import numpy as np

from . import _fastcore
from .dag import _ids_to_indices, _indices_to_ids_sentinel, _sources_to_indices

__all__ = [
    "stitch_fragments",
    "heal_skeleton",
    "reroot_rewire",
]


def _augment_with_radius(coords, parent_ix, radius, use_radius):
    """Append a scaled segment-radius column to `coords`.

    The extra column turns the nearest-neighbour search into a 4D one, which
    makes nodes of similar calibre look "closer" and so biases healing towards
    joining fragments of similar thickness. We use the mean radius of the
    *segment* a node belongs to rather than the node's own radius, which is far
    less noisy.
    """
    if radius is None:
        raise ValueError("`use_radius` requires `radius` to be provided")

    radius = np.asarray(radius, dtype=np.float64)
    if len(radius) != len(parent_ix):
        raise ValueError("`radius` must have one entry per node")
    # Missing radii would poison the segment mean; navis treats them as 0.
    radius = np.nan_to_num(radius, nan=0.0)

    # Isolated nodes (a root with no children) form no segment, so they get no
    # segment radius -- fall back to their own radius.
    radius_seg = radius.copy()
    for seg in _fastcore.break_segments(parent_ix):
        radius_seg[seg] = radius[seg].mean()

    # `use_radius` doubles as the weight: True -> 1.0, larger -> radius matters more.
    radius_seg *= float(use_radius)

    return np.column_stack((coords, radius_seg))


def stitch_fragments(
    node_ids,
    parent_ids,
    coords,
    mask=None,
    max_dist=None,
    radius=None,
    use_radius=False,
    threads=None,
):
    """Find minimal-length edges to reconnect a fragmented skeleton.

    This is the low-level primitive behind :func:`heal_skeleton`: given a
    skeleton that consists of several disconnected fragments (connected
    components), it returns the set of new edges that would connect those
    fragments into a single tree while minimising the total added length. It
    does *not* modify the skeleton.

    Parameters
    ----------
    node_ids :   (N, ) array
                 Array of node IDs.
    parent_ids : (N, ) array
                 Array of parent IDs for each node. Root nodes' parents
                 must be -1.
    coords :     (N, 3) array
                 Array of coordinates for each node.
    mask :       (N, ) bool array, optional
                 If provided, only nodes where ``mask`` is ``True`` may be used
                 as bridge endpoints. A fragment without any unmasked node can
                 not be connected.
    max_dist :   float, optional
                 Maximum length for any single new edge. Fragment pairs whose
                 closest eligible nodes are farther apart than this are left
                 disconnected. ``None`` means no limit.
    radius :     (N, ) array, optional
                 Radius for each node. Only required if ``use_radius`` is set.
    use_radius : bool | float
                 If truthy, node radii are taken into account when measuring
                 distances, which biases healing towards connecting fragments of
                 similar calibre. Pass a float to weight the effect: higher
                 values give radius more influence. Note that ``max_dist`` is
                 then measured in this augmented space too.
    threads :    int, optional
                 Number of threads to use. If ``None`` uses all available cores.
                 If you are stitching many skeletons across several *processes*,
                 prefer :func:`navis_fastcore.set_num_threads` — it costs nothing
                 per call, whereas this builds a thread pool each time.

    Returns
    -------
    edges :      (M, 2) array
                 Pairs of node IDs to connect. At most ``(#fragments - 1)`` rows.
    distances :  (M, ) float32 array
                 Euclidean length of each new edge. If ``use_radius`` is set,
                 this includes the radius component.

    Examples
    --------
    >>> import navis_fastcore as fastcore
    >>> import numpy as np
    >>> node_ids = np.arange(4)
    >>> # Two fragments: {0, 1} and {2, 3}
    >>> parent_ids = np.array([-1, 0, -1, 2])
    >>> coords = np.array([[0, 0, 0], [1, 0, 0], [10, 0, 0], [11, 0, 0]], dtype=float)
    >>> edges, dists = fastcore.stitch_fragments(node_ids, parent_ids, coords)
    >>> edges
    array([[1, 2]])
    >>> dists
    array([9.], dtype=float32)

    """
    node_ids = np.asarray(node_ids)
    parent_ix = _ids_to_indices(node_ids, parent_ids)
    coords = np.asarray(coords, dtype=np.float64, order="C")
    assert coords.ndim == 2 and coords.shape[1] == 3, "`coords` must be (N, 3)"
    assert len(coords) == len(node_ids), "`coords` must have one row per node"

    if use_radius:
        coords = _augment_with_radius(coords, parent_ix, radius, use_radius)
    coords = np.ascontiguousarray(coords, dtype=np.float64)

    # Connected components (index -> component label).
    components = _fastcore.connected_components(parent_ix)

    if mask is not None:
        mask = np.asarray(mask, dtype=bool, order="C")
        assert len(mask) == len(node_ids), "`mask` must match `node_ids` length"

    if max_dist is None:
        max_dist = np.inf

    edges_ix, distances = _fastcore.stitch_fragments(
        coords,
        components,
        mask=mask,
        max_dist=float(max_dist),
        threads=None if threads is None else int(threads),
    )

    # Map node indices back to IDs.
    if len(edges_ix):
        edges = node_ids[edges_ix]
    else:
        edges = np.empty((0, 2), dtype=node_ids.dtype)

    return edges, distances


def heal_skeleton(
    node_ids,
    parent_ids,
    coords,
    method="ALL",
    max_dist=None,
    min_size=None,
    mask=None,
    radius=None,
    use_radius=False,
    threads=None,
):
    """Heal a fragmented skeleton by reconnecting its fragments.

    Rust re-implementation of the core of ``navis.heal_skeleton``: it finds the
    minimal-length set of bridges between the skeleton's connected components
    (see :func:`stitch_fragments`) and regenerates a single rooted tree.

    Parameters
    ----------
    node_ids :   (N, ) array
                 Array of node IDs.
    parent_ids : (N, ) array
                 Array of parent IDs for each node. Root nodes' parents
                 must be -1.
    coords :     (N, 3) array
                 Array of coordinates for each node.
    method :     "ALL" | "LEAFS"
                 Which nodes may form new edges. ``"ALL"`` (default) considers
                 every node; ``"LEAFS"`` restricts bridge endpoints to leaf and
                 root nodes (faster, occasionally suboptimal attach points).
    max_dist :   float, optional
                 Maximum length for any single new edge. Gaps larger than this
                 are left unhealed (the result may stay fragmented). ``None``
                 means no limit.
    min_size :   int, optional
                 Fragments with fewer than this many nodes are excluded from
                 healing and stay disconnected.
    mask :       (N, ) bool array, optional
                 If provided, only these nodes may be used as bridge endpoints.
                 Combined (AND) with the ``method`` restriction.
    radius :     (N, ) array, optional
                 Radius for each node. Only required if ``use_radius`` is set.
    use_radius : bool | float
                 If truthy, node radii are taken into account when measuring
                 distances, which prioritises connecting fragments of similar
                 calibre. Pass a float to weight the effect: higher values give
                 radius more influence. To keep this robust we use the mean
                 radius of the segment a node belongs to, not the node's own.
                 Note that ``max_dist`` is then measured in this augmented space
                 too.
    threads :    int, optional
                 Number of threads to use. If ``None`` uses all available cores.
                 If you are healing many skeletons across several *processes*
                 (e.g. ``navis.heal_skeleton(nl, parallel=True)``), set this to a
                 small number or — better, since it costs nothing per call — call
                 :func:`navis_fastcore.set_num_threads` once per worker.

    Returns
    -------
    new_parent_ids : (N, ) array
                 New parent IDs, in the same order as ``node_ids``. If the
                 skeleton could be fully healed this describes a single tree
                 (one root); otherwise it may still contain several roots.

    Examples
    --------
    >>> import navis_fastcore as fastcore
    >>> import numpy as np
    >>> node_ids = np.arange(4)
    >>> parent_ids = np.array([-1, 0, -1, 2])   # two fragments
    >>> coords = np.array([[0, 0, 0], [1, 0, 0], [10, 0, 0], [11, 0, 0]], dtype=float)
    >>> fastcore.heal_skeleton(node_ids, parent_ids, coords)
    array([-1,  0,  1,  2])

    """
    method = str(method).upper()
    if method not in ("ALL", "LEAFS"):
        raise ValueError(f'Unknown method "{method}"')

    node_ids = np.asarray(node_ids)
    parent_ix = _ids_to_indices(node_ids, parent_ids)
    coords = np.asarray(coords, dtype=np.float64, order="C")
    assert coords.ndim == 2 and coords.shape[1] == 3, "`coords` must be (N, 3)"
    assert len(coords) == len(node_ids), "`coords` must have one row per node"

    if use_radius:
        coords = _augment_with_radius(coords, parent_ix, radius, use_radius)
    coords = np.ascontiguousarray(coords, dtype=np.float64)

    components = _fastcore.connected_components(parent_ix)

    # Build the candidate mask from the various restrictions.
    candidate = np.ones(len(node_ids), dtype=bool)

    if method == "LEAFS":
        # classify_nodes: 0=root, 1=leaf, 2=branch, 3=slab
        node_type = _fastcore.classify_nodes(parent_ix)
        candidate &= np.isin(node_type, (0, 1))

    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        assert len(mask) == len(node_ids), "`mask` must match `node_ids` length"
        candidate &= mask

    if min_size is not None:
        # Exclude nodes belonging to fragments smaller than `min_size`.
        labels, counts = np.unique(components, return_counts=True)
        small = set(labels[counts < min_size])
        if small:
            candidate &= ~np.isin(components, list(small))

    if max_dist is None:
        max_dist = np.inf

    # 1. Find the bridging edges (in node-index space).
    edges_ix, _ = _fastcore.stitch_fragments(
        coords,
        components,
        mask=candidate,
        max_dist=float(max_dist),
        threads=None if threads is None else int(threads),
    )

    # 2. Regenerate the parent array. Prefer the existing (first) root so the
    #    healed skeleton keeps its orientation where possible.
    roots = np.where(parent_ix < 0)[0]
    preferred_root = int(roots[0]) if len(roots) else -1
    edges_ix = np.asarray(edges_ix, dtype=np.int32).reshape(-1, 2)
    new_parent_ix = _fastcore.reroot_rewire(parent_ix, edges_ix, preferred_root)

    # 3. Map parent indices back to IDs (-1 stays -1 for roots).
    return _indices_to_ids_sentinel(node_ids, new_parent_ix)


def reroot_rewire(node_ids, parent_ids, new_edges, root=None):
    """Regenerate the parent array after adding a set of undirected edges.

    Takes an *edited edge set* — the skeleton's own child→parent edges plus
    `new_edges` — and orients all of it away from a root, which is what turns a bag
    of edges back into a rooted forest. This is step 2 of
    :func:`heal_skeleton`, exposed on its own for callers who choose their own
    edges rather than taking the minimal bridges from
    :func:`stitch_fragments`.

    Parameters
    ----------
    node_ids :   (N, ) array
                 Array of node IDs.
    parent_ids : (N, ) array
                 Array of parent IDs for each node. Root nodes' parents
                 must be -1.
    new_edges :  (M, 2) array
                 Undirected edges to add, as pairs of node IDs. May be empty, in
                 which case this is a pure re-orientation.
    root :       node ID, optional
                 Preferred root. Its component is rooted here; every other
                 component is rooted at its lowest-index node. ``None`` auto-picks
                 for all of them.

    Returns
    -------
    new_parent_ids : (N, ) array
                 New parent IDs, aligned with `node_ids`, roots as -1. Always a
                 valid forest: `new_edges` that would close a cycle are back-edges
                 to the orienting walk and are dropped, and a component the added
                 edges did not reach simply keeps its own root.

    Notes
    -----
    Not the same as [`navis_fastcore.reroot`][], which re-roots an *unchanged*
    forest by reversing only the edges between each new root and the old one, and
    leaves components nobody named byte-identical. This one re-orients everything,
    because after an edge set changes there is no "unchanged" to preserve. Use
    `reroot` to move a root; use this when the edges themselves moved.

    Examples
    --------
    >>> import navis_fastcore as fastcore
    >>> import numpy as np
    >>> node_ids = np.arange(4)
    >>> parent_ids = np.array([-1, 0, -1, 2])       # two fragments
    >>> fastcore.reroot_rewire(node_ids, parent_ids, [(1, 2)])
    array([-1,  0,  1,  2])

    Rooting the joined skeleton at node 3 instead:

    >>> fastcore.reroot_rewire(node_ids, parent_ids, [(1, 2)], root=3)
    array([ 1,  2,  3, -1])

    """
    node_ids = np.asarray(node_ids)
    parent_ix = _ids_to_indices(node_ids, parent_ids)

    new_edges = np.asarray(new_edges)
    if new_edges.size == 0:
        edges_ix = np.empty((0, 2), dtype=np.int32)
    else:
        if new_edges.ndim != 2 or new_edges.shape[1] != 2:
            raise ValueError(
                f"`new_edges` must be of shape (M, 2), got {new_edges.shape}"
            )
        # Flat, so an unknown ID is reported once for the whole array rather than
        # per column.
        edges_ix = _sources_to_indices(
            node_ids, new_edges.ravel(), what="`new_edges` endpoints"
        ).reshape(-1, 2).astype(np.int32, copy=False)

    if root is None:
        root_ix = -1
    else:
        root_ix = int(_sources_to_indices(node_ids, [root], what="`root`")[0])

    new_parent_ix = _fastcore.reroot_rewire(parent_ix, edges_ix, root_ix)

    return _indices_to_ids_sentinel(node_ids, new_parent_ix)
