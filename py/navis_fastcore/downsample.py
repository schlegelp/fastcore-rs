import numpy as np

from . import _fastcore
from .dag import (
    _dropped_to_ids,
    _ids_to_indices,
    _indices_to_ids_sentinel,
    _prep_coords,
    _prep_weights,
    _sources_to_indices,
)


def _preserve_mask(preserve, node_ids):
    """Turn a list of node IDs that must survive into the core's per-node mask."""
    if preserve is None:
        return None
    mask = np.zeros(len(node_ids), dtype=bool)
    mask[_sources_to_indices(node_ids, preserve, what="preserved nodes")] = True
    return mask


__all__ = [
    "downsample_skeleton",
    "simplify_rdp",
    "simplify_vw",
    "resample_skeleton",
    "smooth_skeleton",
    "smooth_skeleton_gaussian",
]


def downsample_skeleton(node_ids, parent_ids, factor, preserve=None, weights=None):
    """Keep every Nth node, dropping the rest.

    The plain "make this skeleton smaller" operation: it pays no attention to
    geometry, so reach for it when the skeleton is already evenly sampled and you
    just want fewer nodes. Roots, branch points and leafs always survive, so the
    result is still the same neuron - only its unbranched stretches are sampled
    `factor` times more coarsely.

    See [`navis_fastcore.simplify_rdp`][] and [`navis_fastcore.simplify_vw`][] for
    the geometry-aware alternatives, which spend the same node budget where the
    neuron actually curves.

    Parameters
    ----------
    node_ids :   (N, ) array
                 Array of node IDs.
    parent_ids : (N, ) array
                 Array of parent IDs for each node. Root nodes' parents
                 must be -1.
    factor :     int
                 Keep one node in every `factor`, counting from each segment's
                 distal end. ``1`` keeps everything; the useful range starts at 2.
    preserve :   (M, ) array, optional
                 IDs of extra nodes that must survive - nodes carrying synapses,
                 say, or the ends of a region of interest.
    weights :    (N, ) float32 array, optional
                 Array of distances for each child -> parent connection.
                 If ``None`` all node-to-node distances are set to 1.

    Returns
    -------
    node_ids :   (M, ) array
                 The surviving node IDs, in their original relative order.
    parent_ids : (M, ) array
                 Their new parent IDs. Roots are -1.
    weights :    (M, ) float32 array or None
                 Length of each node's edge to its new parent, i.e. the summed
                 length of the chain it replaces. Roots are 0. ``None`` exactly
                 when `weights` was ``None``.
    node_map :   (N, ) array
                 For each **input** node, the ID of the surviving node its data
                 belongs to now - indexed like `node_ids`, valued in the returned
                 `node_ids`. Surviving nodes map to themselves; a dropped node maps
                 to whichever end of its chain is nearer, measured in `weights` (in
                 hops if `weights` is ``None``), with ties going towards the root.
                 Use it to re-attach anything you keep per node, such as synapses.

    Examples
    --------
    >>> import navis_fastcore as fastcore
    >>> import numpy as np
    >>> node_ids = np.arange(7)
    >>> parent_ids = np.array([-1, 0, 1, 2, 1, 4, 5])

    Every second node of each segment, plus the root, the branch point and the leafs:

    >>> fastcore.downsample_skeleton(node_ids, parent_ids, 2)[:3]
    (array([0, 1, 3, 4, 6]), array([-1,  0,  1,  1,  4]), None)

    A factor nothing can satisfy leaves just the root, the branch point and the two
    leafs - and total cable length is preserved, the dropped nodes' edges having
    moved into the edges that replaced them:

    >>> weights = np.array([0, 1, 1, 1, 1, 1, 1], dtype=np.float32)
    >>> ids, _, w, node_map = fastcore.downsample_skeleton(
    ...     node_ids, parent_ids, 100, weights=weights
    ... )
    >>> ids
    array([0, 1, 3, 6])
    >>> w
    array([0., 1., 2., 3.], dtype=float32)

    The dropped nodes 2, 4 and 5 hand their data to whichever survivor is nearer:

    >>> node_map
    array([0, 1, 1, 3, 1, 6, 6])

    """
    factor = int(factor)
    if factor < 1:
        raise ValueError(f"`factor` must be >= 1, got {factor}")
    elif factor == np.inf:
        raise ValueError(f"`factor` must be finite, got {factor}")

    parent_ix = _ids_to_indices(node_ids, parent_ids)
    weights = _prep_weights(weights, node_ids)

    kept, new_parent_ix, new_weights, node_map_ix = _fastcore.downsample_skeleton(
        parent_ix, factor, preserve=_preserve_mask(preserve, node_ids), weights=weights
    )

    return _dropped_to_ids(node_ids, kept, new_parent_ix, new_weights, node_map_ix)


def simplify_rdp(
    node_ids, parent_ids, coords, epsilon, preserve=None, weights=None, threads=None
):
    """Drop the nodes that don't bend a neurite (Ramer-Douglas-Peucker).

    Where [`navis_fastcore.downsample_skeleton`][] thins by counting, this thins by
    *shape*: a node survives only if removing it would move the traced path by more
    than `epsilon`. Long straight stretches collapse to their two ends while a tight
    curve keeps every node it needs, so the same tolerance buys a much better
    skeleton per node than a fixed factor does.

    Roots, branch points and leafs always survive, and each replacement edge carries
    the length of the chain it stands in for - so geodesic distances stay right even
    where the geometry has been cut across.

    Parameters
    ----------
    node_ids :   (N, ) array
                 Array of node IDs.
    parent_ids : (N, ) array
                 Array of parent IDs for each node. Root nodes' parents
                 must be -1.
    coords :     (N, 3) array
                 Array of coordinates for each node.
    epsilon :    float
                 How far the simplified path may stray from the original, in the
                 units of `coords`. ``0`` still drops nodes that are *exactly*
                 collinear, and nothing else.
    preserve :   (M, ) array, optional
                 IDs of extra nodes that must survive - nodes carrying synapses,
                 say, or the ends of a region of interest.
    weights :    (N, ) float32 array, optional
                 Array of distances for each child -> parent connection.
                 If ``None`` all node-to-node distances are set to 1.
    threads :    int, optional
                 Number of threads to use. ``None`` uses all available cores.

    Returns
    -------
    node_ids :   (M, ) array
                 The surviving node IDs, in their original relative order.
    parent_ids : (M, ) array
                 Their new parent IDs. Roots are -1.
    weights :    (M, ) float32 array or None
                 Length of each node's edge to its new parent. ``None`` exactly
                 when `weights` was ``None``.
    node_map :   (N, ) array
                 For each **input** node, the ID of the surviving node its data
                 belongs to now - indexed like `node_ids`, valued in the returned
                 `node_ids`. Surviving nodes map to themselves; a dropped node maps
                 to whichever end of its chain is nearer, measured in `weights` (in
                 hops if `weights` is ``None``), with ties going towards the root.
                 Use it to re-attach anything you keep per node, such as synapses.

    Examples
    --------
    >>> import navis_fastcore as fastcore
    >>> import numpy as np
    >>> node_ids = np.arange(5)
    >>> parent_ids = np.array([-1, 0, 1, 2, 3])

    A straight run with one node nudged off the line:

    >>> coords = np.array(
    ...     [[0, 0, 0], [1, 0, 0], [2, 1, 0], [3, 0, 0], [4, 0, 0]], dtype=float
    ... )

    A tolerance below the bump keeps it...

    >>> fastcore.simplify_rdp(node_ids, parent_ids, coords, 0.5)[0]
    array([0, 2, 4])

    ...and one above it does not:

    >>> fastcore.simplify_rdp(node_ids, parent_ids, coords, 2.0)[0]
    array([0, 4])

    """
    assert epsilon >= 0, f"`epsilon` must be non-negative, got {epsilon}"

    parent_ix = _ids_to_indices(node_ids, parent_ids)
    coords = _prep_coords(coords, node_ids)
    weights = _prep_weights(weights, node_ids)

    kept, new_parent_ix, new_weights, node_map_ix = _fastcore.simplify_rdp(
        parent_ix,
        coords,
        float(epsilon),
        preserve=_preserve_mask(preserve, node_ids),
        weights=weights,
        threads=threads,
    )

    return _dropped_to_ids(node_ids, kept, new_parent_ix, new_weights, node_map_ix)


def simplify_vw(
    node_ids, parent_ids, coords, min_area, preserve=None, weights=None, threads=None
):
    """Drop the nodes that contribute least area (Visvalingam-Whyatt).

    The other geometry-aware thinning. Where [`navis_fastcore.simplify_rdp`][] asks
    how far the path *moves*, this asks how much area each node adds to it and
    repeatedly removes whichever node adds least. The difference shows under
    aggressive simplification: RDP will happily keep one spike and flatten
    everything around it, while Visvalingam-Whyatt sheds detail evenly and so keeps
    a neurite looking like itself.

    Parameters
    ----------
    node_ids :   (N, ) array
                 Array of node IDs.
    parent_ids : (N, ) array
                 Array of parent IDs for each node. Root nodes' parents
                 must be -1.
    coords :     (N, 3) array
                 Array of coordinates for each node.
    min_area :   float
                 Remove a node while the triangle it forms with its two surviving
                 neighbours is smaller than this, in the *squared* units of
                 `coords`. ``0`` or less is a no-op.
    preserve :   (M, ) array, optional
                 IDs of extra nodes that must survive - nodes carrying synapses,
                 say, or the ends of a region of interest.
    weights :    (N, ) float32 array, optional
                 Array of distances for each child -> parent connection.
                 If ``None`` all node-to-node distances are set to 1.
    threads :    int, optional
                 Number of threads to use. ``None`` uses all available cores.

    Returns
    -------
    node_ids :   (M, ) array
                 The surviving node IDs, in their original relative order.
    parent_ids : (M, ) array
                 Their new parent IDs. Roots are -1.
    weights :    (M, ) float32 array or None
                 Length of each node's edge to its new parent. ``None`` exactly
                 when `weights` was ``None``.
    node_map :   (N, ) array
                 For each **input** node, the ID of the surviving node its data
                 belongs to now - indexed like `node_ids`, valued in the returned
                 `node_ids`. Surviving nodes map to themselves; a dropped node maps
                 to whichever end of its chain is nearer, measured in `weights` (in
                 hops if `weights` is ``None``), with ties going towards the root.
                 Use it to re-attach anything you keep per node, such as synapses.

    Examples
    --------
    >>> import navis_fastcore as fastcore
    >>> import numpy as np
    >>> node_ids = np.arange(5)
    >>> parent_ids = np.array([-1, 0, 1, 2, 3])

    Two bumps off a straight line, one ten times taller than the other. The small
    one goes:

    >>> coords = np.array(
    ...     [[0, 0, 0], [1, 0.1, 0], [2, 0, 0], [3, 1, 0], [4, 0, 0]], dtype=float
    ... )
    >>> fastcore.simplify_vw(node_ids, parent_ids, coords, 0.5)[0]
    array([0, 2, 3, 4])

    """
    assert min_area >= 0, f"`min_area` must be non-negative, got {min_area}"

    parent_ix = _ids_to_indices(node_ids, parent_ids)
    coords = _prep_coords(coords, node_ids)
    weights = _prep_weights(weights, node_ids)

    kept, new_parent_ix, new_weights, node_map_ix = _fastcore.simplify_vw(
        parent_ix,
        coords,
        float(min_area),
        preserve=_preserve_mask(preserve, node_ids),
        weights=weights,
        threads=threads,
    )

    return _dropped_to_ids(node_ids, kept, new_parent_ix, new_weights, node_map_ix)


def resample_skeleton(node_ids, parent_ids, coords, spacing, threads=None):
    """Place nodes at a fixed spacing along every neurite.

    The inverse problem to [`navis_fastcore.downsample_skeleton`][]: rather than
    thinning what is there, this re-samples each segment from scratch, so a skeleton
    whose node density varies tenfold between neurites comes out evenly sampled
    throughout. It is the step most morphometrics want in front of them - anything
    that averages a quantity *per node* is otherwise weighted by how finely each
    neurite happened to be traced.

    Each segment is divided into ``round(length / spacing)`` equal parts (at least
    one), so both of its endpoints land exactly and no runt edge is left over. A
    segment shorter than ``spacing / 2`` collapses to a single straight edge.

    Parameters
    ----------
    node_ids :   (N, ) array
                 Array of node IDs.
    parent_ids : (N, ) array
                 Array of parent IDs for each node. Root nodes' parents
                 must be -1.
    coords :     (N, 3) array
                 Array of coordinates for each node.
    spacing :    float
                 Target distance between adjacent nodes, in the units of `coords`.
    threads :    int, optional
                 Number of threads to use. ``None`` uses all available cores.

    Returns
    -------
    node_ids :   (M, ) array
                 Node IDs for the resampled skeleton. Roots, branch points and
                 leafs keep their original ID and come first, in their original
                 order; the interpolated nodes get fresh IDs counting up from
                 ``max(node_ids) + 1``.
    parent_ids : (M, ) array
                 Their parent IDs. Roots are -1.
    coords :     (M, 3) array
                 Their coordinates.
    source :     (M, 2) int32 array
                 For each new node, the *indices* into the original `node_ids` of
                 the edge it sits on: column 0 the child (distal) end, column 1 the
                 parent (proximal) end. A node carried over unchanged has its own
                 index in both columns.
    alpha :      (M, ) float64 array
                 How far along that edge each new node lies, from the child end.
                 Zero for a node carried over unchanged.
    node_map :   (N, ) array
                 The other direction: for each **input** node, the ID of the output
                 node nearest it along the neurite, with ties going towards the root.
                 Indexed like `node_ids`, valued in the returned `node_ids`. Nodes
                 carried over map to themselves.

    Notes
    -----
    `source`/`alpha` and `node_map` point opposite ways, and which you want depends
    on what you are moving.

    `source` and `alpha` carry a per-node *column* forward, so this function does not
    have to know what else you keep per node. Radius, label, confidence and anything
    else numeric interpolate the same way, over the whole output at once:

    ```python
    new_radius = (
        radius[source[:, 0]] * (1 - alpha) + radius[source[:, 1]] * alpha
    )
    ```

    `node_map` re-homes whatever is *attached* to a node - a synapse, a soma tag, a
    manual annotation. That question cannot be answered from `source` and `alpha`:
    an input node between two output nodes has no output row of its own, so the
    mapping does not invert.

    ```python
    synapses["node_id"] = pd.Series(node_map, index=node_ids)[synapses["node_id"]].values
    ```

    Examples
    --------
    >>> import navis_fastcore as fastcore
    >>> import numpy as np
    >>> node_ids = np.arange(3)
    >>> parent_ids = np.array([-1, 0, 1])
    >>> coords = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float)

    Two units of cable at a spacing of 0.5 gives four edges, so five nodes. The
    root and the leaf keep their IDs and lead the way; the three new ones follow:

    >>> ids, parents, xyz, source, alpha, node_map = fastcore.resample_skeleton(
    ...     node_ids, parent_ids, coords, 0.5
    ... )
    >>> ids
    array([0, 2, 3, 4, 5])
    >>> parents
    array([-1,  3,  4,  5,  0])

    The new nodes are laid down walking each segment from its distal end, which here
    is the leaf - hence the descending x:

    >>> xyz[:, 0]
    array([0. , 2. , 1.5, 1. , 0.5])

    Input node 1 sat at x = 1, where new node 4 now is, so that is where its data
    goes; the root and the leaf were carried over and map to themselves:

    >>> node_map
    array([0, 4, 2])

    """
    spacing = float(spacing)
    if not spacing > 0:
        raise ValueError(f"`spacing` must be positive, got {spacing}")

    parent_ix = _ids_to_indices(node_ids, parent_ids)
    coords = _prep_coords(coords, node_ids)

    new_parent_ix, new_coords, source, alpha, node_map_ix = _fastcore.resample_skeleton(
        parent_ix, coords, spacing, threads=threads
    )

    # Carried-over nodes keep their IDs; the interpolated ones are new, so they need
    # IDs that cannot collide with an existing one. `source[:, 0] == source[:, 1]`
    # marks the carried-over rows - the core guarantees they come first and in input
    # order, but we do not rely on that here.
    node_ids = np.asarray(node_ids)
    carried = source[:, 0] == source[:, 1]
    dtype = node_ids.dtype if np.issubdtype(node_ids.dtype, np.integer) else np.int64

    new_node_ids = np.empty(len(new_parent_ix), dtype=dtype)
    new_node_ids[carried] = node_ids[source[carried, 0]]
    n_new = int((~carried).sum())
    if n_new:
        start = int(node_ids.max()) + 1 if len(node_ids) else 0
        new_node_ids[~carried] = np.arange(start, start + n_new, dtype=dtype)

    return (
        new_node_ids,
        _indices_to_ids_sentinel(new_node_ids, new_parent_ix),
        new_coords,
        source,
        alpha,
        _indices_to_ids_sentinel(new_node_ids, node_map_ix),
    )


def smooth_skeleton(node_ids, parent_ids, coords, window=5, threads=None):
    """Smooth a skeleton with a moving average along each neurite.

    Takes the tracing jitter out of a skeleton without touching its topology or its
    node count: every node keeps its ID and its parent, and only its coordinates
    move. Roots, branch points and leafs are pinned - a branch point that drifted
    would drag three neurites apart - so this is safe to run before measuring angles,
    tortuosity or tangent vectors, all of which a raw traced skeleton overstates.

    The window shrinks symmetrically as it approaches a segment's ends, which keeps
    the smoothed path centred on the original rather than letting it pull towards
    the middle.

    See [`navis_fastcore.smooth_skeleton_gaussian`][] for the version whose kernel is
    a distance rather than a node count.

    Parameters
    ----------
    node_ids :   (N, ) array
                 Array of node IDs.
    parent_ids : (N, ) array
                 Array of parent IDs for each node. Root nodes' parents
                 must be -1.
    coords :     (N, 3) array
                 Array of coordinates for each node.
    window :     int
                 Nodes in the window, counting the node itself. Even values round
                 down to the odd value below, since the window is symmetric.
                 ``0`` and ``1`` are no-ops.
    threads :    int, optional
                 Number of threads to use. ``None`` uses all available cores.

    Returns
    -------
    coords :     (N, 3) float64 array
                 New coordinates, in the same order as `node_ids`.

    Notes
    -----
    There is no ``node_map`` here, unlike the functions that drop or add nodes: this
    changes coordinates only, so every node keeps its ID and its parent and anything
    attached to a node is still attached to it afterwards. The one thing that does go
    stale is a *copy* of a node's position taken beforehand.

    Examples
    --------
    >>> import navis_fastcore as fastcore
    >>> import numpy as np
    >>> node_ids = np.arange(5)
    >>> parent_ids = np.array([-1, 0, 1, 2, 3])
    >>> coords = np.array(
    ...     [[0, 0, 0], [1, 1, 0], [2, -1, 0], [3, 1, 0], [4, 0, 0]], dtype=float
    ... )
    >>> fastcore.smooth_skeleton(node_ids, parent_ids, coords, window=3)[:, 1]
    array([0.        , 0.        , 0.33333333, 0.        , 0.        ])

    """
    parent_ix = _ids_to_indices(node_ids, parent_ids)
    coords = _prep_coords(coords, node_ids)

    return _fastcore.smooth_skeleton(parent_ix, coords, int(window), threads=threads)


def smooth_skeleton_gaussian(
    node_ids, parent_ids, coords, sigma, truncate=4.0, threads=None
):
    """Smooth a skeleton with a Gaussian kernel along each neurite.

    The same operation as [`navis_fastcore.smooth_skeleton`][] with a softer,
    scale-based kernel: `sigma` is a distance in the units of `coords` rather than a
    count of nodes, so the amount of smoothing does not change when the skeleton is
    resampled. That is usually what you want - and it is why the kernel measures
    distance *along* the neurite rather than between the points, which would let the
    far arm of a hairpin pull on the near one.

    Segment ends are pinned by reflecting the neurite about them, so a node one step
    in from a leaf is smoothed against a symmetric neighbourhood rather than being
    dragged inwards by a one-sided one.

    Parameters
    ----------
    node_ids :   (N, ) array
                 Array of node IDs.
    parent_ids : (N, ) array
                 Array of parent IDs for each node. Root nodes' parents
                 must be -1.
    coords :     (N, 3) array
                 Array of coordinates for each node.
    sigma :      float
                 Kernel width, as a distance along the neurite.
    truncate :   float
                 How many `sigma` out to keep summing. 4 covers all but 1e-4 of the
                 kernel's mass.
    threads :    int, optional
                 Number of threads to use. ``None`` uses all available cores.

    Returns
    -------
    coords :     (N, 3) float64 array
                 New coordinates, in the same order as `node_ids`.

    Notes
    -----
    There is no ``node_map`` here, unlike the functions that drop or add nodes: this
    changes coordinates only, so every node keeps its ID and its parent and anything
    attached to a node is still attached to it afterwards. The one thing that does go
    stale is a *copy* of a node's position taken beforehand.

    Examples
    --------
    >>> import navis_fastcore as fastcore
    >>> import numpy as np
    >>> node_ids = np.arange(5)
    >>> parent_ids = np.array([-1, 0, 1, 2, 3])
    >>> coords = np.array(
    ...     [[0, 0, 0], [1, 1, 0], [2, -1, 0], [3, 1, 0], [4, 0, 0]], dtype=float
    ... )

    The ends stay put, the wobble in between is flattened:

    >>> smoothed = fastcore.smooth_skeleton_gaussian(
    ...     node_ids, parent_ids, coords, sigma=2.0
    ... )
    >>> bool(np.abs(smoothed[2, 1]) < np.abs(coords[2, 1]))
    True
    >>> np.allclose(smoothed[[0, 4]], coords[[0, 4]])
    True

    """
    sigma = float(sigma)
    if not sigma > 0:
        raise ValueError(f"`sigma` must be positive, got {sigma}")
    truncate = float(truncate)
    if truncate < 0:
        raise ValueError(f"`truncate` must be non-negative, got {truncate}")

    parent_ix = _ids_to_indices(node_ids, parent_ids)
    coords = _prep_coords(coords, node_ids)

    return _fastcore.smooth_skeleton_gaussian(
        parent_ix, coords, sigma, truncate, threads=threads
    )
