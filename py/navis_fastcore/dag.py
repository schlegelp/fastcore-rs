import numpy as np

from . import _fastcore
from ._points import _prep_points

__all__ = [
    "geodesic_matrix",
    "geodesic_nearest",
    "geodesic_farthest",
    "geodesic_pairs",
    "connected_components",
    "synapse_flow_centrality",
    "generate_segments",
    "break_segments",
    "segment_coords",
    "prune_twigs",
    "strahler_index",
    "subtree_height",
    "dist_to_root",
    "parent_dist",
    "classify_nodes",
    "has_cycles",
    "descendants",
    "paths_to_root",
    "reroot",
    "contract_nodes",
    "simplify_skeleton",
    "adjacency",
    "longest_path",
    "longest_paths",
    "descendant_counts",
    "betweenness",
]


def generate_segments(node_ids, parent_ids, weights=None):
    """Generate linear segments maximizing segment lengths.

    Parameters
    ----------
    node_ids :   (N, ) array
                 Array node IDs.
    parent_ids : (N, ) array
                 Array of parent IDs for each node. Root nodes' parents
                 must be -1.
    weights :    (N, ) float32 array, optional
                 Array of distances for each child -> parent connection.
                 If ``None`` all node-to-node distances are set to 1.

    Returns
    -------
    segments :   list of arrays
                 Segments as list of arrays, sorted from longest to shortest.
                 Each segment starts with a leaf and stops with a branch point
                 or root node.
    lengths :    array
                 Length of each segment, measured from its **first node to its
                 last**: the physical distance between them if `weights` was
                 given, otherwise the number of edges. Because a segment stops
                 *at* a branch point, that terminal node's own edge continues
                 into the parent segment and is not counted here - so a
                 single-node segment has length 0.

    Examples
    --------
    >>> import navis_fastcore as fastcore
    >>> import numpy as np
    >>> node_ids = np.arange(7)
    >>> parent_ids = np.array([-1, 0, 1, 2, 1, 4, 5])
    >>> segs, length = fastcore.generate_segments(node_ids, parent_ids)
    >>> segs
    [array([6, 5, 4, 1, 0]), array([3, 2, 1])]
    >>> length
    array([4, 2], dtype=int32)

    """
    # Convert parent IDs into indices
    parent_ix = _ids_to_indices(node_ids, parent_ids)

    # Make sure weights are float32
    if weights is not None:
        weights = np.asarray(weights, dtype=np.float32, order="C")
        assert len(weights) == len(node_ids), (
            "`weights` must have the same length as `node_ids`"
        )

    # Get the segments (this will be a list of arrays of node indices)
    segments, lengths = _fastcore.generate_segments(parent_ix, weights=weights)

    if lengths is not None:
        lengths = np.asarray(lengths, dtype=np.float32)
    else:
        # Edges, not nodes: with every edge weighing 1 this is what the weighted
        # branch above returns, so `weights=None` stays equivalent to `weights=ones`.
        lengths = np.array([len(s) - 1 for s in segments], dtype=np.int32)

    # Map node indices back to IDs
    seg_ids = [node_ids[s] for s in segments]

    return seg_ids, lengths


def break_segments(node_ids, parent_ids):
    """Break neuron into linear segments connecting ends, branches and root.

    Parameters
    ----------
    node_ids :   (N, ) array
                 Array node IDs.
    parent_ids : (N, ) array
                 Array of parent IDs for each node. Root nodes' parents
                 must be -1.

    Returns
    -------
    segments :   list of arrays
                 Segments as list of arrays.

    Examples
    --------
    >>> import navis_fastcore as fastcore
    >>> import numpy as np
    >>> node_ids = np.arange(7)
    >>> parent_ids = np.array([-1, 0, 1, 2, 1, 4, 5])
    >>> fastcore.break_segments(node_ids, parent_ids)
    [array([1, 0]), array([3, 2, 1]), array([6, 5, 4, 1])]

    """
    # Convert parent IDs into indices
    parent_ix = _ids_to_indices(node_ids, parent_ids)

    # Get the segments (this will be a list of arrays of node indices)
    segments = _fastcore.break_segments(parent_ix)

    # Map node indices back to IDs
    seg_ids = [node_ids[s] for s in segments]

    return seg_ids


def segment_coords(
    node_ids,
    parent_ids,
    coords,
    weights=None,
    node_colors=None,
):
    """Generate coordinates for linear segments.

    This is useful for plotting the skeleton of a neuron.

    Parameters
    ----------
    node_ids :      (N, ) array
                    Array node IDs.
    parent_ids :    (N, ) array
                    Array of parent IDs for each node. Root nodes' parents
                    must be -1.
    coords :        (N, 3) array
                    Array of coordinates for each node.
    node_colors :   (N, ) numpy.ndarray, optional
                    A color for each node in `node_ids`. If provided, will
                    also return a list of colors sorted to match coordinates.

    Returns
    -------
    seg_coords :    list of arrays
                    Note that these are views into the original `coords` array!
    colors :        list of colors
                    If `node_colors` provided will return a copy of it sorted
                    to match `seg_coords`.

    Examples
    --------
    >>> import navis_fastcore as fastcore
    >>> import numpy as np
    >>> node_ids = np.arange(7)
    >>> parent_ids = np.array([-1, 0, 1, 2, 1, 4, 5])
    >>> coords = np.random.RandomState(42).rand(7, 3)
    >>> fastcore.segment_coords(node_ids, parent_ids, coords)
    [array([[0.43194502, 0.29122914, 0.61185289],
           [0.18340451, 0.30424224, 0.52475643],
           [0.83244264, 0.21233911, 0.18182497],
           [0.59865848, 0.15601864, 0.15599452],
           [0.37454012, 0.95071431, 0.73199394]]), array([[0.70807258, 0.02058449, 0.96990985],
           [0.05808361, 0.86617615, 0.60111501],
           [0.59865848, 0.15601864, 0.15599452]])]

    """
    # Convert parent IDs into indices
    parent_ix = _ids_to_indices(node_ids, parent_ids)

    if weights is not None:
        weights = np.asarray(weights, dtype=np.float32, order="C")
        assert len(weights) == len(node_ids), (
            "`weights` must have the same length as `node_ids`"
        )

    # Get the segments (this will be a list of arrays of node indices)
    segments, _ = _fastcore.generate_segments(parent_ix, weights=weights)

    # Translate into coordinates via a single batched index + split
    # (faster than one fancy-index call per segment)
    all_indices = np.concatenate(segments)
    split_at = np.cumsum([len(s) for s in segments[:-1]])
    seg_coords = np.split(coords[all_indices], split_at)

    # Apply colors if provided
    if not isinstance(node_colors, type(None)):
        colors = np.split(node_colors[all_indices], split_at)

        return seg_coords, colors

    return seg_coords


def geodesic_matrix(
    node_ids,
    parent_ids,
    directed=False,
    sources=None,
    targets=None,
    weights=None,
):
    """Calculate geodesic ("along-the-arbor") distances.

    Notes
    -----
    Under-the-hood, this uses two different implementations depending on whether
    a full all-by-all or a partial (via `sources`/`targets`) matrix is requested.
    The partial implementation is faster and more memory efficient for small-ish
    subsets of nodes. However, for subsets that include a large portion of the
    nodes, it may be faster to calculate the full matrix and then subset it.

    Parameters
    ----------
    node_ids :   (N, ) array
                 Array of node IDs.
    parent_ids : (N, ) array
                 Array of parent IDs for each node. Root nodes' parents
                 must be -1.
    directed :   bool, optional
                 If ``True`` will only return distances in the direction of
                 the child -> parent (i.e. towards the root) relationship.
    sources :    iterable, optional
                 Source node IDs. If ``None`` all nodes are used as sources.
    targets :    iterable, optional
                 Target node IDs. If ``None`` all nodes are used as targets.
    weights :    (N, ) float32 array, optional
                 Array of distances for each child -> parent connection.
                 If ``None`` all node to node distances are set to 1.

    Returns
    -------
    matrix :    float32 (single) array
                Geodesic distances. Unreachable nodes are set to -1. If
                `source` and/or `targets` are provided, the matrix will be
                ordered accordingly.

    Examples
    --------
    >>> import navis_fastcore as fastcore
    >>> import numpy as np
    >>> node_ids = np.arange(7)
    >>> parent_ids = np.array([-1, 0, 1, 2, 1, 4, 5])
    >>> fastcore.geodesic_matrix(node_ids, parent_ids)
    array([[0., 1., 2., 3., 2., 3., 4.],
           [1., 0., 1., 2., 1., 2., 3.],
           [2., 1., 0., 1., 2., 3., 4.],
           [3., 2., 1., 0., 3., 4., 5.],
           [2., 1., 2., 3., 0., 1., 2.],
           [3., 2., 3., 4., 1., 0., 1.],
           [4., 3., 4., 5., 2., 1., 0.]], dtype=float32)
    >>> fastcore.geodesic_matrix(
    ...     node_ids, parent_ids,
    ...     sources=[0, 1], targets=[5, 6]
    ...     )
    array([[3., 4.],
           [2., 3.]], dtype=float32)

    """
    # Convert parent IDs into indices
    parent_ix = _ids_to_indices(node_ids, parent_ids)

    if weights is not None:
        weights = np.asarray(weights, dtype=np.float32, order="C")
        assert len(weights) == len(node_ids), (
            "`weights` must have the same length as `node_ids`"
        )

    # Translate sources and targets into indices (if provided)
    # This will also de-duplicate the IDs!
    if sources is not None:
        sources_ix = np.where(np.isin(node_ids, sources))[0].astype(np.int32)
        assert len(sources), "`sources` must not be empty"
    else:
        sources_ix = None

    if targets is not None:
        targets_ix = np.where(np.isin(node_ids, targets))[0].astype(np.int32)
        assert len(targets), "`targets` must not be empty"
    else:
        targets_ix = None

    # Calculate distances
    dists = _fastcore.geodesic_distances(
        parent_ix,
        sources=sources_ix,
        targets=targets_ix,
        weights=weights,
        directed=directed,
    )

    # If sources and targets are provided, we need to order the matrix
    if sources is not None:
        id2ix = {nid: ix for ix, nid in enumerate(node_ids[sources_ix])}
        dists = dists[[id2ix[nid] for nid in sources]]

    if targets is not None:
        id2ix = {nid: ix for ix, nid in enumerate(node_ids[targets_ix])}
        dists = dists[:, [id2ix[nid] for nid in targets]]

    return dists


def geodesic_nearest(
    node_ids,
    parent_ids,
    sources=None,
    targets=None,
    directed=False,
    weights=None,
):
    """Find the nearest target for each source.

    This is a memory-efficient companion to :func:`geodesic_matrix`: instead of
    materialising the full ``sources x targets`` distance matrix it only keeps,
    for each source, the distance to and ID of the *nearest* target. It uses a
    linear-time algorithm and therefore scales to several 100k nodes.

    A source that is itself a target is matched to the nearest *other* (distinct)
    target, never to itself.

    Parameters
    ----------
    node_ids :   (N, ) array
                 Array of node IDs.
    parent_ids : (N, ) array
                 Array of parent IDs for each node. Root nodes' parents
                 must be -1.
    sources :    iterable, optional
                 Source node IDs. If ``None`` all nodes are used as sources.
    targets :    iterable, optional
                 Target node IDs. If ``None`` all nodes are used as targets.
    directed :   bool, optional
                 If ``True`` only consider targets in the direction of the
                 child -> parent (i.e. towards the root) relationship.
    weights :    (N, ) float32 array, optional
                 Array of distances for each child -> parent connection.
                 If ``None`` all node to node distances are set to 1.

    Returns
    -------
    distances :  float32 (single) array
                 Distance from each source to its nearest target. Sources
                 without a reachable target are set to ``-1``. Ordered to match
                 `sources` (or `node_ids` if `sources` is ``None``).
    nearest :    array
                 Node ID of the nearest target for each source, in the same
                 order as `distances`. Sources without a reachable target are
                 set to ``-1``.

    Examples
    --------
    >>> import navis_fastcore as fastcore
    >>> import numpy as np
    >>> node_ids = np.arange(7)
    >>> parent_ids = np.array([-1, 0, 1, 2, 1, 4, 5])
    >>> dist, nearest = fastcore.geodesic_nearest(
    ...     node_ids, parent_ids,
    ...     sources=[0, 3], targets=[5, 6]
    ...     )
    >>> dist
    array([3., 4.], dtype=float32)
    >>> nearest
    array([5, 5])

    """
    # Convert parent IDs into indices
    parent_ix = _ids_to_indices(node_ids, parent_ids)

    node_ids = np.asarray(node_ids)

    if weights is not None:
        weights = np.asarray(weights, dtype=np.float32, order="C")
        assert len(weights) == len(node_ids), (
            "`weights` must have the same length as `node_ids`"
        )

    # Translate sources and targets into indices (if provided).
    # This will also de-duplicate the IDs!
    if sources is not None:
        sources_ix = np.where(np.isin(node_ids, sources))[0].astype(np.int32)
        assert len(sources_ix), "`sources` must not be empty"
    else:
        sources_ix = None

    if targets is not None:
        targets_ix = np.where(np.isin(node_ids, targets))[0].astype(np.int32)
        assert len(targets_ix), "`targets` must not be empty"
    else:
        targets_ix = None

    # Calculate nearest distances and target indices (in node-index space)
    distances, nearest_ix = _fastcore.geodesic_nearest(
        parent_ix,
        sources=sources_ix,
        targets=targets_ix,
        weights=weights,
        directed=directed,
    )

    # If sources are provided, re-order to match the order they were passed in
    if sources is not None:
        id2ix = {nid: ix for ix, nid in enumerate(node_ids[sources_ix])}
        order = [id2ix[nid] for nid in sources]
        distances = distances[order]
        nearest_ix = nearest_ix[order]

    # Translate the nearest target node indices back into node IDs (-1 = no target)
    nearest = _indices_to_ids_sentinel(node_ids, nearest_ix)

    return distances, nearest


def geodesic_farthest(
    node_ids,
    parent_ids,
    sources=None,
    targets=None,
    directed=False,
    weights=None,
):
    """Find the farthest target for each source.

    This is the mirror image of :func:`geodesic_nearest`: it uses the same
    linear-time algorithm but keeps, for each source, the distance to and ID of
    the *farthest* target. Like its counterpart it never materialises the full
    ``sources x targets`` distance matrix and therefore scales to several 100k
    nodes.

    A source that is itself a target is matched to the farthest *other*
    (distinct) target, never to itself.

    Parameters
    ----------
    node_ids :   (N, ) array
                 Array of node IDs.
    parent_ids : (N, ) array
                 Array of parent IDs for each node. Root nodes' parents
                 must be -1.
    sources :    iterable, optional
                 Source node IDs. If ``None`` all nodes are used as sources.
    targets :    iterable, optional
                 Target node IDs. If ``None`` all nodes are used as targets.
    directed :   bool, optional
                 If ``True`` only consider targets in the direction of the
                 child -> parent (i.e. towards the root) relationship. Note that
                 with non-negative weights the farthest such target is the target
                 ancestor closest to the root.
    weights :    (N, ) float32 array, optional
                 Array of distances for each child -> parent connection.
                 If ``None`` all node to node distances are set to 1.

    Returns
    -------
    distances :  float32 (single) array
                 Distance from each source to its farthest target. Sources
                 without a reachable target are set to ``-1``. Ordered to match
                 `sources` (or `node_ids` if `sources` is ``None``).
    farthest :   array
                 Node ID of the farthest target for each source, in the same
                 order as `distances`. Sources without a reachable target are
                 set to ``-1``.

    Examples
    --------
    >>> import navis_fastcore as fastcore
    >>> import numpy as np
    >>> node_ids = np.arange(7)
    >>> parent_ids = np.array([-1, 0, 1, 2, 1, 4, 5])
    >>> dist, farthest = fastcore.geodesic_farthest(
    ...     node_ids, parent_ids,
    ...     sources=[0, 3], targets=[5, 6]
    ...     )
    >>> dist
    array([4., 5.], dtype=float32)
    >>> farthest
    array([6, 6])

    """
    # Convert parent IDs into indices
    parent_ix = _ids_to_indices(node_ids, parent_ids)

    node_ids = np.asarray(node_ids)

    if weights is not None:
        weights = np.asarray(weights, dtype=np.float32, order="C")
        assert len(weights) == len(node_ids), (
            "`weights` must have the same length as `node_ids`"
        )

    # Translate sources and targets into indices (if provided).
    # This will also de-duplicate the IDs!
    if sources is not None:
        sources_ix = np.where(np.isin(node_ids, sources))[0].astype(np.int32)
        assert len(sources_ix), "`sources` must not be empty"
    else:
        sources_ix = None

    if targets is not None:
        targets_ix = np.where(np.isin(node_ids, targets))[0].astype(np.int32)
        assert len(targets_ix), "`targets` must not be empty"
    else:
        targets_ix = None

    # Calculate farthest distances and target indices (in node-index space)
    distances, farthest_ix = _fastcore.geodesic_farthest(
        parent_ix,
        sources=sources_ix,
        targets=targets_ix,
        weights=weights,
        directed=directed,
    )

    # If sources are provided, re-order to match the order they were passed in
    if sources is not None:
        id2ix = {nid: ix for ix, nid in enumerate(node_ids[sources_ix])}
        order = [id2ix[nid] for nid in sources]
        distances = distances[order]
        farthest_ix = farthest_ix[order]

    # Translate the farthest target node indices back into node IDs (-1 = no target)
    farthest = _indices_to_ids_sentinel(node_ids, farthest_ix)

    return distances, farthest


def geodesic_pairs(
    node_ids,
    parent_ids,
    pairs,
    directed=False,
    weights=None,
    threads=None,
):
    """Calculate geodesic ("along-the-arbor") distances between pairs of nodes.

    This uses a simple algorithm that calculates distances using brute force.
    It's fast because we parallelize the calculation of each pair of nodes.

    Parameters
    ----------
    node_ids :   (N, ) array
                 Array of node IDs.
    parent_ids : (N, ) array
                 Array of parent IDs for each node. Root nodes' parents
                 must be -1.
    pairs :      (N, 2) array
                 Pairs of node IDs for which to calculate distances.
    directed :   bool, optional
                 If ``True`` will only return distances in the direction of
                 the child -> parent (i.e. towards the root) relationship.
    weights :    (N, ) float32 array, optional
                 Array of distances for each child -> parent connection.
                 If ``None`` all node to node distances are set to 1.
    threads :    int, optional
                 Number of threads to use. If ``None`` uses all available cores.
                 If you are running this across several *processes*, prefer
                 :func:`navis_fastcore.set_num_threads` — it costs nothing per
                 call, whereas this builds a thread pool each time.

    Returns
    -------
    matrix :    float32 (single) array
                Geodesic distances. Unreachable nodes are set to -1.

    Examples
    --------
    >>> import navis_fastcore as fastcore
    >>> import numpy as np
    >>> node_ids = np.arange(7)
    >>> parent_ids = np.array([-1, 0, 1, 2, 1, 4, 5])
    >>> pairs = np.array([(0, 1), (0, 2)])
    >>> fastcore.geodesic_pairs(node_ids, parent_ids, pairs)
    array([1., 2.], dtype=float32)

    """
    # Convert parent IDs into indices
    parent_ix = _ids_to_indices(node_ids, parent_ids)

    pairs = np.asarray(pairs)
    assert pairs.ndim == 2 and pairs.shape[1] == 2, "`pairs` must be of shape (N, 2)"

    if weights is not None:
        weights = np.asarray(weights, dtype=np.float32, order="C")
        assert len(weights) == len(node_ids), (
            "`weights` must have the same length as `node_ids`"
        )

    # Calculate distances
    dists = _fastcore.geodesic_pairs(
        parent_ix,
        pairs_source=_ids_to_indices(node_ids, pairs[:, 0]),
        pairs_target=_ids_to_indices(node_ids, pairs[:, 1]),
        weights=weights,
        directed=directed,
        threads=None if threads is None else int(threads),
    )

    return dists


def connected_components(node_ids, parent_ids):
    """Get the connected components for this neuron.

    Parameters
    ----------
    node_ids :   (N, ) array
                 Array of node IDs.
    parent_ids : (N, ) array
                 Array of parent IDs for each node. Root nodes' parents
                 must be -1.

    Returns
    -------
    cc :        (N, ) int32 array
                For each node the node ID of its root (= connected component ID).

    Examples
    --------
    Fully connected neuron:

    >>> import navis_fastcore as fastcore
    >>> import numpy as np
    >>> node_ids = np.arange(7)
    >>> parent_ids = np.array([-1, 0, 1, 2, 1, 4, 5])
    >>> fastcore.connected_components(node_ids, parent_ids)
    array([0, 0, 0, 0, 0, 0, 0])

    Introduce a break:

    >>> parent_ids[4] = -1
    >>> fastcore.connected_components(node_ids, parent_ids)
    array([0, 0, 0, 0, 4, 4, 4])

    """
    # Convert parent IDs into indices
    parent_ix = _ids_to_indices(node_ids, parent_ids)

    # Get connected components - this returns indices, not node IDs
    cc = _fastcore.connected_components(parent_ix)

    # Return the root node ID for each node
    return node_ids[cc]


def synapse_flow_centrality(
    node_ids, parent_ids, presynapses, postsynapses, mode="sum"
):
    """Calculate synapse flow centrality for this neuron.

    Please note that this implementation currently produces slightly different
    results than the implementation in navis. I'm not sure why that is but the
    differences seem to be negligible.

    Parameters
    ----------
    node_ids :   (N, ) array
                 Array of int32 node IDs.
    parent_ids : (N, ) array
                 Array of parent IDs for each node. Root nodes' parents
                 must be -1.
    presynapses : (N, ) uint32 array
                 Array of number of presynapses associated with each node.
    postsynapses : (N, ) uint32 array
                 Array of number of postsynapses associated with each node.
    mode :       "centrifugal" | "centripetal" | "sum"
                 The mode to calculate the flow centrality. "centrifugal" will
                 calculate the flow from the root to the leaves, "centripetal"
                 will calculate the flow from the leaves to the root, and "sum"
                 will calculate the sum of both.


    Returns
    -------
    cc :        (N, ) uint32 array
                Synapse flow centrality for each node.

    """
    # Convert parent IDs into indices
    parent_ix = _ids_to_indices(node_ids, parent_ids)

    # Make sure we have the correct data types and order
    presynapses = presynapses.astype(np.uint32, order="C", copy=False)
    postsynapses = postsynapses.astype(np.uint32, order="C", copy=False)

    assert len(presynapses) == len(postsynapses) == len(node_ids)

    # Get connected components - this returns indices, not node IDs
    flow = _fastcore.synapse_flow_centrality(parent_ix, presynapses, postsynapses, mode)

    # Return the flow for each node
    return flow


def parent_dist(node_ids, parent_ids, xyz, root_dist=None):
    """Get child->parent distances for skeleton nodes.

    The edge weights every other function here takes as ``weights``: the euclidean
    distance from each node to its parent. R spells this ``child_to_parent_dists``.

    Parameters
    ----------
    node_ids :   (N, ) array
                 Array of int32 node IDs.
    parent_ids : (N, ) array
                 Array of parent IDs for each node. Root nodes' parents
                 must be -1.
    xyz :        (N, 3) array
                 Array of coordinates for each node.
    root_dist :  int | None
                 ``parent_dist`` for the root's row. Set to ``None``, to leave
                 at ``NaN`` or e.g. to ``0`` to set to 0.

    Returns
    -------
    np.ndarray
                 Array with distances in same order and size as node table.

    """
    # Note: this function is effectively a copy of the one in navis with the
    # main difference being that it uses the fastcore implementation of
    # _ids_to_indices which is ~5X faster than the pandas-based version
    # in navis. Consider using this function for cable length calculations
    # instead of the graph-based one.

    # Convert parent IDs into indices
    parent_ix = _ids_to_indices(node_ids, parent_ids)
    not_root = parent_ix >= 0

    # Calculate distances
    w = np.full(len(parent_ix), root_dist, dtype=np.float32)
    w[not_root] = np.sqrt(((xyz[not_root] - xyz[parent_ix[not_root]]) ** 2).sum(axis=1))

    return w


def _ids_to_indices(node_ids, to_map):
    """Map node IDs to node indices.

    Parameters
    ----------
    node_ids :  (N, )
                Array of node IDs.
    to_map :    (N, )
                Array of IDs to map to indices. Root nodes' parents
                must be -1.

    Returns
    -------
    parent_ix : (N, ) int32 array
                Array with parent indices for each node.

    """
    # Some initial sanity checks
    node_ids = np.asanyarray(node_ids)
    to_map = np.asanyarray(to_map)
    assert node_ids.ndim == 1 and to_map.ndim == 1

    # # We need the IDs to be signed integers and we need the same dtypes.
    # # When the dtypes are different we need to convert them but we need
    # # to be careful to avoid overflow/underflow errors.
    fix_dtypes = False
    if node_ids.dtype != to_map.dtype:
        fix_dtypes = True
    elif node_ids.dtype not in (np.int16, np.int32, np.int64):
        fix_dtypes = True
    elif node_ids.dtype not in (np.int16, np.int32, np.int64):
        fix_dtypes = True

    # Cast to the smallest safe signed integer type.
    # This whole block should not take more than a few tens of microseconds
    if fix_dtypes:
        # Finding the max value takes only a few microseconds even for large arrays.
        # `initial=0` covers the empty case: asking for nothing is a legitimate call
        # (e.g. `descendant_counts(targets=[])`), and `max()` on an empty array raises.
        max_node_ids = node_ids.max(initial=0)
        max_to_map = to_map.max(initial=0)
        for dtype in (np.int16, np.int32, np.int64):
            if (
                np.iinfo(dtype).max >= max_node_ids
                and np.iinfo(dtype).max >= max_to_map
            ):
                node_ids = node_ids.astype(dtype, copy=False)  # cast only if necessary
                to_map = to_map.astype(dtype, copy=False)  # cast only if necessary
                break

    # Dispatch the correct function
    if node_ids.dtype == np.int16:
        return _fastcore.node_indices_16(node_ids, to_map)
    elif node_ids.dtype == np.int32:
        return _fastcore.node_indices_32(node_ids, to_map)
    elif node_ids.dtype == np.int64:
        return _fastcore.node_indices_64(node_ids, to_map)
    else:
        raise ValueError("IDs must be int32 or int64")


def prune_twigs(node_ids, parent_ids, threshold, weights=None, mask=None):
    """Prune twigs shorter than a given threshold.

    Parameters
    ----------
    node_ids :   (N, ) array
                 Array node IDs.
    parent_ids : (N, ) array
                 Array of parent IDs for each node. Root nodes' parents
                 must be -1.
    threshold :  float
                 Twigs shorter than this threshold will be pruned.
    weights :    (N, ) float32 array, optional
                 Array of distances for each child -> parent connection.
                 If ``None`` all node-to-node distances are set to 1.
    mask :       (N, ) bool array, optional
                 Array of booleans to mask nodes that should not be pruned.
                 Importantly, twigs with _any_ masked node will not be pruned.


    Returns
    -------
    keep :       (M, ) integer array
                 Node IDs to keep.

    Examples
    --------
    >>> import navis_fastcore as fastcore
    >>> import numpy as np
    >>> node_ids = np.arange(7)
    >>> parent_ids = np.array([-1, 0, 1, 2, 1, 4, 5])
    >>> fastcore.prune_twigs(node_ids, parent_ids, 2)
    array([0, 1, 4, 5, 6])
    >>> mask = np.array([True, True, True, False, True, True, True])
    >>> fastcore.prune_twigs(node_ids, parent_ids, 2, mask=mask)
    array([0, 1, 2, 3, 4, 5, 6])

    """
    # Convert parent IDs into indices
    parent_ix = _ids_to_indices(node_ids, parent_ids)

    # Make sure weights are float32
    if weights is not None:
        weights = np.asarray(weights, dtype=np.float32, order="C")
        assert len(weights) == len(node_ids), (
            "`weights` must have the same length as `node_ids`"
        )

    # Make sure mask is boolean
    if mask is not None:
        mask = np.asarray(mask, dtype=bool, order="C")
        assert len(mask) == len(node_ids), (
            "`mask` must have the same length as `node_ids"
        )

    # Get the nodes to keep
    keep_idx = _fastcore.prune_twigs(parent_ix, threshold, weights=weights, mask=mask)

    # Map node indices back to IDs
    return node_ids[keep_idx]


def strahler_index(
    node_ids, parent_ids, method="standard", to_ignore=None, min_twig_size=None
):
    """Calculcate Strahler Index.

    Parameters
    ----------
    node_ids :          (N, ) array
                        Array node IDs.
    parent_ids :        (N, ) array
                        Array of parent IDs for each node. Root nodes' parents
                        must be -1.
    method :            'standard' | 'greedy', optional
                        Method used to calculate Strahler indices: 'standard'
                        will use the method described above; 'greedy' will
                        always increase the index at converging branches
                        whether these branches have the same index or not.
    to_ignore :         iterable, optional
                        List of node IDs to ignore. Must be the FIRST node
                        of the branch. Excluded branches will not contribute
                        to Strahler index calculations and instead be assigned
                        the SI of their parent branch.
    min_twig_size :     int, optional
                        If provided, will ignore twigs with fewer nodes than
                        this. Instead, they will be assigned the SI of their
                        parent branch.

    Returns
    -------
    strahler_index :    (N, ) int array
                        Strahler Index for each node.

    Examples
    --------
    >>> import navis_fastcore as fastcore
    >>> import numpy as np
    >>> node_ids = np.arange(8)
    >>> parent_ids = np.array([-1, 0, 1, 2, 1, 4, 5, 5])
    >>> fastcore.strahler_index(node_ids, parent_ids)
    array([2, 2, 1, 1, 2, 2, 1, 1], dtype=int32)

    """
    # Convert parent IDs into indices
    parent_ix = _ids_to_indices(node_ids, parent_ids)

    # Convert to_ignore to indices
    if to_ignore is not None:
        to_ignore = np.where(np.isin(node_ids, to_ignore))[0].astype(np.int32)

    # Convert min_twig_size to int32
    if min_twig_size is not None:
        min_twig_size = np.int32(min_twig_size)

    # Get the Strahler indices
    strahler_index = _fastcore.strahler_index(
        parent_ix, min_twig_size=min_twig_size, to_ignore=to_ignore, method=method
    )

    # Map node indices back to IDs
    return strahler_index


def subtree_height(node_ids, parent_ids, weights=None):
    """Calculate the height of the subtree below each node.

    A node's height is the geodesic distance from it *down* to the farthest leaf
    below it. Leafs therefore have a height of 0, and a root has the length of
    the longest root-to-leaf path in its component.

    Parameters
    ----------
    node_ids :   (N, ) array
                 Array of node IDs.
    parent_ids : (N, ) array
                 Array of parent IDs for each node. Root nodes' parents
                 must be -1.
    weights :    (N, ) float32 array, optional
                 Array of distances for each child -> parent connection.
                 If ``None`` all node to node distances are set to 1. Weights are
                 expected to be non-negative. A root's own entry is never read,
                 so the ``NaN`` :func:`parent_dist` leaves there is harmless.

    Returns
    -------
    heights :    float32 (single) array
                 Height of each node, in the same order as `node_ids`.

    Examples
    --------
    >>> import navis_fastcore as fastcore
    >>> import numpy as np
    >>> node_ids = np.arange(8)
    >>> parent_ids = np.array([-1, 0, 1, 2, 1, 4, 5, 5])
    >>> fastcore.subtree_height(node_ids, parent_ids)
    array([4., 3., 1., 0., 2., 1., 0., 0.], dtype=float32)

    See Also
    --------
    :func:`geodesic_farthest`
                 Answers a different question: its ``directed`` mode looks towards
                 the root, and its undirected mode can leave the subtree entirely.

    """
    # Convert parent IDs into indices
    parent_ix = _ids_to_indices(node_ids, parent_ids)

    if weights is not None:
        weights = np.asarray(weights, dtype=np.float32, order="C")
        assert len(weights) == len(node_ids), (
            "`weights` must have the same length as `node_ids`"
        )

    # Already aligned with `node_ids`, so no index -> ID mapping needed
    return _fastcore.subtree_height(parent_ix, weights=weights)


def dist_to_root(node_ids, parent_ids, sources=None, weights=None):
    """Calculate the distance from each node to its root.

    Parameters
    ----------
    node_ids :   (N, ) array
                 Array of node IDs.
    parent_ids : (N, ) array
                 Array of parent IDs for each node. Root nodes' parents
                 must be -1.
    sources :    iterable, optional
                 Node IDs to measure from. If ``None`` all nodes are used.
    weights :    (N, ) float32 array, optional
                 Array of distances for each child -> parent connection.
                 If ``None`` all node to node distances are set to 1.

    Returns
    -------
    distances :  float32 (single) array
                 Distance to the root for each node. Roots are at distance 0 from
                 themselves. Ordered to match `sources` (or `node_ids` if `sources`
                 is ``None``).

    Examples
    --------
    >>> import navis_fastcore as fastcore
    >>> import numpy as np
    >>> node_ids = np.arange(7)
    >>> parent_ids = np.array([-1, 0, 1, 2, 1, 4, 5])
    >>> fastcore.dist_to_root(node_ids, parent_ids)
    array([0., 1., 2., 3., 2., 3., 4.], dtype=float32)

    """
    # Convert parent IDs into indices
    parent_ix = _ids_to_indices(node_ids, parent_ids)

    node_ids = np.asarray(node_ids)

    if weights is not None:
        weights = np.asarray(weights, dtype=np.float32, order="C")
        assert len(weights) == len(node_ids), (
            "`weights` must have the same length as `node_ids`"
        )

    # Translate sources into indices (if provided).
    # This will also de-duplicate the IDs!
    if sources is not None:
        sources_ix = np.where(np.isin(node_ids, sources))[0].astype(np.int32)
        assert len(sources_ix), "`sources` must not be empty"
    else:
        sources_ix = None

    distances = _fastcore.all_dists_to_root(
        parent_ix, sources=sources_ix, weights=weights
    )

    # If sources are provided, re-order to match the order they were passed in
    if sources is not None:
        id2ix = {nid: ix for ix, nid in enumerate(node_ids[sources_ix])}
        distances = distances[[id2ix[nid] for nid in sources]]

    return distances


def classify_nodes(node_ids, parent_ids):
    """Classify nodes.

    Parameters
    ----------
    node_ids :          (N, ) array
                        Array node IDs.
    parent_ids :        (N, ) array
                        Array of parent IDs for each node. Root nodes' parents
                        must be -1.

    Returns
    -------
    node_type :         (N, ) integer array
                        Node types:
                         - 0: root
                         - 1: leaf
                         - 2: branch point
                         - 3: slab (intermediate node)

    Examples
    --------
    >>> import navis_fastcore as fastcore
    >>> import numpy as np
    >>> node_ids = np.arange(8)
    >>> parent_ids = np.array([-1, 0, 1, 2, 1, 4, 5, 5])
    >>> fastcore.classify_nodes(node_ids, parent_ids)
    array([0, 2, 3, 1, 3, 2, 1, 1], dtype=int32)

    """
    # Convert parent IDs into indices
    parent_ix = _ids_to_indices(node_ids, parent_ids)

    return _fastcore.classify_nodes(parent_ix)


def has_cycles(node_ids, parent_ids):
    """Check whether the parent structure contains a cycle.

    A well-formed skeleton is a rooted forest: walking parents from any node has
    to arrive at a root. Every other function in this module assumes that without
    checking it; this is the check, in a single linear pass.

    Parameters
    ----------
    node_ids :   (N, ) array
                 Array of node IDs.
    parent_ids : (N, ) array
                 Array of parent IDs for each node. Root nodes' parents
                 must be -1.

    Returns
    -------
    bool
                 Whether any node is its own ancestor.

    Notes
    -----
    A parent ID that does not appear in `node_ids` is treated as -1, i.e. as a
    root - the same convention as everywhere else here - so a dangling parent is
    not a cycle.

    Cyclic input is malformed, not merely unusual. The other functions are
    written so that it cannot hang them, but what they hand back is a truncated
    walk rather than an answer. Use this if you need to know rather than survive.

    Examples
    --------
    >>> import navis_fastcore as fastcore
    >>> import numpy as np
    >>> node_ids = np.array([1, 2, 3, 4])
    >>> fastcore.has_cycles(node_ids, np.array([-1, 1, 2, 3]))
    False

    Every node an ancestor of itself, with no root anywhere:

    >>> fastcore.has_cycles(node_ids, np.array([4, 1, 2, 3]))
    True

    """
    # Convert parent IDs into indices
    parent_ix = _ids_to_indices(node_ids, parent_ids)

    return _fastcore.has_cycles(parent_ix)


def _sources_to_indices(node_ids, sources, what="sources"):
    """Map a list of node IDs to indices, preserving order and duplicates.

    Not `np.isin`: that de-duplicates *and* sorts, which would silently reorder a
    caller's request. `_ids_to_indices` keeps the sequence intact and flags anything
    it cannot find with a negative index.
    """
    indices = _ids_to_indices(node_ids, np.asarray(sources))
    missing = indices < 0
    if missing.any():
        bad = np.asarray(sources)[missing]
        raise ValueError(
            f"{len(bad)} {what} not found in `node_ids` (e.g. {bad[0]})"
        )
    return indices


def _prep_weights(weights, node_ids):
    """Coerce optional per-node weights to a contiguous float32 array of the right length.

    Mirrors `mesh._prep_weights` minus the width resolution: the DAG kernels are
    float32 only. Extracted because this block had been copy-pasted to
    fourteen call sites in this module, half of them raising `AssertionError` (which
    `python -O` removes) where the rest raise `ValueError`.
    """
    if weights is None:
        return None
    weights = np.ascontiguousarray(np.asarray(weights, dtype=np.float32).ravel())
    if len(weights) != len(node_ids):
        raise ValueError(
            f"`weights` must have one entry per node: got {len(weights)} "
            f"for {len(node_ids)} nodes"
        )
    return weights


def _prep_coords(coords, node_ids, name="coords"):
    """Coerce per-node coordinates to a contiguous float64 (N, 3) array.

    The shape check itself is `_points._prep_points`, which the transform modules
    already share; this adds the "one row per node" rule, which is what makes a
    coordinate array a *skeleton's* coordinates rather than a point cloud.

    Lives here rather than beside its first caller for the reason `_prep_weights`
    gives: the same three lines had been inlined at every call site, half of them
    raising `AssertionError`, which `python -O` removes.
    """
    coords, _ = _prep_points(coords, name=name)
    if len(coords) != len(node_ids):
        raise ValueError(
            f"`{name}` must have one row per node: got {len(coords)} "
            f"for {len(node_ids)} nodes"
        )
    return coords


def _dropped_to_ids(node_ids, kept, new_parent_ix, new_weights):
    """Map a `(kept, new_parents, new_weights)` triple back into ID space.

    Shared by every method that drops nodes and rewires what is left - the four in
    [`navis_fastcore.downsample`][] plus `simplify_skeleton`. They differ only in
    which nodes they decide to drop, and all of them then owe the caller the same
    thing.
    """
    new_node_ids = _indices_to_ids(node_ids, kept)
    return (
        new_node_ids,
        _indices_to_ids_sentinel(new_node_ids, new_parent_ix),
        new_weights,
    )


def _indices_to_ids(node_ids, indices):
    """Map node indices back to node IDs."""
    return np.asarray(node_ids)[indices]


def _runs_to_ids(node_ids, runs):
    """Map a list of index runs back to node IDs, one array per run."""
    node_ids = np.asarray(node_ids)
    # Empty runs come back as float64 arrays from numpy's default dtype, which cannot
    # index; `intp` keeps them usable without copying the non-empty ones.
    return [node_ids[np.asarray(run, dtype=np.intp)] for run in runs]


def _indices_to_ids_sentinel(node_ids, indices):
    """Map indices back to node IDs, keeping -1 where the index is negative.

    Used wherever a result carries a "no such node" sentinel: roots in a parent
    array, or sources with no reachable target.
    """
    node_ids = np.asarray(node_ids)
    dtype = node_ids.dtype
    if not np.issubdtype(dtype, np.signedinteger):
        # -1 has no unsigned representation. This mirrors the convention navis uses:
        # a uint64 `node_id` column alongside an int64 `parent_id` column.
        dtype = np.int64

    out = np.full(len(indices), -1, dtype=dtype)
    found = indices >= 0
    out[found] = node_ids[indices[found]].astype(dtype, copy=False)
    return out


def _walk(fn, node_ids, parent_ids, sources):
    """Run a per-source traversal and map both ends between IDs and indices.

    `descendants` and `paths_to_root` are the two directions of the same walk and differ
    only in which core function they call.
    """
    parent_ix = _ids_to_indices(node_ids, parent_ids)
    sources_ix = _sources_to_indices(node_ids, sources)

    return _runs_to_ids(node_ids, fn(parent_ix, sources_ix))


def descendants(node_ids, parent_ids, sources):
    """Find the nodes distal to each source, i.e. its sub-tree.

    Parameters
    ----------
    node_ids :   (N, ) array
                 Array of node IDs.
    parent_ids : (N, ) array
                 Array of parent IDs for each node. Root nodes' parents
                 must be -1.
    sources :    iterable
                 Node IDs to collect the sub-tree of.

    Returns
    -------
    subtrees :   list of arrays
                 One array of node IDs per source, in `sources` order. Each starts
                 with the source itself and is in depth-first pre-order, so a node
                 always precedes its own descendants.

    Examples
    --------
    >>> import navis_fastcore as fastcore
    >>> import numpy as np
    >>> node_ids = np.arange(6)
    >>> parent_ids = np.array([-1, 0, 1, 2, 1, 2])
    >>> fastcore.descendants(node_ids, parent_ids, [2])
    [array([2, 3, 5])]

    A leaf is its own only descendant:

    >>> fastcore.descendants(node_ids, parent_ids, [3, 4])
    [array([3]), array([4])]

    See Also
    --------
    [`navis_fastcore.paths_to_root`][]
                 The same walk in the opposite direction.

    """
    return _walk(_fastcore.descendants, node_ids, parent_ids, sources)


def paths_to_root(node_ids, parent_ids, sources):
    """Walk from each source up to its root.

    Parameters
    ----------
    node_ids :   (N, ) array
                 Array of node IDs.
    parent_ids : (N, ) array
                 Array of parent IDs for each node. Root nodes' parents
                 must be -1.
    sources :    iterable
                 Node IDs to walk up from.

    Returns
    -------
    paths :      list of arrays
                 One array of node IDs per source, in `sources` order, ordered
                 source-first / root-last. A source that is itself a root gives a
                 single-element path.

    Examples
    --------
    >>> import navis_fastcore as fastcore
    >>> import numpy as np
    >>> node_ids = np.arange(6)
    >>> parent_ids = np.array([-1, 0, 1, 2, 1, 2])
    >>> fastcore.paths_to_root(node_ids, parent_ids, [3, 4])
    [array([3, 2, 1, 0]), array([4, 1, 0])]

    A root is a path of one:

    >>> fastcore.paths_to_root(node_ids, parent_ids, [0])
    [array([0])]

    """
    return _walk(_fastcore.paths_to_root, node_ids, parent_ids, sources)


def reroot(node_ids, parent_ids, new_roots):
    """Re-root the skeleton at the given node(s).

    Every edge on the path from a new root to its component's old root is reversed;
    the rest of the tree is untouched.

    Parameters
    ----------
    node_ids :   (N, ) array
                 Array of node IDs.
    parent_ids : (N, ) array
                 Array of parent IDs for each node. Root nodes' parents
                 must be -1.
    new_roots :  iterable
                 Node IDs to root their components at. Components containing none
                 of these are left exactly as they were - this re-roots, it does
                 not renumber the rest of the forest. Where two new roots fall in
                 the same component, the first one wins.

    Returns
    -------
    parent_ids : (N, ) array
                 New parent IDs, aligned with `node_ids`. Roots are -1.

    Examples
    --------
    >>> import navis_fastcore as fastcore
    >>> import numpy as np
    >>> node_ids = np.arange(5)
    >>> parent_ids = np.array([-1, 0, 1, 2, 1])
    >>> fastcore.reroot(node_ids, parent_ids, [3])
    array([ 1,  2,  3, -1,  1])

    Re-rooting at the existing root is a no-op:

    >>> fastcore.reroot(node_ids, parent_ids, [0])
    array([-1,  0,  1,  2,  1])

    """
    parent_ix = _ids_to_indices(node_ids, parent_ids)
    roots_ix = _sources_to_indices(node_ids, new_roots, what="new roots")

    new_parent_ix = _fastcore.reroot(parent_ix, roots_ix)

    return _indices_to_ids_sentinel(node_ids, new_parent_ix)


def contract_nodes(node_ids, parent_ids, mapping):
    """Collapse groups of nodes onto a representative and rewire.

    Parameters
    ----------
    node_ids :   (N, ) array
                 Array of node IDs.
    parent_ids : (N, ) array
                 Array of parent IDs for each node. Root nodes' parents
                 must be -1.
    mapping :    (N, ) array
                 For each node, the ID of the node it collapses into. A node
                 mapped to itself survives; nodes sharing a representative are
                 merged into it. Edges that end up inside a group are dropped.

    Returns
    -------
    node_ids :   (M, ) array
                 The surviving node IDs, in their original relative order.
    parent_ids : (M, ) array
                 Their new parent IDs. Roots are -1.

    Raises
    ------
    ValueError
                 If the contraction would produce a cycle - which happens when a
                 node is mapped onto one of its own descendants.

    Examples
    --------
    Collapse nodes 1 and 2 onto node 1:

    >>> import navis_fastcore as fastcore
    >>> import numpy as np
    >>> node_ids = np.arange(5)
    >>> parent_ids = np.array([-1, 0, 1, 2, 2])
    >>> mapping = np.array([0, 1, 1, 3, 4])
    >>> fastcore.contract_nodes(node_ids, parent_ids, mapping)
    (array([0, 1, 3, 4]), array([-1,  0,  1,  1]))

    Note this does not re-root; follow with
    [`navis_fastcore.reroot`][] if you need the result rooted somewhere specific.

    """
    parent_ix = _ids_to_indices(node_ids, parent_ids)

    mapping = np.asarray(mapping)
    if len(mapping) != len(node_ids):
        raise ValueError(
            f"`mapping` must have the same length as `node_ids` "
            f"({len(mapping)} vs {len(node_ids)})"
        )
    mapping_ix = _sources_to_indices(node_ids, mapping, what="mapping targets")

    # A mapping that collapses a node onto one of its own descendants closes a loop; the
    # core refuses it rather than returning something that is not a forest.
    kept, new_parent_ix = _fastcore.contract_nodes(parent_ix, mapping_ix)

    new_node_ids = _indices_to_ids(node_ids, kept)

    return new_node_ids, _indices_to_ids_sentinel(new_node_ids, new_parent_ix)


def simplify_skeleton(node_ids, parent_ids, weights=None):
    """Reduce a skeleton to its roots, leafs and branch points.

    The slab nodes in between carry no topological information. Dropping them
    leaves the same tree at a fraction of the size, with each replacement edge
    carrying the total length of the chain it stands in for - so total cable
    length is preserved.

    Parameters
    ----------
    node_ids :   (N, ) array
                 Array of node IDs.
    parent_ids : (N, ) array
                 Array of parent IDs for each node. Root nodes' parents
                 must be -1.
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

    Examples
    --------
    >>> import navis_fastcore as fastcore
    >>> import numpy as np
    >>> node_ids = np.arange(5)
    >>> parent_ids = np.array([-1, 0, 1, 2, 2])
    >>> weights = np.array([0, 1, 2, 4, 8], dtype=np.float32)
    >>> ids, parents, w = fastcore.simplify_skeleton(
    ...     node_ids, parent_ids, weights=weights
    ... )

    Node 1 was the only slab, so it is gone and nodes 3 and 4 now hang off node 2
    directly. Node 2's new edge carries both the 2 -> 1 and 1 -> 0 lengths:

    >>> ids
    array([0, 2, 3, 4])
    >>> parents
    array([-1,  0,  2,  2])
    >>> w
    array([0., 3., 4., 8.], dtype=float32)

    """
    parent_ix = _ids_to_indices(node_ids, parent_ids)

    weights = _prep_weights(weights, node_ids)

    kept, new_parent_ix, new_weights = _fastcore.simplify_skeleton(
        parent_ix, weights=weights
    )

    return _dropped_to_ids(node_ids, kept, new_parent_ix, new_weights)


def adjacency(node_ids, parent_ids, weights=None, directed=True, transpose=False):
    """Build the skeleton's adjacency matrix in CSR form.

    Parameters
    ----------
    node_ids :   (N, ) array
                 Array of node IDs.
    parent_ids : (N, ) array
                 Array of parent IDs for each node. Root nodes' parents
                 must be -1.
    weights :    (N, ) float32 array, optional
                 Array of distances for each child -> parent connection.
                 If ``None`` all edges weigh 1.
    directed :   bool
                 If ``True`` only the child -> parent edge is emitted. If
                 ``False`` both directions are.
    transpose :  bool
                 If ``True`` flip every edge, so rows are parents and columns
                 children.

    Returns
    -------
    indptr :     (N + 1, ) int32 array
    indices :    (nnz, ) int32 array
    data :       (nnz, ) float32 array
                 The three arrays of a `scipy.sparse.csr_matrix`. Rows and columns
                 follow `node_ids` order, i.e. index `i` is `node_ids[i]`. Column
                 indices are ascending within each row.

    Notes
    -----
    Returning the raw triple rather than a matrix keeps this package free of a
    scipy dependency. To build the matrix - note scipy takes the three arrays in
    the *opposite* order to the conventional CSR description used here::

        from scipy.sparse import csr_matrix
        n = len(node_ids)
        indptr, indices, data = fastcore.adjacency(node_ids, parent_ids)
        A = csr_matrix((data, indices, indptr), shape=(n, n))

    Examples
    --------
    >>> import navis_fastcore as fastcore
    >>> import numpy as np
    >>> node_ids = np.arange(3)
    >>> parent_ids = np.array([-1, 0, 1])
    >>> indptr, indices, data = fastcore.adjacency(node_ids, parent_ids)

    Row `i` holds node `i`'s edge to its parent, so the root's row is empty:

    >>> indptr
    array([0, 0, 1, 2], dtype=int32)
    >>> indices
    array([0, 1], dtype=int32)

    """
    parent_ix = _ids_to_indices(node_ids, parent_ids)

    weights = _prep_weights(weights, node_ids)

    return _fastcore.adjacency(parent_ix, weights, bool(directed), bool(transpose))


def longest_path(node_ids, parent_ids, weights=None):
    """Find the longest path in the skeleton.

    Parameters
    ----------
    node_ids :   (N, ) array
                 Array of node IDs.
    parent_ids : (N, ) array
                 Array of parent IDs for each node. Root nodes' parents
                 must be -1.
    weights :    (N, ) float32 array, optional
                 Array of distances for each child -> parent connection.
                 If ``None`` all node-to-node distances are set to 1.

    Returns
    -------
    path :       (L, ) array
                 Node IDs along the path, **distal first** - so `path[0]` is the
                 far end and `path[-1]` is a root. Ties are broken towards the
                 node that comes first in `node_ids`.

    Notes
    -----
    This is not the (NP-hard) general longest-path problem. In a rooted forest
    every maximal path is fixed by its start node - just follow the parents up -
    so the longest one starts at whichever node is farthest from its own root.

    Examples
    --------
    >>> import navis_fastcore as fastcore
    >>> import numpy as np
    >>> node_ids = np.arange(5)
    >>> parent_ids = np.array([-1, 0, 1, 2, 1])
    >>> fastcore.longest_path(node_ids, parent_ids)
    array([3, 2, 1, 0])

    Weights change which path is longest - here one heavy hop beats two light ones:

    >>> weights = np.array([0, 1, 1, 1, 50], dtype=np.float32)
    >>> fastcore.longest_path(node_ids, parent_ids, weights=weights)
    array([4, 1, 0])

    See Also
    --------
    [`navis_fastcore.longest_paths`][]
                 The `n` longest paths, each peeled off before the next.

    """
    parent_ix = _ids_to_indices(node_ids, parent_ids)

    weights = _prep_weights(weights, node_ids)

    return _indices_to_ids(
        node_ids, _fastcore.longest_path(parent_ix, weights=weights)
    )


def longest_paths(node_ids, parent_ids, n, weights=None, min_length=None):
    """Find the `n` longest paths, peeling each one off before the next.

    Each path is removed from the skeleton before the next is sought, so the
    second path is the longest of what *remains* rather than the second-longest
    of the original - and the paths are pairwise disjoint. Removing a path turns
    the children hanging off it into roots of their own.

    Parameters
    ----------
    node_ids :   (N, ) array
                 Array of node IDs.
    parent_ids : (N, ) array
                 Array of parent IDs for each node. Root nodes' parents
                 must be -1.
    n :          int
                 How many paths to take. Fewer are returned if the skeleton runs
                 out, or if `min_length` stops the search.
    weights :    (N, ) float32 array, optional
                 Array of distances for each child -> parent connection.
                 If ``None`` all node-to-node distances are set to 1.
    min_length : float, optional
                 Stop as soon as a path measures no more than this.

    Returns
    -------
    paths :      list of arrays
                 Up to `n` arrays of node IDs, longest first, each distal-first.

    Warnings
    --------
    `min_length` measures the path's **whole catchment**, not just its own edges:
    every edge whose *parent* lies on the path counts, so each twig hanging off
    the path contributes its first edge too. The comparison is `<=`, and hitting
    it **stops** the search rather than skipping that one path.

    That is inherited from navis' `split_into_fragments`, where it is flagged as
    "preserved as-is from the networkx implementation", and it is kept here
    deliberately so that results do not shift.

    Examples
    --------
    >>> import navis_fastcore as fastcore
    >>> import numpy as np
    >>> node_ids = np.arange(6)
    >>> parent_ids = np.array([-1, 0, 1, 2, 1, 4])
    >>> fastcore.longest_paths(node_ids, parent_ids, 2)
    [array([3, 2, 1, 0]), array([5, 4])]

    Note the second path stops at node 4: node 1 went with the first path, so 4
    is the root of what was left.

    """
    parent_ix = _ids_to_indices(node_ids, parent_ids)

    weights = _prep_weights(weights, node_ids)

    n = int(n)
    if n < 0:
        raise ValueError(f"`n` must be non-negative, got {n}")

    paths = _fastcore.longest_paths(
        parent_ix,
        n,
        weights=weights,
        min_length=None if min_length is None else float(min_length),
    )

    return _runs_to_ids(node_ids, paths)


def descendant_counts(node_ids, parent_ids, targets=None):
    """Count, for each node, how many nodes lie strictly below it.

    Parameters
    ----------
    node_ids :   (N, ) array
                 Array of node IDs.
    parent_ids : (N, ) array
                 Array of parent IDs for each node. Root nodes' parents
                 must be -1.
    targets :    iterable, optional
                 Restrict the count to these node IDs. If ``None`` every node
                 counts.

    Returns
    -------
    counts :     (N, ) int64 array
                 Aligned with `node_ids`. A node is never its own descendant, so
                 a leaf scores 0 even when it is itself a target.

    Notes
    -----
    With `targets=None` this is each node's sub-tree size minus one.

    This is what navis' `betweeness_centrality(from_=...)` actually computed,
    under a name that suggested otherwise - see
    [`navis_fastcore.betweenness`][].

    Examples
    --------
    >>> import navis_fastcore as fastcore
    >>> import numpy as np
    >>> node_ids = np.arange(5)
    >>> parent_ids = np.array([-1, 0, 1, 2, 1])
    >>> fastcore.descendant_counts(node_ids, parent_ids)
    array([4, 3, 1, 0, 0])

    Counting only the leafs below each node:

    >>> fastcore.descendant_counts(node_ids, parent_ids, targets=[3, 4])
    array([2, 2, 1, 0, 0])

    """
    parent_ix = _ids_to_indices(node_ids, parent_ids)

    target_ix = None
    if targets is not None:
        target_ix = _sources_to_indices(node_ids, targets, what="targets")

    return _fastcore.descendant_counts(parent_ix, targets=target_ix)


def betweenness(node_ids, parent_ids, directed=True):
    """Calculate betweenness centrality.

    Parameters
    ----------
    node_ids :   (N, ) array
                 Array of node IDs.
    parent_ids : (N, ) array
                 Array of parent IDs for each node. Root nodes' parents
                 must be -1.
    directed :   bool
                 If ``True`` only count paths running towards the root, i.e. from
                 a node to one of its ancestors. If ``False`` count every
                 unordered pair once.

    Returns
    -------
    betweenness : (N, ) int64 array
                 Number of shortest paths through each node, aligned with
                 `node_ids`. Pairs are only counted within a connected component.

    Notes
    -----
    O(N), not Brandes' O(V*E): shortest paths in a tree are *unique*, so the
    number passing through a node is a closed form rather than a search. Directed,
    a node lies on one path per (strict descendant, strict ancestor) pair.
    Undirected, removing it splits its component into its children's sub-trees
    plus everything above, and it lies between every pair drawn from two
    different parts.

    Counts are `int64` because they grow as the square of the component size - an
    undirected 100k-node skeleton reaches ~5e9, which overflows `int32`.

    Examples
    --------
    >>> import navis_fastcore as fastcore
    >>> import numpy as np
    >>> node_ids = np.arange(5)
    >>> parent_ids = np.array([-1, 0, 1, 2, 3])
    >>> fastcore.betweenness(node_ids, parent_ids)
    array([0, 3, 4, 3, 0])

    Leafs and roots are never *between* anything.

    See Also
    --------
    [`navis_fastcore.descendant_counts`][]
                 What you want if you are counting how much hangs below a node
                 rather than how much routes through it.

    """
    parent_ix = _ids_to_indices(node_ids, parent_ids)

    return _fastcore.betweenness(parent_ix, bool(directed))
