import numpy as np

from . import _fastcore

__all__ = [
    "geodesic_matrix",
    "geodesic_pairs",
    "connected_components",
    "synapse_flow_centrality",
    "generate_segments",
    "break_segments",
    "segment_coords",
    "prune_twigs",
    "strahler_index",
    "classify_nodes",
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
                 Length for each segment. If `weights` is provided this will be
                 the physical length. Otherwise it will be the number of nodes.

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
    array([5, 3], dtype=int32)

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
        lengths = np.array([len(s) for s in segments], dtype=np.int32)

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
    [array([1, 0]), array([3, 2]), array([6, 5, 4])]

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
    colors :        list of colors
                    If `node_colors` provided will return a copy of it sorted
                    to match `seg_coords`.

    Examples
    --------
    >>> import navis_fastcore as fastcore
    >>> import numpy as np
    >>> node_ids = np.arange(7)
    >>> parent_ids = np.array([-1, 0, 1, 2, 1, 4, 5])
    >>> coords = np.random.rand(7, 3)
    >>> fastcore.segment_coords(node_ids, parent_ids, coords)
    [array([[5.30713899e-01, 8.26450947e-01, 2.46805326e-01],
            [1.54144332e-04, 9.07823578e-01, 3.20199043e-01],
            [6.64580597e-01, 3.23724555e-01, 3.18361918e-01],
            [7.16579499e-01, 8.65568868e-02, 7.15686948e-01],
            [5.94874740e-01, 5.95528161e-01, 8.14234930e-01]]),
    array([[0.47814894, 0.84468164, 0.2765942 ],
            [0.21748528, 0.36673489, 0.81449368],
            [0.7165795 , 0.08655689, 0.71568695]])]

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

    # Translate into coordinates
    seg_coords = [coords[s] for s in segments]

    # Apply colors if provided
    if not isinstance(node_colors, type(None)):
        colors = [node_colors[s] for s in segments]

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


def geodesic_pairs(
    node_ids,
    parent_ids,
    pairs,
    directed=False,
    weights=None,
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


def parent_dist(node_ids, parent_ids, xyz, root_dist=None) -> None:
    """Get child->parent distances for skeleton nodes.

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
        # Finding the max value takes only a few microseconds even for large arrays
        max_node_ids = node_ids.max()
        max_to_map = to_map.max()
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


def classify_nodes(node_ids, parent_ids):
    """Classify nodes.

    Note to self: this function is not significantly faster than the
    pandas/numpy-based implementation in navis which is already pretty
    fast. May need to test on larger neurons to see if there is a difference.

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
    array([2, 2, 1, 1, 2, 2, 1, 1], dtype=int32)

    """
    # Convert parent IDs into indices
    parent_ix = _ids_to_indices(node_ids, parent_ids)

    return _fastcore.classify_nodes(parent_ix)
