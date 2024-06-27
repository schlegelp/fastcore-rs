import numpy as np

from . import _fastcore

__all__ = [
    "generate_segments",
    "geodesic_matrix",
    "connected_components",
    "synapse_flow_centrality",
    "segment_coords",
]


def generate_segments(node_ids, parent_ids, weights=None):
    """Generate segments maximizing segment lengths.

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

    """
    # Convert parent IDs into indices
    parent_ix = _ids_to_indices(node_ids, parent_ids)

    # Get the actual path
    segments = _fastcore.generate_segments(parent_ix, weights=weights)

    # Map node indices back to IDs
    seg_ids = [node_ids[s] for s in segments]

    return seg_ids


def segment_coords(
    node_ids,
    parent_ids,
    coords,
    node_colors=None,
):
    """Generate coordinates for linear segments.

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
    seg_coords :    list of tuples
                    [(x, y, z), (x, y, z), ... ]
    colors :        list of colors
                    If `node_colors` provided will return a copy of it sorted
                    to match `seg_coords`.

    """
    # Convert parent IDs into indices
    parent_ix = _ids_to_indices(node_ids, parent_ids)

    # Get the actual paths, these are indices into `node_ids`
    segments = _fastcore.generate_segments(parent_ix, weights=None)

    # Translate into coordinates
    seg_coords = [coords[s] for s in segments]

    # Apply colors if provided
    if not isinstance(node_colors, type(None)):
        colors = [node_colors[s] for s in segments]

        return seg_coords, colors

    return seg_coords


def geodesic_matrix(
    node_ids, parent_ids, directed=False, sources=None, targets=None, weights=None
):
    """Calculate all-by-all geodesic distances.

    This implementation is up to 100x faster than the implementation in navis
    (which uses scipy's `csgraph`).

    Parameters
    ----------
    node_ids :   (N, ) array
                 Array of node IDs.
    parent_ids : (N, ) array
                 Array of parent IDs for each node. Root nodes' parents
                 must be -1.
    directed :   bool, optional
                 If ``True`` will only return distances in the direction of
                 the child -> parent (i.e. toward root) relationship.
    sources :    iterable, optional
                 Source node IDs. If ``None`` all nodes are used as sources.
    targets :    iterable, optional
                 Target node IDs. If ``None`` all nodes are used as targets.
    weights :    (N, ) float32 array, optional
                 Array of distances for each child -> parent connection.
                 If ``None`` all node to node distances are set to 1.

    Returns
    -------
    matrix :    float32 (double) array
                Geodesic distances. Unreachable nodes are set to -1.

    """
    # Convert parent IDs into indices
    parent_ix = _ids_to_indices(node_ids, parent_ids)

    if weights is not None:
        weights = np.asarray(weights, dtype=np.float32, order="C")
        assert len(weights) == len(node_ids), "`weights` must have the same length as `node_ids`"

    if sources is not None:
        sources = np.asarray(sources, dtype=np.int32)
        assert len(sources), "`sources` must not be empty"
    if targets is not None:
        targets = np.asarray(targets, dtype=np.int32)
        assert len(targets), "`targets` must not be empty"

    # Get the actual path
    dists = _fastcore.geodesic_distances(
        parent_ix, sources=sources, targets=targets, weights=weights, directed=directed
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
                For each node the node ID of its root.

    """
    # Convert parent IDs into indices
    parent_ix = _ids_to_indices(node_ids, parent_ids)

    # Get connected components - this returns indices, not node IDs
    cc = _fastcore.connected_components(parent_ix)

    # Return the root node ID for each node
    return node_ids[cc]


def synapse_flow_centrality(node_ids, parent_ids, presynapses, postsynapses):
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
    flow = _fastcore.synapse_flow_centrality(parent_ix, presynapses, postsynapses)

    # Return the root node ID for each node
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


def _ids_to_indices(node_ids, parent_ids):
    """Convert node IDs to indices.

    Parameters
    ----------
    node_ids :  (N, )
                Array of node IDs.
    parent_ids : (N, )
                Array of parent IDs for each node. Root nodes' parents
                must be -1.

    Returns
    -------
    parent_ix : (N, ) int32 (long) array
                Array with parent indices for each node.

    """
    # Some initial sanity checks
    node_ids = np.asanyarray(node_ids)
    parent_ids = np.asanyarray(parent_ids)
    assert node_ids.shape == parent_ids.shape
    assert node_ids.ndim == 1 and parent_ids.ndim == 1

    # Make sure we have the correct data types and order
    # "long" = int64 on 64-bit systems
    node_ids = node_ids.astype("long", order="C", copy=False)
    parent_ids = parent_ids.astype("long", order="C", copy=False)

    return _fastcore.node_indices(node_ids, parent_ids)
