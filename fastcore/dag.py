import numpy as np

from fastcore._fastcore import _node_indices


def generate_segments(node_ids, parent_ids, weights=None):
    """Generate segments maximizing segment lengths.

    Parameters
    ----------
    node_ids :   (N, ) int32 (long) array
                 Array of int32 node IDs.
    parent_ids : (N, ) int (long) array
                 Array of parent IDs for each node. Root nodes' parents
                 must be -1.
    weights :    (N, ) float32 array, optional
                 Array of distances for each child -> parent connection.
                 If ``None`` all node to node distances are set to 1.

    Returns
    -------
    segments :   list of arrays
                 Segments as list of arrays, sorted from longest to shortest.
                 Each segment starts with a leaf and stops with a branch point
                 or root node.

    """
    # Some initial sanity checks
    node_ids = np.asanyarray(node_ids)
    parent_ids = np.asanyarray(parent_ids)
    assert node_ids.shape == parent_ids.shape
    assert node_ids.ndim == 1 and parent_ids.ndim == 1

    # Make sure we have the correct data types
    node_ids = node_ids.astype("long", order="C", copy=False)
    parent_ids = parent_ids.astype("long", order="C", copy=False)

    # Convert parent IDs into indices
    parent_ix = _node_indices(parent_ids, node_ids)

    # Get the actual path
    segments = _generate_segments(parent_ix, weights=weights)

    # Map node indices back to IDs
    seg_ids = [node_ids[s] for s in segments]

    return seg_ids
