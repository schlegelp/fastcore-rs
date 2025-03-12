"""
Wrappers for analogous functions in scipy.sparse.csgraph
"""

import numpy as np

from scipy.sparse import csr_matrix

from .. import _fastcore

from ..dag import geodesic_matrix
from ..dag import connected_components as _connected_components


def dijkstra(
    csgraph,
    directed=True,
    indices=None,
    return_predecessors=False,
    unweighted=False,
    limit=np.inf,
    min_only=False,
    checks=True,
):
    """Wrapper for geodesic_matrix() that mimics the interfaces of scipy.sparse.csgraph.dijkstra().

    Notes:
    1. `min_only=True` is not supported
    2. `return_predecessors=True` is currently not supported (might change in the future)
    3. `checks` (enabled by default) will raise an error if the input graph is not a tree

    """
    if min_only or return_predecessors:
        raise NotImplementedError(
            "min_only=True and return_predecessors=True are not supported."
        )

    # Turn the csr matrix into two numpy arrays: sources and targets
    sources, targets, weights = _csgraph_to_sources_targets(csgraph, checks=checks)

    dmat = geodesic_matrix(
        sources,
        targets,
        directed=directed,
        sources=indices,
        weights=weights if not unweighted else None,
    )

    # Djikstra would return unreachable nodes as np.inf
    dmat[dmat < 0] = np.inf

    if limit is not np.inf:
        dmat[dmat > limit] = np.inf

    return dmat


def connected_components(
    csgraph, directed=True, connection="weak", return_labels=True, checks=True
):
    """Wrapper for connected_components() that mimics the interfaces of scipy.sparse.csgraph.connected_components().

    Notes:
    1. `connection='strong'` makes no sense for trees, so it is not supported.

    """
    if connection == "strong":
        raise NotImplementedError(
            "connection='strong' makes no sense for trees, so it is not supported."
        )

    # Turn the csr matrix into two numpy arrays: sources and targets
    sources, targets, _ = _csgraph_to_sources_targets(csgraph, checks=checks)

    # Get the connected components
    cc = _connected_components(sources, targets)

    # csgraph's connected components returns contiguous labels starting from 0
    # whereas our connectec components returns the ID of the root node
    unique, cc = np.unique(cc, return_inverse=True)

    if not return_labels:
        return len(unique)

    return len(unique), cc


def _csgraph_to_sources_targets(csgraph, checks=True):
    """Convert a csgraph to sources and targets numpy arrays.

    Parameters
    ----------
    csgraph :   csr_matrix
                The input graph.

    checks :    bool, default=True
                Whether to perform checks on the input graph.

    Returns
    -------
    sources :   np.ndarray
                The source nodes of the graph.
    targets :   np.ndarray
                The target nodes of the graph.
    weights :   np.ndarray
                The weights of the graph.

    """
    # If input is not csr matrix, try to convert it
    if not isinstance(csgraph, csr_matrix):
        csgraph = csr_matrix(csgraph)

    if checks:
        if ((csgraph > 0).sum(axis=1).flatten() > 1).any():
            raise ValueError(
                "Input graph is not a tree: each node must have at most one parent."
            )

    # Prepare sources and targets arrays
    sources = np.arange(csgraph.shape[0], dtype=np.int32)
    targets = np.full(csgraph.shape[0], -1, dtype=np.int32)
    weights = np.zeros(csgraph.shape[0], dtype=np.float32)

    # Fill in the target arrays
    x, y = csgraph.nonzero()
    targets[x] = y
    weights[x] = csgraph.data

    # Check for cycles
    if checks and _fastcore.has_cycles(targets):
        raise ValueError("Input graph is not a tree: it contains cycles.")

    return sources, targets, weights
