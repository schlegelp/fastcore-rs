"""
Wrappers for analogous functions in scipy.sparse.csgraph
"""

import numpy as np

from scipy.sparse import csr_matrix

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
    1. `min_only=True` is not supported.
    2. `return_predecessors=True` is currently not supported (might change in the future)
    3. `checks` (enabled by default) will raise an error if the input graph is not a tree.

    """
    if min_only or return_predecessors:
        raise NotImplementedError(
            "min_only=True and return_predecessors=True are not supported."
        )

    # If input is not csr matrix, try to convert it
    if not isinstance(csgraph, csr_matrix):
        csgraph = csr_matrix(csgraph)

    # Turn the csr matrix into two numpy arrays: sources and targets
    sources, targets = csgraph.nonzero()

    # If checks, ensure that the input graph is actually a tree
    # This means each node must only have one parent (out-degree <= 1)
    if checks:
        _, counts = np.unique(sources, return_counts=True)
        if np.any(counts > 1):
            raise ValueError(
                "Input graph is not a tree. Each node must have at most one parent."
            )

    dmat = geodesic_matrix(
        sources,
        targets,
        directed=directed,
        indices=indices,
        weights=csgraph.data if not unweighted else None,
    )

    # Djikstra would return unreachable nodes as np.inf
    dmat[dmat < 0] = np.inf

    if limit is not np.inf:
        dmat[dmat > limit] = np.inf

    return dmat


def connected_components(csgraph, directed=True, connection='weak', return_labels=True, checks=True):
    """Wrapper for connected_components() that mimics the interfaces of scipy.sparse.csgraph.connected_components().

    Notes:
    1. `connection='strong'` makes no sense for trees, so it is not supported.

    """
    if connection == 'strong':
        raise NotImplementedError("connection='strong' makes no sense for trees, so it is not supported.")

    # If input is not csr matrix, try to convert it
    if not isinstance(csgraph, csr_matrix):
        csgraph = csr_matrix(csgraph)

    # Turn the csr matrix into two numpy arrays: sources and targets
    sources, targets = csgraph.nonzero()

    # If checks, ensure that the input graph is actually a tree
    # This means each node must only have one parent (out-degree <= 1)
    if checks:
        _, counts = np.unique(sources, return_counts=True)
        if np.any(counts > 1):
            raise ValueError(
                "Input graph is not a tree. Each node must have at most one parent."
            )

    # Get the connected components
    cc = _connected_components(sources, targets)

    # csgraph's connected components returns contiguous labels starting from 0
    # whereas our connectec components returns the ID of the root node
    unique, cc = np.unique(cc, return_inverse=True)

    if not return_labels:
        return len(unique)

    return len(unique), cc