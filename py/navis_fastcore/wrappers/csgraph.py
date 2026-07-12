"""
Wrappers for analogous functions in scipy.sparse.csgraph
"""

import numpy as np

from scipy.sparse import csr_matrix

from .. import _fastcore

from ..mesh import geodesic_matrix_graph
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
    targets=None,
    threads=None,
):
    """Wrapper for geodesic_matrix_graph() mimicking scipy.sparse.csgraph.dijkstra().

    Unlike scipy's, this runs one Dijkstra per source *in parallel* — scipy's holds the
    GIL, so you cannot get that from Python by threading it yourself.

    Notes:
    1. `min_only=True` is not supported
    2. `return_predecessors=True` is currently not supported (might change in the future)
    3. `checks` is accepted for signature compatibility but is now a no-op: this used to
       route through the tree-only `geodesic_matrix` and so had to reject cyclic graphs.
       It no longer does, so any graph is fine.
    4. `targets` and `threads` are extensions beyond scipy's API. `targets` is the
       important one: scipy has no notion of targets, so it always materialises all N
       columns and makes you slice afterwards. Passing `targets` here means only those
       columns are ever allocated, which is what makes a large graph tractable.

    """
    if min_only or return_predecessors:
        raise NotImplementedError(
            "min_only=True and return_predecessors=True are not supported."
        )

    edges, weights, n_nodes = _csgraph_to_edges(csgraph)

    # scipy returns a 1-D array when `indices` is a scalar.
    scalar_source = indices is not None and np.isscalar(indices)

    dmat = geodesic_matrix_graph(
        edges,
        n_nodes,
        weights=None if unweighted else weights,
        directed=directed,
        sources=None if indices is None else np.atleast_1d(indices),
        targets=targets,
        limit=None if limit == np.inf else limit,
        threads=threads,
    )

    # Dijkstra reports unreachable nodes as np.inf; we use -1 internally.
    dmat = dmat.astype(np.float64)
    dmat[dmat < 0] = np.inf

    return dmat[0] if scalar_source else dmat


def _csgraph_to_edges(csgraph):
    """Convert a csgraph to an edge list, weights and a node count.

    Every *stored* entry is an edge, which is what scipy's csgraph does too — so an
    explicitly stored zero is a zero-weight edge, not a non-edge.
    """
    if not isinstance(csgraph, csr_matrix):
        csgraph = csr_matrix(csgraph)

    coo = csgraph.tocoo()
    edges = np.stack([coo.row, coo.col], axis=1).astype(np.uint32)
    weights = coo.data.astype(np.float32)

    return edges, weights, csgraph.shape[0]


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
