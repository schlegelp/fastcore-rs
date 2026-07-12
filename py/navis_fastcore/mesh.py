import numpy as np

from . import _fastcore

__all__ = [
    "mesh_connected_components",
    "geodesic_matrix_mesh",
    "geodesic_matrix_graph",
    "geodesic_nearest_mesh",
    "geodesic_farthest_mesh",
]


def mesh_connected_components(faces, n_vertices):
    """Find connected components of a triangle mesh.

    Uses Union-Find (DSU) with path-halving. The only extra allocation is a
    single integer array of length ``n_vertices`` — no adjacency list is built.

    Parameters
    ----------
    faces :      (N, 3) array
                 Triangular faces given as rows of three vertex indices.
                 Must be convertible to ``uint32``.
    n_vertices : int
                 Total number of vertices in the mesh. Must be at least
                 ``faces.max() + 1``.

    Returns
    -------
    components : (n_vertices, ) uint32 array
                 For each vertex the index of the root vertex of its connected
                 component. Vertices that share a component will have the same
                 value (the smallest vertex index in that component).

    Examples
    --------
    Two triangles sharing an edge — one component:

    >>> import navis_fastcore as fastcore
    >>> import numpy as np
    >>> faces = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.uint32)
    >>> fastcore.mesh_connected_components(faces, n_vertices=4)
    array([0, 0, 0, 0], dtype=uint32)

    Two disjoint triangles — two components:

    >>> faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.uint32)
    >>> fastcore.mesh_connected_components(faces, n_vertices=6)
    array([0, 0, 0, 3, 3, 3], dtype=uint32)

    """
    faces = np.asarray(faces, dtype=np.uint32, order="C")

    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(
            f"`faces` must be a 2-D array of shape (N, 3), got {faces.shape}"
        )

    return _fastcore.mesh_connected_components(faces, int(n_vertices))


def _prep_mesh(faces, vertices, n_vertices):
    """Validate and coerce the shared (faces, vertices, n_vertices) arguments."""
    faces = np.asarray(faces, dtype=np.uint32, order="C")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(
            f"`faces` must be a 2-D array of shape (F, 3), got {faces.shape}"
        )

    if vertices is not None:
        vertices = np.asarray(vertices, dtype=np.float64, order="C")
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError(
                f"`vertices` must be a 2-D array of shape (V, 3), got {vertices.shape}"
            )
        if n_vertices is None:
            n_vertices = len(vertices)
        elif int(n_vertices) != len(vertices):
            raise ValueError(
                f"`n_vertices` ({n_vertices}) does not match `len(vertices)` "
                f"({len(vertices)})"
            )
    elif n_vertices is None:
        raise ValueError("Provide either `vertices` (for euclidean edge weights) "
                         "or `n_vertices` (for hop counts).")

    n_vertices = int(n_vertices)
    if len(faces) and faces.max() >= n_vertices:
        raise ValueError(
            f"`faces` references vertex {faces.max()} but n_vertices = {n_vertices}"
        )

    return faces, vertices, n_vertices


def _prep_indices(x, n_nodes, what):
    """Coerce an optional index subset to a contiguous uint32 array."""
    if x is None:
        return None
    x = np.ascontiguousarray(np.asarray(x, dtype=np.uint32).ravel())
    if len(x) and x.max() >= n_nodes:
        raise ValueError(
            f"`{what}` contains vertex {x.max()} but there are only {n_nodes} nodes"
        )
    return x


def geodesic_matrix_mesh(
    faces,
    vertices=None,
    n_vertices=None,
    sources=None,
    targets=None,
    limit=None,
    threads=None,
):
    """Calculate geodesic ("along-the-mesh-edge") distances on a triangle mesh.

    This is the mesh counterpart to :func:`~navis_fastcore.geodesic_matrix`, which
    works on skeletons. Where the skeleton version exploits the tree structure, a
    mesh is a general cyclic graph, so this runs a parallel Dijkstra (or a BFS when
    unweighted) over the vertex adjacency derived from ``faces``.

    Notes
    -----
    This is the distance *along mesh edges*, not the exact surface geodesic: shortest
    paths are constrained to run along edges, so on a coarse mesh they overshoot the
    true surface distance.

    Beware the size of the output. A full ``V x V`` matrix is ~107 GB at V=164k, so
    for anything but a small mesh you want ``sources`` and/or ``targets``. Unlike
    ``scipy.sparse.csgraph.dijkstra`` — which has no notion of targets and always
    materialises all ``V`` columns before you can slice them — ``targets`` here means
    only those columns are ever allocated.

    Parameters
    ----------
    faces :      (F, 3) array
                 Triangular faces given as rows of three vertex indices.
    vertices :   (V, 3) array, optional
                 Vertex coordinates. If provided, edges are weighted by their
                 euclidean length. If ``None``, every edge has weight 1 (i.e. the
                 result is a hop count) and you must pass ``n_vertices``.
    n_vertices : int, optional
                 Total number of vertices. Inferred from ``vertices`` if given.
                 Vertices not referenced by any face are simply unreachable.
    sources :    iterable, optional
                 Source vertex indices. If ``None`` all vertices are used.
    targets :    iterable, optional
                 Target vertex indices. If ``None`` all vertices are used. The order
                 is preserved and duplicates are allowed.
    limit :      float, optional
                 Ignore any nodes further away than this. Vertices at exactly
                 ``limit`` are kept (as in scipy). This prunes the search itself, it
                 is not a post-hoc mask.
    threads :    int, optional
                 Number of threads to use. If ``None`` uses all available cores. Set
                 to 1 if you are already inside a multiprocessing pool, to avoid
                 oversubscribing the machine.

    Returns
    -------
    matrix :     (len(sources), len(targets)) float32 array
                 Geodesic distances. Unreachable pairs — disconnected, or beyond
                 ``limit`` — are set to ``-1``.

    Examples
    --------
    Two triangles sharing the 1-2 edge, forming a unit square. Vertices 0 and 3 are
    the opposite corners, so they are *not* directly connected — the shortest path
    between them goes around, via 1 or 2:

    >>> import navis_fastcore as fastcore
    >>> import numpy as np
    >>> faces = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.uint32)
    >>> vertices = np.array([[0, 0, 0],
    ...                      [1, 0, 0],
    ...                      [0, 1, 0],
    ...                      [1, 1, 0]], dtype=np.float64)
    >>> fastcore.geodesic_matrix_mesh(faces, vertices)
    array([[0.       , 1.       , 1.       , 2.       ],
           [1.       , 0.       , 1.4142135, 1.       ],
           [1.       , 1.4142135, 0.       , 1.       ],
           [2.       , 1.       , 1.       , 0.       ]], dtype=float32)

    Without ``vertices`` every edge has weight 1, so you get hop counts instead — the
    shared diagonal 1-2 is now a single hop rather than sqrt(2):

    >>> fastcore.geodesic_matrix_mesh(faces, n_vertices=4)
    array([[0., 1., 1., 2.],
           [1., 0., 1., 1.],
           [1., 1., 0., 1.],
           [2., 1., 1., 0.]], dtype=float32)

    """
    faces, vertices, n_vertices = _prep_mesh(faces, vertices, n_vertices)

    return _fastcore.geodesic_matrix_mesh(
        faces,
        n_vertices,
        vertices,
        _prep_indices(sources, n_vertices, "sources"),
        _prep_indices(targets, n_vertices, "targets"),
        None if limit is None else float(limit),
        None if threads is None else int(threads),
    )


def geodesic_matrix_graph(
    edges,
    n_nodes,
    weights=None,
    directed=False,
    sources=None,
    targets=None,
    limit=None,
    threads=None,
):
    """Calculate geodesic distances over an arbitrary graph.

    The general form of :func:`~navis_fastcore.geodesic_matrix_mesh`. Unlike
    :func:`~navis_fastcore.geodesic_matrix`, this makes no tree assumption — cycles
    are fine.

    Parameters
    ----------
    edges :      (E, 2) array
                 Edges given as rows of two node indices.
    n_nodes :    int
                 Total number of nodes.
    weights :    (E, ) array, optional
                 Length of each edge. If ``None`` all edges have weight 1 (i.e. the
                 result is a hop count). Must be finite and non-negative. Parallel
                 edges collapse to the shortest.
    directed :   bool, optional
                 If ``True`` an edge ``(u, v)`` may only be traversed from ``u`` to
                 ``v``. If ``False`` (default) the graph is treated as undirected.
    sources :    iterable, optional
                 Source node indices. If ``None`` all nodes are used.
    targets :    iterable, optional
                 Target node indices. If ``None`` all nodes are used.
    limit :      float, optional
                 Ignore any nodes further away than this.
    threads :    int, optional
                 Number of threads to use. If ``None`` uses all available cores.

    Returns
    -------
    matrix :     (len(sources), len(targets)) float32 array
                 Geodesic distances; ``-1`` where unreachable.

    Examples
    --------
    A triangle — a cycle, which the skeleton functions would reject. Note the direct
    0-2 edge has weight 5, so the shortest path goes the long way round via 1:

    >>> import navis_fastcore as fastcore
    >>> import numpy as np
    >>> edges = np.array([[0, 1], [1, 2], [2, 0]], dtype=np.uint32)
    >>> weights = np.array([1, 1, 5], dtype=np.float32)
    >>> fastcore.geodesic_matrix_graph(edges, 3, weights=weights)
    array([[0., 1., 2.],
           [1., 0., 1.],
           [2., 1., 0.]], dtype=float32)

    """
    edges = np.asarray(edges, dtype=np.uint32, order="C")
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError(
            f"`edges` must be a 2-D array of shape (E, 2), got {edges.shape}"
        )

    n_nodes = int(n_nodes)
    if len(edges) and edges.max() >= n_nodes:
        raise ValueError(
            f"`edges` references node {edges.max()} but n_nodes = {n_nodes}"
        )

    if weights is not None:
        weights = np.ascontiguousarray(np.asarray(weights, dtype=np.float32).ravel())
        if len(weights) != len(edges):
            raise ValueError(
                f"`weights` must have one entry per edge: got {len(weights)} "
                f"for {len(edges)} edges"
            )

    return _fastcore.geodesic_matrix_graph(
        edges,
        n_nodes,
        weights,
        bool(directed),
        _prep_indices(sources, n_nodes, "sources"),
        _prep_indices(targets, n_nodes, "targets"),
        None if limit is None else float(limit),
        None if threads is None else int(threads),
    )


def geodesic_nearest_mesh(
    faces,
    vertices=None,
    n_vertices=None,
    sources=None,
    targets=None,
    limit=None,
    threads=None,
):
    """For each source vertex, find the nearest target vertex on a mesh.

    A memory-efficient alternative to :func:`~navis_fastcore.geodesic_matrix_mesh`:
    it keeps only the nearest target and the distance to it, so the output is
    ``O(len(sources))`` rather than ``O(len(sources) * len(targets))``. It is also
    *faster*, because the search stops at the first target it settles instead of
    exploring the whole connected component.

    Parameters
    ----------
    faces :      (F, 3) array
                 Triangular faces given as rows of three vertex indices.
    vertices :   (V, 3) array, optional
                 Vertex coordinates for euclidean edge weights. If ``None``, edges
                 have weight 1 and you must pass ``n_vertices``.
    n_vertices : int, optional
                 Total number of vertices. Inferred from ``vertices`` if given.
    sources :    iterable, optional
                 Source vertex indices. If ``None`` all vertices are used.
    targets :    iterable, optional
                 Target vertex indices. If ``None`` all vertices are used.
    limit :      float, optional
                 Ignore any targets further away than this.
    threads :    int, optional
                 Number of threads to use. If ``None`` uses all available cores.

    Returns
    -------
    distances :  (len(sources), ) float32 array
                 Distance from each source to its nearest target; ``-1`` if no
                 target is reachable.
    nearest :    (len(sources), ) int32 array
                 Vertex index of that nearest target; ``-1`` if none is reachable.

    Notes
    -----
    A source that is itself a target is matched to its nearest *distinct* target,
    never to itself (so the distance is never trivially 0). Ties break towards the
    lower vertex index.

    Examples
    --------
    >>> import navis_fastcore as fastcore
    >>> import numpy as np
    >>> faces = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.uint32)
    >>> vertices = np.array([[0, 0, 0],
    ...                      [1, 0, 0],
    ...                      [0, 1, 0],
    ...                      [1, 1, 0]], dtype=np.float64)
    >>> dists, nearest = fastcore.geodesic_nearest_mesh(
    ...     faces, vertices, sources=[0], targets=[2, 3]
    ... )
    >>> dists
    array([1.], dtype=float32)
    >>> nearest
    array([2], dtype=int32)

    """
    faces, vertices, n_vertices = _prep_mesh(faces, vertices, n_vertices)

    return _fastcore.geodesic_nearest_mesh(
        faces,
        n_vertices,
        vertices,
        _prep_indices(sources, n_vertices, "sources"),
        _prep_indices(targets, n_vertices, "targets"),
        None if limit is None else float(limit),
        None if threads is None else int(threads),
    )


def geodesic_farthest_mesh(
    faces,
    vertices=None,
    n_vertices=None,
    sources=None,
    targets=None,
    limit=None,
    threads=None,
):
    """For each source vertex, find the farthest target vertex on a mesh.

    The mirror image of :func:`~navis_fastcore.geodesic_nearest_mesh`, with the same
    ``O(len(sources))`` memory footprint. Unlike ``nearest``, this cannot stop early
    — it has to settle every target — but the farthest one then comes for free, since
    the search settles vertices in increasing order of distance.

    Parameters
    ----------
    faces :      (F, 3) array
                 Triangular faces given as rows of three vertex indices.
    vertices :   (V, 3) array, optional
                 Vertex coordinates for euclidean edge weights. If ``None``, edges
                 have weight 1 and you must pass ``n_vertices``.
    n_vertices : int, optional
                 Total number of vertices. Inferred from ``vertices`` if given.
    sources :    iterable, optional
                 Source vertex indices. If ``None`` all vertices are used.
    targets :    iterable, optional
                 Target vertex indices. If ``None`` all vertices are used.
    limit :      float, optional
                 Ignore any targets further away than this.
    threads :    int, optional
                 Number of threads to use. If ``None`` uses all available cores.

    Returns
    -------
    distances :  (len(sources), ) float32 array
                 Distance from each source to its farthest target; ``-1`` if no
                 target is reachable.
    farthest :   (len(sources), ) int32 array
                 Vertex index of that farthest target; ``-1`` if none is reachable.

    Notes
    -----
    As with ``nearest``, a source that is itself a target is matched to a *distinct*
    target, never to itself.

    Examples
    --------
    >>> import navis_fastcore as fastcore
    >>> import numpy as np
    >>> faces = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.uint32)
    >>> vertices = np.array([[0, 0, 0],
    ...                      [1, 0, 0],
    ...                      [0, 1, 0],
    ...                      [1, 1, 0]], dtype=np.float64)
    >>> dists, farthest = fastcore.geodesic_farthest_mesh(
    ...     faces, vertices, sources=[0], targets=[2, 3]
    ... )
    >>> farthest
    array([3], dtype=int32)

    """
    faces, vertices, n_vertices = _prep_mesh(faces, vertices, n_vertices)

    return _fastcore.geodesic_farthest_mesh(
        faces,
        n_vertices,
        vertices,
        _prep_indices(sources, n_vertices, "sources"),
        _prep_indices(targets, n_vertices, "targets"),
        None if limit is None else float(limit),
        None if threads is None else int(threads),
    )
