import numpy as np

from . import _fastcore

__all__ = [
    "mesh_connected_components",
    "connected_components_graph",
    "level_set_components",
    "contract_vertices",
    "minimum_spanning_tree",
    "unique_edges",
    "geodesic_matrix_mesh",
    "geodesic_matrix_graph",
    "geodesic_nearest_mesh",
    "geodesic_farthest_mesh",
    "geodesic_predecessors",
    "geodesic_path",
    "geodesic_clusters",
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


def unique_edges(
    faces, vertices=None, return_index=False, return_inverse=False, threads=None
):
    """Unique undirected edges of a triangle mesh.

    A fast, multi-threaded equivalent of ``trimesh.Trimesh.edges_unique``:
    output order, dtype and first-occurrence semantics are identical, so the
    results can be used interchangeably. Each face ``(a, b, c)`` contributes
    the edges ``(a, b), (b, c), (c, a)`` to a conceptual ``3 * F`` edge list;
    edges are normalised to ``[min, max]`` and deduplicated. Self-loop edges
    from degenerate faces are kept, as in trimesh.

    Parameters
    ----------
    faces :          (F, 3) array
                     Triangular faces given as rows of three vertex indices.
                     Must be convertible to ``uint32``.
    vertices :       (V, 3) array, optional
                     Vertex positions. If provided, also return the euclidean
                     length of each unique edge (trimesh's
                     ``edges_unique_length``).
    return_index :   bool
                     Also return, per unique edge, the index of its first
                     occurrence in the ``3 * F`` edge list (trimesh's
                     ``edges_unique_idx``).
    return_inverse : bool
                     Also return, per edge in the ``3 * F`` list, the row of
                     its unique edge (trimesh's ``edges_unique_inverse``;
                     reshape to ``(F, 3)`` for ``faces_unique_edges``).
    threads :        int, optional
                     Size of the thread pool. Defaults to all available cores.

    Returns
    -------
    edges :   (n_unique, 2) int64 array
              Unique edges as ``[min, max]`` rows, sorted ascending with the
              *larger* vertex index as the primary key — the same (not
              lexicographic!) order trimesh produces.
    index :   (n_unique, ) int64 array
              Only if ``return_index=True``.
    inverse : (3 * F, ) int64 array
              Only if ``return_inverse=True``.
    lengths : (n_unique, ) float64 array
              Only if ``vertices`` were provided. Always last in the tuple.

    Examples
    --------
    Two triangles sharing the edge ``(1, 2)`` — five unique edges:

    >>> import navis_fastcore as fastcore
    >>> import numpy as np
    >>> faces = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.uint32)
    >>> fastcore.unique_edges(faces)
    array([[0, 1],
           [0, 2],
           [1, 2],
           [1, 3],
           [2, 3]])
    >>> edges, inv = fastcore.unique_edges(faces, return_inverse=True)
    >>> inv.reshape(-1, 3)  # per-face edge ids (faces_unique_edges)
    array([[0, 2, 1],
           [2, 4, 3]])

    With vertex positions, edge lengths come along for the ride — here the
    unit square split along its diagonal:

    >>> vertices = np.array(
    ...     [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float64
    ... )
    >>> edges, lengths = fastcore.unique_edges(faces, vertices)
    >>> lengths.round(3)
    array([1.   , 1.   , 1.414, 1.   , 1.   ])

    """
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
        if len(faces) and faces.max() >= len(vertices):
            raise ValueError(
                f"`faces` references vertex {faces.max()} but there are only "
                f"{len(vertices)} vertices"
            )

    edges, index, inverse, lengths = _fastcore.unique_edges(
        faces,
        vertices,
        bool(return_index),
        bool(return_inverse),
        threads if threads is None else int(threads),
    )
    if vertices is None and not return_index and not return_inverse:
        return edges
    out = (edges,)
    if return_index:
        out += (index,)
    if return_inverse:
        out += (inverse,)
    if vertices is not None:
        out += (lengths,)
    return out


def _prep_edges(edges, n_nodes):
    """Validate and coerce an (E, 2) edge list against a node count."""
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
    return edges, n_nodes


def connected_components_graph(edges, n_nodes):
    """Find connected components of a graph given as an edge list.

    The edge-list counterpart of :func:`~navis_fastcore.mesh_connected_components`,
    using the same Union-Find: a single integer array of length ``n_nodes``, no
    adjacency list. Use this when the graph is not a triangle mesh, or when you
    already hold the deduplicated edges and would rather not walk the faces again.

    Parameters
    ----------
    edges :      (E, 2) array
                 Edges given as rows of two node indices. Direction is ignored;
                 self-loops and parallel edges are harmless.
    n_nodes :    int
                 Total number of nodes. Nodes not named by any edge form
                 components of size one.

    Returns
    -------
    components : (n_nodes, ) uint32 array
                 For each node, the smallest node index in its component.

    Examples
    --------
    A path 0-1-2, a lone edge 3-4, and an isolated node 5:

    >>> import navis_fastcore as fastcore
    >>> import numpy as np
    >>> edges = np.array([[0, 1], [1, 2], [3, 4]], dtype=np.uint32)
    >>> fastcore.connected_components_graph(edges, n_nodes=6)
    array([0, 0, 0, 3, 3, 5], dtype=uint32)

    """
    edges, n_nodes = _prep_edges(edges, n_nodes)
    return _fastcore.connected_components_graph(edges, n_nodes)


def level_set_components(edges, n_nodes, labels):
    """Find the connected components of every level set at once.

    Given a label per node, this finds the connected components of each subgraph
    induced by the nodes sharing a label — all labels in a single pass, by
    unioning an edge only when its two endpoints agree.

    This is the primitive behind "which nodes were reached by the same wavefront
    and are actually touching", where ``labels`` is a (binned) geodesic distance
    and each component is one ring around the structure.

    The point is that it replaces a *loop*. With a general-purpose graph library
    the same result costs one induced-subgraph construction plus one component
    search per distinct label, so a mesh with a thousand levels pays a thousand
    graph builds; here it is one ``O(E)`` sweep over the edges, and the only
    allocations are three ``n_nodes``-sized integer arrays.

    Parameters
    ----------
    edges :        (E, 2) array
                   Edges given as rows of two node indices.
    n_nodes :      int
                   Total number of nodes.
    labels :       (n_nodes, ) array
                   Label per node, convertible to ``int64``. **Negative labels
                   mark excluded nodes**: they join no component and come back as
                   ``-1``. That is what lets you feed the output of a search that
                   could not reach everything straight in — ``geodesic_matrix_*``
                   returns ``-1`` for unreachable — rather than lumping every
                   unreachable node into one bogus level.

    Returns
    -------
    ids :          (n_nodes, ) int32 array
                   Component of each node in ``[0, n_components)``, or ``-1`` for
                   excluded nodes. Ids are contiguous and assigned in order of
                   first appearance scanning nodes low to high, so they are
                   deterministic and can index straight into an accumulator — no
                   separate ``np.unique`` pass needed.
    n_components : int
                   Number of components found.

    Examples
    --------
    A path 0-1-2-3-4 labelled ``0, 0, 0, 1, 1``: one run per label, so two
    components.

    >>> import navis_fastcore as fastcore
    >>> import numpy as np
    >>> edges = np.array([[0, 1], [1, 2], [2, 3], [3, 4]], dtype=np.uint32)
    >>> ids, n = fastcore.level_set_components(edges, 5, [0, 0, 0, 1, 1])
    >>> ids
    array([0, 0, 0, 1, 1], dtype=int32)

    Nodes sharing a label but *not* touching stay separate — here label 0 appears
    at both ends of the path:

    >>> ids, n = fastcore.level_set_components(edges, 5, [0, 1, 1, 1, 0])
    >>> ids
    array([0, 1, 1, 1, 2], dtype=int32)

    Aggregating per component is then a plain ``np.bincount``:

    >>> sizes = np.bincount(ids[ids >= 0], minlength=n)
    >>> sizes
    array([1, 3, 1])

    """
    edges, n_nodes = _prep_edges(edges, n_nodes)

    labels = np.ascontiguousarray(np.asarray(labels, dtype=np.int64).ravel())
    if len(labels) != n_nodes:
        raise ValueError(
            f"`labels` must have one entry per node: got {len(labels)} for "
            f"{n_nodes} nodes"
        )

    return _fastcore.level_set_components(edges, n_nodes, labels)


def contract_vertices(edges, mapping, threads=None):
    """Contract nodes onto new ids and return the simplified edge list.

    Both endpoints of every edge are pushed through ``mapping``; edges that end up
    with both ends on the same new node (self-loops) are dropped, and the rest are
    deduplicated. This is igraph's ``contract_vertices()`` followed by
    ``simplify()``, fused — and, unlike igraph's version, it does not rewrite a
    graph object in place, so contracting does not cost a copy of the graph.

    Parameters
    ----------
    edges :   (E, 2) array
              Edges given as rows of two node indices.
    mapping : (n_old, ) array
              New id for each old node, i.e. ``mapping[old] = new``. Ids need not
              be contiguous, but the output is only as compact as the ids you
              supply.
    threads : int, optional
              Number of threads to use. If ``None`` uses all available cores.

    Returns
    -------
    edges :   (n_unique, 2) int64 array
              The surviving edges as ``[min, max]`` rows, sorted ascending by
              ``(max, min)`` — the same ordering
              :func:`~navis_fastcore.unique_edges` produces.

    Examples
    --------
    A square 0-1-2-3 with a diagonal, collapsing ``{0, 1} -> 0`` and
    ``{2, 3} -> 1``. The 0-1 and 2-3 edges become self-loops and vanish; the
    remaining three all become 0-1 and collapse to a single edge:

    >>> import navis_fastcore as fastcore
    >>> import numpy as np
    >>> edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0], [0, 2]], dtype=np.uint32)
    >>> fastcore.contract_vertices(edges, [0, 0, 1, 1])
    array([[0, 1]])

    """
    edges = np.asarray(edges, dtype=np.uint32, order="C")
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError(
            f"`edges` must be a 2-D array of shape (E, 2), got {edges.shape}"
        )

    mapping = np.ascontiguousarray(np.asarray(mapping, dtype=np.uint32).ravel())
    if len(edges) and edges.max() >= len(mapping):
        raise ValueError(
            f"`edges` references node {edges.max()} but `mapping` only covers "
            f"{len(mapping)} nodes"
        )

    return _fastcore.contract_vertices(
        edges, mapping, None if threads is None else int(threads)
    )


def minimum_spanning_tree(edges, n_nodes, weights=None, maximize=False, threads=None):
    """Find the minimum (or maximum) spanning forest of a graph.

    Kruskal's algorithm on the same Union-Find as the component search: sort the
    edges by weight, keep the ones that join two different components.
    Disconnected input is fine — each component contributes its own tree, so this
    is really a spanning *forest*, matching igraph's ``spanning_tree()`` and
    scipy's ``minimum_spanning_tree``.

    Parameters
    ----------
    edges :    (E, 2) array
               Edges given as rows of two node indices.
    n_nodes :  int
               Total number of nodes.
    weights :  (E, ) array, optional
               Weight per edge. If ``None`` every edge counts as equal (any
               spanning forest, edges preferred in input order). Must be finite;
               negative weights are allowed.
    maximize : bool
               Return the *maximum* spanning forest instead. This exists so you do
               not have to pass ``1 / weights`` to invert the ordering — a
               transform that both loses precision and blows up on the zero
               weights that legitimately occur.
    threads :  int, optional
               Number of threads to use. If ``None`` uses all available cores.

    Returns
    -------
    indices :  (n_nodes - n_components, ) int64 array
               Row indices *into* ``edges``, ordered by weight — not the edges
               themselves, so you can index whatever per-edge data you hold
               (weights, ids, attributes) with the same array.

    Examples
    --------
    A triangle with weights 1, 2, 3 — the spanning tree takes the two cheap edges
    and rejects the one that would close the cycle:

    >>> import navis_fastcore as fastcore
    >>> import numpy as np
    >>> edges = np.array([[0, 1], [1, 2], [0, 2]], dtype=np.uint32)
    >>> weights = np.array([1, 2, 3], dtype=np.float32)
    >>> keep = fastcore.minimum_spanning_tree(edges, 3, weights)
    >>> edges[keep]
    array([[0, 1],
           [1, 2]], dtype=uint32)

    Ask for the maximum instead and it takes the two expensive ones:

    >>> keep = fastcore.minimum_spanning_tree(edges, 3, weights, maximize=True)
    >>> edges[keep]
    array([[0, 2],
           [1, 2]], dtype=uint32)

    """
    edges, n_nodes = _prep_edges(edges, n_nodes)

    if weights is not None:
        weights = np.ascontiguousarray(np.asarray(weights, dtype=np.float32).ravel())
        if len(weights) != len(edges):
            raise ValueError(
                f"`weights` must have one entry per edge: got {len(weights)} "
                f"for {len(edges)} edges"
            )

    return _fastcore.minimum_spanning_tree(
        edges,
        n_nodes,
        weights,
        bool(maximize),
        None if threads is None else int(threads),
    )


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


def _prep_weights(weights, edges):
    """Coerce optional edge weights to a contiguous float32 array of the right length."""
    if weights is None:
        return None
    weights = np.ascontiguousarray(np.asarray(weights, dtype=np.float32).ravel())
    if len(weights) != len(edges):
        raise ValueError(
            f"`weights` must have one entry per edge: got {len(weights)} "
            f"for {len(edges)} edges"
        )
    return weights


def geodesic_predecessors(
    edges,
    n_nodes,
    weights=None,
    directed=False,
    sources=None,
    limit=None,
    threads=None,
):
    """Shortest path tree(s) - distances *and* the route to each node.

    The predecessor-returning counterpart to
    :func:`~navis_fastcore.geodesic_matrix_graph`. Use this when you need the path
    itself; use ``geodesic_matrix_graph`` when the distance is enough, and
    :func:`~navis_fastcore.geodesic_path` when you want the node sequences rather
    than the raw chains.

    Because this takes a bare edge list there is no index to build or invalidate
    between calls, which is what algorithms that re-weight the graph every iteration
    (TEASAR zeroes the edges along each path it extracts, then searches again) need.

    Parameters
    ----------
    edges :      (E, 2) array
                 Edges given as rows of two node indices.
    n_nodes :    int
                 Total number of nodes.
    weights :    (E, ) array, optional
                 Length of each edge. If ``None`` all edges weigh 1. Must be finite
                 and non-negative. **Zero weights are explicitly allowed** - they
                 are how a penalised-path search makes an already-extracted route
                 free to re-traverse.
    directed :   bool, optional
                 If ``True`` an edge ``(u, v)`` may only be traversed from ``u`` to
                 ``v``.
    sources :    iterable, optional
                 Source nodes, one shortest path tree each. If ``None`` all nodes
                 are used.
    limit :      float, optional
                 Ignore any nodes further away than this.
    threads :    int, optional
                 Number of threads to use. If ``None`` uses all available cores.

    Returns
    -------
    distances :  (len(sources), n_nodes) float32 array
                 As ``geodesic_matrix_graph``: ``-1`` where unreachable.
    predecessors : (len(sources), n_nodes) int32 array
                 For each node, the node before it on the shortest path back to that
                 row's source. ``-1`` for the source itself and for unreachable
                 nodes - so a single ``>= 0`` test both walks the path and
                 terminates it.

    Notes
    -----
    Among equal-length paths the predecessor is the one reached first in the
    search's own deterministic order, so results are reproducible run to run and do
    not depend on ``threads``.

    Examples
    --------
    A triangle whose direct 0-2 edge is expensive, so the shortest path to 2 goes
    the long way round via 1:

    >>> import navis_fastcore as fastcore
    >>> import numpy as np
    >>> edges = np.array([[0, 1], [1, 2], [2, 0]], dtype=np.uint32)
    >>> weights = np.array([1, 1, 5], dtype=np.float32)
    >>> dists, pred = fastcore.geodesic_predecessors(
    ...     edges, 3, weights=weights, sources=[0]
    ... )
    >>> dists
    array([[0., 1., 2.]], dtype=float32)
    >>> pred
    array([[-1,  0,  1]], dtype=int32)

    """
    edges, n_nodes = _prep_edges(edges, n_nodes)
    return _fastcore.geodesic_predecessors(
        edges,
        n_nodes,
        _prep_weights(weights, edges),
        bool(directed),
        _prep_indices(sources, n_nodes, "sources"),
        None if limit is None else float(limit),
        None if threads is None else int(threads),
    )


def geodesic_path(edges, n_nodes, source, targets, weights=None, directed=False):
    """Node sequences of the shortest paths from ``source`` to each target.

    The convenience form of :func:`~navis_fastcore.geodesic_predecessors` for the
    common single-source case: one search, with the predecessor chains walked in
    Rust rather than in Python. Because every target is known up front the search
    also stops as soon as the last of them settles, so a short path in a large graph
    costs a ball, not a sweep.

    Parameters
    ----------
    edges :      (E, 2) array
                 Edges given as rows of two node indices.
    n_nodes :    int
                 Total number of nodes.
    source :     int
                 Source node index.
    targets :    iterable
                 Target node indices.
    weights :    (E, ) array, optional
                 Length of each edge. If ``None`` all edges weigh 1. Zero weights
                 are allowed.
    directed :   bool, optional
                 If ``True`` an edge ``(u, v)`` may only be traversed from ``u`` to
                 ``v``.

    Returns
    -------
    paths :      list of (L, ) uint32 arrays
                 One per target, ordered source-first / target-last (so ``path[0]``
                 is always ``source``). Empty array where the target is
                 unreachable; the single-element ``[source]`` where the target *is*
                 the source.

    Examples
    --------
    >>> import navis_fastcore as fastcore
    >>> import numpy as np
    >>> edges = np.array([[0, 1], [1, 2], [2, 0]], dtype=np.uint32)
    >>> weights = np.array([1, 1, 5], dtype=np.float32)
    >>> fastcore.geodesic_path(edges, 3, 0, [2], weights=weights)
    [array([0, 1, 2], dtype=uint32)]

    An unreachable target gives an empty path:

    >>> edges = np.array([[0, 1], [2, 3]], dtype=np.uint32)
    >>> fastcore.geodesic_path(edges, 4, 0, [1, 3])
    [array([0, 1], dtype=uint32), array([], dtype=uint32)]

    """
    edges, n_nodes = _prep_edges(edges, n_nodes)
    source = int(source)
    if not 0 <= source < n_nodes:
        raise ValueError(f"`source` is node {source} but n_nodes = {n_nodes}")

    targets = _prep_indices(targets, n_nodes, "targets")
    if targets is None:
        raise ValueError("`targets` must be given")

    return _fastcore.geodesic_path(
        edges,
        n_nodes,
        source,
        targets,
        _prep_weights(weights, edges),
        bool(directed),
    )


def geodesic_clusters(edges, n_nodes, max_dist, weights=None, seeds=None):
    """Greedily partition nodes into connected clusters of bounded radius.

    Repeatedly takes an unassigned node as a seed and grows a cluster outwards from
    it, absorbing any node reachable within ``max_dist`` that no earlier cluster has
    already claimed. Collapsing each cluster to its centroid gives a coarser graph
    whose nodes are spaced by roughly ``max_dist``, which is what makes this useful
    as mesh or skeleton downsampling.

    The radius is the **true geodesic distance from the seed**, not the length of
    the walk that happened to reach it - so a node close to a seed is never excluded
    merely because a traversal arrived at it the long way round.

    Parameters
    ----------
    edges :      (E, 2) array
                 Edges given as rows of two node indices. Treated as undirected.
    n_nodes :    int
                 Total number of nodes. Isolated nodes each become their own
                 cluster.
    max_dist :   float
                 Maximum distance from a cluster's seed. Must be finite and
                 non-negative.
    weights :    (E, ) array, optional
                 Length of each edge. If ``None`` all edges weigh 1, i.e.
                 ``max_dist`` is a hop count.
    seeds :      iterable, optional
                 Nodes to use as seeds, in order of preference. Any node left
                 unassigned afterwards becomes a seed in ascending index order. If
                 ``None``, seeds are picked in ascending index order throughout. A
                 seed an earlier cluster already claimed is skipped.

    Returns
    -------
    labels :     (n_nodes, ) int32 array
                 Cluster of each node, contiguous in ``[0, n_clusters)`` and
                 numbered in the order the clusters were grown. Every node is
                 labelled.
    n_clusters : int

    Notes
    -----
    The greedy outer loop is inherently sequential - cluster *n* depends on
    everything every earlier cluster claimed - so there is no ``threads`` argument.

    Examples
    --------
    A path 0-1-...-5 with a radius of one hop:

    >>> import navis_fastcore as fastcore
    >>> import numpy as np
    >>> edges = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]], dtype=np.uint32)
    >>> labels, n = fastcore.geodesic_clusters(edges, 6, 1)
    >>> labels
    array([0, 0, 1, 1, 2, 2], dtype=int32)
    >>> n
    3

    Seeding from the middle instead:

    >>> labels, n = fastcore.geodesic_clusters(edges, 6, 1, seeds=[3])
    >>> labels
    array([1, 1, 0, 0, 0, 2], dtype=int32)

    """
    edges, n_nodes = _prep_edges(edges, n_nodes)
    max_dist = float(max_dist)
    if not np.isfinite(max_dist) or max_dist < 0:
        raise ValueError(f"`max_dist` must be finite and non-negative, got {max_dist}")

    return _fastcore.geodesic_clusters(
        edges,
        n_nodes,
        max_dist,
        _prep_weights(weights, edges),
        _prep_indices(seeds, n_nodes, "seeds"),
    )
