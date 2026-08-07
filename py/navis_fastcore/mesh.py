import warnings

import numpy as np

from . import _fastcore

__all__ = [
    "mesh_connected_components",
    "connected_components_graph",
    "level_set_components",
    "contract_vertices",
    "minimum_spanning_tree",
    "bridges",
    "parents_from_edges",
    "geodesic_mst_mesh",
    "geodesic_mst_graph",
    "unique_edges",
    "geodesic_matrix_mesh",
    "geodesic_matrix_graph",
    "geodesic_nearest_mesh",
    "geodesic_farthest_mesh",
    "geodesic_predecessors",
    "geodesic_path",
    "geodesic_clusters",
    "simplify_mesh",
    "simplify_mesh_lossless",
    "smooth_mesh",
    "GeodesicGraph",
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
    output order and first-occurrence semantics are identical, so the results
    can be used interchangeably. The one difference is the dtype: edges come
    back as ``uint32``, not trimesh's ``int64``, because they are node ids.
    Each face ``(a, b, c)`` contributes the edges ``(a, b), (b, c), (c, a)`` to
    a conceptual ``3 * F`` edge list; edges are normalised to ``[min, max]`` and
    deduplicated. Self-loop edges from degenerate faces are kept, as in trimesh.

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
    edges :   (n_unique, 2) uint32 array
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
           [2, 3]], dtype=uint32)
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
    faces = _prep_faces(faces)

    if vertices is not None:
        vertices = _prep_vertices(vertices)
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
    edges :   (n_unique, 2) uint32 array
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
    array([[0, 1]], dtype=uint32)

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
               negative weights are allowed. A float64 array is compared at that
               width rather than being narrowed, which can change *which* edges are
               kept where two weights tie at float32 and not at float64.
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

    return _fastcore.minimum_spanning_tree(
        edges,
        n_nodes,
        # No `dtype`: the output is row indices. The weights' own width is honoured
        # because it decides the sort order where two weights are close enough to
        # compare equal at float32 and not at float64.
        _prep_weights(weights, edges)[0],
        bool(maximize),
        None if threads is None else int(threads),
    )


def bridges(edges, n_nodes):
    """Find the edges whose removal would disconnect their component.

    Tarjan's algorithm: one depth-first sweep tracking, per node, the earliest
    node reachable from its subtree by a single back edge. A tree edge ``(u, v)``
    is a bridge exactly when nothing under ``v`` can climb above it, i.e. there is
    no second route around it.

    The counterpart to :func:`~navis_fastcore.minimum_spanning_tree` rather than a
    variant of it: the MST asks which edges to *keep* to stay connected, this asks
    which ones may not be *dropped*. That is the question behind "prune this graph
    but do not shatter it", where you have a set of edges you would like gone and
    need to know which of them are load-bearing.

    Parameters
    ----------
    edges :   (E, 2) array
              Edges given as rows of two node indices. Treated as undirected.
    n_nodes : int
              Total number of nodes.

    Returns
    -------
    mask :    (E, ) bool array
              ``True`` for each edge that is a bridge. A mask rather than a list of
              indices because the next move is nearly always to filter a parallel
              array; ``np.flatnonzero`` recovers the indices when it is not.

    Notes
    -----
    Parallel edges are honoured: two nodes joined twice are joined by a cycle, so
    neither of those edges is a bridge. Self-loops are never bridges. This is why
    ``bridges`` does not share the deduplicated adjacency the geodesic searches in
    this module use — that would fuse a parallel pair into one edge and report a
    bridge that is not there.

    Examples
    --------
    Every edge of a tree is a bridge:

    >>> import navis_fastcore as fastcore
    >>> import numpy as np
    >>> path = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.uint32)
    >>> fastcore.bridges(path, 4)
    array([ True,  True,  True])

    Close it into a ring and none of them is:

    >>> ring = np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.uint32)
    >>> fastcore.bridges(ring, 4)
    array([False, False, False, False])

    Two triangles joined by a single edge — only the link:

    >>> edges = np.array([[0, 1], [1, 2], [2, 0],
    ...                   [3, 4], [4, 5], [5, 3],
    ...                   [2, 3]], dtype=np.uint32)
    >>> fastcore.bridges(edges, 6)
    array([False, False, False, False, False, False,  True])

    """
    edges, n_nodes = _prep_edges(edges, n_nodes)
    return _fastcore.bridges(edges, n_nodes)


def parents_from_edges(edges, n_nodes, weights=None, roots=None):
    """Orient a graph into a rooted spanning forest — one parent per node.

    The missing half of "I have an edge list and I want a tree".
    :func:`~navis_fastcore.minimum_spanning_tree` picks *which* edges survive; this
    picks which way they point, which is what turns a bag of undirected edges into
    something you can walk, root, or write out as SWC. Cycles in the input are
    fine — each component contributes a spanning tree of itself, so this doubles as
    the cycle-breaker ``networkx.bfs_tree`` is usually pressed into.

    One search covers the whole graph. The obvious construction — a shortest-path
    tree per component — is what
    :func:`~navis_fastcore.geodesic_predecessors` gives you, and it costs
    ``O(components * n_nodes)`` in *output alone*: on a mesh that shatters into four
    thousand specks that is a two-gigabyte array to answer a question whose answer
    is one ``n_nodes``-long column. Here the components are swept one after another
    into that single column, so the cost is ``O(V + E)`` however finely the graph is
    fragmented.

    Parameters
    ----------
    edges :   (E, 2) array
              Edges given as rows of two node indices. Direction is ignored.
    n_nodes : int
              Total number of nodes. Nodes named by no edge are isolated roots.
    weights : (E, ) array, optional
              Length of each edge. ``None`` gives the breadth-first tree; weights
              give the shortest-path tree, which is a different (and generally
              deeper) spanning tree. Neither is the minimum spanning tree — for
              that, run :func:`~navis_fastcore.minimum_spanning_tree` first and
              orient the edges it keeps. A float64 array is accumulated at that
              width rather than being narrowed, which can change which tree comes
              out. There is no ``dtype`` argument because both outputs are ids.
    roots :   iterable, optional
              Nodes to root at. If ``None`` each component is rooted at its lowest
              node index — the same representative
              :func:`~navis_fastcore.connected_components_graph` labels components
              by. Components holding none of ``roots`` fall back to that, so the
              result is always a complete forest. Two roots in the *same* component
              split it into two trees, which is well defined (each node goes to
              whichever root is nearer) and occasionally what you want.

    Returns
    -------
    parents : (n_nodes, ) int32 array
              Parent of each node, ``-1`` for a root.
    order :   (n_nodes, ) uint32 array
              Every node in the order it settled. A node always settles after its
              parent, so this is a topological order — relabel by it and parents are
              guaranteed to have lower ids than their children, which is exactly the
              SWC requirement. It comes free: the search already visits nodes in this
              order, and deriving it afterwards from ``parents`` would cost another
              traversal.

    Notes
    -----
    Among equal-length routes the parent is whichever settled first, which is
    deterministic but otherwise arbitrary — as it is for any spanning tree of a
    graph with more than one.

    Examples
    --------
    A path, written "backwards" to show the orientation comes from the search and
    not from the order the endpoints happen to be in:

    >>> import navis_fastcore as fastcore
    >>> import numpy as np
    >>> edges = np.array([[1, 0], [2, 1], [3, 2]], dtype=np.uint32)
    >>> parents, order = fastcore.parents_from_edges(edges, 4)
    >>> parents
    array([-1,  0,  1,  2], dtype=int32)

    Root it at the far end instead and every link reverses:

    >>> parents, order = fastcore.parents_from_edges(edges, 4, roots=[3])
    >>> parents
    array([ 1,  2,  3, -1], dtype=int32)
    >>> order
    array([3, 2, 1, 0], dtype=uint32)

    Cycles are broken; two components each get their own root:

    >>> edges = np.array([[0, 1], [1, 2], [2, 0], [4, 5]], dtype=np.uint32)
    >>> parents, order = fastcore.parents_from_edges(edges, 6)
    >>> parents
    array([-1,  0,  0, -1, -1,  4], dtype=int32)

    ``order`` relabels a forest so parents come before their children:

    >>> new_ids = np.empty(len(order), dtype=np.int64)
    >>> new_ids[order] = np.arange(len(order))
    >>> new_parents = np.where(parents < 0, -1, new_ids[parents])
    >>> bool(np.all(new_parents < new_ids))
    True

    """
    edges, n_nodes = _prep_edges(edges, n_nodes)
    return _fastcore.parents_from_edges(
        edges,
        n_nodes,
        # No `dtype`: both outputs are ids. The weights' own width still decides
        # which spanning tree comes out, so it is honoured rather than flattened.
        _prep_weights(weights, edges)[0],
        _prep_indices(roots, n_nodes, "roots"),
    )


def _prep_mst_nodes(nodes, n_nodes):
    """Coerce and check the node subset the geodesic MST spans."""
    nodes = _prep_indices(nodes, n_nodes, "nodes", unique=True)
    if nodes is None:
        raise ValueError("`nodes` is required: pass the nodes you want to span.")
    return nodes


def geodesic_mst_mesh(
    faces, nodes, vertices=None, n_vertices=None, limit=None, threads=None, dtype=None
):
    """Minimum spanning tree over a subset of mesh vertices, by geodesic distance.

    The tree that reconnects a scatter of surviving vertices through the mesh they
    were carved out of — the last step of a skeletonisation, where the mesh has been
    thinned to a few thousand vertices that must be rejoined along the surface
    rather than through space.

    The obvious route is to ask for the ``k x k`` geodesic matrix and hand it to a
    matrix MST. That materialises ``k**2`` distances to use ``k - 1`` of them —
    400 MB at ``k = 10_000``, before the ``O(k**2)`` MST itself — and it needs ``k``
    separate searches to fill. This never forms the matrix. Instead, following
    Mehlhorn's construction for the distance network, one multi-source search
    partitions *every* vertex by which of ``nodes`` is nearest, and then each mesh
    edge whose endpoints fall in different cells offers one candidate: joining their
    two owners at ``d(u) + w(u, v) + d(v)``. An MST over those candidates is an MST
    of the full distance network, so one sweep and one Kruskal replace ``k`` searches
    and a dense matrix.

    The returned weights come back exactly equal to the geodesic distances between
    the pairs they join, so they are usable as lengths and not merely as an ordering.

    Parameters
    ----------
    faces :      (F, 3) array
                 Triangular faces given as rows of three vertex indices.
    nodes :      (K, ) array
                 Vertices to span. Must be distinct.
    vertices :   (V, 3) array, optional
                 Vertex positions. If given, mesh edges are weighted by their
                 euclidean length; if ``None`` distances are hop counts.
    n_vertices : int, optional
                 Total number of vertices. Inferred from ``vertices`` when that is
                 given; required otherwise.
    limit :      float, optional
                 Do not join vertices farther apart than this. The result is then
                 the MST of the graph on ``nodes`` keeping only pairs within
                 ``limit``, which is a *forest* when that graph is disconnected —
                 the same trade ``scipy.sparse.csgraph.dijkstra(limit=...)`` offers,
                 except that here it also prunes the sweep, so it buys time rather
                 than merely discarding results.
    threads :    int, optional
                 Number of threads to use. If ``None`` uses all available cores.
    dtype :      float32 | float64, optional
                 Width the distances are accumulated and returned at. Defaults to
                 float32. Unlike the graph functions there is no input dtype to read
                 this off: ``vertices`` are *coordinates*, taken at float64 either
                 way, and each edge length is computed from them at that width and
                 rounded once on the way into the graph. Ask for float64 when the
                 paths are long enough for the per-hop accumulation to matter — and
                 note the output doubles in size.

    Returns
    -------
    edges :      (M, 2) int64 array
                 Rows of *positions in* ``nodes``, not vertex indices — so
                 ``nodes[edges]`` maps back, and any per-node data you hold indexes
                 the same way. Ascending by weight, as
                 :func:`~navis_fastcore.minimum_spanning_tree`.
    weights :    (M, ) array
                 Geodesic distance across each of those edges, in the resolved
                 ``dtype``.

    ``M`` is ``len(nodes) - 1`` when every node can reach every other within
    ``limit``, and less when they cannot: vertices in different components of the
    mesh are never joined.

    Examples
    --------
    Two triangles sharing an edge, spanning three of the four vertices:

    >>> import navis_fastcore as fastcore
    >>> import numpy as np
    >>> faces = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.uint32)
    >>> verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float)
    >>> edges, weights = fastcore.geodesic_mst_mesh(faces, [0, 1, 3], verts)
    >>> edges
    array([[0, 1],
           [1, 2]])
    >>> np.round(weights, 3)
    array([1., 1.], dtype=float32)

    The rows index ``nodes``, so map them back to vertex ids yourself:

    >>> np.asarray([0, 1, 3])[edges]
    array([[0, 1],
           [1, 3]])

    """
    faces, vertices, n_vertices = _prep_mesh(faces, vertices, n_vertices)
    nodes = _prep_mst_nodes(nodes, n_vertices)

    return _fastcore.geodesic_mst_mesh(
        faces,
        n_vertices,
        nodes,
        vertices,
        _prep_limit(limit),
        None if threads is None else int(threads),
        _width_of(dtype),
    )


def geodesic_mst_graph(
    edges, n_nodes, nodes, weights=None, limit=None, threads=None, dtype=None
):
    """Minimum spanning tree over a subset of graph nodes, by geodesic distance.

    The edge-list form of :func:`~navis_fastcore.geodesic_mst_mesh`, which explains
    why this never builds the ``k x k`` distance matrix the question seems to call
    for. Always undirected — a minimum spanning tree of a directed graph is a
    different problem (an arborescence) with a different algorithm.

    Parameters
    ----------
    edges :   (E, 2) array
              Edges given as rows of two node indices.
    n_nodes : int
              Total number of nodes.
    nodes :   (K, ) array
              Nodes to span. Must be distinct.
    weights : (E, ) array, optional
              Length of each edge. If ``None`` all edges weigh 1, i.e. distances are
              hop counts.
    limit :   float, optional
              Do not join nodes farther apart than this. See
              :func:`~navis_fastcore.geodesic_mst_mesh`.
    threads : int, optional
              Number of threads to use. If ``None`` uses all available cores.
    dtype :   float32 | float64, optional
              Width the distances are accumulated and returned at. If ``None``
              (default) it follows ``weights``: float64 in, float64 out.

    Returns
    -------
    edges :   (M, 2) int64 array
              Rows of *positions in* ``nodes``, ascending by weight.
    weights : (M, ) array
              Geodesic distance across each of those edges, in the resolved
              ``dtype``.

    Examples
    --------
    Two paths joined at their middle. Spanning the four endpoints costs three edges,
    each two hops long:

    >>> import navis_fastcore as fastcore
    >>> import numpy as np
    >>> edges = np.array([[0, 1], [1, 2], [1, 3], [3, 4]], dtype=np.uint32)
    >>> mst, weights = fastcore.geodesic_mst_graph(edges, 5, nodes=[0, 2, 4])
    >>> np.asarray([0, 2, 4])[mst]
    array([[0, 2],
           [0, 4]])
    >>> weights
    array([2., 3.], dtype=float32)

    Nodes in different components are never joined, so the result is a forest:

    >>> edges = np.array([[0, 1], [2, 3]], dtype=np.uint32)
    >>> mst, weights = fastcore.geodesic_mst_graph(edges, 4, nodes=[0, 1, 2, 3])
    >>> mst
    array([[0, 1],
           [2, 3]])

    """
    edges, n_nodes = _prep_edges(edges, n_nodes)
    nodes = _prep_mst_nodes(nodes, n_nodes)

    weights, float64 = _prep_weights(weights, edges, dtype)
    return _fastcore.geodesic_mst_graph(
        edges,
        n_nodes,
        nodes,
        weights,
        _prep_limit(limit),
        None if threads is None else int(threads),
        float64,
    )


def _prep_faces(faces):
    """Coerce an (F, 3) face array to contiguous uint32.

    Split out of `_prep_mesh` because `unique_edges` and the `caps` functions want
    faces without the `n_vertices` bookkeeping that surrounds them there.
    """
    faces = np.asarray(faces, dtype=np.uint32, order="C")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(
            f"`faces` must be a 2-D array of shape (F, 3), got {faces.shape}"
        )
    return faces


def _prep_vertices(vertices):
    """Coerce a (V, 3) coordinate array to contiguous float64."""
    vertices = np.asarray(vertices, dtype=np.float64, order="C")
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(
            f"`vertices` must be a 2-D array of shape (V, 3), got {vertices.shape}"
        )
    return vertices


def _prep_mesh(faces, vertices, n_vertices):
    """Validate and coerce the shared (faces, vertices, n_vertices) arguments."""
    faces = _prep_faces(faces)

    if vertices is not None:
        vertices = _prep_vertices(vertices)
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


def _prep_indices(x, n_nodes, what, unique=False):
    """Coerce an optional index subset to a contiguous uint32 array.

    ``unique`` additionally rejects repeats, for the callers that renumber the
    subset and so cannot give one node two new ids. The core asserts it too, but
    only after the wrapper has handed the array over — so checking here is what
    turns a caller's mistake into a ``ValueError`` rather than a panic.
    """
    if x is None:
        return None
    x = np.ascontiguousarray(np.asarray(x, dtype=np.uint32).ravel())
    if len(x) and x.max() >= n_nodes:
        raise ValueError(
            f"`{what}` contains vertex {x.max()} but there are only {n_nodes} nodes"
        )
    if unique and len(x) and _has_duplicates(x, n_nodes):
        raise ValueError(f"`{what}` must not contain duplicates")
    return x


def _has_duplicates(x, n_nodes):
    """Does this range-checked index array repeat itself?

    Two algorithms, because the callers sit on opposite sides of the crossover.
    Sorting costs `O(k log k)` and touches only the subset; a bitmap over the node
    space costs `O(n_nodes + k)` and touches all of it. `geodesic_mst_*` spans a
    few hundred nodes of a large graph, where the sort never leaves cache;
    `GeodesicGraph.subset` routinely keeps most of the graph, where the sort is 25x
    the bitmap (43 ms vs 1.8 ms at `k = n_nodes = 1M`). The crossover measures at
    1-3% of `n_nodes`, and picking wrong costs well under a millisecond either way.
    """
    if len(x) * 64 < n_nodes:
        return len(np.unique(x)) != len(x)
    seen = np.zeros(n_nodes, dtype=bool)
    seen[x] = True
    return int(np.count_nonzero(seen)) != len(x)


def _prep_limit(limit):
    """Coerce an optional distance bound, rejecting what a search cannot use.

    ``None`` and ``+inf`` are the same input to the core, which reads a missing
    bound as infinity, so both pass through as ``None``.
    """
    if limit is None:
        return None
    limit = float(limit)
    if np.isnan(limit) or limit < 0:
        raise ValueError(f"`limit` must be non-negative, got {limit}")
    return None if np.isinf(limit) else limit


def geodesic_matrix_mesh(
    faces,
    vertices=None,
    n_vertices=None,
    sources=None,
    targets=None,
    limit=None,
    threads=None,
    dtype=None,
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
    dtype :      float32 | float64, optional
                 Width the distances are accumulated and returned at. Defaults to
                 float32. Unlike the graph functions there is no input dtype to read
                 this off: ``vertices`` are *coordinates*, taken at float64 either
                 way, and each edge length is computed from them at that width and
                 rounded once on the way into the graph. Ask for float64 when the
                 paths are long enough for the per-hop accumulation to matter — and
                 note the output doubles in size.

    Returns
    -------
    matrix :     (len(sources), len(targets)) array
                 Geodesic distances in the resolved ``dtype``. Unreachable pairs —
                 disconnected, or beyond ``limit`` — are set to ``-1``.

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
        _prep_limit(limit),
        None if threads is None else int(threads),
        _width_of(dtype),
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
    dtype=None,
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
    dtype :      float32 | float64, optional
                 Width the distances are accumulated and returned at. If ``None``
                 (default) it follows ``weights``: a float64 array in gives float64
                 out, anything else gives float32. See the note below.

    Returns
    -------
    matrix :     (len(sources), len(targets)) array
                 Geodesic distances in the resolved dtype; ``-1`` where unreachable.

    Notes
    -----
    Dijkstra sums one weight per hop, so a path of ``k`` hops carries up to ``k``
    roundings. float32 is right for mesh and skeleton work — a 24-bit mantissa
    resolves a 100 mm neuron to ~6 nm, and the matrix is by far the largest thing
    this allocates. float64 earns its keep when the *accumulation* is long rather
    than the graph large (tens of thousands of hops), when weights span a wide
    dynamic range, or when you are comparing against
    ``scipy.sparse.csgraph``, which works in float64 unconditionally.

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

    Hand it float64 weights and the distances come back float64:

    >>> fastcore.geodesic_matrix_graph(edges, 3, weights=weights.astype(np.float64))
    array([[0., 1., 2.],
           [1., 0., 1.],
           [2., 1., 0.]])

    ``dtype`` overrides that in either direction — here asking for float64 from a
    float32 input, which is what you want when the weights were measured coarsely
    but the paths are long enough for the accumulation to matter:

    >>> fastcore.geodesic_matrix_graph(edges, 3, weights=weights,
    ...                                dtype=np.float64).dtype
    dtype('float64')

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

    weights, float64 = _prep_weights(weights, edges, dtype)
    return _fastcore.geodesic_matrix_graph(
        edges,
        n_nodes,
        weights,
        bool(directed),
        _prep_indices(sources, n_nodes, "sources"),
        _prep_indices(targets, n_nodes, "targets"),
        _prep_limit(limit),
        None if threads is None else int(threads),
        float64,
    )


def geodesic_nearest_mesh(
    faces,
    vertices=None,
    n_vertices=None,
    sources=None,
    targets=None,
    limit=None,
    threads=None,
    dtype=None,
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
    dtype :      float32 | float64, optional
                 Width the distances are accumulated and returned at. Defaults to
                 float32. Unlike the graph functions there is no input dtype to read
                 this off: ``vertices`` are *coordinates*, taken at float64 either
                 way, and each edge length is computed from them at that width and
                 rounded once on the way into the graph. Ask for float64 when the
                 paths are long enough for the per-hop accumulation to matter — and
                 note the output doubles in size.

    Returns
    -------
    distances :  (len(sources), ) array
                 Distance from each source to its nearest target, in the resolved
                 ``dtype``; ``-1`` if no target is reachable.
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
        _prep_limit(limit),
        None if threads is None else int(threads),
        _width_of(dtype),
    )


def geodesic_farthest_mesh(
    faces,
    vertices=None,
    n_vertices=None,
    sources=None,
    targets=None,
    limit=None,
    threads=None,
    dtype=None,
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
    dtype :      float32 | float64, optional
                 Width the distances are accumulated and returned at. Defaults to
                 float32. Unlike the graph functions there is no input dtype to read
                 this off: ``vertices`` are *coordinates*, taken at float64 either
                 way, and each edge length is computed from them at that width and
                 rounded once on the way into the graph. Ask for float64 when the
                 paths are long enough for the per-hop accumulation to matter — and
                 note the output doubles in size.

    Returns
    -------
    distances :  (len(sources), ) array
                 Distance from each source to its farthest target, in the resolved
                 ``dtype``; ``-1`` if no target is reachable.
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
        _prep_limit(limit),
        None if threads is None else int(threads),
        _width_of(dtype),
    )


#: The two widths a geodesic search runs at. ``float16`` is absent deliberately:
#: Dijkstra accumulates one addition per hop, and float16 runs out of mantissa
#: within a handful of them.
_F32 = np.dtype(np.float32)
_F64 = np.dtype(np.float64)


def _width_of(dtype):
    """Validate an explicit ``dtype`` and say whether it is float64.

    For the callers whose width is *only* ever explicit — the mesh functions, whose
    ``vertices`` are coordinates rather than distances and so carry no width to read.
    """
    if dtype is None:
        return False
    d = np.dtype(dtype)
    if d not in (_F32, _F64):
        raise ValueError(
            f"`dtype` must be float32 or float64, got {d}. Distances are "
            "accumulated one edge at a time, so anything narrower runs out "
            "of mantissa within a few hops."
        )
    return d == _F64


def _prep_weights(weights, edges, dtype=None):
    """Coerce optional edge weights, and resolve the width, in one step.

    Returns ``(weights, float64)`` — deliberately both, because the extension needs
    the array *and* a flag for the unweighted case, and the two must agree. Resolving
    the width separately and then remembering to pass it here as well is exactly the
    mistake this signature makes unavailable.

    The width is the caller's ``dtype`` if given, else the weights' *own* dtype —
    float64 in, float64 out, the rule :func:`~navis_fastcore.linkage` already follows
    for score matrices — else float32.

    Only something that actually carries a ``dtype`` counts as having stated one. A
    plain list does not: ``np.asarray([1.0, 2.0])`` is float64 by numpy's default
    rather than by anyone's intent, and honouring that would quietly double the output
    of every caller who passes a list of Python floats. Pass ``dtype`` if that is what
    you meant.

    Weights already at the resolved width pass through without a copy; anything else
    is cast, which is what makes a list of Python floats or a column of ints work.
    """
    if dtype is not None:
        float64 = _width_of(dtype)
    else:
        # `is not None` first, and not merely for tidiness: `np.dtype` reads `None` as
        # float64 and `np.dtype.__eq__` coerces its operand, so `None == _F64` is
        # *True* and every unweighted call would quietly come back float64. A dtype
        # numpy cannot name (a pandas nullable column) compares unequal rather than
        # raising, and so lands on the float32 default, which casts fine below.
        given = getattr(weights, "dtype", None)
        float64 = given is not None and given == _F64

    if weights is None:
        return None, float64

    weights = np.ascontiguousarray(
        np.asarray(weights, dtype=_F64 if float64 else _F32).ravel()
    )
    if len(weights) != len(edges):
        raise ValueError(
            f"`weights` must have one entry per edge: got {len(weights)} "
            f"for {len(edges)} edges"
        )
    return weights, float64


def geodesic_predecessors(
    edges,
    n_nodes,
    weights=None,
    directed=False,
    sources=None,
    limit=None,
    threads=None,
    dtype=None,
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
    dtype :      float32 | float64, optional
                 Width the distances are accumulated and returned at, as
                 :func:`~navis_fastcore.geodesic_matrix_graph`. If ``None`` it
                 follows ``weights``.

    Returns
    -------
    distances :  (len(sources), n_nodes) array
                 In the resolved dtype. As ``geodesic_matrix_graph``: ``-1`` where
                 unreachable.
    predecessors : (len(sources), n_nodes) int32 array
                 For each node, the node before it on the shortest path back to that
                 row's source. ``-1`` for the source itself and for unreachable
                 nodes - so a single ``>= 0`` test both walks the path and
                 terminates it. Node ids, so int32 at either width.

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
    weights, float64 = _prep_weights(weights, edges, dtype)
    return _fastcore.geodesic_predecessors(
        edges,
        n_nodes,
        weights,
        bool(directed),
        _prep_indices(sources, n_nodes, "sources"),
        _prep_limit(limit),
        None if threads is None else int(threads),
        float64,
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
                 are allowed. A float64 array is accumulated at that width rather
                 than being narrowed, which can change which route wins. There is no
                 ``dtype`` argument because the paths are node ids.
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
        # No `dtype`: the output is node ids, so the width is invisible in it. The
        # weights' own width is still honoured, because it decides which route wins.
        _prep_weights(weights, edges)[0],
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
                 ``max_dist`` is a hop count. A float64 array is accumulated at that
                 width rather than being narrowed, which can change which nodes fall
                 inside a ball. There is no ``dtype`` argument because the labels
                 carry no width of their own.
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
        # As `geodesic_path`: the labels carry no width, but which nodes fall inside
        # a ball is evaluated at the weights' own.
        _prep_weights(weights, edges)[0],
        _prep_indices(seeds, n_nodes, "seeds"),
    )


def _prep_mesh_edit(faces, vertices, lock):
    """Validate the arguments every mesh-editing entry point shares.

    ``simplify_mesh``, ``simplify_mesh_lossless`` and ``smooth_mesh`` all take the
    same triple and want the same three things done to it, including the ``lock``
    mask both families spell the same way.
    """
    faces, vertices, n_vertices = _prep_mesh(faces, vertices, None)
    if not np.isfinite(vertices).all():
        # A non-finite coordinate would reach the collapse guards, and every
        # comparison against NaN is false, so it would be silently accepted.
        raise ValueError("`vertices` must be finite")
    return faces, vertices, _prep_mask(lock, n_vertices, "lock")


def _prep_target(ratio, n_faces):
    """Check exactly one face budget was named, and that it is in range.

    What a ratio *means* — the rounding, the floor of one face — lives in the core
    so it is defined once rather than once per binding. What is left here is the
    same job every other ``_prep_*`` does: turn a caller's mistake into a
    ``ValueError`` rather than letting it reach the core and panic.
    """
    if (ratio is None) == (n_faces is None):
        raise ValueError("Provide exactly one of `ratio` or `n_faces`.")
    if n_faces is not None:
        n_faces = int(n_faces)
        if n_faces < 0:
            raise ValueError(f"`n_faces` must be non-negative, got {n_faces}")
        return None, n_faces
    ratio = float(ratio)
    if not np.isfinite(ratio) or not 0 < ratio <= 1:
        raise ValueError(f"`ratio` must be in (0, 1], got {ratio}")
    return ratio, None


def simplify_mesh(
    faces,
    vertices,
    ratio=None,
    n_faces=None,
    aggressiveness=7.0,
    preserve_border=False,
    lock=None,
):
    """Simplify a triangle mesh, tracking where every vertex went.

    Iteratively contracts the edge whose collapse costs least, where the cost is the
    Garland-Heckbert quadric error: the summed squared distance from the merged
    vertex to the planes of every face that met at the two it replaces. This is the
    algorithm ``pyfqmr`` runs — a port of the same MIT-licensed original — with the
    one thing no implementation in this space returns, ``vertex_map``.

    Non-manifold input is fine. Meshes out of EM segmentation routinely have edges
    shared by three faces, bowtie vertices and zero-area triangles; nothing here
    checks for manifoldness, and each collapse guard skips what it cannot handle
    rather than failing.

    Parameters
    ----------
    faces :           (F, 3) array
                      Triangular faces given as rows of three vertex indices.
                      Must be convertible to ``uint32``.
    vertices :        (V, 3) array
                      Vertex positions. Must be finite.
    ratio :           float, optional
                      Fraction of the faces to keep, in ``(0, 1]``.
    n_faces :         int, optional
                      Absolute number of faces to keep. Give exactly one of
                      ``ratio`` or ``n_faces``.
    aggressiveness :  float
                      Exponent of the error-threshold sweep. Higher reaches the
                      target in fewer, coarser passes; 5-8 are sensible and 7 is
                      the default everywhere this algorithm appears.
    preserve_border : bool
                      Freeze every vertex the one-ring heuristic calls a boundary.
                      If ``False`` (the default, as in ``pyfqmr``) boundary vertices
                      still collapse among themselves, just never into the interior.
    lock :            (V, ) bool array, optional
                      Vertices that must survive at exactly their input position.
                      A locked vertex may absorb its neighbours but is never itself
                      absorbed or moved. This is how you pin synapse-bearing vertices.

    Returns
    -------
    vertices :   (V', 3) float64 array
                 Positions of the surviving vertices.
    faces :      (F', 3) uint32 array
                 Faces, indexing the returned ``vertices``.
    vertex_map : (V, ) int32 array
                 For each **input** vertex, the index of the **output** vertex it
                 ended up in; ``-1`` where it did not survive. Indexed by input
                 vertex, valued in output vertices — invert it with ``bincount``
                 (see Examples).

    Notes
    -----
    **When a vertex maps to ``-1``.** Being merged is *not* one of those cases —
    that is what the map is for, and a collapsed vertex points at whatever it
    merged into. Decimating a clean closed mesh to 5% of its faces yields no ``-1``
    at all. The rule is that ``vertex_map[i]`` is ``-1`` exactly when the vertex
    ``i`` ended up in is referenced by no surviving face, which happens in four
    situations:

    1. ``i`` is in no face to begin with, so it never takes part in a collapse.
    2. ``i`` appears only in zero-area faces. A face naming the same vertex twice
       carries no plane and no normal, so it is dropped on the way in, which
       reduces this to case 1.
    3. The whole piece ``i`` belonged to was decimated away. A collapse deletes
       exactly the faces holding *both* its endpoints, so a survivor normally keeps
       faces — but the budget is global, with nothing reserved per component, so a
       small disconnected fragment is consumed entirely once the target is tight
       enough. Simplify per component if that matters.
    4. The input is wholly degenerate, so the output mesh is empty.

    Mask with ``vertex_map >= 0`` before aggregating (see Examples).

    **Positions move.** ``vertices_out[vertex_map[i]]`` is **not**
    ``vertices_in[i]``: a collapse moves its survivor to the quadric-optimal point,
    so any vertex that took part in one has shifted. Use ``lock`` where the exact
    position matters.

    **What ``lock`` guarantees** is that a locked vertex is never merged into
    another and never moved. It is not an absolute guarantee against ``-1``: it
    cannot conjure up a vertex that no surviving face references, so cases 1, 2 and
    4 above still apply, and a locked vertex can in principle lose every face it sat
    on if each of them happened to hold both endpoints of some other collapse.
    Locking does set a floor on how far the mesh can shrink — every locked vertex
    survives — so a target below the locked count is not reachable.

    **Determinism.** The result depends on the order of ``faces``, since the sweep
    visits them in input order, but is otherwise deterministic: same input, same
    output, every run. There is no ``threads`` argument because each collapse
    invalidates its own neighbourhood, so the sweep cannot be parallelised without
    changing the answer. The GIL is released for the duration, so simplifying
    several meshes from a thread pool does scale.

    Examples
    --------
    >>> import navis_fastcore as fastcore
    >>> import numpy as np
    >>> # A unit square as two triangles, subdivided once.
    >>> faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.uint32)
    >>> vertices = np.array([[0., 0., 0.], [1., 0., 0.],
    ...                      [0., 1., 0.], [1., 1., 0.]])
    >>> v, f, vmap = fastcore.simplify_mesh(faces, vertices, n_faces=2)
    >>> len(f)
    2
    >>> vmap
    array([0, 1, 2, 3], dtype=int32)

    Push a per-vertex quantity — synapse counts, say — onto the simplified mesh:

    >>> syn = np.array([3, 0, 1, 2])
    >>> live = vmap >= 0
    >>> np.bincount(vmap[live], weights=syn[live], minlength=len(v))
    array([3., 0., 1., 2.])

    Pin the vertices that carry synapses so they keep their exact positions:

    >>> lock = np.zeros(len(vertices), dtype=bool)
    >>> lock[[0, 3]] = True
    >>> v, f, vmap = fastcore.simplify_mesh(faces, vertices, ratio=0.5, lock=lock)
    >>> np.array_equal(v[vmap[[0, 3]]], vertices[[0, 3]])
    True

    """
    faces, vertices, lock = _prep_mesh_edit(faces, vertices, lock)
    ratio, n_faces = _prep_target(ratio, n_faces)
    return _fastcore.simplify_mesh(
        faces,
        vertices,
        ratio,
        n_faces,
        float(aggressiveness),
        bool(preserve_border),
        lock,
    )


def simplify_mesh_lossless(
    faces,
    vertices,
    epsilon=1e-3,
    max_iterations=9999,
    preserve_border=False,
    lock=None,
):
    """Simplify a triangle mesh without changing its shape.

    Collapses only edges whose quadric error is below ``epsilon`` and repeats until
    a whole pass changes nothing. There is no face budget: this is for shedding
    over-tessellation — coplanar fans, duplicated vertices, degenerate faces —
    rather than for hitting a target. Use :func:`simplify_mesh` for that.

    Parameters
    ----------
    faces :           (F, 3) array
                      Triangular faces given as rows of three vertex indices.
    vertices :        (V, 3) array
                      Vertex positions. Must be finite.
    epsilon :         float
                      Quadric error below which an edge may collapse. This is an
                      **absolute** error with units of squared distance, so it
                      scales with your coordinates: 1e-3 means something quite
                      different in microns than in nanometres.
    max_iterations :  int
                      Cap on the number of passes.
    preserve_border : bool
                      As :func:`simplify_mesh`.
    lock :            (V, ) bool array, optional
                      As :func:`simplify_mesh`.

    Returns
    -------
    vertices :   (V', 3) float64 array
    faces :      (F', 3) uint32 array
    vertex_map : (V, ) int32 array
                 As :func:`simplify_mesh` — for each input vertex, the output vertex
                 it ended up in, or ``-1``.

    Notes
    -----
    "Lossless" is a claim about the *surface*, not the *outline*. A quadric measures
    distance to the planes of the incident faces, and the plane of a flat patch says
    nothing about where that patch ends — so with ``preserve_border=False`` a planar
    region will happily collapse its own boundary inwards at zero measured cost. Pass
    ``preserve_border=True`` on open meshes.

    The circumstances under which a vertex maps to ``-1`` rather than to the vertex
    it merged into, and what ``lock`` does and does not guarantee, are as
    :func:`simplify_mesh`. Case 3 there — a whole piece consumed — arrives by a
    different route here: there is no face budget, but ``epsilon`` is an *absolute*
    error, so a component small enough that all of its edges fall under it collapses
    away entirely. A lone triangle does this at any size; a tetrahedron 0.001 across
    does it at the default ``epsilon``, whether or not a larger mesh surrounds it.

    Examples
    --------
    A flat 4x4 grid of vertices. The four interior ones are exactly coplanar with
    their neighbours, so removing them costs nothing; the rim is held in place.

    >>> import navis_fastcore as fastcore
    >>> import numpy as np
    >>> vertices = np.array([[i, j, 0] for i in range(4) for j in range(4)],
    ...                     dtype=float)
    >>> faces = np.array(
    ...     [t for i in range(3) for j in range(3)
    ...        for t in ([i * 4 + j, (i + 1) * 4 + j, (i + 1) * 4 + j + 1],
    ...                  [i * 4 + j, (i + 1) * 4 + j + 1, i * 4 + j + 1])],
    ...     dtype=np.uint32,
    ... )
    >>> v, f, vmap = fastcore.simplify_mesh_lossless(faces, vertices,
    ...                                              preserve_border=True)
    >>> len(faces), len(f)
    (18, 12)
    >>> bool(np.abs(v[:, 2]).max() < 1e-9)  # still flat
    True

    The four interior vertices — 5, 6, 9 and 10 — merged into a single one:

    >>> vmap[[5, 6, 9, 10]]
    array([7, 7, 7, 7], dtype=int32)

    """
    faces, vertices, lock = _prep_mesh_edit(faces, vertices, lock)
    epsilon = float(epsilon)
    if not np.isfinite(epsilon) or epsilon < 0:
        raise ValueError(f"`epsilon` must be finite and non-negative, got {epsilon}")
    return _fastcore.simplify_mesh_lossless(
        faces,
        vertices,
        epsilon,
        int(max_iterations),
        bool(preserve_border),
        lock,
    )


def smooth_mesh(
    faces,
    vertices,
    method="taubin",
    iterations=10,
    lamb=None,
    mu=None,
    alpha=None,
    beta=None,
    weights="uniform",
    preserve_border=False,
    lock=None,
    volume_correction=False,
    threads=None,
):
    """Smooth a triangle mesh.

    Moves vertices and touches nothing else: the face array, the vertex count and the
    vertex order all come back unchanged, so anything you have indexed by vertex —
    synapses, radii, labels — is still attached to the vertex it was attached to.

    Three methods, chosen with ``method``:

    ``"taubin"`` (the default)
        Alternating shrink and inflate passes, tuned so the two cancel below a
        cut-off frequency. Removes noise without removing the shape, and is the
        default for that reason.
    ``"laplacian"``
        The plain diffusion step: simple, effective, and it **shrinks**. At
        ``lamb=0.5`` and five iterations — what ``navis.smooth_mesh`` ships — a
        neuron mesh loses 88% of its enclosed volume. Reach for it when the mesh is
        a means to an end rather than when its volume means something, or pair it
        with ``volume_correction``.
    ``"humphrey"``
        The HC filter of Vollmer et al., which fights shrinkage by pulling each
        vertex back towards where it started rather than towards a lower frequency.
        The gentler of the two on fine detail worth keeping.

    Parameters
    ----------
    faces :             (F, 3) array
                        Triangular faces given as rows of three vertex indices.
                        Must be convertible to ``uint32``.
    vertices :          (V, 3) array
                        Vertex positions. Must be finite.
    method :            "taubin" | "laplacian" | "humphrey"
                        Which filter to run. See above.
    iterations :        int
                        Passes to run. For ``"taubin"`` one pass is a full
                        ``lamb``-then-``mu`` pair, i.e. two sweeps over the mesh —
                        not one, as ``trimesh.smoothing.filter_taubin`` counts them.
                        Counting half-steps lets an odd ``iterations`` end on a
                        shrink that nothing undoes.
    lamb :              float, optional
                        Diffusion speed for ``"laplacian"`` and ``"taubin"``, in
                        ``[0, 1]``. Larger is more aggressive. Defaults to 0.5.
    mu :                float, optional
                        Inflating pass for ``"taubin"``. Must be negative and larger
                        in magnitude than ``lamb``. Defaults to -0.53.
    alpha :             float, optional
                        For ``"humphrey"``: how hard vertices are pulled back
                        towards their original positions, in ``[0, 1]``. Defaults
                        to 0.1.
    beta :              float, optional
                        For ``"humphrey"``: how much of that pull-back lands on the
                        vertex itself rather than on its one-ring, in ``[0, 1]``.
                        Defaults to 0.5.
    weights :           "uniform" | "inverse_distance" | "cotangent"
                        How each vertex's one-ring is weighted. ``"uniform"`` counts
                        every neighbour equally and also regularises the *sampling*,
                        which means it slides vertices along the surface where the
                        tessellation is uneven. ``"cotangent"`` is the discrete
                        Laplace-Beltrami operator: it depends on the shape rather
                        than on the triangulation, so it moves vertices along the
                        normal and leaves them alone within the surface. That is
                        usually what you want on meshes out of EM segmentation,
                        whose triangles vary wildly in size and aspect.
    preserve_border :   bool
                        Pin every vertex on a mesh boundary — an endpoint of an edge
                        used by exactly one face. Without this an open mesh's rim
                        rolls inwards under any of these filters, because a boundary
                        vertex's one-ring lies entirely to one side of it.
    lock :              (V, ) bool array, optional
                        Vertices that must not move; they come back at bitwise the
                        same coordinates. Unioned with ``preserve_border``, not an
                        alternative to it. A locked vertex still pulls on its
                        neighbours, which is what makes it a boundary condition
                        rather than a hole. Same name and same meaning as
                        :func:`simplify_mesh`'s ``lock``.
    volume_correction : bool
                        Rescale the result about its centroid so the enclosed volume
                        matches the input's. Warns and leaves the mesh unscaled if
                        the mesh has no usable volume — see Notes.
    threads :           int, optional
                        Number of threads to use. ``None`` uses all available cores.

    Returns
    -------
    vertices :          (V, 3) float64 array
                        New positions, in the same order as the input.

    Notes
    -----
    **The volume correction scales about the centroid, not the origin.** This is the
    one place where the result deliberately differs from
    ``trimesh.smoothing.filter_laplacian``, which is what ``navis.smooth_mesh``
    calls today. Upstream rescales by ``(vol_before / vol_after) ** (1/3)`` about the
    origin, which is not a shape operation: on the 722817260 test neuron at navis'
    own defaults it displaces the mesh by 41 um, and the mesh is 19-26 um across. It
    is also not translation invariant — the same mesh smoothed at two different
    offsets comes out two different shapes, and far enough from the origin the volume
    ratio goes negative and the cube root returns NaN. Scaling about the mesh's own
    centroid is the same size change with none of that.

    The correction also runs **once, at the end**, which is not an approximation of
    running it every iteration but exactly equal to it: every filter here is an
    affine combination of a vertex and a normalised average of its neighbours, and
    those commute with a uniform scaling. Upstream pays a full pass over the faces
    per iteration — 40% of its runtime — for a result it could have had at the end.

    **When the volume is undefined.** On a closed mesh the correction is exactly what
    it says. A mesh that is *not* closed still usually gets one, and deliberately:
    both measurements cone every face back to the same anchor, so their ratio stays a
    consistent measure of how much the surface shrank even where neither number is an
    enclosed volume on its own. That matters because meshes worth smoothing are almost
    never watertight — the 722817260 test neuron is not — and refusing on that basis
    would refuse on nearly every mesh this exists for.

    What is left is the genuinely undecidable case: the ratio of the two signed
    volumes is zero, infinite, NaN or negative, so it has no cube root worth taking. A
    flat sheet is the clean example, with both volumes exactly zero. There the
    vertices come back smoothed but unscaled and a ``RuntimeWarning`` says so.
    Consistently inverted winding is *not* in that set — both volumes come out
    negative, the ratio is positive, and the correction is as valid as ever.

    **Non-manifold input is fine**, as for :func:`simplify_mesh`. An edge shared by
    three faces, a face naming the same vertex twice, a duplicated face and a vertex
    no face mentions are all merely data; nothing here reads more topology than
    "which vertices are adjacent to which". A vertex in no face never moves.

    Examples
    --------
    >>> import navis_fastcore as fastcore
    >>> import numpy as np
    >>> # A 5x5 grid with its middle vertex lifted out of the plane.
    >>> faces = np.array([[i * 5 + j, (i + 1) * 5 + j, (i + 1) * 5 + j + 1]
    ...                   for i in range(4) for j in range(4)]
    ...                  + [[i * 5 + j, (i + 1) * 5 + j + 1, i * 5 + j + 1]
    ...                     for i in range(4) for j in range(4)], dtype=np.uint32)
    >>> vertices = np.array([[i, j, 0.] for i in range(5) for j in range(5)])
    >>> vertices[12, 2] = 1.0
    >>> v = fastcore.smooth_mesh(faces, vertices, method="laplacian", lamb=1.0,
    ...                          iterations=1)
    >>> float(v[12, 2])  # back in the plane its six neighbours span
    0.0

    Pin the rim so an open mesh does not roll inwards:

    >>> v = fastcore.smooth_mesh(faces, vertices, preserve_border=True)
    >>> bool(np.array_equal(v[0], vertices[0]))
    True

    """
    faces, vertices, lock = _prep_mesh_edit(faces, vertices, lock)

    iterations = int(iterations)
    if iterations < 0:
        raise ValueError(f"`iterations` must be non-negative, got {iterations}")

    # `method`, `weights`, which parameters belong to which method, and the ranges are
    # all checked one layer down, against the tables in the Rust core that own them —
    # so there is one copy of each rule rather than one per binding surface. They come
    # back as ``ValueError`` either way.
    verts, volumes = _fastcore.smooth_mesh(
        faces,
        vertices,
        method,
        iterations,
        None if lamb is None else float(lamb),
        None if mu is None else float(mu),
        None if alpha is None else float(alpha),
        None if beta is None else float(beta),
        weights,
        bool(preserve_border),
        lock,
        bool(volume_correction),
        threads=threads,
    )
    # `volumes` is only ever set when a correction was asked for and could not be
    # made, so this is the whole of the "undefined" branch. Silence here would be
    # the failure mode worth avoiding: the caller asked for a volume-preserving
    # smooth and got a plain one.
    if volumes is not None:
        before, after = volumes
        warnings.warn(
            f"`volume_correction` was requested but the mesh has no usable enclosed "
            f"volume (signed volume {before:.6g} before smoothing, {after:.6g} "
            f"after), so the vertices were returned unscaled. This is expected for a "
            f"mesh that is not closed.",
            RuntimeWarning,
            stacklevel=2,
        )
    return verts


def _prep_mask(mask, n, what):
    """Coerce an optional boolean mask to a contiguous bool array of length ``n``.

    Deliberately a no-op for the array a caller in a tiling loop actually holds (a
    C-contiguous ``np.bool_`` array): ``asarray`` and ``ascontiguousarray`` both pass it
    straight through, so the mask is borrowed rather than copied on every iteration.
    """
    if mask is None:
        return None
    mask = np.ascontiguousarray(np.asarray(mask, dtype=bool).ravel())
    if len(mask) != n:
        raise ValueError(f"`{what}` must have {n} entries, got {len(mask)}")
    return mask


class GeodesicGraph:
    """A graph prepared once for many geodesic queries.

    The module-level geodesic functions each build an adjacency index from your edge
    list, answer one question and throw it away. That is the right trade for a single
    all-pairs sweep, and the wrong one for algorithms that ask *many small* questions
    of the same graph - the index build is O(E) over the whole graph, so it dwarfs a
    query that only ever explores a small ball.

    This class is for that second pattern. It builds the index once, keeps the search
    scratch space alive between calls, and lets each query cost only the ball it
    explores.

    Parameters
    ----------
    edges :      (E, 2) array
                 Edges given as rows of two node indices. Treated as undirected.
    n_nodes :    int
                 Total number of nodes.
    weights :    (E, ) array, optional
                 Length of each edge. If ``None`` all edges weigh 1, i.e. distances
                 are hop counts (and the searches run as BFS rather than Dijkstra).
    directed :   bool
                 If ``True`` an edge ``(u, v)`` may only be traversed from ``u`` to
                 ``v``, and every method takes its "outward from here" reading - see
                 Notes.
    item_nodes : (M, ) array, optional
                 The node each of ``M`` *items* is attached to. :meth:`grow`,
                 :meth:`farthest_seed` and :meth:`item_components` then count and return
                 items rather than nodes - see Notes. If ``None`` (default) each node is
                 its own single item, so items and nodes coincide.

    Attributes
    ----------
    n_nodes :    int
    n_items :    int
                 Equals ``n_nodes`` unless ``item_nodes`` was given.
    item_nodes : (n_items, ) uint32 array
                 The node each item sits on.

    Notes
    -----
    **What's here.** Most methods are the module-level functions with the index build
    taken out, and answer exactly what their counterpart does: :meth:`distances`
    (:func:`~navis_fastcore.geodesic_matrix_graph`), :meth:`nearest`, :meth:`farthest`,
    :meth:`predecessors`, :meth:`path`, :meth:`clusters` and :meth:`components`. The two
    with no counterpart - :meth:`grow` and :meth:`farthest_seed` - are the ones that only
    make sense against a graph you keep, since they are called in a loop. :meth:`subset`
    carves out an induced subgraph without going back to your edge list.

    **When the reuse pays.** Hoisting the index build out of the call is worth real time
    exactly when each query is *small* relative to the graph - many short paths, a
    :meth:`nearest` with a tight ``limit``, :meth:`grow`. On a 40k-vertex mesh, 500
    short-path queries run ~100x faster through this class than through
    :func:`~navis_fastcore.geodesic_path`. It buys nothing measurable when a single query
    already sweeps the graph: one 50-source distance matrix on that mesh costs 90 ms
    either way, against a 1 ms build. The other reason to reach for the class is simply
    that you have a graph, and would rather not re-pass ``edges``/``weights``/``directed``
    to every call.

    **Width.** float32 only, and the one place the module's "your dtype in, your dtype
    out" rule does not apply: float64 ``weights`` are accepted but narrowed, and every
    distance this class returns is float32. That is deliberate rather than an omission.
    The class exists for "large graph, many small queries", which is exactly the case
    where float32 is the right width and where doubling the several node-sized arrays it
    holds resident for a whole run would be felt. If you need float64, use the module-level
    functions, which rebuild the index per call and take a ``dtype``.

    **Items.** Optionally each node carries zero or more *items* - points of a cloud
    attached to the graph, one entry of a resampled surface say. By default the
    distinction vanishes entirely.

    The rule for which index space a method speaks is short: :meth:`grow`,
    :meth:`farthest_seed` and :meth:`item_components` count and return **items**;
    everything else takes and returns graph **nodes**, exactly as the free function it
    mirrors does. So growth follows the graph but is measured in cloud points - which is
    what keeps a patch of a cloud far sparser than its mesh connected, since the empty
    nodes in between conduct without contributing - while a distance matrix stays a
    matrix over the graph.

    **Direction.** With ``directed=True``, :meth:`grow` gathers the *out*-reachable ball,
    :meth:`farthest_seed` measures distance *from* the done set, and :meth:`clusters`
    grows out-balls (so it differs from :func:`~navis_fastcore.geodesic_clusters`, which
    is always undirected). :meth:`components` still reports *weakly* connected
    components - a search has to start somewhere.

    **Threading.** Queries share mutable scratch space, so concurrent calls on one
    instance serialise. Build one instance per thread if you need real parallelism. The
    ``threads`` argument on the matrix-style methods is unaffected: those parallelise
    internally over sources.

    Examples
    --------
    >>> import navis_fastcore as fastcore
    >>> import numpy as np
    >>> edges = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]], dtype=np.uint32)
    >>> g = fastcore.GeodesicGraph(edges, 6)
    >>> g
    GeodesicGraph(n_nodes=6, n_items=6)

    Grow a ball of three nodes around node 2 - nearest first:

    >>> g.grow(2, 3)
    array([2, 1, 3], dtype=uint32)

    Tile the whole graph into disjoint connected fragments by feeding what has already
    been claimed back in as ``forbidden``:

    >>> claimed = np.zeros(6, dtype=bool)
    >>> frags = []
    >>> while not claimed.all():
    ...     frag = g.grow(int(np.argmax(~claimed)), 2, forbidden=claimed)
    ...     claimed[frag] = True
    ...     frags.append(frag.tolist())
    >>> frags
    [[0, 1], [2, 3], [4, 5]]

    """

    def __init__(self, edges, n_nodes, weights=None, directed=False, item_nodes=None):
        edges, n_nodes = _prep_edges(edges, n_nodes)
        self._adopt(
            _fastcore.GeodesicGraph(
                edges,
                n_nodes,
                # `np.float32` spelled out: this class is float32 only (see the
                # class docstring), and a defaulted argument would not say so.
                _prep_weights(weights, edges, np.float32)[0],
                bool(directed),
                _prep_indices(item_nodes, n_nodes, "item_nodes"),
            )
        )

    @classmethod
    def _wrap(cls, inner, parent_nodes, parent_items):
        """Wrap an already-built Rust graph - the second construction path.

        :meth:`subset` gets its graph back from Rust fully formed; rebuilding it through
        ``__init__`` from an edge list is exactly the work it exists to avoid. Funnelling
        both paths through :meth:`_adopt` keeps that shortcut from quietly producing an
        object that is missing whatever ``__init__`` sets.
        """
        self = cls.__new__(cls)
        self._adopt(inner, parent_nodes, parent_items)
        return self

    def _adopt(self, inner, parent_nodes=None, parent_items=None):
        """Take ownership of a Rust graph and cache what never changes about it.

        ``n_nodes``/``n_items`` are fixed for the object's lifetime, and reading them
        crosses into Rust and takes the graph's lock - which `grow` and `farthest_seed`
        would otherwise pay on every call of a tight loop, to re-learn a constant.
        """
        self._graph = inner
        self.n_nodes = inner.n_nodes
        self.n_items = inner.n_items
        #: For a graph returned by :meth:`subset`, the parent's node/item indices.
        self.parent_nodes = parent_nodes
        self.parent_items = parent_items

    @property
    def item_nodes(self):
        """The node each item sits on, as a ``(n_items, )`` uint32 array."""
        return self._graph.item_nodes

    def __repr__(self):
        return repr(self._graph)

    # -- the module-level geodesic functions, without the per-call index build --

    def distances(self, sources=None, targets=None, limit=None, threads=None):
        """Pairwise geodesic distances between `sources` and `targets` (nodes).

        The same query as :func:`~navis_fastcore.geodesic_matrix_graph`, minus the
        adjacency build. Use this when you slice the same graph repeatedly - a batch of
        sources at a time, say - where the free function would rebuild its index on each
        call.

        Parameters
        ----------
        sources :  iterable, optional
                   Node indices. ``None`` (default) means all nodes.
        targets :  iterable, optional
                   Node indices. ``None`` (default) means all nodes. Beware the output
                   size: a full ``V x V`` matrix is ~107 GB at V=164k.
        limit :    float, optional
                   Ignore anything farther than this; such pairs come back as ``-1``.
        threads :  int, optional
                   Size of the thread pool. Defaults to all available cores.

        Returns
        -------
        matrix :   (len(sources), len(targets)) float32 array
                   ``-1`` where unreachable, or beyond `limit`.

        Examples
        --------
        >>> import navis_fastcore as fastcore
        >>> import numpy as np
        >>> edges = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.uint32)
        >>> g = fastcore.GeodesicGraph(edges, 4)
        >>> g.distances(sources=[0], targets=[3])
        array([[3.]], dtype=float32)

        """
        return self._graph.distances(
            _prep_indices(sources, self.n_nodes, "sources"),
            _prep_indices(targets, self.n_nodes, "targets"),
            _prep_limit(limit),
            None if threads is None else int(threads),
        )

    def nearest(self, sources=None, targets=None, limit=None, threads=None):
        """For each source node, the distance to its nearest target and which one.

        As :func:`~navis_fastcore.geodesic_nearest_mesh`, minus the adjacency build.
        ``O(sources)`` output instead of ``O(sources x targets)``, and faster than the
        matrix too - each search stops at the first target it settles.

        A source that is itself a target is matched to its nearest *distinct* target.
        Sources with no reachable distinct target get ``-1`` / ``-1``.

        Parameters
        ----------
        sources, targets, limit, threads
                   As :meth:`distances`.

        Returns
        -------
        distances : (len(sources), ) float32 array
        indices :   (len(sources), ) int32 array
                    Index *into the graph*, not into `targets`.

        """
        return self._graph.nearest(
            _prep_indices(sources, self.n_nodes, "sources"),
            _prep_indices(targets, self.n_nodes, "targets"),
            _prep_limit(limit),
            None if threads is None else int(threads),
        )

    def farthest(self, sources=None, targets=None, limit=None, threads=None):
        """For each source node, the distance to its farthest target and which one.

        The mirror of :meth:`nearest`; see
        :func:`~navis_fastcore.geodesic_farthest_mesh`. This one cannot stop early - it
        has to settle every target - but the farthest is then free, since the kernels
        settle in increasing distance order.

        Returns
        -------
        distances : (len(sources), ) float32 array
        indices :   (len(sources), ) int32 array

        """
        return self._graph.farthest(
            _prep_indices(sources, self.n_nodes, "sources"),
            _prep_indices(targets, self.n_nodes, "targets"),
            _prep_limit(limit),
            None if threads is None else int(threads),
        )

    def predecessors(self, sources=None, limit=None, threads=None):
        """Shortest-path trees: distances *and* the route to every node.

        As :func:`~navis_fastcore.geodesic_predecessors`, minus the adjacency build.
        Use :meth:`path` when you want the node sequences rather than the raw chains.

        Parameters
        ----------
        sources :  iterable, optional
                   Node indices, one tree each. ``None`` (default) means all nodes.
        limit, threads
                   As :meth:`distances`.

        Returns
        -------
        distances :    (len(sources), n_nodes) float32 array, ``-1`` where unreachable.
        predecessors : (len(sources), n_nodes) int32 array
                       The node before each node on its shortest path back to that row's
                       source; ``-1`` for the source itself and for unreachable nodes.

        """
        return self._graph.predecessors(
            _prep_indices(sources, self.n_nodes, "sources"),
            _prep_limit(limit),
            None if threads is None else int(threads),
        )

    def path(self, source, targets):
        """Node sequences of the shortest paths from `source` to each of `targets`.

        As :func:`~navis_fastcore.geodesic_path`, minus the adjacency build. One search,
        stopped as soon as the last target settles.

        Parameters
        ----------
        source :  int
                  Node index.
        targets : iterable
                  Node indices.

        Returns
        -------
        paths :   list of uint32 arrays
                  One per target, source-first and target-last. An unreachable target
                  gives an empty array; a target equal to `source` gives ``[source]``.

        Examples
        --------
        >>> import navis_fastcore as fastcore
        >>> import numpy as np
        >>> edges = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.uint32)
        >>> g = fastcore.GeodesicGraph(edges, 4)
        >>> g.path(0, [3])[0]
        array([0, 1, 2, 3], dtype=uint32)

        """
        source = int(source)
        if not 0 <= source < self.n_nodes:
            raise ValueError(f"`source` is node {source} but n_nodes = {self.n_nodes}")
        targets = _prep_indices(targets, self.n_nodes, "targets")
        if targets is None:
            raise ValueError("`targets` must be given")
        return self._graph.path(source, targets)

    def ball(self, sources, max_dist=None):
        """Every node within `max_dist` of any of `sources`, and its nearest source.

        One multi-source search, so this costs the ball it returns rather than one sweep
        per source - and it returns the ball itself, not a ``(n_nodes, )`` array with the
        ball buried in it. Both halves matter for the pattern this exists for: sweeping a
        graph a neighbourhood at a time, thousands of small radii against one big graph.

        The nearest equivalent elsewhere,
        ``scipy.sparse.csgraph.dijkstra(..., min_only=True, limit=...)``, allocates and
        fills three node-sized arrays per call whatever the radius, which for a small one
        costs more than the search does.

        Parameters
        ----------
        sources :  iterable
                   Node indices. May repeat.
        max_dist : float, optional
                   Radius, inclusive, in the graph's own metric (hop counts when the graph
                   is unweighted). ``None`` (default) for no bound, which makes this "the
                   nearest source of every reachable node".

        Returns
        -------
        nodes :     (N, ) uint32 array
                    The nodes within reach, in increasing-distance order. Every source is
                    in here at distance 0. Nodes farther than `max_dist`, and nodes no
                    source reaches, are absent rather than flagged.
        distances : (N, ) float32 array
                    Distance from each node to its nearest source.
        sources :   (N, ) uint32 array
                    That source, as a node index. A source's own entry is itself. Ties are
                    broken deterministically but arbitrarily.

        Examples
        --------
        >>> import navis_fastcore as fastcore
        >>> import numpy as np

        A path of 7 nodes, seeded from both ends - one hop out from each:

        >>> edges = np.array([[i, i + 1] for i in range(6)], dtype=np.uint32)
        >>> g = fastcore.GeodesicGraph(edges, 7)
        >>> nodes, dist, src = g.ball([0, 6], 1)
        >>> order = np.argsort(nodes)   # settle order interleaves the two frontiers
        >>> nodes[order]
        array([0, 1, 5, 6], dtype=uint32)
        >>> dist[order]
        array([0., 1., 1., 0.], dtype=float32)
        >>> src[order]
        array([0, 0, 6, 6], dtype=uint32)

        """
        sources = _prep_indices(sources, self.n_nodes, "sources")
        if sources is None:
            raise ValueError("`sources` must be given")
        max_dist = np.inf if max_dist is None else float(max_dist)
        if not max_dist >= 0:  # also catches NaN
            raise ValueError(f"`max_dist` must be non-negative, got {max_dist}")
        return self._graph.ball(sources, max_dist)

    def set_weights(self, edges, weights):
        """Re-weight edges in place, leaving the graph otherwise untouched.

        For algorithms that *change* the graph as they run - TEASAR zeroing each path it
        extracts so later ones may re-traverse it for free is the case this was added for.
        Rebuilding the graph after each change costs O(E) against an edit of a few hundred
        edges, and gives up the reason for holding a prepared graph in the first place;
        this costs O(edits) and a binary search apiece.

        Parameters
        ----------
        edges :   (K, 2) array
                  Edges to re-weight, as node pairs. Each must be an edge the graph
                  actually has, or you get a `ValueError` naming it - re-weighting a
                  non-existent edge is far more likely to be a bug than an intent to add
                  one, and this cannot add one anyway. Order is irrelevant and repeats are
                  allowed, last write winning.
        weights : (K, ) array or scalar
                  Their new weights, finite and non-negative. A single value applies to all
                  of them, which is the shape a caller zeroing a whole path writes.

        Notes
        -----
        Only available on a graph built *with* weights: there is no weight array on an
        unweighted one to write into, and materialising one would quietly turn every later
        search from a BFS into a Dijkstra.

        Distances change, so the incremental field behind :meth:`farthest_seed` is
        discarded here - a minimum folded under the old weights cannot be corrected under
        the new ones. Interleaving `farthest_seed` with re-weighting therefore pays a cold
        start after each edit; not interleaving them costs nothing. Component labels
        survive, since which edges exist has not changed.

        Examples
        --------
        >>> import navis_fastcore as fastcore
        >>> import numpy as np
        >>> edges = np.array([[0, 1], [1, 2], [0, 2]], dtype=np.uint32)
        >>> g = fastcore.GeodesicGraph(edges, 3, weights=[1.0, 1.0, 5.0])
        >>> g.distances(sources=[0], targets=[2])
        array([[2.]], dtype=float32)
        >>> g.set_weights([[0, 2]], [0.5])
        >>> g.distances(sources=[0], targets=[2])
        array([[0.5]], dtype=float32)

        """
        edges, _ = _prep_edges(edges, self.n_nodes)
        weights = np.asarray(weights, dtype=np.float32).ravel()
        if len(weights) == 1:
            # One value for all of them - the shape a caller zeroing a path writes. `repeat`
            # of a length-1 array is a no-op, so the K == 1 case needs no special casing.
            weights = np.repeat(weights, len(edges))
        return self._graph.set_weights(
            edges, _prep_weights(weights, edges, np.float32)[0]
        )

    def clusters(self, max_dist, seeds=None):
        """Greedily partition *nodes* into connected clusters of bounded radius.

        As :func:`~navis_fastcore.geodesic_clusters`, minus the adjacency build. This is
        the radius-bounded sibling of :meth:`grow`'s count-bounded ball: use this when
        the cluster's physical extent is what must be fixed, :meth:`grow` when its size
        is.

        On a graph built with ``directed=True`` this grows *out*-balls, so it differs
        from the always-undirected free function.

        Parameters
        ----------
        max_dist : float
                   Maximum distance from a cluster's seed.
        seeds :    iterable, optional
                   Preferred seed nodes, in order of preference. Any node still
                   unassigned afterwards seeds a cluster of its own.

        Returns
        -------
        labels :     (n_nodes, ) int32 array of contiguous cluster ids.
        n_clusters : int

        """
        return self._graph.clusters(
            float(max_dist), _prep_indices(seeds, self.n_nodes, "seeds")
        )

    def components(self):
        """Component label of each *node*.

        Labels are node indices - the smallest node index in the component, the same
        convention :func:`~navis_fastcore.connected_components_graph` uses. See
        :meth:`item_components` for the per-item view.

        Returns
        -------
        labels : (n_nodes, ) uint32 array

        """
        return self._graph.components()

    def subset(self, nodes):
        """The subgraph induced on `nodes`, as a graph in its own right.

        New node ``i`` is old node ``nodes[i]``. Edges with an endpoint outside the
        subset are dropped, and items go wherever their node did - an item whose node was
        dropped goes with it. The result carries ``parent_nodes`` and ``parent_items``
        so anything computed on it can be mapped back.

        The subgraph is carved out of the adjacency already built rather than re-derived,
        so this never returns to your original edge list - which is the point, since
        masking and renumbering an edge list in numpy is both slower and easy to get
        wrong. Restricting to one connected component is the motivating case.

        Distances within a subset are *not* generally the parent's: a shortest path that
        left the subset is gone. Taking a whole connected component is the case where
        they do agree.

        Parameters
        ----------
        nodes : iterable or (n_nodes, ) bool array
                Nodes to keep, as indices or as a mask. Indices may be in any order but
                must not repeat; the order becomes the subgraph's node order.

        Returns
        -------
        subgraph : GeodesicGraph

        Examples
        --------
        Pull out the largest connected component:

        >>> import navis_fastcore as fastcore
        >>> import numpy as np
        >>> edges = np.array([[0, 1], [2, 3], [3, 4]], dtype=np.uint32)
        >>> g = fastcore.GeodesicGraph(edges, 5)
        >>> labels = g.components()
        >>> sub = g.subset(labels == np.bincount(labels).argmax())
        >>> sub.n_nodes
        3
        >>> sub.parent_nodes            # which original nodes these are
        array([2, 3, 4], dtype=uint32)
        >>> sub.distances(sources=[0], targets=[2])
        array([[2.]], dtype=float32)

        """
        nodes = np.asarray(nodes)
        if nodes.dtype == bool:
            if nodes.shape != (self.n_nodes,):
                raise ValueError(
                    f"a boolean `nodes` mask must have {self.n_nodes} entries, "
                    f"got {nodes.shape}"
                )
            nodes = np.flatnonzero(nodes)
        nodes = _prep_indices(nodes, self.n_nodes, "nodes", unique=True)
        inner, kept_items = self._graph.subset(nodes)
        return type(self)._wrap(inner, nodes, kept_items)

    def grow(self, seed, size, forbidden=None, return_distances=False):
        """Grow a connected region of up to ``size`` items outwards from ``seed``.

        Settles nodes in order of increasing geodesic distance from the seed's node,
        collecting the items on each until ``size`` are gathered. The region is
        therefore the geodesic *ball* around the seed that happens to hold ``size``
        items - not whatever a depth-first walk stumbled into - and it is always
        connected, since every node reached bar the seed's own is reached through one
        settled before it.

        This is the count-bounded sibling of
        :func:`~navis_fastcore.geodesic_clusters`, which bounds by radius instead. Use
        this one when the fragment *size* is what must be fixed, e.g. tiling a neuron
        into equal-length inputs for a neural network.

        Parameters
        ----------
        seed :      int
                    Item to grow from.
        size :      int
                    How many items to gather. You get fewer only when the reachable
                    region runs out of eligible items.
        forbidden : (n_items, ) bool array, optional
                    Items an earlier fragment already claimed. Claimed items are never
                    collected, and a node whose items are *all* claimed becomes a wall
                    that growth will not cross - which is what makes repeated calls
                    carve the graph into disjoint *connected* fragments rather than
                    letting a later fragment tunnel through an earlier one. Nodes
                    carrying no items at all always conduct. If ``None``, nothing is
                    claimed, so successive calls are independent and may overlap.
        return_distances : bool
                    Also return each item's distance to the seed. It costs nothing - the
                    search settles items in distance order, so it already holds the
                    number - and it is what you need to make a patch *non-uniform*, e.g.
                    to thin it radially into a dense core with a sparse, far-reaching
                    halo. Items sharing a node share a distance exactly, since an item's
                    position is its node's.

        Returns
        -------
        region :    (<= size, ) uint32 array
                    Item indices, seed-first and in increasing-distance order.
        distances : (<= size, ) float32 array
                    Only if ``return_distances=True``. Each item's distance to the seed,
                    in the graph's own metric (hop counts when ``weights=None``), and
                    non-decreasing by construction.

        Examples
        --------
        A path of 7 nodes carrying a sparse cloud - two points at each end, nothing in
        between. The empty nodes conduct, so the patch spans them:

        >>> import navis_fastcore as fastcore
        >>> import numpy as np
        >>> edges = np.array([[i, i + 1] for i in range(6)], dtype=np.uint32)
        >>> g = fastcore.GeodesicGraph(edges, 7, item_nodes=[0, 0, 6, 6])
        >>> g.n_items
        4
        >>> g.grow(0, 4)
        array([0, 1, 2, 3], dtype=uint32)

        The distances come back alongside on request - here both points at each end
        share their node's distance:

        >>> g.grow(0, 4, return_distances=True)[1]
        array([0., 0., 6., 6.], dtype=float32)

        """
        return self._graph.grow(
            int(seed),
            int(size),
            _prep_mask(forbidden, self.n_items, "forbidden"),
            bool(return_distances),
        )

    def farthest_seed(self, done):
        """The undone item geodesically farthest from everything already done.

        Calling this repeatedly with a growing ``done`` set is farthest-point sampling:
        it spreads seeds evenly over the graph instead of letting them clump, which is
        what you want when placing patches, landmarks or cluster centres. Pair it with
        :meth:`grow` - seed, grow, mark what you covered, seed again.

        Only items *reachable* from something in ``done`` are candidates. Unreachable
        ones are infinitely far and would otherwise win every time, so a mesh with a few
        hundred disconnected specks would seed every speck before returning to the main
        body. Once the reachable frontier is exhausted - or when ``done`` is empty - this
        jumps to a fresh component, largest first. Ties go to the lower item index.

        Parameters
        ----------
        done :  (n_items, ) bool array
                Items already seeded or covered.

        Returns
        -------
        seed :  int or None
                ``None`` only when every item is already done.

        Notes
        -----
        The distance field is maintained incrementally, and each update is *pruned*
        against the running field, so a call costs the region the new sources actually
        claim rather than a sweep of the whole graph. The usual implementation of this
        (a fresh multi-source Dijkstra per seed, e.g. via
        ``scipy.sparse.csgraph.dijkstra(..., min_only=True)``) cannot prune that way and
        so goes quadratic in the number of seeds: placing 2560 seeds on a 160k-vertex
        mesh takes ~93 s that way against ~0.35 s here.

        ``done`` is expected to only ever *grow* between calls. It may shrink - the field
        is rebuilt from scratch when that is detected, so the answer stays correct - but
        that call loses the incremental saving.

        Examples
        --------
        >>> import navis_fastcore as fastcore
        >>> import numpy as np
        >>> edges = np.array([[i, i + 1] for i in range(8)], dtype=np.uint32)
        >>> g = fastcore.GeodesicGraph(edges, 9)
        >>> done = np.zeros(9, dtype=bool)
        >>> done[0] = True
        >>> picks = []
        >>> for _ in range(4):
        ...     s = g.farthest_seed(done)
        ...     picks.append(s)
        ...     done[s] = True
        >>> picks  # the far end, then repeated bisection
        [8, 4, 2, 6]

        """
        return self._graph.farthest_seed(_prep_mask(done, self.n_items, "done"))

    def item_components(self):
        """Component label of each item.

        Labels are node indices - specifically the smallest node index in the component,
        the same convention
        :func:`~navis_fastcore.connected_components_graph` uses - not a contiguous range.

        :meth:`farthest_seed` deliberately does not offer a "random seed from the largest
        component"; that would mean owning a random number generator, and a caller who
        cares about reproducibility wants it to be theirs. This is the piece to build it
        from:

        >>> import navis_fastcore as fastcore
        >>> import numpy as np
        >>> edges = np.array([[0, 1], [2, 3], [3, 4]], dtype=np.uint32)
        >>> g = fastcore.GeodesicGraph(edges, 5)
        >>> labels = g.item_components()
        >>> labels
        array([0, 0, 2, 2, 2], dtype=uint32)
        >>> pool = np.flatnonzero(labels == np.bincount(labels).argmax())
        >>> int(np.random.default_rng(0).choice(pool))  # a seed off the largest component
        4

        Returns
        -------
        labels : (n_items, ) uint32 array

        """
        return self._graph.item_components()
