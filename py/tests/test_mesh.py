"""Tests for the mesh module.

The oracle throughout is `scipy.sparse.csgraph.dijkstra`, which is what navis currently
uses for meshes (via a trimesh -> igraph -> scipy sparse detour).
"""

import numpy as np
import pytest

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

import navis_fastcore as fastcore


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


def grid_mesh(n=12, spacing=1.0):
    """An `n x n` grid triangulated along the (0,0)->(1,1) diagonal of each cell.

    Has a closed-form metric (see `test_matches_grid_closed_form`), so it doubles as an
    oracle that does not depend on scipy.
    """
    idx = lambda i, j: i * n + j  # noqa: E731

    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            faces.append([idx(i, j), idx(i + 1, j), idx(i + 1, j + 1)])
            faces.append([idx(i, j), idx(i + 1, j + 1), idx(i, j + 1)])

    verts = np.array(
        [[i * spacing, j * spacing, 0.0] for i in range(n) for j in range(n)],
        dtype=np.float64,
    )
    return np.array(faces, dtype=np.uint32), verts


def scipy_oracle(faces, vertices, n_vertices, sources=None, targets=None,
                 weighted=True, limit=np.inf):
    """Reference implementation, straight through scipy."""
    e = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]).astype(np.int64)
    e.sort(axis=1)
    # REQUIRED. coo/csr *sums* duplicate entries, and every interior edge appears in two
    # faces -- so without deduping first, every interior edge would silently get double
    # weight and this "oracle" would be wrong.
    e = np.unique(e, axis=0)
    e = e[e[:, 0] != e[:, 1]]  # drop self-loops from degenerate faces

    if weighted:
        w = np.linalg.norm(vertices[e[:, 0]] - vertices[e[:, 1]], axis=1)
    else:
        w = np.ones(len(e))

    g = csr_matrix(
        (np.concatenate([w, w]),
         (np.concatenate([e[:, 0], e[:, 1]]), np.concatenate([e[:, 1], e[:, 0]]))),
        shape=(n_vertices, n_vertices),
    )
    d = dijkstra(g, directed=False, indices=sources, limit=limit)
    if sources is None:
        d = d.reshape(n_vertices, n_vertices)
    if targets is not None:
        d = d[:, targets]
    return d


def as_inf(d):
    """Our -1 sentinel -> scipy's inf, so the two are comparable."""
    d = np.asarray(d, dtype=np.float64).copy()
    d[d < 0] = np.inf
    return d


# -----------------------------------------------------------------------------
# geodesic_matrix_mesh
# -----------------------------------------------------------------------------


def test_matches_grid_closed_form():
    """No external oracle: the triangulated grid has an analytic metric."""
    n, s = 10, 0.7
    faces, verts = grid_mesh(n, s)

    d = fastcore.geodesic_matrix_mesh(faces, verts, sources=[0])
    for i in range(n):
        for j in range(n):
            # The diagonal edge advances both coordinates, so the optimal path takes it
            # while it can (sqrt(2) < 2) and then goes straight.
            expect = s * (np.sqrt(2) * min(i, j) + abs(i - j))
            assert d[0, i * n + j] == pytest.approx(expect, abs=1e-4)

    hops = fastcore.geodesic_matrix_mesh(faces, n_vertices=n * n, sources=[0])
    for i in range(n):
        for j in range(n):
            assert hops[0, i * n + j] == max(i, j)


@pytest.mark.parametrize("weighted", [True, False])
def test_matches_scipy_full(weighted):
    faces, verts = grid_mesh(11, 1.3)
    n = 121

    ours = fastcore.geodesic_matrix_mesh(
        faces, verts if weighted else None, n_vertices=n
    )
    ref = scipy_oracle(faces, verts, n, weighted=weighted)

    np.testing.assert_allclose(as_inf(ours), ref, rtol=1e-5)


@pytest.mark.parametrize("weighted", [True, False])
def test_matches_scipy_with_subsets(weighted):
    faces, verts = grid_mesh(12, 0.9)
    n = 144

    # Deliberately unsorted, and with a duplicate, to pin the ordering contract.
    sources = np.array([100, 0, 37, 143], dtype=np.uint32)
    targets = np.array([5, 5, 120, 1], dtype=np.uint32)

    ours = fastcore.geodesic_matrix_mesh(
        faces,
        verts if weighted else None,
        n_vertices=n,
        sources=sources,
        targets=targets,
    )
    ref = scipy_oracle(
        faces, verts, n, sources=sources, targets=targets, weighted=weighted
    )

    assert ours.shape == (4, 4)
    np.testing.assert_allclose(as_inf(ours), ref, rtol=1e-5)


def test_output_order_follows_the_caller_not_sorted_order():
    faces, verts = grid_mesh(8, 1.0)
    full = fastcore.geodesic_matrix_mesh(faces, verts)

    sources = [40, 3, 17]
    targets = [63, 0]
    sub = fastcore.geodesic_matrix_mesh(faces, verts, sources=sources, targets=targets)

    for i, s in enumerate(sources):
        for j, t in enumerate(targets):
            assert sub[i, j] == full[s, t]


def test_duplicate_targets_are_allowed():
    faces, verts = grid_mesh(6, 1.0)
    d = fastcore.geodesic_matrix_mesh(
        faces, verts, sources=[0], targets=[7, 7, 7, 35]
    )
    assert d.shape == (1, 4)
    assert d[0, 0] == d[0, 1] == d[0, 2]


def test_disconnected_components_are_minus_one():
    # Two disjoint triangles.
    faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.uint32)
    verts = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [10, 0, 0], [11, 0, 0], [10, 1, 0]],
        dtype=np.float64,
    )
    d = fastcore.geodesic_matrix_mesh(faces, verts)

    # Cross-check the component structure against the existing DSU implementation.
    cc = fastcore.mesh_connected_components(faces, 6)
    for i in range(6):
        for j in range(6):
            if cc[i] == cc[j]:
                assert d[i, j] >= 0
            else:
                assert d[i, j] == -1


def test_isolated_vertex():
    # Vertex 3 is counted but is in no face.
    faces = np.array([[0, 1, 2]], dtype=np.uint32)
    d = fastcore.geodesic_matrix_mesh(faces, n_vertices=4)
    assert d[3, 3] == 0
    assert (d[3, :3] == -1).all()
    assert (d[:3, 3] == -1).all()


def test_degenerate_face_does_not_corrupt_adjacency():
    # Face (0, 0, 1) has a repeated vertex.
    faces = np.array([[0, 0, 1], [1, 2, 3]], dtype=np.uint32)
    verts = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [1, 1, 0]], dtype=np.float64)
    d = fastcore.geodesic_matrix_mesh(faces, verts)
    assert d[0, 0] == 0  # no self-loop weirdness
    assert d[0, 1] == pytest.approx(1.0)


def test_limit_prunes_and_is_inclusive():
    faces, verts = grid_mesh(10, 1.0)
    n = 100
    full = fastcore.geodesic_matrix_mesh(faces, verts, sources=[0])

    # Vertex 3 is exactly 3 axis-hops from vertex 0.
    exact = float(full[0, 3])
    assert exact == pytest.approx(3.0)

    at = fastcore.geodesic_matrix_mesh(faces, verts, sources=[0], limit=exact)
    assert at[0, 3] == pytest.approx(exact), "distance == limit must be kept"

    under = fastcore.geodesic_matrix_mesh(faces, verts, sources=[0], limit=exact - 1e-3)
    assert under[0, 3] == -1, "distance > limit must be dropped"

    # And the pruned result must agree with scipy's own `limit`.
    ref = scipy_oracle(faces, verts, n, sources=[0], limit=exact)
    np.testing.assert_allclose(as_inf(at), ref, rtol=1e-5)


def test_symmetry_and_triangle_inequality():
    faces, verts = grid_mesh(9, 1.0)
    d = fastcore.geodesic_matrix_mesh(faces, verts)

    np.testing.assert_array_equal(d, d.T)

    rng = np.random.default_rng(0)
    for _ in range(200):
        s, k, t = rng.integers(0, 81, 3)
        assert d[s, t] <= d[s, k] + d[k, t] + 1e-4


def test_threads_do_not_change_the_result():
    faces, verts = grid_mesh(11, 0.6)
    ref = fastcore.geodesic_matrix_mesh(faces, verts, threads=1)
    for n in (2, 4, 8):
        got = fastcore.geodesic_matrix_mesh(faces, verts, threads=n)
        np.testing.assert_array_equal(got, ref)


def test_deterministic():
    faces, verts = grid_mesh(10, 1.0)
    a = fastcore.geodesic_matrix_mesh(faces, verts)
    b = fastcore.geodesic_matrix_mesh(faces, verts)
    np.testing.assert_array_equal(a, b)


def test_validation():
    faces, verts = grid_mesh(5, 1.0)

    with pytest.raises(ValueError, match="faces"):
        fastcore.geodesic_matrix_mesh(np.zeros((3, 4), dtype=np.uint32), verts)

    with pytest.raises(ValueError, match="vertices"):
        fastcore.geodesic_matrix_mesh(faces, np.zeros((25, 2)))

    with pytest.raises(ValueError, match="[Pp]rovide either"):
        fastcore.geodesic_matrix_mesh(faces)

    with pytest.raises(ValueError, match="sources"):
        fastcore.geodesic_matrix_mesh(faces, verts, sources=[999])

    with pytest.raises(ValueError, match="does not match"):
        fastcore.geodesic_matrix_mesh(faces, verts, n_vertices=99)


# -----------------------------------------------------------------------------
# geodesic_matrix_graph
# -----------------------------------------------------------------------------


def test_graph_matches_scipy_on_a_cyclic_graph():
    """The tree-based `geodesic_matrix` cannot do this at all."""
    rng = np.random.default_rng(42)
    n = 60
    # A random connected-ish graph, definitely with cycles.
    edges = rng.integers(0, n, size=(200, 2)).astype(np.uint32)
    edges = edges[edges[:, 0] != edges[:, 1]]
    w = rng.random(len(edges)).astype(np.float32) + 0.1

    ours = fastcore.geodesic_matrix_graph(edges, n, weights=w)

    g = csr_matrix(
        (np.concatenate([w, w]),
         (np.concatenate([edges[:, 0], edges[:, 1]]),
          np.concatenate([edges[:, 1], edges[:, 0]]))),
        shape=(n, n),
    )
    # csr_matrix SUMS duplicates, so build the oracle from a min-reduced edge set.
    ref = dijkstra(g, directed=False)

    # Where our (min-reduced) graph and scipy's (sum-reduced) differ, scipy can only ever
    # be >= ours. Compare on the subset with no duplicate edges.
    key = edges[:, 0].astype(np.int64) * n + edges[:, 1]
    if len(np.unique(key)) == len(key):
        np.testing.assert_allclose(as_inf(ours), ref, rtol=1e-4)
    else:
        assert (as_inf(ours) <= ref + 1e-4).all()


def test_graph_directed():
    edges = np.array([[0, 1], [1, 2]], dtype=np.uint32)
    w = np.array([1.0, 1.0], dtype=np.float32)

    d = fastcore.geodesic_matrix_graph(edges, 3, weights=w, directed=True)
    assert d[0, 2] == 2
    assert d[2, 0] == -1

    u = fastcore.geodesic_matrix_graph(edges, 3, weights=w, directed=False)
    assert u[2, 0] == 2


def test_graph_unweighted_is_hop_count():
    edges = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.uint32)
    d = fastcore.geodesic_matrix_graph(edges, 4)
    np.testing.assert_array_equal(d[0], [0, 1, 2, 3])


# -----------------------------------------------------------------------------
# nearest / farthest
# -----------------------------------------------------------------------------


def test_nearest_agrees_with_the_matrix():
    faces, verts = grid_mesh(9, 1.1)
    n = 81
    targets = np.array([3, 40, 77], dtype=np.uint32)

    full = as_inf(fastcore.geodesic_matrix_mesh(faces, verts, targets=targets))
    dist, near = fastcore.geodesic_nearest_mesh(faces, verts, targets=targets)

    for s in range(n):
        # A source that is itself a target matches its nearest *distinct* target.
        cand = [full[s, j] for j, t in enumerate(targets) if t != s]
        expect = min(cand) if cand else np.inf
        assert as_inf(dist[s : s + 1])[0] == pytest.approx(expect, abs=1e-4)
        if np.isfinite(expect):
            assert full[s, list(targets).index(near[s])] == pytest.approx(expect, abs=1e-4)


def test_farthest_agrees_with_the_matrix():
    faces, verts = grid_mesh(8, 1.0)
    n = 64
    targets = np.array([0, 9, 63], dtype=np.uint32)

    full = as_inf(fastcore.geodesic_matrix_mesh(faces, verts, targets=targets))
    dist, far = fastcore.geodesic_farthest_mesh(faces, verts, targets=targets)

    for s in range(n):
        cand = [full[s, j] for j, t in enumerate(targets) if t != s]
        expect = max(cand) if cand else np.inf
        assert as_inf(dist[s : s + 1])[0] == pytest.approx(expect, abs=1e-4)


def test_nearest_excludes_self():
    faces, verts = grid_mesh(7, 1.0)
    dist, near = fastcore.geodesic_nearest_mesh(
        faces, verts, sources=[0], targets=[0, 1, 48]
    )
    assert near[0] == 1, "must not match itself"
    assert dist[0] == pytest.approx(1.0)


def test_nearest_unreachable_is_minus_one():
    faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.uint32)
    dist, near = fastcore.geodesic_nearest_mesh(
        faces, n_vertices=6, sources=[0, 1], targets=[4]
    )
    np.testing.assert_array_equal(dist, [-1, -1])
    np.testing.assert_array_equal(near, [-1, -1])


# -----------------------------------------------------------------------------
# csgraph wrapper
# -----------------------------------------------------------------------------


def test_csgraph_dijkstra_now_handles_cyclic_graphs():
    """This used to raise 'Input graph is not a tree'."""
    from navis_fastcore.wrappers.csgraph import dijkstra as our_dijkstra

    faces, verts = grid_mesh(8, 1.0)
    n = 64
    e = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]).astype(np.int64)
    e.sort(axis=1)
    e = np.unique(e, axis=0)
    w = np.linalg.norm(verts[e[:, 0]] - verts[e[:, 1]], axis=1)
    g = csr_matrix(
        (np.concatenate([w, w]),
         (np.concatenate([e[:, 0], e[:, 1]]), np.concatenate([e[:, 1], e[:, 0]]))),
        shape=(n, n),
    )

    idx = np.array([0, 17, 63])
    ours = our_dijkstra(g, directed=False, indices=idx)
    ref = dijkstra(g, directed=False, indices=idx)
    np.testing.assert_allclose(ours, ref, rtol=1e-5)


def test_csgraph_dijkstra_scalar_indices_returns_1d():
    from navis_fastcore.wrappers.csgraph import dijkstra as our_dijkstra

    edges = np.array([[0, 1], [1, 2]])
    g = csr_matrix((np.ones(2), (edges[:, 0], edges[:, 1])), shape=(3, 3))
    out = our_dijkstra(g, directed=False, indices=0)
    assert out.ndim == 1
    np.testing.assert_allclose(out, [0, 1, 2])


def test_csgraph_dijkstra_targets_extension():
    """`targets` is ours, not scipy's -- it avoids materialising all N columns."""
    from navis_fastcore.wrappers.csgraph import dijkstra as our_dijkstra

    faces, verts = grid_mesh(8, 1.0)
    n = 64
    e = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]).astype(np.int64)
    e.sort(axis=1)
    e = np.unique(e, axis=0)
    w = np.linalg.norm(verts[e[:, 0]] - verts[e[:, 1]], axis=1)
    g = csr_matrix(
        (np.concatenate([w, w]),
         (np.concatenate([e[:, 0], e[:, 1]]), np.concatenate([e[:, 1], e[:, 0]]))),
        shape=(n, n),
    )

    idx = np.array([0, 17])
    tgt = np.array([63, 5])
    ours = our_dijkstra(g, directed=False, indices=idx, targets=tgt)
    ref = dijkstra(g, directed=False, indices=idx)[:, tgt]

    assert ours.shape == (2, 2)
    np.testing.assert_allclose(ours, ref, rtol=1e-5)


# -----------------------------------------------------------------------------
# Real mesh, if trimesh is around
# -----------------------------------------------------------------------------


def test_matches_scipy_on_a_real_mesh():
    trimesh = pytest.importorskip("trimesh")

    m = trimesh.creation.icosphere(subdivisions=3)
    faces = np.asarray(m.faces, dtype=np.uint32)
    verts = np.asarray(m.vertices, dtype=np.float64)
    n = len(verts)

    sources = np.array([0, n // 3, n - 1], dtype=np.uint32)

    ours = fastcore.geodesic_matrix_mesh(faces, verts, sources=sources)
    ref = scipy_oracle(faces, verts, n, sources=sources)

    np.testing.assert_allclose(as_inf(ours), ref, rtol=1e-5)


# -----------------------------------------------------------------------------
# unique_edges
# -----------------------------------------------------------------------------


def unique_edges_oracle(faces):
    """Reference implementation: trimesh's exact pipeline, in numpy.

    Expand each face to its three edges, row-sort, pack into the same u64 key
    trimesh's `hashable_rows` bit-bangs, then let `np.unique` do the work.
    """
    e = faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2).astype(np.int64)
    e.sort(axis=1)
    keys = (e[:, 1].astype(np.uint64) << np.uint64(32)) | e[:, 0].astype(np.uint64)
    _, idx, inv = np.unique(keys, return_index=True, return_inverse=True)
    return e[idx], idx.astype(np.int64), inv.astype(np.int64)


def test_unique_edges_matches_numpy_oracle():
    faces, _ = grid_mesh(n=10)
    exp_edges, exp_idx, exp_inv = unique_edges_oracle(faces)

    edges, idx, inv = fastcore.unique_edges(faces, return_index=True, return_inverse=True)
    assert edges.dtype == np.int64
    np.testing.assert_array_equal(edges, exp_edges)
    np.testing.assert_array_equal(idx, exp_idx)
    np.testing.assert_array_equal(inv, exp_inv)

    # Bare call returns just the edges, identical to the full path.
    np.testing.assert_array_equal(fastcore.unique_edges(faces), exp_edges)


def test_unique_edges_random_meshes():
    rng = np.random.default_rng(42)
    for n_faces in (1, 7, 1000):
        faces = rng.integers(0, 200, size=(n_faces, 3)).astype(np.uint32)
        exp_edges, exp_idx, exp_inv = unique_edges_oracle(faces)
        edges, idx, inv = fastcore.unique_edges(
            faces, return_index=True, return_inverse=True
        )
        np.testing.assert_array_equal(edges, exp_edges)
        np.testing.assert_array_equal(idx, exp_idx)
        np.testing.assert_array_equal(inv, exp_inv)


def test_unique_edges_keeps_degenerate_self_loops():
    # trimesh does NOT drop self-loop edges from degenerate faces.
    faces = np.array([[0, 0, 1], [1, 2, 3]], dtype=np.uint32)
    edges = fastcore.unique_edges(faces)
    assert [0, 0] in edges.tolist()
    np.testing.assert_array_equal(edges, unique_edges_oracle(faces)[0])


def test_unique_edges_lengths():
    faces, verts = grid_mesh(n=10, spacing=0.7)
    exp_edges, _, _ = unique_edges_oracle(faces)
    exp_len = np.linalg.norm(verts[exp_edges[:, 0]] - verts[exp_edges[:, 1]], axis=1)

    edges, lengths = fastcore.unique_edges(faces, verts)
    assert lengths.dtype == np.float64
    np.testing.assert_array_equal(edges, exp_edges)
    np.testing.assert_allclose(lengths, exp_len, rtol=1e-14)

    # Lengths always come last, whatever else is requested.
    edges, idx, inv, lengths2 = fastcore.unique_edges(
        faces, verts, return_index=True, return_inverse=True
    )
    np.testing.assert_array_equal(lengths2, lengths)

    # Out-of-bounds vertex indices are caught up front.
    with pytest.raises(ValueError, match="references vertex"):
        fastcore.unique_edges(faces, verts[:-5])


def test_unique_edges_empty():
    faces = np.zeros((0, 3), dtype=np.uint32)
    edges, idx, inv = fastcore.unique_edges(faces, return_index=True, return_inverse=True)
    assert edges.shape == (0, 2)
    assert edges.dtype == np.int64
    assert len(idx) == 0 and len(inv) == 0

    _, lengths = fastcore.unique_edges(faces, np.zeros((0, 3)))
    assert len(lengths) == 0


def test_unique_edges_threads_do_not_change_the_result():
    faces, _ = grid_mesh(n=20)
    np.testing.assert_array_equal(
        fastcore.unique_edges(faces, threads=1), fastcore.unique_edges(faces)
    )


def test_unique_edges_validation():
    with pytest.raises(ValueError, match="must be a 2-D array"):
        fastcore.unique_edges(np.zeros((4, 2), dtype=np.uint32))


def test_unique_edges_matches_trimesh():
    trimesh = pytest.importorskip("trimesh")

    m = trimesh.creation.icosphere(subdivisions=3)
    faces = np.asarray(m.faces, dtype=np.uint32)

    edges, idx, inv, lengths = fastcore.unique_edges(
        faces, m.vertices, return_index=True, return_inverse=True
    )

    np.testing.assert_array_equal(edges, m.edges_unique)
    # `edges_unique_idx` is not a public property, only a cache entry populated
    # as a side effect of `edges_unique` (accessed above).
    np.testing.assert_array_equal(idx, m._cache["edges_unique_idx"])
    np.testing.assert_array_equal(inv, m.edges_unique_inverse)
    np.testing.assert_array_equal(inv.reshape(-1, 3), m.faces_unique_edges)
    # trimesh computes the norm through a BLAS dot, so allow float noise.
    np.testing.assert_allclose(lengths, m.edges_unique_length, rtol=1e-14)


# -----------------------------------------------------------------------------
# Graph primitives: components, level sets, contraction, spanning tree
#
# The oracle here is igraph, since these exist to replace exactly the igraph
# calls that skeletor's mesh skeletonization makes.
# -----------------------------------------------------------------------------


def random_graph(n_nodes=200, n_edges=600, seed=0):
    """A random undirected graph, deduplicated and free of self-loops."""
    rng = np.random.default_rng(seed)
    e = rng.integers(0, n_nodes, size=(n_edges, 2))
    e.sort(axis=1)
    e = np.unique(e, axis=0)
    return e[e[:, 0] != e[:, 1]].astype(np.uint32)


def as_partition(labels):
    """Group node indices by label into a comparable set of frozensets."""
    out = {}
    for node, lab in enumerate(labels):
        out.setdefault(lab, set()).add(node)
    return {frozenset(v) for v in out.values()}


def test_connected_components_graph_matches_igraph():
    ig = pytest.importorskip("igraph")

    edges = random_graph()
    n = 200
    ours = fastcore.connected_components_graph(edges, n)

    g = ig.Graph(n=n, edges=edges.tolist(), directed=False)
    ref = [set(cc) for cc in g.connected_components()]

    assert as_partition(ours) == {frozenset(cc) for cc in ref}


def test_connected_components_graph_agrees_with_the_mesh_version():
    """Same mesh, two entry points — the labels themselves must match, not just
    the partition."""
    faces, verts = grid_mesh(n=9)
    n = len(verts)
    edges = fastcore.unique_edges(faces).astype(np.uint32)

    np.testing.assert_array_equal(
        fastcore.connected_components_graph(edges, n),
        fastcore.mesh_connected_components(faces, n),
    )


def test_level_set_components_matches_the_igraph_subgraph_loop():
    """The oracle is the loop this function exists to delete: for each level,
    induce the subgraph on those vertices and find its components."""
    ig = pytest.importorskip("igraph")

    faces, verts = grid_mesh(n=15)
    n = len(verts)
    edges = fastcore.unique_edges(faces).astype(np.uint32)

    # A genuine wavefront: hop distance from a corner, which on this grid is max(i, j).
    dist = fastcore.geodesic_matrix_mesh(faces, n_vertices=n, sources=[0])[0]
    labels = dist.astype(np.int64)

    ids, n_comp = fastcore.level_set_components(edges, n, labels)

    # Oracle: igraph, one induced subgraph per level, exactly as skeletor does it.
    g = ig.Graph(n=n, edges=edges.tolist(), directed=False)
    ref = set()
    for lvl in np.unique(labels):
        ix = np.where(labels == lvl)[0]
        sg = g.subgraph(ix, implementation="create_from_scratch")
        for cc in sg.connected_components():
            ref.add(frozenset(ix[cc].tolist()))

    assert as_partition(ids) == ref
    assert n_comp == len(ref)
    # Ids are dense and every vertex is assigned (all are reachable here).
    assert ids.min() == 0 and ids.max() == n_comp - 1


def test_level_set_components_excludes_negative_labels():
    """`-1` is what geodesic_matrix_* returns for unreachable. Those must be
    dropped, not fused into one phantom level."""
    edges = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.uint32)

    ids, n = fastcore.level_set_components(edges, 4, [-1, -1, 5, 5])
    np.testing.assert_array_equal(ids, [-1, -1, 0, 0])
    assert n == 1

    # Feeding an unreachable distance row straight in must work: two disjoint
    # triangles, seeded only in the first.
    faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.uint32)
    mesh_edges = fastcore.unique_edges(faces).astype(np.uint32)
    dist = fastcore.geodesic_matrix_mesh(faces, n_vertices=6, sources=[0])[0]
    assert (dist[3:] == -1).all()

    ids, n = fastcore.level_set_components(mesh_edges, 6, dist.astype(np.int64))
    np.testing.assert_array_equal(ids[3:], [-1, -1, -1])
    assert n == 2  # vertex 0 at distance 0; vertices 1, 2 at distance 1


def test_level_set_components_all_same_label_is_plain_components():
    """With one label everywhere it degenerates to connected components."""
    edges = random_graph()
    n = 200
    ids, n_comp = fastcore.level_set_components(edges, n, np.zeros(n, dtype=np.int64))
    assert as_partition(ids) == as_partition(
        fastcore.connected_components_graph(edges, n)
    )


def test_contract_vertices_matches_igraph():
    ig = pytest.importorskip("igraph")

    edges = random_graph()
    n = 200
    rng = np.random.default_rng(1)
    mapping = rng.integers(0, 40, size=n).astype(np.uint32)

    ours = fastcore.contract_vertices(edges, mapping)

    g = ig.Graph(n=n, edges=edges.tolist(), directed=False)
    g.contract_vertices(mapping.tolist())
    g = g.simplify()
    ref = np.array(g.get_edgelist(), dtype=np.int64)
    ref.sort(axis=1)
    ref = np.unique(ref, axis=0)

    ours_sorted = ours[np.lexsort((ours[:, 0], ours[:, 1]))]
    ref_sorted = ref[np.lexsort((ref[:, 0], ref[:, 1]))]
    np.testing.assert_array_equal(ours_sorted, ref_sorted)


def test_contract_vertices_identity_is_unique_edges():
    faces, verts = grid_mesh(n=9)
    edges = fastcore.unique_edges(faces)
    mapping = np.arange(len(verts), dtype=np.uint32)

    np.testing.assert_array_equal(
        fastcore.contract_vertices(edges.astype(np.uint32), mapping), edges
    )


def test_minimum_spanning_tree_matches_scipy():
    """Distinct weights make the MST unique, so the edge sets must match exactly."""
    from scipy.sparse.csgraph import minimum_spanning_tree as scipy_mst

    edges = random_graph(n_nodes=150, n_edges=500, seed=3)
    n = 150
    rng = np.random.default_rng(4)
    # Distinct weights -> unique MST.
    w = rng.permutation(len(edges)).astype(np.float32) + 1.0

    keep = fastcore.minimum_spanning_tree(edges, n, w)
    ours = {frozenset(e) for e in edges[keep].tolist()}

    g = csr_matrix((w, (edges[:, 0], edges[:, 1])), shape=(n, n))
    ref_mat = scipy_mst(g).tocoo()
    ref = {frozenset((int(a), int(b))) for a, b in zip(ref_mat.row, ref_mat.col)}

    assert ours == ref
    np.testing.assert_allclose(w[keep].sum(), ref_mat.data.sum(), rtol=1e-6)


def test_minimum_spanning_tree_matches_igraph():
    ig = pytest.importorskip("igraph")

    edges = random_graph(n_nodes=150, n_edges=500, seed=3)
    n = 150
    rng = np.random.default_rng(5)
    w = rng.permutation(len(edges)).astype(np.float32) + 1.0

    keep = fastcore.minimum_spanning_tree(edges, n, w)
    ours = {frozenset(e) for e in edges[keep].tolist()}

    g = ig.Graph(n=n, edges=edges.tolist(), directed=False)
    tree = g.spanning_tree(weights=w.tolist())
    ref = {frozenset(e) for e in tree.get_edgelist()}

    assert ours == ref


def test_minimum_spanning_tree_maximize_beats_the_reciprocal_hack():
    """`maximize=True` must equal what igraph gives for `weights=1/w` -- the trick
    it exists to replace -- and unlike that trick must survive zero weights."""
    ig = pytest.importorskip("igraph")

    edges = random_graph(n_nodes=100, n_edges=300, seed=6)
    n = 100
    rng = np.random.default_rng(7)
    w = rng.permutation(len(edges)).astype(np.float32) + 1.0

    keep = fastcore.minimum_spanning_tree(edges, n, w, maximize=True)
    ours = {frozenset(e) for e in edges[keep].tolist()}

    g = ig.Graph(n=n, edges=edges.tolist(), directed=False)
    ref = {frozenset(e) for e in g.spanning_tree(weights=(1 / w).tolist()).get_edgelist()}
    assert ours == ref

    # A zero weight makes 1/w infinite; ours takes it in stride and, being the
    # *cheapest* edge, it must be in the minimum tree.
    w0 = w.copy()
    w0[0] = 0.0
    keep = fastcore.minimum_spanning_tree(edges, n, w0)
    assert 0 in keep.tolist()


def test_minimum_spanning_tree_of_a_forest():
    """Disconnected input yields one tree per component: n_nodes - n_components edges."""
    edges = random_graph(n_nodes=200, n_edges=300, seed=8)
    n = 200
    n_comp = len(set(fastcore.connected_components_graph(edges, n).tolist()))

    keep = fastcore.minimum_spanning_tree(edges, n)
    assert len(keep) == n - n_comp

    # The result must be acyclic and preserve the component structure.
    assert as_partition(
        fastcore.connected_components_graph(edges[keep], n)
    ) == as_partition(fastcore.connected_components_graph(edges, n))


def test_graph_primitives_validation():
    edges = np.array([[0, 1], [1, 2]], dtype=np.uint32)

    with pytest.raises(ValueError):
        fastcore.connected_components_graph(edges, 2)  # node 2 out of range
    with pytest.raises(ValueError):
        fastcore.level_set_components(edges, 3, [0, 0])  # labels too short
    with pytest.raises(ValueError):
        fastcore.contract_vertices(edges, [0, 0])  # mapping too short
    with pytest.raises(ValueError):
        fastcore.minimum_spanning_tree(edges, 3, [1.0])  # weights too short
    with pytest.raises(ValueError):
        fastcore.minimum_spanning_tree(np.zeros((2, 3), dtype=np.uint32), 3)


def test_graph_primitives_threads_do_not_change_the_result():
    edges = random_graph()
    mapping = (np.arange(200) // 5).astype(np.uint32)
    rng = np.random.default_rng(9)
    w = rng.random(len(edges)).astype(np.float32)

    for t in (1, 2, 4):
        np.testing.assert_array_equal(
            fastcore.contract_vertices(edges, mapping, threads=t),
            fastcore.contract_vertices(edges, mapping, threads=1),
        )
        np.testing.assert_array_equal(
            fastcore.minimum_spanning_tree(edges, 200, w, threads=t),
            fastcore.minimum_spanning_tree(edges, 200, w, threads=1),
        )


# -----------------------------------------------------------------------------
# Predecessors, paths and geodesic clustering
#
# Oracles: scipy's `dijkstra(..., return_predecessors=True)` (note its predecessor
# sentinel is -9999, not -1) and igraph's `get_shortest_paths`.
# -----------------------------------------------------------------------------


def as_csr(edges, n_nodes, weights=None, directed=False):
    """The same graph as a scipy sparse matrix, parallel edges collapsed to the shortest."""
    w = np.ones(len(edges), dtype=np.float64) if weights is None else np.asarray(
        weights, dtype=np.float64
    )
    rows = list(edges[:, 0])
    cols = list(edges[:, 1])
    data = list(w)
    if not directed:
        rows, cols = rows + cols, cols + rows
        data = data + data
    m = csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))
    # `csr_matrix` sums duplicates; take the minimum instead, as fastcore does.
    m = m.tolil()
    for i, j, d in zip(rows, cols, data):
        if m[i, j] > d or m[i, j] == 0:
            m[i, j] = d
    return m.tocsr()


def test_predecessors_match_scipy():
    edges = random_graph(n_nodes=120, n_edges=400, seed=3)
    rng = np.random.default_rng(3)
    w = rng.random(len(edges)).astype(np.float32)
    sources = np.array([0, 7, 55], dtype=np.uint32)

    dist, pred = fastcore.geodesic_predecessors(edges, 120, w, sources=sources)
    ref_d, ref_p = dijkstra(
        as_csr(edges, 120, w), directed=False, indices=sources, return_predecessors=True
    )

    # Distances agree outright.
    np.testing.assert_allclose(np.where(dist < 0, np.inf, dist), ref_d, rtol=1e-5)

    # Predecessors need not be *identical* - equal-length paths are resolved
    # differently - but every chain must exist and weigh the reference distance.
    lookup = {}
    for (u, v), x in zip(edges, w):
        key = (min(u, v), max(u, v))
        lookup[key] = min(lookup.get(key, np.inf), x)

    for r, s in enumerate(sources):
        assert pred[r, s] == -1
        for t in range(120):
            if not np.isfinite(ref_d[r, t]):
                assert dist[r, t] == -1 and pred[r, t] == -1
                continue
            total, cur, hops = 0.0, t, 0
            while cur != s:
                p = pred[r, cur]
                assert p >= 0, f"chain broke at {cur}"
                total += lookup[(min(p, cur), max(p, cur))]
                cur = p
                hops += 1
                assert hops <= 120, "predecessor cycle"
            assert total == pytest.approx(ref_d[r, t], rel=1e-4)


def test_predecessors_unweighted_match_scipy():
    edges = random_graph(n_nodes=80, n_edges=200, seed=5)
    dist, pred = fastcore.geodesic_predecessors(edges, 80, sources=[0])
    ref_d = dijkstra(as_csr(edges, 80), directed=False, indices=[0], unweighted=True)

    np.testing.assert_allclose(np.where(dist < 0, np.inf, dist), ref_d)
    # A hop-count chain must be exactly as long as the distance says.
    for t in range(80):
        if not np.isfinite(ref_d[0, t]):
            continue
        cur, hops = t, 0
        while cur != 0:
            cur = pred[0, cur]
            hops += 1
        assert hops == ref_d[0, t]


def test_geodesic_path_matches_igraph():
    ig = pytest.importorskip("igraph")

    edges = random_graph(n_nodes=150, n_edges=500, seed=11)
    rng = np.random.default_rng(11)
    w = rng.random(len(edges)).astype(np.float32)
    targets = np.arange(150, dtype=np.uint32)

    ours = fastcore.geodesic_path(edges, 150, 0, targets, weights=w)
    g = ig.Graph(n=150, edges=[tuple(e) for e in edges], directed=False)
    theirs = g.get_shortest_paths(0, to=list(targets), weights=list(w.astype(np.float64)))

    lengths = {}
    for (u, v), x in zip(edges, w.astype(np.float64)):
        lengths[(min(u, v), max(u, v))] = x

    def walk(path):
        return sum(lengths[(min(a, b), max(a, b))] for a, b in zip(path[:-1], path[1:]))

    for t, (mine, ref) in enumerate(zip(ours, theirs)):
        if not ref:  # igraph returns [] for unreachable
            assert len(mine) == 0
            continue
        assert mine[0] == 0 and mine[-1] == t
        assert walk(list(mine)) == pytest.approx(walk(ref), rel=1e-5)


def test_geodesic_path_zero_weights_are_free():
    # The TEASAR mechanism: zeroing the edges of a path makes it free to re-traverse.
    edges = np.array([[0, 1], [1, 2], [2, 3], [0, 3]], dtype=np.uint32)
    w = np.array([0.0, 0.0, 0.0, 10.0], dtype=np.float32)
    (path,) = fastcore.geodesic_path(edges, 4, 0, [3], weights=w)
    np.testing.assert_array_equal(path, [0, 1, 2, 3])


def test_geodesic_path_edge_cases():
    edges = np.array([[0, 1], [2, 3]], dtype=np.uint32)
    paths = fastcore.geodesic_path(edges, 4, 0, [0, 1, 3])
    np.testing.assert_array_equal(paths[0], [0])
    np.testing.assert_array_equal(paths[1], [0, 1])
    assert len(paths[2]) == 0

    with pytest.raises(ValueError):
        fastcore.geodesic_path(edges, 4, 9, [0])


def test_geodesic_predecessors_directed_is_one_way():
    edges = np.array([[0, 1], [1, 2]], dtype=np.uint32)
    dist, pred = fastcore.geodesic_predecessors(edges, 3, directed=True, sources=[2])
    np.testing.assert_array_equal(dist[0], [-1, -1, 0])
    np.testing.assert_array_equal(pred[0], [-1, -1, -1])


def test_geodesic_clusters_are_balls_around_their_seeds():
    edges = random_graph(n_nodes=200, n_edges=700, seed=17)
    rng = np.random.default_rng(17)
    w = rng.random(len(edges)).astype(np.float32)
    max_dist = 0.8

    labels, n = fastcore.geodesic_clusters(edges, 200, max_dist, weights=w)
    assert labels.min() >= 0, "every node is labelled"
    assert set(np.unique(labels)) == set(range(n))

    # Seeds are the lowest-index member of each cluster, because clusters are grown
    # in ascending index order when no preferred seeds are given.
    seeds = np.array([np.flatnonzero(labels == c).min() for c in range(n)], np.uint32)
    d = dijkstra(as_csr(edges, 200, w), directed=False, indices=seeds)
    for node, c in enumerate(labels):
        assert d[c, node] <= max_dist + 1e-6

    # And each node belongs to the *first* cluster whose ball contains it.
    for node, c in enumerate(labels):
        first = np.flatnonzero(d[:, node] <= max_dist + 1e-6).min()
        assert first == c


def test_geodesic_clusters_use_true_distance_not_traversal_path():
    # A 5-cycle within 2 hops of node 0 the short way round; a depth-first walk that
    # took the long branch first would wrongly reject node 3.
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]], dtype=np.uint32)
    labels, n = fastcore.geodesic_clusters(edges, 5, 2)
    np.testing.assert_array_equal(labels, [0, 0, 0, 0, 0])
    assert n == 1


def test_geodesic_clusters_preferred_seeds():
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]], dtype=np.uint32)
    labels, n = fastcore.geodesic_clusters(edges, 6, 1, seeds=[3, 3, 1])
    np.testing.assert_array_equal(labels, [1, 1, 0, 0, 0, 2])
    assert n == 3


def test_geodesic_clusters_validation():
    edges = np.array([[0, 1]], dtype=np.uint32)
    with pytest.raises(ValueError):
        fastcore.geodesic_clusters(edges, 2, -1)
    with pytest.raises(ValueError):
        fastcore.geodesic_clusters(edges, 2, np.inf)
    with pytest.raises(ValueError):
        fastcore.geodesic_clusters(edges, 2, 1, weights=[1.0, 2.0])


# -----------------------------------------------------------------------------
# GeodesicGraph
# -----------------------------------------------------------------------------
#
# The oracle here is navis' own pure-Python implementation (`navis.ml.chunk`'s
# `_Geodesic.grow` / `_ConnectedCloud.grow`), ported verbatim below. `GeodesicGraph.grow`
# exists to replace it, so "produces the same fragments" is the property that matters.


def _navis_grow(indptr, indices, data, seed, size, forbidden=None):
    """Verbatim port of `navis.ml.chunk._Geodesic.grow`."""
    import heapq

    region, dists, settled = [], [], set()
    heap = [(0.0, int(seed))]
    while heap and len(region) < size:
        d, u = heapq.heappop(heap)
        if u in settled:
            continue
        settled.add(u)
        region.append(u)
        dists.append(d)
        for j in range(indptr[u], indptr[u + 1]):
            v = int(indices[j])
            if v in settled:
                continue
            if forbidden is not None and forbidden[v]:
                continue
            heapq.heappush(heap, (d + float(data[j]), v))
    return np.array(region, dtype=np.int64), np.array(dists, dtype=float)


def _navis_grow_cloud(indptr, indices, data, by_vtx, svtx, seed, size, forbidden=None):
    """Verbatim port of `navis.ml.chunk._ConnectedCloud.grow`."""
    import heapq

    settled, got, gdist = set(), [], []
    heap = [(0.0, int(svtx[seed]))]
    while heap and len(got) < size:
        d, u = heapq.heappop(heap)
        if u in settled:
            continue
        settled.add(u)
        for s in by_vtx.get(u, ()):
            if forbidden is None or not forbidden[s]:
                got.append(s)
                gdist.append(d)
        for j in range(indptr[u], indptr[u + 1]):
            v = int(indices[j])
            if v in settled:
                continue
            if forbidden is not None:
                sv = by_vtx.get(v)
                if sv is not None and all(forbidden[s] for s in sv):
                    continue
            heapq.heappush(heap, (d + float(data[j]), v))
    return (np.array(got[:size], dtype=np.int64),
            np.array(gdist[:size], dtype=float))


def _csr_parts(edges, n_nodes, weights):
    """The symmetric CSR the navis reference walks, matching `_build_csr`."""
    m = as_csr(edges, n_nodes, weights)
    return m.indptr, m.indices, m.data


def _partition(grow, n_items, size):
    """The `_partition` driver: seed at the first unclaimed item, grow, mark, repeat."""
    claimed = np.zeros(n_items, dtype=bool)
    frags = []
    while not claimed.all():
        seed = int(np.argmax(~claimed))
        frag = np.asarray(grow(seed, size, claimed))
        assert len(frag), "growth from an unclaimed seed cannot be empty"
        claimed[frag] = True
        frags.append(frag)
    return frags


def test_grow_matches_navis_geodesic_reference():
    # Random weights, so no two paths tie and float32-vs-float64 accumulation cannot
    # reorder anything: the fragments must agree item for item, in order.
    n = 200
    edges = random_graph(n_nodes=n, n_edges=900, seed=3)
    rng = np.random.default_rng(3)
    w = rng.random(len(edges)).astype(np.float32)
    indptr, indices, data = _csr_parts(edges, n, w)

    g = fastcore.GeodesicGraph(edges, n, weights=w)
    assert (g.n_nodes, g.n_items) == (n, n)

    for size in (1, 5, 40, 200, 1000):
        for seed in (0, 17, 111, n - 1):
            np.testing.assert_array_equal(
                g.grow(seed, size),
                _navis_grow(indptr, indices, data, seed, size)[0],
                err_msg=f"seed={seed} size={size}",
            )

    # And through a full partition, where `forbidden` grows between calls.
    for size in (3, 25, 60):
        mine = _partition(lambda s, k, f: g.grow(s, k, forbidden=f), n, size)
        theirs = _partition(
            lambda s, k, f: _navis_grow(indptr, indices, data, s, k, f)[0], n, size
        )
        assert len(mine) == len(theirs)
        for a, b in zip(mine, theirs):
            np.testing.assert_array_equal(a, b, err_msg=f"size={size}")


def test_grow_matches_navis_connected_cloud_reference():
    # Same, for the cloud backend: many items per node, and plenty of nodes with none.
    n = 200
    edges = random_graph(n_nodes=n, n_edges=900, seed=5)
    rng = np.random.default_rng(5)
    w = rng.random(len(edges)).astype(np.float32)
    indptr, indices, data = _csr_parts(edges, n, w)

    # ~350 items over 200 nodes, unevenly: some nodes carry several, many carry none.
    item_nodes = np.sort(rng.integers(0, n, size=350)).astype(np.uint32)
    by_vtx = {}
    for i, v in enumerate(item_nodes):
        by_vtx.setdefault(int(v), []).append(i)
    assert len(by_vtx) < n, "the test is only meaningful if some nodes are empty"

    g = fastcore.GeodesicGraph(edges, n, weights=w, item_nodes=item_nodes)
    assert (g.n_nodes, g.n_items) == (n, 350)

    for size in (1, 8, 64, 350, 1000):
        for seed in (0, 42, 200, 349):
            np.testing.assert_array_equal(
                g.grow(seed, size),
                _navis_grow_cloud(indptr, indices, data, by_vtx, item_nodes, seed, size)[0],
                err_msg=f"seed={seed} size={size}",
            )

    for size in (4, 32, 90):
        mine = _partition(lambda s, k, f: g.grow(s, k, forbidden=f), 350, size)
        theirs = _partition(
            lambda s, k, f: _navis_grow_cloud(
                indptr, indices, data, by_vtx, item_nodes, s, k, f
            )[0],
            350,
            size,
        )
        assert len(mine) == len(theirs)
        for a, b in zip(mine, theirs):
            np.testing.assert_array_equal(a, b, err_msg=f"size={size}")


def test_grow_is_a_ball_in_increasing_distance_order():
    # Independent oracle: scipy's distances from the seed. The region must be `size`
    # items ordered by distance, with nothing outside it closer than its edge.
    n = 200
    edges = random_graph(n_nodes=n, n_edges=800, seed=11)
    rng = np.random.default_rng(11)
    w = rng.random(len(edges)).astype(np.float32)
    d = dijkstra(as_csr(edges, n, w), directed=False, indices=[7])[0]

    g = fastcore.GeodesicGraph(edges, n, weights=w)
    region = g.grow(7, 50)
    assert len(region) == 50
    assert region[0] == 7
    assert np.all(np.diff(d[region]) >= -1e-6), "settle order is distance order"
    outside = np.setdiff1d(np.arange(n), region)
    assert d[outside].min() >= d[region].max() - 1e-6


def test_grow_partition_is_an_exact_cover_of_connected_fragments():
    faces, verts = grid_mesh(12)
    n = len(verts)
    edges, lengths = fastcore.unique_edges(faces, verts)
    edges = edges.astype(np.uint32)

    for weights in (None, lengths.astype(np.float32)):
        g = fastcore.GeodesicGraph(edges, n, weights=weights)
        for size in (1, 7, 33, 144, 500):
            frags = _partition(lambda s, k, f: g.grow(s, k, forbidden=f), n, size)
            seen = np.zeros(n, dtype=int)
            for frag in frags:
                assert len(frag) <= size
                seen[frag] += 1
            assert np.all(seen == 1), f"size={size}: exact cover"

            # Every fragment is a single connected piece — the property the walls exist
            # to preserve. Checked with scipy, not with fastcore.
            for frag in frags:
                sub = as_csr(edges, n, weights)[np.ix_(frag, frag)]
                from scipy.sparse.csgraph import connected_components

                assert connected_components(sub, directed=False)[0] == 1


def test_grow_walls_and_conduits():
    # Path 0-1-2-3-4 with items only on nodes 0, 2 and 4: the odd nodes are conduits.
    edges = np.array([[i, i + 1] for i in range(4)], dtype=np.uint32)
    g = fastcore.GeodesicGraph(edges, 5, item_nodes=[0, 2, 4])

    # Empty nodes conduct, so all three items are one connected region.
    np.testing.assert_array_equal(g.grow(0, 3), [0, 1, 2])

    # Claiming the middle item walls its node off: item 2 is now unreachable.
    np.testing.assert_array_equal(
        g.grow(0, 3, forbidden=np.array([False, True, False])), [0]
    )
    # Claiming an item beyond it changes nothing about the conduit in between.
    np.testing.assert_array_equal(
        g.grow(0, 3, forbidden=np.array([False, False, True])), [0, 1]
    )
    # Without `forbidden` there are no walls at all.
    np.testing.assert_array_equal(g.grow(0, 3, forbidden=None), [0, 1, 2])


def test_grow_stays_within_its_connected_component():
    edges = np.array([[0, 1], [1, 2], [2, 0], [3, 4], [4, 5], [5, 3]], dtype=np.uint32)
    g = fastcore.GeodesicGraph(edges, 6)
    assert sorted(g.grow(0, 100).tolist()) == [0, 1, 2]
    assert sorted(g.grow(4, 100).tolist()) == [3, 4, 5]


def test_grow_repeated_queries_are_reproducible():
    # The scratch space outlives the query; if `reset` left anything behind, an
    # interleaved query would perturb the repeat.
    n = 150
    edges = random_graph(n_nodes=n, n_edges=600, seed=23)
    rng = np.random.default_rng(23)
    w = rng.random(len(edges)).astype(np.float32)
    g = fastcore.GeodesicGraph(edges, n, weights=w)

    first = g.grow(5, 40)
    for _ in range(5):
        g.grow(0, n)
        g.grow(n - 1, 3)
        np.testing.assert_array_equal(g.grow(5, 40), first)


def test_grow_degenerate_requests():
    edges = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.uint32)
    g = fastcore.GeodesicGraph(edges, 4)
    assert len(g.grow(2, 0)) == 0
    assert g.grow(2, 1).tolist() == [2]

    # No edges at all: every node is its own component.
    g = fastcore.GeodesicGraph(np.zeros((0, 2), dtype=np.uint32), 3)
    assert g.grow(1, 10).tolist() == [1]

    # A graph whose nodes mostly carry nothing.
    g = fastcore.GeodesicGraph(edges, 4, item_nodes=[1])
    assert g.n_items == 1
    assert g.grow(0, 5).tolist() == [0]


def test_grow_accepts_a_non_contiguous_or_non_bool_mask():
    edges = np.array([[i, i + 1] for i in range(5)], dtype=np.uint32)
    g = fastcore.GeodesicGraph(edges, 6)
    # A strided view and a plain list must both work, matching the bool array.
    ref = g.grow(0, 3, forbidden=np.array([False, False, True, False, False, False]))
    strided = np.zeros(12, dtype=bool)
    strided[4] = True  # element 2 of the ::2 view
    np.testing.assert_array_equal(g.grow(0, 3, forbidden=strided[::2]), ref)
    np.testing.assert_array_equal(g.grow(0, 3, forbidden=[0, 0, 1, 0, 0, 0]), ref)


def test_geodesic_graph_validation():
    edges = np.array([[0, 1], [1, 2]], dtype=np.uint32)
    with pytest.raises(ValueError):
        fastcore.GeodesicGraph(edges, 2)  # edge references node 2
    with pytest.raises(ValueError):
        fastcore.GeodesicGraph(edges, 3, weights=[1.0])  # one weight per edge

    g = fastcore.GeodesicGraph(edges, 3)
    with pytest.raises(ValueError):
        g.grow(3, 2)  # seed out of range
    with pytest.raises(ValueError):
        g.grow(0, 2, forbidden=np.zeros(2, dtype=bool))  # wrong mask length


def test_grow_is_an_exact_ball_on_a_tie_rich_graph():
    """Ties may reorder, but the region is still exactly the right set.

    fastcore searches in float32 where the navis reference uses float64. On a graph
    whose edge lengths tie in float32 - a symmetric mesh is full of them - the two can
    settle equally-distant nodes in a different order, so the *sequences* diverge even
    though both are correct. What must not drift is the ball property itself: nothing
    outside the region may be nearer than the farthest node inside it, measured against
    a float64 oracle.
    """
    faces, verts = grid_mesh(20)
    n = len(verts)
    edges, lengths = fastcore.unique_edges(faces, verts)
    edges = edges.astype(np.uint32)
    assert len(np.unique(lengths.astype(np.float32))) < len(lengths), "ties expected"

    g = fastcore.GeodesicGraph(edges, n, weights=lengths.astype(np.float32))
    m = as_csr(edges, n, lengths)
    for seed in (0, 210, n - 1):
        d = dijkstra(m, directed=False, indices=[seed])[0]
        for size in (16, 64, 200):
            region = g.grow(seed, size)
            assert len(region) == size and region[0] == seed
            assert np.all(np.diff(d[region]) >= -1e-5), "settle order is distance order"
            outside = np.setdiff1d(np.arange(n), region)
            assert d[region].max() <= d[outside].min() + 1e-5, (
                f"seed={seed} size={size}: region is not a ball"
            )


# -----------------------------------------------------------------------------
# GeodesicGraph.farthest_seed
# -----------------------------------------------------------------------------


class _NavisSeeder:
    """Verbatim port of `navis.ml.chunk._Geodesic`'s FPS seeding.

    Keeps its own incremental `_fps_min` exactly as navis does - an unpruned multi-source
    Dijkstra over the *whole* graph per fold, which is the cost `farthest_seed` removes.
    """

    def __init__(self, csr):
        self.csr = csr
        self.n_comp, self.labels = connected_components(csr, directed=False)
        self._fps_min = None
        self._fps_seen = None

    def seed(self, done):
        if done.any():
            self._fps_fold(done)
            reachable = np.isfinite(self._fps_min) & ~done
            if reachable.any():
                return int(np.argmax(np.where(reachable, self._fps_min, -np.inf)))
        return self._largest_unset(done)

    def _fps_fold(self, done):
        if self._fps_min is None:
            self._fps_min = np.full(done.shape[0], np.inf)
            self._fps_seen = np.zeros(done.shape[0], dtype=bool)
        new = done & ~self._fps_seen
        if new.any():
            d = dijkstra(
                self.csr, directed=False, indices=np.where(new)[0], min_only=True
            )
            np.minimum(self._fps_min, d, out=self._fps_min)
            self._fps_seen |= done

    def _largest_unset(self, done):
        unset = ~done
        counts = np.bincount(self.labels[unset], minlength=self.n_comp)
        best = int(np.argmax(counts))
        return int(np.flatnonzero(unset & (self.labels == best))[0])


def test_farthest_seed_matches_navis_reference():
    # Random weights, so nothing ties and float32-vs-float64 cannot reorder the argmax:
    # every seed of a long run must agree exactly.
    n = 250
    edges = random_graph(n_nodes=n, n_edges=1100, seed=13)
    rng = np.random.default_rng(13)
    w = rng.random(len(edges)).astype(np.float32)

    g = fastcore.GeodesicGraph(edges, n, weights=w)
    ref = _NavisSeeder(as_csr(edges, n, w))

    mine, theirs = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
    for step in range(60):
        a, b = g.farthest_seed(mine), ref.seed(theirs)
        assert a == b, f"step {step}: fastcore {a}, navis {b}"
        mine[a] = theirs[b] = True


def test_farthest_seed_matches_navis_reference_unweighted():
    n = 250
    edges = random_graph(n_nodes=n, n_edges=1100, seed=29)
    g = fastcore.GeodesicGraph(edges, n)
    ref = _NavisSeeder(as_csr(edges, n))

    mine, theirs = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
    for step in range(60):
        a, b = g.farthest_seed(mine), ref.seed(theirs)
        assert a == b, f"step {step}: fastcore {a}, navis {b}"
        mine[a] = theirs[b] = True


def test_farthest_seed_is_a_true_argmax_on_a_tie_rich_graph():
    # On a mesh, distances tie constantly, so which of several equally-far nodes is picked
    # is not something to pin down. That it is *one of* the farthest is.
    faces, verts = grid_mesh(12)
    n = len(verts)
    edges, lengths = fastcore.unique_edges(faces, verts)
    edges = edges.astype(np.uint32)
    m = as_csr(edges, n, lengths)

    g = fastcore.GeodesicGraph(edges, n, weights=lengths.astype(np.float32))
    done = np.zeros(n, dtype=bool)
    done[0] = True
    for step in range(25):
        s = g.farthest_seed(done)
        d = dijkstra(m, directed=False, indices=np.flatnonzero(done), min_only=True)
        eligible = np.isfinite(d) & ~done
        assert eligible[s], f"step {step}: seed {s} is not an eligible candidate"
        assert d[s] >= d[eligible].max() - 1e-5, (
            f"step {step}: seed {s} at {d[s]} is not farthest ({d[eligible].max()})"
        )
        done[s] = True


def test_farthest_seed_batched_folds_match_one_at_a_time():
    # The `cover` pattern marks a whole grown region done per call, not a single item.
    n = 200
    edges = random_graph(n_nodes=n, n_edges=900, seed=31)
    rng = np.random.default_rng(31)
    w = rng.random(len(edges)).astype(np.float32)

    g = fastcore.GeodesicGraph(edges, n, weights=w)
    done = np.zeros(n, dtype=bool)
    for _ in range(12):
        s = g.farthest_seed(done)
        done[g.grow(s, 12)] = True
        # A graph that has never seen the intermediate states must give the same answer.
        fresh = fastcore.GeodesicGraph(edges, n, weights=w)
        assert g.farthest_seed(done) == fresh.farthest_seed(done)


def test_farthest_seed_with_items():
    # Cloud case: many items per node, plenty of nodes with none. The reference is the
    # brute-force argmax over item distances, computed from scratch each step.
    n = 150
    edges = random_graph(n_nodes=n, n_edges=650, seed=37)
    rng = np.random.default_rng(37)
    w = rng.random(len(edges)).astype(np.float32)
    item_nodes = np.sort(rng.integers(0, n, size=260)).astype(np.uint32)
    m = as_csr(edges, n, w)

    g = fastcore.GeodesicGraph(edges, n, weights=w, item_nodes=item_nodes)
    done = np.zeros(len(item_nodes), dtype=bool)
    done[0] = True
    for step in range(30):
        s = g.farthest_seed(done)
        d = dijkstra(
            m, directed=False, indices=np.unique(item_nodes[done]), min_only=True
        )
        di = d[item_nodes]  # each item inherits its node's distance
        eligible = np.isfinite(di) & ~done
        assert eligible[s]
        assert di[s] >= di[eligible].max() - 1e-5, f"step {step}"
        done[s] = True


def test_farthest_seed_prefers_reachable_then_largest_component():
    # A 6-path, a 3-island and a lone node: the path must be exhausted first, then the
    # bigger island, then the singleton.
    edges = np.array(
        [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [6, 7], [7, 8]], dtype=np.uint32
    )
    g = fastcore.GeodesicGraph(edges, 10)
    done = np.zeros(10, dtype=bool)
    done[0] = True
    order = []
    for _ in range(9):
        s = g.farthest_seed(done)
        order.append(s)
        done[s] = True
    assert order[:5] == [5, 2, 1, 3, 4]
    assert order[5:8] == [6, 8, 7]
    assert order[8] == 9
    assert g.farthest_seed(done) is None


def test_farthest_seed_empty_done_starts_on_largest_component():
    edges = np.array([[0, 1], [2, 3], [3, 4], [4, 5]], dtype=np.uint32)
    g = fastcore.GeodesicGraph(edges, 6)
    assert g.farthest_seed(np.zeros(6, dtype=bool)) == 2


def test_farthest_seed_rebuilds_when_done_shrinks():
    n = 120
    edges = random_graph(n_nodes=n, n_edges=500, seed=41)
    rng = np.random.default_rng(41)
    w = rng.random(len(edges)).astype(np.float32)

    g = fastcore.GeodesicGraph(edges, n, weights=w)
    done = np.zeros(n, dtype=bool)
    for _ in range(10):
        done[g.farthest_seed(done)] = True

    shrunk = np.zeros(n, dtype=bool)
    shrunk[7] = True
    fresh = fastcore.GeodesicGraph(edges, n, weights=w)
    assert g.farthest_seed(shrunk) == fresh.farthest_seed(shrunk)


def test_spaced_driver_places_distinct_evenly_spread_seeds():
    # End-to-end `_spaced`: k distinct seeds, and every one at least as far from its
    # nearest neighbour as a random draw would manage.
    faces, verts = grid_mesh(15)
    n = len(verts)
    edges, lengths = fastcore.unique_edges(faces, verts)
    edges = edges.astype(np.uint32)
    g = fastcore.GeodesicGraph(edges, n, weights=lengths.astype(np.float32))

    chosen = np.zeros(n, dtype=bool)
    seeds = []
    while len(seeds) < 20:
        s = g.farthest_seed(chosen)
        assert s is not None and not chosen[s], "seeds must be distinct"
        seeds.append(s)
        chosen[s] = True

    def min_separation(idx):
        """Distance from each chosen node to its nearest other chosen node."""
        d = dijkstra(as_csr(edges, n, lengths), directed=False, indices=idx)[:, idx]
        return np.min(d + np.diag(np.full(len(idx), np.inf)), axis=1)

    sep = min_separation(seeds)
    rng = np.random.default_rng(0)
    worst_random = min(
        min_separation(rng.choice(n, size=len(seeds), replace=False)).min()
        for _ in range(10)
    )
    assert sep.min() > worst_random, "FPS must spread better than a random draw"


def test_item_components():
    edges = np.array([[1, 2], [4, 5]], dtype=np.uint32)
    g = fastcore.GeodesicGraph(edges, 6)
    np.testing.assert_array_equal(g.item_components(), [0, 1, 1, 3, 4, 4])
    # Matches the free function, which callers may already rely on.
    np.testing.assert_array_equal(
        g.item_components(), fastcore.connected_components_graph(edges, 6)
    )
    g = fastcore.GeodesicGraph(edges, 6, item_nodes=[5, 0, 2])
    np.testing.assert_array_equal(g.item_components(), [4, 0, 1])


def test_farthest_seed_validation():
    edges = np.array([[0, 1], [1, 2]], dtype=np.uint32)
    g = fastcore.GeodesicGraph(edges, 3)
    with pytest.raises(ValueError):
        g.farthest_seed(np.zeros(2, dtype=bool))
    assert g.farthest_seed(np.ones(3, dtype=bool)) is None
    # A list and a strided view are both accepted, like `forbidden`.
    assert g.farthest_seed([True, False, False]) == 2


# -----------------------------------------------------------------------------
# GeodesicGraph: the mirrored free functions, and subset
# -----------------------------------------------------------------------------


def test_geodesic_graph_methods_agree_with_the_free_functions():
    # The contract of the whole class: keeping the index changes the cost, never the
    # answer. If these ever diverge, callers cannot safely migrate off the free functions.
    faces, verts = grid_mesh(11)
    n = len(verts)
    edges, lengths = fastcore.unique_edges(faces, verts)
    edges = edges.astype(np.uint32)
    w = lengths.astype(np.float32)
    srcs, tgts = [0, 60, 120], [7, 55, 99]

    for weights in (None, w):
        g = fastcore.GeodesicGraph(edges, n, weights=weights)
        kw = dict(weights=weights)

        np.testing.assert_array_equal(
            g.distances(sources=srcs, targets=tgts, threads=1),
            fastcore.geodesic_matrix_graph(
                edges, n, sources=srcs, targets=tgts, threads=1, **kw
            ),
        )
        np.testing.assert_array_equal(
            g.distances(sources=srcs, limit=3.0, threads=1),
            fastcore.geodesic_matrix_graph(
                edges, n, sources=srcs, limit=3.0, threads=1, **kw
            ),
        )
        for mine, theirs in zip(
            g.predecessors(sources=srcs, threads=1),
            fastcore.geodesic_predecessors(edges, n, sources=srcs, threads=1, **kw),
        ):
            np.testing.assert_array_equal(mine, theirs)
        for a, b in zip(g.path(0, tgts), fastcore.geodesic_path(edges, n, 0, tgts, **kw)):
            np.testing.assert_array_equal(a, b)

        labels, k = g.clusters(2.5, seeds=srcs)
        rl, rk = fastcore.geodesic_clusters(edges, n, 2.5, seeds=srcs, **kw)
        np.testing.assert_array_equal(labels, rl)
        assert k == rk

        np.testing.assert_array_equal(
            g.components(), fastcore.connected_components_graph(edges, n)
        )

    # nearest / farthest have only a mesh-shaped free function, so compare on the mesh.
    g = fastcore.GeodesicGraph(edges, n, weights=w)
    for method, ref in (
        (g.nearest, fastcore.geodesic_nearest_mesh),
        (g.farthest, fastcore.geodesic_farthest_mesh),
    ):
        for mine, theirs in zip(
            method(sources=srcs, targets=tgts, threads=1),
            ref(faces, verts, sources=srcs, targets=tgts, threads=1),
        ):
            np.testing.assert_allclose(mine, theirs, rtol=1e-6)


def test_geodesic_graph_directed():
    # A one-way chain 0->1->2->3.
    edges = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.uint32)
    g = fastcore.GeodesicGraph(edges, 4, directed=True)

    np.testing.assert_array_equal(g.grow(0, 10), [0, 1, 2, 3])
    np.testing.assert_array_equal(g.grow(3, 10), [3])
    np.testing.assert_array_equal(g.distances(sources=[0], threads=1), [[0, 1, 2, 3]])
    np.testing.assert_array_equal(
        g.distances(sources=[3], threads=1), [[-1, -1, -1, 0]]
    )
    # Agrees with the free function's directed mode.
    np.testing.assert_array_equal(
        g.distances(threads=1),
        fastcore.geodesic_matrix_graph(edges, 4, directed=True, threads=1),
    )
    # Components are *weakly* connected, so the chain stays one piece.
    np.testing.assert_array_equal(g.components(), [0, 0, 0, 0])
    # And undirected sees it all both ways.
    u = fastcore.GeodesicGraph(edges, 4)
    np.testing.assert_array_equal(u.grow(3, 10), [3, 2, 1, 0])


def test_subset_matches_a_graph_built_from_the_surviving_edges():
    # Not merely "same distances" but the same graph, down to the neighbour order that
    # decides every tie-break - so a subset is safe to treat as the real thing.
    faces, verts = grid_mesh(9)
    n = len(verts)
    edges, lengths = fastcore.unique_edges(faces, verts)
    edges = edges.astype(np.uint32)
    w = lengths.astype(np.float32)
    g = fastcore.GeodesicGraph(edges, n, weights=w)

    keep = np.array(
        [v for v in range(n) if 2 <= v // 9 <= 6 and 2 <= v % 9 <= 6], dtype=np.uint32
    )
    keep[[0, 7]] = keep[[7, 0]]  # deliberately not ascending
    sub = g.subset(keep)
    np.testing.assert_array_equal(sub.parent_nodes, keep)
    np.testing.assert_array_equal(sub.parent_items, keep)
    assert sub.n_nodes == len(keep)

    # Rebuild the same subgraph the long way, from a filtered edge list.
    new_id = np.full(n, -1, dtype=np.int64)
    new_id[keep] = np.arange(len(keep))
    mask = (new_id[edges[:, 0]] >= 0) & (new_id[edges[:, 1]] >= 0)
    fresh = fastcore.GeodesicGraph(
        new_id[edges[mask]].astype(np.uint32), len(keep), weights=w[mask]
    )

    np.testing.assert_array_equal(
        sub.distances(threads=1), fresh.distances(threads=1)
    )
    np.testing.assert_array_equal(sub.components(), fresh.components())
    np.testing.assert_array_equal(sub.clusters(2.0)[0], fresh.clusters(2.0)[0])
    for seed in (0, 5, 12):
        np.testing.assert_array_equal(sub.grow(seed, 9), fresh.grow(seed, 9))


def test_subset_of_a_component_preserves_parent_distances():
    edges = np.array([[0, 1], [1, 2], [2, 0], [3, 4], [4, 5]], dtype=np.uint32)
    w = np.array([1, 2, 4, 1, 1], dtype=np.float32)
    g = fastcore.GeodesicGraph(edges, 6, weights=w)

    labels = g.components()
    comp = np.flatnonzero(labels == labels[3]).astype(np.uint32)
    sub = g.subset(comp)
    np.testing.assert_array_equal(
        sub.distances(threads=1), g.distances(sources=comp, targets=comp, threads=1)
    )
    # A bool mask is accepted too and means the same thing.
    np.testing.assert_array_equal(
        g.subset(labels == labels[3]).parent_nodes, sub.parent_nodes
    )


def test_subset_carries_items_and_drops_the_orphans():
    edges = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.uint32)
    g = fastcore.GeodesicGraph(edges, 4, item_nodes=[0, 0, 2, 3])
    sub = g.subset([2, 0])
    # Items come out grouped by new node: node 0 (old 2) has item 2, node 1 (old 0) has
    # items 0 and 1. The item on the dropped node 3 is gone.
    np.testing.assert_array_equal(sub.parent_items, [2, 0, 1])
    np.testing.assert_array_equal(sub.item_nodes, [0, 1, 1])
    assert (sub.n_nodes, sub.n_items) == (2, 3)

    # Subsetting a plain graph must keep item i == node i, whatever order `nodes` came in.
    plain = fastcore.GeodesicGraph(edges, 4).subset([3, 1, 0])
    np.testing.assert_array_equal(plain.item_nodes, [0, 1, 2])
    np.testing.assert_array_equal(plain.parent_items, [3, 1, 0])


def test_subset_is_chainable_and_maps_back():
    faces, verts = grid_mesh(8)
    n = len(verts)
    edges, lengths = fastcore.unique_edges(faces, verts)
    g = fastcore.GeodesicGraph(edges.astype(np.uint32), n, weights=lengths.astype(np.float32))

    first = g.subset(np.arange(0, 40, dtype=np.uint32))
    second = first.subset(np.arange(0, 20, dtype=np.uint32))
    # Composing the two maps must land on the same original nodes as subsetting once.
    original = first.parent_nodes[second.parent_nodes]
    np.testing.assert_array_equal(original, np.arange(20))
    direct = g.subset(np.arange(0, 20, dtype=np.uint32))
    np.testing.assert_array_equal(
        second.distances(threads=1), direct.distances(threads=1)
    )


def test_subset_validation():
    g = fastcore.GeodesicGraph(np.array([[0, 1], [1, 2]], dtype=np.uint32), 4)
    with pytest.raises(ValueError):
        g.subset([0, 1, 1])  # repeated node
    with pytest.raises(ValueError):
        g.subset([0, 9])  # out of range
    with pytest.raises(ValueError):
        g.subset(np.zeros(3, dtype=bool))  # mask of the wrong length
    empty = g.subset([])
    assert (empty.n_nodes, empty.n_items) == (0, 0)


def test_subset_reuses_the_parent_index_rather_than_rebuilding():
    # Behavioural proxy for "carved from the parent": a subset of a graph whose edge list
    # we then mutate must be unaffected, i.e. it cannot be re-reading our input.
    edges = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.uint32)
    g = fastcore.GeodesicGraph(edges, 4)
    sub = g.subset([0, 1, 2])
    edges[:] = 0
    np.testing.assert_array_equal(sub.distances(sources=[0], threads=1), [[0, 1, 2]])


# -----------------------------------------------------------------------------
# GeodesicGraph.grow: distances, and the foveation they enable
# -----------------------------------------------------------------------------


def test_grow_distances_match_the_navis_reference():
    # Same oracle as the indices: navis' own `_Geodesic.grow` / `_ConnectedCloud.grow`,
    # which now return distances alongside. Random weights, so nothing ties.
    n = 200
    edges = random_graph(n_nodes=n, n_edges=900, seed=61)
    rng = np.random.default_rng(61)
    w = rng.random(len(edges)).astype(np.float32)
    indptr, indices, data = _csr_parts(edges, n, w)

    g = fastcore.GeodesicGraph(edges, n, weights=w)
    for size in (1, 10, 75, 200):
        for seed in (0, 33, 199):
            idx, dist = g.grow(seed, size, return_distances=True)
            ridx, rdist = _navis_grow(indptr, indices, data, seed, size)
            np.testing.assert_array_equal(idx, ridx)
            np.testing.assert_allclose(dist, rdist, rtol=1e-5, atol=1e-6)

    # Cloud backend: several items per node, many nodes with none.
    item_nodes = np.sort(rng.integers(0, n, size=350)).astype(np.uint32)
    by_vtx = {}
    for i, v in enumerate(item_nodes):
        by_vtx.setdefault(int(v), []).append(i)
    g = fastcore.GeodesicGraph(edges, n, weights=w, item_nodes=item_nodes)
    for size in (1, 20, 128, 350):
        for seed in (0, 111, 349):
            idx, dist = g.grow(seed, size, return_distances=True)
            ridx, rdist = _navis_grow_cloud(
                indptr, indices, data, by_vtx, item_nodes, seed, size
            )
            np.testing.assert_array_equal(idx, ridx)
            np.testing.assert_allclose(dist, rdist, rtol=1e-5, atol=1e-6)


def test_grow_distances_agree_with_scipy_and_are_sorted():
    n = 200
    edges = random_graph(n_nodes=n, n_edges=800, seed=67)
    rng = np.random.default_rng(67)
    w = rng.random(len(edges)).astype(np.float32)
    g = fastcore.GeodesicGraph(edges, n, weights=w)

    idx, dist = g.grow(9, 60, return_distances=True)
    ref = dijkstra(as_csr(edges, n, w), directed=False, indices=[9])[0]
    np.testing.assert_allclose(dist, ref[idx], rtol=1e-5, atol=1e-6)
    assert np.all(np.diff(dist) >= 0), "distances are non-decreasing"
    assert dist[0] == 0.0


def test_grow_distances_are_shared_within_a_node():
    # An item's position is its node's, so items on one node must share a distance
    # *exactly* - a radial thinning keyed on these would drift otherwise.
    edges = np.array([[i, i + 1] for i in range(3)], dtype=np.uint32)
    g = fastcore.GeodesicGraph(edges, 4, item_nodes=[0, 1, 1, 1, 3])
    idx, dist = g.grow(0, 5, return_distances=True)
    np.testing.assert_array_equal(idx, [0, 1, 2, 3, 4])
    np.testing.assert_array_equal(dist, [0, 1, 1, 1, 3])
    assert len(np.unique(dist[1:4])) == 1


def test_grow_distances_stay_in_lockstep_with_indices():
    edges = np.array([[i, i + 1] for i in range(3)], dtype=np.uint32)
    g = fastcore.GeodesicGraph(edges, 4, item_nodes=[0, 1, 1, 1])

    # Budget fills mid-node.
    idx, dist = g.grow(0, 3, return_distances=True)
    assert len(idx) == len(dist) == 3
    # Growth stops at a wall.
    idx, dist = g.grow(
        0, 4, forbidden=np.array([False, True, True, True]), return_distances=True
    )
    np.testing.assert_array_equal(idx, [0])
    np.testing.assert_array_equal(dist, [0.0])
    # Zero budget gives two empty arrays, and the default still gives a bare one.
    idx, dist = g.grow(0, 0, return_distances=True)
    assert len(idx) == 0 and len(dist) == 0
    assert isinstance(g.grow(0, 3), np.ndarray)


def test_grow_supports_navis_foveation():
    """End-to-end: the `fovea` pipeline driven off fastcore agrees with navis'.

    `_Foveated.grow` grows an oversized candidate pool and thins it radially back to
    `n_points`. Everything after the grow is pure numpy; what it needs from the backend
    is the ``(indices, distances)`` pair, so this pins that the two agree.
    """
    n = 300
    edges = random_graph(n_nodes=n, n_edges=1400, seed=71)
    rng = np.random.default_rng(71)
    w = rng.random(len(edges)).astype(np.float32)
    indptr, indices, data = _csr_parts(edges, n, w)
    item_nodes = np.sort(rng.integers(0, n, size=600)).astype(np.uint32)
    by_vtx = {}
    for i, v in enumerate(item_nodes):
        by_vtx.setdefault(int(v), []).append(i)
    g = fastcore.GeodesicGraph(edges, n, weights=w, item_nodes=item_nodes)

    n_points, reach, fovea = 32, 8, 4
    for falloff in (None, 2.0):
        for seed in (0, 250, 599):
            mine = g.grow(seed, reach * n_points, return_distances=True)
            theirs = _navis_grow_cloud(
                indptr, indices, data, by_vtx, item_nodes, seed, reach * n_points
            )
            # Thin both with the same draw; identical inputs must give identical patches.
            sel_a, focus_a = _radial_thin(
                len(mine[0]), n_points, fovea, falloff, mine[1],
                np.random.default_rng(0),
            )
            sel_b, focus_b = _radial_thin(
                len(theirs[0]), n_points, fovea, falloff, theirs[1],
                np.random.default_rng(0),
            )
            np.testing.assert_array_equal(mine[0][sel_a], theirs[0][sel_b])
            np.testing.assert_allclose(focus_a, focus_b)

            # And the patch has the shape foveation promises: full-density core, then
            # a thinned halo reaching well beyond a uniform patch of the same budget.
            assert len(sel_a) == n_points
            # `_focus` is a central difference, so the *last* fovea point already
            # straddles the transition into the halo; everything inside it is at 1.0.
            assert np.all(focus_a[: fovea - 1] == 1.0), "the fovea is at full density"
            assert focus_a[-1] < 1.0, "the periphery is thinned"
            uniform = g.grow(seed, n_points, return_distances=True)[1]
            assert mine[1][sel_a].max() >= uniform.max(), "the halo reaches further"


def _radial_thin(m, size, fovea, falloff, dist, rng):
    """Verbatim port of `navis.ml.chunk._radial_thin`."""
    if m <= size:
        sel = np.arange(m)
        return sel, _focus(sel)
    k = min(int(fovea), size)
    n = size - k
    if n == 0:
        sel = np.arange(k)
        return sel, _focus(sel)
    u = (np.arange(n) + rng.random(n)) / n
    if falloff is None:
        sel = k + np.round((m - k) ** u).astype(np.int64) - 1
    else:
        r = dist[k:]
        pos = r[r > 0]
        r0 = float(r[0]) if r[0] > 0 else (float(pos[0]) if len(pos) else 1.0)
        weight = (r0**2 + r**2) ** (-falloff / 2)
        cum = np.cumsum(weight)
        sel = k + np.searchsorted(cum, u * cum[-1], side="right")
    sel = np.concatenate([np.arange(k), _spread(np.clip(sel, k, m - 1), k, m)])
    return sel, _focus(sel)


def _focus(sel):
    """Verbatim port of `navis.ml.chunk._focus`."""
    if len(sel) < 2:
        return np.ones(len(sel), dtype=float)
    return 1.0 / np.gradient(sel.astype(float))


def _spread(sel, lo, hi):
    """Verbatim port of `navis.ml.chunk._spread`."""
    i = np.arange(len(sel))
    out = np.maximum.accumulate(sel - i) + i
    return np.minimum(out, hi - len(sel) + i)


# -----------------------------------------------------------------------------
# GeodesicGraph.ball
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("seed", [0, 7, 61])
def test_ball_matches_scipy_min_only(weighted, seed):
    """The oracle is `dijkstra(..., min_only=True, limit=...)`, which asks exactly this."""
    n = 200
    edges = random_graph(n_nodes=n, n_edges=600, seed=seed)
    rng = np.random.default_rng(seed)
    w = rng.random(len(edges)).astype(np.float32) if weighted else None
    g = fastcore.GeodesicGraph(edges, n, weights=w)
    m = as_csr(edges, n, w)

    sources = rng.choice(n, size=4, replace=False).astype(np.uint32)
    limit = 1.5 if weighted else 3.0
    nodes, dist, src = g.ball(sources, limit)

    ref = dijkstra(m, directed=False, indices=sources, limit=limit, min_only=True)
    assert set(nodes.tolist()) == set(np.flatnonzero(ref <= limit).tolist())
    np.testing.assert_allclose(dist, ref[nodes], atol=1e-5)

    # Every reported source must be *a* nearest one - which need not be scipy's, since
    # equidistant sources are a tie either way
    per_source = g.distances(sources=sources, threads=1)
    per_source[per_source < 0] = np.inf
    for v, s, d in zip(nodes, src, dist):
        assert s in sources
        assert per_source[:, v].min() == pytest.approx(d, abs=1e-5)
        assert per_source[list(sources).index(s), v] == pytest.approx(d, abs=1e-5)


def test_ball_returns_increasing_distances_and_includes_its_sources():
    edges = np.array([[i, i + 1] for i in range(6)], dtype=np.uint32)
    g = fastcore.GeodesicGraph(edges, 7)

    # Two sources at opposite ends. The contract is the *set*, its distances and its
    # attribution, plus that distances come back non-decreasing - not how two equidistant
    # frontiers interleave, which the docs call arbitrary. Pinning that would turn a
    # legitimate change of frontier order into a test failure.
    nodes, dist, src = g.ball([0, 6], 1)
    assert (np.diff(dist) >= 0).all()
    order = np.argsort(nodes)
    np.testing.assert_array_equal(nodes[order], [0, 1, 5, 6])
    np.testing.assert_array_equal(dist[order], [0, 1, 1, 0])
    np.testing.assert_array_equal(src[order], [0, 0, 6, 6])

    # Unbounded: every reachable node, nearest source and all
    nodes, dist, src = g.ball([0])
    np.testing.assert_array_equal(nodes, np.arange(7))
    np.testing.assert_array_equal(dist, np.arange(7))
    np.testing.assert_array_equal(src, np.zeros(7))


def test_ball_skips_unreachable_and_survives_repeat_sources():
    # Two components; a source in one says nothing about the other
    edges = np.array([[0, 1], [2, 3]], dtype=np.uint32)
    g = fastcore.GeodesicGraph(edges, 5)
    nodes, _, _ = g.ball([0])
    np.testing.assert_array_equal(np.sort(nodes), [0, 1])

    # A repeated source is one source, not two frontier entries
    np.testing.assert_array_equal(g.ball([0, 0, 0])[0], g.ball([0])[0])

    assert len(g.ball([])[0]) == 0
    assert len(g.ball([4])[0]) == 1  # isolated node: only itself


def test_ball_is_reusable_and_matches_a_fresh_graph():
    """Scratch is shared between calls, so a stale one would show up as drift."""
    n = 120
    edges = random_graph(n_nodes=n, n_edges=400, seed=3)
    rng = np.random.default_rng(3)
    w = rng.random(len(edges)).astype(np.float32)
    g = fastcore.GeodesicGraph(edges, n, weights=w)

    first = g.ball([0, 1], 0.8)
    for _ in range(5):
        g.ball(rng.choice(n, size=3).astype(np.uint32), 2.0)
        g.grow(int(rng.integers(n)), 10)
    again = g.ball([0, 1], 0.8)
    for a, b in zip(first, again):
        np.testing.assert_array_equal(a, b)

    fresh = fastcore.GeodesicGraph(edges, n, weights=w).ball([0, 1], 0.8)
    for a, b in zip(first, fresh):
        np.testing.assert_array_equal(a, b)


def test_ball_rejects_bad_arguments():
    g = fastcore.GeodesicGraph(np.array([[0, 1]], dtype=np.uint32), 2)
    with pytest.raises(ValueError, match="sources"):
        g.ball([2])
    with pytest.raises(ValueError, match="max_dist"):
        g.ball([0], -1)


# -----------------------------------------------------------------------------
# GeodesicGraph.set_weights
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 7, 61])
def test_set_weights_matches_a_rebuilt_graph(seed):
    """An edited graph must be indistinguishable from one built with the new weights."""
    n = 150
    edges = random_graph(n_nodes=n, n_edges=500, seed=seed)
    rng = np.random.default_rng(seed)
    w = rng.random(len(edges)).astype(np.float32)
    g = fastcore.GeodesicGraph(edges, n, weights=w)

    pick = rng.choice(len(edges), size=len(edges) // 3, replace=False)
    new = rng.random(len(pick)).astype(np.float32)
    g.set_weights(edges[pick], new)

    w2 = w.copy()
    w2[pick] = new
    ref = fastcore.GeodesicGraph(edges, n, weights=w2)
    np.testing.assert_allclose(g.distances(threads=1), ref.distances(threads=1), atol=1e-5)


def test_set_weights_keeps_an_undirected_graph_symmetric():
    """Both arcs of an undirected edge must move, whichever way round it is given."""
    edges = np.array([[0, 1], [1, 2], [0, 2]], dtype=np.uint32)
    g = fastcore.GeodesicGraph(edges, 3, weights=[1.0, 1.0, 5.0])
    assert g.distances(sources=[0], targets=[2], threads=1)[0, 0] == 2.0

    g.set_weights([[2, 0]], [0.5])  # reversed pair
    d = g.distances(threads=1)
    np.testing.assert_allclose(d, d.T)
    assert d[0, 2] == pytest.approx(0.5)


def test_set_weights_broadcasts_a_scalar_and_takes_repeats():
    edges = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.uint32)
    g = fastcore.GeodesicGraph(edges, 4, weights=[1.0, 1.0, 1.0])

    g.set_weights(edges, 0.0)  # the shape a caller zeroing a path writes
    assert g.distances(sources=[0], targets=[3], threads=1)[0, 0] == 0.0

    # Last write wins on a repeat
    g.set_weights([[0, 1], [0, 1]], [9.0, 2.0])
    assert g.distances(sources=[0], targets=[1], threads=1)[0, 0] == 2.0


def test_set_weights_rejects_what_it_cannot_do():
    g = fastcore.GeodesicGraph(np.array([[0, 1]], dtype=np.uint32), 3, weights=[1.0])
    with pytest.raises(ValueError, match="no edge 0 - 2"):
        g.set_weights([[0, 2]], [1.0])  # not an edge: this cannot add one

    unweighted = fastcore.GeodesicGraph(np.array([[0, 1]], dtype=np.uint32), 3)
    with pytest.raises(ValueError, match="weights=None"):
        unweighted.set_weights([[0, 1]], [1.0])

    with pytest.raises(ValueError, match="one entry per edge"):
        g.set_weights([[0, 1]], [1.0, 2.0])


def test_set_weights_on_a_subset_edits_only_the_subset():
    """`subset` carves out its own adjacency, so its weights are its own."""
    edges = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.uint32)
    g = fastcore.GeodesicGraph(edges, 4, weights=[1.0, 1.0, 1.0])
    sub = g.subset([1, 2, 3])  # new node i is old node parent_nodes[i]

    sub.set_weights([[0, 1]], [7.0])  # sub's (0, 1) is the parent's (1, 2)
    assert sub.distances(sources=[0], targets=[1], threads=1)[0, 0] == 7.0
    assert g.distances(sources=[1], targets=[2], threads=1)[0, 0] == 1.0


def test_set_weights_directed_leaves_the_reverse_arc_alone():
    edges = np.array([[0, 1], [1, 0]], dtype=np.uint32)
    g = fastcore.GeodesicGraph(edges, 2, weights=[1.0, 1.0], directed=True)

    g.set_weights([[0, 1]], [5.0])
    assert g.distances(sources=[0], targets=[1], threads=1)[0, 0] == 5.0
    assert g.distances(sources=[1], targets=[0], threads=1)[0, 0] == 1.0


def test_set_weights_restarts_farthest_seed_rather_than_lying():
    """The incremental FPS field is a minimum under the old weights - it cannot be kept."""
    edges = np.array([[i, i + 1] for i in range(8)], dtype=np.uint32)
    w = np.ones(8, dtype=np.float32)
    g = fastcore.GeodesicGraph(edges, 9, weights=w)

    done = np.zeros(9, dtype=bool)
    done[0] = True
    assert g.farthest_seed(done) == 8

    # Make the far end cheap to reach and the near end expensive; the answer must move
    w2 = w.copy()
    w2[4:] = 0.0
    g.set_weights(edges[4:], 0.0)
    assert g.farthest_seed(done) == fastcore.GeodesicGraph(
        edges, 9, weights=w2
    ).farthest_seed(done)
