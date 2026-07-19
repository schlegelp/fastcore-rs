"""Tests for the mesh module.

The oracle throughout is `scipy.sparse.csgraph.dijkstra`, which is what navis currently
uses for meshes (via a trimesh -> igraph -> scipy sparse detour).
"""

import numpy as np
import pytest

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

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
