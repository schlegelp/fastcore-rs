"""Tests for the caps module — finding and closing the holes in a mesh.

The oracle for the boundary search is a plain Python dict counting faces per
undirected edge: `O(F)` and obviously correct, which is exactly what the fast
path is not. For the capping itself the oracles are geometric — a cap has to
close the hole, use every ring vertex and wind against the ring — because there
is no single right triangulation to compare against.
"""

from collections import defaultdict

import numpy as np
import pytest

import navis_fastcore as fastcore
from meshes import grid_mesh, uv_sphere


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def dict_boundary(faces):
    """Reference implementation: count faces per undirected edge, keep the ones."""
    seen = defaultdict(list)
    for f in np.asarray(faces).tolist():
        for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            seen[(min(a, b), max(a, b))].append((a, b))
    return [v[0] for v in seen.values() if len(v) == 1]


def as_set(halfedges):
    return set(map(tuple, np.asarray(halfedges, dtype=np.int64).tolist()))


def as_rings(rings, offsets):
    return [rings[offsets[i] : offsets[i + 1]] for i in range(len(offsets) - 1)]


def face_normal(tri, vertices):
    a, b, c = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]
    return np.cross(b - a, c - a)


def punched(faces, frac, seed=0):
    """Drop a fraction of the faces at random, opening holes all over the mesh."""
    rng = np.random.default_rng(seed)
    return faces[rng.random(len(faces)) > frac]


# -----------------------------------------------------------------------------
# boundary_halfedges
# -----------------------------------------------------------------------------


def test_closed_mesh_has_no_boundary():
    # A UV sphere built without poles is open at top and bottom; sealing those
    # rings by hand would just be the code under test. Use a tetrahedron, which
    # is closed by construction.
    tetra = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.uint32)
    assert len(fastcore.boundary_halfedges(tetra)) == 0


def test_grid_boundary_is_its_outer_edge():
    n = 6
    faces, _ = grid_mesh(n)
    b = fastcore.boundary_halfedges(faces)

    assert len(b) == 4 * (n - 1)
    for v in np.unique(b):
        i, j = divmod(int(v), n)
        assert 0 in (i, j) or n - 1 in (i, j), f"interior vertex {v} on the boundary"


@pytest.mark.parametrize("frac", [0.0, 0.02, 0.2, 0.6])
def test_matches_dict_oracle(frac):
    faces, _ = uv_sphere(16, 16)
    faces = punched(faces, frac) if frac else faces

    got = as_set(fastcore.boundary_halfedges(faces))
    assert got == as_set(dict_boundary(faces))
    # Only the half-edge the *surviving* face winds is reported, never its reverse
    # as well. (A self-loop off a degenerate face is its own reverse, so exempt.)
    assert not any((b, a) in got for a, b in got if a != b)


def test_boundary_is_in_halfedge_order():
    """Rows come in `3F` edge-list order — what `trace_loops` greedily depends on."""
    faces, _ = grid_mesh(5)
    b = fastcore.boundary_halfedges(faces)

    order = {}
    for i, f in enumerate(faces.tolist()):
        for e, (a, b_) in enumerate(((f[0], f[1]), (f[1], f[2]), (f[2], f[0]))):
            order.setdefault((a, b_), i * 3 + e)
    positions = [order[(int(a), int(c))] for a, c in b]
    assert positions == sorted(positions)


def test_threads_do_not_change_the_answer():
    faces = punched(uv_sphere(20, 20)[0], 0.1)
    ref = fastcore.boundary_halfedges(faces)
    for t in (1, 2, 4):
        assert np.array_equal(fastcore.boundary_halfedges(faces, threads=t), ref)


# -----------------------------------------------------------------------------
# exposed_halfedges
# -----------------------------------------------------------------------------


def test_dropping_one_corner_exposes_the_opposite_edge():
    # Two triangles sharing (1, 2). Dropping vertex 0 kills the first face and
    # leaves (1, 2) with only the second on it, wound the way that one winds it.
    faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.uint32)

    e = fastcore.exposed_halfedges(faces, np.array([True, False, False, False]))
    assert e.tolist() == [[2, 1]]

    e = fastcore.exposed_halfedges(faces, np.array([False, False, False, True]))
    assert e.tolist() == [[1, 2]]


def test_pre_existing_boundary_is_not_exposed():
    """A hole the mesh came with is not this function's business."""
    faces, _ = grid_mesh(6)
    # Drop a vertex right on the grid's outer edge. The outer edge stays boundary
    # but was boundary already, so only the newly opened edges are reported.
    dropped = np.zeros(36, dtype=bool)
    dropped[2] = True

    exposed = as_set(fastcore.exposed_halfedges(faces, dropped))
    was_boundary = {
        (min(a, b), max(a, b)) for a, b in as_set(fastcore.boundary_halfedges(faces))
    }
    assert exposed
    assert not any((min(a, b), max(a, b)) in was_boundary for a, b in exposed)


@pytest.mark.parametrize("n_dropped", [1, 5, 40])
def test_exposed_is_the_new_part_of_the_boundary(n_dropped):
    """What a cut exposes = the subset's boundary minus what was boundary before."""
    faces, _ = uv_sphere(14, 14)
    n_vertices = 14 * 14
    rng = np.random.default_rng(n_dropped)
    dropped = np.zeros(n_vertices, dtype=bool)
    dropped[rng.choice(n_vertices, n_dropped, replace=False)] = True

    kept = faces[~dropped[faces].any(axis=1)]
    before = as_set(fastcore.boundary_halfedges(faces))
    after = as_set(fastcore.boundary_halfedges(kept))

    assert as_set(fastcore.exposed_halfedges(faces, dropped)) == after - before


def test_dropping_everything_or_nothing_exposes_nothing():
    faces, _ = uv_sphere(10, 10)
    n = 100
    assert len(fastcore.exposed_halfedges(faces, np.zeros(n, dtype=bool))) == 0
    assert len(fastcore.exposed_halfedges(faces, np.ones(n, dtype=bool))) == 0


def test_exposed_threads_do_not_change_the_answer():
    faces, _ = uv_sphere(16, 16)
    rng = np.random.default_rng(3)
    dropped = rng.random(256) < 0.2
    ref = fastcore.exposed_halfedges(faces, dropped)
    for t in (1, 2, 4):
        assert np.array_equal(fastcore.exposed_halfedges(faces, dropped, threads=t), ref)


# -----------------------------------------------------------------------------
# trace_loops
# -----------------------------------------------------------------------------


def test_grid_boundary_traces_into_one_ring():
    n = 5
    faces, _ = grid_mesh(n)
    rings, offsets = fastcore.trace_loops(fastcore.boundary_halfedges(faces))

    assert len(offsets) == 2
    assert len(rings) == 4 * (n - 1)
    assert len(set(rings.tolist())) == len(rings), "no vertex twice in one ring"


def test_every_ring_edge_is_a_real_halfedge():
    faces = punched(uv_sphere(16, 16)[0], 0.1)
    he = fastcore.boundary_halfedges(faces)
    rings, offsets = fastcore.trace_loops(he)

    available = as_set(he)
    used = []
    for r in as_rings(rings, offsets):
        assert len(r) >= 3
        used += [(int(r[i]), int(r[(i + 1) % len(r)])) for i in range(len(r))]

    assert set(used) <= available
    assert len(used) == len(set(used)), "a half-edge landed in two rings"


def test_covers_more_than_a_cycle_basis():
    """Two triangles meeting at one vertex: both rings, not one."""
    he = np.array([[0, 1], [1, 2], [2, 0], [0, 3], [3, 4], [4, 0]], dtype=np.uint32)
    rings, offsets = fastcore.trace_loops(he)

    assert len(offsets) == 3
    assert len(rings) == 6


def test_dead_ends_are_dropped_not_hung_on():
    he = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.uint32)
    rings, offsets = fastcore.trace_loops(he)

    assert len(rings) == 0
    assert offsets.tolist() == [0]


def test_two_vertex_ring_is_dropped():
    """A ring needs three vertices to be a hole; a doubled-back pair is not one."""
    he = np.array([[0, 1], [1, 0]], dtype=np.uint32)
    rings, offsets = fastcore.trace_loops(he)

    assert len(rings) == 0
    assert offsets.tolist() == [0]


# -----------------------------------------------------------------------------
# triangulate_rings
# -----------------------------------------------------------------------------


def test_square_hole_caps_against_its_ring():
    # Counter-clockwise seen from +z, so the cap must point at -z.
    vertices = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float64
    )
    caps = fastcore.triangulate_rings(np.arange(4), [0, 4], vertices)

    assert len(caps) == 2
    for tri in caps:
        assert face_normal(tri, vertices)[2] < 0

    # Reverse the ring and the cap follows it round.
    caps = fastcore.triangulate_rings(np.arange(4)[::-1].copy(), [0, 4], vertices)
    for tri in caps:
        assert face_normal(tri, vertices)[2] > 0


def test_non_convex_ring_stays_inside_itself():
    """An L-shape: a fan would put triangles outside the polygon, ear-clipping does not."""
    ring = np.array(
        [[0, 0], [3, 0], [3, 1], [1, 1], [1, 3], [0, 3]], dtype=np.float64
    )
    vertices = np.column_stack([ring, np.zeros(len(ring))])
    caps = fastcore.triangulate_rings(np.arange(len(ring)), [0, len(ring)], vertices)

    assert len(caps) == len(ring) - 2
    # The L has area 5; a correct triangulation covers exactly that.
    area = sum(abs(np.linalg.norm(face_normal(t, vertices))) / 2 for t in caps)
    assert area == pytest.approx(5.0)


def test_cap_closes_what_it_covers():
    n = 7
    faces, vertices = grid_mesh(n)
    rings, offsets = fastcore.trace_loops(fastcore.boundary_halfedges(faces))
    caps = fastcore.triangulate_rings(rings, offsets, vertices)

    assert len(caps) == sum(len(r) - 2 for r in as_rings(rings, offsets))
    assert len(fastcore.boundary_halfedges(np.vstack((faces, caps)))) == 0


def test_many_holes_all_get_closed():
    faces, vertices = uv_sphere(20, 20)
    faces = punched(faces, 0.1, seed=5)

    he = fastcore.boundary_halfedges(faces)
    rings, offsets = fastcore.trace_loops(he)
    caps = fastcore.triangulate_rings(rings, offsets, vertices)

    ring_list = as_rings(rings, offsets)
    assert len(ring_list) > 10, "the fixture should have plenty of holes"
    assert len(caps) == sum(len(r) - 2 for r in ring_list)

    # Each ring's cap uses that ring's vertices and no others — the property the
    # global containment check below cannot see.
    at = 0
    for r in ring_list:
        cap = caps[at : at + len(r) - 2]
        at += len(r) - 2
        assert set(cap.ravel().tolist()) == set(r.tolist())
    assert at == len(caps)


def test_ring_through_the_same_vertex_twice_still_closes():
    """Greedy tracing can name a non-manifold vertex twice in one ring.

    The polygon then touches itself, so neither ear-clipping attempt can find
    ``n - 2`` ears and the fan has to take over. These are the real coordinates of
    one such ring off a punched neuron mesh, kept because they are also the input
    that sends ``mapbox_earcut`` — what navis reached for before this module
    existed — into an infinite loop on its best-fit-plane retry.
    """
    vertices = np.array(
        [
            [5571.95996094, 22467.96875, 16704.00390625],
            [5618.43554688, 22463.57617188, 16698.08398438],
            [5519.92675781, 22390.08789062, 16695.66992188],
            [5576.05859375, 22375.62695312, 16725.99609375],
            [5618.43554688, 22463.57617188, 16698.08398438],  # == row 1
            [5611.96044922, 22447.96679688, 16756.00585938],
        ]
    )
    caps = fastcore.triangulate_rings(np.arange(6), [0, 6], vertices)

    assert len(caps) == 4
    assert set(caps.ravel().tolist()) == set(range(6))


def test_degenerate_ring_still_closes():
    """Collinear vertices name no plane at all — the fan is the last resort."""
    vertices = np.array(
        [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=np.float64
    )
    caps = fastcore.triangulate_rings(np.arange(4), [0, 4], vertices)

    assert len(caps) == 2
    assert set(caps.ravel().tolist()) == {0, 1, 2, 3}


def test_non_planar_ring_still_closes():
    """A ring that self-intersects when flattened: falls through to the retries."""
    vertices = np.array(
        [[0, 0, 0], [1, 0, 3], [2, 0, 0], [2, 2, 3], [1, 1, 0], [0, 2, 3]],
        dtype=np.float64,
    )
    caps = fastcore.triangulate_rings(np.arange(6), [0, 6], vertices)

    assert len(caps) == 4
    assert set(caps.ravel().tolist()) == set(range(6))


def test_triangle_ring_is_one_face():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    caps = fastcore.triangulate_rings(np.arange(3), [0, 3], vertices)

    assert len(caps) == 1
    assert face_normal(caps[0], vertices)[2] < 0


def test_rings_are_capped_independently():
    """Two rings in one call give the same faces as two calls of one ring each."""
    faces, vertices = uv_sphere(14, 14)
    rings, offsets = fastcore.trace_loops(fastcore.boundary_halfedges(faces))
    assert len(offsets) - 1 == 2, "an open UV sphere has a ring at each pole"

    together = fastcore.triangulate_rings(rings, offsets, vertices)
    apart = np.vstack(
        [
            fastcore.triangulate_rings(
                rings[offsets[i] : offsets[i + 1]],
                [0, offsets[i + 1] - offsets[i]],
                vertices,
            )
            for i in range(2)
        ]
    )
    assert np.array_equal(together, apart)


def test_triangulate_threads_do_not_change_the_answer():
    faces, vertices = uv_sphere(18, 18)
    faces = punched(faces, 0.15, seed=2)
    rings, offsets = fastcore.trace_loops(fastcore.boundary_halfedges(faces))

    ref = fastcore.triangulate_rings(rings, offsets, vertices)
    for t in (1, 2, 4):
        assert np.array_equal(
            fastcore.triangulate_rings(rings, offsets, vertices, threads=t), ref
        )


# -----------------------------------------------------------------------------
# End to end
# -----------------------------------------------------------------------------


def test_capping_a_cut_closes_exactly_the_cut():
    """The `subset_neuron` path: find before cutting, remap, cap after."""
    faces, vertices = uv_sphere(16, 16)
    n_vertices = len(vertices)

    # Away from the sphere's own two openings, so every exposed edge belongs to a
    # ring that closes — see `test_a_cut_reaching_an_existing_hole_leaves_a_chain`.
    rng = np.random.default_rng(11)
    interior = np.arange(4 * 16, 12 * 16)
    dropped = np.zeros(n_vertices, dtype=bool)
    dropped[rng.choice(interior, 30, replace=False)] = True

    exposed = fastcore.exposed_halfedges(faces, dropped)

    # Subset: keep the faces whose corners all survive, renumber what is left.
    kept = np.flatnonzero(~dropped)
    renumber = np.full(n_vertices, -1, dtype=np.int64)
    renumber[kept] = np.arange(len(kept))
    sub_faces = renumber[faces[~dropped[faces].any(axis=1)]].astype(np.uint32)
    sub_vertices = vertices[kept]
    sub_exposed = renumber[exposed].astype(np.uint32)

    before = as_set(fastcore.boundary_halfedges(sub_faces))
    rings, offsets = fastcore.trace_loops(sub_exposed)
    assert len(rings) == len(sub_exposed), "every exposed edge traced into a ring"
    caps = fastcore.triangulate_rings(rings, offsets, sub_vertices)

    # Exactly the edges the cut exposed are closed; the sphere's own openings, which
    # the cut did not make, are left standing.
    after = as_set(fastcore.boundary_halfedges(np.vstack((sub_faces, caps))))
    assert after == before - as_set(sub_exposed)
    assert after, "the sphere's poles are still open"


def test_a_cut_reaching_an_existing_hole_leaves_a_chain():
    """A cut running into an opening the mesh came with does not close on itself.

    `exposed_halfedges` deliberately leaves out edges that were boundary already,
    so what it reports there is an open chain rather than a ring, and
    `trace_loops` abandons it. That is the intended division of labour: capping
    an opening the mesh came with is `boundary_halfedges`' job, not this one's.
    """
    n = 6
    faces, _ = grid_mesh(n)  # the grid's outer edge is boundary from the start
    dropped = np.zeros(n * n, dtype=bool)
    dropped[[1, 2]] = True  # right on that outer edge

    exposed = fastcore.exposed_halfedges(faces, dropped)
    assert len(exposed), "the cut does expose interior edges"

    rings, offsets = fastcore.trace_loops(exposed)
    assert len(rings) == 0, "an open chain is not a ring"


def test_filling_every_hole_leaves_a_closed_mesh():
    """The `fill_holes` path: boundary of the whole mesh, then cap it."""
    faces, vertices = uv_sphere(18, 18)
    faces = punched(faces, 0.08, seed=9)

    he = fastcore.boundary_halfedges(faces)
    rings, offsets = fastcore.trace_loops(he)
    caps = fastcore.triangulate_rings(rings, offsets, vertices)

    filled = np.vstack((faces, caps))
    assert len(fastcore.boundary_halfedges(filled)) == 0
    assert filled.shape[1] == 3
    # Only faces were added — no vertex moved and none appeared.
    assert caps.max() < len(vertices)


# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------


def test_empty_inputs():
    empty_f = np.empty((0, 3), dtype=np.uint32)
    empty_v = np.empty((0, 3), dtype=np.float64)

    assert fastcore.boundary_halfedges(empty_f).shape == (0, 2)
    assert fastcore.exposed_halfedges(empty_f, np.empty(0, dtype=bool)).shape == (0, 2)

    rings, offsets = fastcore.trace_loops(np.empty((0, 2), dtype=np.uint32))
    assert len(rings) == 0 and offsets.tolist() == [0]
    assert fastcore.triangulate_rings(rings, offsets, empty_v).shape == (0, 3)


def test_dtypes():
    faces, vertices = grid_mesh(4)
    assert fastcore.boundary_halfedges(faces).dtype == np.uint32
    assert (
        fastcore.exposed_halfedges(faces, np.zeros(16, dtype=bool)).dtype == np.uint32
    )
    rings, offsets = fastcore.trace_loops(fastcore.boundary_halfedges(faces))
    assert rings.dtype == np.uint32
    assert offsets.dtype == np.int64
    assert fastcore.triangulate_rings(rings, offsets, vertices).dtype == np.uint32


def test_accepts_other_integer_dtypes():
    """navis carries faces as int64; the wrapper casts rather than refusing."""
    faces, vertices = grid_mesh(5)
    ref = fastcore.boundary_halfedges(faces)
    assert np.array_equal(fastcore.boundary_halfedges(faces.astype(np.int64)), ref)
    assert np.array_equal(fastcore.boundary_halfedges(faces.astype(np.int32)), ref)


def test_bad_shapes_raise():
    faces, vertices = grid_mesh(4)

    with pytest.raises(ValueError, match="shape"):
        fastcore.boundary_halfedges(np.zeros((4, 2), dtype=np.uint32))
    with pytest.raises(ValueError, match="shape"):
        fastcore.trace_loops(np.zeros((4, 3), dtype=np.uint32))
    with pytest.raises(ValueError, match="1-D"):
        fastcore.exposed_halfedges(faces, np.zeros((4, 4), dtype=bool))
    with pytest.raises(ValueError, match="shape"):
        fastcore.triangulate_rings([0, 1, 2], [0, 3], np.zeros((3, 2)))


def test_out_of_range_indices_raise():
    faces, vertices = grid_mesh(4)

    with pytest.raises(ValueError, match="only covers"):
        fastcore.exposed_halfedges(faces, np.zeros(4, dtype=bool))
    with pytest.raises(ValueError, match="only"):
        fastcore.triangulate_rings([0, 1, 99], [0, 3], vertices)


def test_malformed_offsets_raise():
    vertices = np.zeros((4, 3))

    with pytest.raises(ValueError, match="at least one entry"):
        fastcore.triangulate_rings([0, 1, 2], [], vertices)
    with pytest.raises(ValueError, match="must run from 0"):
        fastcore.triangulate_rings([0, 1, 2], [0, 2], vertices)
    with pytest.raises(ValueError, match="must run from 0"):
        fastcore.triangulate_rings([0, 1, 2], [1, 3], vertices)
    with pytest.raises(ValueError, match="non-decreasing"):
        fastcore.triangulate_rings([0, 1, 2], [0, 2, 1, 3], vertices)
