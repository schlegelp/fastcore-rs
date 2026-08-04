"""Tests for mesh simplification.

Two kinds of test here, and the split matters.

The **oracle** tests compare against `pyfqmr`, which wraps the same MIT-licensed
`Simplify.h` this is a port of. Both carry positions at float64 (upstream's `vec3f`
is `double x, y, z` despite the name), so the two implementations agree to the last
few bits: on a clean mesh the face arrays come out *identical* and the positions
within ~1e-12. That is a much stronger check than "roughly the same decimation",
and it is what makes a hand-transcribed port of 500 lines of C++ index arithmetic
trustworthy. `pyfqmr` is a test-only dependency; those tests skip without it.

The **invariant** tests need no oracle and cover what `pyfqmr` cannot check,
because it does not have the feature: the vertex map, vertex pinning, and the
degenerate and non-manifold inputs that neuron meshes are actually made of.
"""

import numpy as np
import pytest

import navis_fastcore as fastcore
from meshes import check_simplify_invariants, grid_mesh, uv_sphere


# -----------------------------------------------------------------------------
# Against pyfqmr
# -----------------------------------------------------------------------------
#
# Marked rather than `importorskip`-ed at module scope: that would take the whole
# file down with it, and everything below this section is an invariant that needs no
# oracle — which is exactly the half worth still running on a machine without pyfqmr.

try:
    import pyfqmr
except ImportError:  # pragma: no cover - depends on the environment
    pyfqmr = None

needs_pyfqmr = pytest.mark.skipif(pyfqmr is None, reason="pyfqmr is not installed")


def pyfqmr_simplify(faces, verts, target, aggressiveness=7.0, preserve_border=False):
    s = pyfqmr.Simplify()
    s.setMesh(verts, faces.astype(np.int32))
    s.simplify_mesh(
        target_count=target,
        aggressiveness=aggressiveness,
        preserve_border=preserve_border,
        verbose=False,
    )
    v, f, _ = s.getMesh()
    return np.asarray(v, dtype=np.float64), np.asarray(f, dtype=np.uint32)


@needs_pyfqmr
@pytest.mark.parametrize("ratio", [0.75, 0.5, 0.25, 0.1])
@pytest.mark.parametrize("preserve_border", [False, True])
@pytest.mark.parametrize("aggressiveness", [5.0, 7.0])
def test_matches_pyfqmr(ratio, preserve_border, aggressiveness):
    """Same connectivity, to the face array; same positions, to ~1e-12.

    Not "a similar decimation" — the *same* one. Both implementations run the same
    deterministic threshold sweep over the same float64 arithmetic, so any real
    divergence in the port shows up here immediately.
    """
    faces, verts = uv_sphere(30, 30)
    target = max(1, round(ratio * len(faces)))

    want_v, want_f = pyfqmr_simplify(
        faces, verts, target, aggressiveness, preserve_border
    )
    got_v, got_f, vmap = fastcore.simplify_mesh(
        faces,
        verts,
        n_faces=target,
        aggressiveness=aggressiveness,
        preserve_border=preserve_border,
    )

    np.testing.assert_array_equal(got_f, want_f)
    np.testing.assert_allclose(got_v, want_v, atol=1e-9)
    check_simplify_invariants((got_v, got_f, vmap), faces, verts)


@needs_pyfqmr
def test_matches_pyfqmr_on_an_open_mesh():
    """A grid has a boundary, which is the case the `border` heuristic governs."""
    faces, verts = grid_mesh(15)
    target = len(faces) // 3

    want_v, want_f = pyfqmr_simplify(faces, verts, target, preserve_border=True)
    got_v, got_f, _ = fastcore.simplify_mesh(
        faces, verts, n_faces=target, preserve_border=True
    )

    np.testing.assert_array_equal(got_f, want_f)
    np.testing.assert_allclose(got_v, want_v, atol=1e-9)


# -----------------------------------------------------------------------------
# The vertex map
# -----------------------------------------------------------------------------


def test_identity_target_is_a_no_op():
    faces, verts = uv_sphere(10, 10)
    v, f, vmap = fastcore.simplify_mesh(faces, verts, ratio=1.0)

    np.testing.assert_array_equal(f, faces)
    np.testing.assert_array_equal(v, verts)
    np.testing.assert_array_equal(vmap, np.arange(len(verts)))


def test_map_transfers_a_per_vertex_quantity():
    """The motivating use case: move synapse counts onto the simplified mesh.

    Nothing may be double-counted or invented, so the transferred total has to equal
    the input total over the vertices that survived.
    """
    faces, verts = uv_sphere(20, 20)
    rng = np.random.default_rng(0)
    syn = rng.integers(0, 5, size=len(verts))

    v, f, vmap = fastcore.simplify_mesh(faces, verts, ratio=0.2)
    live = vmap >= 0
    counts = np.bincount(vmap[live], weights=syn[live], minlength=len(v))

    assert counts.sum() == syn[live].sum()
    assert len(counts) == len(v)


def test_map_beats_a_random_assignment():
    """Each vertex maps to something *near* it.

    Structural invariants would all still pass if the map were shuffled, so pin the
    thing they cannot: the mean distance from a vertex to its image has to be far
    below what an arbitrary assignment would give.
    """
    faces, verts = uv_sphere(20, 20)
    v, f, vmap = fastcore.simplify_mesh(faces, verts, ratio=0.25)

    live = vmap >= 0
    actual = np.linalg.norm(v[vmap[live]] - verts[live], axis=1).mean()

    rng = np.random.default_rng(0)
    shuffled = rng.permutation(vmap[live])
    chance = np.linalg.norm(v[shuffled] - verts[live], axis=1).mean()

    assert actual < chance / 5, f"map is no better than chance: {actual} vs {chance}"


def test_unreferenced_vertices_map_to_minus_one():
    faces, verts = uv_sphere(10, 10)
    padded = np.vstack([verts, [[9.0, 9.0, 9.0], [8.0, 8.0, 8.0]]])

    v, f, vmap = fastcore.simplify_mesh(faces, padded, ratio=0.5)
    check_simplify_invariants((v, f, vmap), faces, padded)
    assert (vmap[-2:] == -1).all(), "vertices in no face cannot survive"


def test_map_is_onto_under_heavy_decimation():
    faces, verts = uv_sphere(24, 24)
    out = fastcore.simplify_mesh(faces, verts, ratio=0.05)
    check_simplify_invariants(out, faces, verts)
    assert len(out[0]) < len(verts) / 8


# -----------------------------------------------------------------------------
# Pinning
# -----------------------------------------------------------------------------


def test_locked_vertices_survive_at_exactly_their_position():
    faces, verts = uv_sphere(20, 20)
    lock = np.zeros(len(verts), dtype=bool)
    lock[::7] = True

    v, f, vmap = fastcore.simplify_mesh(faces, verts, ratio=0.15, lock=lock)
    check_simplify_invariants((v, f, vmap), faces, verts)

    assert (vmap[lock] >= 0).all(), "a locked vertex was decimated away"
    # Bitwise. A locked vertex's position is never recomputed, which is what lets a
    # caller key data off the coordinates rather than the index.
    np.testing.assert_array_equal(v[vmap[lock]], verts[lock])
    # ...and each lands in its own output vertex, so nothing is merged into it that
    # would make two pinned vertices indistinguishable.
    assert len(set(vmap[lock].tolist())) == lock.sum()


def test_locking_everything_blocks_all_collapses():
    faces, verts = uv_sphere(10, 10)
    lock = np.ones(len(verts), dtype=bool)

    v, f, vmap = fastcore.simplify_mesh(faces, verts, n_faces=1, lock=lock)
    np.testing.assert_array_equal(f, faces)
    np.testing.assert_array_equal(v, verts)
    np.testing.assert_array_equal(vmap, np.arange(len(verts)))


def test_locked_vertices_still_absorb_neighbours():
    """The asymmetric rule: pinned vertices may take neighbours in.

    Freezing both directions would stall the sweep as soon as the pinned set got
    dense, which is exactly the synapse case. Here a sparse pinned set must not stop
    the target being reached.
    """
    faces, verts = uv_sphere(20, 20)
    lock = np.zeros(len(verts), dtype=bool)
    lock[::20] = True
    target = len(faces) // 4

    v, f, vmap = fastcore.simplify_mesh(faces, verts, n_faces=target, lock=lock)
    assert len(f) <= target

    preimages = np.bincount(vmap[vmap >= 0], minlength=len(v))
    assert preimages[vmap[lock]].max() > 1, "no pinned vertex absorbed anything"


# -----------------------------------------------------------------------------
# Lossless
# -----------------------------------------------------------------------------


def test_lossless_removes_coplanar_interior_without_moving_the_rim():
    faces, verts = grid_mesh(10)
    v, f, vmap = fastcore.simplify_mesh_lossless(faces, verts, preserve_border=True)
    check_simplify_invariants((v, f, vmap), faces, verts)

    assert len(f) < len(faces)
    assert np.abs(v[:, 2]).max() < 1e-9, "left the plane"
    # The border is frozen, so the footprint is untouched.
    assert v[:, 0].min() == 0.0 and v[:, 0].max() == 9.0
    assert v[:, 1].min() == 0.0 and v[:, 1].max() == 9.0


def test_lossless_barely_touches_a_sphere():
    """Nothing on a curved surface is free to remove, so almost nothing goes."""
    faces, verts = uv_sphere(15, 15)
    v, f, vmap = fastcore.simplify_mesh_lossless(faces, verts, epsilon=1e-12)
    check_simplify_invariants((v, f, vmap), faces, verts)
    assert len(f) > 0.9 * len(faces)


def test_lossless_welds_duplicate_vertices():
    """Two coincident vertices: the zero-length edge between them costs nothing."""
    verts = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
    )
    faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.uint32)
    out = fastcore.simplify_mesh_lossless(faces, verts)
    check_simplify_invariants(out, faces, verts)


# -----------------------------------------------------------------------------
# Inputs that break the alternatives
# -----------------------------------------------------------------------------


def test_non_manifold_input_is_tolerated():
    """Three faces on one edge, plus a bowtie vertex.

    This is the input class that rules out every halfedge-based crate: they either
    refuse the mesh outright (`alum`) or silently drop the third face
    (`baby_shark`). Here it just has to work.
    """
    faces = np.array(
        [[0, 1, 2], [0, 1, 4], [0, 1, 5], [3, 6, 7], [3, 8, 9]], dtype=np.uint32
    )
    verts = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            [5.0, 5.0, 0.0],
            [0.5, -1.0, 0.0],
            [0.5, 0.0, 1.0],
            [6.0, 5.0, 0.0],
            [5.5, 6.0, 0.0],
            [4.0, 5.0, 0.0],
            [4.5, 4.0, 0.0],
        ]
    )
    out = fastcore.simplify_mesh(faces, verts, n_faces=2)
    check_simplify_invariants(out, faces, verts)


def test_degenerate_faces_do_not_poison_the_result():
    """A zero-area face and a face naming a vertex twice.

    Upstream normalises a zero-length normal into NaN, and NaN then defeats both
    collapse guards because every comparison against it is false.
    """
    faces, verts = uv_sphere(10, 10)
    n = len(verts)
    verts = np.vstack([verts, [[3.0, 0.0, 0.0], [4.0, 0.0, 0.0], [5.0, 0.0, 0.0]]])
    faces = np.vstack([faces, [[n, n + 1, n + 2], [0, 0, 1]]]).astype(np.uint32)

    out = fastcore.simplify_mesh(faces, verts, ratio=0.5)
    check_simplify_invariants(out, faces, verts)
    assert np.isfinite(out[0]).all()


def test_empty_and_trivial_inputs():
    empty_f = np.zeros((0, 3), dtype=np.uint32)
    empty_v = np.zeros((0, 3), dtype=np.float64)

    v, f, vmap = fastcore.simplify_mesh(empty_f, empty_v, ratio=0.5)
    assert len(v) == 0 and len(f) == 0 and len(vmap) == 0

    # Vertices but no faces: everything is an orphan.
    verts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    v, f, vmap = fastcore.simplify_mesh(empty_f, verts, ratio=0.5)
    check_simplify_invariants((v, f, vmap), empty_f, verts)
    assert (vmap == -1).all()

    # A single triangle cannot collapse without deleting itself.
    one = np.array([[0, 1, 2]], dtype=np.uint32)
    check_simplify_invariants(fastcore.simplify_mesh(one, verts, n_faces=0), one, verts)


def test_deterministic_across_runs():
    faces, verts = uv_sphere(18, 18)
    a = fastcore.simplify_mesh(faces, verts, ratio=0.25)
    b = fastcore.simplify_mesh(faces, verts, ratio=0.25)

    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])
    np.testing.assert_array_equal(a[2], b[2])


# -----------------------------------------------------------------------------
# Argument validation
# -----------------------------------------------------------------------------


def test_target_must_be_named_exactly_once():
    faces, verts = uv_sphere(6, 6)
    with pytest.raises(ValueError, match="exactly one"):
        fastcore.simplify_mesh(faces, verts)
    with pytest.raises(ValueError, match="exactly one"):
        fastcore.simplify_mesh(faces, verts, ratio=0.5, n_faces=10)


@pytest.mark.parametrize("ratio", [0.0, -0.5, 1.5, np.nan, np.inf])
def test_bad_ratio_is_rejected(ratio):
    faces, verts = uv_sphere(6, 6)
    with pytest.raises(ValueError, match="`ratio` must be"):
        fastcore.simplify_mesh(faces, verts, ratio=ratio)


def test_bad_arguments_are_rejected():
    faces, verts = uv_sphere(6, 6)

    with pytest.raises(ValueError, match="`n_faces` must be"):
        fastcore.simplify_mesh(faces, verts, n_faces=-1)

    with pytest.raises(ValueError, match="must have"):
        fastcore.simplify_mesh(faces, verts, ratio=0.5, lock=np.zeros(3, dtype=bool))

    with pytest.raises(ValueError, match="must be finite"):
        bad = verts.copy()
        bad[0, 0] = np.nan
        fastcore.simplify_mesh(faces, bad, ratio=0.5)

    with pytest.raises(ValueError, match="references vertex"):
        fastcore.simplify_mesh(faces, verts[:-1], ratio=0.5)

    with pytest.raises(ValueError, match=r"shape \(F, 3\)"):
        fastcore.simplify_mesh(faces[:, :2], verts, ratio=0.5)

    with pytest.raises(ValueError, match="`epsilon` must be"):
        fastcore.simplify_mesh_lossless(faces, verts, epsilon=-1.0)


def test_accepts_unconverted_input():
    """Lists and the wrong dtypes go through the same coercion as the rest of the API."""
    faces, verts = uv_sphere(8, 8)
    out = fastcore.simplify_mesh(
        faces.astype(np.int64).tolist(), verts.astype(np.float32), ratio=0.5
    )
    check_simplify_invariants(out, faces, verts)
