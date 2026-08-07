"""Tests for mesh smoothing.

Two kinds of test, the same split as `test_simplify.py`.

The **oracle** tests compare against `trimesh.smoothing`, which implements the same
three filters over the same uniform umbrella. On a real neuron mesh the two agree to
~1e-11 on coordinates spanning 25,880 units — floating-point noise — which is what
makes this a check on the arithmetic rather than on the general shape of the result.
`trimesh` is a test-only dependency; those tests skip without it.

The **invariant** tests need no oracle and cover the half `trimesh` cannot check
because it does the opposite: locking, boundary preservation, translation
equivariance, and a volume correction that scales about the mesh rather than about the
origin. Two of them assert directly that `trimesh` gets those wrong, which is the
reason this module exists and worth having a regression on.
"""

import warnings

import numpy as np
import pytest

import navis_fastcore as fastcore
from meshes import grid_mesh, uv_sphere

try:
    import trimesh as tm
except ImportError:  # pragma: no cover - depends on the environment
    tm = None

needs_trimesh = pytest.mark.skipif(tm is None, reason="trimesh is not installed")

METHODS = ("laplacian", "taubin", "humphrey")
WEIGHTS = ("uniform", "inverse_distance", "cotangent")


def noisy_grid(n=12, amplitude=0.4):
    """A flat grid displaced out of its plane at the highest frequency it can carry.

    The clean surface is `z = 0`, so residual `z` is exactly the noise and any sliding
    *within* the plane — which the uniform umbrella does plenty of — does not
    contaminate the measurement.
    """
    faces, verts = grid_mesh(n)
    verts = verts.copy()
    verts[:, 2] = np.where(np.arange(len(verts)) % 2 == 0, amplitude, -amplitude)
    return faces, verts


def signed_volume(faces, verts):
    """Enclosed volume by the divergence theorem, about the mesh's own centroid.

    About the centroid so that this stays usable on a mesh at EM coordinates, where
    anchoring at the origin costs six digits to cancellation.
    """
    p = verts - verts.mean(axis=0)
    a, b, c = p[faces[:, 0]], p[faces[:, 1]], p[faces[:, 2]]
    return np.einsum("ij,ij->i", np.cross(a, b), c).sum() / 6.0


# -----------------------------------------------------------------------------
# Against trimesh
# -----------------------------------------------------------------------------


@needs_trimesh
def test_laplacian_matches_trimesh():
    faces, verts = uv_sphere(24, 24)
    m = tm.Trimesh(verts.copy(), faces.copy(), process=False)
    ref = tm.smoothing.filter_laplacian(
        m, lamb=0.5, iterations=10, volume_constraint=False
    )
    out = fastcore.smooth_mesh(faces, verts, method="laplacian", lamb=0.5, iterations=10)
    assert np.allclose(out, ref.vertices, atol=1e-12)


@needs_trimesh
def test_taubin_matches_trimesh_at_twice_the_iterations():
    """One `iterations` here is a full lambda/mu pair; trimesh counts half-steps.

    Pinning that down against the oracle is the point: the factor of two is a
    deliberate divergence, so it needs a test that fails if it silently changes.
    """
    faces, verts = uv_sphere(24, 24)
    m = tm.Trimesh(verts.copy(), faces.copy(), process=False)
    ref = tm.smoothing.filter_taubin(m, lamb=0.5, nu=0.53, iterations=20)
    out = fastcore.smooth_mesh(
        faces, verts, method="taubin", lamb=0.5, mu=-0.53, iterations=10
    )
    assert np.allclose(out, ref.vertices, atol=1e-12)


@needs_trimesh
def test_humphrey_matches_trimesh():
    faces, verts = uv_sphere(24, 24)
    m = tm.Trimesh(verts.copy(), faces.copy(), process=False)
    ref = tm.smoothing.filter_humphrey(m, alpha=0.1, beta=0.5, iterations=10)
    out = fastcore.smooth_mesh(
        faces, verts, method="humphrey", alpha=0.1, beta=0.5, iterations=10
    )
    assert np.allclose(out, ref.vertices, atol=1e-12)


@needs_trimesh
def test_inverse_distance_matches_trimesh_for_one_iteration():
    """Only for one, and deliberately.

    `trimesh` builds its operator once from the input geometry and reuses it, so from
    the second iteration onwards its weights describe a mesh that no longer exists.
    Here they are recomputed from the current positions every pass, which is the flow
    the weighting is supposed to discretise. The two therefore agree exactly for the
    first iteration and diverge on purpose after it.
    """
    faces, verts = uv_sphere(20, 20)
    m = tm.Trimesh(verts.copy(), faces.copy(), process=False)
    op = tm.smoothing.laplacian_calculation(m, equal_weight=False)
    ref = tm.smoothing.filter_laplacian(
        m, lamb=0.5, iterations=1, volume_constraint=False, laplacian_operator=op
    )
    out = fastcore.smooth_mesh(
        faces, verts, method="laplacian", lamb=0.5, iterations=1,
        weights="inverse_distance",
    )
    assert np.allclose(out, ref.vertices, atol=1e-12)


@needs_trimesh
def test_volume_correction_beats_trimesh_where_trimesh_is_wrong():
    """The claim the module exists for, as a regression against the thing it fixes.

    `trimesh`'s volume constraint scales about the origin, so it translates the mesh
    and stops being a function of its shape. Both halves are asserted: that upstream
    displaces a mesh sitting at plausible EM coordinates by more than its own size,
    and that this does not.
    """
    faces, verts = uv_sphere(20, 20)
    verts = verts + 1e3  # a mesh that does not happen to sit on the origin
    extent = np.ptp(verts, axis=0).max()

    ref = tm.smoothing.filter_laplacian(
        tm.Trimesh(verts.copy(), faces.copy(), process=False),
        lamb=0.5,
        iterations=5,
        volume_constraint=True,
    )
    upstream_shift = np.linalg.norm(np.asarray(ref.vertices).mean(0) - verts.mean(0))
    assert upstream_shift > extent, "fixture no longer reproduces the upstream bug"

    out = fastcore.smooth_mesh(
        faces, verts, method="laplacian", lamb=0.5, iterations=5,
        volume_correction=True,
    )
    plain = fastcore.smooth_mesh(
        faces, verts, method="laplacian", lamb=0.5, iterations=5
    )
    # The correction changes size and not position.
    assert np.allclose(out.mean(0), plain.mean(0), atol=1e-9)
    assert abs(signed_volume(faces, out) / signed_volume(faces, verts) - 1) < 1e-9


@needs_trimesh
def test_faster_than_trimesh():
    """Not a benchmark — a floor, so a rewrite that quietly loses the point fails.

    The margin on a real mesh is ~150x; asking for 5x here keeps this from being a
    flaky timing test on a loaded machine while still catching a regression that
    reintroduces per-vertex Python work.
    """
    import time

    faces, verts = uv_sphere(120, 120)
    m = tm.Trimesh(verts.copy(), faces.copy(), process=False)

    t0 = time.perf_counter()
    tm.smoothing.filter_laplacian(m, lamb=0.5, iterations=10)
    upstream = time.perf_counter() - t0

    fastcore.smooth_mesh(faces, verts, method="laplacian", iterations=1)  # warm
    t0 = time.perf_counter()
    fastcore.smooth_mesh(
        faces, verts, method="laplacian", iterations=10, volume_correction=True
    )
    ours = time.perf_counter() - t0

    assert ours * 5 < upstream, f"fastcore {ours:.3f}s vs trimesh {upstream:.3f}s"


# -----------------------------------------------------------------------------
# Invariants
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("weights", WEIGHTS)
def test_shape_and_dtype_survive(method, weights):
    faces, verts = uv_sphere(12, 12)
    out = fastcore.smooth_mesh(faces, verts, method=method, weights=weights)
    assert out.shape == verts.shape
    assert out.dtype == np.float64
    assert np.isfinite(out).all()
    assert not np.array_equal(out, verts), "nothing moved"


@pytest.mark.parametrize("method", METHODS)
def test_zero_iterations_is_the_identity(method):
    # Not swept over `weights`: at zero iterations the weighting is never consulted, so
    # the extra six cases would exercise the same path three times over.
    faces, verts = uv_sphere(10, 10)
    out = fastcore.smooth_mesh(faces, verts, method=method, iterations=0)
    assert np.array_equal(out, verts)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("weights", WEIGHTS)
def test_noise_is_removed(method, weights):
    faces, verts = noisy_grid()
    rough = (verts[:, 2] ** 2).sum()
    out = fastcore.smooth_mesh(
        faces, verts, method=method, weights=weights, iterations=20
    )
    left = (out[:, 2] ** 2).sum()
    # The binding case is HC on cotangent weights, which converges to a fixed point
    # still carrying ~31%: a maximally corrugated grid makes nearly every triangle
    # obtuse, the clamp zeroes most of the cotangents, and HC's pull-back towards the
    # rough input holds what survives in place. Both halves are documented behaviour.
    assert left < 0.35 * rough
    if method != "humphrey":
        assert left < 0.06 * rough


@pytest.mark.parametrize("method", METHODS)
def test_a_flat_grid_is_a_fixed_point(method):
    """Exactly, not nearly: an interior grid vertex is the average of its six
    neighbours, and a sum of six exactly-representable values over six does not round.

    Also what pins down what *pinning* means. Not merely that a frozen vertex ends
    where it started — a filter that let the rim wander mid-iteration and put it back
    afterwards would satisfy that — but that it never acts on its neighbours from
    anywhere else. Get that wrong and the interior next to the rim is dragged by an
    excursion that officially never happened, which is what this catches.
    """
    faces, verts = grid_mesh(7)
    out = fastcore.smooth_mesh(
        faces, verts, method=method, iterations=25, preserve_border=True
    )
    assert np.array_equal(out, verts)


@pytest.mark.parametrize("method", METHODS)
def test_locked_vertices_do_not_move(method):
    faces, verts = uv_sphere(12, 12)
    lock = np.zeros(len(verts), dtype=bool)
    lock[::5] = True
    out = fastcore.smooth_mesh(faces, verts, method=method, lock=lock, iterations=10)
    assert np.array_equal(out[lock], verts[lock]), "a locked vertex moved"
    assert not np.array_equal(out[~lock], verts[~lock])


def test_preserve_border_pins_the_rim():
    faces, verts = noisy_grid(9)
    out = fastcore.smooth_mesh(
        faces, verts, method="laplacian", iterations=10, preserve_border=True
    )
    ij = np.arange(len(verts))
    rim = (ij // 9 == 0) | (ij // 9 == 8) | (ij % 9 == 0) | (ij % 9 == 8)
    assert np.array_equal(out[rim], verts[rim]), "a rim vertex moved"
    assert abs(out[40, 2]) < 0.05, "the interior did not smooth"


def test_preserve_border_and_lock_compose():
    faces, verts = noisy_grid(5)
    lock = np.zeros(len(verts), dtype=bool)
    lock[12] = True  # interior, so not covered by the border flags
    out = fastcore.smooth_mesh(
        faces, verts, iterations=5, preserve_border=True, lock=lock
    )
    assert np.array_equal(out[12], verts[12])
    assert np.array_equal(out[0], verts[0])
    assert not np.array_equal(out[7], verts[7])


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("correction", [False, True])
def test_translation_equivariance(method, correction):
    """Smoothing a mesh 100 um from the origin gives the same shape as at the origin.

    `trimesh` fails this outright at this offset — its volume constraint returns NaN —
    and it is why both the volume measurement and the scaling here are anchored to the
    mesh rather than to the coordinate system.
    """
    faces, verts = uv_sphere(20, 20)
    offset = 1e5
    kw = dict(method=method, iterations=10, volume_correction=correction)
    here = fastcore.smooth_mesh(faces, verts, **kw)
    there = fastcore.smooth_mesh(faces, verts + offset, **kw)
    # Unit-radius mesh at 1e5, so f64 resolves it to ~1e-11 there.
    assert np.abs(here - (there - offset)).max() < 1e-8


@pytest.mark.parametrize("method", METHODS)
def test_scale_equivariance(method):
    """The claim that lets the volume correction run once at the end rather than every
    iteration: the filters commute with a uniform scaling, so the two are equal.

    Swept over all three weightings in the Rust suite; here, as for translation, the
    job is only to show the arguments survive the binding.
    """
    faces, verts = uv_sphere(16, 16)
    kw = dict(method=method, iterations=8)
    a = fastcore.smooth_mesh(faces, verts * 7.5, **kw)
    b = fastcore.smooth_mesh(faces, verts, **kw) * 7.5
    assert np.abs(a - b).max() < 1e-10


def test_volume_correction_restores_the_volume():
    faces, verts = uv_sphere(24, 24)
    v0 = signed_volume(faces, verts)
    out = fastcore.smooth_mesh(
        faces, verts, method="laplacian", iterations=10, volume_correction=True
    )
    assert abs(signed_volume(faces, out) / v0 - 1) < 1e-9


def test_taubin_holds_the_volume_laplacian_loses():
    faces, verts = uv_sphere(24, 24)
    v0 = signed_volume(faces, verts)
    # The `lap` half is also the "there is something for the correction to undo"
    # premise of `test_volume_correction_restores_the_volume`, asserted here once
    # rather than in both on the same fixture.
    lap = signed_volume(
        faces, fastcore.smooth_mesh(faces, verts, method="laplacian", iterations=20)
    )
    tau = signed_volume(
        faces, fastcore.smooth_mesh(faces, verts, method="taubin", iterations=20)
    )
    assert lap < 0.75 * v0
    assert tau > 0.95 * v0


def test_open_mesh_warns_instead_of_scaling_by_nonsense():
    faces, verts = noisy_grid(6)
    with pytest.warns(RuntimeWarning, match="no usable enclosed volume"):
        out = fastcore.smooth_mesh(
            faces, verts, method="laplacian", iterations=5, volume_correction=True
        )
    assert np.isfinite(out).all()
    assert abs(out[21, 2]) < 0.2, "undefined means unscaled, not unsmoothed"


def test_no_warning_when_the_volume_is_fine():
    """The warning has to be rare enough to mean something, so the closed-mesh path
    must be silent."""
    faces, verts = uv_sphere(16, 16)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fastcore.smooth_mesh(faces, verts, iterations=5, volume_correction=True)


def test_threads_do_not_change_the_result():
    faces, verts = uv_sphere(16, 16)
    kw = dict(method="taubin", weights="cotangent", iterations=8,
              volume_correction=True)
    assert np.array_equal(
        fastcore.smooth_mesh(faces, verts, threads=1, **kw),
        fastcore.smooth_mesh(faces, verts, threads=4, **kw),
    )


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("weights", WEIGHTS)
def test_degenerate_geometry_is_merely_data(method, weights):
    """The shapes EM meshes are actually made of."""
    faces = np.array(
        [
            [0, 1, 2],
            [0, 1, 2],  # duplicate
            [1, 2, 3],
            [4, 4, 5],  # names a vertex twice
            [5, 6, 7],  # zero-area: three collinear points
        ],
        dtype=np.uint32,
    )
    verts = np.array(
        [
            [0.0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0.5],
            [2, 0, 0], [3, 0, 0], [4, 0, 0], [5, 0, 0],
            [9, 9, 9],  # referenced by nothing
        ]
    )
    out = fastcore.smooth_mesh(faces, verts, method=method, weights=weights,
                               iterations=5)
    assert np.isfinite(out).all()
    assert np.array_equal(out[8], verts[8]), "an unreferenced vertex moved"


def test_an_empty_mesh_is_a_no_op():
    faces = np.zeros((0, 3), dtype=np.uint32)
    verts = np.array([[0.0, 0, 0], [1, 2, 3]])
    out = fastcore.smooth_mesh(faces, verts, weights="cotangent", iterations=10)
    assert np.array_equal(out, verts)


def test_coincident_vertices_do_not_blow_up():
    faces = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.uint32)
    verts = np.array([[0.0, 0, 0], [0, 0, 0], [1, 0, 0], [1, 1, 0]])
    for weights in WEIGHTS:
        out = fastcore.smooth_mesh(faces, verts, weights=weights, iterations=5)
        assert np.isfinite(out).all()


# -----------------------------------------------------------------------------
# Rejected input
# -----------------------------------------------------------------------------
#
# All of these must be `ValueError`, not the `PanicException` the core's own asserts
# would produce — that is the whole job of the pre-checks in the wrapper.


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(method="nope"), "`method` must be one of"),
        (dict(weights="nope"), "must be one of"),
        (dict(method="laplacian", lamb=1.5), r"`lamb` must be in \[0, 1\]"),
        (dict(method="laplacian", lamb=-0.1), r"`lamb` must be in \[0, 1\]"),
        (dict(method="taubin", lamb=0.5, mu=-0.4), "larger in magnitude"),
        (dict(method="taubin", lamb=0.5, mu=0.53), "larger in magnitude"),
        (dict(method="taubin", lamb=0.5, mu=-1.5), r"`mu` must be in \[-1, 0\)"),
        (dict(method="humphrey", alpha=2.0), r"`alpha` must be in \[0, 1\]"),
        (dict(method="humphrey", beta=-1.0), r"`beta` must be in \[0, 1\]"),
        (dict(iterations=-1), "`iterations` must be non-negative"),
        # Parameters that belong to another method are refused, not ignored.
        (dict(method="taubin", alpha=0.3), "does not apply"),
        (dict(method="laplacian", mu=-0.53), "does not apply"),
        (dict(method="humphrey", lamb=0.5), "does not apply"),
    ],
)
def test_bad_parameters_raise_value_error(kwargs, match):
    faces, verts = grid_mesh(4)
    with pytest.raises(ValueError, match=match):
        fastcore.smooth_mesh(faces, verts, **kwargs)


def test_bad_shapes_raise_value_error():
    faces, verts = grid_mesh(4)
    with pytest.raises(ValueError, match="must be a 2-D array of shape"):
        fastcore.smooth_mesh(faces[:, :2], verts)
    with pytest.raises(ValueError, match="must be a 2-D array of shape"):
        fastcore.smooth_mesh(faces, verts[:, :2])
    with pytest.raises(ValueError, match="must have .* entries"):
        fastcore.smooth_mesh(faces, verts, lock=np.zeros(3, dtype=bool))
    with pytest.raises(ValueError, match="references vertex"):
        fastcore.smooth_mesh(np.array([[0, 1, 99]], dtype=np.uint32), verts)
    with pytest.raises(ValueError, match="must be finite"):
        bad = verts.copy()
        bad[0, 0] = np.nan
        fastcore.smooth_mesh(faces, bad)
