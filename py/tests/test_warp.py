"""Landmark-based transforms: thin-plate spline and moving least squares.

Where `navis` is importable these check us against it directly - `navis.transforms`
delegates to `morphops` and `molesq`, which are the reference implementations. Where it is
not, the properties checked here (landmark interpolation, exact recovery of a global
affine) pin the maths on their own.
"""

import pickle

import numpy as np
import pytest

import navis_fastcore as fastcore

# The reference implementations agree with us to ~1e-14 *relative to the spatial extent of
# the landmarks*. These tolerances are on data spanning ~100 units, so they leave several
# orders of magnitude of headroom. A change that pushes agreement to 1e-6 means something is
# subtly wrong - do not loosen them.
ATOL = 1e-9


@pytest.fixture(scope="module")
def landmarks():
    """A non-degenerate landmark pair with a genuinely non-affine component."""
    rng = np.random.default_rng(42)
    src = rng.uniform(0, 100, size=(40, 3))
    trg = src * [1.3, 0.8, 1.1] + [5.0, -3.0, 12.0]
    trg += rng.normal(0, 4.0, size=src.shape)  # the warp the affine cannot explain
    return src, trg


@pytest.fixture(scope="module")
def points():
    rng = np.random.default_rng(7)
    return rng.uniform(0, 100, size=(500, 3))


CLASSES = [fastcore.TpsTransform, fastcore.MlsTransform]
IDS = ["tps", "mls"]


# ---------------------------------------------------------------------------
# Against navis / the reference implementations
# ---------------------------------------------------------------------------


def _rel_err(got, ref, src):
    """Max deviation relative to the spatial extent of the landmarks."""
    return np.abs(got - ref).max() / np.linalg.norm(src.max(axis=0) - src.min(axis=0))


def test_tps_matches_morphops(landmarks, points):
    """The whole point: same numbers as the implementation we are replacing.

    Compared against `morphops` rather than `navis.transforms.TPStransform`, because that
    class *is* these four lines - and importing `navis` drags in its full optional
    dependency stack, which makes the check flaky for reasons unrelated to the maths.
    """
    mops = pytest.importorskip("morphops", reason="morphops not installed")
    # morphops still calls `np.row_stack`, removed in numpy 2. navis hotfixes this the same
    # way; without it the reference cannot run at all on a current numpy.
    if not hasattr(np, "row_stack"):
        np.row_stack = np.vstack

    src, trg = landmarks
    W, A = mops.tps_coefs(src, trg)
    ref = mops.P_matrix(points) @ A + mops.K_matrix(points, src) @ W

    got = fastcore.TpsTransform(src, trg).xform(points)
    assert _rel_err(got, ref, src) < 1e-10


def test_mls_matches_molesq(landmarks, points):
    """Same, for moving least squares: `molesq` is what `navis` delegates to."""
    molesq = pytest.importorskip("molesq", reason="molesq not installed")
    src, trg = landmarks

    ref = molesq.Transformer(src, trg).transform(points)
    got = fastcore.MlsTransform(src, trg).xform(points)
    assert _rel_err(got, ref, src) < 1e-10


def test_mls_reverse_matches_molesq(landmarks, points):
    """`direction="inverse"` must mean what `molesq`'s `reverse=True` means."""
    molesq = pytest.importorskip("molesq", reason="molesq not installed")
    src, trg = landmarks

    ref = molesq.Transformer(src, trg).transform(points, reverse=True)
    got = fastcore.MlsTransform(src, trg, direction="inverse").xform(points)
    assert _rel_err(got, ref, src) < 1e-10


# ---------------------------------------------------------------------------
# Properties that hold regardless of implementation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", CLASSES, ids=IDS)
def test_landmarks_map_to_targets(landmarks, cls):
    """Both transforms are interpolating: a source landmark lands on its partner."""
    src, trg = landmarks
    got = cls(src, trg).xform(src)
    assert np.allclose(got, trg, atol=1e-6)


@pytest.mark.parametrize("cls", CLASSES, ids=IDS)
def test_recovers_global_affine(cls):
    """Affine landmarks in, that exact affine out - everywhere, including far outside."""
    rng = np.random.default_rng(3)
    src = rng.uniform(0, 10, size=(20, 3))

    def affine(v):
        return v @ np.array([[2.0, 0.1, 0.0], [0.0, -1.0, 0.25], [0.3, 0.0, 0.75]]) + [
            3.0,
            4.0,
            5.0,
        ]

    trg = affine(src)
    tr = cls(src, trg)

    pts = rng.uniform(-500, 500, size=(50, 3))  # well outside the landmark hull
    assert np.allclose(tr.xform(pts), affine(pts), atol=1e-6)


@pytest.mark.parametrize("cls", CLASSES, ids=IDS)
def test_matrix_affine(cls):
    """`matrix_affine` is the homogeneous form of the recovered global affine."""
    rng = np.random.default_rng(11)
    src = rng.uniform(0, 10, size=(20, 3))
    linear = np.array([[2.0, 0.1, 0.0], [0.0, -1.0, 0.25], [0.3, 0.0, 0.75]])
    offset = np.array([3.0, 4.0, 5.0])
    trg = src @ linear + offset

    m = cls(src, trg).matrix_affine
    assert m.shape == (4, 4)
    assert np.allclose(m[3], [0, 0, 0, 1])
    # `linear` is applied as a row-vector product, so the matrix form is its transpose.
    assert np.allclose(m[:3, :3], linear.T, atol=1e-8)
    assert np.allclose(m[:3, 3], offset, atol=1e-8)


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", CLASSES, ids=IDS)
def test_single_point_roundtrips_as_1d(landmarks, cls):
    src, trg = landmarks
    tr = cls(src, trg)
    out = tr.xform(np.array([50.0, 50.0, 50.0]))
    assert out.shape == (3,)
    assert np.allclose(out, tr.xform(np.array([[50.0, 50.0, 50.0]]))[0])


@pytest.mark.parametrize("cls", CLASSES, ids=IDS)
def test_dataframe_input(landmarks, points, cls):
    """DataFrames with x/y/z columns work for both landmarks and points."""
    pd = pytest.importorskip("pandas")
    src, trg = landmarks
    src_df = pd.DataFrame(src, columns=["x", "y", "z"])
    trg_df = pd.DataFrame(trg, columns=["x", "y", "z"])
    pts_df = pd.DataFrame(points, columns=["x", "y", "z"])

    assert np.allclose(cls(src_df, trg_df).xform(pts_df), cls(src, trg).xform(points))


@pytest.mark.parametrize("cls", CLASSES, ids=IDS)
def test_dataframe_missing_columns(landmarks, cls):
    pd = pytest.importorskip("pandas")
    src, trg = landmarks
    bad = pd.DataFrame(src, columns=["x", "y", "not_z"])
    with pytest.raises(ValueError, match="x/y/z columns"):
        cls(bad, trg)


@pytest.mark.parametrize("cls", CLASSES, ids=IDS)
def test_rejects_mismatched_landmarks(landmarks, cls):
    src, trg = landmarks
    with pytest.raises(ValueError, match="must match"):
        cls(src, trg[:-1])


@pytest.mark.parametrize("cls", CLASSES, ids=IDS)
def test_rejects_non_3d(cls):
    bad = np.zeros((10, 2))
    with pytest.raises(ValueError, match=r"\(N, 3\)"):
        cls(bad, bad)


def test_tps_needs_four_landmarks():
    pts = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]])
    with pytest.raises(ValueError, match="at least 4 landmarks"):
        fastcore.TpsTransform(pts, pts)


def test_mls_rejects_bad_direction(landmarks):
    src, trg = landmarks
    with pytest.raises(ValueError, match="forward"):
        fastcore.MlsTransform(src, trg, direction="sideways")


# ---------------------------------------------------------------------------
# Threading
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", CLASSES, ids=IDS)
@pytest.mark.parametrize("n_cores", [1, 2, 4])
def test_thread_count_does_not_change_result(landmarks, points, cls, n_cores):
    """Chunking is over independent points, so results must be bit-identical."""
    src, trg = landmarks
    tr = cls(src, trg)
    assert np.array_equal(tr.xform(points, n_cores=n_cores), tr.xform(points))


@pytest.mark.parametrize("cls", CLASSES, ids=IDS)
def test_empty_input(landmarks, cls):
    src, trg = landmarks
    out = cls(src, trg).xform(np.zeros((0, 3)))
    assert out.shape == (0, 3)


# ---------------------------------------------------------------------------
# Object protocol
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", CLASSES, ids=IDS)
def test_len_and_repr(landmarks, cls):
    src, trg = landmarks
    tr = cls(src, trg)
    assert len(tr) == len(src)
    assert "landmarks=40" in repr(tr)


@pytest.mark.parametrize("cls", CLASSES, ids=IDS)
def test_equality(landmarks, cls):
    src, trg = landmarks
    assert cls(src, trg) == cls(src, trg)
    assert cls(src, trg) != cls(src, trg * 2)
    assert cls(src, trg) != "not a transform"


@pytest.mark.parametrize("cls", CLASSES, ids=IDS)
def test_copy(landmarks, points, cls):
    src, trg = landmarks
    tr = cls(src, trg)
    assert np.array_equal(tr.copy().xform(points), tr.xform(points))


@pytest.mark.parametrize("cls", CLASSES, ids=IDS)
def test_negation_inverts_direction(landmarks, cls):
    """Negation maps targets back onto sources. Both are refits, not true inverses."""
    src, trg = landmarks
    back = (-cls(src, trg)).xform(trg)
    assert np.allclose(back, src, atol=1e-6)


@pytest.mark.parametrize("cls", CLASSES, ids=IDS)
def test_pickle_roundtrip(landmarks, points, cls):
    src, trg = landmarks
    tr = cls(src, trg)
    restored = pickle.loads(pickle.dumps(tr))
    assert np.allclose(restored.xform(points), tr.xform(points))
    assert len(restored) == len(tr)


def test_tps_pickle_does_not_refit(landmarks):
    """The coefficients travel with the pickle, so unpickling is exact, not a re-solve."""
    src, trg = landmarks
    tr = fastcore.TpsTransform(src, trg)
    restored = pickle.loads(pickle.dumps(tr))
    assert np.array_equal(restored.W, tr.W)
    assert np.array_equal(restored.A, tr.A)


def test_mls_direction_and_negation(landmarks, points):
    """`direction="inverse"` and negating a forward transform are the same thing."""
    src, trg = landmarks
    inverse = fastcore.MlsTransform(src, trg, direction="inverse")
    negated = -fastcore.MlsTransform(src, trg)
    assert inverse.direction == "inverse" == negated.direction
    assert np.array_equal(inverse.xform(points), negated.xform(points))
    assert np.array_equal(inverse.source, trg)
    assert np.array_equal(inverse.target, src)


# ---------------------------------------------------------------------------
# TPS coefficients
# ---------------------------------------------------------------------------


def test_tps_from_coefs(landmarks, points):
    """Rebuilding from W/A reproduces the transform without refitting."""
    src, trg = landmarks
    tr = fastcore.TpsTransform(src, trg)
    rebuilt = fastcore.TpsTransform.from_coefs(tr.source, tr.W, tr.A)
    assert np.array_equal(rebuilt.xform(points), tr.xform(points))
    assert rebuilt.target is None


def test_tps_from_coefs_matches_numpy_fit(landmarks, points):
    """A caller may fit with LAPACK and hand us the coefficients; same answer either way."""
    src, trg = landmarks
    m = len(src)

    # Build and solve the saddle-point system the way morphops does.
    k = np.linalg.norm(src[:, None, :] - src[None, :, :], axis=-1)
    p = np.column_stack([np.ones(m), src])
    L = np.zeros((m + 4, m + 4))
    L[:m, :m] = k
    L[:m, m:] = p
    L[m:, :m] = p.T
    rhs = np.vstack([trg, np.zeros((4, 3))])
    q = np.linalg.solve(L, rhs)

    rebuilt = fastcore.TpsTransform.from_coefs(src, q[:m], q[m:])
    assert np.allclose(rebuilt.xform(points), fastcore.TpsTransform(src, trg).xform(points),
                       atol=ATOL)


def test_tps_from_coefs_validates_shapes(landmarks):
    src, trg = landmarks
    tr = fastcore.TpsTransform(src, trg)
    with pytest.raises(ValueError, match=r"`W` must be"):
        fastcore.TpsTransform.from_coefs(src, tr.W[:-1], tr.A)
    with pytest.raises(ValueError, match=r"`A` must be"):
        fastcore.TpsTransform.from_coefs(src, tr.W, tr.A[:-1])


def test_tps_from_coefs_cannot_negate_without_target(landmarks):
    src, trg = landmarks
    tr = fastcore.TpsTransform(src, trg)
    rebuilt = fastcore.TpsTransform.from_coefs(src, tr.W, tr.A)
    with pytest.raises(ValueError, match="landmarks_target"):
        -rebuilt


def test_tps_coefficient_shapes(landmarks):
    src, trg = landmarks
    tr = fastcore.TpsTransform(src, trg)
    assert tr.W.shape == (len(src), 3)
    assert tr.A.shape == (4, 3)
    assert np.array_equal(tr.source, src)
    assert np.array_equal(tr.target, trg)


def test_tps_singular_landmarks_raise():
    """Duplicate landmarks pulling apart is unsolvable - say so, don't return nonsense."""
    src = np.array([[0.0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    trg = np.array([[0.0, 0, 0], [5, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    with pytest.raises(ValueError, match="singular"):
        fastcore.TpsTransform(src, trg)


# ---------------------------------------------------------------------------
# Scale
# ---------------------------------------------------------------------------


def test_large_landmark_count_is_tractable():
    """The reference MLS cannot run this shape at all; peak memory here is the output."""
    rng = np.random.default_rng(1)
    src = rng.uniform(0, 1000, size=(2000, 3))
    trg = src + rng.normal(0, 5, size=src.shape)
    pts = rng.uniform(0, 1000, size=(5000, 3))

    out = fastcore.MlsTransform(src, trg).xform(pts)
    assert out.shape == pts.shape
    assert np.isfinite(out).all()
