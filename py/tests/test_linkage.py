"""Tests for hierarchical clustering.

SciPy is the oracle throughout: the whole point of this module is to be a cheaper
route to the same answer, so "the same answer" is what gets checked.
"""

import numpy as np
import pytest
from scipy.cluster.hierarchy import fcluster, linkage as sp_linkage
from scipy.spatial.distance import squareform

import navis_fastcore as fastcore

METHODS = ["single", "complete", "average", "weighted", "ward", "centroid", "median"]

#: f32 carries ~7 decimal digits, so agreement with SciPy's f64 is checked at that
#: scale rather than to the bit.
F32_RTOL = 1e-5


def clustered_scores(n=200, k=8, dim=12, seed=0, dtype=np.float32, asymmetric=True):
    """A score matrix with real cluster structure.

    Pure noise makes for a bad oracle test: merge order is then decided by ties and
    float noise, so two correct implementations can legitimately disagree. Actual
    clusters give a well-separated, stable answer.
    """
    rng = np.random.default_rng(seed)
    centers = rng.random((k, dim))
    labels = rng.integers(0, k, n)
    pts = centers[labels] + rng.normal(0, 0.15, (n, dim))

    d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
    d /= d.max()
    scores = 1.0 - d
    if asymmetric:
        # NBLAST is not symmetric; perturb so symmetry= actually has work to do.
        scores = scores + rng.normal(0, 0.01, (n, n))
    np.fill_diagonal(scores, 1.0)
    return np.ascontiguousarray(scores, dtype=dtype), labels


def reference(scores, method, symmetry="mean", transform="one_minus"):
    """The numpy/scipy pipeline this module replaces."""
    m = scores.astype(np.float64)
    if symmetry == "mean":
        m = (m + m.T) / 2
    elif symmetry == "min":
        m = np.minimum(m, m.T)
    elif symmetry == "max":
        m = np.maximum(m, m.T)
    if transform == "one_minus":
        m = 1.0 - m
    np.fill_diagonal(m, 0.0)
    return sp_linkage(squareform(m, checks=False), method=method)


# ---------------------------------------------------------------------------
# condensed_distances
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("symmetry", ["none", "mean", "min", "max"])
@pytest.mark.parametrize("transform", ["one_minus", "none"])
def test_condensed_matches_squareform(symmetry, transform):
    scores, _ = clustered_scores(n=60, dtype=np.float64)

    got = fastcore.condensed_distances(scores, symmetry=symmetry, transform=transform)

    m = scores.copy()
    if symmetry == "mean":
        m = (m + m.T) / 2
    elif symmetry == "min":
        m = np.minimum(m, m.T)
    elif symmetry == "max":
        m = np.maximum(m, m.T)
    if transform == "one_minus":
        m = 1.0 - m
    np.fill_diagonal(m, 0.0)
    want = squareform(m, checks=False)

    assert got.shape == want.shape
    np.testing.assert_allclose(got, want, rtol=1e-12)


def test_condensed_preserves_dtype():
    """The memory argument dies if f32 silently becomes f64."""
    scores, _ = clustered_scores(n=40, dtype=np.float32)
    assert fastcore.condensed_distances(scores).dtype == np.float32
    assert fastcore.condensed_distances(scores.astype(np.float64)).dtype == np.float64


def test_condensed_ignores_diagonal():
    """Self-scores on the diagonal must not leak into the result."""
    scores, _ = clustered_scores(n=30, dtype=np.float64)
    base = fastcore.condensed_distances(scores)
    poisoned = scores.copy()
    np.fill_diagonal(poisoned, np.nan)
    np.testing.assert_array_equal(base, fastcore.condensed_distances(poisoned))


def test_condensed_f_order_matches_c_order():
    """F-order is handled by transposing the view, not by copying."""
    scores, _ = clustered_scores(n=40, dtype=np.float64)
    f = np.asfortranarray(scores)
    assert not f.flags.c_contiguous
    for symmetry in ["none", "mean", "min", "max"]:
        np.testing.assert_allclose(
            fastcore.condensed_distances(f, symmetry=symmetry),
            fastcore.condensed_distances(scores, symmetry=symmetry),
            rtol=1e-12,
        )


# ---------------------------------------------------------------------------
# linkage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", METHODS)
def test_linkage_matches_scipy_f64(method):
    scores, _ = clustered_scores(n=200, dtype=np.float64)

    got = fastcore.linkage(scores, method=method)
    want = reference(scores, method)

    assert got.shape == want.shape
    assert got.dtype == np.float64
    # Merge heights, and the merges themselves, must agree.
    np.testing.assert_allclose(got[:, 2], want[:, 2], rtol=1e-9, atol=1e-12)
    np.testing.assert_array_equal(got[:, :2], want[:, :2])
    np.testing.assert_array_equal(got[:, 3], want[:, 3])


@pytest.mark.parametrize("method", METHODS)
def test_linkage_matches_scipy_f32(method):
    """f32 in, f32 clustering — must still land on SciPy's f64 answer."""
    scores, _ = clustered_scores(n=200, dtype=np.float32)

    got = fastcore.linkage(scores, method=method)
    want = reference(scores, method)

    np.testing.assert_allclose(got[:, 2], want[:, 2], rtol=F32_RTOL, atol=1e-6)
    np.testing.assert_array_equal(got[:, :2], want[:, :2])


@pytest.mark.parametrize("method", ["single", "complete", "average", "ward"])
def test_partitions_agree_with_scipy(method):
    """The thing users actually consume: cluster assignments, via fcluster."""
    scores, labels = clustered_scores(n=300, k=8, dtype=np.float32)
    k = 8

    got = fcluster(fastcore.linkage(scores, method=method), k, criterion="maxclust")
    want = fcluster(reference(scores, method), k, criterion="maxclust")

    # Cluster *ids* are arbitrary; the partition is what must match.
    assert _same_partition(got, want)


def _same_partition(a, b):
    return {frozenset(np.flatnonzero(a == i)) for i in np.unique(a)} == {
        frozenset(np.flatnonzero(b == i)) for i in np.unique(b)
    }


def test_linkage_recovers_known_clusters():
    """End to end: well-separated ground truth must come back out."""
    scores, labels = clustered_scores(n=300, k=6, dim=20, seed=3, dtype=np.float32)
    got = fcluster(fastcore.linkage(scores, method="average"), 6, criterion="maxclust")
    assert _same_partition(got, labels)


@pytest.mark.parametrize("symmetry", ["none", "mean", "min", "max"])
def test_linkage_symmetry_modes_match_scipy(symmetry):
    scores, _ = clustered_scores(n=150, dtype=np.float64)
    got = fastcore.linkage(scores, method="average", symmetry=symmetry)
    if symmetry == "none":
        m = 1.0 - scores
        np.fill_diagonal(m, 0.0)
        # `none` reads the upper triangle only, so mirror it to build the oracle.
        want = sp_linkage(squareform(np.triu(m) + np.triu(m, 1).T, checks=False),
                          method="average")
    else:
        want = reference(scores, "average", symmetry=symmetry)
    np.testing.assert_allclose(got[:, 2], want[:, 2], rtol=1e-9, atol=1e-12)


# ---------------------------------------------------------------------------
# condensed input
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", METHODS)
def test_linkage_from_condensed_matches_square(method):
    scores, _ = clustered_scores(n=150, dtype=np.float64)
    cond = fastcore.condensed_distances(scores)

    from_cond = fastcore.linkage(cond, method=method)
    from_square = fastcore.linkage(scores, method=method)

    np.testing.assert_allclose(from_cond, from_square, rtol=1e-12)


def test_copy_true_preserves_input():
    scores, _ = clustered_scores(n=80, dtype=np.float32)
    cond = fastcore.condensed_distances(scores)
    before = cond.copy()
    fastcore.linkage(cond, method="average", copy=True)
    np.testing.assert_array_equal(cond, before)


def test_copy_false_consumes_input():
    """Documented behaviour: in-place clustering leaves the input as scratch."""
    scores, _ = clustered_scores(n=80, dtype=np.float32)
    cond = fastcore.condensed_distances(scores)
    before = cond.copy()

    z_inplace = fastcore.linkage(cond, method="average", copy=False)
    z_copied = fastcore.linkage(before.copy(), method="average", copy=True)

    np.testing.assert_allclose(z_inplace, z_copied, rtol=1e-12)
    assert not np.array_equal(cond, before), "copy=False should have consumed the input"


def test_copy_false_rejects_readonly():
    scores, _ = clustered_scores(n=40, dtype=np.float32)
    cond = fastcore.condensed_distances(scores)
    cond.flags.writeable = False
    with pytest.raises(ValueError, match="read-only"):
        fastcore.linkage(cond, method="average", copy=False)


def test_copy_false_rejects_strided():
    scores, _ = clustered_scores(n=60, dtype=np.float32)
    cond = fastcore.condensed_distances(scores)
    with pytest.raises(ValueError, match="C-contiguous"):
        fastcore.linkage(cond[::2], method="average", copy=False)


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


def test_rejects_non_square():
    with pytest.raises(ValueError, match="square"):
        fastcore.linkage(np.zeros((3, 4), dtype=np.float32))


def test_rejects_float16():
    """float16 lacks the mantissa for thousands of accumulated merge updates."""
    scores, _ = clustered_scores(n=20, dtype=np.float32)
    with pytest.raises(TypeError, match="float32 or float64"):
        fastcore.linkage(scores.astype(np.float16))


def test_rejects_bad_condensed_length():
    with pytest.raises(ValueError, match="n\\(n-1\\)/2"):
        fastcore.linkage(np.zeros(5, dtype=np.float64), method="average")


def test_rejects_non_finite():
    scores, _ = clustered_scores(n=20, dtype=np.float64)
    scores[3, 7] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        fastcore.linkage(scores, method="average")


def test_non_finite_names_the_row():
    scores, _ = clustered_scores(n=20, dtype=np.float64)
    scores[5, 9] = np.inf
    with pytest.raises(ValueError, match="row 5"):
        fastcore.linkage(scores, method="average")


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"method": "nonesuch"}, "unknown `method`"),
        ({"symmetry": "nonesuch"}, "unknown `symmetry`"),
        ({"transform": "nonesuch"}, "unknown `transform`"),
    ],
)
def test_rejects_unknown_options(kwargs, match):
    scores, _ = clustered_scores(n=20, dtype=np.float32)
    with pytest.raises(ValueError, match=match):
        fastcore.linkage(scores, **kwargs)


def test_rejects_strided_square():
    scores, _ = clustered_scores(n=40, dtype=np.float32)
    with pytest.raises(ValueError, match="contiguous"):
        fastcore.linkage(scores[::2, ::2])


def test_rejects_3d():
    with pytest.raises(ValueError, match="3D"):
        fastcore.linkage(np.zeros((2, 2, 2), dtype=np.float32))


def test_n_cores_does_not_change_the_answer():
    scores, _ = clustered_scores(n=200, dtype=np.float32)
    a = fastcore.linkage(scores, method="ward", n_cores=1)
    b = fastcore.linkage(scores, method="ward", n_cores=4)
    np.testing.assert_array_equal(a, b)
