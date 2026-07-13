"""Match extraction: the Rust kernels vs an independent numpy reference.

The reference is written from scratch here rather than imported from navis, deliberately:
navis' numba top-N kernel seeds its buffer with `-inf` regardless of `distances`, so its
distance path selects nothing and returns all -1. Checking against it would enshrine a bug.
"""

import numpy as np
import pytest

import navis_fastcore as fastcore

DTYPES = [np.float16, np.float32, np.float64]


def make_scores(n_rows=61, n_cols=47, seed=0, dtype=np.float32):
    rng = np.random.default_rng(seed)
    # Spread over [-1, 1) so the negative-extremum branch of `percentage` is exercised.
    return (rng.random((n_rows, n_cols)) * 2 - 1).astype(dtype)


def ranked(scores, g, axis, distances, skip=-1):
    """Reference: (index, value) of one group, best first, ties by lower index."""
    group = scores[g, :] if axis == 0 else scores[:, g]
    pairs = [
        (j, v)
        for j, v in enumerate(group)
        if not np.isnan(v) and (skip < 0 or j != skip)
    ]
    # Sort by (value, index): descending value for similarities, ascending for distances.
    pairs.sort(key=lambda p: (p[1] if distances else -p[1], p[0]))
    return pairs


def ref_cutoff(best, threshold, percentage, distances):
    if threshold is not None:
        return threshold
    band = abs(best * percentage)
    return best + band if distances else best - band


# ---------------------------------------------------------------------------
# top_matches
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("distances", [False, True])
@pytest.mark.parametrize("n", [1, 3, 8])
def test_top_matches_vs_reference(dtype, axis, distances, n):
    scores = make_scores(dtype=dtype)
    idx, val = fastcore.top_matches(scores, n, axis=axis, distances=distances)

    n_groups = scores.shape[axis]
    assert idx.shape == (n_groups, n)
    assert val.shape == (n_groups, n)
    assert idx.dtype == np.int64
    assert val.dtype == dtype  # scores come back at the width they went in at

    for g in range(n_groups):
        want = ranked(scores, g, axis, distances)
        assert list(idx[g]) == [j for j, _ in want[:n]]
        np.testing.assert_array_equal(val[g], [v for _, v in want[:n]])


def test_distances_uses_min():
    """navis' numba kernel fails this: seeded with -inf, its distance path selects nothing."""
    scores = np.array([[5.0, 1.0, 9.0, 3.0]], dtype=np.float32)
    idx, val = fastcore.top_matches(scores, 2, distances=True)
    np.testing.assert_array_equal(idx, [[1, 3]])
    np.testing.assert_array_equal(val, [[1.0, 3.0]])


def test_ties_prefer_lower_index():
    scores = np.full((1, 5), 0.5, dtype=np.float32)
    idx, _ = fastcore.top_matches(scores, 3)
    np.testing.assert_array_equal(idx, [[0, 1, 2]])


def test_nan_is_never_a_match():
    nan = np.nan
    scores = np.array([[nan, nan, nan], [1.0, nan, 2.0]], dtype=np.float32)

    idx, val = fastcore.top_matches(scores, 2)
    np.testing.assert_array_equal(idx[0], [-1, -1])  # all-NaN group
    assert np.isnan(val[0]).all()
    np.testing.assert_array_equal(idx[1], [2, 0])  # NaN never selected
    np.testing.assert_array_equal(val[1], [2.0, 1.0])

    offsets, indices, _ = fastcore.matches_above(scores, threshold=0.0)
    np.testing.assert_array_equal(offsets, [0, 0, 2])  # first group empty
    np.testing.assert_array_equal(indices, [2, 0])

    # A percentage band on an all-NaN group must not compute inf - inf.
    offsets, _, _ = fastcore.matches_above(scores, percentage=0.1)
    assert offsets[1] == 0


def test_skip_self_drops_the_diagonal():
    scores = make_scores(30, 30, seed=1)
    np.fill_diagonal(scores, 10.0)  # make the self-hit unmissable

    for axis in (0, 1):
        idx, _ = fastcore.top_matches(scores, 3, axis=axis, skip_self=True)
        for g in range(30):
            assert g not in idx[g]
            assert list(idx[g]) == [j for j, _ in ranked(scores, g, axis, False, skip=g)[:3]]

        offsets, indices, _ = fastcore.matches_above(
            scores, threshold=-2.0, axis=axis, skip_self=True
        )
        for g in range(30):
            assert g not in indices[offsets[g]:offsets[g + 1]]

    cnt = fastcore.count_matches(scores, threshold=-2.0, skip_self=True)
    np.testing.assert_array_equal(cnt, 29)  # everything except the self-hit


def test_skip_self_needs_a_square_matrix():
    scores = make_scores(5, 7)
    with pytest.raises(ValueError, match="square"):
        fastcore.top_matches(scores, 1, skip_self=True)


def test_explicit_skip_array():
    scores = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    # Skip column 2 for row 0; skip nothing for row 1.
    idx, _ = fastcore.top_matches(scores, 1, skip_self=np.array([2, -1]))
    np.testing.assert_array_equal(idx.ravel(), [1, 2])


# ---------------------------------------------------------------------------
# Memory layout: the whole point is that nothing gets copied
# ---------------------------------------------------------------------------


def test_f_contiguous_view_is_free():
    """`scores.T` must take the zero-copy path, not a 40 GB transpose."""
    scores = make_scores(23, 31, seed=2)
    t = scores.T
    assert t.base is scores  # a view, not a copy
    assert t.flags.f_contiguous and not t.flags.c_contiguous

    via_view = fastcore.top_matches(t, 4, axis=0)
    via_copy = fastcore.top_matches(np.ascontiguousarray(t), 4, axis=0)
    via_axis = fastcore.top_matches(scores, 4, axis=1)

    for a, b in zip(via_view, via_copy):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(via_view, via_axis):
        np.testing.assert_array_equal(a, b)


def test_memmap_is_not_densified(tmp_path):
    path = tmp_path / "scores.dat"
    scores = make_scores(40, 40, seed=3)
    scores.tofile(path)

    mm = np.memmap(path, dtype=np.float32, mode="r", shape=(40, 40))
    idx, val = fastcore.top_matches(mm, 3)
    # Still a memmap afterwards - the kernel borrowed it rather than materialising it.
    assert isinstance(mm, np.memmap)
    np.testing.assert_array_equal(idx, fastcore.top_matches(scores, 3)[0])


def test_strided_view_is_refused_not_copied():
    scores = make_scores(10, 10)
    with pytest.raises(ValueError, match="contiguous"):
        fastcore.top_matches(scores[:, ::2], 1)


# ---------------------------------------------------------------------------
# matches_above / count_matches
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("distances", [False, True])
@pytest.mark.parametrize("crit", [{"threshold": 0.2}, {"percentage": 0.25}])
def test_matches_above_vs_reference(dtype, axis, distances, crit):
    scores = make_scores(dtype=dtype)
    offsets, indices, values = fastcore.matches_above(
        scores, axis=axis, distances=distances, **crit
    )

    n_groups = scores.shape[axis]
    assert offsets.shape == (n_groups + 1,)
    assert offsets[0] == 0
    assert offsets[-1] == len(indices) == len(values)
    assert (np.diff(offsets) >= 0).all()
    assert indices.dtype == np.uint32
    assert offsets.dtype == np.int64
    assert values.dtype == dtype

    counts = fastcore.count_matches(scores, axis=axis, distances=distances, **crit)
    np.testing.assert_array_equal(counts, np.diff(offsets))

    for g in range(n_groups):
        want = ranked(scores, g, axis, distances)
        cut = ref_cutoff(
            want[0][1], crit.get("threshold"), crit.get("percentage"), distances
        )
        want = [(j, v) for j, v in want if (v <= cut if distances else v >= cut)]

        lo, hi = offsets[g], offsets[g + 1]
        assert hi - lo == len(want), f"group {g}"
        np.testing.assert_array_equal(indices[lo:hi], [j for j, _ in want])
        np.testing.assert_array_equal(values[lo:hi], [v for _, v in want])


def test_threshold_boundary_is_inclusive():
    scores = np.array([[0.3, 0.2999999, 0.4]], dtype=np.float32)
    _, indices, _ = fastcore.matches_above(scores, threshold=0.3)
    np.testing.assert_array_equal(indices, [2, 0])


def test_percentage_with_a_negative_best():
    """`m - |m * p|`: the abs() is what widens the band the right way for a negative best."""
    scores = np.array([[-0.5, -0.54, -0.56, -2.0]], dtype=np.float32)
    _, indices, values = fastcore.matches_above(scores, percentage=0.1)  # keep >= -0.55
    np.testing.assert_array_equal(indices, [0, 1])
    np.testing.assert_allclose(values, [-0.5, -0.54])


def test_long_format_recipe():
    """The CSR return has to expand to a tidy table with no Python loop."""
    scores = np.array(
        [[1.0, 0.2, 0.9], [0.2, 1.0, 0.4], [0.9, 0.4, 1.0]], dtype=np.float32
    )
    offsets, indices, values = fastcore.matches_above(scores, threshold=0.5)

    counts = np.diff(offsets)
    query = np.repeat(np.arange(len(counts)), counts)
    rank = np.arange(len(indices)) - np.repeat(offsets[:-1], counts)

    np.testing.assert_array_equal(query, [0, 0, 1, 2, 2])
    np.testing.assert_array_equal(indices, [0, 2, 1, 2, 0])
    np.testing.assert_array_equal(rank, [0, 1, 0, 0, 1])
    np.testing.assert_allclose(values, [1.0, 0.9, 1.0, 1.0, 0.9])


def test_max_matches_guards_the_allocation():
    scores = make_scores(50, 50)
    with pytest.raises(ValueError, match="max_matches"):
        fastcore.matches_above(scores, threshold=-1.5, max_matches=10)

    # count_matches is how you find out what a cutoff will cost you first.
    assert fastcore.count_matches(scores, threshold=-1.5).sum() > 10


# ---------------------------------------------------------------------------
# Invariants and errors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("axis", [0, 1])
def test_n_cores_invariant(axis):
    # 2000 columns with 4 workers forces more than one stripe per worker.
    scores = make_scores(200, 2000, seed=4)
    a = fastcore.top_matches(scores, 5, axis=axis, n_cores=1)
    b = fastcore.top_matches(scores, 5, axis=axis, n_cores=4)
    c = fastcore.top_matches(scores, 5, axis=axis)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)
    for x, y in zip(a, c):
        np.testing.assert_array_equal(x, y)

    x = fastcore.matches_above(scores, threshold=0.9, axis=axis, n_cores=1)
    y = fastcore.matches_above(scores, threshold=0.9, axis=axis, n_cores=4)
    for a_, b_ in zip(x, y):
        np.testing.assert_array_equal(a_, b_)


def test_errors():
    scores = make_scores(5, 8)

    with pytest.raises(ValueError, match="exactly one"):
        fastcore.matches_above(scores)
    with pytest.raises(ValueError, match="exactly one"):
        fastcore.matches_above(scores, threshold=0.5, percentage=0.1)
    with pytest.raises(ValueError, match="axis"):
        fastcore.top_matches(scores, 1, axis=2)
    with pytest.raises(ValueError, match="`n` must be"):
        fastcore.top_matches(scores, 0)
    with pytest.raises(ValueError, match="`n` must be"):
        fastcore.top_matches(scores, 9)  # only 8 candidates
    with pytest.raises(ValueError, match="percentage"):
        fastcore.matches_above(scores, percentage=1.5)
    with pytest.raises(TypeError, match="float16, float32 or float64"):
        fastcore.top_matches(scores.astype(np.int64), 1)
    with pytest.raises(ValueError, match="2-dimensional"):
        fastcore.top_matches(scores.ravel(), 1)


def test_accepts_a_dataframe():
    pd = pytest.importorskip("pandas")
    scores = make_scores(6, 6, seed=5)
    df = pd.DataFrame(scores, index=list("abcdef"), columns=list("abcdef"))
    idx, val = fastcore.top_matches(df, 2)
    np.testing.assert_array_equal(idx, fastcore.top_matches(scores, 2)[0])
