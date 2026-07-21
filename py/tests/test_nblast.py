"""NBLAST tests: the Rust shull+aann pipeline vs an independent numpy reference."""
import csv as _csv
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.spatial import cKDTree

import navis_fastcore as fastcore

# `navis_fastcore.nblast` resolves to the re-exported *function*, not the module,
# so the private helpers are imported by name.
from navis_fastcore.nblast import (
    Dotprop,
    Synapses,
    _combine,
    _symmetrize,
    _tangents_and_alpha,
)

DATA = Path(__file__).parent
SMAT_DIR = DATA.parent.parent / "fastcore" / "fastcore.data"


def load_dotprop(name, k=5):
    swc = pd.read_csv(DATA / name, comment="#", header=None, sep=r"\s+")
    xyz = swc[[2, 3, 4]].values.astype(np.float64) / 1000  # nm -> um
    vect, alpha = _tangents_and_alpha(xyz, k=k)
    return Dotprop(xyz, vect, alpha)


def load_smat_csv(alpha=False):
    """Parse an embedded FCWB matrix into (values, left dist edges, left dot edges)."""
    fname = "smat_alpha_fcwb.csv" if alpha else "smat_fcwb.csv"
    with open(SMAT_DIR / fname) as f:
        rows = list(_csv.reader(f))
    left = lambda label: float(label.split(",")[0].lstrip("("))
    dot_edges = np.array([left(c) for c in rows[0][1:]])
    dist_edges, values = [], []
    for r in rows[1:]:
        dist_edges.append(left(r[0]))
        values.append([float(x) for x in r[1:]])
    return np.array(values), np.array(dist_edges), np.array(dot_edges)


def digitize(edges, v):
    """Rust `digitize` in numpy: (#edges <= v) - 1, clamped to [0, len-1]."""
    b = np.searchsorted(edges, v, side="right") - 1
    return np.clip(b, 0, len(edges) - 1)


def reference_scores(
    q_dps, t_dps, values, dist_edges, dot_edges,
    normalize=True, use_alpha=False, limit_dist=None,
):
    """Exact-NN (scipy) + FCWB scoring reference; row = query, col = target.

    Mirrors navis: `use_alpha` scales the dot product by sqrt(alpha_q * alpha_t);
    `limit_dist` caps over-bound points at (dist=limit, dot=0).
    """
    qp, qv, qa = [d.points for d in q_dps], [d.vect for d in q_dps], [d.alpha for d in q_dps]
    tp, tv, ta = [d.points for d in t_dps], [d.vect for d in t_dps], [d.alpha for d in t_dps]
    trees = [cKDTree(p) for p in tp]

    if use_alpha:
        self_hit = [values[0, digitize(dot_edges, qa[i])].sum() for i in range(len(q_dps))]
    else:
        self_hit = [len(p) * values[0, -1] for p in qp]

    M = np.empty((len(q_dps), len(t_dps)), dtype=np.float64)
    for i in range(len(q_dps)):
        for j in range(len(t_dps)):
            if limit_dist:
                d, idx = trees[j].query(qp[i], distance_upper_bound=limit_dist)
                over = d == np.inf
                d = np.where(over, limit_dist, d)
                idx = np.where(over, 0, idx)
            else:
                d, idx = trees[j].query(qp[i])
                over = np.zeros(len(d), dtype=bool)
            dots = np.abs((qv[i] * tv[j][idx]).sum(axis=1))
            if use_alpha:
                dots = dots * np.sqrt(qa[i] * ta[j][idx])
            dots = np.where(over, 0.0, dots)
            raw = values[digitize(dist_edges, d), digitize(dot_edges, dots)].sum()
            M[i, j] = raw / self_hit[i] if normalize else raw
    return M


@pytest.fixture(scope="module")
def dotprops():
    return [load_dotprop("722817260.swc"), load_dotprop("754534424.swc")]


def _navis_dotprops(navis, dotprops, use_alpha=False):
    """navis Dotprops reusing our points/vect (units set so nblast preflight is happy)."""
    return [
        navis.core.Dotprops(
            d.points,
            k=None,
            vect=d.vect,
            alpha=d.alpha if use_alpha else None,
            units="1 micrometer",
        )
        for d in dotprops
    ]


def test_allbyall_matches_reference(dotprops):
    values, de, ve = load_smat_csv()
    M_fast = np.asarray(fastcore.nblast_allbyall(dotprops))
    M_ref = reference_scores(dotprops, dotprops, values, de, ve)

    # k=1 on a Delaunay graph is exact, so only float32 rounding should remain.
    assert np.allclose(np.diag(M_fast), 1.0, atol=1e-5)
    assert np.abs(M_fast - M_ref).max() < 1e-4, np.abs(M_fast - M_ref).max()


def test_query_target_matches_allbyall(dotprops):
    M_all = np.asarray(fastcore.nblast_allbyall(dotprops))
    M_qt = np.asarray(fastcore.nblast(dotprops, dotprops))
    assert np.allclose(M_all, M_qt, atol=1e-6)


def test_symmetry_mean(dotprops):
    M = np.asarray(fastcore.nblast_allbyall(dotprops))
    M_mean = np.asarray(fastcore.nblast_allbyall(dotprops, symmetry="mean"))
    assert np.allclose(M_mean, (M + M.T) / 2, atol=1e-6)


def test_symmetry_forward_is_noop(dotprops):
    M = np.asarray(fastcore.nblast_allbyall(dotprops))
    M_fwd = np.asarray(fastcore.nblast_allbyall(dotprops, symmetry="forward"))
    assert np.array_equal(M, M_fwd)


@pytest.mark.parametrize("symmetry", ["min", "max"])
def test_symmetry_minmax_allbyall(dotprops, symmetry):
    """All-by-all min/max, which the mean-only test above leaves uncovered."""
    M = np.asarray(fastcore.nblast_allbyall(dotprops))
    got = np.asarray(fastcore.nblast_allbyall(dotprops, symmetry=symmetry))
    want = (np.minimum if symmetry == "min" else np.maximum)(M, M.T)
    assert np.allclose(got, want, atol=1e-6)


# ---------------------------------------------------------------------------
# _symmetrize / _combine
#
# Both were rewritten to stop allocating n x n temporaries. The contract is that
# they produce *exactly* what the expressions they replaced produced, so these pin
# them bit-for-bit rather than approximately.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "symmetry, naive",
    [
        ("mean", lambda a, b: (a + b) / 2),
        (True, lambda a, b: (a + b) / 2),
        ("min", np.minimum),
        ("max", np.maximum),
    ],
)
def test_symmetrize_matches_naive_expression(symmetry, naive, dtype):
    rng = np.random.default_rng(0)
    M = np.ascontiguousarray(rng.random((37, 37)), dtype=dtype)
    want = naive(M, M.T)

    got = _symmetrize(M, symmetry)

    np.testing.assert_array_equal(got, want)
    assert got is M, "_symmetrize must work in place"
    np.testing.assert_array_equal(got, got.T, err_msg="result must be symmetric")


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "symmetry, naive",
    [
        ("mean", lambda a, b: (a + b) / 2),
        (True, lambda a, b: (a + b) / 2),
        ("min", np.minimum),
        ("max", np.maximum),
    ],
)
def test_combine_matches_naive_expression(symmetry, naive, dtype):
    rng = np.random.default_rng(1)
    A = np.ascontiguousarray(rng.random((11, 23)), dtype=dtype)
    B = np.ascontiguousarray(rng.random((23, 11)), dtype=dtype)
    want = naive(A, B.T)

    got = _combine(A, B.T, symmetry)

    np.testing.assert_array_equal(got, want)
    assert got is A, "_combine must consume its first argument in place"


def test_symmetrize_leaves_diagonal_alone():
    """NBLAST puts self-scores on the diagonal; symmetrising must not touch them."""
    rng = np.random.default_rng(2)
    M = np.ascontiguousarray(rng.random((16, 16)), dtype=np.float32)
    diag = np.diag(M).copy()
    _symmetrize(M, "mean")
    np.testing.assert_array_equal(np.diag(M), diag)


@pytest.mark.parametrize("bad", ["nonesuch", 0.5, "Mean"])
def test_unknown_symmetry_raises(bad):
    rng = np.random.default_rng(3)
    M = np.ascontiguousarray(rng.random((8, 8)), dtype=np.float32)
    with pytest.raises(ValueError, match="Unknown symmetry"):
        _symmetrize(M, bad)
    with pytest.raises(ValueError, match="Unknown symmetry"):
        _combine(M, M.copy(), bad)


def test_nblast_symmetry_rectangular(dotprops):
    """Rectangular symmetry combines forward with the reverse's transpose."""
    q, t = dotprops[:1], dotprops
    F = np.asarray(fastcore.nblast(q, t))
    R = np.asarray(fastcore.nblast(t, q))
    for sym, expect in (
        ("mean", (F + R.T) / 2),
        ("min", np.minimum(F, R.T)),
        ("max", np.maximum(F, R.T)),
    ):
        M = np.asarray(fastcore.nblast(q, t, symmetry=sym))
        assert np.allclose(M, expect, atol=1e-6), sym


def test_custom_smat_tuple_matches_default(dotprops):
    """Passing the embedded matrix explicitly must equal the default path."""
    values, de, ve = load_smat_csv()
    M_default = np.asarray(fastcore.nblast_allbyall(dotprops))
    M_custom = np.asarray(fastcore.nblast_allbyall(dotprops, smat=(values, de, ve)))
    assert np.allclose(M_default, M_custom, atol=1e-6)


def test_use_alpha_matches_reference(dotprops):
    # use_alpha auto-selects the alpha-calibrated FCWB matrix (as navis does).
    values, de, ve = load_smat_csv(alpha=True)
    M = np.asarray(fastcore.nblast_allbyall(dotprops, use_alpha=True))
    R = reference_scores(dotprops, dotprops, values, de, ve, use_alpha=True)
    assert np.allclose(np.diag(M), 1.0, atol=1e-4)
    assert np.abs(M - R).max() < 2e-3, np.abs(M - R).max()


def test_use_alpha_changes_scores(dotprops):
    plain = np.asarray(fastcore.nblast_allbyall(dotprops))
    alpha = np.asarray(fastcore.nblast_allbyall(dotprops, use_alpha=True))
    assert not np.allclose(plain, alpha)


def test_use_alpha_requires_alpha(dotprops):
    bare = [Dotprop(d.points, d.vect) for d in dotprops]  # alpha defaults to None
    with pytest.raises(ValueError):
        fastcore.nblast_allbyall(bare, use_alpha=True)


def test_limit_dist_matches_reference(dotprops):
    values, de, ve = load_smat_csv()
    # These two neurons nearly overlap (max NN dist < 1 um), so the bound has to
    # be sub-micron to actually clip points and exercise the over-limit path.
    limit = 0.3
    M = np.asarray(fastcore.nblast_allbyall(dotprops, limit_dist=limit))
    R = reference_scores(dotprops, dotprops, values, de, ve, limit_dist=limit)
    plain = np.asarray(fastcore.nblast_allbyall(dotprops))
    assert np.allclose(np.diag(M), 1.0, atol=1e-4)
    assert np.abs(M - R).max() < 2e-3, np.abs(M - R).max()
    assert not np.allclose(M, plain)  # the bound must actually clip something


def test_limit_dist_auto_matches_numeric(dotprops):
    _, de, _ = load_smat_csv()
    M_auto = np.asarray(fastcore.nblast_allbyall(dotprops, limit_dist="auto"))
    M_num = np.asarray(fastcore.nblast_allbyall(dotprops, limit_dist=de[-1] * 1.05))
    assert np.array_equal(M_auto, M_num)


def test_precision_dtypes(dotprops):
    M32 = np.asarray(fastcore.nblast_allbyall(dotprops, precision=32))
    M64 = np.asarray(fastcore.nblast_allbyall(dotprops, precision=64))
    M16 = np.asarray(fastcore.nblast_allbyall(dotprops, precision=16))
    assert M32.dtype == np.float32
    assert M64.dtype == np.float64
    assert M16.dtype == np.float16
    assert np.allclose(M32, M64, atol=1e-5)


def test_n_cores_invariant(dotprops):
    M = np.asarray(fastcore.nblast_allbyall(dotprops))
    M1 = np.asarray(fastcore.nblast_allbyall(dotprops, n_cores=1))
    assert np.array_equal(M, M1)


# --- float32 coordinates ---------------------------------------------------
#
# Coordinates are used at the dtype they arrive as: float32 dotprops build a
# float32 spatial index, which roughly halves the resident set of a large run.
#
# Scores stay equivalent but not identical. A score is a sum of lookups from a
# coarsely binned matrix, so most of the ~1e-7 relative shift in the distances
# changes nothing at all; what does move the total is the handful of points whose
# nearest neighbour flips, or whose distance crosses a bin edge. On these two
# neurons (4.3k/4.7k points, coordinates up to ~37) that comes to 8.5e-5 relative
# without alpha and 1.3e-4 with it, so 1e-3 leaves roughly an order of margin.
F32_RTOL, F32_ATOL = 1e-3, 1e-5


def _as32(dps):
    return [
        Dotprop(
            np.ascontiguousarray(d.points, dtype=np.float32),
            np.ascontiguousarray(d.vect, dtype=np.float32),
            None if d.alpha is None else d.alpha,
        )
        for d in dps
    ]


def _widen(dps):
    """Explicitly float64 copies - the coordinates a float32 input falls back to."""
    return [
        Dotprop(
            np.ascontiguousarray(d.points, dtype=np.float64),
            np.ascontiguousarray(d.vect, dtype=np.float64),
            None if d.alpha is None else d.alpha,
        )
        for d in dps
    ]


def test_f32_coords_reach_rust_unconverted(dotprops):
    """The whole point: nothing upcasts on the way in."""
    from navis_fastcore.nblast import _as_clouds, _coord_dtype

    dps32 = _as32(dotprops)
    assert _coord_dtype(dps32) == np.float32
    assert _coord_dtype(dotprops) == np.float64
    points, vects, _ = _as_clouds(dps32)
    assert points[0].dtype == np.float32 and vects[0].dtype == np.float32
    # And the buffer is the caller's, not a converted copy - a copy would cost the
    # memory the narrow width was chosen to save.
    assert points[0] is dps32[0].points


def test_f32_coords_match_f64(dotprops):
    M64 = np.asarray(fastcore.nblast_allbyall(dotprops, precision=64))
    M32 = np.asarray(fastcore.nblast_allbyall(_as32(dotprops), precision=64))
    assert M32.shape == M64.shape
    assert np.allclose(M32, M64, rtol=F32_RTOL, atol=F32_ATOL)
    # Normalisation is width-independent, so the diagonal is exactly 1 either way.
    assert np.allclose(np.diag(M32), 1.0, atol=1e-12)


def test_f32_coords_match_f64_query_target(dotprops):
    Q64 = np.asarray(fastcore.nblast(dotprops, dotprops, precision=64))
    Q32 = np.asarray(fastcore.nblast(_as32(dotprops), _as32(dotprops), precision=64))
    assert np.allclose(Q32, Q64, rtol=F32_RTOL, atol=F32_ATOL)


def test_f32_coords_match_f64_knn(dotprops):
    dps = dotprops * 4  # 8 neurons, so k=3 has something to choose between
    i64, s64 = fastcore.nblast_knn(dps, k=3, n_candidates=8, precision=64)
    i32, s32 = fastcore.nblast_knn(_as32(dps), k=3, n_candidates=8, precision=64)
    assert i32.shape == i64.shape
    for r64, r32 in zip(i64, i32):
        assert len(set(r64) & set(r32)) >= len(r64) - 1
    assert np.allclose(s32, s64, rtol=F32_RTOL, atol=F32_ATOL)


def test_f32_coords_use_alpha(dotprops):
    """Alpha is a score-side weight and stays float64 even at float32 coordinates."""
    M64 = fastcore.nblast_allbyall(dotprops, use_alpha=True, precision=64)
    M32 = fastcore.nblast_allbyall(_as32(dotprops), use_alpha=True, precision=64)
    assert np.allclose(np.asarray(M32), np.asarray(M64), rtol=F32_RTOL, atol=F32_ATOL)


def test_f32_coords_limit_dist(dotprops):
    """`limit_dist` compares against the bound in float64 whatever the storage is."""
    M64 = fastcore.nblast_allbyall(dotprops, limit_dist="auto", precision=64)
    M32 = fastcore.nblast_allbyall(_as32(dotprops), limit_dist="auto", precision=64)
    assert np.allclose(np.asarray(M32), np.asarray(M64), rtol=F32_RTOL, atol=F32_ATOL)


def test_mixed_coord_dtypes_fall_back_to_f64(dotprops):
    """A half-float32 set must not error, and must run wholly at float64.

    Widening the float32 side is the right call: the two sides have to agree on a
    width, and widening is lossless where narrowing is not. Note the comparison is
    against *explicitly widened* clouds rather than the originals - the float32 side
    has already lost precision, and running it at float64 does not restore it.
    """
    mixed = [_as32(dotprops)[0], dotprops[1]]
    got = np.asarray(fastcore.nblast_allbyall(mixed, precision=64))
    want = np.asarray(fastcore.nblast_allbyall(_widen(mixed), precision=64))
    assert np.array_equal(got, want)


def test_f32_coords_across_query_and_target_agree(dotprops):
    """float32 query against float64 target: one width is chosen for both sides."""
    q32 = _as32(dotprops)
    got = np.asarray(fastcore.nblast(q32, dotprops, precision=64))
    want = np.asarray(fastcore.nblast(_widen(q32), dotprops, precision=64))
    assert np.array_equal(got, want)


def test_navis_lookup2d_smat(dotprops):
    """A navis Lookup2d passed as `smat` must round-trip to the default path."""
    navis = pytest.importorskip("navis")
    sf = navis.nbl.smat.smat_fcwb(alpha=False)  # identical to our embedded FCWB
    M_default = np.asarray(fastcore.nblast_allbyall(dotprops))
    M_lookup = np.asarray(fastcore.nblast_allbyall(dotprops, smat=sf))
    assert np.allclose(M_default, M_lookup, atol=1e-6)


def test_parity_vs_navis(dotprops):
    navis = pytest.importorskip("navis")
    dp = _navis_dotprops(navis, dotprops)
    M_navis = navis.nblast_allbyall(dp, progress=False).to_numpy()
    M_fast = np.asarray(fastcore.nblast_allbyall(dotprops))
    off = ~np.eye(len(dotprops), dtype=bool)
    assert np.abs(M_fast - M_navis)[off].max() < 1e-3, np.abs(M_fast - M_navis)[off].max()


def test_parity_vs_navis_use_alpha(dotprops):
    navis = pytest.importorskip("navis")
    dp = _navis_dotprops(navis, dotprops, use_alpha=True)
    M_navis = navis.nblast_allbyall(dp, use_alpha=True, progress=False).to_numpy()
    M_fast = np.asarray(fastcore.nblast_allbyall(dotprops, use_alpha=True))
    off = ~np.eye(len(dotprops), dtype=bool)
    assert np.abs(M_fast - M_navis)[off].max() < 2e-3, np.abs(M_fast - M_navis)[off].max()


def test_parity_vs_navis_limit_dist(dotprops):
    navis = pytest.importorskip("navis")
    dp = _navis_dotprops(navis, dotprops)
    M_navis = navis.nblast_allbyall(dp, limit_dist=0.3, progress=False).to_numpy()
    M_fast = np.asarray(fastcore.nblast_allbyall(dotprops, limit_dist=0.3))
    off = ~np.eye(len(dotprops), dtype=bool)
    assert np.abs(M_fast - M_navis)[off].max() < 2e-3, np.abs(M_fast - M_navis)[off].max()


# --- smart NBLAST -----------------------------------------------------------


def test_nblast_pairs_matches_dense(dotprops):
    """The sparse `nblast_pairs` primitive equals the dense matrix at those cells."""
    from navis_fastcore import _fastcore

    pts = [d.points for d in dotprops]
    vec = [d.vect for d in dotprops]
    dense = np.asarray(fastcore.nblast(dotprops, dotprops, precision=64))
    qi = np.array([0, 0, 1, 1], dtype=np.int64)
    tj = np.array([0, 1, 0, 1], dtype=np.int64)
    s = np.asarray(_fastcore.nblast_pairs(pts, vec, pts, vec, qi, tj, precision=64))
    assert np.allclose(s, dense[qi, tj], atol=1e-6)


def test_smart_all_selected_matches_full(dotprops):
    # t=0 percentile selects every cell, so smart == the full all-by-all.
    dense = np.asarray(fastcore.nblast_allbyall(dotprops, precision=64))
    out, mask = fastcore.nblast_smart(
        dotprops, t=0, criterion="percentile", precision=64, return_mask=True
    )
    assert mask.all()
    assert np.allclose(out, dense, atol=1e-6)


def test_smart_symmetry_mean_all_selected(dotprops):
    dense = np.asarray(fastcore.nblast_allbyall(dotprops, symmetry="mean", precision=64))
    out = np.asarray(fastcore.nblast_smart(dotprops, t=0, symmetry="mean", precision=64))
    assert np.allclose(out, dense, atol=1e-6)


def test_smart_criterion_N_count(dotprops):
    _, mask = fastcore.nblast_smart(dotprops, t=1, criterion="N", return_mask=True)
    assert (mask.sum(axis=1) == 1).all()


def test_smart_unselected_keep_coarse(dotprops):
    # top-1 per row is the self-match (diagonal); off-diagonal stays coarse.
    out, mask = fastcore.nblast_smart(
        dotprops, t=1, criterion="N", downsample=2, precision=64, return_mask=True
    )
    dq = [Dotprop(d.points[::2], d.vect[::2]) for d in dotprops]
    coarse = np.asarray(fastcore.nblast_allbyall(dq, precision=64))
    unsel = ~mask
    assert unsel.any()
    assert np.allclose(out[unsel], coarse[unsel], atol=1e-6)


def test_smart_return_mask_shape(dotprops):
    res = fastcore.nblast_smart(dotprops, return_mask=True)
    assert isinstance(res, tuple) and len(res) == 2
    out, mask = res
    assert out.shape == (len(dotprops), len(dotprops))
    assert mask.shape == out.shape
    assert mask.dtype == bool


def test_smart_query_target_rectangular(dotprops):
    # t=0 selects all -> smart query/target == the full rectangular nblast.
    q, t = dotprops[:1], dotprops
    dense = np.asarray(fastcore.nblast(q, t, precision=64))
    out = np.asarray(fastcore.nblast_smart(q, t, t=0, precision=64))
    assert out.shape == (1, len(dotprops))
    assert np.allclose(out, dense, atol=1e-6)


def test_smart_parity_vs_navis(dotprops):
    navis = pytest.importorskip("navis")
    dp = _navis_dotprops(navis, dotprops)
    ids = [d.id for d in dp]
    off = ~np.eye(len(dotprops), dtype=bool)
    n = len(dotprops)

    # Selecting every cell reduces smart to a full NBLAST in both engines. We ask for
    # that with criterion "N" (t = all targets): navis only accepts 0 < t < 100 for
    # "percentile", so the t=0 that selects everything here is rejected there.
    res = navis.nblast_smart(dp, t=n, criterion="N", progress=False)
    M_navis = res.loc[ids, ids].to_numpy()
    M_fast = np.asarray(fastcore.nblast_smart(dotprops, t=n, criterion="N", precision=64))
    assert np.abs(M_fast - M_navis)[off].max() < 2e-3, np.abs(M_fast - M_navis)[off].max()

    # With the default percentile only the top cells per row are recomputed at full
    # resolution and the rest keep their coarse score, so this also pins the pre-pass
    # (downsampling) and the selection to navis'.
    res = navis.nblast_smart(dp, t=90, progress=False)
    M_navis = res.loc[ids, ids].to_numpy()
    M_fast = np.asarray(fastcore.nblast_smart(dotprops, t=90, precision=64))
    assert np.abs(M_fast - M_navis)[off].max() < 2e-3, np.abs(M_fast - M_navis)[off].max()


# --- syNBLAST ---------------------------------------------------------------


@pytest.fixture(scope="module")
def synapses(dotprops):
    """Synthetic connectors from real neuron geometry: every node is a synapse with
    a deterministic 0/1 type, packed as an (N, 4) [x, y, z, type] array."""
    rng = np.random.default_rng(42)
    out = []
    for d in dotprops:
        xyz = np.ascontiguousarray(d.points, dtype=np.float64)
        ty = rng.integers(0, 2, size=(len(xyz), 1)).astype(np.float64)
        out.append(Synapses(np.hstack([xyz, ty])))
    return out


def _extract_connectors(neurons, by_type, cn_types=None):
    pts, types = [], []
    for n in neurons:
        cn = np.asarray(getattr(n, "connectors", n), dtype=np.float64)
        ty = np.rint(cn[:, 3]).astype(np.int64) if cn.shape[1] >= 4 else np.zeros(len(cn), np.int64)
        p = cn[:, :3]
        if cn_types is not None:
            sel = np.isin(ty, np.asarray(cn_types))
            p, ty = p[sel], ty[sel]
        if not by_type:
            ty = np.zeros(len(p), np.int64)
        pts.append(p)
        types.append(ty)
    return pts, types


def reference_synblast(
    neurons_q, neurons_t, values, dist_edges, dot_edges,
    by_type=False, cn_types=None, normalize=True,
):
    """Exact-NN (scipy), same-type-only syNBLAST reference; row = query, col = target.

    Scores each nearest-neighbour distance in the matrix's dot=1 column (synapses
    have no tangent). Query connectors whose type is absent in the target take the
    worst (farthest) bin.
    """
    qp, qt = _extract_connectors(neurons_q, by_type, cn_types)
    tp, tt = _extract_connectors(neurons_t, by_type, cn_types)
    dot_col = int(digitize(dot_edges, 1.0))
    worst = values[-1, dot_col]
    self_hit = [len(p) * values[0, dot_col] for p in qp]

    M = np.empty((len(qp), len(tp)), dtype=np.float64)
    for i in range(len(qp)):
        for j in range(len(tp)):
            raw = 0.0
            for ty in np.unique(qt[i]):
                qm = qt[i] == ty
                tm = tt[j] == ty
                if tm.any():
                    d, _ = cKDTree(tp[j][tm]).query(qp[i][qm])
                    raw += values[digitize(dist_edges, d), dot_col].sum()
                else:
                    raw += worst * qm.sum()
            M[i, j] = raw / self_hit[i] if normalize else raw
    return M


def test_synblast_matches_reference(synapses):
    values, de, ve = load_smat_csv()
    M_fast = np.asarray(fastcore.synblast(synapses, precision=64))
    M_ref = reference_synblast(synapses, synapses, values, de, ve)
    assert np.allclose(np.diag(M_fast), 1.0, atol=1e-6)
    assert np.abs(M_fast - M_ref).max() < 1e-3, np.abs(M_fast - M_ref).max()


def test_synblast_by_type_matches_reference(synapses):
    values, de, ve = load_smat_csv()
    M_fast = np.asarray(fastcore.synblast(synapses, by_type=True, precision=64))
    M_ref = reference_synblast(synapses, synapses, values, de, ve, by_type=True)
    assert np.abs(M_fast - M_ref).max() < 1e-3, np.abs(M_fast - M_ref).max()


def test_synblast_by_type_differs_from_pooled(synapses):
    pooled = np.asarray(fastcore.synblast(synapses, precision=64))
    by_type = np.asarray(fastcore.synblast(synapses, by_type=True, precision=64))
    # Restricting matches to same-type synapses must change off-diagonal scores.
    off = ~np.eye(len(synapses), dtype=bool)
    assert not np.allclose(pooled[off], by_type[off])


def test_synblast_diagonal_is_one(synapses):
    M = np.asarray(fastcore.synblast(synapses, precision=64))
    assert np.allclose(np.diag(M), 1.0, atol=1e-6)


def test_synblast_query_target_matches_allbyall(synapses):
    M_all = np.asarray(fastcore.synblast(synapses, precision=64))
    M_qt = np.asarray(fastcore.synblast(synapses, synapses, precision=64))
    assert np.allclose(M_all, M_qt, atol=1e-5)


def test_synblast_symmetry_mean(synapses):
    M = np.asarray(fastcore.synblast(synapses, precision=64))
    M_mean = np.asarray(fastcore.synblast(synapses, symmetry="mean", precision=64))
    assert np.allclose(M_mean, (M + M.T) / 2, atol=1e-6)


def test_synblast_symmetry_rectangular(synapses):
    q, t = synapses[:1], synapses
    F = np.asarray(fastcore.synblast(q, t, precision=64))
    R = np.asarray(fastcore.synblast(t, q, precision=64))
    for sym, expect in (
        ("mean", (F + R.T) / 2),
        ("min", np.minimum(F, R.T)),
        ("max", np.maximum(F, R.T)),
    ):
        M = np.asarray(fastcore.synblast(q, t, symmetry=sym, precision=64))
        assert np.allclose(M, expect, atol=1e-6), sym


def test_synblast_cn_types_filter(synapses):
    values, de, ve = load_smat_csv()
    M_fast = np.asarray(fastcore.synblast(synapses, cn_types=[0], precision=64))
    M_ref = reference_synblast(synapses, synapses, values, de, ve, cn_types=[0])
    assert np.abs(M_fast - M_ref).max() < 1e-3, np.abs(M_fast - M_ref).max()


def test_synblast_missing_type_worst_score():
    values, de, ve = load_smat_csv()
    dot_col = int(digitize(ve, 1.0))
    self_p, worst = values[0, dot_col], values[-1, dot_col]
    # Two type-0 connectors co-located with the target, one type-1 absent in target.
    q = Synapses(np.array([[0, 0, 0, 0], [0, 0, 0, 0], [9, 9, 9, 1]], dtype=float))
    t = Synapses(np.array([[0, 0, 0, 0], [0, 0, 0, 0]], dtype=float))
    raw = np.asarray(fastcore.synblast([q], [t], by_type=True, normalize=False, precision=64))
    assert np.isclose(raw[0, 0], 2 * self_p + worst, atol=1e-6)


def test_synblast_custom_smat_matches_default(synapses):
    values, de, ve = load_smat_csv()
    M_default = np.asarray(fastcore.synblast(synapses, precision=64))
    M_custom = np.asarray(fastcore.synblast(synapses, smat=(values, de, ve), precision=64))
    assert np.allclose(M_default, M_custom, atol=1e-6)


def test_synblast_precision_dtypes(synapses):
    M32 = np.asarray(fastcore.synblast(synapses, precision=32))
    M64 = np.asarray(fastcore.synblast(synapses, precision=64))
    M16 = np.asarray(fastcore.synblast(synapses, precision=16))
    assert M32.dtype == np.float32
    assert M64.dtype == np.float64
    assert M16.dtype == np.float16
    assert np.allclose(M32, M64, atol=1e-5)


def test_synblast_by_type_requires_type_column():
    xyz_only = [Synapses(np.random.rand(10, 3)) for _ in range(2)]
    with pytest.raises(ValueError):
        fastcore.synblast(xyz_only, by_type=True)


def test_synblast_n_cores_invariant(synapses):
    M = np.asarray(fastcore.synblast(synapses, precision=64))
    M1 = np.asarray(fastcore.synblast(synapses, n_cores=1, precision=64))
    assert np.array_equal(M, M1)


def test_synblast_parity_vs_navis(synapses):
    navis = pytest.importorskip("navis")
    # Build minimal navis neurons carrying our synapses as connectors (in microns).
    try:
        neurons = []
        for i, s in enumerate(synapses):
            cn = s.connectors
            nodes = pd.DataFrame(
                {
                    "node_id": [0],
                    "parent_id": [-1],
                    "x": [0.0],
                    "y": [0.0],
                    "z": [0.0],
                    "radius": [1.0],
                }
            )
            n = navis.TreeNeuron(nodes, units="1 micrometer", id=i)
            n.connectors = pd.DataFrame(
                {
                    "connector_id": np.arange(len(cn)),
                    "x": cn[:, 0],
                    "y": cn[:, 1],
                    "z": cn[:, 2],
                    "type": cn[:, 3].astype(int),
                    "node_id": 0,
                }
            )
            neurons.append(n)
        nl = navis.NeuronList(neurons)
        M_navis = navis.synblast(
            nl, nl, by_type=True, normalized=True, progress=False
        ).to_numpy()
    except Exception as e:  # pragma: no cover - depends on navis internals/version
        pytest.skip(f"could not build navis synblast reference: {e}")

    M_fast = np.asarray(fastcore.synblast(synapses, by_type=True, precision=64))
    off = ~np.eye(len(synapses), dtype=bool)
    assert np.abs(M_fast - M_navis)[off].max() < 2e-3, np.abs(M_fast - M_navis)[off].max()


# ---------------------------------------------------------------------------
# nblast_knn
# ---------------------------------------------------------------------------

#: A coarse-enough signature grid that every neuron shares the single feature, so
#: the candidate stage is exhaustive and `nblast_knn` must reproduce the dense
#: top-k exactly. This is the correctness anchor for the approximate pipeline.
EXHAUSTIVE = dict(voxel=1e6, n_dirs=1, splat=False)


@pytest.fixture(scope="module")
def knn_dotprops(dotprops):
    """A population with genuine near-neighbours: jittered copies of the real pair."""
    rng = np.random.default_rng(0)
    out = []
    for d in dotprops:
        out.append(d)
        for _ in range(4):
            pts = d.points + rng.normal(scale=0.5, size=d.points.shape)
            out.append(Dotprop(np.ascontiguousarray(pts), d.vect, d.alpha))
    return out


def dense_topk(M, k, symmetry="mean"):
    """Top-k of a dense score matrix under `symmetry`, as (idx, scores)."""
    if symmetry == "mean":
        S = (M + M.T) / 2
    elif symmetry == "min":
        S = np.minimum(M, M.T)
    elif symmetry == "max":
        S = np.maximum(M, M.T)
    else:
        S = M.copy()
    np.fill_diagonal(S, -np.inf)
    idx = np.argsort(-S, axis=1)[:, :k]
    return idx, np.take_along_axis(S, idx, axis=1)


@pytest.mark.parametrize("symmetry", ["mean", "forward", "min", "max"])
def test_knn_exhaustive_matches_allbyall(knn_dotprops, symmetry):
    n, k = len(knn_dotprops), 4
    M = fastcore.nblast_allbyall(knn_dotprops, precision=64)
    want_idx, want_sc = dense_topk(M, k, symmetry)
    idx, sc = fastcore.nblast_knn(
        knn_dotprops, k=k, n_candidates=n - 1, symmetry=symmetry,
        precision=64, **EXHAUSTIVE,
    )
    assert np.allclose(sc, want_sc, atol=1e-12)
    # Indices may legitimately swap on an exact tie; scores may not.
    distinct = np.diff(want_sc, axis=1, prepend=np.inf) != 0
    assert np.array_equal(idx[distinct], want_idx[distinct])


def test_knn_scores_are_exact_even_when_approximate(knn_dotprops):
    """Whatever survives shortlisting is scored by the same kernel as the dense path."""
    k = 3
    M = fastcore.nblast_allbyall(knn_dotprops, precision=64)
    S = (M + M.T) / 2
    idx, sc = fastcore.nblast_knn(knn_dotprops, k=k, n_candidates=2, precision=64)
    for i in range(len(knn_dotprops)):
        for c in range(k):
            if idx[i, c] >= 0:
                assert abs(sc[i, c] - S[i, idx[i, c]]) < 1e-12


@pytest.mark.parametrize("seed", range(8))
def test_knn_matches_nblast_on_tied_neighbours(seed):
    """Grid-quantised clouds: `nblast_knn` and `nblast` must resolve NN ties alike.

    The two paths hand different query sets to the aann index (`nblast` all clouds
    at once, `nblast_knn` one candidate subset per target), which yields a
    different Morton order and so a different descent seed. While aann broke ties
    by keeping the incumbent, that made the winner among *exactly* equidistant
    target points path-dependent, and the two disagreed by ~1e-3 here. aann 0.2.1
    resolves ties to the lowest vertex index instead.

    Continuous coordinates essentially never tie, so this needs the grid snapping
    below - it is what makes real (resampled/voxelised) neurons tie readily.
    """
    rng = np.random.default_rng(seed)

    def cloud(n=400, step=2.0):
        p = np.round(np.cumsum(rng.normal(size=(n, 3)), axis=0) * 2.0 / step) * step
        p = np.unique(p, axis=0)  # dedupe: coincident points make NN indices moot
        v = rng.normal(size=(len(p), 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        return Dotprop(np.ascontiguousarray(p), np.ascontiguousarray(v))

    dps = [cloud(), cloud()]
    D = np.linalg.norm(
        dps[0].points[:, None, :] - dps[1].points[None, :, :], axis=-1
    )
    n_ties = sum((D[i] == D[i].min()).sum() > 1 for i in range(len(D)))
    assert n_ties, "no tied nearest neighbours - this seed does not test anything"

    want = fastcore.nblast(dps, dps, precision=64)[0, 1]
    got = fastcore.nblast_knn(
        dps, k=1, n_candidates=10, symmetry=None, precision=64
    )[1][0, 0]
    assert got == pytest.approx(want, abs=1e-12)


def test_knn_shapes_rows_sorted_and_exclude_self(knn_dotprops):
    n, k = len(knn_dotprops), 5
    idx, sc = fastcore.nblast_knn(knn_dotprops, k=k, n_candidates=n - 1)
    assert idx.shape == (n, k) and sc.shape == (n, k)
    assert idx.dtype == np.int64
    for i in range(n):
        assert i not in idx[i], f"row {i} contains itself"
        real = sc[i][idx[i] >= 0]
        assert np.all(np.diff(real) <= 0), f"row {i} not descending"


def test_knn_pads_when_k_exceeds_population(knn_dotprops):
    n = len(knn_dotprops)
    k = n + 3
    idx, sc = fastcore.nblast_knn(knn_dotprops, k=k, n_candidates=n - 1, **EXHAUSTIVE)
    assert np.all(idx[:, n - 1:] == -1)
    assert np.all(np.isneginf(sc[:, n - 1:]))


def test_knn_mean_fixes_containment_asymmetry():
    """A short neuron inside a long one: forward disagrees by row, mean does not."""
    long_pts = np.stack([np.arange(120) * 0.5, np.zeros(120), np.zeros(120)], axis=1)
    short_pts = long_pts[:20].copy()
    vect_l = np.tile([1.0, 0.0, 0.0], (len(long_pts), 1))
    vect_s = np.tile([1.0, 0.0, 0.0], (len(short_pts), 1))
    dps = [Dotprop(long_pts, vect_l), Dotprop(short_pts, vect_s)]

    M = fastcore.nblast_allbyall(dps, precision=64)
    long_to_short, short_to_long = M[0, 1], M[1, 0]
    assert short_to_long > long_to_short + 0.2  # the asymmetry exists

    _, fwd = fastcore.nblast_knn(dps, k=1, n_candidates=1, symmetry="forward",
                                 precision=64, **EXHAUSTIVE)
    _, mean = fastcore.nblast_knn(dps, k=1, n_candidates=1, symmetry="mean",
                                  precision=64, **EXHAUSTIVE)
    assert fwd[1, 0] - fwd[0, 0] > 0.2                  # forward rows disagree
    assert abs(mean[0, 0] - mean[1, 0]) < 1e-12         # mean rows agree
    assert long_to_short < mean[0, 0] < short_to_long   # and sit between


@pytest.mark.parametrize("precision,dtype", [(32, np.float32), (64, np.float64)])
def test_knn_precision(knn_dotprops, precision, dtype):
    _, sc = fastcore.nblast_knn(knn_dotprops, k=3, precision=precision)
    assert sc.dtype == dtype


def test_knn_use_alpha_changes_scores(knn_dotprops):
    _, plain = fastcore.nblast_knn(knn_dotprops, k=3, n_candidates=len(knn_dotprops) - 1,
                                   precision=64, **EXHAUSTIVE)
    _, alpha = fastcore.nblast_knn(knn_dotprops, k=3, n_candidates=len(knn_dotprops) - 1,
                                   use_alpha=True, precision=64, **EXHAUSTIVE)
    assert not np.allclose(plain, alpha)


@pytest.mark.parametrize("bad", ["nonsense", 1.5])
def test_knn_unknown_symmetry_raises(knn_dotprops, bad):
    with pytest.raises(ValueError, match="[Uu]nknown symmetry"):
        fastcore.nblast_knn(knn_dotprops, k=2, symmetry=bad)


def test_knn_rejects_bad_k(knn_dotprops):
    with pytest.raises(ValueError, match="k must be"):
        fastcore.nblast_knn(knn_dotprops, k=0)


def test_knn_empty_input():
    idx, sc = fastcore.nblast_knn([], k=5)
    assert idx.shape == (0, 5) and sc.shape == (0, 5)


# --- nblast_knn: query -> target (rectangular) ------------------------------

@pytest.fixture(scope="module")
def knn_split(knn_dotprops):
    """Split the k-NN population into disjoint query / target sets."""
    return knn_dotprops[:4], knn_dotprops[4:]


def dense_rect(query, targets, symmetry):
    """Dense (nq, nt) score matrix under `symmetry`, via the existing nblast path."""
    F = fastcore.nblast(query, targets, precision=64)
    if symmetry == "forward":
        return F
    R = fastcore.nblast(targets, query, precision=64).T
    return {"mean": (F + R) / 2, "min": np.minimum(F, R),
            "max": np.maximum(F, R)}[symmetry]


@pytest.mark.parametrize("symmetry", ["mean", "forward", "min", "max"])
def test_knn_rect_exhaustive_matches_dense(knn_split, symmetry):
    q, t = knn_split
    k = 3
    S = dense_rect(q, t, symmetry)
    want = np.sort(S, axis=1)[:, ::-1][:, :k]
    idx, sc = fastcore.nblast_knn(q, target=t, k=k, symmetry=symmetry,
                                  n_candidates=len(t), precision=64, **EXHAUSTIVE)
    assert np.allclose(sc, want, atol=1e-12)
    for i in range(len(q)):
        for c in range(k):
            assert abs(sc[i, c] - S[i, idx[i, c]]) < 1e-12


def test_knn_rect_indexes_targets_not_queries(knn_split):
    q, t = knn_split
    idx, _ = fastcore.nblast_knn(q, target=t, k=3, n_candidates=len(t), **EXHAUSTIVE)
    assert idx.shape == (len(q), 3)
    assert idx.max() < len(t), "indices must address the target list"
    assert idx.min() >= 0


def test_knn_rect_keeps_self_when_sets_overlap(knn_dotprops):
    """Unlike the square form, a neuron present in both sets matches itself at 1.0."""
    idx, sc = fastcore.nblast_knn(knn_dotprops, target=knn_dotprops, k=1,
                                  n_candidates=len(knn_dotprops), precision=64,
                                  **EXHAUSTIVE)
    assert np.array_equal(idx[:, 0], np.arange(len(knn_dotprops)))
    assert np.allclose(sc[:, 0], 1.0, atol=1e-9)


def test_knn_rect_pads_beyond_target_count(knn_split):
    q, t = knn_split
    k = len(t) + 2
    idx, sc = fastcore.nblast_knn(q, target=t, k=k, n_candidates=len(t), **EXHAUSTIVE)
    assert np.all(idx[:, len(t):] == -1)
    assert np.all(np.isneginf(sc[:, len(t):]))
    assert np.all(idx[:, :len(t)] >= 0), "backfill should fill up to nt"


def test_knn_rect_none_targets_is_the_square_form(knn_dotprops):
    a = fastcore.nblast_knn(knn_dotprops, k=3, n_candidates=5)
    b = fastcore.nblast_knn(knn_dotprops, target=None, k=3, n_candidates=5)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
