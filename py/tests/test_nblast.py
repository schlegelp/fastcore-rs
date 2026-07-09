"""NBLAST tests: the Rust shull+aann pipeline vs an independent numpy reference."""
import csv as _csv
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.spatial import cKDTree

import navis_fastcore as fastcore
from navis_fastcore.nblast import Dotprop, Synapses, _tangents_and_alpha

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
    # t=0 selects every cell in both engines, so smart reduces to a full NBLAST.
    res = navis.nblast_smart(dp, t=0, progress=False)
    M_navis = res.loc[ids, ids].to_numpy()
    M_fast = np.asarray(fastcore.nblast_smart(dotprops, t=0, precision=64))
    off = ~np.eye(len(dotprops), dtype=bool)
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
