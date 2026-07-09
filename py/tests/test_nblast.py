"""NBLAST tests: the Rust shull+aann pipeline vs an independent numpy reference."""
import csv as _csv
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.spatial import cKDTree

import navis_fastcore as fastcore
from navis_fastcore.nblast import Dotprop, _tangents_and_alpha

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
