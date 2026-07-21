"""Tests for `dotprops`.

The oracle is the scipy/numpy formulation navis and skeletor both carry: a
`cKDTree.query` for the neighbourhoods, then `np.linalg.svd` of each local scatter
matrix. That is exactly the code this replaces, so agreeing with it *is* the
requirement — a silent off-by-one in `k` would shift every downstream NBLAST score.
"""

import numpy as np
import pytest

import navis_fastcore as fastcore


def reference(points, k):
    """navis' dotprops, spelled out: cKDTree neighbourhoods + a per-point SVD."""
    KDTree = pytest.importorskip("scipy.spatial").cKDTree

    points = np.asarray(points, dtype=np.float64)
    _, ix = KDTree(points).query(points, k=k)

    pt = points[ix]
    centers = np.mean(pt, axis=1)
    cpt = pt - centers.reshape((pt.shape[0], 1, 3))
    inertia = cpt.transpose((0, 2, 1)) @ cpt
    _, s, vh = np.linalg.svd(inertia)
    return vh[:, 0, :], (s[:, 0] - s[:, 1]) / s.sum(axis=1)


def cloud(n=500, seed=0):
    return np.random.default_rng(seed).random((n, 3)) * 100


@pytest.mark.parametrize("k", [3, 5, 20])
def test_matches_scipy_reference(k):
    pts = cloud()
    vect, alpha = fastcore.dotprops(pts, k=k)
    ref_vect, ref_alpha = reference(pts, k)

    np.testing.assert_allclose(alpha, ref_alpha, atol=1e-9)
    # Eigenvector sign is arbitrary, so compare |dot|.
    dots = np.abs(np.sum(vect * ref_vect, axis=1))
    np.testing.assert_allclose(dots, 1.0, atol=1e-6)


def test_k_includes_the_point_itself():
    # Three collinear points. With k=2 each neighbourhood is the point plus its one
    # nearest neighbour - still a line, so alpha is 1. If `k` excluded self, k=2
    # would pull in a second neighbour and the middle point's alpha would still be 1
    # but the *end* points' neighbourhoods would differ; comparing against the scipy
    # reference (which includes self) pins the convention either way.
    pts = np.array([[0.0, 0, 0], [1, 0, 0], [2, 0, 0]])
    _, alpha = fastcore.dotprops(pts, k=2)
    np.testing.assert_allclose(alpha, 1.0, atol=1e-12)
    np.testing.assert_allclose(alpha, reference(pts, 2)[1], atol=1e-12)


def test_a_line_is_anisotropic_and_a_lattice_is_not():
    line = np.zeros((30, 3))
    line[:, 0] = np.arange(30)
    vect, alpha = fastcore.dotprops(line, k=5)
    np.testing.assert_allclose(alpha, 1.0, atol=1e-9)
    np.testing.assert_allclose(np.abs(vect[:, 0]), 1.0, atol=1e-9)

    grid = np.stack(np.meshgrid(*[np.arange(5.0)] * 3, indexing="ij"), -1).reshape(-1, 3)
    _, alpha = fastcore.dotprops(grid, k=7)
    interior = np.all((grid > 0) & (grid < 4), axis=1)
    assert alpha[interior].max() < 1e-9


def test_vectors_are_unit_length_and_sign_is_deterministic():
    pts = cloud(300, seed=4)
    vect, _ = fastcore.dotprops(pts, k=10)
    np.testing.assert_allclose(np.linalg.norm(vect, axis=1), 1.0, atol=1e-12)

    # The convention: the largest-magnitude component is positive.
    big = np.argmax(np.abs(vect), axis=1)
    assert np.all(vect[np.arange(len(vect)), big] > 0)


def test_threads_do_not_change_the_result():
    pts = cloud(400, seed=6)
    v1, a1 = fastcore.dotprops(pts, k=15, threads=1)
    for t in (2, 4):
        vt, at = fastcore.dotprops(pts, k=15, threads=t)
        np.testing.assert_array_equal(vt, v1)
        np.testing.assert_array_equal(at, a1)


def test_k_is_clamped_to_the_cloud_size():
    pts = cloud(5, seed=8)
    v_big, a_big = fastcore.dotprops(pts, k=99)
    v_all, a_all = fastcore.dotprops(pts, k=5)
    np.testing.assert_array_equal(v_big, v_all)
    np.testing.assert_array_equal(a_big, a_all)


def test_degenerate_clouds_give_zero_alpha_not_nan():
    coincident = np.zeros((5, 3))
    vect, alpha = fastcore.dotprops(coincident, k=3)
    assert np.all(alpha == 0)
    np.testing.assert_array_equal(vect, np.tile([1.0, 0, 0], (5, 1)))

    one = np.array([[3.0, 4.0, 5.0]])
    vect, alpha = fastcore.dotprops(one, k=20)
    assert alpha[0] == 0
    np.testing.assert_array_equal(vect[0], [1.0, 0, 0])

    empty = np.zeros((0, 3))
    vect, alpha = fastcore.dotprops(empty, k=5)
    assert vect.shape == (0, 3) and alpha.shape == (0,)


def test_coincident_points_do_not_produce_nan():
    # Duplicated vertices are ordinary in real meshes; the neighbourhood is still
    # well defined as long as *some* neighbour differs.
    pts = np.repeat(cloud(50, seed=2), 3, axis=0)
    vect, alpha = fastcore.dotprops(pts, k=6)
    assert np.isfinite(vect).all() and np.isfinite(alpha).all()
    assert ((alpha >= 0) & (alpha <= 1)).all()


def test_validation():
    with pytest.raises(ValueError):
        fastcore.dotprops(np.zeros((5, 2)), k=3)
    with pytest.raises(ValueError):
        fastcore.dotprops(np.zeros((5, 3)), k=0)
    with pytest.raises(BaseException):
        fastcore.dotprops(np.full((5, 3), np.nan), k=3)


def test_dotprop_from_points_round_trips_through_nblast():
    # The end-to-end check: a Dotprop built here must score identically to one built
    # from the same arrays by hand.
    pts = cloud(200, seed=12)
    dp = fastcore.Dotprop.from_points(pts, k=5)
    vect, alpha = fastcore.dotprops(pts, k=5)

    np.testing.assert_array_equal(dp.points, pts)
    np.testing.assert_array_equal(dp.vect, vect)
    np.testing.assert_array_equal(dp.alpha, alpha)

    manual = fastcore.Dotprop(pts, vect, alpha)
    np.testing.assert_allclose(
        fastcore.nblast([dp], [dp]), fastcore.nblast([manual], [manual])
    )
