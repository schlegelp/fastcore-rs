"""Elastix transforms, checked against Elastix's own `transformix` output.

The ground truth in `transformix_golden.csv` was produced by the `transformix` binary itself, so
these are not self-consistency checks: they pin us to Elastix's behaviour, including its habit of
returning points outside the control-point grid **unchanged** rather than failing them.
"""

import pickle

import numpy as np
import pytest

import navis_fastcore as fastcore

# Measured margin is ~5e-7 (transformix's own print precision, six decimals). If a change lands
# at 1e-5, something is subtly wrong -- do not loosen this.
ATOL = 1e-4

CASES = ["affine", "translation", "euler", "euler_zyx", "similarity", "bspline", "add"]


@pytest.fixture(scope="module")
def bspline(elastix_dir):
    """The B-spline fixture, composed over an affine."""
    return fastcore.ElastixTransform(elastix_dir / "bspline.txt")


@pytest.fixture(scope="module")
def in_grid(elastix_dir, bspline):
    """Does a *source* point reach the spline's valid region (i.e. after the initial affine)?"""
    affine = fastcore.ElastixTransform(elastix_dir / "affine.txt")
    size = bspline.grid_size[0]
    origin = bspline.grid_origin[0]
    spacing = bspline.grid_spacing[0]

    def check(points):
        u = (affine.xform(np.atleast_2d(points)) - origin) / spacing
        return np.all((u >= 1.0) & (u < size - 2), axis=1)

    return check


# --- Against transformix ---------------------------------------------------------------


@pytest.mark.parametrize("case", CASES)
def test_matches_transformix(case, elastix_dir, elastix_golden):
    xf = fastcore.ElastixTransform(elastix_dir / f"{case}.txt")
    got = xf.xform(elastix_golden["input"])
    np.testing.assert_allclose(got, elastix_golden[case], atol=ATOL)


def test_compute_zyx_changes_the_rotation_order(elastix_dir, elastix_golden):
    """`euler.txt` and `euler_zyx.txt` carry identical angles and differ only in `ComputeZYX`.

    The default (false) is Rz*Rx*Ry, not the Rz*Ry*Rx its name suggests.
    """
    a = fastcore.ElastixTransform(elastix_dir / "euler.txt").xform(elastix_golden["input"])
    b = fastcore.ElastixTransform(elastix_dir / "euler_zyx.txt").xform(elastix_golden["input"])
    assert np.abs(a - b).max() > 1.0


# --- The grid boundary -----------------------------------------------------------------


def test_out_of_grid_points_are_returned_unchanged(bspline, elastix_dir):
    """Elastix's rule, and the exact opposite of CMTK's, which reports such points as FAILED."""
    far = np.array([[-5000.0, -5000.0, -5000.0]])

    # ...but the affine still applies: it is a separate step of the chain.
    affine = fastcore.ElastixTransform(elastix_dir / "affine.txt")
    np.testing.assert_allclose(bspline.xform(far), affine.xform(far), atol=1e-9)

    out = bspline.xform(far, out_of_bounds="nan")
    assert np.isnan(out).all()


def test_out_of_bounds_only_affects_out_of_grid_points(bspline, elastix_golden):
    """Points inside the grid must be unaffected by the flag."""
    got_id = bspline.xform(elastix_golden["input"])
    got_nan = bspline.xform(elastix_golden["input"], out_of_bounds="nan")

    inside = ~np.isnan(got_nan).any(axis=1)
    assert inside.sum() > 0 and (~inside).sum() > 0, "fixture must span the boundary"
    np.testing.assert_allclose(got_id[inside], got_nan[inside], atol=1e-12)


def test_bad_out_of_bounds_raises(bspline):
    with pytest.raises(ValueError, match="out_of_bounds"):
        bspline.xform(np.zeros((1, 3)), out_of_bounds="bogus")


# --- The inverse -----------------------------------------------------------------------


def test_inverse_is_forward_consistent(bspline, elastix_golden):
    """The guarantee: what we hand back really is a preimage."""
    fwd = bspline.xform(elastix_golden["input"])
    back = bspline.xform_inv(fwd)
    assert np.isfinite(back).all()
    np.testing.assert_allclose(bspline.xform(back), fwd, atol=1e-6)


def test_inverse_round_trips_for_sources_inside_the_grid(bspline, elastix_golden, in_grid):
    """A source point inside the grid has a *unique* warp preimage, so we recover it exactly.

    (The fixture is injective: det(I + J) > 0 throughout. A folded registration would not be.)
    """
    src = elastix_golden["input"]
    inside = in_grid(src)
    assert inside.sum() > 20, "fixture must have points inside the grid"

    back = bspline.xform_inv(bspline.xform(src))
    np.testing.assert_allclose(back[inside], src[inside], atol=1e-6)


def test_a_target_can_have_two_preimages(bspline, elastix_golden, in_grid):
    """The forward map is `p + disp(p)` inside the grid and `p` outside -- discontinuous, and so
    **not injective**. A target near the boundary can have a preimage on each branch.

    We prefer the warp branch, so for such a point the inverse returns a *different* point than
    you started from. That is not a defect and cannot be fixed; it is what a 2-to-1 map means.
    What we guarantee is that the answer is a genuine preimage -- which this asserts.
    """
    src = elastix_golden["input"]
    fwd = bspline.xform(src)
    back = bspline.xform_inv(fwd)

    off = np.abs(back - src).max(axis=1) > 1e-6
    assert off.sum() == 1, "fixture should carry exactly one such point"
    assert not in_grid(src)[off].any(), "...and its source must be outside the grid"

    # ...and what we returned really does map to the target.
    np.testing.assert_allclose(bspline.xform(back[off]), fwd[off], atol=1e-6)


def test_seed_iter_does_not_change_a_converged_answer(bspline, elastix_golden):
    """On a grid this small the seed is not *needed* -- see the real-file test below, where it is.

    This only pins that turning it off does not corrupt an answer the solver already gets right.
    """
    fwd = bspline.xform(elastix_golden["input"])
    np.testing.assert_allclose(
        bspline.xform_inv(fwd, seed_iter=0), bspline.xform_inv(fwd), atol=1e-6
    )


def test_add_chains_cannot_be_inverted(elastix_dir):
    """`T(x) = T0(x) + T1(x) - x` does not decompose into invertible hops. We refuse."""
    xf = fastcore.ElastixTransform(elastix_dir / "add.txt")
    with pytest.raises(ValueError, match="cannot be inverted"):
        xf.xform_inv(np.zeros((1, 3)))
    # ...but it still transforms forwards
    assert np.isfinite(xf.xform(np.zeros((1, 3)))).all()


def test_initial_guess(bspline, elastix_golden):
    """Handed the exact answer as a guess, the solver lands on it -- including for the ambiguous
    point, where the guess is what breaks the tie."""
    src = elastix_golden["input"]
    fwd = bspline.xform(src)
    back = bspline.xform_inv(fwd, initial_guess=src)
    np.testing.assert_allclose(back, src, atol=1e-6)

    with pytest.raises(ValueError, match="one point per input point"):
        bspline.xform_inv(fwd, initial_guess=src[:2])


# --- Chaining --------------------------------------------------------------------------


def test_initial_transform_is_resolved_from_the_files_own_directory(bspline):
    """`bspline.txt` names `affine.txt` by a bare relative filename."""
    assert bspline.kinds == [["linear", "bspline"]]
    assert len(bspline) == 1
    assert bspline.affine.shape == (4, 4)
    assert bspline.grid_size.tolist() == [[10, 10, 8]]


def test_chain_equals_manual_double_application(elastix_dir, elastix_golden):
    path = elastix_dir / "affine.txt"
    one = fastcore.ElastixTransform(path)
    two = fastcore.ElastixTransform([path, path])
    np.testing.assert_allclose(
        two.xform(elastix_golden["input"]),
        one.xform(one.xform(elastix_golden["input"])),
        atol=1e-9,
    )


def test_invert_flag_reverses_a_hop(elastix_dir, elastix_golden):
    """The `invert` flag must agree exactly with `xform_inv`."""
    path = elastix_dir / "bspline.txt"
    xf = fastcore.ElastixTransform(path)
    fwd = xf.xform(elastix_golden["input"])
    np.testing.assert_array_equal(
        fastcore.ElastixTransform(path, invert=True).xform(fwd), xf.xform_inv(fwd)
    )


def test_invert_length_mismatch_raises(elastix_dir):
    with pytest.raises(ValueError, match="one flag per transform"):
        fastcore.ElastixTransform([elastix_dir / "affine.txt"], invert=[True, False])


# --- Input handling --------------------------------------------------------------------


def test_single_point_in_single_point_out(bspline):
    got = bspline.xform([30.0, 25.0, 20.0])
    assert got.shape == (3,)
    np.testing.assert_allclose(got, bspline.xform(np.array([[30.0, 25.0, 20.0]]))[0])


def test_int_input_is_coerced(bspline):
    np.testing.assert_allclose(
        bspline.xform(np.array([[30, 25, 20]])),
        bspline.xform(np.array([[30.0, 25.0, 20.0]])),
    )


def test_bad_shape_raises(bspline):
    with pytest.raises(ValueError, match=r"\(N, 3\)"):
        bspline.xform(np.zeros((4, 2)))


def test_missing_file_raises(elastix_dir):
    with pytest.raises(ValueError, match="could not read"):
        fastcore.ElastixTransform(elastix_dir / "nope.txt")


def test_a_file_that_is_not_a_transform_raises(elastix_dir, tmp_path):
    """`navis-flybrains` ships a `template_to_BANC.txt` holding nothing but a filename."""
    p = tmp_path / "pointer.txt"
    p.write_text("3_elastix_Bspline_fine.txt\n")
    with pytest.raises(ValueError, match="not an elastix transform"):
        fastcore.ElastixTransform(p)


# --- Plumbing --------------------------------------------------------------------------


def test_pickle_round_trip(bspline, elastix_golden):
    """Pickling must re-load from disk, not ship the coefficients (BANC's file is 56 MB)."""
    blob = pickle.dumps(bspline)
    assert len(blob) < 1000, f"pickle is {len(blob)} bytes -- is it carrying the grid?"

    revived = pickle.loads(blob)
    np.testing.assert_allclose(
        revived.xform(elastix_golden["input"]), bspline.xform(elastix_golden["input"])
    )


def test_n_cores_does_not_change_the_answer(bspline, elastix_golden):
    fwd = bspline.xform(elastix_golden["input"])
    np.testing.assert_array_equal(
        bspline.xform_inv(fwd, n_cores=1), bspline.xform_inv(fwd)
    )


def test_repr(bspline):
    assert "ElastixTransform" in repr(bspline)
    assert "linear+bspline" in repr(bspline)


def test_load_elastix_transform_factory(elastix_dir):
    xf = fastcore.load_elastix_transform(elastix_dir / "affine.txt")
    assert isinstance(xf, fastcore.ElastixTransform)


def test_large_batch_is_finite(bspline):
    rng = np.random.default_rng(0)
    pts = rng.uniform(0, 100, (100_000, 3))
    assert np.isfinite(bspline.xform(pts)).all()
