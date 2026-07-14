"""CMTK transforms: the Rust kernels vs CMTK's own `streamxform`.

The ground truth in `streamxform_golden.csv` was produced by the CMTK binary itself, so
these are not self-consistency checks - they pin us to CMTK's behaviour, including its habit
of *failing* on points whose inverse does not converge (which we return as NaN).
"""

import pickle

import numpy as np
import pytest

import navis_fastcore as fastcore

# `atol=1e-4` is the tolerance of the reference implementation's own golden tests. Our
# measured margin is ~4e-7, i.e. three orders of magnitude of headroom. If a change pushes a
# result to 1e-5, something is subtly wrong - do not loosen this.
ATOL = 1e-4
RTOL = 1e-5


@pytest.fixture(scope="module")
def reg(cmtk_dir):
    return fastcore.CmtkRegistration(cmtk_dir)


# ---------------------------------------------------------------------------
# Against streamxform
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case,kwargs",
    [
        ("affine_forward", dict(transform="affine")),
        ("warp_forward", dict(transform="warp")),
    ],
)
def test_forward_matches_streamxform(reg, golden, case, kwargs):
    out = reg.xform(golden["input"], **kwargs)
    np.testing.assert_allclose(out, golden[case], atol=ATOL, rtol=RTOL, equal_nan=True)


@pytest.mark.parametrize(
    "case,kwargs",
    [
        ("affine_inverse", dict(transform="affine")),
        ("warp_inverse", dict(transform="warp")),
    ],
)
def test_inverse_matches_streamxform(reg, golden, case, kwargs):
    out = reg.xform_inv(golden["input"], **kwargs)
    np.testing.assert_allclose(out, golden[case], atol=ATOL, rtol=RTOL, equal_nan=True)


def test_warp_inverse_reproduces_streamxform_failures(reg, golden):
    """CMTK reports two of the five sample points as FAILED; we must return NaN there.

    Getting a finite answer for these would be *worse* than useless: it would silently
    disagree with every other CMTK-based tool.
    """
    out = reg.xform_inv(golden["input"])
    assert np.isnan(out[0]).all()
    assert np.isnan(out[4]).all()
    assert np.isfinite(out[[1, 2, 3]]).all()


# ---------------------------------------------------------------------------
# Round-trips
# ---------------------------------------------------------------------------


def test_warp_roundtrip(reg):
    pts = np.array(
        [[50.0, 50.0, 50.0], [100.0, 100.0, 20.0], [250.0, 150.0, 60.0], [300.0, 120.0, 70.0]]
    )
    back = reg.xform_inv(reg.xform(pts))
    np.testing.assert_allclose(back, pts, atol=1e-4)


def test_affine_roundtrip_is_exact(reg):
    pts = np.array([[0.0, 0.0, 0.0], [123.0, 45.0, 67.0], [500.0, 250.0, 100.0]])
    back = reg.xform_inv(reg.xform(pts, transform="affine"), transform="affine")
    np.testing.assert_allclose(back, pts, atol=1e-10)


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------


def test_single_point_in_single_point_out(reg):
    single = reg.xform(np.array([50.0, 50.0, 50.0]))
    assert single.shape == (3,)
    batch = reg.xform(np.array([[50.0, 50.0, 50.0]]))
    assert batch.shape == (1, 3)
    np.testing.assert_allclose(single, batch[0])


def test_int_input_is_coerced(reg):
    out = reg.xform(np.array([[50, 50, 50]], dtype=np.int64))
    np.testing.assert_allclose(out, reg.xform(np.array([[50.0, 50.0, 50.0]])))


def test_list_input_works(reg):
    np.testing.assert_allclose(
        reg.xform([[50.0, 50.0, 50.0]]), reg.xform(np.array([[50.0, 50.0, 50.0]]))
    )


def test_bad_shape_raises(reg):
    with pytest.raises(ValueError, match=r"\(N, 3\)"):
        reg.xform(np.zeros((4, 2)))


def test_bad_transform_raises(reg):
    with pytest.raises(ValueError, match="warp.*affine"):
        reg.xform(np.zeros((1, 3)), transform="bogus")


def test_missing_path_raises():
    with pytest.raises(ValueError, match="no such file or directory"):
        fastcore.CmtkRegistration("/nonexistent/nope.list")


def test_initial_guess_length_mismatch_raises(reg):
    with pytest.raises(ValueError, match="one point per input point"):
        reg.xform_inv(np.zeros((3, 3)), initial_guess=np.zeros((2, 3)))


# ---------------------------------------------------------------------------
# Behaviour flags
# ---------------------------------------------------------------------------


def test_out_of_domain_is_nan_by_default_and_fallback_fills_it(reg):
    """CMTK reports points outside the domain box as FAILED; we return NaN, by default.

    Returning a clamped-boundary extrapolation instead (which is what the reference
    implementation does by default) would silently disagree with every CMTK-based tool.
    """
    pts = np.array([[-5000.0, -5000.0, -5000.0], [100.0, 100.0, 20.0]])

    strict = reg.xform(pts)
    assert np.isnan(strict[0]).all()
    assert np.isfinite(strict[1]).all()

    filled = reg.xform(pts, fallback_to_affine=True)
    np.testing.assert_allclose(filled[0], reg.xform(pts, transform="affine")[0], atol=1e-12)
    # The in-domain point is untouched by the fallback.
    np.testing.assert_allclose(filled[1], strict[1])


def test_fallback_to_affine_survives_an_inverted_registration(cmtk_dir):
    """Regression: `fallback_to_affine` used to be a silent no-op when `invert=True`.

    This is the direction navis walks about half of its bridging edges in - it passes
    `affine_fallback=True` by default through `TransformSequence` - so an inverted edge
    silently dropped every point the binary path would have rescued.
    """
    reg = fastcore.CmtkRegistration(cmtk_dir)
    pts = np.array([[-5000.0, -5000.0, -5000.0], [100.0, 100.0, 20.0]])

    strict = reg.xform(pts, invert=True)
    assert np.isnan(strict[0]).all()
    assert np.isfinite(strict[1]).all()

    filled = reg.xform(pts, invert=True, fallback_to_affine=True)
    assert np.isfinite(filled).all()

    # ...and it falls back to the *inverse* affine, not the forward one. A wrong-direction
    # fallback would be worse than a NaN: a plausible number pointing into the wrong space.
    inv_affine = reg.xform(pts, invert=True, transform="affine")
    fwd_affine = reg.xform(pts, transform="affine")
    np.testing.assert_allclose(filled[0], inv_affine[0], atol=1e-12)
    assert np.abs(inv_affine[0] - fwd_affine[0]).max() > 1.0, "else this proves nothing"


def test_fallback_to_affine_survives_xform_inv(reg):
    """The same hole from the other side: `xform_inv` could not fall back at all."""
    pts = np.array([[-5000.0, -5000.0, -5000.0], [100.0, 100.0, 20.0]])

    assert np.isnan(reg.xform_inv(pts)[0]).all()

    filled = reg.xform_inv(pts, fallback_to_affine=True)
    np.testing.assert_allclose(
        filled[0], reg.xform_inv(pts, transform="affine")[0], atol=1e-12
    )


def test_chain_and_hop_fallbacks_differ_on_a_chain(cmtk_dir):
    """`fallback_to_affine` falls back over the *whole chain*, as nat/navis do.

    Verified against the binary: `streamxform --affine-only` composes the affine of every
    registration in the list, and navis re-runs the failed rows through it from the ORIGINAL
    point. So a point that survives hop 1 and fails hop 2 loses its hop-1 warp.

    `"hop"` keeps that warp and swaps the affine in only where it actually failed. Better,
    probably - but a silent departure, so it has to be asked for.
    """
    chain = fastcore.CmtkRegistration([cmtk_dir, cmtk_dir])
    pts = np.array([[5.0, 179.0, 38.0]])  # inside the domain for hop 1, outside it for hop 2

    assert np.isnan(chain.xform(pts)).any(), "this point must fail, or the test proves nothing"

    by_chain = chain.xform(pts, fallback_to_affine=True)
    by_hop = chain.xform(pts, fallback_to_affine="hop")
    assert np.isfinite(by_chain).all() and np.isfinite(by_hop).all()

    # True == "chain" == the whole chain, affine-only, from the original point
    np.testing.assert_allclose(by_chain, chain.xform(pts, transform="affine"), atol=1e-12)
    np.testing.assert_allclose(by_chain, chain.xform(pts, fallback_to_affine="chain"), atol=1e-12)

    # ...and "hop" lands somewhere else entirely, because it kept the hop-1 warp
    assert np.abs(by_chain - by_hop).max() > 1.0


def test_on_a_single_registration_chain_and_hop_agree(reg):
    """The two only diverge once there is more than one hop to disagree about."""
    pts = np.array([[-5000.0, -5000.0, -5000.0]])
    np.testing.assert_allclose(
        reg.xform(pts, fallback_to_affine="chain"),
        reg.xform(pts, fallback_to_affine="hop"),
        atol=1e-12,
    )


def test_bad_fallback_value_raises(reg):
    with pytest.raises(ValueError, match="chain"):
        reg.xform(np.zeros((1, 3)), fallback_to_affine="bogus")


def test_extrapolation_can_be_opted_into(reg):
    out = reg.xform(np.array([[-5000.0, -5000.0, -5000.0]]), allow_extrapolation=True)
    assert np.isfinite(out).all()


def test_domain_is_the_world_box_not_the_lattice_extent(reg):
    """CMTK's domain is `0 <= p <= domain`, NOT the control-point lattice's extent.

    The lattice is padded one cell outside the domain (origin == -spacing), so a point with
    lattice coordinate u just below 1 sits inside the lattice box but *outside* the world
    domain box. CMTK fails it. The reference implementation accepts it.
    """
    domain = reg.spacing[0] * (reg.dims[0] - 3)
    origin = -reg.spacing[0]

    # u_x = 0.5 -> inside the lattice box [0, dims-3), but world x < 0 -> outside the domain.
    p = np.array([[origin[0] + 0.5 * reg.spacing[0][0], 100.0, 50.0]])
    assert p[0, 0] < 0
    assert np.isnan(reg.xform(p)).all()

    # The domain corners themselves are in.
    assert np.isfinite(reg.xform(np.array([[0.0, 0.0, 0.0]]))).all()
    assert np.isfinite(reg.xform(np.array([domain - 1e-6]))).all()


def test_clamp_to_domain_off_recovers_out_of_domain_preimages(reg, golden):
    """The two points CMTK calls FAILED do have preimages - they just lie outside the
    image domain. `clamp_to_domain=False` finds them, at the price of disagreeing with
    CMTK. This is the one semantic knob in the port, so pin both sides of it.
    """
    pts = golden["input"]

    faithful = reg.xform_inv(pts)  # default: clamped, CMTK-faithful
    assert np.isnan(faithful[[0, 4]]).all()

    loose = reg.xform_inv(pts, clamp_to_domain=False)
    assert np.isfinite(loose[0]).all(), "the preimage exists once the domain box is lifted"

    # It is a real preimage: it maps forward onto the target. Verifying that needs
    # `allow_extrapolation=True` -- the preimage is by construction outside the domain box,
    # so the CMTK-faithful forward transform refuses to evaluate there (which is the whole
    # reason CMTK rejects it).
    domain = reg.spacing[0] * (reg.dims[0] - 3)
    assert (loose[0] < 0).any() or (loose[0] > domain).any(), "preimage is out of domain"
    assert np.isnan(reg.xform(loose[0])).all(), "...so the faithful forward declines it"
    np.testing.assert_allclose(
        reg.xform(loose[0], allow_extrapolation=True), pts[0], atol=1e-6
    )

    # Points that solve *inside* the domain are unaffected by the flag.
    np.testing.assert_allclose(faithful[[1, 2, 3]], loose[[1, 2, 3]], atol=1e-9)


def test_initial_guess_is_accepted(reg, golden):
    # Seeding with the answer must not change the answer.
    target = reg.xform(golden["input"][[1, 2, 3]])
    want = reg.xform_inv(target)
    got = reg.xform_inv(target, initial_guess=want)
    np.testing.assert_allclose(got, want, atol=1e-6)


# ---------------------------------------------------------------------------
# Chains
# ---------------------------------------------------------------------------


def test_chain_matches_manual_double_application(reg, cmtk_dir):
    chain = fastcore.CmtkRegistration([cmtk_dir, cmtk_dir])
    assert len(chain) == 2
    pts = np.array([[50.0, 50.0, 50.0], [100.0, 100.0, 20.0]])
    np.testing.assert_allclose(chain.xform(pts), reg.xform(reg.xform(pts)), atol=1e-12)


def test_invert_flag_is_the_inverse_transform(reg):
    """One parse, both directions: `invert` is a property of the traversal, not the object."""
    pts = np.array([[100.0, 100.0, 20.0], [250.0, 150.0, 60.0]])
    np.testing.assert_allclose(reg.xform(reg.xform(pts), invert=True), pts, atol=1e-4)


def test_invert_is_not_the_same_knob_as_xform_inv(cmtk_dir, tiny_dir):
    """`invert` flips each hop in place; `xform_inv` also *reverses* the chain.

    For a single hop they agree. For a chain they must not - and only `invert` can express a
    mixed-direction traversal, which is exactly what a bridging graph hands you. Affine-only,
    so no point can fall out of a domain and make the comparison vacuous through NaN.
    """
    chain = fastcore.CmtkRegistration([cmtk_dir, tiny_dir])
    pts = np.array([[100.0, 100.0, 20.0]])

    flipped = chain.xform(pts, transform="affine", invert=True)
    reversed_ = chain.xform_inv(pts, transform="affine")
    assert np.isfinite(flipped).all() and np.isfinite(reversed_).all()
    assert not np.allclose(flipped, reversed_, atol=1e-6)

    # hop 0 forwards, hop 1 backwards - no whole-chain spelling exists for this
    mixed = chain.xform(pts, transform="affine", invert=[False, True])
    assert np.isfinite(mixed).all()
    assert not np.allclose(mixed, flipped, atol=1e-6)


def test_chain_inverse_reverses_order(reg, cmtk_dir):
    chain = fastcore.CmtkRegistration([cmtk_dir, cmtk_dir])
    pts = np.array([[100.0, 100.0, 20.0], [250.0, 150.0, 60.0]])
    np.testing.assert_allclose(chain.xform_inv(chain.xform(pts)), pts, atol=1e-4)


def test_invert_length_mismatch_raises(reg):
    with pytest.raises(ValueError, match="one flag per registration"):
        reg.xform(np.zeros((2, 3)), invert=[True, False])


# ---------------------------------------------------------------------------
# Loading & introspection
# ---------------------------------------------------------------------------


def test_gz_and_plain_both_load(cmtk_dir, tiny_dir):
    # The bundled JFRC2 registration is gzipped; the tiny fixture is plain text. Gzip is
    # sniffed from magic bytes, not the file name.
    gz = fastcore.CmtkRegistration(cmtk_dir)
    plain = fastcore.CmtkRegistration(tiny_dir)
    np.testing.assert_array_equal(gz.dims, [[59, 27, 11]])
    np.testing.assert_array_equal(plain.dims, [[5, 5, 5]])
    assert gz.version == ["1.1"]
    assert plain.version == ["2.4"]


def test_introspection(reg):
    assert reg.has_spline == [True]
    assert reg.affine.shape == (4, 4)
    np.testing.assert_allclose(reg.affine[3], [0, 0, 0, 1])
    np.testing.assert_allclose(reg.spacing[0], [11.3642, 13.2453, 16.8741], atol=1e-3)
    assert "CmtkRegistration" in repr(reg)


def test_load_cmtk_registration_factory(cmtk_dir):
    reg = fastcore.load_cmtk_registration(cmtk_dir)
    assert isinstance(reg, fastcore.CmtkRegistration)
    assert len(reg) == 1


def test_pickle_roundtrip(reg):
    # navis/joblib fan out over processes, so the object must survive a pickle. It re-loads
    # from disk rather than shipping ~420 KB of coefficients through the pipe.
    pts = np.array([[50.0, 50.0, 50.0], [250.0, 150.0, 60.0]])
    revived = pickle.loads(pickle.dumps(reg))
    np.testing.assert_allclose(revived.xform(pts), reg.xform(pts))


# ---------------------------------------------------------------------------
# Parallelism
# ---------------------------------------------------------------------------


def test_n_cores_does_not_change_the_answer(reg, golden):
    # Each point is solved independently, so the result must be bit-identical regardless of
    # how the work was split.
    one = reg.xform_inv(golden["input"], n_cores=1)
    many = reg.xform_inv(golden["input"])
    np.testing.assert_array_equal(np.isnan(one), np.isnan(many))
    np.testing.assert_array_equal(one[~np.isnan(one)], many[~np.isnan(many)])


def test_large_batch_is_finite(reg):
    rng = np.random.default_rng(0)
    pts = rng.uniform([0, 0, 0], [600, 300, 120], size=(100_000, 3))
    out = reg.xform(pts)
    assert out.shape == (100_000, 3)
    assert np.isfinite(out).all()
