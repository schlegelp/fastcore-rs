"""Elastix transforms against the *real* registrations `navis-flybrains` ships.

Skipped unless navis-flybrains is present. These files are 3-56 MB, so they cannot be checked
into the repo, but they are the ones anybody actually uses -- and two of them are four-deep
chains, which the synthetic fixtures do not cover.

They are also the only place the inverse's *seeding* can be exercised. FANC's and BANC's
displacements reach 10-25 grid cells, where a solver started at the bare target lands nowhere
near the answer. You cannot make a small synthetic grid deform that hard without the warp
folding, so `test_elastix.py` cannot show this and does not try.
"""

import importlib.util
import os
import re
from pathlib import Path

import numpy as np
import pytest

import navis_fastcore as fastcore


def _flybrains_data():
    """Locate `flybrains/data`, without importing flybrains.

    `import flybrains` drags in navis and GitPython, which we neither need nor want here -- all
    we want is the directory. `find_spec` resolves the package location without executing it.
    `$FASTCORE_FLYBRAINS_DATA` overrides, for a source checkout.
    """
    env = os.environ.get("FASTCORE_FLYBRAINS_DATA")
    if env:
        return Path(env)
    spec = importlib.util.find_spec("flybrains")
    if spec is None or not spec.submodule_search_locations:
        return None
    return Path(spec.submodule_search_locations[0]) / "data"


DATA = _flybrains_data()
if DATA is None or not DATA.is_dir():
    pytest.skip(
        "navis-flybrains not found (set $FASTCORE_FLYBRAINS_DATA to a source checkout's "
        "flybrains/data to run these)",
        allow_module_level=True,
    )

BSPLINES = [
    "FANC_JRCVNC2018F/TransformParameters.FixedFANC.txt",
    "FANC_JRCVNC2018F/TransformParameters.FixedTemplate.Bspline.txt",
    "BANC_JRC2018F/BANC_to_template.txt",
    "BANC_JRC2018F/3_elastix_Bspline_fine.txt",
    "BANC_JRCVNC2018F/BANC_to_template.txt",
    "BANC_JRCVNC2018F/3_elastix_Bspline_fine.txt",
]
ALL = BSPLINES + ["FANC_JRCVNC2018F/TransformParameters.FixedTemplate.affine.txt"]

AVAILABLE = [f for f in ALL if (DATA / f).is_file()]
HAVE_BSPLINE = [f for f in BSPLINES if (DATA / f).is_file()]

#: `BANC_to_template` is an extreme warp -- median displacement 163 um (ten grid cells), max 391 um
#: -- and it folds, so a slice of it has no recoverable preimage and comes back NaN. Everything
#: else inverts completely. Note flybrains never needs *this* file's inverse: it ships
#: `3_elastix_Bspline_fine.txt` for the reverse direction, and that one inverts with zero failures.
MAX_NAN_FRACTION = {"BANC_to_template.txt": 0.02}


def _image_box(path):
    """The fixed image's world box -- where the data actually is, as opposed to the
    control-point grid, which extends well outside it into territory the registration never
    constrained."""
    text = "\n".join(
        line for line in open(path) if not line.startswith("(TransformParameters ")
    )

    def key(k):
        m = re.search(rf"\({k}\s+([^)]*)\)", text)
        return np.array([float(v) for v in m.group(1).split()])

    origin = key("Origin")
    return origin, origin + key("Size") * key("Spacing")


@pytest.mark.parametrize("name", AVAILABLE)
def test_real_files_load(name):
    xf = fastcore.ElastixTransform(DATA / name)
    assert len(xf) == 1
    assert xf.kinds[0], "no steps resolved"


@pytest.mark.parametrize(
    "name", [f for f in AVAILABLE if f.endswith("3_elastix_Bspline_fine.txt")]
)
def test_four_deep_chains_resolve(name):
    """BANC's `3_fine -> 2_coarse -> 1_affine -> 0_manual` chain, each named by a bare relative
    filename and resolved against its own directory."""
    xf = fastcore.ElastixTransform(DATA / name)
    assert xf.kinds == [["linear", "linear", "bspline", "bspline"]]


@pytest.mark.parametrize("name", HAVE_BSPLINE)
def test_inverse_is_forward_consistent_on_real_data(name):
    """The guarantee, on the real thing: whatever we return is a genuine preimage.

    Round-trip identity is *not* asserted, and could not be: these warps fold, so several points
    map to the same place and no inverse can recover which one you started from.
    """
    xf = fastcore.ElastixTransform(DATA / name)
    lo, hi = _image_box(DATA / name)
    rng = np.random.default_rng(0)
    pts = rng.uniform(lo, hi, (2000, 3))

    y = xf.xform(pts)
    x = xf.xform_inv(y)

    failed = np.isnan(x).any(axis=1)
    budget = MAX_NAN_FRACTION.get(Path(name).name, 0.0)
    assert failed.mean() <= budget, (
        f"{failed.sum()}/{len(pts)} rows failed to invert (budget {budget:.0%})"
    )
    assert np.abs(xf.xform(x[~failed]) - y[~failed]).max() < 1e-3


@pytest.mark.parametrize("name", HAVE_BSPLINE)
def test_seeding_and_the_lattice_both_earn_their_keep(name):
    """The regression guard for `seed_iter` and `lattice_points`.

    Neither can be demonstrated on a small synthetic grid, so this is the only thing standing
    between them and a well-meaning simplification. Turning either off must lose points that the
    full solver keeps -- on at least one real registration.
    """
    xf = fastcore.ElastixTransform(DATA / name)
    lo, hi = _image_box(DATA / name)
    rng = np.random.default_rng(1)
    pts = rng.uniform(lo, hi, (2000, 3))
    y = xf.xform(pts)

    def n_failed(**kw):
        return int(np.isnan(xf.xform_inv(y, **kw)).any(axis=1).sum())

    full = n_failed()
    assert n_failed(seed_iter=0) >= full
    assert n_failed(lattice_points=0) >= full


def test_the_naive_start_really_does_lose_points():
    """Concretely, on FANC: seeding LM at the bare target strands it on the flat extrapolation
    plateau, because the warp displaces points by up to ~20 grid cells."""
    name = "FANC_JRCVNC2018F/TransformParameters.FixedFANC.txt"
    if name not in AVAILABLE:
        pytest.skip("FANC transform not available")

    xf = fastcore.ElastixTransform(DATA / name)
    size, origin, spacing = xf.grid_size[0], xf.grid_origin[0], xf.grid_spacing[0]
    lo, hi = origin + 1.0 * spacing, origin + (size - 2) * spacing
    rng = np.random.default_rng(1)
    pts = rng.uniform(lo, hi, (2000, 3))
    y = xf.xform(pts)

    naive = int(np.isnan(xf.xform_inv(y, seed_iter=0, lattice_points=0)).any(axis=1).sum())
    full = int(np.isnan(xf.xform_inv(y)).any(axis=1).sum())
    assert naive > 10, "the naive start was supposed to lose points here"
    assert full < naive / 4
