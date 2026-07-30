from pathlib import Path

import pytest

import topologies

#: Fixtures shared with the Rust test suite. They live in the core crate so that
#: `cargo test -p fastcore` stands alone; `scripts/bundle-r-core.py` copies only `src/` and
#: `fastcore.data/`, so nothing here is shipped in the R tarball.
DATA = Path(__file__).resolve().parents[2] / "fastcore" / "testdata"


# --------------------------------------------------------------- skeleton topologies
#
# Built once at import: `real_swc` reads a CSV, and parametrizing on the builders
# would re-read it for every test.

SMALL = [build() for build in topologies.SMALL]
STRESS = [build() for build in topologies.STRESS]


@pytest.fixture(params=SMALL, ids=[t.name for t in SMALL])
def topo(request):
    """Each shape in the default topology matrix (see `topologies.py`)."""
    return request.param


@pytest.fixture(params=STRESS, ids=[t.name for t in STRESS], scope="session")
def stress_topo(request):
    """Shapes that only fail at scale: 100k-deep chain, 100k-wide star, big SWC."""
    return request.param


def pytest_addoption(parser):
    parser.addoption(
        "--baseline",
        action="store_true",
        default=False,
        help="Record a new performance baseline instead of checking against it.",
    )


def pytest_configure(config):
    # Also declared in `py/pyproject.toml`, but that file is only read when
    # pytest's rootdir is `py/`; registering here keeps the marker known (and the
    # "unknown mark" warning quiet) wherever pytest was invoked from.
    config.addinivalue_line(
        "markers",
        "benchmark: performance and memory regression guards (see tests/test_perf.py)",
    )


def pytest_collection_modifyitems(config, items):
    """Skip `benchmark`-marked tests unless they were explicitly asked for.

    Done here rather than with an `addopts = -m 'not benchmark'` in
    `py/pyproject.toml`: that config is only found when pytest's rootdir resolves
    to `py/`, so running `pytest` from the repository root would silently pull a
    two-minute wall-clock-gated suite into an ordinary test run. A conftest is
    loaded no matter where pytest was invoked from.
    """
    requested = config.getoption("--baseline") or "benchmark" in (
        config.getoption("-m") or ""
    )
    if requested:
        return

    skip = pytest.mark.skip(reason="benchmark: run with `-m benchmark`")
    for item in items:
        if "benchmark" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def cmtk_dir():
    """The real JFRC2 -> FCWB bridging registration (gzipped, 59x27x11 spline)."""
    return DATA / "JFRC2_FCWB.list"


@pytest.fixture(scope="session")
def tiny_dir():
    """A hand-written plain-text registration: 5x5x5 lattice, `absolute no`."""
    return DATA / "tiny_warp.list"


def _read_golden(path):
    """A `case,i,x,y,z` golden file -> {case: (N, 3) array}, rows ordered by `i`."""
    import numpy as np

    rows = np.genfromtxt(
        path, delimiter=",", dtype=None, names=True, encoding="utf-8"
    )
    out = {}
    for case in np.unique(rows["case"]):
        sel = rows[rows["case"] == case]
        sel = sel[np.argsort(sel["i"])]
        out[str(case)] = np.stack([sel["x"], sel["y"], sel["z"]], axis=1)
    return out


@pytest.fixture(scope="session")
def golden():
    """`streamxform`'s own output for 5 sample points, keyed by case."""
    return _read_golden(DATA / "streamxform_golden.csv")


@pytest.fixture(scope="session")
def elastix_dir():
    """Synthetic elastix fixtures: one file per transform type, plus a Compose chain."""
    return DATA / "elastix"


@pytest.fixture(scope="session")
def elastix_golden(elastix_dir):
    """`transformix`'s own output for 41 sample points, keyed by case."""
    return _read_golden(elastix_dir / "transformix_golden.csv")
