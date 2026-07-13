from pathlib import Path

import pytest

#: Fixtures shared with the Rust test suite. They live in the core crate so that
#: `cargo test -p fastcore` stands alone; `scripts/bundle-r-core.py` copies only `src/` and
#: `fastcore.data/`, so nothing here is shipped in the R tarball.
DATA = Path(__file__).resolve().parents[2] / "fastcore" / "testdata"


@pytest.fixture(scope="session")
def cmtk_dir():
    """The real JFRC2 -> FCWB bridging registration (gzipped, 59x27x11 spline)."""
    return DATA / "JFRC2_FCWB.list"


@pytest.fixture(scope="session")
def tiny_dir():
    """A hand-written plain-text registration: 5x5x5 lattice, `absolute no`."""
    return DATA / "tiny_warp.list"


@pytest.fixture(scope="session")
def golden():
    """`streamxform`'s own output for 5 sample points, keyed by case."""
    import numpy as np

    rows = np.genfromtxt(
        DATA / "streamxform_golden.csv", delimiter=",", dtype=None, names=True, encoding="utf-8"
    )
    out = {}
    for case in np.unique(rows["case"]):
        sel = rows[rows["case"] == case]
        sel = sel[np.argsort(sel["i"])]
        out[str(case)] = np.stack([sel["x"], sel["y"], sel["z"]], axis=1)
    return out
