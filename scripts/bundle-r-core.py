#!/usr/bin/env python3
"""Bundle the `fastcore` core crate into the R package so it builds standalone.

The R bindings crate (`R/nat.fastcore/src/rust`) normally depends on the sibling
`fastcore` crate through the Cargo workspace. But R-universe (and CRAN) build the
package from a source tarball of `R/nat.fastcore/` alone, where neither the
workspace root nor the sibling `fastcore/` crate exists. To make the tarball
self-contained we keep a *committed copy* of the core crate source inside the
package at `R/nat.fastcore/src/rust/fastcore/`, and the R crate depends on it via
`fastcore = { path = "fastcore" }`.

This script refreshes that bundled copy from the canonical `fastcore/` crate and
keeps the versions in lockstep with the workspace version (the single source of
truth in the root Cargo.toml). Run it after changing anything under `fastcore/`.

Usage:
    python scripts/bundle-r-core.py           # refresh the bundled copy
    python scripts/bundle-r-core.py --check    # exit 1 if the bundle is stale
"""

import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CARGO_TOML = ROOT / "Cargo.toml"
CORE = ROOT / "fastcore"
BUNDLE = ROOT / "R" / "nat.fastcore" / "src" / "rust" / "fastcore"
R_CRATE_CARGO = ROOT / "R" / "nat.fastcore" / "src" / "rust" / "Cargo.toml"

# Directories copied verbatim from the core crate into the bundle. `fastcore.data`
# is required because fastcore/src/nblast.rs pulls the score matrices in at compile
# time via include_bytes!("../fastcore.data/...").
COPY_DIRS = ("src", "fastcore.data")


def workspace_version():
    """Read `[workspace.package] version` from the root Cargo.toml."""
    try:
        import tomllib

        with open(CARGO_TOML, "rb") as f:
            version = tomllib.load(f)["workspace"]["package"]["version"]
    except ModuleNotFoundError:  # Python < 3.11
        section = re.search(
            r"^\[workspace\.package\]$(.*?)(?=^\[|\Z)",
            CARGO_TOML.read_text(),
            flags=re.M | re.S,
        )
        match = section and re.search(
            r'^version\s*=\s*["\']([^"\']+)["\']', section.group(1), flags=re.M
        )
        if not match:
            sys.exit(f"error: no [workspace.package] version found in {CARGO_TOML}")
        version = match.group(1)
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        sys.exit(f"error: workspace version {version!r} is not of the form X.Y.Z")
    return version


def bundled_core_cargo(version):
    """The core crate's Cargo.toml with `version.workspace = true` made concrete."""
    text = (CORE / "Cargo.toml").read_text()
    new, n = re.subn(
        r"^version\.workspace\s*=\s*true\s*$",
        f'version = "{version}"',
        text,
        flags=re.M,
    )
    if n != 1:
        sys.exit(
            f"error: expected exactly one `version.workspace = true` in "
            f"{CORE / 'Cargo.toml'}, found {n}"
        )
    return new


def expected_files(version):
    """Map of bundle-relative path -> file bytes for the whole bundled copy."""
    files = {}
    files["Cargo.toml"] = bundled_core_cargo(version).encode()
    for d in COPY_DIRS:
        base = CORE / d
        for path in sorted(base.rglob("*")):
            if path.is_file():
                files[str((Path(d) / path.relative_to(base)))] = path.read_bytes()
    return files


def current_files():
    """Committed bundle contents, excluding build artifacts."""
    files = {}
    if not BUNDLE.exists():
        return files
    for path in sorted(BUNDLE.rglob("*")):
        if path.is_file() and "target" not in path.relative_to(BUNDLE).parts:
            files[str(path.relative_to(BUNDLE))] = path.read_bytes()
    return files


def r_crate_version():
    match = re.search(
        r'^version\s*=\s*"([^"]+)"', R_CRATE_CARGO.read_text(), flags=re.M
    )
    return match.group(1) if match else None


def main():
    check = "--check" in sys.argv[1:]
    version = workspace_version()
    expected = expected_files(version)
    rel = BUNDLE.relative_to(ROOT)

    stale = expected != current_files() or r_crate_version() != version

    if not stale:
        print(f"{rel}: bundled fastcore {version} (in sync)")
        return
    if check:
        sys.exit(
            f"error: {rel} is out of sync with the fastcore crate "
            f"(expected version {version}); run "
            "`python scripts/bundle-r-core.py` and commit the result"
        )

    if BUNDLE.exists():
        shutil.rmtree(BUNDLE)
    for name, data in expected.items():
        dest = BUNDLE / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)

    # Keep the R crate's own version pinned to the workspace version too.
    text = R_CRATE_CARGO.read_text()
    text = re.sub(r'^version\s*=\s*"[^"]+"', f'version = "{version}"', text, count=1, flags=re.M)
    R_CRATE_CARGO.write_text(text)

    print(f"{rel}: refreshed bundled fastcore -> {version}")


if __name__ == "__main__":
    main()
