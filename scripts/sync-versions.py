#!/usr/bin/env python3
"""Sync the workspace version into files that cannot inherit it from Cargo.

The single source of truth for the package version is `[workspace.package]
version` in the root Cargo.toml. The Rust crates inherit it via
`version.workspace = true` and the Python package via maturin's
`dynamic = ["version"]`. The R package's DESCRIPTION file, however, must
contain a literal version string — this script keeps it in sync.

To bump the version:
    1. Edit `version` under `[workspace.package]` in the root Cargo.toml.
    2. Run `python scripts/sync-versions.py` and commit the changes.

Usage:
    python scripts/sync-versions.py           # rewrite out-of-sync files
    python scripts/sync-versions.py --check   # exit 1 if anything is out of sync
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CARGO_TOML = ROOT / "Cargo.toml"
DESCRIPTION = ROOT / "R" / "nat.fastcore" / "DESCRIPTION"


def workspace_version():
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
        sys.exit(
            f"error: workspace version {version!r} is not of the form X.Y.Z; "
            "refusing to write it to the R DESCRIPTION"
        )
    return version


def main():
    check = "--check" in sys.argv[1:]
    version = workspace_version()

    # Anchored to the start of the line so e.g. `Config/rextendr/version:`
    # is never touched.
    pattern = re.compile(r"^Version:\s*(\S+)$", flags=re.M)
    text = DESCRIPTION.read_text()
    match = pattern.search(text)
    if not match:
        sys.exit(f"error: no 'Version:' line found in {DESCRIPTION}")

    if match.group(1) == version:
        print(f"{DESCRIPTION.relative_to(ROOT)}: {version} (in sync)")
        return
    if check:
        sys.exit(
            f"error: {DESCRIPTION.relative_to(ROOT)} has version {match.group(1)} "
            f"but the workspace version is {version}; "
            "run `python scripts/sync-versions.py` and commit the result"
        )
    DESCRIPTION.write_text(pattern.sub(f"Version: {version}", text))
    print(f"{DESCRIPTION.relative_to(ROOT)}: {match.group(1)} -> {version}")


if __name__ == "__main__":
    main()
