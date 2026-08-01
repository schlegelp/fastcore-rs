[![docs](https://github.com/schlegelp/fastcore-rs/actions/workflows/docs.yaml/badge.svg)](https://schlegelp.github.io/fastcore-rs/)

# fastcore-rs

Rust re-implementation of `navis` and `nat` core functions with a
focus on efficient algorithms for working with
[rooted trees](https://en.wikipedia.org/wiki/Tree_(graph_theory)#Rooted_tree),
a special case of directed acyclic graphs (DAG) used to represent neurons.

We provide [R](./R/nat.fastcore/) and [Python](./py) bindings.

## Documentation

The [docs](https://schlegelp.github.io/fastcore-rs/) cover all three surfaces —
the Rust core crate plus the Python and R bindings — and include a matrix of which
functions are available in which language.

## Usage

See the README for the [navis](./py) and [nat](./R/nat.fastcore/) wrappers for instructions on installation and usage.

## Versioning

The package version is tracked in a single place: `[workspace.package] version`
in the root [`Cargo.toml`](./Cargo.toml). The Rust crates inherit it via
`version.workspace = true` and the Python package via maturin
(`dynamic = ["version"]`). To bump the version, edit that one field and run:

```sh
python scripts/sync-versions.py
python scripts/bundle-r-core.py
```

Three files need a literal copy of the version, all of them under the R package,
which builds outside the workspace: `R/nat.fastcore/DESCRIPTION` (written by the
first script) and the `Cargo.toml` of both the R bindings crate and its bundled
copy of `fastcore` (written by the second). CI checks all three stay in sync via
the same scripts with `--check`.
