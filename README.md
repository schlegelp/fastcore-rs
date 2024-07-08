[![docs](https://github.com/schlegelp/fastcore-rs/actions/workflows/docs.yaml/badge.svg)](https://schlegelp.github.io/fastcore-rs/)

# fastcore-rs
Rust re-implementation of `navis` core functions. Check out the [documentation](https://schlegelp.github.io/fastcore-rs/) for details.

## TO-DOs
- [x] geodesic distances
- [x] connected components
- [x] generation of linear segments
- [x] synapse flow centrality
- [x] Strahler index
- [ ] flow centrality
- [ ] CI tests
- [ ] NBLAST (started prototype)
- [ ] shortest paths
- [ ] classify nodes
- [ ] cater for `i32` node IDs which are currently cast up to `i64`
- [ ] faster version of `navis.connected_subgraph`

### Additional Notes
- internally, we use `i32` to represent node indices which means we can't
  process neurons with more than 2,147,483,647 nodes (should be fine though)

## Install

We provide precompiled binaries for all major Python versions, CPU architectures and operating systems.

From [PyPI](https://pypi.org/project/navis-fastcore):

```bash
pip install navis-fastcore
```

If that fails, try building from source (see below).

## Usage

`navis` will automatically use `fastcore` if it is available.
The integration is still work in progress, so for now you
should install `navis` from Github to make sure you have the
latest version.

## Build

1. [Install rust](https://www.rust-lang.org/tools/install)
2. Clone this repository
3. `cd` into `fastcore-rs` directory
4. Create a virtual environment: `python3 -m venv .venv`
5. Activate virtual environment: `source .venv/bin/activate`
6. Compile via either:
   - `maturin develop --release` which will compile the
     extension into the `fastcore/` directory
   - `maturin build --release` to build wheel in `/target/wheels`
7. To install the Python package either do:
  -  `pip install -e .` to install in editable mode
  - `pip install targets/wheels/navis_fastcore....whl`

Note that unless you compiled with the `--release` flag,
timings will be much slower (up to 10x) than in a release build!

## Test
First make sure `pytest` is installed:
```
pip install pytest -U
```

Then run the test-suite like so:
```
pytest --verbose -s
```
