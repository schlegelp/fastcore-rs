# fastcore-rs [WIP]
Rust re-implementation of `navis` core functions. This is an experiment to test
replacing Cython with Rust for
[navis-fastcore](https://github.com/navis-org/fastcore) to enable cross-platform
usage of the library.

## Notes
- internally, we use `i32` to represent node indices which means we can't
  process neurons with more than 2,147,483,647 nodes (should be fine)

## TO-DOs
- [x] geodesic distances
- [x] generation of segments
- [x] Nearest-neighbor lookup (via `bosque`)
- [x] synapse flow centrality
- [ ] NBLAST
- [ ] shortest paths
- [ ] cater for `i32` node IDs (currently only `i64` supported)

## Build
1. `cd` into directory
2. Activate virtual environment: `source .venv/bin/activate`
3. Run `maturin develop` (use `maturin build --release` to build wheel)

## Test

First make sure `pytest` is installed:
```
pip install pytest -U
```

Then run the test-suite like so:
```
pytest --verbose -s
```

Note that unless you compiled with `maturin develop --release` the timings will
be much slower (up to 10x) than in a release build.
