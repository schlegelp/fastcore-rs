# fastcore-rs [WIP]
Rust re-implementation of `navis` core functions. This is an experiment to test
replacing Cython with Rust for
[navis-fastcore](https://github.com/navis-org/fastcore) to enable cross-platform
usage of the library.

## TO-DOs
- [x] geodesic distances
- [x] generation of segments
- [x] Nearest-neighbor lookup (via `bosque`)
- [ ] shortest paths

## Build
1. `cd` into directory
2. Activate virtual environment: `source .venv/bin/activate`
3. Run `maturin develop`

## Test

First make sure `pytest` is installed:
```
pip install pytest -U
```

Then run the test-suite like so:
```
pytest --verbose -s
```

