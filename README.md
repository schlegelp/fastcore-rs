# fastcore-rs [WIP]
Rust implementation of `navis` functions. This is an
experiment to test replacing Cython with Rust for
[navis-fastcore](https://github.com/navis-org/fastcore).

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

