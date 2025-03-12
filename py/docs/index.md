# `navis-fastcore`

Re-implementation of [`navis`](https://github.com/navis-org/navis) core function in Rust.

Importantly, functions in `fastcore` are generalized and
do not depend on `navis` itself. That way third-party libraries
can make use of `fastcore` without the rather heavy dependency
of `navis`. Currently, `fastcore` only requires `numpy`!

In the future we might try to also provide R bindings for the
underlying Rust functions so that they can also be used in
e.g. the excellent [natverse](https://natverse.org/).

At this point `fastcore` covers two areas where re-implementation
made immediate sense:

1. [Operations on rooted tree graphs (= skeletons)](Trees/index.md)
2. [NBLAST to get around the limitations of multi-processing in Python (experimental)](NBLAST/index.md)

Details & available functions can be found the respective sections!

## Install

Pre-compiled binaries are on PyPI and can be installed via:

```bash
pip install navis-fastcore
```

Please see the [Github repo](https://github.com/schlegelp/fastcore-rs) if you need/want to build from source.

## Usage

`navis` will automatically use `fastcore` where appropriate. See the
API docs if you want to use `fastcore` directly.