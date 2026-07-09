# NBLAST

`fastcore` implements NBLAST as a pure-Rust pipeline: a Delaunay neighbourhood
graph is built per neuron (via [`shull`](https://github.com/schlegelp/shull)),
nearest neighbours are found by graph descent (via
[`aann`](https://github.com/schlegelp/aann)), and the scoring runs in Rust. The
whole all-by-all releases the GIL and parallelises across cores, streaming each
pair's nearest-neighbour result so peak memory stays low.

A "dotprop" is any object exposing `points` (an `(N, 3)` array of coordinates)
and `vect` (an `(N, 3)` array of **unit** tangent vectors). When `use_alpha` is
enabled it must additionally expose `alpha` (an `(N,)` array).

## All-by-all

```python
import navis_fastcore as fastcore

import numpy as np
from collections import namedtuple

Dotprop = namedtuple("Dotprop", ["points", "vect"])

# 10 example dotprops (use real tangent vectors in practice)
dps = [Dotprop(np.random.rand(100, 3), np.random.rand(100, 3)) for _ in range(10)]

# (10, 10) float32 matrix; row = query, column = target, diagonal = 1.0
scores = fastcore.nblast_allbyall(dps)

# Symmetric scores (also 'min' / 'max')
scores_sym = fastcore.nblast_allbyall(dps, symmetry="mean")
```

## Query vs target

```python
# (len(query), len(target)) score matrix
scores = fastcore.nblast(query_dps, target_dps)

# 'mean' / 'min' / 'max' also combine with the reverse (target-vs-query) NBLAST
scores = fastcore.nblast(query_dps, target_dps, symmetry="mean")
```

## Options

Both `nblast_allbyall` and `nblast` accept the same options:

- `smat`: the scoring matrix. `None` (default) uses the embedded FCWB matrix.
  You may also pass a navis `Lookup2d`, or a `(values, dist_edges, dot_edges)`
  tuple where the edges are the ascending **left** bin boundaries.
- `normalize` (default `True`): divide each score by the query's self-hit so a
  perfect self-match scores 1.0.
- `symmetry` (default `None`): `None` / `"forward"` returns the raw forward
  (asymmetric) matrix; `"mean"`, `"min"` or `"max"` combine it with the reverse
  direction (its transpose for `nblast_allbyall`, an explicit reverse NBLAST for
  the rectangular `nblast`).
- `use_alpha` (default `False`): weight each point's dot product by
  `sqrt(alpha_query * alpha_target)`, emphasising locally linear (backbone)
  regions. Requires each dotprop to expose a per-point `alpha`. With no explicit
  `smat`, this auto-selects the alpha-calibrated FCWB matrix (as navis does);
  an explicit `smat` is used as given.
- `limit_dist` (default `None`): a distance upper bound. A query point whose
  nearest neighbour is farther than this is scored at the matrix's "far +
  orthogonal" corner (`aann` prunes such searches). Pass a number, or `"auto"`
  for `1.05 ×` the last distance-bin edge (as in navis).
- `n_cores` (default `None`): cap the number of worker threads. `None` uses all
  available cores.
- `precision` (default `32`): dtype of the returned matrix — `16`, `32` or `64`
  (or `"half"` / `"single"` / `"double"`). The scoring math always runs in
  float64; `precision` only sets the storage width of the result.
- `progress` (default `False`): show a progress bar over the scoring pairs
  (drawn from Rust to stderr; plain reprinted text under Jupyter).
