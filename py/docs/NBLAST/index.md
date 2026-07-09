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

## Smart NBLAST

`nblast_smart` is a two-pass approximation for large comparisons. It first runs a
cheap "pre-NBLAST" on **downsampled** dotprops, then keeps only the best-scoring
targets per query and recomputes *those* pairs at full resolution. Unselected cells
keep their coarse pre-pass score. This mirrors navis' `nblast_smart` (its `scores`
argument is spelled `symmetry` here, matching the other functions).

```python
# Keep the top 10% of targets per query (percentile 90) for the full pass
scores = fastcore.nblast_smart(query_dps, target_dps, t=90, criterion="percentile")

# All-by-all; also return the boolean mask of cells recomputed at full resolution
scores, mask = fastcore.nblast_smart(dps, t=90, return_mask=True)

# Other selection criteria: an absolute score threshold, or a fixed number per query
scores = fastcore.nblast_smart(dps, t=0.3, criterion="score")
scores = fastcore.nblast_smart(dps, t=10, criterion="N")
```

Extra options beyond the shared ones below: `t` / `criterion` select the candidate
targets (`"percentile"`, `"score"` or `"N"`), `downsample` (default `10`) sets the
pre-pass point stride, and `return_mask` additionally returns which cells were
recomputed. Because fastcore's dense NBLAST is already fast, `nblast_smart` pays off
mainly for **large all-by-all** comparisons where the full-resolution scoring
dominates; on small inputs the extra pre-pass can make it a wash.

## syNBLAST (synapse-based)

`synblast` compares neurons by their **synapses** (connectors) instead of their
skeleton points: for every query connector it finds the nearest target connector
*of the same type* and scores that distance through the same lookup matrix with the
dot product fixed at 1 (synapses carry no tangent vector). This mirrors navis'
`synblast` (its `scores` argument is spelled `symmetry` here).

A "synapse cloud" is any object exposing `connectors` — an `(N, 3)` or `(N, 4)`
array of `[x, y, z, (type)]`, where the optional 4th column is a numeric connector
type (e.g. `0` = presynapse, `1` = postsynapse). The `Synapses` namedtuple is a
minimal container for one.

```python
import navis_fastcore as fastcore
from navis_fastcore import Synapses

import numpy as np

# 5 neurons; each connector is [x, y, z, type] with type in {0, 1}
neurons = [
    Synapses(np.hstack([np.random.rand(200, 3) * 10, np.random.randint(0, 2, (200, 1))]))
    for _ in range(5)
]

# (5, 5) all-by-all matrix; diagonal = 1.0
scores = fastcore.synblast(neurons)

# Only compare like-typed synapses (pre-vs-pre, post-vs-post)
scores = fastcore.synblast(neurons, by_type=True)

# Query vs target, symmetric, restricted to presynapses
scores = fastcore.synblast(neurons[:2], neurons, symmetry="mean", cn_types=[0])
```

`synblast` shares `smat`, `normalize`, `symmetry`, `n_cores`, `precision` and
`progress` with `nblast` (see below), plus two synapse-specific options: `by_type`
(default `False`) restricts matches to same-type connectors, and `cn_types` keeps
only connectors whose type is in the given set before scoring. It does not take
`use_alpha` or `limit_dist` (neither applies to synapses).

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
