# NBLAST

See [Concepts › NBLAST](../concepts/nblast.md) for how the scoring works and what
the Rust pipeline does under the hood. This page covers the Python API.

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
  the rectangular `nblast`). Symmetrising is done **in place and allocates
  nothing** — the numpy spelling `(M + M.T) / 2` would build two full `n x n`
  temporaries, and even `np.add(M, M.T, out=M)` still builds one, since numpy sees
  the output overlapping `M.T` and defensively copies. At 100k neurons that is
  80 GB of peak that no longer exists.
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

## API

<!-- `navis_fastcore.nblast` is ambiguous: the module shadows the re-exported
     function, and griffe resolves the module first. Address the function by its
     defining path and render it under its short name. -->
::: navis_fastcore.nblast.nblast
    options:
      show_root_full_path: false

::: navis_fastcore.nblast_allbyall

::: navis_fastcore.nblast_smart

::: navis_fastcore.synblast

::: navis_fastcore.Synapses

## Extracting matches

Once you have a score matrix, pulling the best matches back out of it is its own problem —
the matrices are big (a few 100k on a side is tens of GB), so the extraction must not copy
or transpose them. Three criteria, all reading the matrix at its native width (`float16`,
`float32` or `float64`) and returning plain numpy arrays:

```python
import numpy as np
import navis_fastcore as fastcore

scores = fastcore.nblast_allbyall(dps)          # (n, n) float32

# The 5 best targets per query, best first. `skip_self` drops the diagonal, which
# would otherwise be every neuron's own top hit.
indices, values = fastcore.top_matches(scores, 5, skip_self=True)

# Everything above an absolute cutoff, or within 5% of each query's own best match.
# Ragged, so returned CSR-style: query `q` owns indices[offsets[q]:offsets[q + 1]].
offsets, indices, values = fastcore.matches_above(scores, threshold=0.5)
offsets, indices, values = fastcore.matches_above(scores, percentage=0.05)
```

Expand the ragged result into a long-format table without a Python loop:

```python
counts = np.diff(offsets)
query = np.repeat(np.arange(len(counts)), counts)
rank = np.arange(len(indices)) - np.repeat(offsets[:-1], counts)
```

Notes:

- `axis=1` gives matches per *target* instead of per query. It costs no more than `axis=0`
  and does **not** transpose: the kernel walks column stripes of the row-major buffer, so
  the matrix is read exactly once either way. Passing a transposed view (`scores.T`) is
  likewise free.
- The matrix is borrowed, never copied — including a `np.memmap`. Nothing here will cast
  your dtype behind your back, so a strided view or an unsupported dtype raises rather
  than silently materialising tens of GB.
- `distances=True` if lower is better.
- `NaN` is never a match. A query with no valid scores yields `-1`/`NaN` slots
  (`top_matches`) or an empty group (`matches_above`).
- Ties break toward the lower index, so results don't depend on the thread count.
- `matches_above` counts before it allocates, so `max_matches` can refuse an over-broad
  cutoff instead of exhausting the machine. Use `count_matches` to size a result first.

::: navis_fastcore.top_matches

::: navis_fastcore.matches_above

::: navis_fastcore.count_matches

## Clustering

The other thing you do with a score matrix is cluster it. The textbook route —
symmetrise, convert similarity to distance, condense, then
`scipy.cluster.hierarchy.linkage` — allocates another `n x n` array at almost every step,
and at 100k neurons that, not the clustering, is what exhausts the machine:

```python
# What this replaces. Each line materialises a fresh n x n array.
m = (scores + scores.T) / 2
m = 1 - m
Z = linkage(squareform(m, checks=False), method="ward")
```

`fastcore.linkage` fuses the first three steps into a single pass that writes the
condensed distance vector directly, then clusters that buffer in place:

```python
import navis_fastcore as fastcore
from scipy.cluster.hierarchy import fcluster, dendrogram

scores = fastcore.nblast_allbyall(dps)          # (n, n) float32

# Symmetrise, 1 - score, condense and cluster - no n x n temporary anywhere.
Z = fastcore.linkage(scores, method="ward")

# Z is a SciPy linkage matrix, so the rest of the ecosystem just works.
labels = fcluster(Z, 10, criterion="maxclust")
```

The condensed distances are available on their own if you want them:

```python
cond = fastcore.condensed_distances(scores, symmetry="mean", transform="one_minus")
Z = fastcore.linkage(cond, method="average", copy=False)   # clusters in place
```

Notes:

- **`float32` stays `float32`.** `scipy.cluster.hierarchy.linkage` up-casts its input to
  `float64` unconditionally, so handing it a `float32` matrix to save memory instead costs
  you a second, doubled copy of the condensed matrix — plus a `bool` temporary the size of
  the input for its finiteness check. Here the condensed matrix is the only allocation, and
  the finiteness check rides along on the fused pass for free.
- Measured on a 14-core machine at `n = 40,000`, `method="ward"`, `float32` input:
  **38.2 s / 12.5 GB peak** for the numpy+SciPy pipeline versus **10.8 s / 9.6 GB** here.
  The output is the same dendrogram.
- `Z` matches SciPy's layout exactly — `(n-1, 4)`, `float64`, singletons labelled `0..n`
  and the cluster formed at step `i` labelled `n + i`, rows ordered by increasing
  distance — so `fcluster`, `dendrogram` and `cut_tree` all take it directly.
- `symmetry` mirrors `nblast_allbyall`'s: use `"none"` if the matrix is already symmetric,
  which is also the fastest path since it reads the buffer strictly sequentially.
- Clustering consumes its input as scratch. From a square matrix that never matters (the
  buffer is its own); from a condensed vector, `copy=False` clusters in place and halves
  peak memory at the cost of your array.
- The linkage itself is single-threaded and cannot be interrupted — `Ctrl-C` is honoured up
  to the end of the condensing pass only.

::: navis_fastcore.linkage

::: navis_fastcore.condensed_distances
