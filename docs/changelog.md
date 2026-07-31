# Changelog

Notable changes in `fastcore-rs`. The version number is shared across all three
surfaces — the Rust crate `fastcore`, Python's `navis-fastcore` and R's
`nat.fastcore` — so a release often touches only some of them; where that matters
it is called out.

Tags, source archives and the original announcements are on
[GitHub](https://github.com/schlegelp/fastcore-rs/releases).

## Unreleased

**Six new tree primitives**, filling the gap between what `navis` asks igraph for
internally and what `fastcore` could answer. Each looks like a general graph algorithm but
is a linear pass over the parent vector on a rooted forest, so building a graph object to
answer it costs more than the answer does. See the new
[Topology](python/topology.md) page.

- `descendants` / `paths_to_root` — the two directions of the same walk: everything below a
  node, and everything above it. `descendants` replaces igraph's
  `subcomponent(v, mode="IN")` and is what makes "cut the skeleton here" a masking
  operation rather than a graph rebuild.
- `reroot` — re-orient a forest at given nodes, reversing only the edges between each new
  root and the old one. Components nobody names are left byte-identical. Generalises
  `topo::reroot_rewire`, which takes one preferred root plus a set of new edges.
- `contract_nodes` — collapse groups of nodes onto a representative and rewire what is
  left. Edges internal to a group are dropped rather than turned into self-loops; a mapping
  that would close a cycle is refused rather than silently returning a non-forest.
- `simplify_skeleton` — keep only roots, leafs and branch points, with each replacement edge
  carrying the total length of the chain it stands in for, so cable length is preserved
  exactly. On the example skeleton that is 4332 nodes down to 1290.
- `adjacency` — the CSR triple (`indptr`, `indices`, `data`) of the skeleton's adjacency
  matrix, with column indices sorted within each row. Handing back the raw arrays rather
  than a matrix keeps this package free of a scipy dependency.

**Four more**, completing the set navis needs:

- `longest_path` / `longest_paths` — the longest path from a node to its root, and the `n`
  longest taken in turn with each peeled off before the next is sought. Not the NP-hard
  general problem: in a rooted forest every maximal path is fixed by its start node, so this
  is a distances-to-root question. Ties break towards the lowest index, matching
  `numpy.argmax`, which is what navis' implementation relies on for a stable answer.
- `betweenness` — betweenness centrality in **O(N)** rather than Brandes' O(V·E). Shortest
  paths in a tree are unique, so the count through a node is a closed form: descendants ×
  ancestors when directed, and a sum of products over the parts it separates when not.
  Counts are `int64`, because an undirected 100k-node skeleton reaches ~5e9.
- `descendant_counts` — how many nodes, or how many of a given set, lie strictly below each
  node. See the note below on why this exists.

!!! note "`betweenness` is *not* navis' `betweeness_centrality(from_=...)`"

    navis' `from_` branch does not compute betweenness at all. It walks root→source paths
    and tallies every node except the source, which counts, for each node, **how many of
    `from_` lie below it** — a descendant count. That is why `descendant_counts` is a
    separate function rather than a `sources` argument here: naming it `betweenness` would
    have made the two behaviours indistinguishable at the call site. navis'
    `find_main_branchpoint(method="betweenness")` — which is that function's *default* —
    is the one caller, and wants `descendant_counts`.

Python and Rust only for now; the R bindings will follow once the signatures have settled.

Every one of these is pinned against igraph in the parity suite and against
brute-force references under `hypothesis`, across a fixture matrix that includes
100k-deep chains — the traversals are iterative precisely because the recursive versions
segfault there.

### Breaking
- `generate_segments` now measures a segment's length from its **first node to its last**.
  It previously summed the weight vector over *every* node in the segment, including the
  terminal one — but a segment stops *at* a branch point, whose own child→parent edge
  continues into the parent segment. Every segment ending at a branch point was therefore
  one edge too long; segments ending at a root were already correct, because a root's
  weight slot is 0. Unweighted lengths change with it, from a node count to an edge count,
  so that `weights=None` stays equivalent to `weights=ones` — the same correction
  `dist_to_root` had in 0.6.0. Segments themselves are unchanged; only `lengths` moves
  (and, where lengths tie differently, the order they are sorted in). Affects all three
  surfaces.

### Fixes
- results carrying a "no such node" sentinel no longer wrap around on `uint64` node IDs.
  `geodesic_nearest`, `geodesic_farthest` and `heal_skeleton` built their output with
  `dtype=node_ids.dtype` and then wrote `-1` into it, so an unreachable source or a root
  came back as 18446744073709551615 rather than -1 — on exactly the uint64 IDs
  segmentation backends hand out. All of them now go through one helper that promotes to
  `int64` when the ID dtype cannot represent the sentinel.
- `_ids_to_indices` no longer raises on an empty ID array when the node and target dtypes
  differ — it took `max()` of both unconditionally. Reachable from any function taking an
  optional set of node IDs (`descendant_counts(targets=[])` and friends) whenever node IDs
  are `uint64` and the target array is `int64`, which is navis' normal convention.
- `geodesic_matrix(directed=True)` no longer leaks across zero-weight edges when `sources`
  or `targets` are given. The partial backend used depth as a proxy for ancestry, which
  holds only while every edge weight is strictly positive: a zero-weight edge gives an
  ancestor the *same* depth, so it slipped through the guard and was written at distance 0
  — reporting a parent as reachable from its child's direction. Coincident nodes are
  routine in traced and resampled skeletons. The all-by-all backend, `geodesic_nearest`,
  `geodesic_farthest` and `geodesic_pairs` were unaffected.
- `geodesic_nearest`, `geodesic_farthest` and `heal_skeleton` now return `-1` rather than
  `18446744073709551615` for their "no such node" sentinel when node IDs are `uint64`.
  Each built its output with `np.full(..., -1, dtype=node_ids.dtype)`, where `-1` wraps
  around; they now share the helper that already handled this for `reroot` and friends,
  which falls back to `int64` for an unsigned ID column. The sentinel marks a source with
  no reachable target (any skeleton with more than one component) and the healed
  skeleton's root, so this was easy to hit on segmentation-derived IDs.

## 0.9.0 (2026-07-28)

**`mesh.GeodesicGraph` — build the adjacency index once, query it many times.** The
[geodesic free functions](python/geodesic.md) each build an index, answer one question and
throw it away: the right trade for a single sweep, the wrong one for algorithms asking
*many small* questions of one graph. Every existing query is available as a method
(`distances`, `nearest`, `farthest`, `predecessors`, `path`, `clusters`, `components`), so
migrating is mechanical, and `subset` carves an induced subgraph out of the built CSR
instead of sending you back to numpy to mask and renumber an edge list. 500 short-path
queries on a 40k-vertex mesh run ~100x faster as methods. It buys nothing measurable when
one query already sweeps the graph.

Four operations are new, because they only make sense against a graph you keep:

- `grow` — a connected region of a fixed *number* of nodes (or of attached cloud points),
  plus each one's distance to the seed. 33x per patch on a 160k-vertex mesh; 37x over a
  53k-point cloud.
- `farthest_seed` — the next farthest-point seed, for spreading patches evenly. An
  incrementally folded and self-pruning distance field plus a lazily-corrected max-heap:
  2560 seeds on a 160k-vertex mesh, 92.5 s → 0.35 s (265x).
- `ball` — everything within a radius of a *set* of nodes, how far, and which source is
  nearest. Returns the ball itself rather than three node-sized arrays with the ball buried
  in them.
- `set_weights` — re-weight edges in place, O(edits log valence) against the O(E) of a
  rebuild. TEASAR zeroing each path it extracts is the motivating case.

Pinned against `navis`' own pure-Python implementations (`navis.ml.chunk`) and scipy. One
caveat: the search is float32 where those references are float64, so on graphs with edge
lengths that tie in float32 the settle *order* of equally-distant nodes can differ. Regions
are still exact balls — verified against a float64 oracle.

**Pyodide / JupyterLite wheels (experimental).** A `wasm32-unknown-emscripten` wheel is now
built, run against the full test suite under Pyodide in CI, and published to PyPI alongside
the native wheels under PEP 783. Emscripten cannot spawn threads, so NBLAST and the
transforms run serially there and are not interruptible — a documented degradation rather
than a runtime panic.

Python-facing release; R is unchanged.

## 0.8.0 (2026-07-28)

**NBLAST peak memory cut ~45%** — 2.22 GB → 1.26 GB on 6.9M points. Two independent
reductions: f32 coordinate storage, and `aann` 0.3.0's u32 neighbourhood graph. Scoring is
unchanged — the accumulator, tangent dot products and every score-matrix lookup still run
in f64; only storage and the descent's own distance comparisons narrow. Python selects the
width from the input dtype, so float32 `points`/`vect` now reach the index with no copy
(previously they were silently upcast, which cost *more* than passing f64). Expect ~1e-4
relative movement on scores: on 40k real neurons, 98.7% of k-NN rows are bit-identical and
top-1 agreement is 99.98%.

**`dotprops()` — tangent vectors and alpha from a bare point cloud.** An exact k-d tree
k-NN plus a parallel Jacobi symmetric-3x3 eigensolve, replacing `cKDTree.query` + N SVDs:
10.5x at N=10k, 15.5x at N=100k. This was the last thing pulling in scipy, so
`navis-fastcore` no longer imports it anywhere. Also available as `Dotprop.from_points()`.

**Graph primitives on `mesh`, straight off an edge list** — no graph object to build first:

- `level_set_components` — connected components of every label's induced subgraph in one
  DSU sweep, replacing a per-level loop (12.2 ms → 0.3 ms on a 41k-vertex mesh)
- `geodesic_path` / `geodesic_predecessors` — the route, not just its length (11.7x
  including build; 2.0x against a *cached* igraph object)
- `geodesic_clusters` — greedy partition into connected clusters of bounded geodesic radius
- `connected_components_graph`, `contract_vertices`, `minimum_spanning_tree`

New APIs are Rust + Python. R gets the synced core and the NBLAST memory work; wrappers for
the new mesh/points functions are still to come.

## 0.7.3 (2026-07-20)

### New
- thin plate spline and moving least squares [landmark transforms](python/landmarks.md)
  (`TpsTransform`, `MlsTransform`) — for when there is no registration file, only matched
  landmarks
- `nblast_knn`: k nearest neighbours without a score matrix

### Fixes
- `MlsTransform.xform` gained a `reverse` parameter
- custom scoring matrices are now checked for conformance
- picked up the upstream tie-breaker fix in `aann`
- Python: `Dotprops` is exposed at top-level

## 0.7.2 (2026-07-20)

### New
- NBLAST support functions across Rust/Python/R: `linkage` and `condensed_distances` in
  Python, `fast_hclust`, `nblast_dist` and `nblast_hclust` in R — hierarchical clustering
  straight off a score matrix, without ever casting it (at 100k a side that would quietly
  materialise tens of GB)

## 0.7.1 (2026-07-19)

### New
- `unique_edges`: the unique undirected edges of a triangle mesh

## 0.7.0 (2026-07-15)

### New
- Rust re-implementations of [CMTK](python/cmtk.md) and [Elastix](python/elastix.md)
  transforms — both several orders of magnitude faster than the original binaries
  (`streamxform` / `transformix`), and neither tool needs to be installed. Bonus:
  `fastcore` can *invert* Elastix transforms, which Elastix itself cannot.
- functions to extract top matches (N, threshold, percentile) from NBLAST matrices:
  `indices, values = fastcore.top_matches(scores, 5, skip_self=True)`

### Improvements
- another ~2x speed-up for NBLAST, from upstream improvements in
  [`aann`](https://github.com/schlegelp/aann)

## 0.6.1 (2026-07-12)

### New
- `subtree_height()` returns, for each node, the geodesic distance to the farthest leaf
- `dist_to_root()` returns, for each node, the distance to its root

## 0.6.0 (2026-07-12)

### New
- `geodesic_farthest(parents, sources, targets)` implements the opposite of
  `geodesic_nearest()`
- `geodesic_matrix_mesh` and `geodesic_farthest/nearest_mesh` compute geodesic distances on
  meshes:
    - they operate directly on faces/vertices; in `navis` this needs a costly round trip
      `mesh` → `igraph` → `csgraph.dijkstra`
    - they use threads
    - they allow defining sources *and* targets (csgraph only does sources), which cuts the
      memory footprint drastically and can be faster

### Improvements
- better performance (speed / scaling / memory) for `classify_nodes`, `strahler_order`,
  `geodesic_matrix`, `geodesic_nearest`, `all_dists_to_root`, `has_cycles`,
  `geodesic_distances`, `geodesic_pairs`, `node_indices_*` and `extract_parent_child`
- guarded `geodesic_distances` against segfaults on very large (100ks of nodes) neurons
- `geodesic_distances` now computes in `f64` internally but still writes `f32`, improving
  relative error from `3.9e-06` to `7.0e-08`

### Fixes
- `generate_segments` now returns segment lengths in the correct order

### Breaking
- `geodesic_pairs` now correctly returns `-1` for unreachable pairs (previously an
  incorrect `1`)
- `dist_to_root` now counts edges instead of nodes (previously the root was reported at
  distance `1`)

## 0.5.1 (2026-07-11)

### Fixes
- refactored `heal_skeleton` to deal with some pathological cases

## 0.5.0 (2026-07-11)

### New
- skeleton [healing](python/healing.md) functions (Rust + Python/R bindings)

## 0.4.0 (2026-07-09)

### Improvements
- NBLAST is rebased on [`aann`](https://github.com/schlegelp/aann) for all-nearest-neighbour
  lookup plus [`shull`](https://github.com/schlegelp/shull) for Delaunay triangulation —
  together a ~5x speed-up over `navis`' built-in implementation
- precompiled binaries for the R bindings are now on
  [R-universe](https://schlegelp.r-universe.dev/builds)

## 0.3.0 (2026-06-28)

### New
- `geodesic_nearest` returns, for each query node, the closest node among a set of targets
  and the distance to it. The full distance matrix is never materialised, so it scales.

## 0.2.0 (2026-06-15)

### New
- `connected_components_mesh()`: fast connected components on meshes, much faster than e.g.
  conversion to igraph and subsequent graph-based CC

### Improvements
- large performance increase for `connected_components()`, especially when the skeleton has
  many roots
- speed-up for `segment_coords()`
- technically supporting Python's free-threading (3.14+), though this is largely untested

## 0.1.0 (2025-03-13)

### Improvements
- support for 16-bit node IDs
- better handling of node and parent IDs with different dtypes
- `csgraph` drop-ins: added [documentation](python/wrappers.md) and proper checks for
  whether the input is a rooted tree

### Fixes
- fixed an incorrect parameter name in the `csgraph` drop-ins

## 0.0.9 (2025-02-06)

### New
- `generate_segments` accepts a `lengths` parameter, sorts segments by total length when
  given, and returns the segment lengths

## 0.0.8 (2025-01-05)

### New
- `prune_twigs()` gained a `mask` parameter to restrict pruning to parts of the neuron

## 0.0.7 (2024-09-16)

### Fixes
- `geodesic_matrix()` no longer returns the distance matrix in the wrong order when
  `sources` and/or `targets` are given

## 0.0.6 (2024-09-16)

### New
- `geodesic_pairs`: geodesic distances between given node pairs

## 0.0.5 (2024-09-07)

### Improvements
- more flexible about 32-bit vs 64-bit node/parent IDs
- more checks before calling into Rust, so exceptions are more helpful

## 0.0.4 (2024-07-25)

### New
- reorganised into a monorepo holding the `fastcore` Rust crate, the `navis-fastcore`
  Python bindings and the `nat.fastcore` R bindings
- `navis-fastcore`: [drop-in replacements](python/wrappers.md) for some `csgraph` functions

### Fixes
- `fastcore`: `break_segments` no longer drops the last node of a sequence

## 0.0.3a (2024-07-09)

### New
- new functions: `prune_twigs()`, `strahler_index()`, `break_segments()` and
  `classify_nodes()`

### Improvements
- `synapse_flow_centrality()`: added a `mode` parameter (centrifugal, centripetal or sum)

### Fixes
- `geodesic_matrix()` now actually respects `directed=True`

### Breaking
- the module was renamed from `fastcore` to `navis-fastcore` to avoid name clashes:

    ```bash
    pip install navis-fastcore
    ```

    ```python
    import navis_fastcore as fastcore
    ```

## 0.0.2 (2024-06-27)

### New
- [documentation](https://schlegelp.github.io/fastcore-rs/)!

### Improvements
- a Rust implementation of `geodesic_matrix` for specific `sources` and/or `targets`, which
  is faster and much more memory efficient than the all-by-all as long as the number of
  sources/targets is small-ish. `fastcore.geodesic_matrix` picks the right one for you.

### Breaking
- removed the `modifier` parameter from `fastcore.segment_coords`

## 0.0.1 (2024-06-22)

Mostly a release to test publishing via CI.
