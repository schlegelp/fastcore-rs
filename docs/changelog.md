# Changelog

Notable changes in `fastcore-rs`. The version number is shared across all three
surfaces — the Rust crate `fastcore`, Python's `navis-fastcore` and R's
`nat.fastcore` — so a release often touches only some of them; where that matters
it is called out.

Tags, source archives and the original announcements are on
[GitHub](https://github.com/schlegelp/fastcore-rs/releases).

## Unreleased

**Mesh simplification that remembers where every vertex went.** `simplify_mesh`
decimates a triangle mesh by quadric-error edge collapse and returns, alongside the
smaller mesh, a `vertex_map`: for each vertex of the original, the index of the vertex
of the simplified mesh it ended up in, or `-1` if it did not survive.

That map is the point. Every other simplifier hands back a mesh and nothing else, so
per-vertex data — synapses, radii, compartment labels — is orphaned by the operation,
and the usual workaround of re-attaching by nearest neighbour afterwards is both slower
and wrong: a collapse moves its survivor to the quadric-optimal point, which is often
nearer some *other* vertex than the one that actually merged into it. `bincount` over
the map replaces the spatial query entirely.

`lock` pins a set of vertices: a locked vertex is never merged into another and never
moved, so it comes back at bitwise the same coordinates. It may still absorb its
neighbours, which is what keeps a face target reachable when the pinned set is large.
`simplify_mesh_lossless` is the other mode — collapse only what costs nothing, run to a
fixed point — for shedding over-tessellation rather than hitting a budget.

This is a port of Sven Forstmann's `Simplify.h` (MIT), the algorithm `pyfqmr` wraps,
rather than a wrapper around an existing crate. Wrapping was the preferred route and
none of the candidates survived two constraints. `meshopt` has exactly the right
semantics but vendors C++ and builds it through `cc`, which would cost the pyodide wheel
and the R source tarball — the same reason `flate2` and `kodama` are pinned to pure-Rust
backends. `alum` and `baby_shark` are pure Rust but built on a halfedge and a corner
table respectively, so they need manifold input: the first returns `Err(ComplexVertex)`,
the second silently drops the offending faces and returns an empty mesh on any build
error. Meshes out of EM segmentation have edges shared by three faces as a matter of
course. And none of the three expose a collapse map, so each would have needed a fork
anyway. The algorithm as written is flat index arrays with no adjacency invariants to
violate, which is what makes non-manifold input merely data.

Because it is the same algorithm, the port is checked against `pyfqmr` directly: on a
clean mesh the two produce *identical* face arrays and positions agreeing to ~1e-12,
across face-count ratios, aggressiveness settings and both border modes. Three
deliberate divergences, all about degenerate geometry: upstream normalises vectors
unconditionally, and since every comparison against the resulting NaN is false, NaN
silently defeats the two guards that exist to reject a bad collapse. Zero-area faces are
dropped, absent normals are represented rather than faked, and a collapse landing on top
of a neighbour is rejected.

Speed is a side effect rather than the aim, but it is not worse: measured end-to-end
against `pyfqmr` on UV spheres at ratio 0.1, **19.5 ms vs 30.5 ms** at 80k faces,
**88 ms vs 129 ms** at 319k, and **259 ms vs 395 ms** at 979k — about 1.5x, scaling
linearly, at roughly 4M input faces/second on one core. Single-threaded by nature —
each collapse invalidates its own neighbourhood — but the GIL is released for the
duration, so simplifying several meshes from a thread pool does scale. Deterministic
run to run; the result does depend on face order, as it does for every implementation
of this family.

No new dependencies. Available on all three surfaces as `simplify_mesh` and
`simplify_mesh_lossless`.

## 0.10.1 (2026-08-03)

**`heal_skeleton` is reproducible again.** Healing the same fragmented neuron twice
could return two different skeletons. The bridge search runs in parallel and prunes
with a per-fragment bound shared across threads; when two nodes tied for their
fragment's shortest bridge, which one got to report it came down to which thread read
that bound first. Ties are routine here, because skeleton coordinates come off a
lattice. The healed skeleton was never *wrong* — the total added cable was identical
every time, since the bound cannot drop below a fragment's true minimum — but which of
several equally short bridges it used varied from run to run.

Each query now searches just past the shared bound, so every node achieving its
fragment's minimum reports it however the threads interleave, and equal-length
candidates are settled on their endpoints. Same bridges, same total length, same
answer every run. Affects `heal_skeleton` and `stitch_fragments` on all three
surfaces.

## 0.10.0 (2026-08-01)

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

**Three graph primitives**, the ones `skeletor` still needs a graph library for. Where the
tree primitives above serve navis, these serve mesh skeletonization — see the
[Meshes](python/mesh.md) page.

- `parents_from_edges` — orient an edge list into a rooted forest: one parent per node, `-1`
  at the roots, cycles broken. `minimum_spanning_tree` picks *which* edges survive; this
  picks which way they point, which is what turns a bag of undirected edges into something
  you can walk, root or write out as SWC. It also returns the order the nodes settled in,
  which is free (the search visits them in it) and is exactly the relabelling that makes
  parents precede their children. One search covers the whole graph rather than one per
  component: the shortest-path-tree-per-component construction costs
  `O(components x n_nodes)` in *output alone*, which on a skeleton shattered into four
  thousand fragments is a 2 GB array for an answer that is one column. At 100k nodes,
  2.9 ms against igraph's 14 ms for one arbor — and 2.7 ms against 4370 ms once it
  fragments, because igraph pays per component and this does not.
- `bridges` — which edges may not be dropped without disconnecting their component
  (Tarjan, on an explicit stack, so a 200k-node strip does not overflow). The counterpart
  to `minimum_spanning_tree` rather than a variant of it: the MST asks what to keep to stay
  connected, this asks what may not be removed. Parallel edges are honoured — two nodes
  joined twice are joined by a cycle, so neither edge is a bridge, which is why this does
  not share the deduplicated adjacency the geodesic searches use. 2.6 ms against igraph's
  13.5 ms at 100k nodes, 2.2 ms against 207 ms fragmented.
- `geodesic_mst_mesh` / `geodesic_mst_graph` — the minimum spanning tree over a *subset* of
  nodes, weighted by geodesic distance through the graph they were carved out of, **without
  materialising the `k x k` distance matrix**. That matrix is `k**2` distances to use
  `k - 1` of them — 400 MB at `k = 10_000` before the `O(k^2)` MST itself, and `k` searches
  to fill it. Following Mehlhorn's distance-network construction, one multi-source sweep
  partitions every node by nearest subset member and each edge straddling two cells offers
  one candidate; an MST over those is an MST of the full distance network. The cost is flat
  in `k` because it is one sweep whatever `k` is: at 100k nodes, 12.7 ms at `k=250` and
  8.3 ms at `k=4000`, against 187 ms and 7820 ms for the matrix route. Returned weights are
  exactly the geodesic distances between the pairs they join, so they are usable as lengths.
  `limit` bounds how far apart two nodes may be and still be joined, and prunes the sweep
  rather than merely discarding results.

**The R bindings caught up.** 22 new functions in `nat.fastcore` — every tree and graph
primitive above, plus the ones that had accumulated unbound before it — taking R from 39
documented capabilities to 58, and to 66 of 77 with the clustering pair below. The
signatures are the ones the Python side settled on, translated to R conventions: 0-based node indices throughout (as the rest of
the DAG family already used), roots and "no such node" as `-1`, and multi-value results
as a named list rather than a tuple.

New optional arguments have R defaults, so `adjacency(parents)` and
`parents_from_edges(edges, n)` work without spelling out a `NULL` per argument. Note this
is *not* true of the bindings that predate this release — those still require every
argument positionally, which is worth fixing before 1.0.

Two things are worth doing before 1.0 alongside that. Argument errors currently reach R
as `Error: User function panicked: <name>`, because extendr discards the panic payload
and the R layer does no validation of its own — where Python raises a message naming the
offending value. The fix belongs in the three shared converters, not per function. And
what remains unbound is `GeodesicGraph` and its methods (a stateful pointer class, so a
different kind of job) plus five of the NBLAST/matches helpers.

Every one of these is pinned against igraph in the parity suite and against
brute-force references under `hypothesis`, across a fixture matrix that includes
100k-deep chains — the traversals are iterative precisely because the recursive versions
segfault there.

**The geodesic searches run in float64 on request.** Dijkstra sums one weight per hop, so
a path of `k` hops carries up to `k` roundings; at float32 and `k` in the tens of
thousands — a densely sampled arbor, a fine mesh — the drift becomes visible against an
exact answer, and comparisons against `scipy.sparse.csgraph`, which works in float64
unconditionally, stop agreeing to the last bits. It also matters when weights span a wide
dynamic range, since `fl(du + w)` loses `w` entirely once `du` exceeds it by 2^24.

The width is now a type parameter on the core rather than baked into it: `Adjacency`,
the search scratch, both kernels and every driver in `mesh` are generic over a new
`mesh::Weight` trait, implemented for `f32` and `f64`. The heap key stays an integer
compare on the raw IEEE bits — see `Weight::Bits` for why that is an exact ordering and
not an approximation — so the float32 path is unchanged, in both results and speed.

In Python the rule is **your dtype in, your dtype out**, the one
[`linkage`](python/wrappers.md) already follows for score matrices: float64 weights give
float64 distances, anything else gives float32.

```python
fastcore.geodesic_matrix_graph(edges, n, weights=w.astype(np.float64))  # -> float64
fastcore.geodesic_matrix_graph(edges, n, weights=w, dtype=np.float64)   # -> float64
```

A new `dtype` argument overrides that in either direction, on
`geodesic_matrix_graph`, `geodesic_matrix_mesh`, `geodesic_nearest_mesh`,
`geodesic_farthest_mesh`, `geodesic_predecessors` and the two `geodesic_mst_*`. Only
something carrying a float64 *dtype* counts as having asked: a list of Python floats does
not, because `np.asarray([1.0, 2.0])` is float64 by numpy's default rather than by
anyone's intent, and honouring it would quietly double the output of every caller who
passes one.

!!! warning "This changes the return dtype for existing callers"

    If you already pass a float64 `weights` **array**, you were getting it cast down to
    float32 and a float32 result; you now get float64 — twice the output memory, and about
    10% slower. Pass `dtype=np.float32` to keep the old behaviour. Callers passing lists,
    int arrays or no weights at all are unaffected.

The mesh functions default to float32 and take `dtype` alone, with no input dtype read
off `vertices`. Those are *coordinates*, taken at float64 either way — each edge length is
computed from them at that width and rounded once on the way in — so reading the
distances' width off them would flip nearly every existing call to float64 and double the
largest thing this library allocates. A full `V x V` matrix is already 107 GB at
`V = 164k`.

`geodesic_path`, `geodesic_clusters`, `parents_from_edges` and `minimum_spanning_tree`
have no `dtype` argument, because none of them returns a distance — but all four honour
the weights' own width, since it decides which route or which tie wins.

`GeodesicGraph` stays float32. It is the type for "large graph, many small queries", which
is exactly the case where float32 is the right width and where doubling the several
node-sized arrays it keeps resident across a whole run would be felt.

R gets a `precision` argument (32 or 64, default 32) on the eleven corresponding
functions, matching `nblast(precision = )`. R has no float32 type, so unlike Python there
is nothing to read the choice off — weights arrive as doubles whatever the caller meant by
them, and the result goes back as doubles either way — so this buys accuracy, not a
different return type.

**Interface polish, ahead of 1.0.** Small things that cost nothing to change now and get
expensive once the API is frozen.

- **Integer returns follow one rule**, written down in the
  [Python overview](python/index.md#integer-return-dtypes): a node id is `uint32`; a node
  id needing a `-1` sentinel, or a dense label such as a cluster id, is `int32`; and a
  *position in an array you passed in* is `int64`. The point is the last — `int64` tells
  you the values index your array rather than the graph, so `nodes[out]` is the node-id
  form and `out` alone is not. Four returns moved to fit (see Breaking). The rule governs
  the index-space API; the tree functions work in *ID space* and hand back values in the
  dtype of the `node_ids` you gave them, which the overview now states explicitly.
- **`from navis_fastcore import *` exports functions only.** The package had no `__all__`,
  so `import *` took the default and dragged in seven submodule objects while dropping
  `__version__` for starting with an underscore. It is now composed from the submodules'
  own `__all__`, so a new function is exported by listing it in one place. `parent_dist`
  is exported for the first time along with it: it was public and documented but in no
  `__all__`, so `fastcore.parent_dist` did not resolve.
- **`has_cycles` is callable from Python.** The core has had it all along and R binds it,
  but on the Python side it lived in the extension module only, used internally by the
  scipy interop shim — so the one function that tells you whether the input to everything
  else is well-formed was the one you could not call. It is now
  `fastcore.has_cycles(node_ids, parent_ids)`, in ID space like the rest of the tree
  family, with a parent ID that is not a node treated as a root rather than as a cycle.
- **Four more that the Rust had and Python did not.** An audit of the crate against the
  bindings, prompted by the one above; what it turned up is now bound, tested and
  documented. Only the first needed new Rust — the rest were already compiled in and
  merely private.
    - `leaf_order` — SciPy's `leaves_list`: the order to place the leaves in so a
      dendrogram draws without crossing branches. It was the one core function with no
      pyo3 wrapper at all, which meant the one step of the clustering story that still
      required scipy was drawing it. Iterative, so a 200k-observation chain does not need
      200k stack frames, and it rejects a linkage matrix naming a cluster that does not
      exist yet rather than walking off the end of it.
    - `nblast_pairs` — NBLAST of an explicit `(query, target)` list, one score per pair
      rather than a matrix. `k` pairs cost `k` comparisons instead of
      `n_query x n_target`, which is the point when a cheaper filter has already told you
      which cells you care about. Smart NBLAST has used this internally for its
      full-resolution pass since it shipped; it is the same primitive, so a target whose
      whole column you request reproduces that column of `nblast` exactly.
    - `reroot_rewire` — turn an edited edge set back into a rooted forest. Step 2 of
      `heal_skeleton`, for callers who choose their own edges rather than taking the
      minimal bridges from `stitch_fragments`. Distinct from `reroot`, which re-orients an
      *unchanged* forest and leaves untouched components byte-identical: once the edge set
      moves there is no unchanged to preserve.
    - `symmetrize` — the in-place symmetrise, on its own. `linkage` and
      `condensed_distances` already fold it into their fused pass, so this is for when
      something *else* has to read the matrix. It is the case numpy cannot do cheaply:
      `(M + M.T) / 2` builds two full `n x n` temporaries and even
      `np.add(M, M.T, out=M)` still builds one, where this allocates nothing.
- **`CmtkRegistration.domain`**, the spline warp's domain box, which R has had as
  `cmtk_domain`. Points outside `[0, domain]` have no spline value — CMTK prints `FAILED`
  and `xform` returns `NaN` — so this is how you predict a `NaN` instead of reconstructing
  the box from `.spacing` and `.dims` yourself.
- **The same two, back the other way, in R.** `symmetrize` and `leaf_order` are now
  exported from `nat.fastcore` as well, so the clustering family reads the same on both
  surfaces. Two differences are forced by R rather than chosen: `symmetrize` returns a
  copy, because R's value semantics forbid writing to the caller's matrix (it is still
  one `n x n` against `(m + t(m)) / 2`'s two), and `leaf_order` takes an `hclust` or its
  merge matrix rather than a SciPy linkage matrix, returning a 1-based ordering in the
  same form as `hclust$order`. It agrees with `stats::hclust`'s own `order` element on
  the trees that package builds, which is what "same child order" has to mean.
- **The capability tables gained a Clustering section.** `linkage`,
  `condensed_distances`, `symmetrize` and `leaf_order` and their R counterparts had
  never appeared in them, so the one family where all three surfaces differ in the
  *object* they hand back — linkage matrix, `dist`, `hclust` — was the one you could not
  look up. Every public Python name now appears in some row.
- `GeodesicGraph.subset` validated its `nodes` argument in three places — the Python
  wrapper, the binding layer and the core. The binding-layer copy is gone; the wrapper now
  uses the same `unique=True` check every other node subset in the package goes through.

### Breaking
- **`spanning_forest` is now `parents_from_edges`.** It sat one word away from
  `minimum_spanning_tree` while answering a different question — that one picks *which*
  edges survive, this picks which way they point — and the new name says what it hands
  back, which is the parent vector the rest of the package consumes. `minimum_spanning_tree`
  keeps its name: it is what scipy, igraph and networkx all call this, including scipy's
  behaviour of returning a forest when the input is disconnected. No alias, since
  `spanning_forest` was never in a release. Affects all three surfaces.
- **Node ids come back as `uint32` where they used to be `int64`**, under the rule above:
  `unique_edges`' `edges`, `contract_vertices`, and `parents_from_edges`' `order` (the last
  of these unreleased). Their `index` / `inverse` companions stay `int64` — those are
  positions in the `3F` edge list, not node ids. Python callers who fed these straight back
  in were already coercing to `uint32`; those coercions are now no-ops rather than copies.
  R is unaffected — the bindings already narrowed to R integers.
- **`matches_above`' `indices` is now `int64`, not `uint32`.** It is a position along the
  scanned axis of the `scores` matrix you passed in — the same quantity `top_matches`
  returns, which was already `int64`. The two disagreed on the dtype of one thing. The
  ragged array is the larger of the two, so this does cost memory; consistency at the call
  site is worth more than the width. `MatchError::AxisTooLong` goes with it — it existed
  only to refuse a scanned axis longer than `u32::MAX`, which is no longer a limit.
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
  `np.full(..., -1, dtype=node_ids.dtype)` and then wrote `-1` into it, where it wraps —
  so an unreachable source or a root came back as 18446744073709551615, on exactly the
  uint64 IDs segmentation backends hand out. All of them now go through the helper that
  already handled this for `reroot` and friends, which promotes to `int64` when the ID
  dtype cannot represent the sentinel. The sentinel marks a source with no reachable
  target (so, any skeleton with more than one component) and the healed skeleton's root,
  which is what made this easy to hit.
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
