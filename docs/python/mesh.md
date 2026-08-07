# Meshes

Routines that treat a triangle mesh as a graph over its vertices, with the edges taken
from the faces.

## Connected components

Labelling the connected components of a triangle mesh, i.e. finding which vertices are
reachable from which through shared faces.

```python
import navis_fastcore as fastcore
import numpy as np

# Two disconnected triangles
faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.uint32)

fastcore.mesh_connected_components(faces, n_vertices=6)
# array([0, 0, 0, 3, 3, 3], dtype=uint32)
```

::: navis_fastcore.mesh_connected_components

## Geodesic distances

The mesh counterpart to [`geodesic_matrix`](dag.md), which works on skeletons. A skeleton
is a tree, so distances there come from walking to the lowest common ancestor. A mesh is a
general cyclic graph, so this runs a Dijkstra per source instead — in parallel, which is
the whole point: `scipy.sparse.csgraph.dijkstra` holds the GIL, so you cannot get that
speedup from Python by threading it yourself.

```python
import navis_fastcore as fastcore
import numpy as np

# Two triangles sharing the 1-2 edge, forming a unit square
faces = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.uint32)
vertices = np.array([[0, 0, 0],
                     [1, 0, 0],
                     [0, 1, 0],
                     [1, 1, 0]], dtype=np.float64)

fastcore.geodesic_matrix_mesh(faces, vertices)
# array([[0.       , 1.       , 1.       , 2.       ],
#        [1.       , 0.       , 1.4142135, 1.       ],
#        [1.       , 1.4142135, 0.       , 1.       ],
#        [2.       , 1.       , 1.       , 0.       ]], dtype=float32)
```

!!! warning "Mind the size of the output"

    A full `V x V` matrix is around 107 GB at V=164k, so for anything but a small mesh you
    want `sources` and/or `targets`.

    `targets` is worth calling out. `scipy.sparse.csgraph.dijkstra` has no notion of
    targets: it always materialises all `V` columns and makes you slice afterwards. Passing
    `targets` here means only those columns are ever allocated — for 200 sources and 100
    targets on a 41k-vertex mesh that is 0.03 MB instead of 70 MB.

    If you only need the *nearest* (or farthest) target, use
    [`geodesic_nearest_mesh`](#navis_fastcore.geodesic_nearest_mesh) instead — its output is
    `O(sources)` rather than `O(sources x targets)`, and it is faster too, because the
    search stops at the first target it settles.

!!! note "This is the along-edge distance"

    Shortest paths are constrained to run along mesh edges, so on a coarse mesh they
    overshoot the true surface geodesic. This is the same approximation navis makes.

::: navis_fastcore.geodesic_matrix_mesh

::: navis_fastcore.geodesic_nearest_mesh

::: navis_fastcore.geodesic_farthest_mesh

## Arbitrary graphs

The same kernel, over an explicit edge list rather than a mesh. Unlike
[`geodesic_matrix`](dag.md), this makes no tree assumption, so cycles are fine.

```python
import navis_fastcore as fastcore
import numpy as np

# A triangle. The direct 0-2 edge has weight 5, so the shortest
# path between them goes the long way round via 1.
edges = np.array([[0, 1], [1, 2], [2, 0]], dtype=np.uint32)
weights = np.array([1, 1, 5], dtype=np.float32)

fastcore.geodesic_matrix_graph(edges, 3, weights=weights)
# array([[0., 1., 2.],
#        [1., 0., 1.],
#        [2., 1., 0.]], dtype=float32)

# The distances follow the weights' dtype: float64 in, float64 out.
fastcore.geodesic_matrix_graph(edges, 3, weights=weights.astype(np.float64))
# array([[0., 1., 2.],
#        [1., 0., 1.],
#        [2., 1., 0.]])
```

See [Float return dtypes](index.md#float-return-dtypes) for when the wider width is
worth asking for, and for the `dtype` argument that overrides the default in either
direction.

::: navis_fastcore.geodesic_matrix_graph

## Graph primitives

The handful of traversal operations that mesh algorithms actually need, taken straight
off an edge list. These exist because reaching for a general-purpose graph library means
paying to *build* a graph object first — on a 41k-vertex mesh that construction alone
costs more than every query you then run against it.

```python
import navis_fastcore as fastcore
import numpy as np

# A path 0-1-2, a lone edge 3-4, and an isolated node 5
edges = np.array([[0, 1], [1, 2], [3, 4]], dtype=np.uint32)

fastcore.connected_components_graph(edges, n_nodes=6)
# array([0, 0, 0, 3, 3, 5], dtype=uint32)
```

::: navis_fastcore.connected_components_graph

### Level sets

[`level_set_components`](#navis_fastcore.level_set_components) is the one worth knowing
about. Given a label per node it finds the connected components of *every* label's
induced subgraph in one pass — which is the inner loop of wavefront-style mesh
skeletonization, where the label is a binned geodesic distance and each component is one
ring around the structure.

Done conventionally that loop costs one subgraph construction plus one component search
per distinct level. Here it is a single sweep over the edges, unioning an edge only when
its endpoints agree:

```python
import navis_fastcore as fastcore
import numpy as np

faces = ...      # your mesh
edges = fastcore.unique_edges(faces)
n = len(vertices)

# Cast a wave from vertex 0 and collapse each ring
dist = fastcore.geodesic_matrix_mesh(faces, n_vertices=n, sources=[0])[0]
rings, n_rings = fastcore.level_set_components(edges, n, dist.astype(np.int64))

# Ring ids are contiguous, so aggregating is a plain bincount
sizes = np.bincount(rings[rings >= 0], minlength=n_rings)
```

Note that `dist` is `-1` where the search could not reach, and negative labels are
*excluded* rather than grouped — so an unreachable region does not become one bogus
level.

On a 41k-vertex mesh with ~200 levels this runs in ~0.3 ms against ~12 ms for the
per-level-subgraph equivalent, on top of the ~28 ms of graph construction it avoids
entirely.

::: navis_fastcore.level_set_components

::: navis_fastcore.contract_vertices

::: navis_fastcore.minimum_spanning_tree

### Turning an edge list into a tree

[`minimum_spanning_tree`](#navis_fastcore.minimum_spanning_tree) picks *which* edges
survive. [`parents_from_edges`](#navis_fastcore.parents_from_edges) picks which way they point —
which is what turns a bag of undirected edges into something you can walk, root, or write
out as SWC. Cycles in the input are fine; each component contributes a spanning tree of
itself, so this doubles as the cycle-breaker `networkx.bfs_tree` is usually pressed into.

```python
import navis_fastcore as fastcore

# Break cycles, orient, and re-index so parents come before their children
keep = fastcore.minimum_spanning_tree(edges, n, weights=lengths)
parents, order = fastcore.parents_from_edges(edges[keep], n)

new_ids = np.empty(n, dtype=np.int64)
new_ids[order] = np.arange(n)
swc_parents = np.where(parents < 0, -1, new_ids[parents])[order]
```

`order` is the second return for exactly that last step. A node always settles after its
parent, so relabelling by it guarantees parents get lower ids than their children — the SWC
requirement — and it comes free, since the search already visits nodes in that order.

**One search, not one per component.** The obvious construction is a shortest-path tree per
component, which is what [`geodesic_predecessors`](#navis_fastcore.geodesic_predecessors)
gives you — and it costs `O(components x n_nodes)` in *output alone*. On a skeleton that
shatters into four thousand fragments that is a 2 GB array to answer a question whose answer
is one `n_nodes`-long column. Here the components are swept one after another into that
single column, so the cost is `O(V + E)` however finely the graph is fragmented:

| 100k-node graph | fastcore | igraph (BFS per component) | networkx (`bfs_tree` per component) |
|---|---|---|---|
| one arbor | 2.9 ms | 14 ms | 365 ms |
| ~4000 fragments | 2.7 ms | 4370 ms | 285 ms |

Weights are optional and change what you get: `None` gives the breadth-first tree, weights
give the shortest-path tree. Neither is the minimum spanning tree — for that, run
`minimum_spanning_tree` first and orient what it keeps, as above.

::: navis_fastcore.parents_from_edges

### Which edges are load-bearing

[`bridges`](#navis_fastcore.bridges) is the counterpart to
[`minimum_spanning_tree`](#navis_fastcore.minimum_spanning_tree) rather than a variant of
it: the MST asks which edges to *keep* to stay connected, this asks which ones may not be
*dropped*. That is the question behind "prune this graph but do not shatter it", where you
have a set of edges you would like gone and need to know which of them are load-bearing.

```python
# Drop the edges you don't want -- except the ones holding the graph together
unwanted = ...                        # bool mask over `edges`
safe = unwanted & ~fastcore.bridges(edges, n)
edges = edges[~safe]
```

Parallel edges are honoured: two nodes joined twice are joined by a cycle, so neither of
those edges is a bridge. That is why this does not share the deduplicated adjacency the
geodesic searches use — that would fuse a parallel pair into one edge and report a bridge
that is not there. Self-loops are never bridges.

It is Tarjan's algorithm on an explicit stack, so a mesh strip tens of thousands of vertices
long does not overflow anything. Against igraph's `Graph.bridges()` on a 100k-node graph:
2.6 ms against 13.5 ms for one arbor, 2.2 ms against 207 ms once it fragments.

::: navis_fastcore.bridges

### Spanning a subset by geodesic distance

Skeletonization ends up here: the mesh has been thinned to a scatter of surviving vertices
that must be rejoined *along the surface* rather than through space. That is a minimum
spanning tree over a `k`-node subset, weighted by geodesic distance in the mesh underneath.

The obvious route is to ask for the `k x k` geodesic matrix and hand it to a matrix MST.
That materialises `k**2` distances to use `k - 1` of them — 400 MB at `k = 10_000`, before
the `O(k^2)` MST itself — and needs `k` separate searches to fill.
[`geodesic_mst_mesh`](#navis_fastcore.geodesic_mst_mesh) never forms the matrix:

```python
edges, weights = fastcore.geodesic_mst_mesh(faces, keep, vertices)

# Rows index `keep`, so map them back yourself
skeleton_edges = keep[edges]
```

Following Mehlhorn's construction for the distance network, one multi-source search
partitions *every* vertex by which of `keep` is nearest, and then each mesh edge whose
endpoints fall in different cells offers one candidate: joining their two owners at
`d(u) + w(u, v) + d(v)`. An MST over those candidates is an MST of the full distance
network, so one sweep and one Kruskal replace `k` searches and a dense matrix. The returned
weights come back exactly equal to the geodesic distances between the pairs they join, so
they are usable as lengths and not merely as an ordering.

The cost is flat in `k`, because it is one sweep whatever `k` is — which is the whole shape
of the table, on a 100k-node graph:

| `k` | fastcore | `k x k` matrix + MST | matrix size |
|---|---|---|---|
| 250 | 12.7 ms | 187 ms | 0.3 MB |
| 1000 | 7.6 ms | 584 ms | 4 MB |
| 4000 | 8.3 ms | 7820 ms | 64 MB |

`limit` bounds how far apart two nodes may be and still be joined. The result is then the
MST of the graph on `nodes` keeping only pairs within `limit`, which is a *forest* when that
graph is disconnected — the same trade `scipy.sparse.csgraph.dijkstra(limit=...)` offers,
except that here it also prunes the sweep, so it buys time rather than merely discarding
results. Nodes in different components of the mesh are never joined either way.

::: navis_fastcore.geodesic_mst_mesh

::: navis_fastcore.geodesic_mst_graph

### Paths, not just distances

[`geodesic_matrix_graph`](#navis_fastcore.geodesic_matrix_graph) answers *how far*;
[`geodesic_path`](#navis_fastcore.geodesic_path) and
[`geodesic_predecessors`](#navis_fastcore.geodesic_predecessors) answer *which way*.

The motivating case is TEASAR-style skeletonization, which extracts a path, zeroes the
edge weights along it so it is free to re-traverse, and searches again — so the graph
changes between every call. That is why these take a bare edge list: there is no index
to build, and nothing to invalidate when the weights move.

```python
import navis_fastcore as fastcore
import numpy as np

edges, lengths = fastcore.unique_edges(faces, vertices)
weights = lengths.astype(np.float32)

# The route from the root to the farthest vertex...
dists, _ = fastcore.geodesic_predecessors(edges, n, weights, sources=[root])
farthest = int(np.argmax(dists[0]))
(path,) = fastcore.geodesic_path(edges, n, root, [farthest], weights=weights)

# ...and make it free to walk again
on_path = np.isin(edges, path).all(axis=1)
weights[on_path] = 0
```

Zero weights are explicitly supported. Among equal-length paths the route is picked
deterministically, so repeated runs give the same skeleton.

::: navis_fastcore.geodesic_path

::: navis_fastcore.geodesic_predecessors

### Clustering by geodesic radius

[`geodesic_clusters`](#navis_fastcore.geodesic_clusters) partitions a graph into
connected clusters of bounded radius: pick an unassigned seed, absorb everything within
`max_dist` of it that no earlier cluster claimed, repeat. Collapsing each cluster to its
centroid is a downsampling step — vertices come out spaced by roughly `max_dist`.

```python
labels, n_clusters = fastcore.geodesic_clusters(edges, n, max_dist=2.0, weights=weights)

# Contiguous labels, so the centroids are one bincount away
centers = np.zeros((n_clusters, 3))
np.add.at(centers, labels, vertices)
centers /= np.bincount(labels, minlength=n_clusters)[:, None]

# ...and the coarse graph is just the contracted edge list
coarse = fastcore.contract_vertices(edges, labels.astype(np.uint32))
```

The radius is the **true geodesic distance from the seed**, not the length of the walk
that reached it. The usual Python implementation of this is a recursive depth-first walk
that accumulates distance along its own traversal path, which both gives worse clusters
(a node close to a seed is dropped because the walk arrived the long way round) and
recurses as deep as the cluster is large.

::: navis_fastcore.geodesic_clusters

### Reusing a graph across many queries

Every function above takes an edge list and builds an adjacency index from it, answers one
question, and throws the index away. [`GeodesicGraph`](#navis_fastcore.GeodesicGraph) is the
same functionality with that build hoisted out. It carries the methods you would expect —

| method | equivalent to |
|---|---|
| `.distances()` | [`geodesic_matrix_graph`](#navis_fastcore.geodesic_matrix_graph) |
| `.nearest()` / `.farthest()` | [`geodesic_nearest_mesh`](#navis_fastcore.geodesic_nearest_mesh) / [`geodesic_farthest_mesh`](#navis_fastcore.geodesic_farthest_mesh) |
| `.predecessors()` / `.path()` | [`geodesic_predecessors`](#navis_fastcore.geodesic_predecessors) / [`geodesic_path`](#navis_fastcore.geodesic_path) |
| `.clusters()` | [`geodesic_clusters`](#navis_fastcore.geodesic_clusters) |
| `.components()` | [`connected_components_graph`](#navis_fastcore.connected_components_graph) |

— each answering exactly what its counterpart does, so migrating is a mechanical change:

```python
g = fastcore.GeodesicGraph(edges, n, weights=weights)

routes = [g.path(a, [b]) for a, b in pairs]   # one index build, not one per pair
```

**When this actually pays.** The build is O(E) over the whole graph, so hoisting it out is
worth real time exactly when each query is *small* relative to the graph — many short paths,
a `nearest` with a tight `limit`, `grow`. On a 40k-vertex mesh, 500 short-path queries run
~100x faster as methods than as free-function calls. It buys nothing measurable when a
single query already sweeps the graph: one 50-source distance matrix on that same mesh takes
90 ms either way, against a 1 ms build. Reach for the class because you have a graph and
want to stop re-passing `edges`/`weights`/`directed` to everything — and take the speedup
where the query pattern happens to earn it.

Two further methods, [`grow`](#navis_fastcore.GeodesicGraph.grow) and
[`farthest_seed`](#navis_fastcore.GeodesicGraph.farthest_seed), have no free-function
counterpart at all — they are the ones that are inherently called in a loop, and are
covered below.

`.subset(nodes)` carves out an induced subgraph without ever returning to your edge list —
masking and renumbering an edge list in numpy is both slower and easy to get subtly wrong:

```python
labels = g.components()
biggest = g.subset(labels == np.bincount(labels).argmax())
biggest.parent_nodes          # which original nodes these are
```

Note that distances inside a subset are not generally the parent's — a shortest path that
left the subset is gone. Taking a whole connected component, as above, is the case where
they agree.

### Growing fixed-size regions

[`geodesic_clusters`](#navis_fastcore.geodesic_clusters) fixes each cluster's *radius*.
When what has to be fixed is its *size* — tiling a neuron into equal-length inputs for a
neural network, say — use [`GeodesicGraph`](#navis_fastcore.GeodesicGraph) instead. It
grows outwards from a seed and stops once it has gathered the requested number of
points, so each region is the geodesic ball that happens to hold exactly that many.

This one is a class rather than a function because the calling pattern is different: a
tiling driver asks *many small* questions of one graph, and building the adjacency index
per call — O(E) over the whole graph, against a query that explores a handful of nodes —
would cost far more than the searches themselves. Build once, query in a loop:

```python
g = fastcore.GeodesicGraph(edges, n, weights=weights)

claimed = np.zeros(n, dtype=bool)
fragments = []
while not claimed.all():
    frag = g.grow(int(np.argmax(~claimed)), size=64, forbidden=claimed)
    claimed[frag] = True
    fragments.append(frag)
```

Feeding `claimed` back in as `forbidden` is what makes the fragments a genuine partition:
an already-claimed point is never collected again, and a node whose points are *all*
claimed becomes a wall, so a later fragment cannot tunnel through an earlier one and
come out somewhere unrelated. Every fragment is therefore a single connected piece.

Pass `return_distances=True` and each point's distance to the seed comes back alongside.
The search settles points in distance order, so it already holds the number and returning it
is free. It is what makes a patch *non-uniform*: thinning a grown region by radius — a dense
core with a sparse, far-reaching halo, so one point budget buys both local detail and
long-range context — needs to know each point's radius.

```python
idx, dist = g.grow(seed, size=8 * 1024, return_distances=True)   # oversized candidate pool
keep = thin_by_radius(dist)                                       # your falloff of choice
patch = idx[keep]
```

Points sharing a node share a distance *exactly*, since a point's position is its node's, so
a thinning keyed on these will not drift.

Points do not have to *be* the graph's nodes. Pass `item_nodes` to attach a cloud —
a resampled surface, say — to the graph, and growth counts and returns cloud points
while still travelling along the graph. Nodes carrying no point simply conduct, which is
what keeps a patch connected when the cloud is far sparser than the mesh beneath it:

```python
# `source_id[i]` is the mesh vertex that cloud point `i` was sampled from
g = fastcore.GeodesicGraph(edges, n_vertices, weights=lengths, item_nodes=source_id)
patch = g.grow(seed, size=1024)      # 1024 *cloud points*, geodesically connected
```

### Spreading seeds evenly

Growing regions from *arbitrary* seeds clumps them. [`GeodesicGraph.farthest_seed`](#navis_fastcore.GeodesicGraph.farthest_seed)
picks the point geodesically farthest from everything chosen so far — farthest-point
sampling — so a sequence of them tiles the graph evenly instead:

```python
g = fastcore.GeodesicGraph(edges, n, weights=weights)

chosen = np.zeros(n, dtype=bool)
patches = []
for _ in range(k):
    seed = g.farthest_seed(chosen)
    if seed is None:                     # nothing left to seed
        break
    patch = g.grow(seed, size=64)
    chosen[patch] = True                 # or just `chosen[seed] = True` to overlap less
    patches.append(patch)
```

Only points *reachable* from something already chosen are candidates, and the search only
jumps to a fresh component (largest first) once the reachable frontier is exhausted. That
rule is what stops a mesh with a few hundred disconnected specks from seeding every speck
before it returns to the main body.

The cost is what makes this usable at scale. The obvious implementation re-runs a
multi-source Dijkstra over the whole graph per seed, which is quadratic in the seed count —
placing 2560 seeds on a 160k-vertex mesh takes ~93 s through
`scipy.sparse.csgraph.dijkstra(..., min_only=True)`. Here the distance field is updated
incrementally *and* the update is pruned against the running field, so each fold costs only
the region the new seed actually claims; the same 2560 seeds take ~0.35 s.

This assumes `chosen` only ever grows between calls, which is what the loop above does. It
is allowed to shrink — the field is rebuilt when that is detected, so the answer stays
correct — but that call pays a cold start.

### Sweeping outwards from a set

[`GeodesicGraph.ball`](#navis_fastcore.GeodesicGraph.ball) answers "what is within `r` of
*any* of these nodes, how far, and which one is nearest" — one multi-source search rather
than one search per source:

```python
nodes, dist, src = g.ball(seeds, max_dist=r)
```

The query is `scipy.sparse.csgraph.dijkstra(..., min_only=True, limit=r)`, and the
difference is what comes back: the ball itself, rather than three node-sized arrays with
the ball buried in them. For a radius that covers a fraction of the graph, allocating and
filling those arrays costs more than the search does — 275 µs against 5 µs on a 26k-vertex
mesh at a radius reaching ~170 nodes.

That gap is the point, because the callers that want this ask thousands of times: walking a
mesh invalidating a neighbourhood at a time, growing regions and marking what they covered.
Ties — a node equidistant from two sources — go to whichever settled first, deterministic
but otherwise arbitrary, exactly as `min_only` is elsewhere.

### Changing the graph as you go

Some algorithms re-weight as they run. TEASAR is the standard example: it extracts the
longest path, zeroes it so later paths may re-traverse it for free, and repeats — which is
what makes it pick *branches* rather than re-walk the trunk.

[`set_weights`](#navis_fastcore.GeodesicGraph.set_weights) edits those arcs in place:

```python
g = fastcore.GeodesicGraph(edges, n, weights=lengths)
...
path = g.path(root, [farthest])[0]
g.set_weights(np.column_stack((path[:-1], path[1:])), 0.0)   # free to re-traverse
```

Rebuilding the graph to change a few hundred edges costs O(E) and gives up the reason for
holding a prepared graph at all — 728 µs against 2.6 µs for the edit, on that same mesh. It
cannot *add* an edge, only re-weight one the graph already has, since growing the adjacency
would mean rebuilding it; a pair that is not an edge raises rather than being ignored, an
edge list out of step with the graph being the likely cause.

Distances change, so the incremental field behind `farthest_seed` is discarded on each edit
— interleaving the two costs a cold start per edit. Component labels survive.

::: navis_fastcore.GeodesicGraph

## Simplification

Every mesh simplifier gives you a smaller mesh. The problem is what happens to the data
you had attached to the *old* vertices — synapses, radii, compartment labels. Decimating
a neuron mesh with `pyfqmr` or `meshopt` orphans all of it, and re-attaching by nearest
neighbour afterwards is both slower and wrong: an edge collapse moves its survivor to the
quadric-optimal point, which is frequently nearer some *other* vertex than the one that
actually merged into it.

[`simplify_mesh`](#navis_fastcore.simplify_mesh) returns that correspondence as a third
array. `vertex_map[i]` is the simplified vertex that original vertex `i` ended up in, or
`-1` if it did not survive:

```python
verts, faces, vmap = fastcore.simplify_mesh(faces, vertices, ratio=0.1)

# Push a per-vertex quantity onto the simplified mesh. `bincount` over the map is the
# whole operation -- no spatial query, no tolerance to pick.
live = vmap >= 0
counts = np.bincount(vmap[live], weights=synapses[live], minlength=len(verts))
```

The map is the *forward* direction — indexed by original vertex, valued in simplified
vertices — because that is the direction aggregation needs. It is `int32` rather than
`uint32` for the `-1`.

### Pinning vertices

Positions move under decimation, so a vertex that carried a synapse is no longer exactly
where the synapse was. Where that matters, `lock` freezes it:

```python
lock = np.zeros(len(vertices), dtype=bool)
lock[synapse_vertices] = True

verts, faces, vmap = fastcore.simplify_mesh(faces, vertices, ratio=0.1, lock=lock)
assert np.array_equal(verts[vmap[synapse_vertices]], vertices[synapse_vertices])
```

A locked vertex is never merged into another and never moved — the equality above is
bitwise, not approximate. It may still *absorb* its neighbours, which is what keeps the
face target reachable when the pinned set is large; freezing a vertex's whole one-ring
instead would stall the sweep as soon as you pinned a few thousand synapses. The floor is
that every locked vertex survives, so a target below the locked count cannot be met.

### Lossless

[`simplify_mesh_lossless`](#navis_fastcore.simplify_mesh_lossless) collapses only edges
whose quadric error is under `epsilon` and runs to a fixed point. It has no face budget:
it is for shedding over-tessellation — coplanar fans, duplicate vertices, degenerate
faces — rather than hitting a target.

"Lossless" is a claim about the *surface*, not the *outline*. A quadric measures distance
to the planes of the incident faces, and the plane of a flat patch says nothing about
where that patch ends, so on an open mesh a planar region will collapse its own boundary
inwards at zero measured cost. Pass `preserve_border=True` there.

### Notes

**Non-manifold input is fine.** Nothing here checks for manifoldness, and each collapse
guard skips what it cannot handle rather than failing. This is the reason the algorithm is
implemented here rather than wrapped from a crate: everything built on a halfedge or
corner table either refuses a mesh with an edge shared by three faces or silently drops
the offending faces, and meshes out of EM segmentation are full of them.

**It is the same algorithm `pyfqmr` runs** — a port of Sven Forstmann's `Simplify.h` — so
expect comparable speed rather than a speed-up. On a clean mesh the two agree to the face
array, which is what the test suite checks. What is new is the vertex map, the pinning,
and that no C++ toolchain is involved, which is what lets the same code build for
pyodide and for the R source tarball.

**Determinism.** Same input, same output, every run and every machine — the sweep is
single-threaded and index-ordered. The result does depend on the order of `faces`, since
triangles are visited in input order; that is normal for this family of algorithm and not
a determinism failure.

::: navis_fastcore.simplify_mesh

::: navis_fastcore.simplify_mesh_lossless

## Smoothing

The other half of mesh cleanup: moving vertices to take the noise out of a surface,
without changing how many there are or which faces they form. The face array and the
vertex order come back untouched, so anything you have indexed by vertex is still
attached to the vertex it was attached to.

```python
smoothed = fastcore.smooth_mesh(faces, vertices)
```

That default is Taubin's λ|μ filter, and it is the default because the obvious
alternative is a trap.

### Why not plain Laplacian

The plain Laplacian step — average each vertex with its neighbours, take a fraction of
the way — removes high frequencies quickly and low ones slowly. A closed surface's
enclosed volume *is* a low frequency, so it removes that too. At `lamb=0.5` and five
iterations, which is what `navis.smooth_mesh` ships today, a neuron mesh comes out having
lost **88% of its volume**.

Taubin alternates a shrinking λ pass with an inflating μ pass, tuned so the two cancel
below a cut-off frequency and reinforce above it. Same fixture, twenty iterations: it
holds its volume to within 5%.

```python
# Explicitly, if you want the Laplacian anyway
smoothed = fastcore.smooth_mesh(faces, vertices, method="laplacian", lamb=0.5,
                                iterations=5)
```

!!! warning "One Taubin iteration is a full λ/μ pair"

    Two sweeps over the mesh, not one — `trimesh.smoothing.filter_taubin` counts
    half-steps. Counting half-steps lets an odd `iterations` end on a λ pass, which is a
    shrink that nothing undoes, and the whole point of the filter is that the passes come
    in pairs. `iterations=10` here equals `iterations=20` there, and the two agree to
    ~1e-11.

`method="humphrey"` is the third option — the HC filter of Vollmer et al., which fights
shrinkage by pulling each vertex back towards where it started rather than towards a
lower frequency. It is the gentler of the two on fine detail worth keeping.

### Weights

`weights` chooses how a vertex's one-ring is averaged. The default `"uniform"` counts
every neighbour equally, which also regularises the *sampling*: where the tessellation is
uneven it slides vertices along the surface towards even spacing. Sometimes that is what
you want; often it is drift you did not ask for.

`"cotangent"` is the discrete Laplace–Beltrami operator — each edge weighted by the
cotangents of the two angles opposite it. It is a function of the surface rather than of
the triangulation, so it moves vertices along the normal and leaves them alone within the
surface. On a UV sphere, whose rings crowd together at the poles, it drifts less than half
as far as the uniform umbrella for the same amount of smoothing.

```python
smoothed = fastcore.smooth_mesh(faces, vertices, weights="cotangent")
```

Cotangents go negative on obtuse triangles, and a negative weight pushes a vertex *away*
from its neighbour, so those contributions are clamped to zero — the usual remedy. A
vertex whose weights all vanish that way falls back to the uniform umbrella. The cost is
that on a surface of mostly-obtuse triangles cotangent weighting degrades towards uniform,
which is the right way to fail.

Unlike `trimesh`, which builds its operator once from the input geometry and reuses it,
the geometry-dependent weightings here are recomputed from the current positions every
pass — the flow they are supposed to discretise rather than a snapshot taken before the
first step. It costs nothing, because the weights are never materialised at all: each is
derived as the one-ring is walked, which is about what reading it back from an array would
have cost anyway.

### Boundaries and pinning

A boundary vertex's one-ring lies entirely to one side of it, so an open mesh's rim rolls
inwards under any of these filters. `preserve_border=True` pins it — a boundary vertex
being an endpoint of an edge used by exactly one face:

```python
smoothed = fastcore.smooth_mesh(faces, vertices, preserve_border=True)
```

`lock` freezes an arbitrary set on top of that — the same name and the same meaning as
[`simplify_mesh`](#navis_fastcore.simplify_mesh)'s. A locked vertex comes back at bitwise
the same coordinates but still pulls on its neighbours, which is what makes it a boundary
condition rather than a hole:

```python
lock = np.zeros(len(vertices), dtype=bool)
lock[synapse_vertices] = True

smoothed = fastcore.smooth_mesh(faces, vertices, lock=lock, preserve_border=True)
assert np.array_equal(smoothed[lock], vertices[lock])
```

### Volume correction

`volume_correction=True` rescales the result **about its centroid** so the enclosed volume
matches the input's:

```python
smoothed = fastcore.smooth_mesh(faces, vertices, method="laplacian",
                                volume_correction=True)
```

About the centroid is the one place this deliberately differs from
`trimesh.smoothing.filter_laplacian`, and the difference is not cosmetic. Upstream
rescales by `(vol_before / vol_after) ** (1/3)` about the **origin**, which is not a shape
operation:

- **It translates the mesh.** On the 722817260 test neuron at navis' own defaults, the
  constraint displaces the result by 41 µm. The mesh is 19–26 µm across.
- **It is not translation invariant.** The same mesh smoothed at two different offsets
  comes out two different shapes; far enough from the origin the volume ratio goes
  negative and the cube root returns `NaN`.
- **It divides by the smoothed volume**, so a mesh with a hole big enough to make that
  zero is a `ZeroDivisionError` rather than a diagnostic.

The correction here also runs **once, at the end** — which is not an approximation of
running it every iteration but exactly equal to it. Every filter is an affine combination
of a vertex and a normalised average of its neighbours, and those commute with a uniform
scaling, so scaling first and smoothing lands on the same vertices as smoothing and
scaling afterwards. Upstream pays a full pass over the faces and a `(F, 3, 3)` gather per
iteration — 40% of its runtime — for a result it could have had at the end for one pass.

!!! note "When the volume is undefined"

    On a closed mesh the correction is exactly what it says. A mesh that is *not* closed
    still usually gets one, and deliberately: both measurements cone every face back to
    the same anchor, so their ratio stays a consistent measure of how much the surface
    shrank even where neither number is an enclosed volume on its own. That matters
    because meshes worth smoothing are almost never watertight — the 722817260 neuron is
    not.

    What is left is the genuinely undecidable case: the ratio of the two signed volumes is
    zero, infinite, `NaN` or negative, a flat sheet being the clean example. There the
    vertices come back smoothed but unscaled and a `RuntimeWarning` says so. Consistently
    inverted winding is *not* in that set — both volumes come out negative, the ratio is
    positive, and the correction is as valid as ever.

### Notes

**Speed.** On a 421k-vertex / 881k-face mesh, ten iterations with the volume correction:

| | |
|---|---|
| `trimesh.smoothing.filter_laplacian` | 5.42 s |
| `fastcore.smooth_mesh`, uniform | 0.03 s |
| `fastcore.smooth_mesh`, cotangent | 0.06 s |

The arithmetic was never the cost. Upstream spends 57% of its time building the operator —
`vertex_neighbors` is a list of 421k Python lists, 636 MB of heap for 10 MB of vertices —
and another 40% in the volume constraint's per-iteration `vertices[faces]` gather. The
sparse matrix–vector product itself is 42 ms of the 5.4 s.

**Non-manifold input is fine**, as for simplification. An edge shared by three faces, a
face naming the same vertex twice, a duplicated face and a vertex no face mentions are all
merely data; nothing here reads more topology than "which vertices are adjacent to which".
A vertex in no face never moves.

**Determinism.** Same input, same output, every run and at every `threads` setting. The
volume and centroid reductions are folded in fixed-size chunks and summed in order for
exactly that reason — floating-point addition is not associative, so a reduction tree that
depended on how rayon split the work would change the last bit of the scale factor between
runs.

::: navis_fastcore.smooth_mesh

## Capping holes

Subsetting a mesh drops every face that loses a corner, which leaves the cut cross-sections
standing open. These four functions find those openings and triangulate them shut. They are
separate rather than one `fill_holes` because the two ways in enter at different points:
you either have a mesh and want every hole in it closed, or you are about to cut one and
want only the holes the cut itself makes.

Only faces are ever added, never vertices. Every vertex index a caller already holds — in
its own face array, in per-vertex data, in a connector table — still points at what it
pointed at before, which is what lets the cap be applied *after* the subset rather than
during it.

### Every hole in a mesh

```python
import navis_fastcore as fastcore
import numpy as np

halfedges = fastcore.boundary_halfedges(faces)
rings, offsets = fastcore.trace_loops(halfedges)
caps = fastcore.triangulate_rings(rings, offsets, vertices)

faces = np.vstack((faces, caps))
```

### Only the holes a cut makes

Worked out on the *original* faces, before the subset, and applied after it — capping only
adds faces, so every index handed out by the subset still stands.

```python
exposed = fastcore.exposed_halfedges(faces, dropped)

# ... subset the mesh, then remap `exposed` onto the surviving vertices ...
renumber = np.full(len(vertices), -1, dtype=np.int64)
renumber[kept] = np.arange(len(kept))
exposed = renumber[exposed].astype(np.uint32)

rings, offsets = fastcore.trace_loops(exposed)
caps = fastcore.triangulate_rings(rings, offsets, new_vertices)
```

`exposed_halfedges` deliberately leaves out edges that were boundary *already*: those belong
to openings the mesh came with — a neurite truncated at the edge of the dataset, say — and
sealing those is `boundary_halfedges`' job. One consequence is worth knowing: where a cut
runs into an opening the mesh came with, what it exposes is an open chain rather than a
ring, and `trace_loops` abandons it. That hole stays open.

### Why these are here

Almost all of it is `boundary_halfedges`. Grouping the `3F` edges a face array names is the
whole cost of finding a boundary, and the obvious numpy spelling —
`np.unique(keys, return_inverse=True, return_counts=True)` — is a stable argsort: **75 ms of
an 84 ms call** on a 578k-face mesh. That is not a formulation problem. The bare `np.sort` of
the same keys is already 51 ms, so no rearrangement in numpy can win. Sorting bare `u64`
keys in parallel and taking a second pass over the faces to recover each boundary edge's
direction brings the call to 8 ms.

On a 578k-face mesh with ~23k holes punched into it, against the equivalent numpy
implementation:

| | numpy | fastcore | |
|---|---|---|---|
| `boundary_halfedges` | 89 ms | 9.1 ms | 10x |
| `trace_loops` | 38 ms | 0.67 ms | 56x |
| `triangulate_rings` | 89 ms | 0.67 ms | 132x |
| **end to end** | **224 ms** | **11 ms** | **21x** |

And the same mesh on the subset path, 400 twig cuts exposing 4.3k half-edges:

| | numpy | fastcore | |
|---|---|---|---|
| `exposed_halfedges` | 6.4 ms | 0.80 ms | 8x |
| `trace_loops` | 0.97 ms | 0.15 ms | 7x |
| `triangulate_rings` | 2.9 ms | 0.18 ms | 16x |
| **end to end** | **10.7 ms** | **0.87 ms** | **12x** |

One of those numbers deserves a caveat: `triangulate_rings`' 132x is not faster ear-clipping —
the C++ it replaces is the same algorithm — it is the disappearance of a 23,000-iteration
Python loop around it, and most of those rings are three vertices and never reach the
ear-clipper at all.

`trace_loops` is worth a word too, in the other direction. It is a sequential walk and does
not scale with cores at all; what makes it cheap is that it is proportional to the *boundary*
rather than to the mesh, and that its adjacency is a CSR keyed by vertex id rather than a hash
map of per-vertex lists. That is the difference between 0.67 ms and about 5.

### Non-manifold boundaries, and why not a cycle basis

At a non-manifold boundary vertex several half-edges leave at once. `trace_loops` is greedy:
it takes whichever is still free, so every half-edge lands in exactly one ring and the whole
boundary is covered. A cycle basis — `networkx.cycle_basis`, which is what
`trimesh.repair.fill_holes` uses — quietly drops the edges that are not part of a simple
cycle, and those holes stay open.

Being greedy means the decomposition depends on the order the half-edges arrive in, which is
why `boundary_halfedges` and `exposed_halfedges` both return theirs in `3F` edge-list order:
that is the one order that does not depend on how the parallel work happened to be split, so
the same mesh gives the same rings at every `threads` setting.

### How a ring is closed

`triangulate_rings` flattens each ring onto a plane and ear-clips it, trying three things in
order:

1. The ring's area-weighted (Newell) normal. Cheaper than a best-fit plane and, on the rings
   a cut actually produces, it fails slightly less often too.
2. The best-fit plane, from the eigenvectors of the ring's 3x3 scatter matrix.
3. A triangle fan from the ring's first vertex — wonky on a non-convex opening, but always
   closed and always correctly wound.

A ring only gets past step 1 if the flattening self-intersects, which is what makes
ear-clipping run out of ears part way through and yield fewer than the `n - 2` triangles a
simple polygon always does.

The cap winds *against* its ring. The ring runs the way the faces it still has wind it, so a
cap that agreed would have the two disagreeing about which side is out.

The ear-clipping is a Rust port of mapbox's earcut rather than a binding to it, so it needs
no extension module of its own. Against `mapbox_earcut` on the same rings it picks the same
triangles about 93% of the time and an equally valid alternative otherwise — same triangle
count, same total oriented area, same winding — so do not depend on the exact triangles, only
on the hole being closed the right way round.

One case is worth knowing about. Greedy tracing can walk back through a non-manifold boundary
vertex, which leaves a ring that names the same vertex twice — a polygon touching itself.
Neither ear-clipping attempt can find `n - 2` ears there, so the fan takes over. On a punched
neuron mesh roughly 10% of rings are like this, and `mapbox_earcut` can loop **forever** on
the best-fit-plane retry one of them provokes; this implementation returns the fan.

::: navis_fastcore.boundary_halfedges

::: navis_fastcore.exposed_halfedges

::: navis_fastcore.trace_loops

::: navis_fastcore.triangulate_rings

## Projecting for a 2-D renderer

Drawing a mesh flat — what `navis.plot2d` does — takes four steps before a rasteriser
sees anything: project the vertices onto the view plane, drop the faces pointing away
from the viewer, sort what is left along the view axis so that painting it gives correct
occlusion, and lay the survivors out as polygons. `project_mesh_2d` does all four in one
parallel pass.

```python
rings, bbox, ix, depth, normals = fastcore.project_mesh_2d(
    vertices, faces, xy_ix=(0, 1), depth_ix=2, front=1
)
```

The view is axis-aligned and named by column: `xy_ix` are the two coordinate columns that
make up the picture, `depth_ix` is the remaining, into-the-screen one, and `front` says
which end of that axis the viewer is on — coordinates are never flipped, so a
right-to-left view is the caller's business, not the projection's.

Fusing the steps is the whole point. Each one written the obvious vectorised way in numpy,
on an 8.4M-vertex, 16.9M-face neuron:

| step | cost |
|---|---|
| project to `(V, 2)` | 76 ms |
| cull | 226 ms |
| gather the kept faces | 72 ms |
| gather the kept corners into `(K, 3, 2)` | 191 ms |
| close each triangle into a ring | 173 ms |
| bounding box of the result | 534 ms |
| **total** | **1.27 s** |

None of that is arithmetic-bound: they are single-threaded walks over arrays far larger
than any cache, and the two gathers plus the ring layout write 900 MB between them to say
something the mesh already said. Fused, it is **133 ms**, and the bounding box comes out
of the same pass that wrote the rings.

The cull is the part worth explaining. Whether a face points at the viewer is the sign of
its normal's depth component, and that component is a 2x2 determinant of the two *other*
columns of the edge vectors — which, for an axis-aligned view, are exactly the columns
being projected onto. So it never forms the other two components of the cross product and
never reads the depth column. It is the same test the full cross product applies, to the
bit.

Faces come back as **rings**: four points each, the first repeated at the end. That is what
a path fill wants — a closed subpath, no separate close-path instruction — and `rings[:, :3]`
is a view of the plain triangles, not a copy. Emitting triangles and closing them afterwards
is the 173 ms row above plus a second buffer the size of the first.

Two things are optional because most callers do not read them. `order=False` skips the sort
and the depths: a mesh filled as one path in one colour is drawn under the nonzero winding
rule, which is blind to the order its subpaths arrive in, so the sort cannot change a pixel.
`normals=False` skips the per-face normals, which only a caller that is shading has any use
for.

Smooth shading is the one thing this does not do. Averaged vertex normals take a
contribution from every face, back-facing ones included, so they cannot be computed from the
survivors alone; a caller that wants them has to accumulate them itself.

::: navis_fastcore.project_mesh_2d
