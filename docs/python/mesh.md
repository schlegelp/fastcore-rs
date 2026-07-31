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
```

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
edges = fastcore.unique_edges(faces).astype(np.uint32)
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
survive. [`spanning_forest`](#navis_fastcore.spanning_forest) picks which way they point —
which is what turns a bag of undirected edges into something you can walk, root, or write
out as SWC. Cycles in the input are fine; each component contributes a spanning tree of
itself, so this doubles as the cycle-breaker `networkx.bfs_tree` is usually pressed into.

```python
import navis_fastcore as fastcore

# Break cycles, orient, and re-index so parents come before their children
keep = fastcore.minimum_spanning_tree(edges, n, weights=lengths)
parents, order = fastcore.spanning_forest(edges[keep], n)

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

::: navis_fastcore.spanning_forest

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
edges = edges.astype(np.uint32)
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
