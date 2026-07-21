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
