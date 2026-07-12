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
