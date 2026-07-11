# Meshes

The one mesh routine `fastcore` provides: labelling the connected components of a
triangle mesh, i.e. finding which vertices are reachable from which through shared
faces.

```python
import navis_fastcore as fastcore
import numpy as np

# Two disconnected triangles
faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.uint32)

fastcore.mesh_connected_components(faces, n_vertices=6)
# array([0, 0, 0, 1, 1, 1], dtype=uint32)
```

::: navis_fastcore.mesh_connected_components
