import numpy as np

from . import _fastcore

__all__ = [
    "mesh_connected_components",
]


def mesh_connected_components(faces, n_vertices):
    """Find connected components of a triangle mesh.

    Uses Union-Find (DSU) with path-halving. The only extra allocation is a
    single integer array of length ``n_vertices`` — no adjacency list is built.

    Parameters
    ----------
    faces :      (N, 3) array
                 Triangular faces given as rows of three vertex indices.
                 Must be convertible to ``uint32``.
    n_vertices : int
                 Total number of vertices in the mesh. Must be at least
                 ``faces.max() + 1``.

    Returns
    -------
    components : (n_vertices, ) uint32 array
                 For each vertex the index of the root vertex of its connected
                 component. Vertices that share a component will have the same
                 value (the smallest vertex index in that component).

    Examples
    --------
    Two triangles sharing an edge — one component:

    >>> import navis_fastcore as fastcore
    >>> import numpy as np
    >>> faces = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.uint32)
    >>> fastcore.mesh_connected_components(faces, n_vertices=4)
    array([0, 0, 0, 0], dtype=uint32)

    Two disjoint triangles — two components:

    >>> faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.uint32)
    >>> fastcore.mesh_connected_components(faces, n_vertices=6)
    array([0, 0, 0, 3, 3, 3], dtype=uint32)

    """
    faces = np.asarray(faces, dtype=np.uint32, order="C")

    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(
            f"`faces` must be a 2-D array of shape (N, 3), got {faces.shape}"
        )

    return _fastcore.mesh_connected_components(faces, int(n_vertices))
