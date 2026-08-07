"""Projecting a triangle mesh into a view plane, ready to draw.

A 2-D renderer given a mesh has to project the vertices, drop the faces pointing
away from the viewer, sort what is left along the view axis and lay the survivors
out as polygons before it can hand anything to a rasteriser.
:func:`~navis_fastcore.project_mesh_2d` does all four in one parallel pass::

    rings, bbox, ix, depth, normals = fastcore.project_mesh_2d(
        vertices, faces, xy_ix=(0, 1), depth_ix=2, front=1
    )

Written as separate steps these are four walks over hundreds of megabytes that
each produce an intermediate only the next one reads - see the function's docs
for where the time actually goes.
"""

import numpy as np

from . import _fastcore
from .mesh import _prep_faces, _prep_vertices

__all__ = ["project_mesh_2d"]


def project_mesh_2d(
    vertices,
    faces,
    xy_ix=(0, 1),
    depth_ix=2,
    front=1,
    order=True,
    normals=False,
    threads=None,
):
    """Project a mesh into a view plane: cull, sort and lay out, in one pass.

    The view is axis-aligned and named by column: ``xy_ix`` are the two coordinate
    columns that make up the picture and ``depth_ix`` is the remaining,
    into-the-screen one. Coordinates are never flipped - a right-to-left view is
    the caller's business - which is why ``front`` is needed to say which end of
    the depth axis the viewer is on.

    Doing this in one pass is the point. Each step written the obvious vectorised
    way in numpy, on an 8.4M-vertex, 16.9M-face neuron:

    ==========================================  ========
    step                                        cost
    ==========================================  ========
    project to ``(V, 2)``                       76 ms
    cull                                        226 ms
    gather the kept faces                       72 ms
    gather the kept corners into ``(K, 3, 2)``  191 ms
    close each triangle into a ring             173 ms
    bounding box of the result                  534 ms
    ==========================================  ========

    None of that is arithmetic-bound: they are single-threaded walks over arrays
    far larger than any cache, and the two gathers plus the ring layout write
    900 MB between them to say something the mesh already said.

    Parameters
    ----------
    vertices :  (V, 3) array
                Vertex positions. Must be convertible to ``float64``.
    faces :     (F, 3) array
                Triangular faces given as rows of three vertex indices. Must be
                convertible to ``uint32``.
    xy_ix :     (int, int)
                Columns of ``vertices`` that make up the view plane.
    depth_ix :  int
                The remaining column, pointing into the screen. Together with
                ``xy_ix`` this must be ``0``, ``1`` and ``2`` in some order.
    front :     1 | -1
                Direction along ``depth_ix`` that points at the viewer. Faces
                pointing the other way are dropped.
    order :     bool
                Sort the survivors furthest-first and return their depths, which
                is what painting them in order needs to give correct occlusion.
                Turn off when filling the whole mesh as a single path in a single
                colour: a nonzero-winding fill is blind to the order its subpaths
                arrive in, and the sort is the most expensive step left.
    normals :   bool
                Return unit face normals, for shading. Off by default because a
                caller that is not shading has no use for them.
    threads :   int, optional
                Size of the thread pool. Defaults to all available cores.

    Returns
    -------
    rings :     (K, 4, 2) float64 array
                Each surviving face as a *closed* ring of projected corners, its
                first repeated at the end - which is what a path fill wants. For
                plain triangles take ``rings[:, :3]``, which is a view, not a copy.
                Furthest-first if ``order``, else in face order.
    bbox :      (4,) float64 array
                ``[xmin, ymin, xmax, ymax]`` over ``rings``, computed while they
                were written. All-infinite if nothing survived.
    ix :        (K,) int64 array
                Index of each surviving face in ``faces``. ``int64`` because it
                is a position in an array you passed in rather than a vertex id -
                see :doc:`the dtype rules <index>` - and because indexing with it
                is the point, which numpy widens a ``uint32`` array to do anyway.
    depth :     (K,) float64 array or None
                Mean depth of each survivor along ``depth_ix``, in the order
                ``rings`` are in. Not sign-corrected, so it can drive a colour
                ramp directly. ``None`` unless ``order``.
    normals :   (K, 3) float64 array or None
                Unit face normals, zero for a degenerate face. ``None`` unless
                ``normals``.

    Examples
    --------
    Three triangles in the ``z = 5``, ``z = 0`` and ``z = 0`` planes, the last of
    them wound the other way round. Looking down ``+z``, that last one is facing
    away and goes:

    >>> import navis_fastcore as fastcore
    >>> import numpy as np
    >>> vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
    ...                      [0., 0., 5.], [1., 0., 5.], [0., 1., 5.]])
    >>> faces = np.array([[3, 4, 5], [0, 1, 2], [0, 2, 1]], dtype=np.uint32)
    >>> rings, bbox, ix, depth, _ = fastcore.project_mesh_2d(vertices, faces)
    >>> ix
    array([1, 0])

    The viewer is at ``+z``, so the ``z = 0`` triangle is the far one and comes
    back first - painting them in that order is what gets the occlusion right:

    >>> depth
    array([0., 5.])

    Each ring is its triangle's projected corners, closed by a repeat of the first:

    >>> rings[0]
    array([[0., 0.],
           [1., 0.],
           [0., 1.],
           [0., 0.]])
    >>> bool(np.array_equal(rings[:, 3], rings[:, 0]))
    True

    The bounding box covers those rings, and ``rings[:, :3]`` is a view of the
    plain triangles, taken without copying:

    >>> bbox
    array([0., 0., 1., 1.])
    >>> rings[:, :3].base is rings
    True

    Flipping ``front`` turns the mesh around, and the face that was culled is the
    only one left:

    >>> _, _, back, _, _ = fastcore.project_mesh_2d(vertices, faces, front=-1)
    >>> back
    array([2])

    """
    vertices = _prep_vertices(vertices)
    faces = _prep_faces(faces)

    # Vectorised here rather than serially in the core, as for `smooth_mesh`: the core
    # does check, but it has to walk `3 * F` indices to do it.
    if len(faces) and faces.max() >= len(vertices):
        raise ValueError(
            f"`faces` names vertex {faces.max()}, but only {len(vertices)} were given"
        )

    xy_ix = (int(xy_ix[0]), int(xy_ix[1]))
    return _fastcore.project_mesh_2d(
        vertices,
        faces,
        xy_ix,
        int(depth_ix),
        int(front),
        bool(order),
        bool(normals),
        None if threads is None else int(threads),
    )
