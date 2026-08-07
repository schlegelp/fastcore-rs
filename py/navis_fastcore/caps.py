"""Closing the holes cut into a triangle mesh.

Subsetting a mesh drops every face that loses a corner, which leaves the cut
cross-sections standing open. These four functions find those openings and
triangulate them shut; they are separate because the two ways in enter at
different points.

Whole mesh, every opening it has::

    halfedges = fastcore.boundary_halfedges(faces)
    rings, offsets = fastcore.trace_loops(halfedges)
    caps = fastcore.triangulate_rings(rings, offsets, vertices)
    faces = np.vstack((faces, caps))

Only the openings a cut is about to make, worked out *before* subsetting::

    halfedges = fastcore.exposed_halfedges(faces, dropped)
    # ... subset, then remap `halfedges` onto the surviving vertices ...
    rings, offsets = fastcore.trace_loops(halfedges)
    caps = fastcore.triangulate_rings(rings, offsets, vertices)

No vertices are ever added, only faces, so every vertex index a caller already
holds still points at what it pointed at before.
"""

import numpy as np

from . import _fastcore
from .mesh import _prep_faces, _prep_indices, _prep_vertices

__all__ = [
    "boundary_halfedges",
    "exposed_halfedges",
    "trace_loops",
    "triangulate_rings",
]


def boundary_halfedges(faces, threads=None):
    """Find every edge of a mesh that has only one face on it.

    An interior edge has two faces on it and a boundary edge has one, so this is
    a grouping of the ``3 * F`` edges the faces name. That grouping is the whole
    cost, and it is why this is here: ``np.unique(keys, return_inverse=True,
    return_counts=True)`` — the obvious way to write it — is a stable argsort,
    75 ms of an 84 ms call on a 578k-face mesh, and numpy cannot be talked out of
    it (the bare ``np.sort`` of the same keys is already 51 ms). Sorting the bare
    keys in parallel and taking a second pass to recover the direction brings
    that to 8 ms.

    Use :func:`~navis_fastcore.exposed_halfedges` instead where you already know
    which vertices are going away — it never looks at the mesh as a whole.

    Parameters
    ----------
    faces :     (F, 3) array
                Triangular faces given as rows of three vertex indices. Must be
                convertible to ``uint32``.
    threads :   int, optional
                Size of the thread pool. Defaults to all available cores.

    Returns
    -------
    halfedges : (K, 2) uint32 array
                Directed half-edges, wound the way their one remaining face
                winds them — which is what
                :func:`~navis_fastcore.trace_loops` walks and what tells
                :func:`~navis_fastcore.triangulate_rings` which way round to
                wind the cap. Rows come in the order the edges appear in the
                ``3 * F`` edge list.

    Examples
    --------
    Two triangles sharing the edge ``(1, 2)``: that one is interior, the other
    four are boundary.

    >>> import navis_fastcore as fastcore
    >>> import numpy as np
    >>> faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.uint32)
    >>> fastcore.boundary_halfedges(faces)
    array([[0, 1],
           [2, 0],
           [1, 3],
           [3, 2]], dtype=uint32)

    A closed mesh has no boundary at all:

    >>> tetra = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.uint32)
    >>> fastcore.boundary_halfedges(tetra)
    array([], shape=(0, 2), dtype=uint32)

    """
    faces = _prep_faces(faces)
    return _fastcore.boundary_halfedges(
        faces, None if threads is None else int(threads)
    )


def exposed_halfedges(faces, dropped, threads=None):
    """Find the edges a subset is about to expose.

    Call this with the *original* faces, before subsetting.

    A face survives only if all three of its corners do, so an edge ends up on a
    new boundary exactly when it loses a face to the cut but keeps one. Both
    halves of that test are local to the cut, so — unlike
    :func:`~navis_fastcore.boundary_halfedges` — this never has to group the
    edges of the whole mesh. Only a face losing *exactly one* corner can leave an
    edge behind (lose two and there is no edge left with both ends standing),
    which on a real prune is a percent or so of the faces that go.

    Edges that were already boundary are left out: they belong to openings the
    mesh came with — a neurite truncated at the edge of the dataset, say — and
    sealing those is not this function's business.

    Parameters
    ----------
    faces :     (F, 3) array
                Faces of the mesh *before* subsetting.
    dropped :   (V, ) bool array
                For each vertex, whether the subset drops it.
    threads :   int, optional
                Size of the thread pool. Defaults to all available cores.

    Returns
    -------
    halfedges : (K, 2) uint32 array
                Directed half-edges, wound the way the one face they have left
                winds them, with indices into the *original* vertices. Remap
                them onto the surviving vertices before capping.

    Examples
    --------
    Two triangles sharing the edge ``(1, 2)``. Dropping vertex 0 kills the first
    face and leaves ``(1, 2)`` newly open, wound the way the surviving face winds
    it:

    >>> import navis_fastcore as fastcore
    >>> import numpy as np
    >>> faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.uint32)
    >>> fastcore.exposed_halfedges(faces, np.array([True, False, False, False]))
    array([[2, 1]], dtype=uint32)

    Dropping vertex 3 instead costs the second face, and ``(1, 2)`` is exposed
    the other way round:

    >>> fastcore.exposed_halfedges(faces, np.array([False, False, False, True]))
    array([[1, 2]], dtype=uint32)

    """
    faces = _prep_faces(faces)

    dropped = np.ascontiguousarray(dropped, dtype=bool)
    if dropped.ndim != 1:
        raise ValueError(
            f"`dropped` must be a 1-D array of shape (V, ), got {dropped.shape}"
        )
    if len(faces) and faces.max() >= len(dropped):
        raise ValueError(
            f"`faces` references vertex {faces.max()} but `dropped` only covers "
            f"{len(dropped)} vertices"
        )

    return _fastcore.exposed_halfedges(
        faces, dropped, None if threads is None else int(threads)
    )


def trace_loops(halfedges):
    """Walk directed half-edges into closed rings.

    Greedy: at a non-manifold boundary vertex several half-edges leave at once,
    so this takes whichever is still free. Every half-edge lands in exactly one
    ring, which is what makes this cover the whole boundary — a cycle basis
    (``networkx.cycle_basis``, which is what ``trimesh.repair.fill_holes`` uses)
    quietly drops the edges that are not part of a simple cycle.

    A walk that runs into a dead end is abandoned, and so is a ring of fewer than
    three vertices. In both cases the half-edges it consumed stay consumed, so
    this always terminates — but it does mean the rings need not account for
    every half-edge handed in.

    Parameters
    ----------
    halfedges : (K, 2) array
                Directed half-edges, as returned by
                :func:`~navis_fastcore.boundary_halfedges` or
                :func:`~navis_fastcore.exposed_halfedges`.

    Returns
    -------
    rings :     (R, ) uint32 array
                Every ring's vertices, end to end.
    offsets :   (n_rings + 1, ) int64 array
                Where each ring starts and stops: ring ``i`` is
                ``rings[offsets[i]:offsets[i + 1]]``. Flat rather than a list of
                arrays because that is what
                :func:`~navis_fastcore.triangulate_rings` wants, and because a
                list of hundreds of small arrays costs more to build than the
                walk that produced it.

    Examples
    --------
    The four boundary edges of two triangles trace into one ring:

    >>> import navis_fastcore as fastcore
    >>> import numpy as np
    >>> faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.uint32)
    >>> rings, offsets = fastcore.trace_loops(fastcore.boundary_halfedges(faces))
    >>> offsets
    array([0, 4])
    >>> rings[offsets[0]:offsets[1]]
    array([0, 1, 3, 2], dtype=uint32)

    """
    halfedges = np.asarray(halfedges, dtype=np.uint32, order="C")
    if halfedges.ndim != 2 or halfedges.shape[1] != 2:
        raise ValueError(
            f"`halfedges` must be a 2-D array of shape (K, 2), got {halfedges.shape}"
        )
    return _fastcore.trace_loops(halfedges)


def triangulate_rings(rings, offsets, vertices, threads=None):
    """Triangulate boundary rings, wound against the direction they run in.

    Each ring is flattened onto a plane and ear-clipped. Three attempts, in
    order: through the ring's area-weighted (Newell) normal, then through its
    best-fit plane, and failing both a triangle fan from its first vertex — wonky
    on a non-convex opening, but always closed and always correctly wound, which
    is what everything downstream depends on. A ring only gets past the first
    attempt if the flattening self-intersects, which is what makes ear-clipping
    run out of ears part way through.

    The cap winds *against* its ring, because the ring runs the way the faces it
    still has wind it — a cap that agreed would have the two disagreeing about
    which side is out.

    Rings are independent and run one per worker.

    Parameters
    ----------
    rings :     (R, ) array
    offsets :   (n_rings + 1, ) array
                Boundary rings in the flat form
                :func:`~navis_fastcore.trace_loops` returns: ring ``i`` is
                ``rings[offsets[i]:offsets[i + 1]]``. ``offsets`` must be
                non-decreasing and run from ``0`` to ``len(rings)``.
    vertices :  (V, 3) array
                Vertex positions. Must be convertible to ``float64``.
    threads :   int, optional
                Size of the thread pool. Defaults to all available cores.

    Returns
    -------
    faces :     (M, 3) uint32 array
                New faces, indices into ``vertices``, ring by ring. A ring of
                ``k`` vertices always caps to ``k - 2`` triangles.

    Examples
    --------
    A square hole in the ``z = 0`` plane, wound counter-clockwise seen from
    ``+z`` — so the cap comes back wound the other way:

    >>> import navis_fastcore as fastcore
    >>> import numpy as np
    >>> vertices = np.array(
    ...     [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float64
    ... )
    >>> fastcore.triangulate_rings(np.arange(4), [0, 4], vertices)
    array([[0, 3, 2],
           [1, 0, 2]], dtype=uint32)

    """
    offsets = np.asarray(offsets, dtype=np.int64, order="C")
    if offsets.ndim != 1:
        raise ValueError(f"`offsets` must be a 1-D array, got {offsets.shape}")

    vertices = _prep_vertices(vertices)
    rings = _prep_indices(rings, len(vertices), "rings")

    return _fastcore.triangulate_rings(
        rings, offsets, vertices, None if threads is None else int(threads)
    )
