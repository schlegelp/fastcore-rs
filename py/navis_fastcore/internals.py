"""Stripping the surface a neuron mesh keeps on its inside.

An *invagination* is a piece of membrane that bulges into the cell — typically
the boundary of a mitochondrion or a vesicle touching it from within. Segmented
meshes are full of them and they are ruinous for anything that walks the
surface: each one is a tunnel the wave front of a skeletonisation can take a
shortcut through, or split on.

The property that defines an invagination — the cell encloses it — is also what
makes it invisible from outside, which is what :func:`drop_internals` exploits::

    vertices, faces, keep, passes = fastcore.drop_internals(vertices, faces)

:func:`openness` is that test on its own, for callers who want the field rather
than the repair — to colour a mesh by it, or to pick a threshold.

"""

import warnings

from . import _fastcore
from .mesh import _prep_mask, _prep_mesh_edit

__all__ = ["openness", "drop_internals"]


def openness(
    vertices,
    faces,
    mask=None,
    n_rays=16,
    offset=None,
    seed=1985,
    threads=None,
):
    """Fraction of rays leaving each face that escape the mesh.

    For every face, fire a cosine-weighted spray of rays into the hemisphere
    above it and count how many get away. Outer membrane lands at 0.5-1.0, the
    wall of an invagination at 0, and the distribution in between is sparse —
    so the threshold that separates the two is not really a tuning parameter.

    Rays are cast as segments long enough to leave the bounding box, which is as
    good as infinity here, and only ever asked whether they hit *anything*. That
    is a much cheaper question than a collision library's "what did I hit
    first", and it is why this does not use one.

    Parameters
    ----------
    vertices :  (V, 3) array
                Vertex positions.
    faces :     (F, 3) array
                Triangular faces given as rows of three vertex indices.
    mask :      (F, ) bool array, optional
                Cast only from these faces. The whole mesh still blocks rays —
                only the *sources* are restricted — so each value is exactly the
                one a full sweep would have produced.
    n_rays :    int
                Rays per face. 16 is plenty: the signal is bimodal, so this only
                has to resolve "none got out" from "some did". 8 halves the cost
                for no measured difference; 4 is visibly past the edge.
    offset :    float, optional
                How far off the surface to start each ray. Defaults to 5% of the
                median edge length, which keeps this scale-free — neuron meshes
                turn up in nanometres and in microns.
    seed :      int
                Fixes the spray. Drawn by hashing (face, ray) rather than from a
                stream, so the answer does not depend on the thread count, nor
                on whether this was a full sweep or a partial one.
    threads :   int, optional
                Size of the thread pool. Defaults to all available cores.

    Returns
    -------
    (F, ) float64 array
                Fractions in [0, 1] — or one value per selected face, in face
                order, if ``mask`` was given. A degenerate face has no
                hemisphere to sample and comes back as 1.0.

    Examples
    --------
    >>> import navis_fastcore as fc
    >>> import numpy as np
    >>> # A tetrahedron: every face is on the outside.
    >>> vertices = np.array([[0., 0., 0.], [1., 0., 0.],
    ...                      [0., 1., 0.], [0., 0., 1.]])
    >>> faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    >>> bool((fc.openness(vertices, faces) > 0.5).all())
    True

    """
    faces, vertices, _ = _prep_mesh_edit(faces, vertices, None)
    return _fastcore.openness(
        vertices,
        faces,
        # Per *face*, unlike the per-vertex masks the mesh-editing entry points take.
        mask=_prep_mask(mask, len(faces), "mask"),
        n_rays=int(n_rays),
        offset=None if offset is None else float(offset),
        seed=int(seed),
        threads=None if threads is None else int(threads),
    )


def drop_internals(
    vertices,
    faces,
    threshold=0.05,
    n_rays=16,
    smooth=10,
    iterations=3,
    hops=20,
    seed=1985,
    threads=None,
):
    """Strip invaginated surface from a mesh and close it back up.

    One pass scores every face by :func:`openness`, diffuses that field over the
    faces, drops what falls below ``threshold``, drops the components left
    enclosing no volume — the inside-out shreds of pocket wall — and caps the
    holes. Each component is closed before its volume is read, since the cut has
    just opened every one it touched and an open surface has no volume to read —
    and a component the capping cannot close is kept rather than guessed at.
    Then it repeats, because capping a pocket mouth turns a partially-open
    neighbour into a fully buried one. Passes converge fast: 18.7% of faces
    buried, then 0.5%, then 0.2%.

    Only vertices are ever *removed*; caps re-use the ones already on the
    boundary. So ``keep`` is all a caller needs to carry a vertex map,
    connectors or any other per-vertex annotation across the repair. The price
    of that rule is that a cap's chords run between vertices the mesh already
    has, so one occasionally lands on an edge the mesh already has and leaves it
    carrying four faces: the mesh comes back closed, but not manifold.

    Note this also closes openings the mesh arrived with — a neurite truncated
    at the edge of the dataset, say. For skeletonisation that is what you want:
    a stump left open splits the wave front just as a pocket mouth does.

    .. warning::

       Faces must be wound **outward**. Rays are fired into the hemisphere the
       face normal points into, so a consistently inward-wound mesh reads as
       entirely buried and comes back empty. A mesh wound *inconsistently* is
       worse, because it fails quietly: the faces that disagree read as buried
       and are cut out of otherwise healthy membrane.

    Parameters
    ----------
    vertices :  (V, 3) array
                Vertex positions.
    faces :     (F, 3) array
                Triangular faces given as rows of three vertex indices.
    threshold : float
                Faces whose smoothed openness falls below this are cut. The
                operating range is 0.05-0.10; the buried mode sits at exactly
                zero, so any small value works, while above ~0.1 the cut starts
                eating real membrane. What says so is the boundary-edge count,
                which is flat across 0.02-0.10 and then explodes — that is the
                cut outrunning the capping.
    n_rays :    int
                Rays per face — see :func:`openness`.
    smooth :    int
                Diffusion rounds applied to the field before thresholding.
                Thresholding a raw field cuts along a ragged contour: on a 3.5 M
                face mesh this takes the result from 7,339 holes with 54,112
                boundary vertices to 513 with 12,094. It moves the contour the cut
                starts from, not how far the cut reaches — the extent comes off the
                raw field, so a pocket narrower than this reaches is still cut whole.
    iterations : int
                Passes of the whole cycle.
    hops :      int, optional
                How far past the previous pass's caps to re-cast. A pass only
                changes the mesh where the last one cut, so the field is only
                stale near the new caps — 97% of what pass 1 finds sits within
                20 faces of one. ``None`` re-casts everything, which costs about
                twice as much and moves no mesh metric more than a percent.
                Worth keeping above ``smooth``, which is how far a stale value
                at the rim can diffuse inward.
    seed :      int
                Fixes the ray spray.
    threads :   int, optional
                Size of the thread pool. Defaults to all available cores.

    Returns
    -------
    vertices :  (V', 3) float64 array
    faces :     (F', 3) uint32 array
    keep :      (V', ) uint32 array
                Index of each surviving vertex in the input.
    passes :    list of dict
                One per pass actually run, with ``faces_before``, ``buried``,
                ``recast``, ``dropped``, ``spared``, ``unclosed``, ``capped``
                and ``faces_after``.

    Warns
    -----
    UserWarning
                If a pass kept surface it could not vouch for. Either
                ``unclosed``, meaning the capping could not close a component
                and so whether it encloses anything was never decided — an open
                surface has no volume to read; or ``spared``, meaning the step
                was about to remove more than half of the surface it judged
                and was held back, because a cell is not mostly cavity and a mesh
                that keeps a few scraps beats a neuron thrown away. Both come back
                repaired, and both are worth looking at: the usual causes are
                boundary rings that pinch, and faces that are not wound outward
                or not wound consistently.

    Examples
    --------
    >>> import navis_fastcore as fc
    >>> import numpy as np
    >>> # A cube with a second, smaller cube floating inside it - the simplest
    >>> # thing that looks like an organelle. Wound outward, as required.
    >>> def cube(lo, hi):
    ...     v = np.array([[lo, lo, lo], [hi, lo, lo], [hi, hi, lo], [lo, hi, lo],
    ...                   [lo, lo, hi], [hi, lo, hi], [hi, hi, hi], [lo, hi, hi]],
    ...                  dtype=float)
    ...     f = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
    ...                   [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
    ...                   [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5]])
    ...     return v, f
    >>> v1, f1 = cube(0, 10)
    >>> v2, f2 = cube(4, 6)
    >>> vertices = np.vstack((v1, v2))
    >>> faces = np.vstack((f1, f2 + len(v1)))
    >>> v, f, keep, passes = fc.drop_internals(vertices, faces)
    >>> len(f)  # the inner cube is gone; the outer one is untouched
    12
    >>> passes[0]["buried"]  # the inner cube's 12 faces, and only those
    12

    """
    faces, vertices, _ = _prep_mesh_edit(faces, vertices, None)
    out = _fastcore.drop_internals(
        vertices,
        faces,
        threshold=float(threshold),
        n_rays=int(n_rays),
        smooth=int(smooth),
        iterations=int(iterations),
        hops=None if hops is None else int(hops),
        seed=int(seed),
        threads=None if threads is None else int(threads),
    )
    # Both of these mean the same thing to a caller — the repair kept surface it could not
    # vouch for — but they have different causes and different answers, so they say so
    # separately. Silence would be the failure mode worth avoiding: the mesh comes back
    # looking repaired either way.
    for i, p in enumerate(out[3]):
        if p["unclosed"]:
            warnings.warn(
                f"drop_internals pass {i}: {p['unclosed']:,} faces sit in components the "
                f"capping could not close, so whether they enclose anything was never "
                f"decided and they were kept. The mesh is repaired but may keep inside-out "
                f"surface. This usually means boundary rings that pinch on a vertex.",
                UserWarning,
                stacklevel=2,
            )
        if p["spared"]:
            warnings.warn(
                f"drop_internals pass {i}: the step that drops components enclosing no "
                f"volume was about to remove {p['spared']:,} faces, more than half of the "
                f"surface it judged, so it was held back. The mesh is repaired but may keep "
                f"inside-out scraps. Check that the faces are wound outward and "
                f"consistently.",
                UserWarning,
                stacklevel=2,
            )
    return out
