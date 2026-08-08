"""Arranging shapes on a page without letting any of them touch.

Three primitives, meant to be used together::

    masks = fastcore.rasterize_segments(coords, edges, scale=40, pad=1)
    positions, variant, grid = fastcore.pack_masks(
        masks, page_shape=(1300, 980)
    )

:func:`~navis_fastcore.rasterize_segments` turns line work - a skeleton's
node-to-parent edges, a mesh's face edges - into binary masks;
:func:`~navis_fastcore.pack_masks` lays those out so that no two share a pixel,
which lets a shape sit inside the loop of another as long as no ink meets.
:func:`~navis_fastcore.pack_rectangles` is the cheaper bounding-box alternative,
useful on its own or as a starting point for the mask packing.

The mask packer replaces the cross-correlation you would write in numpy - see its
docstring for why an early-exiting scan over bitsets beats an FFT that scores
every position at once.
"""

import numpy as np

from . import _fastcore

__all__ = ["rasterize_segments", "pack_masks", "pack_rectangles"]


def _prep_shape_coords(coords, what):
    """Coerce an (N, 2) coordinate array to contiguous float64."""
    coords = np.asarray(coords, dtype=np.float64, order="C")
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"`{what}` must be a 2-D array of shape (N, 2), got {coords.shape}")
    return coords


def _prep_shape_edges(edges, n_coords, what):
    """Coerce an (E, 2) index array to contiguous uint32 and range-check it.

    The range check lives here, not in Rust: it is one vectorised pass over an array
    numpy already holds, against a serial walk of every edge with the GIL held - which
    on a mesh-sized batch costs more than a third of the rasterise it guards.
    """
    edges = np.asarray(edges, order="C")
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError(f"`{what}` must be a 2-D array of shape (E, 2), got {edges.shape}")
    if len(edges):
        if edges.min() < 0:
            raise ValueError(f"`{what}` must not contain negative indices")
        if edges.max() >= n_coords:
            raise ValueError(
                f"`{what}` names vertex {edges.max()}, but there are only "
                f"{n_coords} vertices"
            )
    return np.ascontiguousarray(edges, dtype=np.uint32)


def rasterize_segments(
    coords, edges, scale=1.0, pad=0, fill=False, turn=0, threads=None
):
    """Mark every pixel a set of line segments passes through.

    Each shape is shifted so its own lower-left corner sits at ``(pad, pad)`` and
    scaled by ``scale``; the mask comes out just big enough to hold it with ``pad``
    pixels of margin on every side. Shapes are rasterised one per core.

    Doing this in numpy means interpolating every edge and scattering the result,
    which materialises one element per *pixel-step of every edge*: on a mesh with
    300k edges averaging five pixels each that is a 1.5M-element index array, plus
    the same again for the parameter and both coordinates, to set a few tens of
    thousands of distinct pixels. Here the walk writes straight into the mask.

    Parameters
    ----------
    coords :    (N, 2) array | list of (N, 2) arrays
                The two in-plane coordinates of each shape, in whatever units. A
                single array is taken as a single shape.
    edges :     (E, 2) array | list of (E, 2) arrays
                Index pairs into the matching ``coords``. Vertices that no edge
                names are ignored - but if a shape has no edges at all, every one
                of its vertices is marked, so a bare point cloud still rasterises.
    scale :     float, default 1.0
                Pixels per coordinate unit.
    pad :       int, default 0
                Margin left around the shape, in pixels, and the radius the result
                is dilated by. Two masks that do not overlap then leave ``2 * pad``
                pixels of clear space between the lines themselves.
    fill :      bool | sequence of bool, default False
                Fill the interior afterwards - what a solid shape wants, since what
                it occupies is its silhouette rather than the wireframe of its
                edges. Unlike ``scale`` and ``pad``, which describe the page and so
                are necessarily shared, this describes the shape: pass one flag per
                shape to rasterise a mix of solid and wireframe in one call.
    turn :      int, default 0
                Quarter turns counter-clockwise to apply before rasterising, taken
                mod 4 - ``1`` is ``(u, v) -> (-v, u)``. All four are offered because
                a caller whose own axes are mirrored needs the other handedness:
                composed with a flipped axis, a counter-clockwise turn *is* the
                clockwise one. Turning here rather than rotating the coordinates
                first saves a copy of them, which for a large mesh is the biggest
                array in play.
    threads :   int, optional
                Number of threads to use. `None` (default) uses all cores.

    Returns
    -------
    masks :     (h, w) bool array | list of (h, w) bool arrays
                One mask per shape, matching the shape of ``coords``. A shape whose
                coordinates are not all finite comes back as a ``(0, 0)`` mask.

    Examples
    --------
    >>> import navis_fastcore as fastcore
    >>> import numpy as np
    >>> coords = np.array([[0., 0.], [4., 4.]])
    >>> edges = np.array([[0, 1]])
    >>> mask = fastcore.rasterize_segments(coords, edges)
    >>> mask.astype(int)
    array([[1, 0, 0, 0, 0],
           [0, 1, 0, 0, 0],
           [0, 0, 1, 0, 0],
           [0, 0, 0, 1, 0],
           [0, 0, 0, 0, 1]])

    """
    # A bare (N, 2) array is one shape; anything else is a sequence of them.
    single = isinstance(coords, np.ndarray) and coords.ndim == 2
    if single:
        coords, edges = [coords], [edges]

    coords = [_prep_shape_coords(c, f"coords[{i}]") for i, c in enumerate(coords)]
    edges = [
        _prep_shape_edges(e, len(c), f"edges[{i}]")
        for i, (c, e) in enumerate(zip(coords, edges))
    ]

    fill = np.broadcast_to(np.asarray(fill, dtype=bool), (len(coords),))

    masks = _fastcore.rasterize_segments(
        coords,
        edges,
        float(scale),
        int(pad),
        [bool(f) for f in fill],
        int(turn) % 4,
        None if threads is None else int(threads),
    )
    return masks[0] if single else masks


def pack_masks(masks, page_shape, grid=None, cost=None, optional=False, threads=None):
    """Place binary masks onto a page so that no two of them share a pixel.

    Masks go down largest-first, each at the best position at which not one of its
    pixels collides with what is already there - so a shape may sit inside the loop
    of another as long as no ink actually meets.

    The obvious way to do this is a cross correlation:
    ``fftconvolve(page, mask[::-1, ::-1], mode="valid")`` gives the number of shared
    pixels at every position at once, and the free ones are the zeros. It is also
    the wrong computation - an exact overlap *count* everywhere, in floating point,
    when the question is boolean and the answer is wanted at one position. Measured
    on 200 synthetic arbors at 100 px per page unit it is about 70% of the layout,
    and it grows as ``O(N^2 res^2)``. This visits positions in the order you score
    them and stops at the first free one, testing collisions 64 pixels at a time.

    Parameters
    ----------
    masks :         list of 2-D bool arrays | list of lists of them
                    One entry per item. An entry may be a single mask, or a list of
                    the variants to try for that item - i.e. upright and, if
                    rotation is allowed, turned a quarter turn. The two forms mix
                    freely, so the output of
                    :func:`~navis_fastcore.rasterize_segments` can be handed
                    straight over. An item with no usable variant counts as one
                    that did not fit.
    page_shape :    (height, width)
                    Page size in pixels.
    grid :          (height, width) bool array, optional
                    A page that is already partly occupied, e.g. one holding an
                    earlier set of shapes, or one with everything outside some
                    silhouette marked as taken. Defaults to an empty page.
    cost :          (height, width) array, optional
                    What counts as the "best" position: each mask goes where this
                    is lowest under its centre. Defaults to as far down, and then
                    as far left, as the mask will go - which fills a rectangle
                    neatly but would pile everything into the bottom of any other
                    shape.
    optional :      bool, default False
                    If True, masks that fit nowhere are skipped instead of failing
                    the packing.
    threads :       int, optional
                    Number of threads to use. `None` (default) uses all cores.

    Returns
    -------
    positions :     (N, 2) array of lower-left ``(x, y)`` corners in pixels
                    ``NaN`` for masks that were skipped. Note the order: ``(x, y)``
                    here, against ``(height, width)`` for ``page_shape``, ``grid``
                    and ``cost``.
    variant :       (N, ) int array
                    Which variant was used. Meaningless where ``positions`` is NaN.
    grid :          (height, width) bool array
                    The page with everything drawn onto it - hand it back in as
                    ``grid`` to pack a second set into whatever room is left. The
                    ``grid`` you passed in is not modified.

    All three are ``None`` if a mask fit nowhere and ``optional=False``.

    Examples
    --------
    >>> import navis_fastcore as fastcore
    >>> import numpy as np
    >>> square = np.ones((4, 4), dtype=bool)
    >>> pos, variant, grid = fastcore.pack_masks([square, square], page_shape=(8, 8))
    >>> pos
    array([[0., 0.],
           [4., 0.]])

    """
    page_shape = (int(page_shape[0]), int(page_shape[1]))
    if min(page_shape) < 0:
        raise ValueError(f"`page_shape` must not be negative, got {page_shape}")

    # A bare 2-D array is an item with one variant - the same rule `rasterize_segments`
    # applies to `coords`, so its output composes with this. Tested on the array itself
    # rather than with `np.ndim`, which would have to build an array out of the variant
    # list to answer, and raises on the ragged one a quarter turn produces.
    prepped = [
        [
            np.asarray(v, dtype=bool)
            for v in ([m] if isinstance(m, np.ndarray) and m.ndim == 2 else m)
        ]
        for m in masks
    ]

    if grid is not None:
        grid = np.asarray(grid, dtype=bool)
    if cost is not None:
        cost = np.ascontiguousarray(cost, dtype=np.float64)

    out = _fastcore.pack_masks(
        prepped,
        page_shape,
        grid,
        cost,
        bool(optional),
        None if threads is None else int(threads),
    )
    if out is None:
        return None, None, None
    return out


def pack_rectangles(sizes, page_size, allow_rotation=False, optional=False, free=None):
    """Pack rectangles into a page using the MaxRects heuristic.

    Rectangles are inserted largest-first into the free rectangle that leaves the
    least slack (best short side fit), which packs them tightly against each other
    and against the edges of the page.

    Cheap, but a bounding box is mostly empty for anything branching - see
    :func:`~navis_fastcore.pack_masks` for packing the shapes themselves.

    Parameters
    ----------
    sizes :             (N, 2) array of widths and heights
    page_size :         (width, height)
    allow_rotation :    bool, default False
                        Whether rectangles may be turned a quarter turn.
    optional :          bool, default False
                        If True, rectangles that fit nowhere are skipped instead of
                        failing the packing.
    free :              (M, 4) array, optional
                        Free space ``(x, y, width, height)`` to pack into, e.g. what
                        an earlier call left over. Defaults to the whole page.

    Returns
    -------
    positions :         (N, 2) array of lower-left ``(x, y)`` corners
                        ``NaN`` for rectangles that were skipped.
    rotated :           (N, ) bool array
    free :              (M, 4) array of the free space that is left

    All three are ``None`` if a rectangle fit nowhere and ``optional=False``.

    Examples
    --------
    >>> import navis_fastcore as fastcore
    >>> pos, rotated, free = fastcore.pack_rectangles(
    ...     [[4, 4], [4, 4]], page_size=(8, 4)
    ... )
    >>> pos
    array([[0., 0.],
           [4., 0.]])

    """
    sizes = np.asarray(sizes, dtype=np.float64, order="C")
    if sizes.ndim != 2 or sizes.shape[1] != 2:
        raise ValueError(f"`sizes` must be a 2-D array of shape (N, 2), got {sizes.shape}")

    if free is not None:
        free = np.asarray(free, dtype=np.float64, order="C")
        if free.ndim != 2 or free.shape[1] != 4:
            raise ValueError(f"`free` must be a 2-D array of shape (M, 4), got {free.shape}")

    page_size = (float(page_size[0]), float(page_size[1]))

    out = _fastcore.pack_rectangles(
        sizes, page_size, bool(allow_rotation), bool(optional), free
    )
    if out is None:
        return None, None, None
    return out
