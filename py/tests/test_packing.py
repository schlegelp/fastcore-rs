"""Tests for the packing module — rasterising shapes and laying them out on a page.

The oracle throughout is the way you would write this in numpy and scipy: interpolate
every edge and scatter it for the rasteriser, and a cross correlation
(``fftconvolve(page, mask[::-1, ::-1], mode="valid")``) for the packer, which scores
every position at once so the free ones are the zeros. The fast paths answer the same
questions completely differently — a walk that writes straight into a bit-packed image,
and a scan over those bitsets that visits positions in score order and stops at the
first free one — so agreeing with the oracle is the whole specification.

One deliberate difference: both packers order items largest-first with a *stable* sort,
where ``np.argsort`` defaults to an unstable one. Ties therefore go in input order here
and in an arbitrary order there, so the oracles below pass ``kind="stable"``.
"""

import numpy as np
import pytest
from scipy import ndimage
from scipy.signal import fftconvolve

import navis_fastcore as fastcore

EPS = 1e-9


# -----------------------------------------------------------------------------
# Oracles
# -----------------------------------------------------------------------------


def ref_rasterize(coords, edges, scale=1.0, pad=0, fill=False, turn=0):
    """Interpolate every edge and scatter it, the numpy way."""
    coords = np.asarray(coords, dtype=float)
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    u, v = coords[:, 0], coords[:, 1]
    for _ in range(turn % 4):
        u, v = -v, u
    if not (np.isfinite(u).all() and np.isfinite(v).all()) or not len(coords):
        return np.zeros((0, 0), dtype=bool)

    x = (u - u.min()) * scale + pad
    y = (v - v.min()) * scale + pad
    mask = np.zeros(
        (int(np.ceil(y.max())) + pad + 1, int(np.ceil(x.max())) + pad + 1), dtype=bool
    )

    if len(edges):
        child, parent = edges[:, 0], edges[:, 1]
        dx, dy = x[parent] - x[child], y[parent] - y[child]
        steps = (
            np.maximum(np.ceil(np.maximum(np.abs(dx), np.abs(dy))), 1).astype(int) + 1
        )
        i = np.repeat(np.arange(len(child)), steps)
        t = np.arange(len(i)) - np.repeat(np.cumsum(steps) - steps, steps)
        t = t / (steps[i] - 1)
        x, y = x[child][i] + t * dx[i], y[child][i] + t * dy[i]

    mask[np.rint(y).astype(int), np.rint(x).astype(int)] = True

    if fill:
        mask = ndimage.binary_fill_holes(np.pad(mask, 1))[1:-1, 1:-1]
    if pad:
        r = np.arange(-pad, pad + 1)
        mask = ndimage.binary_dilation(mask, r[:, None] ** 2 + r**2 <= pad**2)
    return mask


def ref_pack_masks(masks, page_shape, optional=False, grid=None, cost=None):
    """Score every position with a cross correlation and take the best zero."""
    height, width = page_shape
    if grid is None:
        grid = np.zeros((height, width), dtype=bool)
    positions = np.full((len(masks), 2), np.nan)
    variant = np.zeros(len(masks), dtype=int)

    for i in np.argsort([-v[0].size for v in masks], kind="stable"):
        top = height
        if cost is None:
            used = np.nonzero(grid.any(axis=1))[0]
            top = used[-1] + 1 if len(used) else 0
        best = None
        for k, mask in enumerate(masks[i]):
            h, w = mask.shape
            if h > height or w > width or h == 0 or w == 0:
                continue
            region = grid[: min(height, top + h)].astype(float)
            overlap = fftconvolve(region, mask[::-1, ::-1].astype(float), mode="valid")
            ys, xs = np.nonzero(overlap < 0.5)
            if not len(ys):
                continue
            if cost is None:
                j = np.lexsort((xs, ys))[0]
                score = (ys[j], xs[j])
            else:
                at = cost[ys + h // 2, xs + w // 2]
                j = np.argmin(at)
                score = (at[j], ys[j], xs[j])
            if best is None or score < best[0]:
                best = (score, ys[j], xs[j], k)
        if best is None:
            if optional:
                continue
            return None, None, None
        _, y, x, k = best
        h, w = masks[i][k].shape
        grid[y : y + h, x : x + w] |= masks[i][k]
        positions[i] = (x, y)
        variant[i] = k
    return positions, variant, grid


def _prune_free(free):
    free = free[(free[:, 2] > EPS) & (free[:, 3] > EPS)]
    x0, y0 = free[:, 0], free[:, 1]
    x1, y1 = x0 + free[:, 2], y0 + free[:, 3]
    inside = (
        (x0[:, None] >= x0[None, :] - EPS)
        & (y0[:, None] >= y0[None, :] - EPS)
        & (x1[:, None] <= x1[None, :] + EPS)
        & (y1[:, None] <= y1[None, :] + EPS)
    )
    np.fill_diagonal(inside, False)
    ix = np.arange(len(free))
    inside &= ~(inside & inside.T & (ix[:, None] < ix[None, :]))
    return free[~inside.any(axis=1)]


def _insert_rect(free, w, h, allow_rotation):
    options = [(w, h, False)]
    if allow_rotation:
        options.append((h, w, True))
    best = None
    for w_, h_, rot in options:
        slack = free[:, 2:] - [w_, h_]
        fits = (slack >= -EPS).all(axis=1)
        if not fits.any():
            continue
        slack = np.abs(slack)
        short = np.where(fits, slack.min(axis=1), np.inf)
        long = np.where(fits, slack.max(axis=1), np.inf)
        j = np.lexsort((long, short))[0]
        if best is None or (short[j], long[j]) < best[:2]:
            best = (short[j], long[j], j, w_, h_, rot)
    if best is None:
        return None
    _, _, j, w, h, rot = best
    px, py = free[j, :2]
    hit = (
        (free[:, 0] < px + w - EPS)
        & (free[:, 0] + free[:, 2] > px + EPS)
        & (free[:, 1] < py + h - EPS)
        & (free[:, 1] + free[:, 3] > py + EPS)
    )
    pieces = []
    for fx, fy, fw, fh in free[hit]:
        pieces += [
            [fx, fy, px - fx, fh],
            [px + w, fy, fx + fw - px - w, fh],
            [fx, fy, fw, py - fy],
            [fx, py + h, fw, fy + fh - py - h],
        ]
    free = np.vstack([free[~hit], pieces]) if pieces else free[~hit]
    return px, py, rot, _prune_free(free)


def ref_pack_rectangles(sizes, page_size, allow_rotation=False, optional=False):
    """MaxRects, written out in numpy."""
    sizes = np.asarray(sizes, dtype=float)
    free = np.array([[0.0, 0.0, *page_size]])
    positions = np.full((len(sizes), 2), np.nan)
    rotated = np.zeros(len(sizes), dtype=bool)
    for i in np.argsort(-sizes.max(axis=1), kind="stable"):
        fit = _insert_rect(free, *sizes[i], allow_rotation)
        if fit is None:
            if optional:
                continue
            return None, None, None
        px, py, rotated[i], free = fit
        positions[i] = (px, py)
    return positions, rotated, free


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


def arbor(rng, n_pts):
    """A branching random walk — a stand-in for a neuron in projection."""
    pts = np.zeros((n_pts, 2))
    parent = np.zeros(n_pts, dtype=np.int64)
    for i in range(1, n_pts):
        p = rng.integers(0, i) if rng.random() < 0.05 else i - 1
        parent[i] = p
        pts[i] = pts[p] + rng.normal(0, 1, 2) * [3, 1]
    edges = np.column_stack([np.arange(1, n_pts), parent[1:]]).astype(np.uint32)
    return pts, edges


@pytest.fixture(scope="module")
def arbors():
    rng = np.random.default_rng(42)
    return [arbor(rng, int(rng.integers(200, 900))) for _ in range(24)]


@pytest.fixture(scope="module")
def masks(arbors):
    coords = [c for c, _ in arbors]
    edges = [e for _, e in arbors]
    upright = fastcore.rasterize_segments(coords, edges, scale=0.35, pad=1)
    turned = fastcore.rasterize_segments(coords, edges, scale=0.35, pad=1, turn=1)
    return [[u, t] for u, t in zip(upright, turned)]


PAGE = (220, 170)


@pytest.fixture(scope="module")
def radial_cost():
    rows, cols = np.ogrid[0 : PAGE[0], 0 : PAGE[1]]
    return np.hypot(rows - PAGE[0] / 2, cols - PAGE[1] / 2)


# -----------------------------------------------------------------------------
# rasterize_segments
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("pad", [0, 1, 3])
@pytest.mark.parametrize("fill", [False, True])
@pytest.mark.parametrize("turn", [0, 1, 2, 3])
def test_rasterize_matches_the_numpy_scatter(arbors, pad, fill, turn):
    coords = [c for c, _ in arbors]
    edges = [e for _, e in arbors]
    got = fastcore.rasterize_segments(
        coords, edges, scale=0.4, pad=pad, fill=fill, turn=turn
    )
    for i, (c, e) in enumerate(arbors):
        want = ref_rasterize(c, e, scale=0.4, pad=pad, fill=fill, turn=turn)
        assert got[i].shape == want.shape, f"shape differs for arbor {i}"
        assert np.array_equal(got[i], want), (
            f"{(got[i] != want).sum()} pixels differ for arbor {i}"
        )


def test_rasterize_fill_can_differ_per_shape(arbors):
    """A mix of solid and wireframe shapes in one call."""
    coords = [c for c, _ in arbors]
    edges = [e for _, e in arbors]
    flags = [i % 2 == 0 for i in range(len(arbors))]
    got = fastcore.rasterize_segments(coords, edges, scale=0.4, pad=1, fill=flags)
    for i, (c, e) in enumerate(arbors):
        want = ref_rasterize(c, e, scale=0.4, pad=1, fill=flags[i])
        assert np.array_equal(got[i], want), f"arbor {i} (fill={flags[i]}) differs"


def test_rasterize_rejects_a_mismatched_fill(arbors):
    coords = [c for c, _ in arbors]
    edges = [e for _, e in arbors]
    with pytest.raises(ValueError):
        fastcore.rasterize_segments(coords, edges, fill=[True, False])


def test_rasterize_takes_a_single_shape(arbors):
    c, e = arbors[0]
    single = fastcore.rasterize_segments(c, e, scale=0.4, pad=1)
    batched = fastcore.rasterize_segments([c], [e], scale=0.4, pad=1)
    assert isinstance(single, np.ndarray)
    assert np.array_equal(single, batched[0])


def test_rasterize_walks_a_diagonal_without_gaps():
    mask = fastcore.rasterize_segments(
        np.array([[0.0, 0.0], [10.0, 10.0]]), np.array([[0, 1]])
    )
    assert mask.shape == (11, 11)
    assert np.array_equal(np.nonzero(mask)[0], np.arange(11))
    assert np.array_equal(np.nonzero(mask)[1], np.arange(11))


def test_rasterize_fills_a_closed_outline():
    # A square outline: unfilled it is a rim, filled it is solid.
    coords = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
    rim = fastcore.rasterize_segments(coords, edges)
    solid = fastcore.rasterize_segments(coords, edges, fill=True)
    assert rim.sum() == 4 * 10
    assert solid.all()


def test_rasterize_without_edges_marks_the_points():
    coords = np.array([[0.0, 0.0], [3.0, 2.0]])
    mask = fastcore.rasterize_segments(coords, np.zeros((0, 2)))
    assert mask.sum() == 2
    assert mask[0, 0] and mask[2, 3]


def test_rasterize_rejects_a_bad_edge():
    coords = np.array([[0.0, 0.0], [1.0, 1.0]])
    with pytest.raises(ValueError, match="only 2 vertices"):
        fastcore.rasterize_segments(coords, np.array([[0, 7]]))
    with pytest.raises(ValueError, match="negative"):
        fastcore.rasterize_segments(coords, np.array([[0, -1]]))


def test_rasterize_rejects_bad_shapes():
    with pytest.raises(ValueError, match=r"\(N, 2\)"):
        fastcore.rasterize_segments(np.zeros((4, 3)), np.zeros((0, 2)))
    with pytest.raises(ValueError, match=r"\(E, 2\)"):
        fastcore.rasterize_segments(np.zeros((4, 2)), np.zeros((2, 3)))
    with pytest.raises(ValueError, match="coordinate arrays"):
        fastcore.rasterize_segments([np.zeros((4, 2))], [], scale=1.0)
    with pytest.raises(ValueError, match="positive"):
        fastcore.rasterize_segments(np.zeros((4, 2)), np.zeros((0, 2)), scale=0)


def test_rasterize_of_non_finite_coords_is_empty():
    coords = np.array([[0.0, 0.0], [np.nan, 1.0]])
    assert fastcore.rasterize_segments(coords, np.array([[0, 1]])).shape == (0, 0)


# -----------------------------------------------------------------------------
# pack_masks
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("n_variants", [1, 2])
def test_pack_masks_matches_the_correlation_bottom_left(masks, n_variants):
    ms = [v[:n_variants] for v in masks]
    want = ref_pack_masks([[m.copy() for m in v] for v in ms], PAGE, optional=True)
    got = fastcore.pack_masks(ms, PAGE, optional=True)

    np.testing.assert_array_equal(want[0], got[0])
    # `variant` is meaningless where nothing was placed
    placed = np.isfinite(got[0]).all(axis=1)
    np.testing.assert_array_equal(want[1][placed], got[1][placed])
    np.testing.assert_array_equal(want[2], got[2])


@pytest.mark.parametrize("n_variants", [1, 2])
def test_pack_masks_matches_the_correlation_under_a_cost(masks, radial_cost, n_variants):
    ms = [v[:n_variants] for v in masks]
    want = ref_pack_masks(
        [[m.copy() for m in v] for v in ms], PAGE, optional=True, cost=radial_cost.copy()
    )
    got = fastcore.pack_masks(ms, PAGE, cost=radial_cost, optional=True)

    np.testing.assert_array_equal(want[0], got[0])
    np.testing.assert_array_equal(want[2], got[2])


def test_pack_masks_never_overlaps(masks):
    pos, variant, grid = fastcore.pack_masks(masks, PAGE, optional=True)
    canvas = np.zeros(PAGE, dtype=bool)
    for i, (p, k) in enumerate(zip(pos, variant)):
        if not np.isfinite(p).all():
            continue
        x, y = int(p[0]), int(p[1])
        m = masks[i][k]
        assert y + m.shape[0] <= PAGE[0] and x + m.shape[1] <= PAGE[1], "off the page"
        win = canvas[y : y + m.shape[0], x : x + m.shape[1]]
        assert not (win & m).any(), f"mask {i} overlaps something already placed"
        win |= m
    np.testing.assert_array_equal(canvas, grid)


def test_pack_masks_lets_a_shape_nest_inside_a_loop():
    """The point of packing masks rather than boxes."""
    ring = np.zeros((9, 9), dtype=bool)
    ring[0, :] = ring[-1, :] = ring[:, 0] = ring[:, -1] = True
    small = np.ones((3, 3), dtype=bool)

    pos, _, _ = fastcore.pack_masks([[ring], [small]], (9, 9))
    np.testing.assert_array_equal(pos[0], [0, 0])
    np.testing.assert_array_equal(pos[1], [1, 1])

    # A bounding-box packer has no chance on the same page
    assert fastcore.pack_rectangles([[9, 9], [3, 3]], (9, 9))[0] is None


def test_pack_masks_fails_or_skips_when_something_does_not_fit():
    big = np.ones((4, 4), dtype=bool)
    assert fastcore.pack_masks([[big], [big]], (4, 4)) == (None, None, None)

    pos, _, _ = fastcore.pack_masks([[big], [big]], (4, 4), optional=True)
    assert np.isfinite(pos).all(axis=1).sum() == 1


def test_pack_masks_respects_a_prefilled_grid():
    grid = np.zeros((8, 8), dtype=bool)
    grid[:4] = True
    pos, _, out = fastcore.pack_masks(
        [[np.ones((4, 8), dtype=bool)]], (8, 8), grid=grid
    )
    np.testing.assert_array_equal(pos[0], [0, 4])
    assert out.all()
    assert not grid[4:].any(), "the caller's grid must not be modified in place"


def test_pack_masks_uses_rotation_when_it_is_the_only_way():
    upright = np.ones((6, 2), dtype=bool)
    turned = np.ones((2, 6), dtype=bool)
    pos, variant, _ = fastcore.pack_masks([[upright, turned]], (2, 6))
    assert variant[0] == 1
    np.testing.assert_array_equal(pos[0], [0, 0])


def test_pack_masks_pulls_towards_the_cost_minimum():
    h, w = 21, 21
    rows, cols = np.ogrid[0:h, 0:w]
    cost = np.hypot(rows - 10, cols - 10)
    pos, _, _ = fastcore.pack_masks([[np.ones((3, 3), dtype=bool)]], (h, w), cost=cost)
    # The centre of a 3x3 at corner (y, x) is (y + 1, x + 1)
    np.testing.assert_array_equal(pos[0], [9, 9])


def test_pack_masks_backfills_into_the_leftover_grid(masks):
    """The two-pass use: pack one set, then squeeze another into what is left."""
    first, second = masks[:12], masks[12:]
    pos, _, grid = fastcore.pack_masks(first, PAGE, optional=True)
    ink = grid.sum()
    pos2, _, grid2 = fastcore.pack_masks(second, PAGE, grid=grid, optional=True)
    assert grid2.sum() > ink, "the second pass should have added something"
    assert (grid2 | grid).sum() == grid2.sum(), "it must not erase the first pass"


def test_pack_masks_takes_bare_masks_and_variant_lists(masks):
    """`rasterize_segments`' output must go straight into `pack_masks`."""
    bare = [v[0] for v in masks]
    a = fastcore.pack_masks(bare, PAGE, optional=True)
    b = fastcore.pack_masks([[m] for m in bare], PAGE, optional=True)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[2], b[2])

    # The two forms may be mixed in one call
    mixed = [bare[0]] + [list(v) for v in masks[1:]]
    pos, _, _ = fastcore.pack_masks(mixed, PAGE, optional=True)
    assert np.isfinite(pos).all(axis=1).any()


def test_pack_masks_rejects_a_mismatched_grid_or_cost():
    m = [[np.ones((2, 2), dtype=bool)]]
    with pytest.raises(ValueError, match="but the page is"):
        fastcore.pack_masks(m, (8, 8), grid=np.zeros((4, 4), dtype=bool))
    with pytest.raises(ValueError, match="but the page is"):
        fastcore.pack_masks(m, (8, 8), cost=np.zeros((4, 4)))


# -----------------------------------------------------------------------------
# pack_rectangles
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("allow_rotation", [False, True])
def test_pack_rectangles_matches_the_numpy_maxrects(allow_rotation):
    rng = np.random.default_rng(3)
    sizes = rng.uniform(1, 6, size=(40, 2))
    want = ref_pack_rectangles(sizes, (20.0, 20.0), allow_rotation, optional=True)
    got = fastcore.pack_rectangles(
        sizes, (20.0, 20.0), allow_rotation=allow_rotation, optional=True
    )
    np.testing.assert_allclose(want[0], got[0], rtol=0, atol=1e-9)
    np.testing.assert_array_equal(want[1], got[1])


def test_pack_rectangles_never_overlaps():
    rng = np.random.default_rng(5)
    sizes = rng.uniform(1, 6, size=(40, 2))
    pos, rotated, _ = fastcore.pack_rectangles(sizes, (20.0, 20.0), optional=True)
    boxes = [
        (p[0], p[1], *(s[::-1] if r else s))
        for p, s, r in zip(pos, sizes, rotated)
        if np.isfinite(p).all()
    ]
    assert len(boxes) > 10
    for i, a in enumerate(boxes):
        assert a[0] + a[2] <= 20 + EPS and a[1] + a[3] <= 20 + EPS
        for b in boxes[i + 1 :]:
            assert not (
                a[0] < b[0] + b[2] - EPS
                and b[0] < a[0] + a[2] - EPS
                and a[1] < b[1] + b[3] - EPS
                and b[1] < a[1] + a[3] - EPS
            ), f"{a} overlaps {b}"


def test_pack_rectangles_fails_or_skips_when_something_does_not_fit():
    sizes = [[5, 5], [5, 5]]
    assert fastcore.pack_rectangles(sizes, (8, 8)) == (None, None, None)
    pos, _, _ = fastcore.pack_rectangles(sizes, (8, 8), optional=True)
    assert np.isfinite(pos).all(axis=1).sum() == 1


def test_pack_rectangles_carries_free_space_into_a_second_call():
    _, _, free = fastcore.pack_rectangles([[4, 8]], (8, 8))
    pos, _, _ = fastcore.pack_rectangles([[4, 8]], (8, 8), free=free)
    assert pos[0][0] >= 4 - EPS, "the second should go beside the first"


def test_pack_rectangles_rejects_bad_shapes():
    with pytest.raises(ValueError, match=r"\(N, 2\)"):
        fastcore.pack_rectangles(np.zeros((4, 3)), (8, 8))
    with pytest.raises(ValueError, match=r"\(M, 4\)"):
        fastcore.pack_rectangles(np.zeros((4, 2)), (8, 8), free=np.zeros((1, 3)))
