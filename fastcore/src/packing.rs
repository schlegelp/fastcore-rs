//! Laying shapes out on a page without letting any of them touch.
//!
//! Two packers, for the two ways of asking the question:
//!
//! - [`pack_rectangles`] packs bounding boxes, by the MaxRects heuristic. Cheap, and
//!   tight against the page — but a box is mostly empty for anything branching, so the
//!   result is loose against the shapes themselves.
//! - [`pack_masks`] packs the shapes, as bitmaps. A shape may reach into another's empty
//!   space — even sit inside a loop of it — as long as no pixel of one meets a pixel of
//!   the other.
//!
//! # Why [`pack_masks`] is not a cross-correlation
//!
//! The obvious way to test every position of a mask against the page at once is a
//! correlation: `fftconvolve(page, mask[::-1, ::-1], mode="valid")` gives the number of
//! shared pixels at each of the ~1.3M positions of a 1300x980 page, and the free ones are
//! the zeros. It is a single vectorised call, which is why it is what you write in numpy.
//!
//! It is also the wrong computation. It produces an exact overlap *count* at every
//! position, in floating point, when the question is boolean and the answer is wanted at
//! exactly one position — the best one. Measured on 200 synthetic arbors at 100 px per
//! page unit, that call is around 70% of the layout, 5 ms per shape when packing bottom-up
//! and 17 ms when a cost surface forces the correlation over the full page; the rest goes
//! on `nonzero` and `argmin` over a 1.3M-element result that is discarded immediately. It
//! grows as `O(N^2 res^2)`, so doubling the resolution quadruples it.
//!
//! What this module does instead:
//!
//! - **Visit positions in the order the caller scores them.** Both cost models — bottom-up
//!   and a cost surface — define a total order on positions that does not depend on the
//!   shape, so the best free position is the *first* free one in that order. The scan
//!   stops there. A correlation cannot stop early; that is the whole difference.
//! - **Test collisions as bit operations.** A page row is `u64` words, a mask row is `u64`
//!   words, and the test is an `AND` — 64 pixels per instruction, no allocation, and the
//!   page stays in cache ([`crate::raster::Bitmap`]).
//! - **Reject on the densest rows first.** A position that collides usually collides on
//!   the shape's heaviest row, so [`Shape`] tries those first and most rejections cost one
//!   or two words.
//! - **Skip what is provably empty.** Everything above the highest occupied row is free, so
//!   a bottom-up scan is bounded by the fill line rather than by the page.
//!
//! The scan runs on every core, and so do the variants of a shape against each other — see
//! [`first_matching`] for why the whole range goes to rayon in one piece. Placement itself
//! is sequential and stays that way: each shape goes down against the page the last one
//! left behind, which is what the packing *means*.

use rayon::prelude::*;

use crate::raster::Bitmap;
use crate::threads::with_pool;

/// Pixels per storage word, as in [`crate::raster`].
const BITS: usize = u64::BITS as usize;

/// Slack below which two edges count as flush. Rectangle packing only — the mask packer
/// is on a pixel grid and needs no tolerance.
const EPS: f64 = 1e-9;

/// One row of a mask, ready to be tested against a page row.
struct Probe<'a> {
    /// Which row of the mask this is — the offset from the placement's own row.
    row: usize,
    /// The words of `bits` that carry pixels; the rest are known to be zero.
    lo: usize,
    hi: usize,
    /// The row itself. Held rather than re-sliced out of the mask on every candidate
    /// position: `collides` runs hundreds of millions of times in a layout, and the slice
    /// is a constant of the probe.
    bits: &'a [u64],
    /// Pixels in the row. Only the ordering below reads it.
    count: u32,
}

/// A mask prepared for collision testing.
///
/// Holds its non-empty rows ordered by how many pixels they carry. Testing the busiest
/// rows first is what makes a rejection cheap: most candidate positions collide, and the
/// sooner one is found the sooner the position is abandoned.
struct Shape<'a> {
    map: &'a Bitmap,
    probes: Vec<Probe<'a>>,
}

impl<'a> Shape<'a> {
    fn new(map: &'a Bitmap) -> Self {
        let mut probes: Vec<Probe<'a>> = (0..map.height())
            .filter_map(|row| {
                let bits = map.row(row);
                let lo = bits.iter().position(|&w| w != 0)?;
                let hi = bits.iter().rposition(|&w| w != 0)? + 1;
                let count = bits[lo..hi].iter().map(|w| w.count_ones()).sum();
                Some(Probe { row, lo, hi, bits, count })
            })
            .collect();
        probes.sort_unstable_by_key(|p| std::cmp::Reverse(p.count));
        Self { map, probes }
    }

    /// Whether any pixel of the shape meets a pixel of `page` with its corner at `(y, x)`.
    ///
    /// The caller must have checked that the shape fits inside the page there; this only
    /// guards the word indices, not the pixel ones.
    #[inline]
    fn collides(&self, page: &Bitmap, y: usize, x: usize) -> bool {
        let (wb, s) = (x / BITS, x % BITS);
        let stride = page.stride();
        for p in &self.probes {
            let prow = page.row(y + p.row);
            for j in p.lo..p.hi {
                let m = p.bits[j];
                if m == 0 {
                    continue;
                }
                // A mask word spans 64 consecutive columns, so at offset `x` it lands
                // across two page words - the same bit offset for the whole row.
                if wb + j < stride && prow[wb + j] & (m << s) != 0 {
                    return true;
                }
                if s > 0 && wb + j + 1 < stride && prow[wb + j + 1] & (m >> (BITS - s)) != 0 {
                    return true;
                }
            }
        }
        false
    }
}

/// Where [`pack_masks`] put everything.
pub struct Packing {
    /// Per item, the `(x, y)` of its lower-left corner in pixels. `None` for an item that
    /// was skipped, which only happens when `optional` was set.
    pub positions: Vec<Option<(usize, usize)>>,
    /// Per item, which of its variants was used. Meaningless where `positions` is `None`.
    pub variant: Vec<usize>,
    /// The page with everything drawn onto it — ready to be handed back in as `grid` to
    /// pack a second set of shapes into whatever room is left.
    pub grid: Bitmap,
}

/// Scan `0..n` for the first index `test` accepts.
///
/// `find_first` and not `find_any`, because here the order *is* the cost model: the first
/// hit is the best position, not merely a workable one. Rayon keeps a shared bound on the
/// lowest match found so far and drops any split lying entirely above it, so stopping early
/// costs a few extra tests on the threads that had started further up — not a scan of the
/// page.
///
/// That bound is also why this hands rayon the whole range at once. It used to walk it in
/// blocks that quadrupled in size, on the theory that a scan hitting something early should
/// not pay for a page of parallel tests. Measured, the blocks only cost: each is a barrier,
/// so the slowest worker in one holds up every other, and on a machine whose cores are not
/// all the same speed that compounds into a cliff. Packing 120 arbors onto a 1300x980 page
/// under a cost surface, on 10 performance cores plus 4 efficiency ones (ms):
///
/// | threads | 8 | 10 | 12 | 14 |
/// |---|---|---|---|---|
/// | in blocks | 426 | 383 | 663 | 685 |
/// | whole range | 372 | 311 | 310 | **300** |
///
/// Bottom-up the two are the same code: that scan is over page *rows*, so `n` never
/// reached even the smallest first block and the loop always ran exactly once.
fn first_matching<F>(n: usize, test: F) -> Option<usize>
where
    F: Fn(usize) -> bool + Sync,
{
    (0..n).into_par_iter().find_first(|&i| test(i))
}

/// The best position found for one variant: `(cost, y, x)`, compared lexicographically.
///
/// `cost` is `0.0` when packing bottom-up, where the order is by `y` then `x` — so the
/// same comparison picks the winner under either cost model, and the position is the tail
/// of the score rather than a second copy of it.
type Candidate = (f64, usize, usize);

/// Lowest free row, then leftmost free column in it.
fn best_bottom_left(page: &Bitmap, shape: &Shape, ceiling: usize) -> Option<Candidate> {
    let (h, w) = (shape.map.height(), shape.map.width());
    let (y_max, x_max) = (page.height().checked_sub(h)?, page.width().checked_sub(w)?);

    // A row at a time, the columns within it serially. Flattening the two into one parallel
    // scan over every `(y, x)` would balance the work perfectly and is 1.7x *slower*: the
    // inner walk shares one page row and stops at its first free column, and neither
    // survives handing every position to the scheduler separately.
    let first_x = |y: usize| (0..=x_max).find(|&x| !shape.collides(page, y, x));
    // Above the fill line the page is empty by definition, so the scan is bounded by it
    // rather than by the page - and is guaranteed to succeed at `ceiling` if not before.
    let y = first_matching(y_max.min(ceiling) + 1, |y| first_x(y).is_some())?;
    first_x(y).map(|x| (0.0, y, x))
}

/// Free position whose *centre* reads lowest on the cost surface.
///
/// `order` is the page's cells sorted by cost — computed once for the whole packing, since
/// it does not depend on the shape. The centre offset is a constant shift per variant, and
/// shifting every position by the same amount does not reorder them, so the same list
/// serves every shape.
fn best_by_cost(
    page: &Bitmap,
    shape: &Shape,
    cost: &[f64],
    order: &[u32],
) -> Option<Candidate> {
    let (h, w) = (shape.map.height(), shape.map.width());
    let (y_max, x_max) = (page.height().checked_sub(h)?, page.width().checked_sub(w)?);
    let width = page.width();
    let (dy, dx) = (h / 2, w / 2);

    let corner = |k: usize| -> Option<(usize, usize)> {
        let c = order[k] as usize;
        let (cy, cx) = (c / width, c % width);
        let y = cy.checked_sub(dy)?;
        let x = cx.checked_sub(dx)?;
        (y <= y_max && x <= x_max).then_some((y, x))
    };

    let k = first_matching(order.len(), |k| {
        corner(k).is_some_and(|(y, x)| !shape.collides(page, y, x))
    })?;
    let (y, x) = corner(k)?;
    Some((cost[order[k] as usize], y, x))
}

/// Place binary masks onto a page so that no two of them share a pixel.
///
/// Masks go down largest-first, each at the best position at which not one of its pixels
/// collides with what is already there — so a shape may sit inside the loop of another as
/// long as no ink actually meets.
///
/// # Arguments
///
/// - `masks`: one entry per item, holding the variants to try — upright and, where
///   rotation is allowed, turned a quarter turn. An item with no usable variant is
///   treated as one that did not fit.
/// - `page`: `(height, width)` in pixels.
/// - `grid`: a page that is already partly occupied, e.g. one holding an earlier set of
///   shapes, or one with the area outside some silhouette marked as taken. Must be
///   `page`-sized. Defaults to empty.
/// - `cost`: `height * width` values, row-major, saying what counts as a good position:
///   each shape goes where this is lowest under its centre. Defaults to as far down, and
///   then as far left, as the shape will go — which fills a rectangle neatly but would
///   pile everything into the bottom of any other shape.
/// - `optional`: skip masks that fit nowhere instead of failing the packing.
///
/// # Returns
///
/// `None` if some mask fit nowhere and `optional` was not set.
///
/// # Panics
///
/// If `grid` is not `page`-sized, or `cost` is not `height * width` long.
pub fn pack_masks(
    masks: &[Vec<Bitmap>],
    page: (usize, usize),
    grid: Option<Bitmap>,
    cost: Option<&[f64]>,
    optional: bool,
    threads: Option<usize>,
) -> Option<Packing> {
    let (height, width) = page;
    let mut grid = grid.unwrap_or_else(|| Bitmap::new(height, width));
    assert_eq!(
        (grid.height(), grid.width()),
        (height, width),
        "`grid` must be the size of the page"
    );
    if let Some(c) = cost {
        assert_eq!(
            c.len(),
            height * width,
            "`cost` must have one value per pixel of the page"
        );
    }

    let mut positions = vec![None; masks.len()];
    let mut variant = vec![0usize; masks.len()];

    // Largest first, by the biggest variant - which for the usual quarter-turn pair is
    // any of them, but nothing here requires the variants of an item to be congruent.
    let mut order: Vec<usize> = (0..masks.len()).collect();
    order.sort_by_key(|&i| {
        std::cmp::Reverse(masks[i].iter().map(Bitmap::area).max().unwrap_or(0))
    });

    with_pool(threads, || {
        // One sort of the whole page, reused by every shape in the packing.
        let cost_order = cost.map(|c| {
            let mut ix: Vec<u32> = (0..c.len() as u32).collect();
            ix.par_sort_unstable_by(|&a, &b| {
                c[a as usize]
                    .partial_cmp(&c[b as usize])
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.cmp(&b))
            });
            ix
        });

        // One past the highest occupied row: everything above it is free.
        let mut ceiling = (0..height).rev().find(|&y| !grid.row_is_empty(y)).map_or(0, |y| y + 1);

        for i in order {
            // Every variant is scanned against the same untouched page, so they go at once.
            // Worth doing even though each scan is itself parallel: a scan stops at its
            // first hit and its rows cost anything from one word to a full page-width
            // probe, so it never fills the machine on its own — running the variants
            // together lets each cover the other's stalls. Measured on 120 arbors and two
            // variants, 14 threads: 88 -> 73 ms bottom-up, 300 -> 275 ms under a cost
            // surface.
            let found: Vec<Option<Candidate>> = masks[i]
                .par_iter()
                .map(|map| {
                    if map.height() == 0 || map.width() == 0 {
                        return None;
                    }
                    let shape = Shape::new(map);
                    match (cost, &cost_order) {
                        (Some(c), Some(o)) => best_by_cost(&grid, &shape, c, o),
                        _ => best_bottom_left(&grid, &shape, ceiling),
                    }
                })
                .collect();

            // Collected first and compared here rather than reduced in parallel: rayon's
            // `min_by` leaves the order it combines in unspecified, and a strict `<` in
            // input order is what keeps ties going to the earliest variant however the
            // scans interleaved.
            let best = found
                .into_iter()
                .enumerate()
                .filter_map(|(k, cand)| Some((cand?, k)))
                .reduce(|best, next| if next.0 < best.0 { next } else { best });

            let Some(((_, y, x), k)) = best else {
                if optional {
                    continue;
                }
                return None;
            };

            grid.paint(&masks[i][k], y, x);
            // A placed shape fits on the page by construction, so this cannot run past it.
            ceiling = ceiling.max(y + masks[i][k].height());
            positions[i] = Some((x, y));
            variant[i] = k;
        }

        Some(Packing {
            positions,
            variant,
            grid,
        })
    })
}

/// An axis-aligned rectangle: lower-left corner, then size.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rect {
    pub x: f64,
    pub y: f64,
    pub w: f64,
    pub h: f64,
}

/// Where [`pack_rectangles`] put everything.
pub struct RectPacking {
    /// Per item, the `(x, y)` of its lower-left corner. `None` for a skipped one.
    pub positions: Vec<Option<(f64, f64)>>,
    /// Per item, whether it went in turned a quarter turn.
    pub rotated: Vec<bool>,
    /// The free space that is left, to carry into a further call.
    pub free: Vec<Rect>,
}

/// Drop free rectangles wholly contained in another one.
fn prune_free(free: Vec<Rect>) -> Vec<Rect> {
    let free: Vec<Rect> = free
        .into_iter()
        .filter(|r| r.w > EPS && r.h > EPS)
        .collect();
    let inside = |a: &Rect, b: &Rect| {
        a.x >= b.x - EPS
            && a.y >= b.y - EPS
            && a.x + a.w <= b.x + b.w + EPS
            && a.y + a.h <= b.y + b.h + EPS
    };
    free.iter()
        .enumerate()
        .filter(|&(i, a)| {
            // Identical rectangles contain each other, which would drop both - keep the first.
            !free.iter().enumerate().any(|(j, b)| {
                i != j && inside(a, b) && !(inside(b, a) && i < j)
            })
        })
        .map(|(_, r)| *r)
        .collect()
}

/// Fit a `w` x `h` rectangle into the free space and cut it out.
///
/// `free` is left untouched unless the rectangle actually goes in.
fn insert_rect(
    free: &mut Vec<Rect>,
    w: f64,
    h: f64,
    allow_rotation: bool,
) -> Option<(f64, f64, bool)> {
    let options: &[(f64, f64, bool)] = if allow_rotation {
        &[(w, h, false), (h, w, true)]
    } else {
        &[(w, h, false)]
    };

    // Best short side fit: least slack on the tighter axis, ties broken on the other.
    let mut best: Option<(f64, f64, usize, f64, f64, bool)> = None;
    for &(w_, h_, rot) in options {
        for (j, f) in free.iter().enumerate() {
            let (sw, sh) = (f.w - w_, f.h - h_);
            if sw < -EPS || sh < -EPS {
                continue;
            }
            let (short, long) = (sw.abs().min(sh.abs()), sw.abs().max(sh.abs()));
            if best.is_none_or(|b| (short, long) < (b.0, b.1)) {
                best = Some((short, long, j, w_, h_, rot));
            }
        }
    }

    let (_, _, j, w, h, rot) = best?;
    let (px, py) = (free[j].x, free[j].y);

    // Every free rectangle the placement overlaps is cut into the up-to-four maximal
    // rectangles left of it.
    let (mut kept, mut pieces) = (Vec::with_capacity(free.len()), Vec::new());
    for f in free.iter().copied() {
        let hits = f.x < px + w - EPS
            && f.x + f.w > px + EPS
            && f.y < py + h - EPS
            && f.y + f.h > py + EPS;
        if !hits {
            kept.push(f);
            continue;
        }
        pieces.extend_from_slice(&[
            Rect { x: f.x, y: f.y, w: px - f.x, h: f.h },                     // left
            Rect { x: px + w, y: f.y, w: f.x + f.w - px - w, h: f.h },        // right
            Rect { x: f.x, y: f.y, w: f.w, h: py - f.y },                     // below
            Rect { x: f.x, y: py + h, w: f.w, h: f.y + f.h - py - h },        // above
        ]);
    }
    kept.append(&mut pieces);

    *free = prune_free(kept);
    Some((px, py, rot))
}

/// Pack rectangles into a page by the MaxRects heuristic.
///
/// Inserted largest-first into the free rectangle that leaves the least slack, which packs
/// them tightly against each other and against the edges of the page.
///
/// # Arguments
///
/// - `sizes`: `(width, height)` per item.
/// - `page`: `(width, height)`.
/// - `allow_rotation`: whether an item may go in turned a quarter turn.
/// - `optional`: skip items that fit nowhere instead of failing the packing.
/// - `free`: space to pack into, e.g. what an earlier call left over. Defaults to the
///   whole page.
///
/// # Returns
///
/// `None` if some rectangle fit nowhere and `optional` was not set.
pub fn pack_rectangles(
    sizes: &[(f64, f64)],
    page: (f64, f64),
    allow_rotation: bool,
    optional: bool,
    free: Option<Vec<Rect>>,
) -> Option<RectPacking> {
    let mut free = free.unwrap_or_else(|| {
        vec![Rect { x: 0.0, y: 0.0, w: page.0, h: page.1 }]
    });
    let mut positions = vec![None; sizes.len()];
    let mut rotated = vec![false; sizes.len()];

    let mut order: Vec<usize> = (0..sizes.len()).collect();
    order.sort_by(|&a, &b| {
        let key = |i: usize| sizes[i].0.max(sizes[i].1);
        key(b).partial_cmp(&key(a)).unwrap_or(std::cmp::Ordering::Equal)
    });

    for i in order {
        let (w, h) = sizes[i];
        match insert_rect(&mut free, w, h, allow_rotation) {
            Some((px, py, rot)) => {
                positions[i] = Some((px, py));
                rotated[i] = rot;
            }
            None if optional => continue,
            None => return None,
        }
    }

    Some(RectPacking {
        positions,
        rotated,
        free,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(h: usize, w: usize) -> Bitmap {
        let mut b = Bitmap::new(h, w);
        for y in 0..h {
            for x in 0..w {
                b.set(y, x);
            }
        }
        b
    }

    fn ring(n: usize) -> Bitmap {
        let mut b = Bitmap::new(n, n);
        for i in 0..n {
            b.set(0, i);
            b.set(n - 1, i);
            b.set(i, 0);
            b.set(i, n - 1);
        }
        b
    }

    /// Nothing may overlap, and everything must sit inside the page.
    fn assert_disjoint(masks: &[Vec<Bitmap>], p: &Packing, page: (usize, usize)) {
        let mut seen = Bitmap::new(page.0, page.1);
        for (i, pos) in p.positions.iter().enumerate() {
            let Some((x, y)) = *pos else { continue };
            let m = &masks[i][p.variant[i]];
            assert!(y + m.height() <= page.0 && x + m.width() <= page.1, "item {i} off the page");
            for my in 0..m.height() {
                for mx in 0..m.width() {
                    if m.get(my, mx) {
                        assert!(!seen.get(y + my, x + mx), "item {i} overlaps at {my},{mx}");
                        seen.set(y + my, x + mx);
                    }
                }
            }
        }
    }

    #[test]
    fn bottom_left_fills_from_the_corner() {
        let masks = vec![vec![solid(4, 4)], vec![solid(4, 4)], vec![solid(4, 4)]];
        let p = pack_masks(&masks, (12, 8), None, None, false, None).unwrap();
        // Two across the bottom, the third on top of them
        let mut got: Vec<_> = p.positions.iter().map(|q| q.unwrap()).collect();
        got.sort();
        assert_eq!(got, vec![(0, 0), (0, 4), (4, 0)]);
        assert_disjoint(&masks, &p, (12, 8));
    }

    /// The point of packing masks rather than boxes: a shape may sit in another's hole.
    #[test]
    fn a_shape_nests_inside_the_loop_of_another() {
        let masks = vec![vec![ring(9)], vec![solid(3, 3)]];
        let p = pack_masks(&masks, (9, 9), None, None, false, None).unwrap();
        assert_eq!(p.positions[0], Some((0, 0)));
        // The ring leaves rows/cols 1..=7 clear inside it, and bottom-left takes the
        // first of those. A bounding-box packer could not place it on this page at all.
        assert_eq!(p.positions[1], Some((1, 1)), "the small square should nest");
        assert_disjoint(&masks, &p, (9, 9));
    }

    #[test]
    fn a_mask_that_does_not_fit_fails_the_packing() {
        let masks = vec![vec![solid(4, 4)], vec![solid(4, 4)]];
        assert!(pack_masks(&masks, (4, 4), None, None, false, None).is_none());
    }

    #[test]
    fn optional_skips_what_does_not_fit() {
        let masks = vec![vec![solid(4, 4)], vec![solid(4, 4)]];
        let p = pack_masks(&masks, (4, 4), None, None, true, None).unwrap();
        assert_eq!(p.positions.iter().filter(|q| q.is_some()).count(), 1);
        assert_disjoint(&masks, &p, (4, 4));
    }

    #[test]
    fn a_prefilled_grid_is_respected() {
        let mut grid = Bitmap::new(8, 8);
        grid.paint(&solid(4, 8), 0, 0); // bottom half taken
        let masks = vec![vec![solid(4, 8)]];
        let p = pack_masks(&masks, (8, 8), Some(grid), None, false, None).unwrap();
        assert_eq!(p.positions[0], Some((0, 4)));
    }

    #[test]
    fn rotation_is_used_when_it_is_the_only_way() {
        // A 2x6 page and a 6x2 shape: only the turned variant fits.
        let masks = vec![vec![solid(6, 2), solid(2, 6)]];
        let p = pack_masks(&masks, (2, 6), None, None, false, None).unwrap();
        assert_eq!(p.variant[0], 1);
        assert_eq!(p.positions[0], Some((0, 0)));
    }

    #[test]
    fn cost_pulls_shapes_towards_its_minimum() {
        let (h, w) = (21, 21);
        // Cheapest at the centre, rising outwards
        let cost: Vec<f64> = (0..h * w)
            .map(|c| {
                let (y, x) = ((c / w) as f64, (c % w) as f64);
                ((y - 10.0).powi(2) + (x - 10.0).powi(2)).sqrt()
            })
            .collect();
        let masks = vec![vec![solid(3, 3)]];
        let p = pack_masks(&masks, (h, w), None, Some(&cost), false, None).unwrap();
        // Centre of a 3x3 at corner (y, x) is (y + 1, x + 1), so the corner lands at (9, 9)
        assert_eq!(p.positions[0], Some((9, 9)));
    }

    #[test]
    fn cost_packs_successive_shapes_outwards() {
        let (h, w) = (31, 31);
        let cost: Vec<f64> = (0..h * w)
            .map(|c| {
                let (y, x) = ((c / w) as f64, (c % w) as f64);
                ((y - 15.0).powi(2) + (x - 15.0).powi(2)).sqrt()
            })
            .collect();
        let masks: Vec<Vec<Bitmap>> = (0..8).map(|_| vec![solid(5, 5)]).collect();
        let p = pack_masks(&masks, (h, w), None, Some(&cost), false, None).unwrap();
        assert_disjoint(&masks, &p, (h, w));
        // The first one down takes the middle
        let centres: Vec<f64> = p
            .positions
            .iter()
            .map(|q| {
                let (x, y) = q.unwrap();
                (((y + 2) as f64 - 15.0).powi(2) + ((x + 2) as f64 - 15.0).powi(2)).sqrt()
            })
            .collect();
        assert!(centres.iter().cloned().fold(f64::INFINITY, f64::min) < 1e-9);
    }

    #[test]
    fn word_boundaries_are_handled() {
        // Shapes wider than a word, at offsets that straddle one.
        let masks: Vec<Vec<Bitmap>> = (0..6).map(|_| vec![solid(10, 70)]).collect();
        let p = pack_masks(&masks, (30, 210), None, None, false, None).unwrap();
        assert!(p.positions.iter().all(|q| q.is_some()));
        assert_disjoint(&masks, &p, (30, 210));
        assert_eq!(p.grid.count_ones(), 6 * 10 * 70);
    }

    /// Nothing about the answer may depend on how wide the search ran — not the scan over
    /// positions, and not the variants of a shape racing each other.
    #[test]
    fn the_packing_does_not_depend_on_the_thread_count() {
        let (h, w) = (61, 83);
        let cost: Vec<f64> = (0..h * w)
            .map(|c| {
                let (y, x) = ((c / w) as f64, (c % w) as f64);
                (y - 30.0).hypot(x - 41.0)
            })
            .collect();
        // Variants that are *not* congruent, so which one wins is a real decision, and
        // enough of them to leave ties for the ordering to break.
        let masks: Vec<Vec<Bitmap>> = (0..14)
            .map(|i| vec![solid(3 + i % 4, 5 + i % 3), solid(5 + i % 3, 3 + i % 4), ring(6)])
            .collect();

        for cost in [None, Some(cost.as_slice())] {
            let one = pack_masks(&masks, (h, w), None, cost, true, Some(1)).unwrap();
            assert!(
                one.positions.iter().all(|p| p.is_some()),
                "nothing to compare if the shapes did not go down"
            );
            for n in [2, 4, 8] {
                let many = pack_masks(&masks, (h, w), None, cost, true, Some(n)).unwrap();
                assert_eq!(one.positions, many.positions, "positions differ on {n} threads");
                assert_eq!(one.variant, many.variant, "variants differ on {n} threads");
                assert_eq!(one.grid, many.grid, "pages differ on {n} threads");
            }
        }
    }

    #[test]
    fn rectangles_pack_into_the_page() {
        let sizes = vec![(4.0, 4.0), (4.0, 4.0), (4.0, 4.0), (4.0, 4.0)];
        let p = pack_rectangles(&sizes, (8.0, 8.0), false, false, None).unwrap();
        let mut got: Vec<(i64, i64)> = p
            .positions
            .iter()
            .map(|q| {
                let (x, y) = q.unwrap();
                (x as i64, y as i64)
            })
            .collect();
        got.sort();
        assert_eq!(got, vec![(0, 0), (0, 4), (4, 0), (4, 4)]);
    }

    #[test]
    fn rectangles_that_do_not_fit_fail_the_packing() {
        let sizes = vec![(5.0, 5.0), (5.0, 5.0)];
        assert!(pack_rectangles(&sizes, (8.0, 8.0), false, false, None).is_none());
        let p = pack_rectangles(&sizes, (8.0, 8.0), false, true, None).unwrap();
        assert_eq!(p.positions.iter().filter(|q| q.is_some()).count(), 1);
    }

    #[test]
    fn rectangles_rotate_when_allowed() {
        let sizes = vec![(8.0, 2.0)];
        assert!(pack_rectangles(&sizes, (2.0, 8.0), false, false, None).is_none());
        let p = pack_rectangles(&sizes, (2.0, 8.0), true, false, None).unwrap();
        assert!(p.rotated[0]);
    }

    /// Packed rectangles must not overlap - the free-list splitting is where that is easy
    /// to get wrong.
    #[test]
    fn packed_rectangles_never_overlap() {
        let sizes: Vec<(f64, f64)> = (1..=12)
            .map(|i| ((i % 5 + 1) as f64, (i % 3 + 1) as f64))
            .collect();
        let p = pack_rectangles(&sizes, (20.0, 20.0), false, true, None).unwrap();
        let placed: Vec<(f64, f64, f64, f64)> = p
            .positions
            .iter()
            .zip(&sizes)
            .filter_map(|(q, s)| q.map(|(x, y)| (x, y, s.0, s.1)))
            .collect();
        assert!(placed.len() > 6, "expected most of them to fit");
        for (i, a) in placed.iter().enumerate() {
            assert!(a.0 + a.2 <= 20.0 + EPS && a.1 + a.3 <= 20.0 + EPS, "off the page");
            for b in &placed[i + 1..] {
                let overlap = a.0 < b.0 + b.2 - EPS
                    && b.0 < a.0 + a.2 - EPS
                    && a.1 < b.1 + b.3 - EPS
                    && b.1 < a.1 + a.3 - EPS;
                assert!(!overlap, "{a:?} overlaps {b:?}");
            }
        }
    }

    #[test]
    fn free_space_carries_into_a_second_call() {
        let first = pack_rectangles(&[(4.0, 8.0)], (8.0, 8.0), false, false, None).unwrap();
        let second = pack_rectangles(
            &[(4.0, 8.0)],
            (8.0, 8.0),
            false,
            false,
            Some(first.free.clone()),
        )
        .unwrap();
        let (x0, _) = first.positions[0].unwrap();
        let (x1, _) = second.positions[0].unwrap();
        assert!((x0 - x1).abs() >= 4.0 - EPS, "the second should go beside the first");
    }
}
