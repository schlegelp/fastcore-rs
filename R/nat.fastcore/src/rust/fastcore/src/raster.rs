//! Binary images as bit-packed rows, and turning line work into one.
//!
//! [`Bitmap`] is a 1-bit image stored 64 pixels to a `u64`. That packing is the whole
//! point of the module: the operations built on it — dilation, hole filling, and above
//! all the collision test in [`crate::packing`] — are pure boolean algebra over pixels,
//! so a word-at-a-time implementation does 64 pixels per instruction and a `bool` array
//! does one. It also makes a page-sized image small enough to stay in cache: a 1300x980
//! page is 160 kB packed against 1.3 MB as bytes.
//!
//! [`rasterize_segments`] is the producer: it walks a set of line segments — a skeleton's
//! node-to-parent edges, a mesh's face edges — and marks every pixel they pass through.
//! Done in numpy this is the standard "interpolate every edge, then scatter" trick, which
//! materialises one element per *pixel-step of every edge*: on a mesh with 300k edges
//! averaging five pixels each that is a 1.5M-element index array, plus the same again for
//! the parameter and both coordinates, to set a few tens of thousands of distinct pixels.
//! Here the walk writes straight into the bitmap, and a long enough one does so on every
//! core at once (see [`MIN_EDGES_PER_CHUNK`]).

use ndarray::ArrayView2;
use rayon::prelude::*;

use crate::threads::with_pool;

/// Pixels per storage word.
const BITS: usize = u64::BITS as usize;

/// Fewest edges [`rasterize_segments`] will give a core of its own.
///
/// The split itself is one chunk per worker, as everywhere else in this crate. This says
/// only how fine that may get — and, at twice this, when a walk is long enough to be worth
/// cutting up at all.
///
/// [`rasterize_segments_batch`] parallelises over shapes, which is enough when there are
/// many — but the biggest shape is then a floor on the whole batch however many cores
/// there are, and a neuron list is not uniform: one projection neuron's mesh can carry a
/// hundred times the edges of a local interneuron's. Splitting the walk puts that floor
/// back under the machine's control, and because the split is a nested `par_iter` the
/// chunks are simply stolen by whichever threads finished their own shapes first.
///
/// A chunk costs a bitmap of the *whole shape* to draw into and an OR of it to fold back
/// in, so the floor is really about the bitmap and not about the edges: split a sparse
/// shape too finely and the empty bitmaps cost more than the walk they save. Measured at
/// 14 threads on a 30k-edge arbor whose bitmap is 7.4 MB, forced to split: 8.6 ms in one
/// piece, 11.8 ms in chunks of 2048. That shape does not reach two chunks at this value
/// and so is left alone. What does reach them pays off — a 300k-edge mesh outline goes
/// 4.18 -> 1.9 ms, and a batch of 4 of those among 20 small shapes 4.46 -> 2.83 ms.
const MIN_EDGES_PER_CHUNK: usize = 16_384;

/// A 1-bit-per-pixel image, packed into `u64` words row by row.
///
/// Row `0` is whichever end of the image the caller treats as first; nothing in here
/// cares. Column `x` of row `y` lives in bit `x % 64` of word `x / 64` of that row.
///
/// # Invariant
///
/// The bits past `width` in a row's last word are always zero. Every method that writes
/// maintains that, and the ones that read rely on it — the collision test compares whole
/// words, so a stray bit in the padding would read as a pixel that is not there.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Bitmap {
    height: usize,
    width: usize,
    /// Words per row. `width.div_ceil(64)`.
    stride: usize,
    bits: Vec<u64>,
}

/// Bits `0..width % 64` of a row's last word — the ones that are really pixels.
fn tail_mask(width: usize) -> u64 {
    match width % BITS {
        0 => u64::MAX,
        n => (1u64 << n) - 1,
    }
}

/// Shift a row's bits toward higher `x` by `s`, in place. Bits pushed off the end are lost.
fn shift_up(row: &mut [u64], s: usize) {
    if s == 0 {
        return;
    }
    let (wo, bo) = (s / BITS, s % BITS);
    let n = row.len();
    if wo >= n {
        row.fill(0);
        return;
    }
    // Descending, so a word is read before it is overwritten.
    for j in (0..n).rev() {
        let hi = if j >= wo { row[j - wo] } else { 0 };
        row[j] = if bo == 0 {
            hi
        } else {
            let lo = if j > wo { row[j - wo - 1] >> (BITS - bo) } else { 0 };
            (hi << bo) | lo
        };
    }
}

/// Shift a row's bits toward lower `x` by `s`, in place. Bits pushed off the end are lost.
fn shift_down(row: &mut [u64], s: usize) {
    if s == 0 {
        return;
    }
    let (wo, bo) = (s / BITS, s % BITS);
    let n = row.len();
    if wo >= n {
        row.fill(0);
        return;
    }
    for j in 0..n {
        let lo = if j + wo < n { row[j + wo] } else { 0 };
        row[j] = if bo == 0 {
            lo
        } else {
            let hi = if j + wo + 1 < n {
                row[j + wo + 1] << (BITS - bo)
            } else {
                0
            };
            (lo >> bo) | hi
        };
    }
}

impl Bitmap {
    /// An all-zero `height` x `width` image.
    pub fn new(height: usize, width: usize) -> Self {
        let stride = width.div_ceil(BITS);
        Self {
            height,
            width,
            stride,
            bits: vec![0; height * stride],
        }
    }

    /// Rows.
    pub fn height(&self) -> usize {
        self.height
    }

    /// Columns.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Words per row.
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// Pixels — `height * width`. What [`crate::packing::pack_masks`] sorts on.
    pub fn area(&self) -> usize {
        self.height * self.width
    }

    /// Set pixels.
    pub fn count_ones(&self) -> usize {
        self.bits.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// Row `y`'s words.
    #[inline]
    pub fn row(&self, y: usize) -> &[u64] {
        &self.bits[y * self.stride..(y + 1) * self.stride]
    }

    #[inline]
    fn row_mut(&mut self, y: usize) -> &mut [u64] {
        let s = self.stride;
        &mut self.bits[y * s..(y + 1) * s]
    }

    /// Whether pixel `(y, x)` is set. Out-of-range coordinates read as unset.
    #[inline]
    pub fn get(&self, y: usize, x: usize) -> bool {
        if y >= self.height || x >= self.width {
            return false;
        }
        self.row(y)[x / BITS] >> (x % BITS) & 1 == 1
    }

    /// Set pixel `(y, x)`. Out-of-range coordinates are ignored.
    #[inline]
    pub fn set(&mut self, y: usize, x: usize) {
        if y >= self.height || x >= self.width {
            return;
        }
        let s = self.stride;
        self.bits[y * s + x / BITS] |= 1u64 << (x % BITS);
    }

    /// Whether row `y` has no pixels at all.
    #[inline]
    pub fn row_is_empty(&self, y: usize) -> bool {
        self.row(y).iter().all(|&w| w == 0)
    }

    /// Restore the padding invariant on every row.
    fn clear_tails(&mut self) {
        if self.stride == 0 {
            return;
        }
        let m = tail_mask(self.width);
        let (stride, height) = (self.stride, self.height);
        for y in 0..height {
            self.bits[y * stride + stride - 1] &= m;
        }
    }

    /// Build from a row-major `bool` slice of `height * width` values.
    ///
    /// # Panics
    ///
    /// If `flat.len() != height * width`.
    pub fn from_bools(flat: &[bool], height: usize, width: usize) -> Self {
        assert_eq!(
            flat.len(),
            height * width,
            "expected {height} x {width} = {} values, got {}",
            height * width,
            flat.len()
        );
        let mut out = Self::new(height, width);
        for y in 0..height {
            let src = &flat[y * width..(y + 1) * width];
            let dst = out.row_mut(y);
            for (x, &v) in src.iter().enumerate() {
                if v {
                    dst[x / BITS] |= 1u64 << (x % BITS);
                }
            }
        }
        out
    }

    /// Unpack into a row-major `bool` vector of `height * width` values.
    pub fn to_bools(&self) -> Vec<bool> {
        let mut out = vec![false; self.height * self.width];
        for y in 0..self.height {
            let src = self.row(y);
            let dst = &mut out[y * self.width..(y + 1) * self.width];
            for (x, v) in dst.iter_mut().enumerate() {
                *v = src[x / BITS] >> (x % BITS) & 1 == 1;
            }
        }
        out
    }

    /// OR `other` into `self` with its top-left corner at `(y, x)`.
    ///
    /// Anything reaching past an edge of `self` is dropped. The whole row moves by the
    /// same bit offset — `x % 64` — because a mask word covers 64 consecutive columns,
    /// so each one lands across at most two words of the destination.
    pub fn paint(&mut self, other: &Bitmap, y: usize, x: usize) {
        let (wb, s) = (x / BITS, x % BITS);
        let stride = self.stride;
        if stride == 0 {
            return;
        }
        let tail = tail_mask(self.width);
        for i in 0..other.height {
            let ty = y + i;
            if ty >= self.height {
                break;
            }
            let base = ty * stride;
            for (j, &m) in other.row(i).iter().enumerate() {
                if m == 0 {
                    continue;
                }
                if wb + j < stride {
                    self.bits[base + wb + j] |= m << s;
                }
                if s > 0 && wb + j + 1 < stride {
                    self.bits[base + wb + j + 1] |= m >> (BITS - s);
                }
            }
            // Only a row that was written can have gained a stray bit past `width`, so
            // the tails are restored here rather than by a sweep of the whole page —
            // which, against a mask a fraction of its height, would cost more than the
            // paint it follows.
            self.bits[base + stride - 1] &= tail;
        }
    }

    /// OR `other` into `self`, which must be the same size.
    ///
    /// Both sides hold the padding invariant and OR cannot set a bit neither had, so the
    /// result holds it too — no `clear_tails` needed.
    ///
    /// This is `paint(other, 0, 0)` with the aligned case written out: one flat pass the
    /// compiler can vectorise, against `paint`'s row-major walk with a zero-word test and a
    /// tail mask per row. Worth the duplication only because it folds whole shape-sized
    /// bitmaps together in [`rasterize_segments`]; anything at an offset should use `paint`.
    fn union(&mut self, other: &Bitmap) {
        // `zip` would otherwise silently stop at the shorter of the two and quietly drop
        // half the drawing.
        debug_assert_eq!(
            (self.height, self.width),
            (other.height, other.width),
            "union needs matching sizes"
        );
        for (a, b) in self.bits.iter_mut().zip(&other.bits) {
            *a |= b;
        }
    }

    /// Grow every set pixel into a disk of radius `r`, returning a new bitmap.
    ///
    /// Matches `scipy.ndimage.binary_dilation` with a `dy^2 + dx^2 <= r^2` structuring
    /// element. Rows are done one `dy` at a time: for a disk the `dx` offsets at a given
    /// `dy` are the contiguous range `-dxmax..=dxmax`, and a contiguous range is a
    /// horizontal dilation, which takes `log2(dxmax)` OR-shifts rather than `2*dxmax+1`.
    pub fn dilate_disk(&self, r: usize) -> Bitmap {
        if r == 0 {
            return self.clone();
        }
        let mut out = Bitmap::new(self.height, self.width);
        let r2 = (r * r) as i64;
        // Reused across every row so the doubling below allocates once, not h * (2r+1) times.
        let mut work = vec![0u64; self.stride];
        let mut up = vec![0u64; self.stride];
        let mut down = vec![0u64; self.stride];

        for dy in -(r as i64)..=(r as i64) {
            let dxmax = ((r2 - dy * dy) as f64).sqrt().floor() as usize;
            for y in 0..self.height {
                let ty = y as i64 + dy;
                if ty < 0 || ty >= self.height as i64 || self.row_is_empty(y) {
                    continue;
                }
                work.copy_from_slice(self.row(y));
                // Dilating by `a` and then by `b` dilates by `a + b`, so double the step
                // until the remaining radius is used up.
                let (mut done, mut step) = (0usize, 1usize);
                while done < dxmax {
                    let s = step.min(dxmax - done);
                    up.copy_from_slice(&work);
                    down.copy_from_slice(&work);
                    shift_up(&mut up, s);
                    shift_down(&mut down, s);
                    for j in 0..self.stride {
                        work[j] |= up[j] | down[j];
                    }
                    done += s;
                    step *= 2;
                }
                let dst = out.row_mut(ty as usize);
                for j in 0..self.stride {
                    dst[j] |= work[j];
                }
            }
        }
        out.clear_tails();
        out
    }

    /// Set every unset pixel that cannot reach outside the image 4-connected.
    ///
    /// `scipy.ndimage.binary_fill_holes`, which is what turns a mesh's wireframe into the
    /// silhouette it actually occupies. Flood-filled by spans rather than pixel by pixel.
    pub fn fill_holes(&mut self) {
        if self.height == 0 || self.width == 0 {
            return;
        }
        let (h, w) = (self.height, self.width);
        let mut seen = Bitmap::new(h, w);
        let mut stack: Vec<(usize, usize)> = Vec::new();

        // The background reaches the outside iff it reaches a border pixel, so those are
        // the seeds — the same thing scipy's pad-fill-unpad does.
        for x in 0..w {
            if !self.get(0, x) {
                stack.push((0, x));
            }
            if h > 1 && !self.get(h - 1, x) {
                stack.push((h - 1, x));
            }
        }
        for y in 0..h {
            if !self.get(y, 0) {
                stack.push((y, 0));
            }
            if w > 1 && !self.get(y, w - 1) {
                stack.push((y, w - 1));
            }
        }

        while let Some((y, x)) = stack.pop() {
            if self.get(y, x) || seen.get(y, x) {
                continue;
            }
            // Widen to the whole free span through (y, x), then mark it in one go.
            let mut lo = x;
            while lo > 0 && !self.get(y, lo - 1) && !seen.get(y, lo - 1) {
                lo -= 1;
            }
            let mut hi = x;
            while hi + 1 < w && !self.get(y, hi + 1) && !seen.get(y, hi + 1) {
                hi += 1;
            }
            for cx in lo..=hi {
                seen.set(y, cx);
            }
            // One seed per free run in the rows above and below - pushing every pixel
            // would revisit the whole span for each of them.
            for ny in [y.wrapping_sub(1), y + 1] {
                if ny >= h {
                    continue;
                }
                let mut cx = lo;
                while cx <= hi {
                    if !self.get(ny, cx) && !seen.get(ny, cx) {
                        stack.push((ny, cx));
                        while cx <= hi && !self.get(ny, cx) {
                            cx += 1;
                        }
                    }
                    cx += 1;
                }
            }
        }

        // A hole is background the flood never reached.
        for j in 0..self.bits.len() {
            self.bits[j] |= !(self.bits[j] | seen.bits[j]);
        }
        self.clear_tails();
    }
}

/// How [`rasterize_segments`] turns coordinates into pixels.
#[derive(Clone, Copy, Debug)]
pub struct RasterOptions {
    /// Pixels per coordinate unit.
    pub scale: f64,
    /// Margin left around the line work, in pixels, and the radius the result is dilated
    /// by. Two shapes whose bitmaps do not overlap then keep `2 * pad` pixels of clear
    /// space between the lines themselves.
    pub pad: usize,
    /// Fill the interior afterwards. What a solid shape wants: what it occupies is its
    /// silhouette and not the wireframe of its edges.
    pub fill: bool,
    /// Quarter turns counter-clockwise to apply before rasterising, taken mod 4. `1` is
    /// `(u, v) -> (-v, u)`.
    ///
    /// All four are here because the caller's own axes may be flipped relative to these:
    /// composed with a mirrored axis, a counter-clockwise turn *is* the clockwise one, so
    /// a `bool` would give half of such callers a point-reflected shape. Turning here
    /// rather than rotating the coordinates first saves a copy of them per variant, which
    /// for a large mesh is the biggest array in play.
    pub turn: u8,
}

impl Default for RasterOptions {
    fn default() -> Self {
        Self {
            scale: 1.0,
            pad: 0,
            fill: false,
            turn: 0,
        }
    }
}

/// Mark every pixel the given line segments pass through.
///
/// `coords` is `(N, 2)` — the two in-plane coordinates, in whatever units — and `edges`
/// is `(E, 2)` index pairs into it. The shape is shifted so its own lower-left corner
/// sits at `(pad, pad)` and scaled by `RasterOptions::scale`; the bitmap comes out just
/// big enough to hold it with `pad` pixels of margin on every side.
///
/// Vertices no edge names are ignored, since the segments are what is being drawn — but
/// if there are no edges at all, every vertex is marked, so a bare point cloud still
/// rasterises to something.
///
/// Past [`MIN_EDGES_PER_CHUNK`] edges the walk runs on the ambient rayon pool; callers who
/// need to cap it wrap it in [`with_pool`], which is what [`rasterize_segments_batch`]
/// does. Unlike this crate's other entry points there is no `threads` argument, because
/// the batch is the one that owns the pool — building one per shape would spawn a pool per
/// neuron.
///
/// # Panics
///
/// If `edges` names a vertex that is not in `coords`.
pub fn rasterize_segments(
    coords: ArrayView2<'_, f64>,
    edges: ArrayView2<'_, u32>,
    opts: &RasterOptions,
) -> Bitmap {
    assert_eq!(coords.ncols(), 2, "`coords` must be (N, 2)");
    assert_eq!(edges.ncols(), 2, "`edges` must be (E, 2)");
    let n = coords.nrows();

    // Turned and bounded in one pass, into a buffer the edge walk then indexes. In a
    // triangle mesh a vertex is an endpoint of a dozen edges, so transforming per
    // endpoint would redo each of these a dozen times - and through ndarray's 2-D
    // indexing rather than a slice.
    let mut pts: Vec<(f64, f64)> = Vec::with_capacity(n);
    let mut lo = (f64::INFINITY, f64::INFINITY);
    let mut hi = (f64::NEG_INFINITY, f64::NEG_INFINITY);
    let mut bad = false;
    for i in 0..n {
        let (u, v) = (coords[[i, 0]], coords[[i, 1]]);
        // Tracked explicitly rather than left to the bounds: `f64::min` *ignores* a NaN and
        // returns the other operand, so a NaN coordinate would leave `lo`/`hi` looking
        // perfectly finite and rasterise to a quietly wrong shape. Accumulated and checked
        // after the loop rather than returned from inside it — an early return out of the
        // middle of this is a loop-carried branch that stops the min/max reduction
        // vectorising, and it costs the good path about half the pass (0.46 -> 0.24 ms on
        // 300k points) to hurry along the one that is about to throw everything away.
        bad |= !u.is_finite() || !v.is_finite();
        let p = match opts.turn % 4 {
            1 => (-v, u),
            2 => (-u, -v),
            3 => (v, -u),
            _ => (u, v),
        };
        lo = (lo.0.min(p.0), lo.1.min(p.1));
        hi = (hi.0.max(p.0), hi.1.max(p.1));
        pts.push(p);
    }
    // Either a coordinate was not finite, or there were no points at all and there is no
    // extent to size a bitmap from.
    if bad || !lo.0.is_finite() || !hi.0.is_finite() {
        return Bitmap::new(0, 0);
    }

    // Into pixels, in place - the untransformed points are not wanted again.
    let pad = opts.pad as f64;
    for p in pts.iter_mut() {
        *p = (
            (p.0 - lo.0) * opts.scale + pad,
            (p.1 - lo.1) * opts.scale + pad,
        );
    }

    let width = ((hi.0 - lo.0) * opts.scale + pad).ceil() as usize + opts.pad + 1;
    let height = ((hi.1 - lo.1) * opts.scale + pad).ceil() as usize + opts.pad + 1;

    if edges.nrows() == 0 {
        let mut out = Bitmap::new(height, width);
        for &(x, y) in &pts {
            out.set(round(y), round(x));
        }
        return finish(out, opts);
    }

    // Drawing a range of edges into a bitmap: the whole walk when it is short, one chunk of
    // it per worker when it is not.
    let draw = |out: &mut Bitmap, range: std::ops::Range<usize>| {
        for e in range {
            let (a, b) = (edges[[e, 0]] as usize, edges[[e, 1]] as usize);
            assert!(
                a < n && b < n,
                "`edges` names vertex {}, but there are only {n} vertices",
                a.max(b)
            );
            let ((x0, y0), (x1, y1)) = (pts[a], pts[b]);
            let (dx, dy) = (x1 - x0, y1 - y0);

            // One step per pixel of the longer axis, both ends included - a coarser walk
            // would leave gaps wherever the shape is sampled more sparsely than the grid.
            let steps = dx.abs().max(dy.abs()).ceil().max(1.0) as usize;
            for k in 0..=steps {
                let t = k as f64 / steps as f64;
                let (x, y) = (x0 + t * dx, y0 + t * dy);
                out.set(round(y), round(x));
            }
        }
    };

    let n_edges = edges.nrows();
    let out = if n_edges < 2 * MIN_EDGES_PER_CHUNK {
        // Too short to be worth cutting up - and kept away from rayon altogether rather
        // than merely handed to it as a single chunk, because *asking* rayon anything,
        // `current_num_threads` included, builds the global pool as a side effect and
        // [`crate::threads::set_num_threads`] can never resize it afterwards. Most shapes
        // in a collage take this path, so it is the one that decides whether a caller can
        // still size the pool.
        let mut out = Bitmap::new(height, width);
        draw(&mut out, 0..n_edges);
        out
    } else {
        // One chunk per worker of the pool actually in force — under `with_pool(Some(1))`,
        // or on emscripten where nothing can spawn, that is one. Sizing by worker rather
        // than by a fixed edge count is what bounds the chunk bitmaps: a 3M-edge mesh would
        // otherwise build 183 of them on any machine, each the size of the whole shape.
        let n_chunks = rayon::current_num_threads().min(n_edges / MIN_EDGES_PER_CHUNK);
        let chunk = n_edges.div_ceil(n_chunks.max(1));

        // Each chunk draws into its own bitmap and they are OR-ed together. Setting a pixel
        // is idempotent and OR is associative, so the result is bit-identical to one serial
        // walk whatever order the chunks finish in. `reduce_with` and not `reduce`: rayon
        // calls a `reduce` identity once per sequential job, which would build and fold in
        // a second empty bitmap per chunk for nothing.
        (0..n_edges.div_ceil(chunk))
            .into_par_iter()
            .map(|c| {
                let mut part = Bitmap::new(height, width);
                draw(&mut part, c * chunk..((c + 1) * chunk).min(n_edges));
                part
            })
            .reduce_with(|mut a, b| {
                a.union(&b);
                a
            })
            .expect("there is at least one chunk")
    };

    finish(out, opts)
}

/// Round half to even, the way `np.rint` does.
///
/// `f64::round` rounds halves away from zero instead, which lands a pixel one over
/// wherever a coordinate falls exactly between two of them — not rare, since a shape is
/// placed on the grid by a scale factor that is frequently a whole number of pixels.
#[inline]
fn round(v: f64) -> usize {
    v.round_ties_even() as usize
}

/// Fill and dilate, in that order: dilating first would grow the outline inwards and
/// close off thin interiors that are really open.
fn finish(mut out: Bitmap, opts: &RasterOptions) -> Bitmap {
    if opts.fill {
        out.fill_holes();
    }
    if opts.pad > 0 {
        out = out.dilate_disk(opts.pad);
    }
    out
}

/// [`rasterize_segments`] over many shapes at once, one per core.
///
/// Worth the batching: a collage re-rasterises every shape at every step of its scale
/// search, and the shapes are independent, so this is the one place in the layout where
/// parallelism is free. Shapes long enough to split their own walk (see
/// [`MIN_EDGES_PER_CHUNK`]) do that too, on the same pool — which is what keeps one outsized
/// mesh in the list from setting the pace for all of it.
///
/// `fill`, when given, overrides [`RasterOptions::fill`] for each shape. Unlike `scale`
/// and `pad`, which are properties of the *page* and so are necessarily shared, whether a
/// shape is solid or a wireframe is a property of the shape — as much as its `coords` and
/// `edges` are. A caller with a mix would otherwise have to sort its shapes into two
/// batches and stitch the results back together.
///
/// # Panics
///
/// If `fill` is given and is not one flag per shape.
pub fn rasterize_segments_batch(
    shapes: &[(ArrayView2<'_, f64>, ArrayView2<'_, u32>)],
    opts: &RasterOptions,
    fill: Option<&[bool]>,
    threads: Option<usize>,
) -> Vec<Bitmap> {
    if let Some(f) = fill {
        assert_eq!(
            f.len(),
            shapes.len(),
            "`fill` must have one flag per shape, got {} for {} shapes",
            f.len(),
            shapes.len()
        );
    }
    with_pool(threads, || {
        shapes
            .par_iter()
            .enumerate()
            .map(|(i, (coords, edges))| {
                let opts = RasterOptions {
                    fill: fill.map_or(opts.fill, |f| f[i]),
                    ..*opts
                };
                rasterize_segments(coords.view(), edges.view(), &opts)
            })
            .collect()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array2};

    fn bm(rows: &[&str]) -> Bitmap {
        let h = rows.len();
        let w = rows[0].len();
        let mut b = Bitmap::new(h, w);
        for (y, r) in rows.iter().enumerate() {
            for (x, c) in r.chars().enumerate() {
                if c == '#' {
                    b.set(y, x);
                }
            }
        }
        b
    }

    fn show(b: &Bitmap) -> Vec<String> {
        (0..b.height())
            .map(|y| {
                (0..b.width())
                    .map(|x| if b.get(y, x) { '#' } else { '.' })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn set_and_get_round_trip_past_a_word_boundary() {
        let mut b = Bitmap::new(3, 200);
        for x in [0, 63, 64, 65, 127, 128, 199] {
            b.set(1, x);
        }
        for x in 0..200 {
            assert_eq!(b.get(1, x), [0, 63, 64, 65, 127, 128, 199].contains(&x));
        }
        assert!(b.row_is_empty(0) && b.row_is_empty(2));
        assert_eq!(b.count_ones(), 7);
    }

    #[test]
    fn bools_round_trip() {
        let b = bm(&["..#.", "#..#", "...."]);
        let flat = b.to_bools();
        assert_eq!(Bitmap::from_bools(&flat, 3, 4), b);
    }

    /// The padding bits past `width` must stay clear, or the collision test sees pixels
    /// that are not in the image.
    #[test]
    fn shifting_never_leaks_past_the_width() {
        let mut b = Bitmap::new(1, 70);
        b.set(0, 69);
        let d = b.dilate_disk(3);
        assert_eq!(d.count_ones(), 4, "{:?}", show(&d));
        assert!((66..70).all(|x| d.get(0, x)));
    }

    #[test]
    fn paint_places_at_an_arbitrary_offset() {
        let mut page = Bitmap::new(6, 100);
        let stamp = bm(&["##", ".#"]);
        page.paint(&stamp, 2, 63);
        assert!(page.get(2, 63) && page.get(2, 64) && page.get(3, 64));
        assert!(!page.get(3, 63));
        assert_eq!(page.count_ones(), 3);
    }

    #[test]
    fn paint_clips_at_the_edges() {
        let mut page = Bitmap::new(3, 3);
        page.paint(&bm(&["##", "##"]), 2, 2);
        assert_eq!(show(&page), vec!["...", "...", "..#"]);
    }

    #[test]
    fn dilate_disk_matches_the_structuring_element() {
        // r = 1 -> the four-neighbourhood, since only dy^2 + dx^2 <= 1 qualifies
        let d = bm(&[".....", ".....", "..#..", ".....", "....."]).dilate_disk(1);
        assert_eq!(
            show(&d),
            vec![".....", "..#..", ".###.", "..#..", "....."]
        );
        // r = 2 -> also the diagonals, since 1 + 1 <= 4
        let d = bm(&[".....", ".....", "..#..", ".....", "....."]).dilate_disk(2);
        assert_eq!(
            show(&d),
            vec!["..#..", ".###.", "#####", ".###.", "..#.."]
        );
    }

    #[test]
    fn fill_holes_closes_an_interior_but_not_a_bay() {
        let mut b = bm(&["#####", "#...#", "#.#.#", "#...#", "#####"]);
        b.fill_holes();
        assert_eq!(show(&b), vec!["#####"; 5]);

        // Open to the outside on the right - nothing to fill
        let open = bm(&["#####", "#....", "#.#..", "#....", "#####"]);
        let mut b = open.clone();
        b.fill_holes();
        assert_eq!(b, open);
    }

    /// Background that only reaches the outside by a long, winding path is the case a
    /// naive row-by-row propagation gets wrong — it fills the corridor.
    #[test]
    fn fill_holes_leaves_an_open_spiral_alone() {
        let spiral = bm(&[
            "#########",
            "#.......#",
            "#.#####.#",
            "#.#...#.#",
            "#.#.#.#.#",
            "#.#.#.#.#",
            "#.#.#...#",
            "#.#.#####",
            "#.#......",
        ]);
        let mut b = spiral.clone();
        b.fill_holes();
        assert_eq!(b, spiral, "an open corridor is not a hole");

        // Seal both mouths and the whole corridor becomes one.
        let mut sealed = spiral.clone();
        for x in 0..9 {
            sealed.set(8, x);
        }
        let mut b = sealed;
        b.fill_holes();
        assert_eq!(b.count_ones(), 81, "a sealed corridor is a hole throughout");
    }

    #[test]
    fn a_diagonal_segment_leaves_no_gaps() {
        let coords = array![[0.0, 0.0], [10.0, 10.0]];
        let edges = array![[0u32, 1u32]];
        let b = rasterize_segments(
            coords.view(),
            edges.view(),
            &RasterOptions {
                scale: 1.0,
                ..Default::default()
            },
        );
        assert_eq!((b.height(), b.width()), (11, 11));
        for i in 0..11 {
            assert!(b.get(i, i), "missing pixel at {i}, {i}");
        }
        assert_eq!(b.count_ones(), 11);
    }

    #[test]
    fn scale_and_pad_size_the_bitmap() {
        let coords = array![[0.0, 0.0], [2.0, 1.0]];
        let edges = array![[0u32, 1u32]];
        let b = rasterize_segments(
            coords.view(),
            edges.view(),
            &RasterOptions {
                scale: 10.0,
                pad: 3,
                ..Default::default()
            },
        );
        // 2 units * 10 px/unit = 20, plus 3 px of margin at each end, plus the +1
        assert_eq!((b.height(), b.width()), (10 + 3 + 3 + 1, 20 + 3 + 3 + 1));
        // Nothing may sit in the margin the padding reserved
        assert!(!b.get(0, 0));
    }

    /// Every quarter turn, on a shape with no symmetry left to hide a wrong one: an "L"
    /// of a long arm and a short one.
    #[test]
    fn turn_rotates_by_quarter_turns() {
        let coords = array![[0.0, 0.0], [4.0, 0.0], [0.0, 1.0]];
        let edges = array![[0u32, 1u32], [0u32, 2u32]];
        let of = |turn: u8| {
            let b = rasterize_segments(
                coords.view(),
                edges.view(),
                &RasterOptions {
                    turn,
                    ..Default::default()
                },
            );
            show(&b)
        };
        // Row 0 is the bottom, and each bitmap is cropped to the turned shape's own
        // extent. 1 and 3 are mirror images of each other - which is the whole reason
        // this is a quarter-turn count and not a bool.
        assert_eq!(of(0), vec!["#####", "#...."]);
        assert_eq!(of(1), vec!["##", ".#", ".#", ".#", ".#"]);
        assert_eq!(of(2), vec!["....#", "#####"]);
        assert_eq!(of(3), vec!["#.", "#.", "#.", "#.", "##"]);
        // Four of them is the identity
        assert_eq!(of(4), of(0));
    }

    #[test]
    fn batch_takes_fill_per_shape() {
        // A closed square: solid when filled, a rim when not.
        let coords = array![[0.0, 0.0], [6.0, 0.0], [6.0, 6.0], [0.0, 6.0]];
        let edges = array![[0u32, 1u32], [1, 2], [2, 3], [3, 0]];
        let shapes = vec![
            (coords.view(), edges.view()),
            (coords.view(), edges.view()),
        ];
        let out = rasterize_segments_batch(
            &shapes,
            &RasterOptions::default(),
            Some(&[false, true]),
            None,
        );
        assert_eq!(out[0].count_ones(), 4 * 6, "the first should be a rim");
        assert_eq!(out[1].count_ones(), 7 * 7, "the second should be solid");

        // Without the override both follow the options
        let out = rasterize_segments_batch(
            &shapes,
            &RasterOptions {
                fill: true,
                ..Default::default()
            },
            None,
            None,
        );
        assert!(out.iter().all(|b| b.count_ones() == 7 * 7));
    }

    #[test]
    #[should_panic(expected = "one flag per shape")]
    fn batch_rejects_a_mismatched_fill() {
        let coords = array![[0.0, 0.0], [1.0, 1.0]];
        let edges = array![[0u32, 1u32]];
        let shapes = vec![(coords.view(), edges.view())];
        rasterize_segments_batch(&shapes, &RasterOptions::default(), Some(&[true, false]), None);
    }

    #[test]
    fn no_edges_falls_back_to_the_points() {
        let coords = array![[0.0, 0.0], [3.0, 2.0]];
        let edges = ndarray::Array2::<u32>::zeros((0, 2));
        let b = rasterize_segments(coords.view(), edges.view(), &RasterOptions::default());
        assert_eq!(b.count_ones(), 2);
        assert!(b.get(0, 0) && b.get(2, 3));
    }

    #[test]
    fn non_finite_coordinates_give_an_empty_bitmap() {
        let coords = array![[0.0, 0.0], [f64::NAN, 1.0]];
        let edges = array![[0u32, 1u32]];
        let b = rasterize_segments(coords.view(), edges.view(), &RasterOptions::default());
        assert_eq!((b.height(), b.width()), (0, 0));
    }

    #[test]
    #[should_panic(expected = "only 2 vertices")]
    fn an_edge_out_of_range_is_rejected() {
        let coords = array![[0.0, 0.0], [1.0, 1.0]];
        let edges = array![[0u32, 5u32]];
        rasterize_segments(coords.view(), edges.view(), &RasterOptions::default());
    }

    /// A long walk is cut into one chunk per worker and the chunks are OR-ed back together,
    /// which must land on exactly the same pixels as walking it in one piece.
    ///
    /// All three draws here are of the same figure — repeating an edge cannot set a pixel
    /// the first pass did not — so the only thing that differs is how many chunks the walk
    /// was cut into, which is what the pool size decides.
    #[test]
    fn a_split_walk_matches_a_serial_one() {
        let n = 240;
        let coords: Vec<f64> = (0..n)
            .flat_map(|i| {
                let a = std::f64::consts::TAU * i as f64 / n as f64;
                // Not a circle: a wobble puts edges at every angle and length.
                let r = 1.0 + 0.4 * (7.0 * a).sin();
                [r * a.cos(), r * a.sin()]
            })
            .collect();
        let coords = Array2::from_shape_vec((n, 2), coords).unwrap();

        let ring: Vec<u32> = (0..n)
            .flat_map(|i| [i as u32, ((i + 1) % n) as u32])
            .collect();
        let once = Array2::from_shape_vec((n, 2), ring.clone()).unwrap();

        // Long enough that four workers really do take a chunk each.
        let reps = 4 * MIN_EDGES_PER_CHUNK / n + 1;
        let many = Array2::from_shape_vec((n * reps, 2), ring.repeat(reps)).unwrap();

        let opts = RasterOptions { scale: 40.0, ..Default::default() };
        let draw = |edges: &Array2<u32>, threads: usize| {
            with_pool(Some(threads), || {
                rasterize_segments(coords.view(), edges.view(), &opts)
            })
        };

        let serial = draw(&once, 1);
        assert!(serial.count_ones() > 0);
        assert_eq!(draw(&many, 1), serial, "one worker should not split at all");
        assert_eq!(draw(&many, 4), serial, "the split walk drew something else");
    }
}
