//! Extracting top matches from a score matrix.
//!
//! Given an `(n_query, n_target)` score matrix — typically NBLAST output — pull out
//! the best matches per query (or per target). Three criteria, mirroring navis'
//! `extract_matches`:
//!
//! - **top-N** ([`top_matches`]) — the `n` best cells per group. Dense output.
//! - **threshold** ([`matches_above`] with [`Criterion::Threshold`]) — every cell at or
//!   beyond an absolute cutoff. Ragged output.
//! - **percentage** ([`matches_above`] with [`Criterion::Percentage`]) — every cell within
//!   `p` of the group's *own* best value. Ragged output.
//!
//! [`count_matches`] is the counting half of [`matches_above`] on its own: it reports how
//! many matches each group *would* yield without allocating them, so a caller can size a
//! result before committing to it.
//!
//! # Why this module exists in Rust
//!
//! The matrices are big — a few 100k on a side, i.e. tens of GB. Everything here is
//! therefore written to be **memory-bandwidth bound**:
//!
//! - The matrix is **never copied and never transposed**. Per-column matches
//!   ([`MatchAxis::Cols`]) are served by walking *column stripes* of the row-major buffer,
//!   so the matrix is read exactly once with full cache-line utilisation. Walking a
//!   row-major matrix column-wise instead would fetch a 64-byte line to consume 4 bytes.
//! - The top-N inner loop rejects a non-qualifying cell in **one** comparison, against the
//!   current worst-of-the-best. Scanning all `n` buffer slots per cell (as the obvious
//!   implementation does) costs `n`× the work on the overwhelmingly common path.
//! - `distances` is monomorphised into the kernel via [`Dir`], not branched on per cell.
//!
//! # Conventions
//!
//! - **Ties break toward the lower index**, in every criterion, so results are
//!   deterministic and independent of the thread count.
//! - **`NaN` is never a match.** All comparisons are strict, and a `NaN` operand makes them
//!   false. A group with no valid cells yields `-1`/`NaN` slots (top-N) or an empty group
//!   (ragged).
//! - Scores are read at their native width (`f16`, `f32` or `f64`) and returned at it.

use half::f16;
use indicatif::ProgressBar;
use ndarray::ArrayView2;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::fmt;
use std::sync::atomic::AtomicBool;

use crate::nblast::{is_cancelled, make_bar, with_pool};

// ---------------------------------------------------------------------------
// Element type
// ---------------------------------------------------------------------------

/// An element of a score matrix: `f16`, `f32` or `f64`.
///
/// This is deliberately a local trait rather than [`num::Float`], which is not implemented
/// for [`half::f16`] — and `f16` is exactly the width someone with a 300k² matrix reaches
/// for, since it halves the 360 GB it would take at `f32`.
pub trait Score: Copy + PartialOrd + Send + Sync + 'static {
    const ZERO: Self;
    const NAN: Self;
    const INFINITY: Self;
    const NEG_INFINITY: Self;

    /// Narrow an `f64` (a user-supplied threshold) to this width. One cast per call, not
    /// per cell.
    fn from_f64(x: f64) -> Self;

    /// The `percentage` cutoff for a group whose best (finite) value is `m`.
    ///
    /// `m - |m * p|` for similarities, `m + |m * p|` for distances. The `abs()` is what
    /// widens the band in the right direction when `m` is **negative** — which NBLAST
    /// scores can be.
    fn perc_cutoff(m: Self, p: f64, distances: bool) -> Self;
}

impl Score for f32 {
    const ZERO: Self = 0.0;
    const NAN: Self = f32::NAN;
    const INFINITY: Self = f32::INFINITY;
    const NEG_INFINITY: Self = f32::NEG_INFINITY;

    #[inline]
    fn from_f64(x: f64) -> Self {
        x as f32
    }

    #[inline]
    fn perc_cutoff(m: Self, p: f64, distances: bool) -> Self {
        // Computed at f32, matching what numpy does for an f32 array against a weak
        // Python float (NEP 50).
        let band = (m * p as f32).abs();
        if distances {
            m + band
        } else {
            m - band
        }
    }
}

impl Score for f64 {
    const ZERO: Self = 0.0;
    const NAN: Self = f64::NAN;
    const INFINITY: Self = f64::INFINITY;
    const NEG_INFINITY: Self = f64::NEG_INFINITY;

    #[inline]
    fn from_f64(x: f64) -> Self {
        x
    }

    #[inline]
    fn perc_cutoff(m: Self, p: f64, distances: bool) -> Self {
        let band = (m * p).abs();
        if distances {
            m + band
        } else {
            m - band
        }
    }
}

impl Score for f16 {
    const ZERO: Self = f16::ZERO;
    const NAN: Self = f16::NAN;
    const INFINITY: Self = f16::INFINITY;
    const NEG_INFINITY: Self = f16::NEG_INFINITY;

    #[inline]
    fn from_f64(x: f64) -> Self {
        f16::from_f64(x)
    }

    #[inline]
    fn perc_cutoff(m: Self, p: f64, distances: bool) -> Self {
        // f16 has ~3 decimal digits, so the band is computed at f32 and rounded back
        // rather than accumulating two roundings at f16.
        let mf = m.to_f32();
        let band = (mf * p as f32).abs();
        f16::from_f32(if distances { mf + band } else { mf - band })
    }
}

// ---------------------------------------------------------------------------
// Direction (similarity vs distance), monomorphised
// ---------------------------------------------------------------------------

/// Which way is "better". Resolved once per call and monomorphised into the kernels, so
/// the hot loop never branches on `distances`.
trait Dir: Send + Sync {
    /// A sentinel that any real value beats: `-inf` for similarities, `+inf` for distances.
    ///
    /// Seeding a distance search with `-inf` (as the numba version in navis does) means
    /// `v < -inf` is never true and *nothing* is ever selected.
    fn worst<T: Score>() -> T;
    /// Strictly better. `NaN` on either side yields `false` — this is what excludes `NaN`.
    fn better<T: Score>(a: T, b: T) -> bool;
    /// Qualifies against a cutoff (inclusive). `NaN` on either side yields `false`.
    fn passes<T: Score>(v: T, t: T) -> bool;
}

/// Higher is better.
struct Sim;

impl Dir for Sim {
    #[inline(always)]
    fn worst<T: Score>() -> T {
        T::NEG_INFINITY
    }
    #[inline(always)]
    fn better<T: Score>(a: T, b: T) -> bool {
        a > b
    }
    #[inline(always)]
    fn passes<T: Score>(v: T, t: T) -> bool {
        v >= t
    }
}

/// Lower is better.
struct Dist;

impl Dir for Dist {
    #[inline(always)]
    fn worst<T: Score>() -> T {
        T::INFINITY
    }
    #[inline(always)]
    fn better<T: Score>(a: T, b: T) -> bool {
        a < b
    }
    #[inline(always)]
    fn passes<T: Score>(v: T, t: T) -> bool {
        v <= t
    }
}

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Which cells count as matches, for the ragged criteria.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Criterion {
    /// Every cell with `score >= t` (`<= t` when `distances`).
    Threshold(f64),
    /// Every cell within `p` (0-1) of the group's own best value.
    Percentage(f64),
}

/// Which axis matches are grouped by. Reported indices always run along the *other* axis.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatchAxis {
    /// navis `axis=0`: one group per row; indices are column indices.
    Rows,
    /// navis `axis=1`: one group per column; indices are row indices.
    Cols,
}

impl MatchAxis {
    /// The same request against a transposed view of the matrix.
    pub fn flip(self) -> Self {
        match self {
            MatchAxis::Rows => MatchAxis::Cols,
            MatchAxis::Cols => MatchAxis::Rows,
        }
    }
}

/// Per-call options.
#[derive(Clone, Copy)]
pub struct MatchOpts<'a> {
    pub axis: MatchAxis,
    /// Lower is better (a distance matrix rather than a similarity matrix).
    pub distances: bool,
    /// For each group, one index along the scanned axis to ignore — its own self-match.
    /// `-1` means "skip nothing for this group". Length must equal the group count.
    pub skip: Option<&'a [i64]>,
    /// Refuse to allocate a ragged result larger than this. `None` = unlimited.
    pub max_matches: Option<u64>,
    /// Cap the rayon worker count (navis' `n_cores`).
    pub threads: Option<usize>,
    pub progress: bool,
    /// Cooperative cancellation, polled inside the scans.
    pub cancel: Option<&'a AtomicBool>,
}

impl Default for MatchOpts<'_> {
    fn default() -> Self {
        MatchOpts {
            axis: MatchAxis::Rows,
            distances: false,
            skip: None,
            max_matches: None,
            threads: None,
            progress: false,
            cancel: None,
        }
    }
}

/// Dense top-N result, row-major `(n_groups, n)`.
#[derive(Debug, Clone)]
pub struct TopN<T> {
    pub n_groups: usize,
    pub n: usize,
    /// `-1` where the group had fewer than `n` valid (non-`NaN`) cells.
    pub indices: Vec<i64>,
    /// `NaN` wherever the paired index is `-1`.
    pub values: Vec<T>,
}

/// Ragged result, CSR-style: group `g` occupies `offsets[g]..offsets[g + 1]`, best first.
#[derive(Debug, Clone)]
pub struct Ragged<T> {
    pub n_groups: usize,
    /// Length `n_groups + 1`; `offsets[0] == 0`. `i64`, not `u64`: it is an index
    /// array, and numpy refuses `uint64` as `np.repeat` counts — which would break the
    /// one recipe every caller of this needs.
    pub offsets: Vec<i64>,
    /// Indices along the scanned axis.
    pub indices: Vec<u32>,
    pub values: Vec<T>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum MatchError {
    /// The view was neither C- nor F-contiguous. Callers normalise F-order away by
    /// transposing the view and flipping the axis; anything else must be made contiguous
    /// by the caller, because a matrix this size must not be silently copied.
    NotContiguous,
    /// `n == 0`, or `n` exceeded the number of candidates along the scanned axis.
    InvalidN { n: usize, n_scan: usize },
    /// `percentage` outside `[0, 1]`, or a non-finite threshold.
    InvalidCriterion(String),
    /// `skip` was not one entry per group.
    SkipLen { got: usize, want: usize },
    /// The scanned axis is longer than `u32::MAX` (ragged indices are `u32`).
    AxisTooLong { len: usize },
    /// The cutoff would produce more matches than `max_matches` allows.
    TooManyMatches { total: u64, limit: u64 },
    Cancelled,
}

impl fmt::Display for MatchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MatchError::NotContiguous => write!(
                f,
                "`scores` must be C- or F-contiguous (got a strided view); pass a \
                 contiguous array — a matrix this size will not be copied for you"
            ),
            MatchError::InvalidN { n, n_scan } => write!(
                f,
                "`n` must be in 1..={n_scan} (the length of the scanned axis), got {n}"
            ),
            MatchError::InvalidCriterion(msg) => write!(f, "{msg}"),
            MatchError::SkipLen { got, want } => {
                write!(f, "`skip` must have one entry per group: got {got}, want {want}")
            }
            MatchError::AxisTooLong { len } => write!(
                f,
                "the scanned axis has {len} entries, which exceeds the u32 index range"
            ),
            MatchError::TooManyMatches { total, limit } => write!(
                f,
                "this cutoff yields {total} matches (limit {limit}); tighten the cutoff or \
                 raise `max_matches`"
            ),
            MatchError::Cancelled => write!(f, "interrupted"),
        }
    }
}

impl std::error::Error for MatchError {}

// ---------------------------------------------------------------------------
// Shared setup
// ---------------------------------------------------------------------------

/// The row-major buffer plus the group/scan extents implied by `axis`.
struct Layout<'a, T> {
    raw: &'a [T],
    n_rows: usize,
    n_cols: usize,
    /// Number of output groups.
    n_groups: usize,
    /// Length of each group's scan.
    n_scan: usize,
    by_rows: bool,
}

fn layout<'a, T: Score>(
    scores: &ArrayView2<'a, T>,
    opts: &MatchOpts,
) -> Result<Layout<'a, T>, MatchError> {
    // `to_slice` (not `as_slice`) hands back the *view's* lifetime, so the layout outlives
    // the local binding. It is `None` for anything but a row-major contiguous view, which
    // is exactly the guard we want: a strided view must never enter a hot loop.
    let raw = scores.to_slice().ok_or(MatchError::NotContiguous)?;
    let (n_rows, n_cols) = (scores.nrows(), scores.ncols());
    let by_rows = opts.axis == MatchAxis::Rows;
    let (n_groups, n_scan) = if by_rows {
        (n_rows, n_cols)
    } else {
        (n_cols, n_rows)
    };
    if n_scan > u32::MAX as usize {
        return Err(MatchError::AxisTooLong { len: n_scan });
    }
    if let Some(s) = opts.skip {
        if s.len() != n_groups {
            return Err(MatchError::SkipLen {
                got: s.len(),
                want: n_groups,
            });
        }
    }
    Ok(Layout {
        raw,
        n_rows,
        n_cols,
        n_groups,
        n_scan,
        by_rows,
    })
}

/// Stripe width for the column kernel: wide enough that every stripe row-segment spans
/// whole cache lines, narrow enough that a stripe's working set stays in L1d, and small
/// enough to give every worker at least two stripes.
fn stripe_width<T>(n_cols: usize, threads: usize) -> usize {
    let per_line = (64 / std::mem::size_of::<T>()).max(1);
    let target = n_cols.div_ceil(threads.max(1) * 2).clamp(128, 8192);
    // Round up to a whole cache line, but never past the matrix.
    target.div_ceil(per_line).saturating_mul(per_line).min(n_cols).max(1)
}

/// A group is empty when its extremum never moved off the sentinel — i.e. it held no
/// comparable value at all. Encoding that as a `NaN` cutoff makes every subsequent
/// `passes()` false, so the group falls out naturally without a special case in the loops.
#[inline]
fn cutoff_from_extremum<T: Score, D: Dir>(m: T, crit: Criterion, distances: bool) -> T {
    match crit {
        Criterion::Threshold(t) => T::from_f64(t),
        Criterion::Percentage(p) => {
            if m == D::worst::<T>() {
                T::NAN // no valid cell in this group
            } else {
                T::perc_cutoff(m, p, distances)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Top-N
// ---------------------------------------------------------------------------

/// Insert `v` into a buffer already known to be sorted best-first, given that `v` beats the
/// current worst-of-the-best. Empty slots hold the `worst()` sentinel, so they are
/// displaced like any other loser and no fill counter is needed.
#[inline(always)]
fn insert<T: Score, D: Dir>(idx: &mut [i64], val: &mut [T], v: T, j: i64) {
    let mut k = val.len() - 1;
    while k > 0 && D::better(v, val[k - 1]) {
        val[k] = val[k - 1];
        idx[k] = idx[k - 1];
        k -= 1;
    }
    val[k] = v;
    idx[k] = j;
}

/// Slots that never took a value carry the `±inf` sentinel. Report them as `NaN` instead:
/// a `-inf` would read as a real score once it lands in a DataFrame, whereas `NaN` pairs
/// unambiguously with the `-1` index.
#[inline]
fn seal_topn<T: Score>(idx: &[i64], val: &mut [T]) {
    for (k, &i) in idx.iter().enumerate() {
        if i < 0 {
            val[k] = T::NAN;
        }
    }
}

/// Top-N per row. Each group is one contiguous row and the output row *is* the
/// accumulator, so this allocates nothing at all.
#[allow(clippy::too_many_arguments)]
fn topn_rows<T: Score, D: Dir>(
    l: &Layout<T>,
    n: usize,
    skip: Option<&[i64]>,
    indices: &mut [i64],
    values: &mut [T],
    cancel: Option<&AtomicBool>,
    bar: Option<&ProgressBar>,
) {
    let n_cols = l.n_cols;
    l.raw
        .par_chunks(n_cols)
        .zip(indices.par_chunks_mut(n))
        .zip(values.par_chunks_mut(n))
        .enumerate()
        .for_each(|(g, ((row, idx_out), val_out))| {
            if is_cancelled(cancel) {
                return;
            }
            idx_out.fill(-1);
            val_out.fill(D::worst::<T>());
            let mut worst = D::worst::<T>();
            let s = skip.map_or(-1, |sk| sk[g]);

            for (j, &v) in row.iter().enumerate() {
                if !D::better(v, worst) {
                    continue; // the common path: one comparison and out
                }
                if s >= 0 && j as i64 == s {
                    continue;
                }
                insert::<T, D>(idx_out, val_out, v, j as i64);
                worst = val_out[n - 1];
            }
            seal_topn(idx_out, val_out);
            if let Some(b) = bar {
                b.inc(1);
            }
        });
}

/// Top-N per column, by column stripes.
///
/// Each task owns a contiguous stripe of columns — and therefore a contiguous, disjoint
/// slice of the output — and walks every row, touching only `w` contiguous elements of it.
/// The matrix is read exactly once, at full cache-line utilisation. The per-column
/// worst-of-the-best lives in its own contiguous `worst` array rather than being read back
/// out of the strided N-buffers, so the reject test compares two dense streams.
#[allow(clippy::too_many_arguments)]
fn topn_cols<T: Score, D: Dir>(
    l: &Layout<T>,
    n: usize,
    skip: Option<&[i64]>,
    w: usize,
    indices: &mut [i64],
    values: &mut [T],
    cancel: Option<&AtomicBool>,
    bar: Option<&ProgressBar>,
) {
    let (raw, n_rows, n_cols) = (l.raw, l.n_rows, l.n_cols);
    indices
        .par_chunks_mut(w * n)
        .zip(values.par_chunks_mut(w * n))
        .enumerate()
        .for_each(|(si, (idx_out, val_out))| {
            if is_cancelled(cancel) {
                return;
            }
            let c0 = si * w;
            let width = idx_out.len() / n; // the last stripe may be narrower
            let c1 = c0 + width;

            idx_out.fill(-1);
            val_out.fill(D::worst::<T>());
            let mut worst = vec![D::worst::<T>(); width];
            let skip_seg = skip.map(|sk| &sk[c0..c1]);

            for r in 0..n_rows {
                if r % 4096 == 0 && is_cancelled(cancel) {
                    return;
                }
                let seg = &raw[r * n_cols + c0..r * n_cols + c1];
                let rr = r as i64;
                for (lc, &v) in seg.iter().enumerate() {
                    if !D::better(v, worst[lc]) {
                        continue;
                    }
                    if let Some(sk) = skip_seg {
                        if sk[lc] == rr {
                            continue;
                        }
                    }
                    let lo = lc * n;
                    insert::<T, D>(&mut idx_out[lo..lo + n], &mut val_out[lo..lo + n], v, rr);
                    worst[lc] = val_out[lo + n - 1];
                }
            }
            seal_topn(idx_out, val_out);
            if let Some(b) = bar {
                b.inc(width as u64);
            }
        });
}

/// Extract the `n` best matches for each group.
///
/// Returns row-major `(n_groups, n)` indices and values, best first. Indices are `-1` (and
/// values `NaN`) where a group held fewer than `n` valid cells.
pub fn top_matches<T: Score>(
    scores: ArrayView2<T>,
    n: usize,
    opts: MatchOpts,
) -> Result<TopN<T>, MatchError> {
    let l = layout(&scores, &opts)?;
    // Nothing to group by: an empty result, not an error. But a *non-empty* set of groups
    // with nothing to scan means `n` genuinely cannot be satisfied, so that does error.
    if l.n_groups == 0 {
        return Ok(TopN {
            n_groups: 0,
            n,
            indices: Vec::new(),
            values: Vec::new(),
        });
    }
    if n == 0 || n > l.n_scan {
        return Err(MatchError::InvalidN {
            n,
            n_scan: l.n_scan,
        });
    }

    let mut indices = vec![-1i64; l.n_groups * n];
    let mut values = vec![T::NAN; l.n_groups * n];

    let bar = opts
        .progress
        .then(|| make_bar("Matches", l.n_groups as u64));

    with_pool(opts.threads, || {
        let w = stripe_width::<T>(l.n_cols, rayon::current_num_threads());
        match (l.by_rows, opts.distances) {
            (true, false) => {
                topn_rows::<T, Sim>(&l, n, opts.skip, &mut indices, &mut values, opts.cancel, bar.as_ref())
            }
            (true, true) => {
                topn_rows::<T, Dist>(&l, n, opts.skip, &mut indices, &mut values, opts.cancel, bar.as_ref())
            }
            (false, false) => topn_cols::<T, Sim>(
                &l, n, opts.skip, w, &mut indices, &mut values, opts.cancel, bar.as_ref(),
            ),
            (false, true) => topn_cols::<T, Dist>(
                &l, n, opts.skip, w, &mut indices, &mut values, opts.cancel, bar.as_ref(),
            ),
        }
    });

    if let Some(b) = bar {
        b.finish_and_clear();
    }
    if is_cancelled(opts.cancel) {
        return Err(MatchError::Cancelled);
    }

    Ok(TopN {
        n_groups: l.n_groups,
        n,
        indices,
        values,
    })
}

// ---------------------------------------------------------------------------
// Ragged: cutoffs, counts, fill
// ---------------------------------------------------------------------------

/// Per-group cutoffs. For `Threshold` this is a constant, so no pass over the matrix is
/// needed; for `Percentage` it takes one pass to find each group's extremum first.
fn cutoffs<T: Score, D: Dir>(
    l: &Layout<T>,
    crit: Criterion,
    skip: Option<&[i64]>,
    w: usize,
    distances: bool,
    cancel: Option<&AtomicBool>,
) -> Vec<T> {
    if let Criterion::Threshold(t) = crit {
        return vec![T::from_f64(t); l.n_groups];
    }

    let mut ext = vec![D::worst::<T>(); l.n_groups];
    let (raw, n_rows, n_cols) = (l.raw, l.n_rows, l.n_cols);

    if l.by_rows {
        raw.par_chunks(n_cols)
            .zip(ext.par_iter_mut())
            .enumerate()
            .for_each(|(g, (row, m))| {
                if is_cancelled(cancel) {
                    return;
                }
                let s = skip.map_or(-1, |sk| sk[g]);
                let mut best = D::worst::<T>();
                for (j, &v) in row.iter().enumerate() {
                    if D::better(v, best) && !(s >= 0 && j as i64 == s) {
                        best = v;
                    }
                }
                *m = best;
            });
    } else {
        ext.par_chunks_mut(w).enumerate().for_each(|(si, seg_ext)| {
            if is_cancelled(cancel) {
                return;
            }
            let c0 = si * w;
            let c1 = c0 + seg_ext.len();
            let skip_seg = skip.map(|sk| &sk[c0..c1]);
            for r in 0..n_rows {
                if r % 4096 == 0 && is_cancelled(cancel) {
                    return;
                }
                let seg = &raw[r * n_cols + c0..r * n_cols + c1];
                let rr = r as i64;
                for (lc, &v) in seg.iter().enumerate() {
                    if D::better(v, seg_ext[lc]) {
                        if let Some(sk) = skip_seg {
                            if sk[lc] == rr {
                                continue;
                            }
                        }
                        seg_ext[lc] = v;
                    }
                }
            }
        });
    }

    ext.iter()
        .map(|&m| cutoff_from_extremum::<T, D>(m, crit, distances))
        .collect()
}

/// How many cells clear each group's cutoff.
fn counts<T: Score, D: Dir>(
    l: &Layout<T>,
    cut: &[T],
    skip: Option<&[i64]>,
    w: usize,
    cancel: Option<&AtomicBool>,
) -> Vec<i64> {
    let mut counts = vec![0i64; l.n_groups];
    let (raw, n_rows, n_cols) = (l.raw, l.n_rows, l.n_cols);

    if l.by_rows {
        raw.par_chunks(n_cols)
            .zip(counts.par_iter_mut())
            .enumerate()
            .for_each(|(g, (row, c))| {
                if is_cancelled(cancel) {
                    return;
                }
                let t = cut[g];
                let s = skip.map_or(-1, |sk| sk[g]);
                let mut k = 0i64;
                for (j, &v) in row.iter().enumerate() {
                    if D::passes(v, t) && !(s >= 0 && j as i64 == s) {
                        k += 1;
                    }
                }
                *c = k;
            });
    } else {
        counts
            .par_chunks_mut(w)
            .zip(cut.par_chunks(w))
            .enumerate()
            .for_each(|(si, (seg_cnt, seg_cut))| {
                if is_cancelled(cancel) {
                    return;
                }
                let c0 = si * w;
                let c1 = c0 + seg_cnt.len();
                let skip_seg = skip.map(|sk| &sk[c0..c1]);
                for r in 0..n_rows {
                    if r % 4096 == 0 && is_cancelled(cancel) {
                        return;
                    }
                    let seg = &raw[r * n_cols + c0..r * n_cols + c1];
                    let rr = r as i64;
                    for (lc, &v) in seg.iter().enumerate() {
                        if D::passes(v, seg_cut[lc]) {
                            if let Some(sk) = skip_seg {
                                if sk[lc] == rr {
                                    continue;
                                }
                            }
                            seg_cnt[lc] += 1;
                        }
                    }
                }
            });
    }
    counts
}

/// Carve a flat buffer into one disjoint sub-slice per group.
fn split_groups<'a, T>(buf: &'a mut [T], counts: &[i64]) -> Vec<&'a mut [T]> {
    let mut rest = buf;
    let mut out = Vec::with_capacity(counts.len());
    for &c in counts {
        let (head, tail) = rest.split_at_mut(c as usize);
        out.push(head);
        rest = tail;
    }
    out
}

/// Order a group best-first, ties by lower index. `NaN`s cannot be present (they never
/// clear a cutoff), so this is a genuine total order and the result does not depend on the
/// sort's stability or on the thread count.
fn sort_group<T: Score, D: Dir>(idx: &mut [u32], val: &mut [T]) {
    let mut pairs: Vec<(u32, T)> = idx.iter().copied().zip(val.iter().copied()).collect();
    pairs.sort_unstable_by(|a, b| {
        if D::better(a.1, b.1) {
            Ordering::Less
        } else if D::better(b.1, a.1) {
            Ordering::Greater
        } else {
            a.0.cmp(&b.0)
        }
    });
    for (k, (i, v)) in pairs.into_iter().enumerate() {
        idx[k] = i;
        val[k] = v;
    }
}

#[allow(clippy::too_many_arguments)]
fn fill<T: Score, D: Dir>(
    l: &Layout<T>,
    cut: &[T],
    skip: Option<&[i64]>,
    w: usize,
    counts: &[i64],
    indices: &mut [u32],
    values: &mut [T],
    cancel: Option<&AtomicBool>,
    bar: Option<&ProgressBar>,
) {
    let (raw, n_rows, n_cols) = (l.raw, l.n_rows, l.n_cols);
    let mut idx_groups = split_groups(indices, counts);
    let mut val_groups = split_groups(values, counts);

    if l.by_rows {
        raw.par_chunks(n_cols)
            .zip(idx_groups.par_iter_mut())
            .zip(val_groups.par_iter_mut())
            .enumerate()
            .for_each(|(g, ((row, idx_out), val_out))| {
                if is_cancelled(cancel) {
                    return;
                }
                let t = cut[g];
                let s = skip.map_or(-1, |sk| sk[g]);
                let mut k = 0usize;
                for (j, &v) in row.iter().enumerate() {
                    if D::passes(v, t) && !(s >= 0 && j as i64 == s) {
                        idx_out[k] = j as u32;
                        val_out[k] = v;
                        k += 1;
                    }
                }
                sort_group::<T, D>(idx_out, val_out);
                if let Some(b) = bar {
                    b.inc(1);
                }
            });
    } else {
        idx_groups
            .par_chunks_mut(w)
            .zip(val_groups.par_chunks_mut(w))
            .zip(cut.par_chunks(w))
            .enumerate()
            .for_each(|(si, ((idx_seg, val_seg), seg_cut))| {
                if is_cancelled(cancel) {
                    return;
                }
                let c0 = si * w;
                let width = idx_seg.len();
                let c1 = c0 + width;
                let skip_seg = skip.map(|sk| &sk[c0..c1]);
                let mut cursor = vec![0usize; width];

                for r in 0..n_rows {
                    if r % 4096 == 0 && is_cancelled(cancel) {
                        return;
                    }
                    let seg = &raw[r * n_cols + c0..r * n_cols + c1];
                    let rr = r as i64;
                    for (lc, &v) in seg.iter().enumerate() {
                        if D::passes(v, seg_cut[lc]) {
                            if let Some(sk) = skip_seg {
                                if sk[lc] == rr {
                                    continue;
                                }
                            }
                            let k = cursor[lc];
                            idx_seg[lc][k] = r as u32;
                            val_seg[lc][k] = v;
                            cursor[lc] = k + 1;
                        }
                    }
                }
                for lc in 0..width {
                    sort_group::<T, D>(idx_seg[lc], val_seg[lc]);
                }
                if let Some(b) = bar {
                    b.inc(width as u64);
                }
            });
    }
}

fn check_criterion(crit: Criterion) -> Result<(), MatchError> {
    match crit {
        Criterion::Threshold(t) if !t.is_finite() => Err(MatchError::InvalidCriterion(format!(
            "`threshold` must be finite, got {t}"
        ))),
        Criterion::Percentage(p) if !(0.0..=1.0).contains(&p) => Err(MatchError::InvalidCriterion(
            format!("`percentage` must be in [0, 1], got {p}"),
        )),
        _ => Ok(()),
    }
}

/// Count the matches each group *would* yield, without materialising them.
///
/// One pass over the matrix (two for [`Criterion::Percentage`]) and no output allocation —
/// the way to size a result on a matrix you cannot afford to guess wrong about.
pub fn count_matches<T: Score>(
    scores: ArrayView2<T>,
    crit: Criterion,
    opts: MatchOpts,
) -> Result<Vec<i64>, MatchError> {
    check_criterion(crit)?;
    let l = layout(&scores, &opts)?;
    if l.n_groups == 0 || l.n_scan == 0 {
        return Ok(vec![0i64; l.n_groups]);
    }

    let out = with_pool(opts.threads, || {
        let w = stripe_width::<T>(l.n_cols, rayon::current_num_threads());
        macro_rules! run {
            ($d:ty) => {{
                let cut = cutoffs::<T, $d>(&l, crit, opts.skip, w, opts.distances, opts.cancel);
                counts::<T, $d>(&l, &cut, opts.skip, w, opts.cancel)
            }};
        }
        if opts.distances {
            run!(Dist)
        } else {
            run!(Sim)
        }
    });

    if is_cancelled(opts.cancel) {
        return Err(MatchError::Cancelled);
    }
    Ok(out)
}

/// Extract every match clearing the cutoff, as a CSR-style ragged result.
///
/// Counts first, so the output is allocated exactly once at exactly the right size — and
/// so `max_matches` can refuse an over-broad cutoff *before* the allocation rather than
/// after the machine has swapped itself to death.
pub fn matches_above<T: Score>(
    scores: ArrayView2<T>,
    crit: Criterion,
    opts: MatchOpts,
) -> Result<Ragged<T>, MatchError> {
    check_criterion(crit)?;
    let l = layout(&scores, &opts)?;
    if l.n_groups == 0 || l.n_scan == 0 {
        return Ok(Ragged {
            n_groups: l.n_groups,
            offsets: vec![0i64; l.n_groups + 1],
            indices: Vec::new(),
            values: Vec::new(),
        });
    }

    let bar = opts
        .progress
        .then(|| make_bar("Matches", l.n_groups as u64));

    let result = with_pool(opts.threads, || -> Result<Ragged<T>, MatchError> {
        let w = stripe_width::<T>(l.n_cols, rayon::current_num_threads());

        macro_rules! run {
            ($d:ty) => {{
                let cut = cutoffs::<T, $d>(&l, crit, opts.skip, w, opts.distances, opts.cancel);
                let cnt = counts::<T, $d>(&l, &cut, opts.skip, w, opts.cancel);
                if is_cancelled(opts.cancel) {
                    return Err(MatchError::Cancelled);
                }

                // Prefix sum -> offsets. The total is known before a single output byte is
                // allocated, which is what makes the guard below meaningful.
                let mut offsets = Vec::with_capacity(l.n_groups + 1);
                let mut acc = 0i64;
                offsets.push(0);
                for &c in &cnt {
                    acc += c;
                    offsets.push(acc);
                }
                let total = acc as u64;
                if let Some(limit) = opts.max_matches {
                    if total > limit {
                        return Err(MatchError::TooManyMatches { total, limit });
                    }
                }

                let mut indices = vec![0u32; total as usize];
                let mut values = vec![T::ZERO; total as usize];
                fill::<T, $d>(
                    &l, &cut, opts.skip, w, &cnt, &mut indices, &mut values, opts.cancel,
                    bar.as_ref(),
                );
                Ok(Ragged {
                    n_groups: l.n_groups,
                    offsets,
                    indices,
                    values,
                })
            }};
        }

        if opts.distances {
            run!(Dist)
        } else {
            run!(Sim)
        }
    });

    if let Some(b) = bar {
        b.finish_and_clear();
    }
    if is_cancelled(opts.cancel) {
        return Err(MatchError::Cancelled);
    }
    result
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
// The reference checks walk a result and its expectation in lockstep by index;
// `enumerate()` would only obscure that they are the same index into both.
#[allow(clippy::needless_range_loop)]
mod tests {
    use super::*;
    use ndarray::Array2;

    /// Deterministic pseudo-random fill; no rand dependency needed for reproducibility.
    fn matrix(n_rows: usize, n_cols: usize, seed: u64) -> Array2<f32> {
        let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            // Spread over [-1, 1) so the negative-extremum paths get exercised.
            // `s >> 40` keeps 24 bits, so 2^24 is what normalises it to [0, 1).
            ((s >> 40) as f32 / 16_777_216.0) * 2.0 - 1.0
        };
        Array2::from_shape_fn((n_rows, n_cols), |_| next())
    }

    /// Reference: (value, index) pairs of one group, ordered best-first / lower-index-first.
    fn ranked(m: &Array2<f32>, g: usize, by_rows: bool, distances: bool, skip: i64) -> Vec<(usize, f32)> {
        let n_scan = if by_rows { m.ncols() } else { m.nrows() };
        let mut v: Vec<(usize, f32)> = (0..n_scan)
            .filter(|&j| skip < 0 || j as i64 != skip)
            .map(|j| (j, if by_rows { m[[g, j]] } else { m[[j, g]] }))
            .filter(|(_, x)| !x.is_nan())
            .collect();
        v.sort_by(|a, b| {
            let c = if distances {
                a.1.partial_cmp(&b.1).unwrap()
            } else {
                b.1.partial_cmp(&a.1).unwrap()
            };
            c.then(a.0.cmp(&b.0))
        });
        v
    }

    fn opts<'a>(axis: MatchAxis, distances: bool) -> MatchOpts<'a> {
        MatchOpts {
            axis,
            distances,
            ..Default::default()
        }
    }

    #[test]
    fn topn_matches_bruteforce() {
        let m = matrix(37, 53, 1);
        for &axis in &[MatchAxis::Rows, MatchAxis::Cols] {
            for &dist in &[false, true] {
                for &n in &[1usize, 3, 10] {
                    let got = top_matches(m.view(), n, opts(axis, dist)).unwrap();
                    let by_rows = axis == MatchAxis::Rows;
                    for g in 0..got.n_groups {
                        let want = ranked(&m, g, by_rows, dist, -1);
                        for k in 0..n {
                            assert_eq!(
                                got.indices[g * n + k], want[k].0 as i64,
                                "axis={axis:?} dist={dist} n={n} g={g} k={k}"
                            );
                            assert_eq!(got.values[g * n + k], want[k].1);
                        }
                    }
                }
            }
        }
    }

    /// The load-bearing test for the stripe kernel: per-column matches on `M` must equal
    /// per-row matches on a materialised, C-contiguous transpose of `M`.
    #[test]
    fn cols_equals_rows_of_materialized_transpose() {
        let m = matrix(41, 67, 2);
        // `to_owned()` on a transposed view keeps its F-order, which the contiguity guard
        // (rightly) rejects — `as_standard_layout` is what actually materialises it C-order.
        let t = m.t().as_standard_layout().to_owned();
        for &dist in &[false, true] {
            let a = top_matches(m.view(), 4, opts(MatchAxis::Cols, dist)).unwrap();
            let b = top_matches(t.view(), 4, opts(MatchAxis::Rows, dist)).unwrap();
            assert_eq!(a.indices, b.indices);
            assert_eq!(a.values, b.values);

            for crit in [Criterion::Threshold(0.3), Criterion::Percentage(0.1)] {
                let a = matches_above(m.view(), crit, opts(MatchAxis::Cols, dist)).unwrap();
                let b = matches_above(t.view(), crit, opts(MatchAxis::Rows, dist)).unwrap();
                assert_eq!(a.offsets, b.offsets, "{crit:?} dist={dist}");
                assert_eq!(a.indices, b.indices);
                assert_eq!(a.values, b.values);
            }
        }
    }

    /// Exercises the remainder stripe and the width clamp / cache-line rounding.
    #[test]
    fn stripe_partition_covers_all_columns() {
        for &n_cols in &[1usize, 2, 63, 64, 65, 127, 129, 1000, 4099] {
            let m = matrix(7, n_cols, n_cols as u64);
            for &threads in &[1usize, 3, 8] {
                let o = MatchOpts {
                    axis: MatchAxis::Cols,
                    threads: Some(threads),
                    ..Default::default()
                };
                let got = top_matches(m.view(), 2, o).unwrap();
                assert_eq!(got.n_groups, n_cols);
                for g in 0..n_cols {
                    let want = ranked(&m, g, false, false, -1);
                    assert_eq!(got.indices[g * 2], want[0].0 as i64, "n_cols={n_cols} t={threads} g={g}");
                    assert_eq!(got.indices[g * 2 + 1], want[1].0 as i64);
                }
            }
        }
    }

    /// The test navis' numba kernel fails: seeded with `-inf`, a distance search selects
    /// nothing and silently returns `-1` for every match.
    #[test]
    fn topn_distances_uses_min() {
        let m = Array2::from_shape_vec((1, 4), vec![5.0f32, 1.0, 9.0, 3.0]).unwrap();
        let got = top_matches(m.view(), 2, opts(MatchAxis::Rows, true)).unwrap();
        assert_eq!(got.indices, vec![1, 3]);
        assert_eq!(got.values, vec![1.0, 3.0]);
    }

    #[test]
    fn ties_prefer_lower_index() {
        let m = Array2::from_elem((1, 5), 0.5f32);
        let got = top_matches(m.view(), 3, opts(MatchAxis::Rows, false)).unwrap();
        assert_eq!(got.indices, vec![0, 1, 2]);

        let r = matches_above(m.view(), Criterion::Threshold(0.0), opts(MatchAxis::Rows, false)).unwrap();
        assert_eq!(r.indices, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn nan_is_never_a_match() {
        let nan = f32::NAN;
        let m = Array2::from_shape_vec((2, 3), vec![nan, nan, nan, 1.0, nan, 2.0]).unwrap();

        let got = top_matches(m.view(), 2, opts(MatchAxis::Rows, false)).unwrap();
        // All-NaN group: -1 / NaN.
        assert_eq!(&got.indices[0..2], &[-1, -1]);
        assert!(got.values[0].is_nan() && got.values[1].is_nan());
        // Partial-NaN group: the two real values, NaN never selected.
        assert_eq!(&got.indices[2..4], &[2, 0]);
        assert_eq!(&got.values[2..4], &[2.0, 1.0]);

        let r = matches_above(m.view(), Criterion::Threshold(0.0), opts(MatchAxis::Rows, false)).unwrap();
        assert_eq!(r.offsets, vec![0, 0, 2]); // first group empty
        assert_eq!(r.indices, vec![2, 0]);

        // Percentage on an all-NaN group must not compute inf - inf.
        let p = matches_above(m.view(), Criterion::Percentage(0.1), opts(MatchAxis::Rows, false)).unwrap();
        assert_eq!(p.offsets[1], 0);
    }

    /// `m - |m * p|`: the abs() is what widens the band in the right direction when the
    /// best score is negative.
    #[test]
    fn percentage_negative_extremum() {
        // Best is -0.5; p = 0.1 => keep >= -0.55.
        let m = Array2::from_shape_vec((1, 4), vec![-0.5f32, -0.54, -0.56, -2.0]).unwrap();
        let r = matches_above(m.view(), Criterion::Percentage(0.1), opts(MatchAxis::Rows, false)).unwrap();
        assert_eq!(r.indices, vec![0, 1]);
        assert_eq!(r.values, vec![-0.5, -0.54]);

        // A best of exactly 0.0 gives a zero-width band.
        let z = Array2::from_shape_vec((1, 3), vec![0.0f32, -0.1, 0.0]).unwrap();
        let r = matches_above(z.view(), Criterion::Percentage(0.5), opts(MatchAxis::Rows, false)).unwrap();
        assert_eq!(r.indices, vec![0, 2]);
    }

    #[test]
    fn threshold_boundary_is_inclusive() {
        let m = Array2::from_shape_vec((1, 3), vec![0.3f32, 0.2999999, 0.4]).unwrap();
        let r = matches_above(m.view(), Criterion::Threshold(0.3), opts(MatchAxis::Rows, false)).unwrap();
        assert_eq!(r.indices, vec![2, 0]);

        let d = matches_above(m.view(), Criterion::Threshold(0.3), opts(MatchAxis::Rows, true)).unwrap();
        assert_eq!(d.indices, vec![1, 0]);
    }

    #[test]
    fn skip_excludes_the_self_match() {
        let m = matrix(20, 20, 3);
        let diag: Vec<i64> = (0..20).collect();
        for &axis in &[MatchAxis::Rows, MatchAxis::Cols] {
            let o = MatchOpts {
                axis,
                skip: Some(&diag),
                ..Default::default()
            };
            let got = top_matches(m.view(), 3, o).unwrap();
            for g in 0..20 {
                let want = ranked(&m, g, axis == MatchAxis::Rows, false, g as i64);
                for k in 0..3 {
                    assert_eq!(got.indices[g * 3 + k], want[k].0 as i64);
                }
                assert!((0..3).all(|k| got.indices[g * 3 + k] != g as i64));
            }

            let r = matches_above(m.view(), Criterion::Threshold(-2.0), o).unwrap();
            let s = r.offsets[1] as usize;
            assert!(!r.indices[0..s].contains(&0u32));
        }
    }

    #[test]
    fn ragged_invariants_and_bruteforce() {
        let m = matrix(23, 31, 4);
        for &axis in &[MatchAxis::Rows, MatchAxis::Cols] {
            for &dist in &[false, true] {
                for crit in [Criterion::Threshold(0.2), Criterion::Percentage(0.25)] {
                    let r = matches_above(m.view(), crit, opts(axis, dist)).unwrap();
                    assert_eq!(r.offsets.len(), r.n_groups + 1);
                    assert_eq!(r.offsets[0], 0);
                    assert_eq!(*r.offsets.last().unwrap() as usize, r.indices.len());
                    assert_eq!(r.indices.len(), r.values.len());
                    assert!(r.offsets.windows(2).all(|w| w[0] <= w[1]));

                    let cnt = count_matches(m.view(), crit, opts(axis, dist)).unwrap();
                    for g in 0..r.n_groups {
                        let lo = r.offsets[g] as usize;
                        let hi = r.offsets[g + 1] as usize;
                        assert_eq!((hi - lo) as i64, cnt[g], "count_matches disagrees");

                        let want = ranked(&m, g, axis == MatchAxis::Rows, dist, -1);
                        let cut = match crit {
                            Criterion::Threshold(t) => t as f32,
                            Criterion::Percentage(p) => {
                                let best = want[0].1;
                                let band = (best * p as f32).abs();
                                if dist { best + band } else { best - band }
                            }
                        };
                        let want: Vec<_> = want
                            .into_iter()
                            .filter(|(_, v)| if dist { *v <= cut } else { *v >= cut })
                            .collect();
                        assert_eq!(hi - lo, want.len(), "{crit:?} {axis:?} dist={dist} g={g}");
                        for (k, (j, v)) in want.iter().enumerate() {
                            assert_eq!(r.indices[lo + k], *j as u32);
                            assert_eq!(r.values[lo + k], *v);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn threads_do_not_change_the_result() {
        let m = matrix(64, 200, 5);
        let mk = |t: Option<usize>| MatchOpts {
            axis: MatchAxis::Cols,
            threads: t,
            ..Default::default()
        };
        let a = top_matches(m.view(), 5, mk(Some(1))).unwrap();
        let b = top_matches(m.view(), 5, mk(Some(3))).unwrap();
        let c = top_matches(m.view(), 5, mk(None)).unwrap();
        assert_eq!(a.indices, b.indices);
        assert_eq!(a.indices, c.indices);
        assert_eq!(a.values, c.values);

        let x = matches_above(m.view(), Criterion::Threshold(0.5), mk(Some(1))).unwrap();
        let y = matches_above(m.view(), Criterion::Threshold(0.5), mk(Some(3))).unwrap();
        assert_eq!(x.offsets, y.offsets);
        assert_eq!(x.indices, y.indices);
    }

    #[test]
    fn large_n_still_correct() {
        // The sorted-insertion buffer is O(n^2 log) for absurd n; prove it stays correct.
        let m = matrix(3, 300, 6);
        let got = top_matches(m.view(), 256, opts(MatchAxis::Rows, false)).unwrap();
        for g in 0..3 {
            let want = ranked(&m, g, true, false, -1);
            for k in 0..256 {
                assert_eq!(got.indices[g * 256 + k], want[k].0 as i64);
            }
        }
    }

    #[test]
    fn max_matches_guard() {
        let m = matrix(10, 10, 7);
        let o = MatchOpts {
            max_matches: Some(5),
            ..Default::default()
        };
        let err = matches_above(m.view(), Criterion::Threshold(-1.5), o).unwrap_err();
        assert!(matches!(err, MatchError::TooManyMatches { limit: 5, .. }));
    }

    #[test]
    fn rejects_bad_input() {
        let m = matrix(5, 8, 8);
        assert!(matches!(
            top_matches(m.view(), 0, opts(MatchAxis::Rows, false)),
            Err(MatchError::InvalidN { .. })
        ));
        assert!(matches!(
            top_matches(m.view(), 9, opts(MatchAxis::Rows, false)),
            Err(MatchError::InvalidN { n: 9, n_scan: 8 })
        ));
        assert!(matches!(
            matches_above(m.view(), Criterion::Percentage(1.5), opts(MatchAxis::Rows, false)),
            Err(MatchError::InvalidCriterion(_))
        ));

        // A strided view must be refused, never silently copied.
        let strided = m.slice(ndarray::s![.., ..;2]);
        assert!(matches!(
            top_matches(strided, 1, opts(MatchAxis::Rows, false)),
            Err(MatchError::NotContiguous)
        ));

        let bad_skip = [0i64; 3];
        let o = MatchOpts {
            skip: Some(&bad_skip),
            ..Default::default()
        };
        assert!(matches!(
            top_matches(m.view(), 1, o),
            Err(MatchError::SkipLen { got: 3, want: 5 })
        ));
    }

    #[test]
    fn degenerate_shapes() {
        let e = Array2::<f32>::zeros((0, 0));
        assert_eq!(top_matches(e.view(), 1, opts(MatchAxis::Rows, false)).unwrap().n_groups, 0);
        let r = matches_above(e.view(), Criterion::Threshold(0.0), opts(MatchAxis::Rows, false)).unwrap();
        assert_eq!(r.offsets, vec![0]);

        let one = Array2::from_elem((1, 1), 1.0f32);
        let got = top_matches(one.view(), 1, opts(MatchAxis::Rows, false)).unwrap();
        assert_eq!(got.indices, vec![0]);

        // Zero-length scanned axis: no candidates, so `n` cannot be satisfied.
        let z = Array2::<f32>::zeros((5, 0));
        assert!(top_matches(z.view(), 1, opts(MatchAxis::Rows, false)).is_err());
        let r = matches_above(z.view(), Criterion::Threshold(0.0), opts(MatchAxis::Rows, false)).unwrap();
        assert_eq!(r.offsets, vec![0; 6]);
    }

    #[test]
    fn all_widths_agree() {
        let m = matrix(9, 17, 9);
        let m64 = m.mapv(|v| v as f64);
        let m16 = m.mapv(f16::from_f32);

        let a = top_matches(m.view(), 3, opts(MatchAxis::Rows, false)).unwrap();
        let b = top_matches(m64.view(), 3, opts(MatchAxis::Rows, false)).unwrap();
        assert_eq!(a.indices, b.indices);
        assert_eq!(a.values.iter().map(|&v| v as f64).collect::<Vec<_>>(), b.values);

        // f16 rounds, so only check it selects sane values (ties may legitimately shift).
        let c = top_matches(m16.view(), 3, opts(MatchAxis::Rows, false)).unwrap();
        assert_eq!(c.n_groups, 9);
        for g in 0..9 {
            let best = c.values[g * 3].to_f32();
            assert!(c.values[g * 3 + 1].to_f32() <= best);
        }
    }
}
