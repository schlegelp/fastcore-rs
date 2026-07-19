//! Hierarchical clustering of a score matrix.
//!
//! Turning an NBLAST all-by-all into a dendrogram is a three-step pipeline —
//! symmetrise, convert similarity to distance, condense to the upper triangle —
//! before the linkage itself. Done in numpy each step materialises another `n x n`
//! array, and at the sizes this crate targets (100–200k neurons a side) that is
//! where the memory goes, not the clustering.
//!
//! [`condense`] fuses all three into a **single pass** that reads the score matrix
//! and writes the condensed distance vector directly, so the only allocation is the
//! `n(n-1)/2` output itself. [`linkage`] then runs agglomerative clustering over
//! that vector **in place**, via [`kodama`] (a Rust port of Müllner's `fastcluster`
//! algorithms: nn-chain for the reducible methods, MST for single, and the generic
//! algorithm for centroid/median). [`linkage_from_scores`] is the two of them
//! together, which is what the bindings call.
//!
//! # Why this module exists in Rust
//!
//! For a 100k x 100k `f32` score matrix, the equivalent
//! `scipy`/`numpy` pipeline peaks north of 100 GB: `(a + a.T) / 2` alone allocates
//! two more full matrices, and `scipy.cluster.hierarchy.linkage` then **up-casts
//! its input to `float64` unconditionally** — so handing it `f32` to save memory
//! instead costs you a second, doubled copy of the condensed matrix, plus a `bool`
//! temporary the size of the input for its finiteness check. Here:
//!
//! - **`f32` stays `f32`.** Dissimilarities are computed and clustered at the width
//!   they arrive in. That halves the condensed matrix, which is the single largest
//!   live allocation in the pipeline.
//! - **No `n x n` temporaries.** Symmetrisation and the similarity→distance
//!   transform happen inside the condensing loop, on values already in registers.
//! - **The finiteness check is free.** It rides along on the fused pass instead of
//!   allocating an `n(n-1)/2` mask.
//!
//! # Conventions
//!
//! The returned linkage matrix `Z` matches SciPy's exactly: `(n-1, 4)`, `f64`, one
//! merge per row as `[cluster1, cluster2, dissimilarity, size]`, singleton clusters
//! labelled `0..n` and the cluster formed at step `i` labelled `n + i`, rows ordered
//! by increasing dissimilarity. It can be handed straight to
//! `scipy.cluster.hierarchy.fcluster` / `dendrogram`.
//!
//! # Cancellation
//!
//! Only [`condense`] is cancellable. `kodama` exposes no per-merge hook, so once
//! linkage proper starts it runs to completion — worth knowing, because at 100k it
//! is the part that takes minutes.

use kodama::Method as KMethod;
use ndarray::{Array2, ArrayView2, ArrayViewMut2};
use rayon::prelude::*;
use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use crate::nblast::{is_cancelled, with_pool};

// ---------------------------------------------------------------------------
// Element type
// ---------------------------------------------------------------------------

/// An element of a score / dissimilarity matrix: `f32` or `f64`.
///
/// A local trait for the same reason [`crate::matches::Score`] is one: it pins down
/// exactly the operations the kernels need. It is *not* extended to `f16` — unlike
/// `matches`, which only ever compares values, clustering accumulates thousands of
/// Lance-Williams updates per cluster and `f16` has nowhere near the mantissa for
/// that. `kodama::Float` is sealed to `f32`/`f64` in any case.
pub trait Dissim: kodama::Float + Send + Sync + 'static {
    const ZERO: Self;
    const ONE: Self;
    const TWO: Self;

    fn is_finite(self) -> bool;
}

impl Dissim for f32 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const TWO: Self = 2.0;

    #[inline(always)]
    fn is_finite(self) -> bool {
        f32::is_finite(self)
    }
}

impl Dissim for f64 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const TWO: Self = 2.0;

    #[inline(always)]
    fn is_finite(self) -> bool {
        f64::is_finite(self)
    }
}

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

/// How to combine the two directions of an asymmetric score matrix.
///
/// NBLAST is not symmetric: `M[i, j]` is neuron `i` queried against `j`, which is
/// not `M[j, i]`. These mirror `navis_fastcore.nblast._combine`, so a matrix
/// clustered here matches one symmetrised there.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Symmetry {
    /// Use the upper triangle as-is. The right choice when the matrix is already
    /// symmetric (e.g. `nblast_allbyall(symmetry="mean")` did the work), and the
    /// fastest, since it reads the buffer strictly sequentially.
    None,
    /// `(M[i,j] + M[j,i]) / 2`.
    Mean,
    /// `min(M[i,j], M[j,i])`.
    Min,
    /// `max(M[i,j], M[j,i])`.
    Max,
}

impl Symmetry {
    /// Parse a symmetry name, matching `navis_fastcore.nblast._combine`. Returns
    /// `None` for anything unrecognised.
    pub fn from_name(name: &str) -> Option<Symmetry> {
        Some(match name {
            "none" => Symmetry::None,
            "mean" => Symmetry::Mean,
            "min" => Symmetry::Min,
            "max" => Symmetry::Max,
            _ => return None,
        })
    }
}

/// How to turn a similarity into a distance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Transform {
    /// The values are already distances. Passed through untouched.
    AsIs,
    /// `1 - score`, the usual NBLAST convention.
    ///
    /// Assumes normalised scores, where a perfect self-match is `1.0`. Raw or
    /// alpha-weighted scores can exceed `1`, which yields a *negative* distance;
    /// that is left alone rather than clamped (clamping would silently change the
    /// data), but note [`Method::Ward`], [`Method::Centroid`] and [`Method::Median`]
    /// are not meaningful on negative distances.
    OneMinus,
}

impl Transform {
    /// Parse a transform name. Returns `None` for anything unrecognised.
    pub fn from_name(name: &str) -> Option<Transform> {
        Some(match name {
            "none" | "as_is" => Transform::AsIs,
            "one_minus" => Transform::OneMinus,
            _ => return None,
        })
    }
}

/// Agglomerative linkage method. One-to-one with SciPy's, and with `kodama`'s.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Method {
    Single,
    Complete,
    Average,
    Weighted,
    Ward,
    Centroid,
    Median,
}

impl Method {
    fn to_kodama(self) -> KMethod {
        match self {
            Method::Single => KMethod::Single,
            Method::Complete => KMethod::Complete,
            Method::Average => KMethod::Average,
            Method::Weighted => KMethod::Weighted,
            Method::Ward => KMethod::Ward,
            Method::Centroid => KMethod::Centroid,
            Method::Median => KMethod::Median,
        }
    }

    /// Parse a SciPy method name. Returns `None` for anything unrecognised.
    pub fn from_name(name: &str) -> Option<Method> {
        Some(match name {
            "single" => Method::Single,
            "complete" => Method::Complete,
            "average" => Method::Average,
            "weighted" => Method::Weighted,
            "ward" => Method::Ward,
            "centroid" => Method::Centroid,
            "median" => Method::Median,
            _ => return None,
        })
    }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LinkageError {
    /// The score matrix was not square.
    NotSquare { rows: usize, cols: usize },
    /// Fewer than two observations — there is nothing to merge.
    TooFewObservations { n: usize },
    /// The view was neither C- nor F-contiguous.
    NotContiguous,
    /// A condensed vector whose length is not `n(n-1)/2` for any `n`.
    BadCondensedLen { len: usize },
    /// The condensed length did not match the `n` the caller declared.
    CondensedLenMismatch { len: usize, n: usize, want: usize },
    /// A `NaN` or infinity reached the dissimilarity matrix. Reported with the row
    /// it came from, since chasing one down in a 100k matrix is otherwise painful.
    NonFinite { row: usize },
    Cancelled,
}

impl fmt::Display for LinkageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LinkageError::NotSquare { rows, cols } => write!(
                f,
                "`scores` must be square to be clustered, got ({rows}, {cols})"
            ),
            LinkageError::TooFewObservations { n } => write!(
                f,
                "clustering needs at least 2 observations, got {n}"
            ),
            LinkageError::NotContiguous => write!(
                f,
                "`scores` must be C- or F-contiguous (got a strided view); pass a \
                 contiguous array — a matrix this size will not be copied for you"
            ),
            LinkageError::BadCondensedLen { len } => write!(
                f,
                "a condensed distance matrix has length n(n-1)/2; {len} is not such \
                 a length for any n"
            ),
            LinkageError::CondensedLenMismatch { len, n, want } => write!(
                f,
                "condensed length {len} does not match n={n} (expected {want})"
            ),
            LinkageError::NonFinite { row } => write!(
                f,
                "the dissimilarity matrix contains non-finite values (NaN or inf), \
                 first seen in row {row}; linkage is undefined for these"
            ),
            LinkageError::Cancelled => write!(f, "cancelled"),
        }
    }
}

impl std::error::Error for LinkageError {}

// ---------------------------------------------------------------------------
// Condensed-matrix geometry
// ---------------------------------------------------------------------------

/// Length of the condensed form for `n` observations: `n(n-1)/2`.
#[inline]
pub fn condensed_len(n: usize) -> usize {
    n * (n - 1) / 2
}

/// Recover `n` from a condensed length, or `None` if it is not a triangular number.
///
/// Solves `n(n-1)/2 = len` by taking the real root and confirming it exactly, so a
/// length that is merely close to triangular is rejected rather than rounded into
/// silently wrong clustering.
pub fn observations_from_condensed(len: usize) -> Option<usize> {
    if len == 0 {
        return None;
    }
    // n = (1 + sqrt(1 + 8*len)) / 2, nudged for f64 rounding at large len, then
    // verified exactly in integer arithmetic.
    let approx = ((1.0 + (1.0 + 8.0 * len as f64).sqrt()) / 2.0).round() as usize;
    for cand in [approx.wrapping_sub(1), approx, approx + 1] {
        if cand >= 2 && condensed_len(cand) == len {
            return Some(cand);
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Fused condensing
// ---------------------------------------------------------------------------

/// Carve a condensed buffer into one sub-slice per row `i`, of length `n - i - 1`.
///
/// This is what lets the fused pass be a safe `rayon` parallel iterator: every row
/// owns a disjoint, contiguous run of the output, so there are no overlapping
/// mutable borrows and no atomics on the hot path.
fn carve_rows<T>(cond: &mut [T], n: usize) -> Vec<&mut [T]> {
    let mut rest = cond;
    let mut rows = Vec::with_capacity(n.saturating_sub(1));
    for i in 0..n.saturating_sub(1) {
        let (head, tail) = rest.split_at_mut(n - i - 1);
        rows.push(head);
        rest = tail;
    }
    rows
}

/// The fused kernel, monomorphised over how the two directions are combined.
fn condense_kernel<T, F>(
    flat: &[T],
    n: usize,
    transform: Transform,
    combine: F,
    cancel: Option<&AtomicBool>,
) -> Result<Vec<T>, LinkageError>
where
    T: Dissim,
    F: Fn(T, T) -> T + Sync + Send,
{
    // Floats have an all-zero bit pattern, so this is `calloc`: pages arrive zeroed
    // and lazily from the OS rather than being touched here. At 100k that is a 20 GB
    // allocation we would otherwise write twice.
    let mut cond = vec![T::ZERO; condensed_len(n)];
    let bad_row = AtomicUsize::new(usize::MAX);

    {
        let rows = carve_rows(&mut cond, n);
        rows.into_par_iter().enumerate().for_each(|(i, row)| {
            if is_cancelled(cancel) || bad_row.load(Ordering::Relaxed) != usize::MAX {
                return;
            }
            let ri = &flat[i * n..(i + 1) * n];
            let mut saw_bad = false;
            for (off, slot) in row.iter_mut().enumerate() {
                let j = i + 1 + off;
                // SAFETY-free indexing: j < n by construction, and the transposed
                // read is bounds-checked against the full buffer below.
                let d = combine(ri[j], flat[j * n + i]);
                let d = match transform {
                    Transform::AsIs => d,
                    Transform::OneMinus => T::ONE - d,
                };
                saw_bad |= !d.is_finite();
                *slot = d;
            }
            if saw_bad {
                bad_row.fetch_min(i, Ordering::Relaxed);
            }
        });
    }

    if is_cancelled(cancel) {
        return Err(LinkageError::Cancelled);
    }
    let bad = bad_row.load(Ordering::Relaxed);
    if bad != usize::MAX {
        return Err(LinkageError::NonFinite { row: bad });
    }
    Ok(cond)
}

/// Fuse symmetrisation, the similarity→distance transform and condensing into one
/// pass over `scores`.
///
/// Returns the upper triangle in row-major (SciPy `squareform`) order: `(0,1),
/// (0,2), … (0,n-1), (1,2), …`. The diagonal is never read, so a matrix whose
/// diagonal holds self-scores rather than zeros needs no fixing up first.
///
/// An F-contiguous matrix is handled without copying: transposing the view makes it
/// C-contiguous, and since [`Symmetry::Mean`], [`Min`](Symmetry::Min) and
/// [`Max`](Symmetry::Max) are symmetric in their arguments the result is unchanged.
/// [`Symmetry::None`] is the exception — there the transpose *is* a different
/// matrix, so the two reads are swapped to compensate.
pub fn condense<T: Dissim>(
    scores: ArrayView2<T>,
    symmetry: Symmetry,
    transform: Transform,
    threads: Option<usize>,
    cancel: Option<&AtomicBool>,
) -> Result<Vec<T>, LinkageError> {
    let (rows, cols) = (scores.nrows(), scores.ncols());
    if rows != cols {
        return Err(LinkageError::NotSquare { rows, cols });
    }
    let n = rows;
    if n < 2 {
        return Err(LinkageError::TooFewObservations { n });
    }

    // Normalise layout: work from a C-contiguous buffer, or refuse.
    let (view, transposed) = if scores.is_standard_layout() {
        (scores, false)
    } else if scores.t().is_standard_layout() {
        (scores.reversed_axes(), true)
    } else {
        return Err(LinkageError::NotContiguous);
    };
    let flat = view.as_slice().ok_or(LinkageError::NotContiguous)?;

    with_pool(threads, || match (symmetry, transposed) {
        // `a` is the sequential read, `b` the strided one. For `None` we want only
        // one of them, and which one depends on whether we transposed.
        (Symmetry::None, false) => condense_kernel(flat, n, transform, |a, _b| a, cancel),
        (Symmetry::None, true) => condense_kernel(flat, n, transform, |_a, b| b, cancel),
        (Symmetry::Mean, _) => {
            condense_kernel(flat, n, transform, |a, b| (a + b) / T::TWO, cancel)
        }
        (Symmetry::Min, _) => {
            condense_kernel(flat, n, transform, |a, b| if b < a { b } else { a }, cancel)
        }
        (Symmetry::Max, _) => {
            condense_kernel(flat, n, transform, |a, b| if b > a { b } else { a }, cancel)
        }
    })
}

// ---------------------------------------------------------------------------
// In-place symmetrising
// ---------------------------------------------------------------------------

/// Shared mutable view over the score matrix.
///
/// The symmetrise writes `M[i, j]` and `M[j, i]` for every `i < j`, so each element
/// is written exactly once — but the writes for a given `i` land in *other* rows,
/// which rayon cannot prove disjoint. This wrapper asserts what the tiling
/// guarantees: distinct tile pairs touch disjoint elements.
struct SharedMat<T>(*mut T);

unsafe impl<T: Send> Send for SharedMat<T> {}
unsafe impl<T: Send> Sync for SharedMat<T> {}

impl<T> SharedMat<T> {
    /// Taken through a method, not the field, so closures capture `&SharedMat`
    /// (which is `Sync`) rather than the bare pointer (which is not).
    #[inline(always)]
    fn ptr(&self) -> *mut T {
        self.0
    }
}

/// Tile edge, in elements. Two `TILE x TILE` f32 tiles are 128 KB together, so a
/// tile pair and its transpose stay resident in L2 for the whole of their work.
const TILE: usize = 128;

/// Make a square score matrix symmetric, **in place and without allocating**.
///
/// This is the square-matrix counterpart of the symmetrisation [`condense`] does
/// inline. It exists because the numpy spelling cannot avoid a copy: `(M + M.T) / 2`
/// builds two full `n x n` temporaries, and even `np.add(M, M.T, out=M)` still costs
/// one, because numpy detects that the output overlaps `M.T` and defensively copies
/// the input. At 100k neurons that temporary is 40 GB.
///
/// The pass is **tiled**. The naive row-wise loop writes `M[j, i]` down a column,
/// touching a fresh cache line per element; walking `TILE x TILE` blocks against
/// their transposed partners instead keeps both sides in L2.
///
/// [`Symmetry::None`] mirrors the upper triangle onto the lower, matching what
/// [`condense`] means by `None` (read the upper triangle).
pub fn symmetrize<T: Dissim>(
    scores: ArrayViewMut2<T>,
    symmetry: Symmetry,
    threads: Option<usize>,
    cancel: Option<&AtomicBool>,
) -> Result<(), LinkageError> {
    let (rows, cols) = (scores.nrows(), scores.ncols());
    if rows != cols {
        return Err(LinkageError::NotSquare { rows, cols });
    }
    let n = rows;
    if n < 2 {
        return Err(LinkageError::TooFewObservations { n });
    }

    // As in `condense`, an F-contiguous matrix is its own transpose viewed C-order.
    // Symmetrising is invariant under that for mean/min/max; `None` swaps which
    // triangle is the source, so the reads are swapped to compensate.
    let (mut view, transposed) = if scores.is_standard_layout() {
        (scores, false)
    } else if scores.t().is_standard_layout() {
        (scores.reversed_axes(), true)
    } else {
        return Err(LinkageError::NotContiguous);
    };
    let flat = view.as_slice_mut().ok_or(LinkageError::NotContiguous)?;
    let shared = SharedMat(flat.as_mut_ptr());

    // Tile pairs (bi, bj) with bi <= bj. Each covers a disjoint set of (i, j) pairs
    // with i < j, and therefore a disjoint set of elements.
    let nb = n.div_ceil(TILE);
    let mut tiles = Vec::with_capacity(nb * (nb + 1) / 2);
    for bi in 0..nb {
        for bj in bi..nb {
            tiles.push((bi, bj));
        }
    }

    fn run<T: Dissim, F>(
        tiles: &[(usize, usize)],
        shared: &SharedMat<T>,
        n: usize,
        combine: F,
        cancel: Option<&AtomicBool>,
    ) where
        F: Fn(T, T) -> T + Sync + Send,
    {
        tiles.par_iter().for_each(|&(bi, bj)| {
            if is_cancelled(cancel) {
                return;
            }
            let p = shared.ptr();
            let i1 = ((bi + 1) * TILE).min(n);
            let j1 = ((bj + 1) * TILE).min(n);
            for i in bi * TILE..i1 {
                // On the diagonal tile only the strict upper triangle is ours.
                let j0 = if bi == bj { i + 1 } else { bj * TILE };
                for j in j0..j1 {
                    // SAFETY: i < j < n, so both offsets are in bounds; and tile
                    // pairs partition the strict upper triangle, so no other
                    // thread writes either element.
                    unsafe {
                        let v = combine(*p.add(i * n + j), *p.add(j * n + i));
                        *p.add(i * n + j) = v;
                        *p.add(j * n + i) = v;
                    }
                }
            }
        });
    }

    with_pool(threads, || match (symmetry, transposed) {
        (Symmetry::None, false) => run(&tiles, &shared, n, |a, _b| a, cancel),
        (Symmetry::None, true) => run(&tiles, &shared, n, |_a, b| b, cancel),
        (Symmetry::Mean, _) => run(&tiles, &shared, n, |a, b| (a + b) / T::TWO, cancel),
        (Symmetry::Min, _) => run(&tiles, &shared, n, |a, b| if b < a { b } else { a }, cancel),
        (Symmetry::Max, _) => run(&tiles, &shared, n, |a, b| if b > a { b } else { a }, cancel),
    });

    if is_cancelled(cancel) {
        return Err(LinkageError::Cancelled);
    }
    Ok(())
}

/// Check a caller-supplied condensed matrix for `NaN`/infinity.
///
/// Separate from [`condense`], which folds the same check into its fused pass for
/// free. Unlike `numpy`'s `np.all(np.isfinite(y))` this allocates nothing — at 100k
/// that mask would be a 5 GB temporary.
pub fn check_finite<T: Dissim>(condensed: &[T], threads: Option<usize>) -> Result<(), LinkageError> {
    let bad = with_pool(threads, || {
        condensed
            .par_iter()
            .position_any(|v| !v.is_finite())
            .is_some()
    });
    if bad {
        // The caller handed us a flat vector, so there is no meaningful row to name.
        Err(LinkageError::NonFinite { row: 0 })
    } else {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Linkage
// ---------------------------------------------------------------------------

/// Agglomerative clustering over a condensed distance matrix, **in place**.
///
/// `condensed` is used as scratch and is left in an arbitrary state — that is what
/// makes this `O(n)` in auxiliary memory on top of the matrix itself, rather than
/// taking the defensive copy `scipy` does. Callers that need their input afterwards
/// must clone it first, and at these sizes should think hard about whether they do.
///
/// Returns the SciPy-compatible `(n-1, 4)` linkage matrix; see the module docs.
pub fn linkage<T: Dissim>(
    condensed: &mut [T],
    n: usize,
    method: Method,
) -> Result<Array2<f64>, LinkageError> {
    if n < 2 {
        return Err(LinkageError::TooFewObservations { n });
    }
    let want = condensed_len(n);
    if condensed.len() != want {
        return Err(LinkageError::CondensedLenMismatch {
            len: condensed.len(),
            n,
            want,
        });
    }

    let dend = kodama::linkage(condensed, n, method.to_kodama());

    let mut z = Array2::<f64>::zeros((n - 1, 4));
    for (k, step) in dend.steps().iter().enumerate() {
        z[[k, 0]] = step.cluster1 as f64;
        z[[k, 1]] = step.cluster2 as f64;
        z[[k, 2]] = step.dissimilarity.to_f64();
        z[[k, 3]] = step.size as f64;
    }
    Ok(z)
}

/// The order in which to place the leaves so a dendrogram draws without crossing
/// branches — the equivalent of SciPy's `leaves_list`.
///
/// A depth-first walk of the merge tree from the root, emitting each singleton as
/// it is reached and descending into `cluster1` before `cluster2`. Any consumer
/// that draws the tree must use the *same* child order as the linkage matrix it
/// came from, which is why this reads `z` rather than taking a pre-built tree.
///
/// The walk is iterative: at 200k observations a recursive one would need 200k
/// stack frames in the degenerate (chained) case.
pub fn leaf_order(z: &Array2<f64>, n: usize) -> Vec<usize> {
    let mut out = Vec::with_capacity(n);
    if n == 0 {
        return out;
    }
    if n == 1 {
        out.push(0);
        return out;
    }
    // Cluster ids follow SciPy: 0..n are singletons, n + i is formed at step i, so
    // the root — the last merge — is `n + (n - 2)`.
    let mut stack = vec![2 * n - 2];
    while let Some(c) = stack.pop() {
        if c < n {
            out.push(c);
        } else {
            let step = c - n;
            // Pushed right-first so the left child pops first.
            stack.push(z[[step, 1]] as usize);
            stack.push(z[[step, 0]] as usize);
        }
    }
    out
}

/// [`condense`] followed by [`linkage`] — the whole pipeline, with the condensed
/// matrix never escaping.
///
/// This is the memory-minimal entry point: peak live memory is the caller's score
/// matrix plus one `n(n-1)/2` buffer, and the buffer is freed on return.
pub fn linkage_from_scores<T: Dissim>(
    scores: ArrayView2<T>,
    method: Method,
    symmetry: Symmetry,
    transform: Transform,
    threads: Option<usize>,
    cancel: Option<&AtomicBool>,
) -> Result<Array2<f64>, LinkageError> {
    let n = scores.nrows();
    let mut cond = condense(scores, symmetry, transform, threads, cancel)?;
    if is_cancelled(cancel) {
        return Err(LinkageError::Cancelled);
    }
    linkage(&mut cond, n, method)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    /// A small, deliberately asymmetric score matrix.
    fn scores4() -> Array2<f64> {
        arr2(&[
            [1.0, 0.9, 0.2, 0.1],
            [0.7, 1.0, 0.3, 0.15],
            [0.25, 0.35, 1.0, 0.8],
            [0.05, 0.1, 0.6, 1.0],
        ])
    }

    #[test]
    fn condensed_len_roundtrips() {
        for n in 2..200usize {
            assert_eq!(observations_from_condensed(condensed_len(n)), Some(n));
        }
        // Large n: the f64 sqrt must not drift.
        for n in [100_000usize, 199_999, 200_000] {
            assert_eq!(observations_from_condensed(condensed_len(n)), Some(n));
        }
        // Non-triangular lengths are rejected, not rounded.
        assert_eq!(observations_from_condensed(0), None);
        assert_eq!(observations_from_condensed(2), None);
        assert_eq!(observations_from_condensed(4), None);
        assert_eq!(observations_from_condensed(7), None);
    }

    #[test]
    fn condense_mean_matches_by_hand() {
        let s = scores4();
        let got = condense(s.view(), Symmetry::Mean, Transform::OneMinus, None, None).unwrap();
        // Upper triangle of 1 - (M + M.T)/2, row-major.
        let want = vec![
            1.0 - (0.9 + 0.7) / 2.0,   // (0,1)
            1.0 - (0.2 + 0.25) / 2.0,  // (0,2)
            1.0 - (0.1 + 0.05) / 2.0,  // (0,3)
            1.0 - (0.3 + 0.35) / 2.0,  // (1,2)
            1.0 - (0.15 + 0.1) / 2.0,  // (1,3)
            1.0 - (0.8 + 0.6) / 2.0,   // (2,3)
        ];
        assert_eq!(got.len(), want.len());
        for (g, w) in got.iter().zip(&want) {
            assert!((g - w).abs() < 1e-12, "got {g}, want {w}");
        }
    }

    #[test]
    fn condense_none_takes_upper_triangle_untouched() {
        let s = scores4();
        let got = condense(s.view(), Symmetry::None, Transform::AsIs, None, None).unwrap();
        assert_eq!(got, vec![0.9, 0.2, 0.1, 0.3, 0.15, 0.8]);
    }

    #[test]
    fn condense_min_max() {
        let s = scores4();
        let lo = condense(s.view(), Symmetry::Min, Transform::AsIs, None, None).unwrap();
        let hi = condense(s.view(), Symmetry::Max, Transform::AsIs, None, None).unwrap();
        assert_eq!(lo, vec![0.7, 0.2, 0.05, 0.3, 0.1, 0.6]);
        assert_eq!(hi, vec![0.9, 0.25, 0.1, 0.35, 0.15, 0.8]);
    }

    /// An F-contiguous matrix must give the same answer as its C-contiguous twin,
    /// without being copied — including for `Symmetry::None`, where the transpose
    /// is genuinely a different matrix and the reads have to swap.
    #[test]
    fn condense_handles_f_order() {
        let s = scores4();
        let f = s.clone().reversed_axes().as_standard_layout().reversed_axes().to_owned();
        assert!(!f.is_standard_layout());
        assert_eq!(f, s);

        for sym in [Symmetry::None, Symmetry::Mean, Symmetry::Min, Symmetry::Max] {
            let c = condense(s.view(), sym, Transform::OneMinus, None, None).unwrap();
            let g = condense(f.view(), sym, Transform::OneMinus, None, None).unwrap();
            assert_eq!(c, g, "F-order disagreed for {sym:?}");
        }
    }

    #[test]
    fn condense_rejects_bad_input() {
        let rect = Array2::<f64>::zeros((3, 4));
        assert!(matches!(
            condense(rect.view(), Symmetry::Mean, Transform::AsIs, None, None),
            Err(LinkageError::NotSquare { rows: 3, cols: 4 })
        ));

        let tiny = Array2::<f64>::zeros((1, 1));
        assert!(matches!(
            condense(tiny.view(), Symmetry::Mean, Transform::AsIs, None, None),
            Err(LinkageError::TooFewObservations { n: 1 })
        ));

        let mut nan = scores4();
        nan[[2, 3]] = f64::NAN;
        assert!(matches!(
            condense(nan.view(), Symmetry::Mean, Transform::AsIs, None, None),
            Err(LinkageError::NonFinite { row: 2 })
        ));
    }

    /// The diagonal is never read, so self-scores on it must not leak into the
    /// result — a matrix with `1.0` down the diagonal is the normal NBLAST shape.
    #[test]
    fn condense_ignores_the_diagonal() {
        let mut s = scores4();
        let base = condense(s.view(), Symmetry::Mean, Transform::OneMinus, None, None).unwrap();
        for i in 0..4 {
            s[[i, i]] = f64::NAN;
        }
        let with_nan_diag =
            condense(s.view(), Symmetry::Mean, Transform::OneMinus, None, None).unwrap();
        assert_eq!(base, with_nan_diag);
    }

    /// Four points in two obvious pairs. After `1 - mean` the within-pair distances
    /// are d(0,1)=0.2 and d(2,3)=0.3, every across-pair distance is >= 0.675, so the
    /// pairs must form first — {0,1} then {2,3} — and the last merge joins them.
    #[test]
    fn linkage_recovers_obvious_structure() {
        let s = scores4();
        for method in [
            Method::Single,
            Method::Complete,
            Method::Average,
            Method::Weighted,
            Method::Ward,
        ] {
            let z = linkage_from_scores(
                s.view(),
                method,
                Symmetry::Mean,
                Transform::OneMinus,
                None,
                None,
            )
            .unwrap();
            assert_eq!(z.shape(), &[3, 4]);

            let pair = |r: usize| {
                let (a, b) = (z[[r, 0]] as usize, z[[r, 1]] as usize);
                (a.min(b), a.max(b))
            };
            assert_eq!(pair(0), (0, 1), "{method:?}: closest pair merges first");
            assert_eq!(pair(1), (2, 3), "{method:?}");
            assert_eq!(pair(2), (4, 5), "{method:?}: roots joined last");
            assert_eq!(z[[2, 3]], 4.0, "{method:?}: root holds every observation");

            // Heights are non-decreasing, which SciPy consumers rely on.
            assert!(z[[0, 2]] <= z[[1, 2]] && z[[1, 2]] <= z[[2, 2]], "{method:?}");
        }
    }

    /// f32 and f64 must agree to f32 precision — the whole memory argument rests on
    /// f32 being a real option.
    #[test]
    fn f32_tracks_f64() {
        let s = scores4();
        let s32 = s.mapv(|v| v as f32);
        for method in [Method::Single, Method::Complete, Method::Average, Method::Ward] {
            let a = linkage_from_scores(
                s.view(),
                method,
                Symmetry::Mean,
                Transform::OneMinus,
                None,
                None,
            )
            .unwrap();
            let b = linkage_from_scores(
                s32.view(),
                method,
                Symmetry::Mean,
                Transform::OneMinus,
                None,
                None,
            )
            .unwrap();
            for r in 0..3 {
                assert_eq!(a[[r, 0]], b[[r, 0]], "{method:?} row {r}");
                assert_eq!(a[[r, 1]], b[[r, 1]], "{method:?} row {r}");
                assert!((a[[r, 2]] - b[[r, 2]]).abs() < 1e-6, "{method:?} row {r}");
            }
        }
    }

    #[test]
    fn linkage_validates_length() {
        let mut cond = vec![0.5f64; 5];
        assert!(matches!(
            linkage(&mut cond, 4, Method::Average),
            Err(LinkageError::CondensedLenMismatch { len: 5, n: 4, want: 6 })
        ));
    }

    #[test]
    fn method_names_round_trip() {
        for (name, m) in [
            ("single", Method::Single),
            ("complete", Method::Complete),
            ("average", Method::Average),
            ("weighted", Method::Weighted),
            ("ward", Method::Ward),
            ("centroid", Method::Centroid),
            ("median", Method::Median),
        ] {
            assert_eq!(Method::from_name(name), Some(m));
        }
        assert_eq!(Method::from_name("Ward"), None);
        assert_eq!(Method::from_name("nonesuch"), None);

        assert_eq!(Symmetry::from_name("mean"), Some(Symmetry::Mean));
        assert_eq!(Symmetry::from_name("none"), Some(Symmetry::None));
        assert_eq!(Symmetry::from_name("Mean"), None);

        assert_eq!(Transform::from_name("one_minus"), Some(Transform::OneMinus));
        assert_eq!(Transform::from_name("as_is"), Some(Transform::AsIs));
        assert_eq!(Transform::from_name("none"), Some(Transform::AsIs));
        assert_eq!(Transform::from_name("1-x"), None);
    }

    /// The in-place symmetrise must equal the numpy expression it replaces.
    #[test]
    fn symmetrize_matches_the_naive_expression() {
        let s = scores4();
        for (sym, f) in [
            (Symmetry::Mean, (|a: f64, b: f64| (a + b) / 2.0) as fn(f64, f64) -> f64),
            (Symmetry::Min, |a, b| a.min(b)),
            (Symmetry::Max, |a, b| a.max(b)),
        ] {
            let mut got = s.clone();
            symmetrize(got.view_mut(), sym, None, None).unwrap();

            let mut want = s.clone();
            for i in 0..4 {
                for j in 0..4 {
                    if i != j {
                        want[[i, j]] = f(s[[i, j]], s[[j, i]]);
                    }
                }
            }
            assert_eq!(got, want, "{sym:?}");
            // And the result really is symmetric.
            for i in 0..4 {
                for j in 0..4 {
                    assert_eq!(got[[i, j]], got[[j, i]]);
                }
            }
        }
    }

    #[test]
    fn symmetrize_none_mirrors_upper_triangle() {
        let s = scores4();
        let mut got = s.clone();
        symmetrize(got.view_mut(), Symmetry::None, None, None).unwrap();
        for i in 0..4 {
            for j in (i + 1)..4 {
                assert_eq!(got[[i, j]], s[[i, j]]);
                assert_eq!(got[[j, i]], s[[i, j]]);
            }
        }
    }

    /// The diagonal is not ours to touch — NBLAST puts self-scores there.
    #[test]
    fn symmetrize_leaves_the_diagonal_alone() {
        let mut s = scores4();
        for i in 0..4 {
            s[[i, i]] = 42.0 + i as f64;
        }
        let mut got = s.clone();
        symmetrize(got.view_mut(), Symmetry::Mean, None, None).unwrap();
        for i in 0..4 {
            assert_eq!(got[[i, i]], 42.0 + i as f64);
        }
    }

    /// Symmetrising then condensing with `None` must equal condensing directly —
    /// the two share the same combine, so this pins them to each other.
    #[test]
    fn symmetrize_then_condense_matches_fused_condense() {
        let s = scores4();
        for sym in [Symmetry::Mean, Symmetry::Min, Symmetry::Max] {
            let fused = condense(s.view(), sym, Transform::OneMinus, None, None).unwrap();

            let mut m = s.clone();
            symmetrize(m.view_mut(), sym, None, None).unwrap();
            let staged = condense(m.view(), Symmetry::None, Transform::OneMinus, None, None)
                .unwrap();

            assert_eq!(fused, staged, "{sym:?}");
        }
    }

    /// Big enough to span many tiles, with a size that is not a tile multiple so the
    /// ragged edge tiles are exercised too.
    #[test]
    fn symmetrize_is_correct_across_tile_boundaries() {
        let n = TILE * 3 + 37;
        let mut rng_state = 0x2545F491_4F6CDD1Du64;
        let mut next = || {
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            (rng_state >> 40) as f64 / 16_777_216.0
        };
        let s = Array2::from_shape_fn((n, n), |_| next());

        let mut got = s.clone();
        symmetrize(got.view_mut(), Symmetry::Mean, None, None).unwrap();

        for i in 0..n {
            assert_eq!(got[[i, i]], s[[i, i]], "diagonal moved at {i}");
            for j in (i + 1)..n {
                let want = (s[[i, j]] + s[[j, i]]) / 2.0;
                assert!((got[[i, j]] - want).abs() < 1e-15, "({i},{j})");
                assert_eq!(got[[i, j]], got[[j, i]], "asymmetric at ({i},{j})");
            }
        }
    }

    #[test]
    fn symmetrize_thread_count_does_not_change_the_answer() {
        let n = TILE * 2 + 5;
        let s = Array2::from_shape_fn((n, n), |(i, j)| ((i * 7 + j * 13) % 101) as f32);
        let mut a = s.clone();
        let mut b = s.clone();
        symmetrize(a.view_mut(), Symmetry::Mean, Some(1), None).unwrap();
        symmetrize(b.view_mut(), Symmetry::Mean, Some(8), None).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn symmetrize_handles_f_order() {
        let s = scores4();
        for sym in [Symmetry::None, Symmetry::Mean, Symmetry::Min, Symmetry::Max] {
            let mut c = s.clone();
            symmetrize(c.view_mut(), sym, None, None).unwrap();

            let mut f = s.clone().reversed_axes().as_standard_layout().reversed_axes().to_owned();
            assert!(!f.is_standard_layout());
            symmetrize(f.view_mut(), sym, None, None).unwrap();

            assert_eq!(c, f, "F-order disagreed for {sym:?}");
        }
    }

    #[test]
    fn symmetrize_rejects_bad_input() {
        let mut rect = Array2::<f64>::zeros((3, 4));
        assert!(matches!(
            symmetrize(rect.view_mut(), Symmetry::Mean, None, None),
            Err(LinkageError::NotSquare { rows: 3, cols: 4 })
        ));
        let mut tiny = Array2::<f64>::zeros((1, 1));
        assert!(matches!(
            symmetrize(tiny.view_mut(), Symmetry::Mean, None, None),
            Err(LinkageError::TooFewObservations { n: 1 })
        ));
    }

    #[test]
    fn leaf_order_is_a_permutation_and_groups_the_pairs() {
        let s = scores4();
        let z = linkage_from_scores(
            s.view(),
            Method::Average,
            Symmetry::Mean,
            Transform::OneMinus,
            None,
            None,
        )
        .unwrap();

        let order = leaf_order(&z, 4);
        let mut sorted = order.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![0, 1, 2, 3], "must be a permutation of the leaves");

        // {0,1} and {2,3} are the two real clusters, so each must appear
        // contiguously — that is exactly what "does not cross" means here.
        let pos: Vec<usize> = (0..4).map(|i| order.iter().position(|&o| o == i).unwrap()).collect();
        assert_eq!(pos[0].abs_diff(pos[1]), 1, "0 and 1 must be adjacent");
        assert_eq!(pos[2].abs_diff(pos[3]), 1, "2 and 3 must be adjacent");
    }

    /// A degenerate chained tree is the deep case; the walk must be iterative
    /// enough to survive it and still return every leaf exactly once.
    #[test]
    fn leaf_order_handles_a_deep_chain() {
        let n = 5000;
        // Points on a line, each a little further out: single linkage chains them.
        let mut cond = Vec::with_capacity(condensed_len(n));
        for i in 0..n {
            for j in (i + 1)..n {
                cond.push((j - i) as f64);
            }
        }
        let z = linkage(&mut cond, n, Method::Single).unwrap();

        let order = leaf_order(&z, n);
        assert_eq!(order.len(), n);
        let mut seen = vec![false; n];
        for &o in &order {
            assert!(!seen[o], "leaf {o} emitted twice");
            seen[o] = true;
        }
        assert!(seen.iter().all(|&s| s));
    }

    #[test]
    fn leaf_order_edge_cases() {
        let z = Array2::<f64>::zeros((0, 4));
        assert_eq!(leaf_order(&z, 0), Vec::<usize>::new());
        assert_eq!(leaf_order(&z, 1), vec![0]);
    }

    #[test]
    fn check_finite_flags_nan() {
        assert!(check_finite(&[1.0f32, 2.0, 3.0], None).is_ok());
        assert!(check_finite(&[1.0f32, f32::NAN, 3.0], None).is_err());
        assert!(check_finite(&[1.0f64, f64::INFINITY], None).is_err());
    }

    #[test]
    fn cancellation_is_observed() {
        let s = scores4();
        let cancel = AtomicBool::new(true);
        assert!(matches!(
            condense(s.view(), Symmetry::Mean, Transform::AsIs, None, Some(&cancel)),
            Err(LinkageError::Cancelled)
        ));
    }
}
