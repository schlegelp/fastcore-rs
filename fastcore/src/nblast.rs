//! NBLAST: a pure-Rust pipeline built on `shull` (3D Delaunay) + `aann-graph`
//! (neighbourhood-graph nearest-neighbour search) + a fast scoring kernel.
//!
//! For an ordered pair of neurons `(i, j)` the score is the *forward* NBLAST
//! score of query `i` against target `j`: for every point of `i` we find its
//! nearest point in `j`, combine the euclidean distance with the absolute dot
//! product of the two tangent vectors via a lookup ("scoring") matrix, and sum.
//! When `normalize` is set the sum is divided by the query's self-hit score so a
//! perfect self-match is 1.0.
//!
//! Everything runs in Rust under `rayon`; each pair's nearest-neighbour result
//! is produced and dropped inside its own closure, so peak memory stays at
//! `O(threads)` NN buffers rather than `O(n^2)`.
//!
//! Feature parity with navis' NBLAST is provided through [`Opts`] and the
//! optional per-point `alpha` arrays:
//!   * `use_alpha` — weight each dot product by `sqrt(alpha_q * alpha_t)` (pass
//!     alpha arrays alongside points/vects).
//!   * `limit_dist` — cap the contribution of points whose nearest neighbour is
//!     beyond a distance bound (navis' `distance_upper_bound`).
//!   * `threads` — cap the rayon worker count for one call (navis' `n_cores`).
//!   * `precision` — the output matrix element type (`f32` / `f64`), chosen by
//!     the [`ScoreOut`] type parameter; the scoring math always runs in `f64`.

use aann::{graph_from_simplices, GroupedQueriesF64, PreparedF64};
use console::Term;
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use ndarray_017::{Array1, Array2};
use rayon::prelude::*;
use shull::delaunay4d;
use std::sync::atomic::{AtomicBool, Ordering};

// ---------------------------------------------------------------------------
// Output element type (precision)
// ---------------------------------------------------------------------------

/// Morton block size for `aann`'s grouped blocked descent. 32 is the measured
/// sweet spot for neuron-scale clouds (thousands of points): the descent's
/// gather-reuse gain plateaus at ~2x around 32-64, so 32 keeps the per-block
/// scratch small while capturing it. See the profiler's block sweep.
pub(crate) const GROUP_BLOCK: usize = 32;

/// Element type of the returned score matrix. The scoring accumulates in `f64`;
/// this only controls the width the final score is stored at (navis' `precision`
/// of 32 / 64). `from_f64` performs the single narrowing cast per cell.
pub trait ScoreOut: Copy + Send + Sync + 'static {
    fn from_f64(x: f64) -> Self;
}

impl ScoreOut for f32 {
    #[inline]
    fn from_f64(x: f64) -> Self {
        x as f32
    }
}

impl ScoreOut for f64 {
    #[inline]
    fn from_f64(x: f64) -> Self {
        x
    }
}

// ---------------------------------------------------------------------------
// Per-call options (parity knobs)
// ---------------------------------------------------------------------------

/// Scalar options shared by both entry points. Point / tangent / alpha data are
/// passed separately (they are per-neuron), everything else lives here.
#[derive(Clone, Copy)]
pub struct Opts<'a> {
    /// Scoring matrix (embedded FCWB by default, or a caller-supplied one).
    pub smat: &'a Smat,
    /// Divide each score by the query's self-hit so a perfect self-match is 1.0.
    pub normalize: bool,
    /// Distance upper bound (navis' `limit_dist`). A query point whose nearest
    /// neighbour is farther than this is scored at the "far + orthogonal" corner
    /// of the matrix (`[dist_bin(limit), dot_bin(0)]`); `None` disables it.
    pub limit_dist: Option<f64>,
    /// Cap the rayon worker count for this call (navis' `n_cores`). `None` (or
    /// `Some(0)`) uses the default global pool.
    pub threads: Option<usize>,
    /// Draw progress bars to stderr: first over the index build, then the
    /// scoring cells.
    pub progress: bool,
    /// Cooperative cancellation flag. When present and set to `true` by the caller
    /// (e.g. the Python binding on a `KeyboardInterrupt`), the index-build and
    /// scoring loops short-circuit and the entry point returns early with a partial
    /// result the caller is expected to discard. `None` disables cancellation.
    pub cancel: Option<&'a AtomicBool>,
}

/// True when `cancel` is present and set — the per-cell / per-index Ctrl-C check.
/// A `Relaxed` load is enough: we only need to *eventually* observe cancellation,
/// not synchronise memory around it.
#[inline]
pub(crate) fn is_cancelled(cancel: Option<&AtomicBool>) -> bool {
    cancel.is_some_and(|c| c.load(Ordering::Relaxed))
}

/// Run `f` on a rayon pool capped to `threads` workers, or on the default global
/// pool when `threads` is `None`/`Some(0)`. A fresh scoped pool is built per call
/// only when a cap is requested, so the common (uncapped) path is zero-overhead.
pub(crate) fn with_pool<R, F>(threads: Option<usize>, f: F) -> R
where
    R: Send,
    F: FnOnce() -> R + Send,
{
    match threads {
        Some(n) if n >= 1 => rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build()
            .expect("failed to build rayon thread pool")
            .install(f),
        _ => f(),
    }
}

// ---------------------------------------------------------------------------
// Scoring matrix
// ---------------------------------------------------------------------------

/// A binned NBLAST scoring matrix plus the *left* edges of its bins.
///
/// `dist_edges.len() == values.nrows()` and `dot_edges.len() == values.ncols()`.
/// Edges are the ascending left boundary of each bin; binning clamps values
/// below the first edge into bin 0 and values above the last edge into the last
/// bin (see [`digitize`]).
pub struct Smat {
    pub values: Array2<f64>,
    pub dist_edges: Vec<f64>,
    pub dot_edges: Vec<f64>,
}

impl Smat {
    #[inline]
    fn dist_bin(&self, d: f64) -> usize {
        digitize(&self.dist_edges, d)
    }

    #[inline]
    fn dot_bin(&self, dp: f64) -> usize {
        digitize(&self.dot_edges, dp)
    }

    /// Self-hit score of a query with `n_query_pts` points (no alpha): a perfect
    /// match has zero distance and perfectly aligned tangents, i.e. the top-right
    /// cell of the matrix, summed over all points.
    #[inline]
    pub fn self_hit(&self, n_query_pts: usize) -> f64 {
        let ncols = self.values.ncols();
        self.values[(0, ncols - 1)] * n_query_pts as f64
    }

    /// Self-hit score when `use_alpha` is on. A self-match still has zero distance
    /// (dist bin 0) but the dot product for point `p` collapses to `alpha[p]`
    /// (`|v·v| * sqrt(alpha_p * alpha_p) = alpha_p`), so the per-point column
    /// varies. Matches navis' `calc_self_hit` alpha branch.
    #[inline]
    pub fn self_hit_alpha(&self, alpha: &[f64]) -> f64 {
        alpha
            .iter()
            .map(|&a| self.values[(0, self.dot_bin(a))])
            .sum()
    }

    /// syNBLAST per-point score for a nearest-neighbour distance `d`.
    ///
    /// Synapses carry no tangent vector, so navis scores them through this same
    /// matrix but with the dot product fixed at `1` (`score_fn(d, 1)`), i.e. the
    /// distance is looked up in the *last* dot-product column. A self-match's
    /// per-point contribution is therefore `syn_score(0.0)`, and a query synapse
    /// whose type is absent in the target is scored at `syn_score(inf)` (the worst,
    /// farthest bin).
    #[inline]
    pub fn syn_score(&self, d: f64) -> f64 {
        self.values[(self.dist_bin(d), self.dot_bin(1.0))]
    }

    /// Build from plain parts (as passed across the Python boundary).
    pub fn from_parts(
        values: Vec<f64>,
        nrows: usize,
        ncols: usize,
        dist_edges: Vec<f64>,
        dot_edges: Vec<f64>,
    ) -> Smat {
        let values = Array2::from_shape_vec((nrows, ncols), values)
            .expect("smat values length must equal nrows * ncols");
        Smat {
            values,
            dist_edges,
            dot_edges,
        }
    }

    /// The navis `limit_dist="auto"` value for this matrix: `1.05 *` the left edge
    /// of the last distance bin (equivalent to navis' `boundaries[-2] * 1.05` for
    /// our inf-terminated matrices, where the last bin is open-ended).
    pub fn auto_limit(&self) -> f64 {
        self.dist_edges.last().copied().unwrap_or(0.0) * 1.05
    }
}

/// Index of the bin `value` falls into, given ascending left `edges`.
///
/// Returns `(# edges <= value) - 1`, clamped to `[0, edges.len() - 1]`. This
/// reproduces the classic NBLAST right-closed `(a, b]` binning with below-range
/// clamped to the first bin and above-range clamped to the last.
#[inline]
fn digitize(edges: &[f64], value: f64) -> usize {
    let c = edges.partition_point(|&e| e <= value);
    if c == 0 {
        0
    } else {
        c - 1
    }
}

/// Parse the `(left,right]` label of a bin and return its left boundary.
fn left_bound(label: &str) -> f64 {
    let left = label.split(',').next().unwrap_or("0");
    left.trim().trim_start_matches('(').parse().unwrap()
}

/// Load the embedded FCWB scoring matrix (21 distance bins x 10 dot bins).
pub fn load_smat() -> Smat {
    parse_smat(include_bytes!("../fastcore.data/smat_fcwb.csv"))
}

/// Load the embedded alpha-weighted FCWB matrix (16 distance bins x 10 dot bins).
///
/// This is the default matrix navis uses for `use_alpha` NBLAST — it is binned
/// and calibrated differently from the plain FCWB matrix, so alpha-weighted
/// scores need it (rather than the plain matrix) to match navis.
pub fn load_smat_alpha() -> Smat {
    parse_smat(include_bytes!("../fastcore.data/smat_alpha_fcwb.csv"))
}

/// Parse a FCWB-style CSV (row/column bin labels + one value per cell) into a
/// [`Smat`] with the ascending left bin edges.
fn parse_smat(data: &[u8]) -> Smat {
    let mut rdr = csv::Reader::from_reader(data);

    // Header row: dot-product (tangent alignment) bin labels; skip the corner.
    let mut dot_edges: Vec<f64> = vec![];
    if let Ok(row) = rdr.headers() {
        dot_edges = row.iter().skip(1).map(left_bound).collect();
    }

    // Remaining rows: distance bin label + one value per dot bin.
    let mut dist_edges: Vec<f64> = vec![];
    let mut flat: Vec<f64> = vec![];
    let mut ncols = 0usize;
    for result in rdr.records() {
        let record = result.unwrap();
        let row: Vec<f64> = record
            .iter()
            .skip(1)
            .map(|x| x.parse::<f64>().unwrap())
            .collect();
        ncols = row.len();
        dist_edges.push(left_bound(record.get(0).unwrap()));
        flat.extend(row);
    }
    let nrows = dist_edges.len();
    let values = Array2::from_shape_vec((nrows, ncols), flat).unwrap();

    Smat {
        values,
        dist_edges,
        dot_edges,
    }
}

// ---------------------------------------------------------------------------
// Neighbourhood-graph index (Delaunay build)
// ---------------------------------------------------------------------------

/// Build an `aann` nearest-neighbour index for one neuron.
///
/// Points are triangulated with `shull` and the resulting tetrahedra are turned
/// into a CSR neighbourhood graph. Points are packed in their original order, so
/// the neighbour indices returned by queries align with the caller's
/// tangent-vector array. Degenerate inputs (fewer than 5 points, or coplanar /
/// cospherical clouds that have no 3D triangulation) fall back to a complete
/// graph, over which graph descent is still exact.
pub fn build_index(points: &[[f64; 3]]) -> PreparedF64 {
    let n = points.len();
    let flat: Vec<f64> = points.iter().flatten().copied().collect();
    let arr: Array2<f64> = Array2::from_shape_vec((n, 3), flat).unwrap();

    let (indptr, indices) = match delaunay4d(arr.view()) {
        Ok((tets, _neighbors, _duplicates)) => {
            let simplices_flat: Vec<u64> = tets.iter().flatten().map(|&v| v as u64).collect();
            let simplices: Array2<u64> =
                Array2::from_shape_vec((tets.len(), 4), simplices_flat).unwrap();
            graph_from_simplices(simplices.view(), n)
        }
        // Rare for real (3D) neurons; keep the pipeline robust rather than panic.
        Err(_) => complete_graph_csr(n),
    };

    PreparedF64::new(arr.view(), indptr.view(), indices.view())
}

/// CSR adjacency of a complete graph on `n` vertices (each vertex adjacent to
/// all others). Used as the fallback when Delaunay triangulation is undefined.
fn complete_graph_csr(n: usize) -> (Array1<usize>, Array1<usize>) {
    let mut indptr: Vec<usize> = Vec::with_capacity(n + 1);
    let mut indices: Vec<usize> = Vec::with_capacity(n.saturating_mul(n.saturating_sub(1)));
    indptr.push(0);
    for k in 0..n {
        for j in 0..n {
            if j != k {
                indices.push(j);
            }
        }
        indptr.push(indices.len());
    }
    (Array1::from(indptr), Array1::from(indices))
}

// ---------------------------------------------------------------------------
// Scoring
// ---------------------------------------------------------------------------

/// Raw (un-normalized) forward NBLAST score for one query -> target pair.
///
/// `dists[p]` / `idx[p]` are the nearest-neighbour distance and target index for
/// query point `p`; `q_vect[p]` and `t_vect[idx[p]]` are the tangent vectors.
///
/// * `q_alpha` / `t_alpha` — when both are `Some`, the dot product is weighted by
///   `sqrt(q_alpha[p] * t_alpha[idx[p]])` (navis' `use_alpha`).
/// * `limit_dist` — when `Some(lim)`, a point whose nearest neighbour is farther
///   than `lim` is scored at `[dist_bin(lim), dot_bin(0)]`, exactly reproducing
///   navis (which caps the distance at the bound and zeroes the dot product /
///   alpha for such "no match within bound" points). Because `aann` returns the
///   true global nearest neighbour, `dists[p] > lim` is equivalent to navis'
///   `distance_upper_bound` finding no neighbour.
#[allow(clippy::too_many_arguments)]
pub fn score_pair(
    dists: &[f64],
    idx: &[usize],
    q_vect: &[[f64; 3]],
    t_vect: &[[f64; 3]],
    q_alpha: Option<&[f64]>,
    t_alpha: Option<&[f64]>,
    limit_dist: Option<f64>,
    smat: &Smat,
) -> f64 {
    let mut raw = 0.0;
    for p in 0..dists.len() {
        let d = dists[p];
        // Over-limit: cap the distance at the bound and treat the dot product as
        // zero (navis zeroes both the dot product and, if used, alpha).
        if let Some(lim) = limit_dist {
            if d > lim {
                raw += smat.values[(smat.dist_bin(lim), smat.dot_bin(0.0))];
                continue;
            }
        }
        let j = idx[p];
        let mut dp = (q_vect[p][0] * t_vect[j][0]
            + q_vect[p][1] * t_vect[j][1]
            + q_vect[p][2] * t_vect[j][2])
            .abs();
        if let (Some(qa), Some(ta)) = (q_alpha, t_alpha) {
            dp *= (qa[p] * ta[j]).sqrt();
        }
        raw += smat.values[(smat.dist_bin(d), smat.dot_bin(dp))];
    }
    raw
}

/// Progress bar over `total` scoring cells (only built when `progress` is on).
///
/// Drawn to stderr — animated in place in a terminal, plain reprinted text under
/// Jupyter. We draw through `ProgressDrawTarget::term_like` rather than
/// `ProgressDrawTarget::stderr()` because the latter self-hides whenever stderr is
/// not a TTY (which is exactly the Jupyter case), leaving notebook users with no
/// feedback at all.
pub(crate) fn scoring_bar(total: u64) -> ProgressBar {
    make_bar("NBLAST", total)
}

/// Progress bar over `total` neuron-index builds (only built when `progress` is
/// on). Precedes the scoring bar; see [`make_bar`] for why it draws through
/// `term_like`.
pub(crate) fn index_bar(total: u64) -> ProgressBar {
    make_bar("Indexing", total)
}

/// Shared constructor for the index / scoring bars: a stderr `ProgressBar`
/// prefixed with `label`. See [`scoring_bar`] for why we draw through
/// `ProgressDrawTarget::term_like` rather than `ProgressDrawTarget::stderr()`.
pub(crate) fn make_bar(label: &str, total: u64) -> ProgressBar {
    let target = ProgressDrawTarget::term_like(Box::new(Term::stderr()));
    let pb = ProgressBar::with_draw_target(Some(total), target);
    pb.set_style(
        ProgressStyle::with_template(&format!(
            "{label} {{bar:40}} {{pos}}/{{len}} [{{elapsed_precise}}] ETA {{eta}}"
        ))
        .unwrap()
        .progress_chars("=>-"),
    );
    // Keep the readout advancing even when ticks arrive sparsely (e.g. Jupyter).
    pb.enable_steady_tick(std::time::Duration::from_millis(250));
    pb
}

/// Build one `aann` index per point cloud, in parallel. When `bar` is `Some`,
/// each finished index ticks it once; when `None` the build stays on its
/// original zero-overhead parallel map (the default `progress = false` path).
///
/// Returns `None` if `cancel` is observed set mid-build (the parallel map
/// short-circuits); the caller treats that as an interrupted call and bails.
pub(crate) fn build_indices(
    clouds: &[Vec<[f64; 3]>],
    bar: Option<&ProgressBar>,
    cancel: Option<&AtomicBool>,
) -> Option<Vec<PreparedF64>> {
    match bar {
        Some(bar) => clouds
            .par_iter()
            .map(|p| {
                if is_cancelled(cancel) {
                    return None;
                }
                let idx = build_index(p);
                bar.inc(1);
                Some(idx)
            })
            .collect(),
        None => clouds
            .par_iter()
            .map(|p| {
                if is_cancelled(cancel) {
                    return None;
                }
                Some(build_index(p))
            })
            .collect(),
    }
}

/// Score every cell of `scores` in parallel via `compute`, ticking a progress bar
/// when `progress`, and short-circuiting (leaving the remaining cells unwritten) as
/// soon as `cancel` is observed set. Shared by every NBLAST/syNBLAST scoring loop so
/// the cooperative Ctrl-C check lives in exactly one place. An interrupted call's
/// partially-filled `scores` is discarded by the caller, so the unwritten tail is
/// harmless.
pub(crate) fn run_scoring<T, F>(
    scores: &mut [T],
    total: u64,
    progress: bool,
    cancel: Option<&AtomicBool>,
    compute: F,
) where
    T: Send,
    F: Fn((usize, &mut T)) + Sync + Send,
{
    if progress {
        let bar = scoring_bar(total);
        let done = scores
            .par_iter_mut()
            .enumerate()
            .try_for_each(|item| -> Result<(), ()> {
                if is_cancelled(cancel) {
                    return Err(());
                }
                compute(item);
                bar.inc(1);
                Ok(())
            });
        // On interrupt leave the bar where it stopped rather than `finish()`ing it
        // to a misleading 100% right before the caller raises `KeyboardInterrupt`.
        if done.is_ok() {
            bar.finish();
        } else {
            bar.abandon();
        }
    } else {
        let _ = scores
            .par_iter_mut()
            .enumerate()
            .try_for_each(|item| -> Result<(), ()> {
                if is_cancelled(cancel) {
                    return Err(());
                }
                compute(item);
                Ok(())
            });
    }
}

/// All-by-all forward NBLAST over `points` / `vects` (one entry per neuron).
///
/// `alphas`, when supplied, are the per-point alpha weights (one array per
/// neuron, same shape as `points`), enabling navis' `use_alpha` weighting; pass
/// `None` to disable it. Returns a flat row-major `n * n` matrix where cell
/// `[i * n + j]` is the score of query `i` against target `j`. With
/// `opts.normalize`, the diagonal is 1.0. The element type `T` selects the output
/// precision.
pub fn nblast_allbyall<T: ScoreOut>(
    points: Vec<Vec<[f64; 3]>>,
    vects: Vec<Vec<[f64; 3]>>,
    alphas: Option<Vec<Vec<f64>>>,
    opts: Opts,
) -> Vec<T> {
    let Opts {
        smat,
        normalize,
        limit_dist,
        threads,
        progress,
        cancel,
    } = opts;

    with_pool(threads, move || {
        let n = points.len();

        // Build every index once, in parallel; reused across all pairs it appears in.
        let idx_bar = progress.then(|| index_bar(n as u64));
        let Some(indices) = build_indices(&points, idx_bar.as_ref(), cancel) else {
            return Vec::new(); // interrupted mid-build; caller discards the result
        };
        if let Some(bar) = idx_bar {
            bar.finish();
        }
        let self_hits: Vec<f64> = (0..n)
            .map(|i| match &alphas {
                Some(a) => smat.self_hit_alpha(&a[i]),
                None => smat.self_hit(points[i].len()),
            })
            .collect();

        // Prepare the grouped query set ONCE (a target-independent concatenation +
        // Morton sort of every neuron's points). `aann` reuses it for every target,
        // turning the per-pair descent into a single per-target blocked descent that
        // shares each target-vertex gather across spatially-adjacent query points
        // (~2x on the descent). `offsets[i]..offsets[i+1]` slices query i's result —
        // finalised back to original point order — out of each target's column.
        let query_refs: Vec<&PreparedF64> = indices.iter().collect();
        let no_perms: Vec<Option<&[i64]>> = vec![None; n];
        let gq = GroupedQueriesF64::prepare(&query_refs, &no_perms);
        let offsets = gq.offsets();

        // Target-major scoring. Each target `j` owns a contiguous column of the
        // column-major buffer `cm` (`cm[j * n + i]` = score of query i vs target j),
        // so the parallel writes are disjoint; we transpose to the row-major result
        // once at the end (cheap next to the descent).
        let mut cm: Vec<T> = vec![T::from_f64(0.0); n * n];
        let score_bar = progress.then(|| scoring_bar(n as u64));
        cm.par_chunks_mut(n).enumerate().for_each(|(j, col)| {
            if is_cancelled(cancel) {
                return; // interrupted; the partial result is discarded by the caller
            }
            // `limit_dist` is the descent's distance upper bound: `aann` prunes and
            // returns the miss marker (inf, |target|) for over-bound points, which
            // `score_pair` caps at the far corner. `finalize = true` returns results
            // in concatenated per-cloud original order, ready to slice per query.
            let (d, ix) = indices[j].query_grouped(&gq, None, GROUP_BLOCK, limit_dist, true);
            let d = d.as_slice().unwrap();
            let ix = ix.as_slice().unwrap();
            for i in 0..n {
                let s = if i == j {
                    // Self-match: normalized -> 1.0; raw -> the self-hit score.
                    if normalize {
                        1.0
                    } else {
                        self_hits[i]
                    }
                } else {
                    let (a, b) = (offsets[i], offsets[i + 1]);
                    let (qa, ta) = match &alphas {
                        Some(al) => (Some(al[i].as_slice()), Some(al[j].as_slice())),
                        None => (None, None),
                    };
                    let raw = score_pair(
                        &d[a..b], &ix[a..b], &vects[i], &vects[j], qa, ta, limit_dist, smat,
                    );
                    if normalize {
                        raw / self_hits[i]
                    } else {
                        raw
                    }
                };
                col[i] = T::from_f64(s);
            }
            if let Some(bar) = &score_bar {
                bar.inc(1);
            }
        });
        match score_bar {
            Some(bar) if !is_cancelled(cancel) => bar.finish(),
            Some(bar) => bar.abandon(),
            None => {}
        }
        if is_cancelled(cancel) {
            return Vec::new(); // interrupted mid-scoring; caller discards
        }

        // Transpose the column-major buffer into the row-major result.
        let mut scores: Vec<T> = vec![T::from_f64(0.0); n * n];
        for j in 0..n {
            let col = &cm[j * n..(j + 1) * n];
            for i in 0..n {
                scores[i * n + j] = col[i];
            }
        }
        scores
    })
}

/// Forward NBLAST of every query neuron against every target neuron.
///
/// `q_alphas` / `t_alphas`, when both supplied, enable `use_alpha` weighting.
/// Returns a flat row-major `n_query * n_target` matrix where cell
/// `[qi * n_target + tj]` is the score of query `qi` against target `tj`. The
/// element type `T` selects the output precision.
#[allow(clippy::too_many_arguments)]
pub fn nblast_query_target<T: ScoreOut>(
    q_points: Vec<Vec<[f64; 3]>>,
    q_vects: Vec<Vec<[f64; 3]>>,
    q_alphas: Option<Vec<Vec<f64>>>,
    t_points: Vec<Vec<[f64; 3]>>,
    t_vects: Vec<Vec<[f64; 3]>>,
    t_alphas: Option<Vec<Vec<f64>>>,
    opts: Opts,
) -> Vec<T> {
    let Opts {
        smat,
        normalize,
        limit_dist,
        threads,
        progress,
        cancel,
    } = opts;

    with_pool(threads, move || {
        let nq = q_points.len();
        let nt = t_points.len();

        // One shared bar spanning both index builds so it reads as a single phase.
        let idx_bar = progress.then(|| index_bar((nq + nt) as u64));
        let (Some(q_idx), Some(t_idx)) = (
            build_indices(&q_points, idx_bar.as_ref(), cancel),
            build_indices(&t_points, idx_bar.as_ref(), cancel),
        ) else {
            return Vec::new(); // interrupted mid-build; caller discards the result
        };
        if let Some(bar) = idx_bar {
            bar.finish();
        }
        let self_hits: Vec<f64> = (0..nq)
            .map(|i| match &q_alphas {
                Some(a) => smat.self_hit_alpha(&a[i]),
                None => smat.self_hit(q_points[i].len()),
            })
            .collect();

        // Prepare the grouped query set ONCE and descend each target against it (see
        // `nblast_allbyall`). Column `tj` of the column-major buffer holds every
        // query's score against target `tj`; `offsets` slices each query out.
        let query_refs: Vec<&PreparedF64> = q_idx.iter().collect();
        let no_perms: Vec<Option<&[i64]>> = vec![None; nq];
        let gq = GroupedQueriesF64::prepare(&query_refs, &no_perms);
        let offsets = gq.offsets();

        let mut cm: Vec<T> = vec![T::from_f64(0.0); nq * nt];
        let score_bar = progress.then(|| scoring_bar(nt as u64));
        cm.par_chunks_mut(nq).enumerate().for_each(|(tj, col)| {
            if is_cancelled(cancel) {
                return;
            }
            let (d, ix) = t_idx[tj].query_grouped(&gq, None, GROUP_BLOCK, limit_dist, true);
            let d = d.as_slice().unwrap();
            let ix = ix.as_slice().unwrap();
            for qi in 0..nq {
                let (a, b) = (offsets[qi], offsets[qi + 1]);
                let (qa, ta) = match (&q_alphas, &t_alphas) {
                    (Some(qa), Some(ta)) => (Some(qa[qi].as_slice()), Some(ta[tj].as_slice())),
                    _ => (None, None),
                };
                let raw = score_pair(
                    &d[a..b], &ix[a..b], &q_vects[qi], &t_vects[tj], qa, ta, limit_dist, smat,
                );
                col[qi] = T::from_f64(if normalize {
                    raw / self_hits[qi]
                } else {
                    raw
                });
            }
            if let Some(bar) = &score_bar {
                bar.inc(1);
            }
        });
        match score_bar {
            Some(bar) if !is_cancelled(cancel) => bar.finish(),
            Some(bar) => bar.abandon(),
            None => {}
        }
        if is_cancelled(cancel) {
            return Vec::new();
        }

        // Transpose column-major (nt columns of nq) into the row-major result.
        let mut scores: Vec<T> = vec![T::from_f64(0.0); nq * nt];
        for tj in 0..nt {
            let col = &cm[tj * nq..(tj + 1) * nq];
            for qi in 0..nq {
                scores[qi * nt + tj] = col[qi];
            }
        }
        scores
    })
}

/// Forward NBLAST for an explicit set of `(query_idx, target_idx)` pairs.
///
/// Builds each query and target index once (in parallel), exactly like
/// [`nblast_query_target`], then scores **only** the requested pairs instead of the
/// full `nq * nt` grid. The returned `Vec<T>` is aligned to `pairs`: element `k` is
/// the forward score of query `pairs[k].0` against target `pairs[k].1`, divided by
/// the query's self-hit when `opts.normalize`. This is the primitive the two-pass
/// "smart" NBLAST uses for its full-resolution second pass over a sparse candidate
/// set of pairs.
///
/// Like the dense paths, this groups the requested pairs by target and runs one
/// grouped blocked descent per target, so it uses the *same* nearest-neighbour
/// descent (and equidistant tie-breaking) as `nblast_query_target`: a target whose
/// full query set is selected reproduces that target's dense column exactly.
#[allow(clippy::too_many_arguments)]
pub fn nblast_pairs<T: ScoreOut>(
    q_points: Vec<Vec<[f64; 3]>>,
    q_vects: Vec<Vec<[f64; 3]>>,
    q_alphas: Option<Vec<Vec<f64>>>,
    t_points: Vec<Vec<[f64; 3]>>,
    t_vects: Vec<Vec<[f64; 3]>>,
    t_alphas: Option<Vec<Vec<f64>>>,
    pairs: Vec<(usize, usize)>,
    opts: Opts,
) -> Vec<T> {
    let Opts {
        smat,
        normalize,
        limit_dist,
        threads,
        progress,
        cancel,
    } = opts;

    with_pool(threads, move || {
        // One shared bar spanning both index builds so it reads as a single phase.
        let idx_bar = progress.then(|| index_bar((q_points.len() + t_points.len()) as u64));
        let (Some(q_idx), Some(t_idx)) = (
            build_indices(&q_points, idx_bar.as_ref(), cancel),
            build_indices(&t_points, idx_bar.as_ref(), cancel),
        ) else {
            return Vec::new(); // interrupted mid-build; caller discards the result
        };
        if let Some(bar) = idx_bar {
            bar.finish();
        }
        let self_hits: Vec<f64> = (0..q_points.len())
            .map(|i| match &q_alphas {
                Some(a) => smat.self_hit_alpha(&a[i]),
                None => smat.self_hit(q_points[i].len()),
            })
            .collect();

        // Group the requested pairs by target so each target is descended once, via
        // the same grouped blocked descent as the dense paths. `by_target[tj]` lists
        // `(pair index, query index)` for every requested pair against target `tj`.
        let nt = t_points.len();
        let mut by_target: Vec<Vec<(usize, usize)>> = vec![Vec::new(); nt];
        for (k, &(qi, tj)) in pairs.iter().enumerate() {
            by_target[tj].push((k, qi));
        }

        // Per target: prepare its query subset, descend once, score its pairs. When
        // every query is selected for a target the subset is the full query set, so
        // the Morton order (hence NN tie-breaking) matches the dense path exactly.
        let bar = progress.then(|| scoring_bar(pairs.len() as u64));
        let per_target: Vec<Vec<(usize, T)>> = (0..nt)
            .into_par_iter()
            .map(|tj| {
                let group = &by_target[tj];
                if group.is_empty() || is_cancelled(cancel) {
                    return Vec::new();
                }
                let subset: Vec<&PreparedF64> =
                    group.iter().map(|&(_, qi)| &q_idx[qi]).collect();
                let no_perms: Vec<Option<&[i64]>> = vec![None; subset.len()];
                let gq = GroupedQueriesF64::prepare(&subset, &no_perms);
                let offsets = gq.offsets();
                let (d, ix) = t_idx[tj].query_grouped(&gq, None, GROUP_BLOCK, limit_dist, true);
                let d = d.as_slice().unwrap();
                let ix = ix.as_slice().unwrap();
                let out: Vec<(usize, T)> = group
                    .iter()
                    .enumerate()
                    .map(|(c, &(k, qi))| {
                        let (a, b) = (offsets[c], offsets[c + 1]);
                        let (qa, ta) = match (&q_alphas, &t_alphas) {
                            (Some(qa2), Some(ta2)) => {
                                (Some(qa2[qi].as_slice()), Some(ta2[tj].as_slice()))
                            }
                            _ => (None, None),
                        };
                        let raw = score_pair(
                            &d[a..b], &ix[a..b], &q_vects[qi], &t_vects[tj], qa, ta, limit_dist,
                            smat,
                        );
                        let s = if normalize { raw / self_hits[qi] } else { raw };
                        (k, T::from_f64(s))
                    })
                    .collect();
                if let Some(b) = &bar {
                    b.inc(group.len() as u64);
                }
                out
            })
            .collect();
        match bar {
            Some(b) if !is_cancelled(cancel) => b.finish(),
            Some(b) => b.abandon(),
            None => {}
        }
        if is_cancelled(cancel) {
            return Vec::new();
        }

        // Scatter each target's pair scores back into `pairs` order (disjoint `k`).
        let mut scores: Vec<T> = vec![T::from_f64(0.0); pairs.len()];
        for group in &per_target {
            for &(k, s) in group {
                scores[k] = s;
            }
        }
        scores
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Tiny 2x2 scoring matrix: dist bins [0,1) / [1,inf), dot bins [0,0.5) / [0.5,1].
    fn test_smat() -> Smat {
        Smat {
            values: Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
            dist_edges: vec![0.0, 1.0],
            dot_edges: vec![0.0, 0.5],
        }
    }

    fn test_opts(smat: &Smat) -> Opts<'_> {
        Opts {
            smat,
            normalize: true,
            limit_dist: None,
            threads: None,
            progress: false,
            cancel: None,
        }
    }

    #[test]
    fn digitize_clamps_and_bins() {
        let edges = [0.0, 1.0, 2.0];
        assert_eq!(digitize(&edges, -5.0), 0); // below range -> first bin
        assert_eq!(digitize(&edges, 0.0), 0); // exactly first edge
        assert_eq!(digitize(&edges, 0.5), 0);
        assert_eq!(digitize(&edges, 1.0), 1); // exactly an edge -> its bin
        assert_eq!(digitize(&edges, 1.5), 1);
        assert_eq!(digitize(&edges, 2.0), 2);
        assert_eq!(digitize(&edges, 99.0), 2); // above range -> last bin
    }

    #[test]
    fn load_smat_has_expected_shape() {
        let smat = load_smat();
        assert_eq!(smat.values.nrows(), 21);
        assert_eq!(smat.values.ncols(), 10);
        assert_eq!(smat.dist_edges.len(), 21);
        assert_eq!(smat.dot_edges.len(), 10);
        // First edges are the left bound of the first bin, i.e. 0.
        assert_eq!(smat.dist_edges[0], 0.0);
        assert_eq!(smat.dot_edges[0], 0.0);
    }

    #[test]
    fn load_smat_alpha_has_expected_shape() {
        let smat = load_smat_alpha();
        // The alpha-calibrated FCWB matrix uses coarser distance bins (16 vs 21).
        assert_eq!(smat.values.nrows(), 16);
        assert_eq!(smat.values.ncols(), 10);
        assert_eq!(smat.dist_edges.len(), 16);
        assert_eq!(smat.dot_edges.len(), 10);
        assert_eq!(smat.dist_edges[0], 0.0);
    }

    #[test]
    fn score_pair_hand_example() {
        let smat = test_smat();
        // Point 0: dist 0.5 -> dist bin 0; dot |0.8| -> dot bin 1 => values[0,1] = 2.0
        // Point 1: dist 1.5 -> dist bin 1; dot |0.3| -> dot bin 0 => values[1,0] = 3.0
        let dists = [0.5, 1.5];
        let idx = [0usize, 1usize];
        let q_vect = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let t_vect = [[0.8, 0.0, 0.0], [0.0, 0.3, 0.0]];
        let raw = score_pair(&dists, &idx, &q_vect, &t_vect, None, None, None, &smat);
        assert!((raw - 5.0).abs() < 1e-12, "got {raw}");
    }

    #[test]
    fn score_pair_alpha_weights_dot() {
        let smat = test_smat();
        // Without alpha the aligned tangents (|dot| = 1) land in dot bin 1.
        // alpha product 1.0 * 0.09 -> sqrt 0.3 scales the dot to 0.3 -> dot bin 0,
        // moving the score from values[0,1]=2.0 to values[0,0]=1.0.
        let dists = [0.5];
        let idx = [0usize];
        let q_vect = [[1.0, 0.0, 0.0]];
        let t_vect = [[1.0, 0.0, 0.0]];
        let qa = [1.0];
        let ta = [0.09];
        let raw = score_pair(&dists, &idx, &q_vect, &t_vect, Some(&qa), Some(&ta), None, &smat);
        assert!((raw - 1.0).abs() < 1e-12, "got {raw}");
    }

    #[test]
    fn score_pair_limit_dist_caps_over_limit() {
        let smat = test_smat();
        // Point 0: dist 0.5 <= 1.0 -> real score values[0,1] = 2.0.
        // Point 1: dist 5.0 > 1.0 -> capped: dist bin(1.0)=1, dot bin(0)=0 => 3.0.
        let dists = [0.5, 5.0];
        let idx = [0usize, 1usize];
        let q_vect = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let t_vect = [[0.8, 0.0, 0.0], [0.0, 0.9, 0.0]];
        let raw = score_pair(&dists, &idx, &q_vect, &t_vect, None, None, Some(1.0), &smat);
        assert!((raw - 5.0).abs() < 1e-12, "got {raw}");
    }

    #[test]
    fn self_hit_scales_with_points() {
        let smat = test_smat();
        // top-right cell = values[0, 1] = 2.0
        assert!((smat.self_hit(3) - 6.0).abs() < 1e-12);
    }

    #[test]
    fn self_hit_alpha_sums_dot_bins() {
        let smat = test_smat();
        // dist bin 0 row = [1.0, 2.0]; alpha 0.3 -> dot bin 0 => 1.0, 0.8 -> bin 1 => 2.0.
        let alpha = [0.3, 0.8];
        assert!((smat.self_hit_alpha(&alpha) - 3.0).abs() < 1e-12);
    }

    #[test]
    fn build_index_delaunay_and_fallback() {
        // 5 distinct points -> a real Delaunay triangulation.
        let tetra = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.25, 0.25, 0.25],
        ];
        let idx = build_index(&tetra);
        assert_eq!(idx.n(), 5);

        // 3 points -> degenerate for shull -> complete-graph fallback, still valid.
        let tri = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let idx = build_index(&tri);
        assert_eq!(idx.n(), 3);
    }

    #[test]
    fn allbyall_diagonal_is_one_when_normalized() {
        let cloud_a = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 0.5],
        ];
        let cloud_b = vec![
            [5.0, 5.0, 5.0],
            [6.0, 5.0, 5.0],
            [5.0, 6.0, 5.0],
            [5.0, 5.0, 6.0],
            [5.5, 5.5, 5.5],
        ];
        let vect: Vec<[f64; 3]> = vec![[1.0, 0.0, 0.0]; 5];
        let points = vec![cloud_a, cloud_b];
        let vects = vec![vect.clone(), vect];
        let smat = load_smat();

        let m: Vec<f32> = nblast_allbyall(points, vects, None, test_opts(&smat));
        assert_eq!(m.len(), 4);
        assert!((m[0] - 1.0).abs() < 1e-6, "diag[0] = {}", m[0]);
        assert!((m[3] - 1.0).abs() < 1e-6, "diag[1] = {}", m[3]);
        assert!(m.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn allbyall_f64_precision_matches_f32() {
        let cloud_a = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 0.5],
        ];
        let cloud_b = vec![
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [2.0, 0.0, 1.0],
            [2.5, 0.5, 0.5],
        ];
        let vect: Vec<[f64; 3]> = vec![[1.0, 0.0, 0.0]; 5];
        let points = vec![cloud_a, cloud_b];
        let vects = vec![vect.clone(), vect];
        let smat = load_smat();

        let m32: Vec<f32> = nblast_allbyall(points.clone(), vects.clone(), None, test_opts(&smat));
        let m64: Vec<f64> = nblast_allbyall(points, vects, None, test_opts(&smat));
        assert_eq!(m32.len(), m64.len());
        for (a, b) in m32.iter().zip(m64.iter()) {
            assert!((*a as f64 - *b).abs() < 1e-5, "{a} vs {b}");
        }
    }

    #[test]
    fn query_target_capped_threads_matches_default() {
        // A worker cap must not change the result, only how it is scheduled.
        let cloud_a = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 0.5],
        ];
        let cloud_b = vec![
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [2.0, 0.0, 1.0],
            [2.5, 0.5, 0.5],
        ];
        let vect: Vec<[f64; 3]> = vec![[1.0, 0.0, 0.0]; 5];
        let smat = load_smat();
        let q = vec![cloud_a];
        let t = vec![cloud_b];
        let qv = vec![vect.clone()];
        let tv = vec![vect];

        let mut opts = test_opts(&smat);
        let default: Vec<f64> =
            nblast_query_target(q.clone(), qv.clone(), None, t.clone(), tv.clone(), None, opts);
        opts.threads = Some(1);
        let capped: Vec<f64> = nblast_query_target(q, qv, None, t, tv, None, opts);
        assert_eq!(default, capped);
    }

    #[test]
    fn pairs_match_dense_query_target() {
        // `nblast_pairs` on a subset of cells must equal the dense matrix at those
        // same (query, target) indices.
        let cloud_a = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 0.5],
        ];
        let cloud_b = vec![
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [2.0, 0.0, 1.0],
            [2.5, 0.5, 0.5],
        ];
        let vect: Vec<[f64; 3]> = vec![[1.0, 0.0, 0.0]; 5];
        let smat = load_smat();
        let q = vec![cloud_a.clone(), cloud_b.clone()];
        let t = vec![cloud_b, cloud_a];
        let qv = vec![vect.clone(), vect.clone()];
        let tv = vec![vect.clone(), vect];
        let opts = test_opts(&smat);

        let nt = t.len();
        let dense: Vec<f64> =
            nblast_query_target(q.clone(), qv.clone(), None, t.clone(), tv.clone(), None, opts);
        let pairs = vec![(0usize, 0usize), (0, 1), (1, 0), (1, 1)];
        let sparse: Vec<f64> = nblast_pairs(q, qv, None, t, tv, None, pairs.clone(), opts);

        for (k, &(qi, tj)) in pairs.iter().enumerate() {
            assert!(
                (sparse[k] - dense[qi * nt + tj]).abs() < 1e-12,
                "pair ({qi},{tj}): {} vs {}",
                sparse[k],
                dense[qi * nt + tj]
            );
        }
    }
}
