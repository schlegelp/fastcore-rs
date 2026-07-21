//! `nblast_knn` — the k nearest neighbours of every neuron under NBLAST, without
//! ever materialising the `n x n` score matrix.
//!
//! At connectomics scale the all-by-all is the wrong shape for a k-NN question: a
//! 164k-neuron dataset is 2.7e10 pairs and a 107 GB matrix, which is not merely
//! slow but larger than most machines, even though the answer wanted from it is a
//! 26 MB k-NN graph (the usual consumer being a UMAP embedding).
//!
//! # Why a cheap spatial pre-filter is sound here
//!
//! The FCWB scoring matrix has **finite support**: every cell beyond its last
//! distance bin (40 um; `limit_dist="auto"` is 42) is ~ -10, so two neurons that
//! do not overlap in space score at that floor and *cannot* be pulled apart by
//! shape. NBLAST similarity is therefore a strictly local function of geometry,
//! and a coarse voxel-occupancy signature is not an arbitrary proxy for it — it is
//! the same information at lower resolution. That is what lets stage 2 discard
//! ~99.8% of pairs before any NBLAST is computed.
//!
//! # The three stages
//!
//! 1. [`build_signatures`] — each neuron becomes a sparse, L2-normalised vector of
//!    voxel occupancy binned by tangent direction, with each point trilinearly
//!    splatted over its 8 surrounding voxels (the splat is worth ~0.05 recall@20:
//!    it stops two neurons that straddle a voxel boundary from sharing no mass).
//! 2. [`candidate_pairs`] — the `n_candidates` best targets per neuron by cosine
//!    similarity of those signatures, via an inverted index over features, then
//!    closed symmetrically so a crowded hub cannot silently drop a partner.
//! 3. [`nblast_knn`] — the **exact** NBLAST score for the surviving pairs only,
//!    reusing the same grouped-descent kernel as the dense paths, combined per
//!    [`Symmetry`] and reduced to the top `k` per row.
//!
//! Only stage 2 is approximate. Stage 3 is lossless by construction: any neuron
//! that belongs in the true top-`k` has a score above the global k-th, so if it is
//! in the candidate set the exact rerank must rank it in. All recall loss lives in
//! candidate generation, and every score returned is an exact NBLAST value.
//!
//! Measured on 163,976 zebrafish neurons: recall@20 = 0.990 at `n_candidates=200`,
//! scoring 0.16% of pairs.

use rayon::prelude::*;
use std::sync::atomic::AtomicBool;

use crate::nblast::{
    build_indices, index_bar, is_cancelled, score_pair, scoring_bar, with_pool, Coord, Opts,
    ScoreOut, GROUP_BLOCK,
};

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

/// How the two directions of a pair are combined before the top-`k` cut.
///
/// This matters more for k-NN than for a full matrix. With a matrix you can
/// symmetrise after the fact by averaging against the transpose; once only `k`
/// neighbours per row are kept, the transpose is gone, so the combine has to
/// happen *before* the cut or not at all. The asymmetry it fixes is real: a small
/// neuron contained in a large one scores high one way (all of it is matched) and
/// low the other (it covers a fraction of the big one), so a forward-only k-NN
/// makes the pair look like neighbours in one row and strangers in the other.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Symmetry {
    /// Row `i` keeps the raw forward score of `i` against each candidate.
    Forward,
    /// `(forward + reverse) / 2` — the k-NN analogue of `(M + M.T) / 2`.
    Mean,
    /// The pessimistic direction; a pair must match well *both* ways.
    Min,
    /// The optimistic direction.
    Max,
}

impl Symmetry {
    #[inline]
    fn combine(self, fwd: f64, rev: f64) -> f64 {
        match self {
            Symmetry::Forward => fwd,
            Symmetry::Mean => 0.5 * (fwd + rev),
            Symmetry::Min => fwd.min(rev),
            Symmetry::Max => fwd.max(rev),
        }
    }
}

/// Options for [`nblast_knn`], on top of the shared NBLAST [`Opts`].
#[derive(Clone, Copy)]
pub struct KnnOpts<'a> {
    /// Scoring matrix, normalisation, `limit_dist`, threads, progress, cancellation.
    pub nblast: Opts<'a>,
    /// Neighbours to return per neuron.
    pub k: usize,
    /// Candidate targets per neuron carried into the exact rerank. This is the
    /// single recall/cost knob. Measured recall@20 on 163,976 real neurons: 0.911
    /// at 50, 0.969 at 100, 0.990 at 200, 0.996 at 400. The budget needed to hold
    /// a given recall grows only about logarithmically with `n`, which is what
    /// makes the rerank `O(n log n)` rather than `O(n^2)`.
    pub n_candidates: usize,
    /// Signature voxel edge, in the units of `points` (um for the FCWB matrix).
    /// 10-20 measured equivalently; below ~10 recall drops as neurons stop sharing
    /// voxels at all.
    pub voxel: f64,
    /// Tangent-direction bins for the signature. 1 disables direction binning; 3
    /// is the cheap "which axis dominates |v|" case.
    pub n_dirs: usize,
    /// Trilinearly splat each point over its 8 surrounding voxels.
    pub splat: bool,
    /// How to combine the two directions of each pair.
    pub symmetry: Symmetry,
}

// ---------------------------------------------------------------------------
// 1. Signatures
// ---------------------------------------------------------------------------

/// Sparse L2-normalised voxel signatures, one row per neuron, in CSR layout.
pub struct Signatures {
    /// Row offsets into `indices` / `data`; length `n_neurons + 1`.
    pub indptr: Vec<usize>,
    /// Feature (voxel x direction bin) ids, ascending within each row.
    pub indices: Vec<u32>,
    /// L2-normalised weights aligned to `indices`.
    pub data: Vec<f32>,
    /// Total feature-space size.
    pub n_features: usize,
}

impl Signatures {
    #[inline]
    pub fn n_rows(&self) -> usize {
        self.indptr.len() - 1
    }

    #[inline]
    fn row(&self, i: usize) -> (&[u32], &[f32]) {
        let (a, b) = (self.indptr[i], self.indptr[i + 1]);
        (&self.indices[a..b], &self.data[a..b])
    }
}

/// `n` reference directions spread over the half-sphere (tangents are sign-free,
/// NBLAST uses `|dot|`), or `None` when direction binning is disabled.
fn direction_codebook(n_dirs: usize) -> Option<Vec<[f64; 3]>> {
    if n_dirs <= 1 || n_dirs == 3 {
        return None; // 1 = no binning; 3 = the dominant-axis special case
    }
    let n = n_dirs as f64;
    Some(
        (0..n_dirs)
            .map(|i| {
                let fi = i as f64 + 0.5;
                let phi = (1.0 - fi / n).acos();
                let theta = std::f64::consts::PI * (1.0 + 5f64.sqrt()) * fi;
                [
                    theta.cos() * phi.sin(),
                    theta.sin() * phi.sin(),
                    phi.cos(),
                ]
            })
            .collect(),
    )
}

/// Direction bin of one unit tangent, ignoring sign.
#[inline]
fn dir_bin(v: &[f64; 3], n_dirs: usize, codebook: Option<&[[f64; 3]]>) -> u32 {
    if n_dirs <= 1 {
        return 0;
    }
    match codebook {
        // n_dirs == 3: which axis dominates |v|.
        None => {
            let (a, b, c) = (v[0].abs(), v[1].abs(), v[2].abs());
            if a >= b && a >= c {
                0
            } else if b >= c {
                1
            } else {
                2
            }
        }
        Some(cb) => {
            let mut best = 0u32;
            let mut best_dot = -1.0;
            for (i, c) in cb.iter().enumerate() {
                let d = (v[0] * c[0] + v[1] * c[1] + v[2] * c[2]).abs();
                if d > best_dot {
                    best_dot = d;
                    best = i as u32;
                }
            }
            best
        }
    }
}

/// The voxel frame signatures are expressed in.
///
/// Feature ids are only comparable between two [`Signatures`] built on the *same*
/// grid, which is why the query/target entry point derives one grid spanning both
/// sets rather than letting each side pick its own bounding box.
#[derive(Clone, Copy, Debug)]
pub struct SigGrid {
    origin: [f64; 3],
    dims: [i64; 3],
    voxel: f64,
    n_dirs: usize,
}

impl SigGrid {
    /// A grid spanning every cloud in every supplied set.
    pub fn spanning<C: Coord>(sets: &[&[Vec<[C; 3]>]], voxel: f64, n_dirs: usize) -> SigGrid {
        assert!(voxel > 0.0, "voxel must be positive");
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        for set in sets {
            for cloud in set.iter() {
                for p in cloud {
                    for d in 0..3 {
                        lo[d] = lo[d].min(p[d].to_f64());
                        hi[d] = hi[d].max(p[d].to_f64());
                    }
                }
            }
        }
        if !lo[0].is_finite() {
            // No points at all; degenerate but not worth panicking over.
            lo = [0.0; 3];
            hi = [0.0; 3];
        }
        let origin: [f64; 3] = std::array::from_fn(|d| lo[d] - voxel);
        let dims: [i64; 3] =
            std::array::from_fn(|d| ((hi[d] + voxel - origin[d]) / voxel).ceil() as i64 + 2);
        SigGrid {
            origin,
            dims,
            voxel,
            n_dirs: n_dirs.max(1),
        }
    }

    #[inline]
    fn n_features(&self) -> usize {
        (self.dims[0] * self.dims[1] * self.dims[2]) as usize * self.n_dirs
    }
}

/// Build the voxel signatures for every neuron, in parallel, on a grid spanning
/// exactly these clouds.
///
/// Weights are `sqrt`-damped before normalising so that a dense arbor crossing one
/// voxel many times cannot dominate the cosine.
pub fn build_signatures<C: Coord>(
    points: &[Vec<[C; 3]>],
    vects: &[Vec<[C; 3]>],
    voxel: f64,
    n_dirs: usize,
    splat: bool,
) -> Signatures {
    let grid = SigGrid::spanning(&[points], voxel, n_dirs);
    build_signatures_on(points, vects, &grid, splat)
}

/// [`build_signatures`] against a caller-supplied grid, so two sets can be made
/// comparable.
pub fn build_signatures_on<C: Coord>(
    points: &[Vec<[C; 3]>],
    vects: &[Vec<[C; 3]>],
    grid: &SigGrid,
    splat: bool,
) -> Signatures {
    let n = points.len();
    let (origin, dims, voxel, n_dirs) = (grid.origin, grid.dims, grid.voxel, grid.n_dirs);
    let codebook = direction_codebook(n_dirs);
    let n_features = grid.n_features();

    let offsets: &[[i64; 3]] = if splat {
        &[
            [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
            [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1],
        ]
    } else {
        &[[0, 0, 0]]
    };

    let rows: Vec<(Vec<u32>, Vec<f32>)> = (0..n)
        .into_par_iter()
        .map(|i| {
            let (pts, vs) = (&points[i], &vects[i]);
            let mut buf: Vec<(u32, f32)> = Vec::with_capacity(pts.len() * offsets.len());
            for (p, v) in pts.iter().zip(vs.iter()) {
                // The grid itself stays f64 whatever `C` is, so which voxel a point
                // lands in - and hence the candidate shortlist - is decided the same
                // way at both widths.
                let g: [f64; 3] = std::array::from_fn(|d| (p[d].to_f64() - origin[d]) / voxel);
                let base: [i64; 3] = std::array::from_fn(|d| g[d].floor() as i64);
                let frac: [f64; 3] = std::array::from_fn(|d| g[d] - base[d] as f64);
                let db = dir_bin(
                    &std::array::from_fn(|d| v[d].to_f64()),
                    n_dirs,
                    codebook.as_deref(),
                );
                for off in offsets {
                    let mut w = 1.0f64;
                    if splat {
                        for d in 0..3 {
                            w *= if off[d] == 1 { frac[d] } else { 1.0 - frac[d] };
                        }
                        if w <= 0.0 {
                            continue;
                        }
                    }
                    let vx: [i64; 3] =
                        std::array::from_fn(|d| (base[d] + off[d]).clamp(0, dims[d] - 1));
                    let cell = (vx[0] * dims[1] + vx[1]) * dims[2] + vx[2];
                    buf.push((cell as u32 * n_dirs as u32 + db, w as f32));
                }
            }

            // Aggregate duplicates per neuron. The grid has far fewer cells than a
            // neuron has splatted points, so collapsing here (rather than globally)
            // is where the ~10x memory saving over a COO build comes from.
            buf.sort_unstable_by_key(|&(c, _)| c);
            let mut idx: Vec<u32> = Vec::new();
            let mut dat: Vec<f32> = Vec::new();
            for (c, w) in buf {
                if idx.last() == Some(&c) {
                    *dat.last_mut().unwrap() += w;
                } else {
                    idx.push(c);
                    dat.push(w);
                }
            }
            for w in dat.iter_mut() {
                *w = w.sqrt();
            }
            let norm = dat.iter().map(|w| (*w as f64) * (*w as f64)).sum::<f64>().sqrt();
            let inv = if norm > 0.0 { 1.0 / norm } else { 0.0 };
            for w in dat.iter_mut() {
                *w = (*w as f64 * inv) as f32;
            }
            (idx, dat)
        })
        .collect();

    let mut indptr = Vec::with_capacity(n + 1);
    indptr.push(0usize);
    let total: usize = rows.iter().map(|(i, _)| i.len()).sum();
    let mut indices = Vec::with_capacity(total);
    let mut data = Vec::with_capacity(total);
    for (i, d) in rows {
        indices.extend_from_slice(&i);
        data.extend_from_slice(&d);
        indptr.push(indices.len());
    }
    Signatures {
        indptr,
        indices,
        data,
        n_features,
    }
}

// ---------------------------------------------------------------------------
// 2. Candidates
// ---------------------------------------------------------------------------

/// Column-major (feature -> neurons) transpose of the signatures: the inverted
/// index the candidate search walks.
struct Inverted {
    indptr: Vec<usize>,
    rows: Vec<u32>,
    data: Vec<f32>,
}

fn invert(sig: &Signatures) -> Inverted {
    let mut counts = vec![0usize; sig.n_features + 1];
    for &f in &sig.indices {
        counts[f as usize + 1] += 1;
    }
    for i in 0..sig.n_features {
        counts[i + 1] += counts[i];
    }
    let indptr = counts;
    let mut cursor = indptr.clone();
    let mut rows = vec![0u32; sig.indices.len()];
    let mut data = vec![0f32; sig.indices.len()];
    for i in 0..sig.n_rows() {
        let (idx, dat) = sig.row(i);
        for (&f, &w) in idx.iter().zip(dat.iter()) {
            let slot = cursor[f as usize];
            rows[slot] = i as u32;
            data[slot] = w;
            cursor[f as usize] += 1;
        }
    }
    Inverted { indptr, rows, data }
}

/// Scratch reused across the queries handled by one worker thread.
///
/// `stamp` avoids clearing the accumulator between queries: an entry is live only
/// when its stamp equals the current query id, so the O(n) reset that would
/// otherwise dominate never happens.
struct Scratch {
    acc: Vec<f32>,
    stamp: Vec<u32>,
    touched: Vec<u32>,
}

impl Scratch {
    fn new(n: usize) -> Self {
        Scratch {
            acc: vec![0.0; n],
            stamp: vec![u32::MAX; n],
            touched: Vec::with_capacity(1024),
        }
    }
}

/// Unordered candidate pairs `(a, b)` with `a < b`, packed as `(a as u64) << 32 | b`,
/// sorted and deduplicated.
///
/// Each neuron proposes its `n_candidates` best targets by signature cosine; the
/// union is then closed symmetrically (a pair survives if *either* endpoint
/// proposed it), so a hub neuron that crowds out a smaller partner from its own
/// list is still compared against it.
pub fn candidate_pairs(
    sig: &Signatures,
    n_candidates: usize,
    cancel: Option<&AtomicBool>,
) -> Vec<u64> {
    let n = sig.n_rows();
    if n < 2 {
        return Vec::new();
    }
    let inv = invert(sig);
    let want = n_candidates.min(n - 1);

    let mut packed: Vec<u64> = (0..n)
        .into_par_iter()
        .map_init(
            || Scratch::new(n),
            |s, i| {
                if is_cancelled(cancel) {
                    return Vec::new();
                }
                s.touched.clear();
                let (idx, dat) = sig.row(i);
                for (&f, &w) in idx.iter().zip(dat.iter()) {
                    let (a, b) = (inv.indptr[f as usize], inv.indptr[f as usize + 1]);
                    for slot in a..b {
                        let j = inv.rows[slot] as usize;
                        let contrib = w * inv.data[slot];
                        if s.stamp[j] != i as u32 {
                            s.stamp[j] = i as u32;
                            s.acc[j] = contrib;
                            s.touched.push(j as u32);
                        } else {
                            s.acc[j] += contrib;
                        }
                    }
                }
                // Drop self, then take the `want` highest by accumulated cosine.
                let mut cand: Vec<u32> =
                    s.touched.iter().copied().filter(|&j| j as usize != i).collect();
                if cand.len() > want {
                    cand.select_nth_unstable_by(want, |&x, &y| {
                        s.acc[y as usize]
                            .partial_cmp(&s.acc[x as usize])
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                    cand.truncate(want);
                }
                cand.iter()
                    .map(|&j| {
                        let (lo, hi) = if (i as u32) < j { (i as u32, j) } else { (j, i as u32) };
                        ((lo as u64) << 32) | hi as u64
                    })
                    .collect()
            },
        )
        .flatten()
        .collect();

    packed.par_sort_unstable();
    packed.dedup();
    packed
}

// ---------------------------------------------------------------------------
// 3. Exact rerank + top-k
// ---------------------------------------------------------------------------

/// Top up rows that ended with fewer than `k` candidates, using nearest-centroid
/// neighbours, and report how many rows needed it.
///
/// A neuron only becomes a candidate for `i` if it shares a signature feature with
/// it, so a tiny fragment with almost no spatial footprint can fall short of `k`.
/// On 20k real neurons this hit 2 rows (0.01%), both fragments of under 20 points
/// — rare, but ragged rows break consumers like UMAP's `precomputed_knn`, which
/// needs exactly `k` per row. The added pairs are scored by the same exact kernel
/// as every other pair; centroid distance only decides *which* pairs get looked
/// at, and for neurons this isolated the true neighbours are all at the matrix
/// floor and mutually indistinguishable anyway.
///
/// Cost is `O(n_short * n)` with a 3-flop inner loop, so it is free in the normal
/// case and merely cheap in the pathological one (a voxel so fine that nothing
/// overlaps).
fn backfill_short_rows<C: Coord>(pairs: &mut Vec<u64>, points: &[Vec<[C; 3]>], k: usize) -> usize {
    let n = points.len();
    let mut counts = vec![0usize; n];
    for &p in pairs.iter() {
        counts[(p >> 32) as usize] += 1;
        counts[(p & 0xffff_ffff) as usize] += 1;
    }
    let short: Vec<usize> = (0..n).filter(|&i| counts[i] < k.min(n - 1)).collect();
    if short.is_empty() {
        return 0;
    }

    let centroids: Vec<[f64; 3]> = points
        .iter()
        .map(|c| {
            if c.is_empty() {
                return [0.0; 3];
            }
            let mut m = [0.0; 3];
            for p in c {
                for d in 0..3 {
                    m[d] += p[d].to_f64();
                }
            }
            for d in 0..3 {
                m[d] /= c.len() as f64;
            }
            m
        })
        .collect();

    // Existing partners of the short rows only, so the lookup stays small.
    let mut have: std::collections::HashMap<usize, std::collections::HashSet<u32>> =
        short.iter().map(|&i| (i, std::collections::HashSet::new())).collect();
    for &p in pairs.iter() {
        let (a, b) = ((p >> 32) as u32, (p & 0xffff_ffff) as u32);
        if let Some(s) = have.get_mut(&(a as usize)) {
            s.insert(b);
        }
        if let Some(s) = have.get_mut(&(b as usize)) {
            s.insert(a);
        }
    }

    let extra: Vec<u64> = short
        .par_iter()
        .flat_map_iter(|&i| {
            let seen = &have[&i];
            let need = k.min(n - 1).saturating_sub(counts[i]);
            let ci = centroids[i];
            let mut by_dist: Vec<(u32, f64)> = (0..n)
                .filter(|&j| j != i && !seen.contains(&(j as u32)))
                .map(|j| {
                    let c = centroids[j];
                    let d = (0..3).map(|t| (ci[t] - c[t]).powi(2)).sum::<f64>();
                    (j as u32, d)
                })
                .collect();
            let take = need.min(by_dist.len());
            if take > 0 && by_dist.len() > take {
                by_dist.select_nth_unstable_by(take - 1, |a, b| a.1.total_cmp(&b.1));
            }
            by_dist
                .into_iter()
                .take(take)
                .map(move |(j, _)| {
                    let (lo, hi) = if (i as u32) < j { (i as u32, j) } else { (j, i as u32) };
                    ((lo as u64) << 32) | hi as u64
                })
                .collect::<Vec<u64>>()
        })
        .collect();

    pairs.extend(extra);
    pairs.par_sort_unstable();
    pairs.dedup();
    short.len()
}

/// Group directed `(query -> target)` work by target, in CSR form.
///
/// Returns `(indptr, query, slot)`: for target `t`, the entries in
/// `query[indptr[t]..indptr[t+1]]` are the queries to score against it, and
/// `slot[..]` says where each result belongs in the directed score buffer.
/// Counting-sort rather than `Vec<Vec<_>>` — at 4e7 directed pairs the latter's
/// per-target allocations cost more than the scores do.
fn group_by_target(pairs: &[u64], n: usize) -> (Vec<usize>, Vec<u32>, Vec<u32>) {
    let mut counts = vec![0usize; n + 1];
    for &p in pairs {
        let (a, b) = ((p >> 32) as usize, (p & 0xffff_ffff) as usize);
        counts[b + 1] += 1; // a -> b
        counts[a + 1] += 1; // b -> a
    }
    for i in 0..n {
        counts[i + 1] += counts[i];
    }
    let indptr = counts;
    let mut cursor = indptr.clone();
    let mut query = vec![0u32; pairs.len() * 2];
    let mut slot = vec![0u32; pairs.len() * 2];
    for (k, &p) in pairs.iter().enumerate() {
        let (a, b) = ((p >> 32) as u32, (p & 0xffff_ffff) as u32);
        let s = cursor[b as usize];
        query[s] = a;
        slot[s] = (2 * k) as u32; // forward: a -> b
        cursor[b as usize] += 1;
        let s = cursor[a as usize];
        query[s] = b;
        slot[s] = (2 * k + 1) as u32; // reverse: b -> a
        cursor[a as usize] += 1;
    }
    (indptr, query, slot)
}

/// k nearest neighbours of every neuron under NBLAST.
///
/// Returns `(idx, scores)`, both flat row-major `n * k`. Row `i` holds neuron
/// `i`'s neighbours in descending score order; a row with fewer than `k`
/// candidates is padded with index `-1` and score `f64::NEG_INFINITY` cast to `T`.
/// Scores are exact NBLAST values (normalised when `opts.nblast.normalize`),
/// combined per `opts.symmetry`.
///
/// The index for each neuron is built **once** and serves as both query and
/// target, which is the main structural saving over calling `nblast_pairs` twice:
/// that path builds separate query and target index sets per call, so the two
/// directions cost four full index builds instead of one.
pub fn nblast_knn<T: ScoreOut, C: Coord>(
    points: Vec<Vec<[C; 3]>>,
    vects: Vec<Vec<[C; 3]>>,
    alphas: Option<Vec<Vec<f64>>>,
    opts: KnnOpts,
) -> (Vec<i64>, Vec<T>) {
    let KnnOpts {
        nblast:
            Opts {
                smat,
                normalize,
                limit_dist,
                threads,
                progress,
                cancel,
            },
        k,
        n_candidates,
        voxel,
        n_dirs,
        splat,
        symmetry,
    } = opts;

    let n = points.len();
    assert_eq!(vects.len(), n, "points and vects must have the same length");
    assert!(k >= 1, "k must be at least 1");
    let empty = || (Vec::new(), Vec::new());

    with_pool(threads, move || {
        if n == 0 {
            return empty();
        }

        // --- stage 1 + 2: signatures -> candidate pairs -------------------
        let sig = build_signatures(&points, &vects, voxel, n_dirs, splat);
        if is_cancelled(cancel) {
            return empty();
        }
        let mut pairs = candidate_pairs(&sig, n_candidates, cancel);
        drop(sig);
        if is_cancelled(cancel) {
            return empty();
        }
        backfill_short_rows(&mut pairs, &points, k);

        // --- stage 3: exact NBLAST on the survivors -----------------------
        // `build_indices` consumes the clouds so each is freed as its index appears;
        // the self-hits only need the counts, so take those first.
        let n_pts: Vec<usize> = points.iter().map(|c| c.len()).collect();
        let idx_bar = progress.then(|| index_bar(n as u64));
        let Some(indices) = build_indices(points, idx_bar.as_ref(), cancel) else {
            return empty();
        };
        if let Some(bar) = idx_bar {
            bar.finish();
        }

        let self_hits: Vec<f64> = (0..n)
            .map(|i| match &alphas {
                Some(a) => smat.self_hit_alpha(&a[i]),
                None => smat.self_hit(n_pts[i]),
            })
            .collect();

        let (grp_ptr, grp_query, grp_slot) = group_by_target(&pairs, n);
        let bar = progress.then(|| scoring_bar((pairs.len() * 2) as u64));

        // One grouped blocked descent per target, exactly as the dense paths do,
        // so the nearest-neighbour tie-breaking matches them cell for cell.
        let per_target: Vec<Vec<(u32, f64)>> = (0..n)
            .into_par_iter()
            .map(|tj| {
                let (a, b) = (grp_ptr[tj], grp_ptr[tj + 1]);
                if a == b || is_cancelled(cancel) {
                    return Vec::new();
                }
                let qs = &grp_query[a..b];
                let subset: Vec<&C::Prepared> =
                    qs.iter().map(|&qi| &indices[qi as usize]).collect();
                let no_perms: Vec<Option<&[i64]>> = vec![None; subset.len()];
                let gq = C::prepare_group(&subset, &no_perms);
                let offs = C::group_offsets(&gq);
                let (d, ix) = C::query_grouped(&indices[tj], &gq, GROUP_BLOCK, limit_dist);
                let d = d.as_slice().unwrap();
                let ix = ix.as_slice().unwrap();
                let out: Vec<(u32, f64)> = qs
                    .iter()
                    .enumerate()
                    .map(|(c, &qi)| {
                        let qi = qi as usize;
                        let (s0, s1) = (offs[c], offs[c + 1]);
                        let (qa, ta) = match &alphas {
                            Some(al) => (Some(al[qi].as_slice()), Some(al[tj].as_slice())),
                            None => (None, None),
                        };
                        let raw = score_pair(
                            &d[s0..s1], &ix[s0..s1], &vects[qi], &vects[tj], qa, ta,
                            limit_dist, smat,
                        );
                        let s = if normalize { raw / self_hits[qi] } else { raw };
                        (grp_slot[a + c], s)
                    })
                    .collect();
                if let Some(bar) = &bar {
                    bar.inc(out.len() as u64);
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
            return empty();
        }

        // Scatter into directed-score order: slot 2k is a->b, slot 2k+1 is b->a.
        let mut directed = vec![0f64; pairs.len() * 2];
        for group in &per_target {
            for &(slot, s) in group {
                directed[slot as usize] = s;
            }
        }
        drop(per_target);

        // --- combine directions, then top-k per row -----------------------
        // Row-major grouping again by counting sort: each pair contributes one
        // entry to each of its two endpoints' rows.
        let mut counts = vec![0usize; n + 1];
        for &p in &pairs {
            counts[(p >> 32) as usize + 1] += 1;
            counts[(p & 0xffff_ffff) as usize + 1] += 1;
        }
        for i in 0..n {
            counts[i + 1] += counts[i];
        }
        let row_ptr = counts;
        let mut cursor = row_ptr.clone();
        let mut row_other = vec![0u32; pairs.len() * 2];
        let mut row_score = vec![0f64; pairs.len() * 2];
        for (kk, &p) in pairs.iter().enumerate() {
            let (a, b) = ((p >> 32) as u32, (p & 0xffff_ffff) as u32);
            let fwd = directed[2 * kk]; // a -> b
            let rev = directed[2 * kk + 1]; // b -> a
            let s = cursor[a as usize];
            row_other[s] = b;
            row_score[s] = symmetry.combine(fwd, rev);
            cursor[a as usize] += 1;
            let s = cursor[b as usize];
            row_other[s] = a;
            // For `Forward`, row b's own view of the pair is its own forward score.
            row_score[s] = symmetry.combine(rev, fwd);
            cursor[b as usize] += 1;
        }
        drop(directed);

        let mut out_idx = vec![-1i64; n * k];
        let mut out_sc = vec![T::from_f64(f64::NEG_INFINITY); n * k];
        out_idx
            .par_chunks_mut(k)
            .zip(out_sc.par_chunks_mut(k))
            .enumerate()
            .for_each(|(i, (oi, os))| {
                let (a, b) = (row_ptr[i], row_ptr[i + 1]);
                let mut ord: Vec<u32> = (0..(b - a) as u32).collect();
                let m = k.min(ord.len());
                if m == 0 {
                    return;
                }
                if ord.len() > m {
                    ord.select_nth_unstable_by(m - 1, |&x, &y| {
                        row_score[a + y as usize]
                            .partial_cmp(&row_score[a + x as usize])
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                    ord.truncate(m);
                }
                ord.sort_unstable_by(|&x, &y| {
                    row_score[a + y as usize]
                        .partial_cmp(&row_score[a + x as usize])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                for (c, &o) in ord.iter().enumerate() {
                    oi[c] = row_other[a + o as usize] as i64;
                    os[c] = T::from_f64(row_score[a + o as usize]);
                }
            });

        (out_idx, out_sc)
    })
}

// ---------------------------------------------------------------------------
// Query -> target (rectangular) k-NN
// ---------------------------------------------------------------------------

/// Candidate `(query, target)` pairs, packed `(qi as u64) << 32 | tj`, sorted.
///
/// Unlike the all-by-all case there is no symmetric closure: the two sides are
/// different index spaces and the deliverable is per-*query* neighbours, so only
/// the query's own shortlist is meaningful. That makes a given `n_candidates`
/// slightly less generous here than in the square case (where closure adds the
/// pairs a target proposed, ~40% more), so raise it if you want matching recall.
fn candidate_pairs_rect(
    sig_q: &Signatures,
    sig_t: &Signatures,
    n_candidates: usize,
    cancel: Option<&AtomicBool>,
) -> Vec<u64> {
    let (nq, nt) = (sig_q.n_rows(), sig_t.n_rows());
    if nq == 0 || nt == 0 {
        return Vec::new();
    }
    let inv = invert(sig_t);
    let want = n_candidates.min(nt);

    let mut packed: Vec<u64> = (0..nq)
        .into_par_iter()
        .map_init(
            || Scratch::new(nt),
            |s, i| {
                if is_cancelled(cancel) {
                    return Vec::new();
                }
                s.touched.clear();
                let (idx, dat) = sig_q.row(i);
                for (&f, &w) in idx.iter().zip(dat.iter()) {
                    let (a, b) = (inv.indptr[f as usize], inv.indptr[f as usize + 1]);
                    for slot in a..b {
                        let j = inv.rows[slot] as usize;
                        let contrib = w * inv.data[slot];
                        if s.stamp[j] != i as u32 {
                            s.stamp[j] = i as u32;
                            s.acc[j] = contrib;
                            s.touched.push(j as u32);
                        } else {
                            s.acc[j] += contrib;
                        }
                    }
                }
                let mut cand: Vec<u32> = s.touched.clone();
                if cand.len() > want {
                    cand.select_nth_unstable_by(want, |&x, &y| {
                        s.acc[y as usize]
                            .partial_cmp(&s.acc[x as usize])
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                    cand.truncate(want);
                }
                cand.iter().map(|&j| ((i as u64) << 32) | j as u64).collect()
            },
        )
        .flatten()
        .collect();

    packed.par_sort_unstable();
    packed.dedup();
    packed
}

/// Top up query rows with fewer than `k` candidates, using nearest target
/// centroids. See [`backfill_short_rows`] for why this exists.
fn backfill_short_rows_rect<C: Coord>(
    pairs: &mut Vec<u64>,
    q_points: &[Vec<[C; 3]>],
    t_points: &[Vec<[C; 3]>],
    k: usize,
) -> usize {
    let (nq, nt) = (q_points.len(), t_points.len());
    let want = k.min(nt);
    let mut counts = vec![0usize; nq];
    for &p in pairs.iter() {
        counts[(p >> 32) as usize] += 1;
    }
    let short: Vec<usize> = (0..nq).filter(|&i| counts[i] < want).collect();
    if short.is_empty() {
        return 0;
    }

    let centroid = |c: &Vec<[C; 3]>| -> [f64; 3] {
        if c.is_empty() {
            return [0.0; 3];
        }
        let mut m = [0.0; 3];
        for p in c {
            for d in 0..3 {
                m[d] += p[d].to_f64();
            }
        }
        for d in 0..3 {
            m[d] /= c.len() as f64;
        }
        m
    };
    let qc: Vec<[f64; 3]> = q_points.iter().map(centroid).collect();
    let tc: Vec<[f64; 3]> = t_points.iter().map(centroid).collect();

    let mut have: std::collections::HashMap<usize, std::collections::HashSet<u32>> =
        short.iter().map(|&i| (i, std::collections::HashSet::new())).collect();
    for &p in pairs.iter() {
        if let Some(s) = have.get_mut(&((p >> 32) as usize)) {
            s.insert((p & 0xffff_ffff) as u32);
        }
    }

    let extra: Vec<u64> = short
        .par_iter()
        .flat_map_iter(|&i| {
            let seen = &have[&i];
            let need = want - counts[i];
            let ci = qc[i];
            let mut by_dist: Vec<(u32, f64)> = (0..nt)
                .filter(|&j| !seen.contains(&(j as u32)))
                .map(|j| {
                    let c = tc[j];
                    (j as u32, (0..3).map(|t| (ci[t] - c[t]).powi(2)).sum::<f64>())
                })
                .collect();
            let take = need.min(by_dist.len());
            if take > 0 && by_dist.len() > take {
                by_dist.select_nth_unstable_by(take - 1, |a, b| a.1.total_cmp(&b.1));
            }
            by_dist
                .into_iter()
                .take(take)
                .map(move |(j, _)| ((i as u64) << 32) | j as u64)
                .collect::<Vec<u64>>()
        })
        .collect();

    pairs.extend(extra);
    pairs.par_sort_unstable();
    pairs.dedup();
    short.len()
}

/// Group `(query, target)` pairs by one side, in CSR form.
///
/// With `by_target`, group `g` collects the pairs whose target is `g` and `other`
/// holds their query indices (the layout the forward pass wants: one descent per
/// target). With `by_target = false` the roles swap, which is what the reverse
/// direction needs since there the query neuron is the *target* of the descent.
fn group_rect(
    pairs: &[u64],
    n_groups: usize,
    by_target: bool,
) -> (Vec<usize>, Vec<u32>, Vec<u32>) {
    let key = |p: u64| -> usize {
        if by_target {
            (p & 0xffff_ffff) as usize
        } else {
            (p >> 32) as usize
        }
    };
    let other = |p: u64| -> u32 {
        if by_target {
            (p >> 32) as u32
        } else {
            (p & 0xffff_ffff) as u32
        }
    };
    let mut counts = vec![0usize; n_groups + 1];
    for &p in pairs {
        counts[key(p) + 1] += 1;
    }
    for i in 0..n_groups {
        counts[i + 1] += counts[i];
    }
    let indptr = counts;
    let mut cursor = indptr.clone();
    let mut others = vec![0u32; pairs.len()];
    let mut slots = vec![0u32; pairs.len()];
    for (k, &p) in pairs.iter().enumerate() {
        let s = cursor[key(p)];
        others[s] = other(p);
        slots[s] = k as u32;
        cursor[key(p)] += 1;
    }
    (indptr, others, slots)
}

/// One direction of the rerank: for every group, one grouped blocked descent of
/// its member clouds into the group's own index, scored and written to `out` at
/// each pair's slot.
///
/// `q_*` here are the *scoring* query side (whose self-hit normalises and whose
/// tangents lead), `t_*` the side being descended into — so the reverse pass just
/// swaps the two argument sets.
#[allow(clippy::too_many_arguments)]
fn score_groups<C: Coord>(
    indptr: &[usize],
    others: &[u32],
    slots: &[u32],
    q_idx: &[C::Prepared],
    q_vects: &[Vec<[C; 3]>],
    q_alphas: &Option<Vec<Vec<f64>>>,
    q_self_hits: &[f64],
    t_idx: &[C::Prepared],
    t_vects: &[Vec<[C; 3]>],
    t_alphas: &Option<Vec<Vec<f64>>>,
    normalize: bool,
    limit_dist: Option<f64>,
    smat: &crate::nblast::Smat,
    cancel: Option<&AtomicBool>,
    bar: Option<&indicatif::ProgressBar>,
    out: &mut [f64],
) {
    let per_group: Vec<Vec<(u32, f64)>> = (0..t_idx.len())
        .into_par_iter()
        .map(|tj| {
            let (a, b) = (indptr[tj], indptr[tj + 1]);
            if a == b || is_cancelled(cancel) {
                return Vec::new();
            }
            let qs = &others[a..b];
            let subset: Vec<&C::Prepared> = qs.iter().map(|&qi| &q_idx[qi as usize]).collect();
            let no_perms: Vec<Option<&[i64]>> = vec![None; subset.len()];
            let gq = C::prepare_group(&subset, &no_perms);
            let offs = C::group_offsets(&gq);
            let (d, ix) = C::query_grouped(&t_idx[tj], &gq, GROUP_BLOCK, limit_dist);
            let d = d.as_slice().unwrap();
            let ix = ix.as_slice().unwrap();
            let out: Vec<(u32, f64)> = qs
                .iter()
                .enumerate()
                .map(|(c, &qi)| {
                    let qi = qi as usize;
                    let (s0, s1) = (offs[c], offs[c + 1]);
                    let (qa, ta) = match (q_alphas, t_alphas) {
                        (Some(qa), Some(ta)) => {
                            (Some(qa[qi].as_slice()), Some(ta[tj].as_slice()))
                        }
                        _ => (None, None),
                    };
                    let raw = score_pair(
                        &d[s0..s1], &ix[s0..s1], &q_vects[qi], &t_vects[tj], qa, ta,
                        limit_dist, smat,
                    );
                    let s = if normalize { raw / q_self_hits[qi] } else { raw };
                    (slots[a + c], s)
                })
                .collect();
            if let Some(bar) = bar {
                bar.inc(out.len() as u64);
            }
            out
        })
        .collect();
    for group in &per_group {
        for &(slot, s) in group {
            out[slot as usize] = s;
        }
    }
}

/// The `k` nearest **targets** for every query, without the `nq x nt` matrix.
///
/// The rectangular counterpart of [`nblast_knn`], standing to it as
/// `nblast_query_target` stands to `nblast_allbyall`. Returns `(idx, scores)`,
/// both flat row-major `nq * k`; `idx` holds indices **into the target list**.
///
/// Two differences from the square form are deliberate: nothing is excluded from
/// a row (a neuron present in both sets legitimately matches itself at 1.0, as in
/// `nblast_query_target`), and there is no symmetric closure of the candidate set
/// (see [`candidate_pairs_rect`]).
#[allow(clippy::too_many_arguments)]
pub fn nblast_knn_query_target<T: ScoreOut, C: Coord>(
    q_points: Vec<Vec<[C; 3]>>,
    q_vects: Vec<Vec<[C; 3]>>,
    q_alphas: Option<Vec<Vec<f64>>>,
    t_points: Vec<Vec<[C; 3]>>,
    t_vects: Vec<Vec<[C; 3]>>,
    t_alphas: Option<Vec<Vec<f64>>>,
    opts: KnnOpts,
) -> (Vec<i64>, Vec<T>) {
    let KnnOpts {
        nblast:
            Opts {
                smat,
                normalize,
                limit_dist,
                threads,
                progress,
                cancel,
            },
        k,
        n_candidates,
        voxel,
        n_dirs,
        splat,
        symmetry,
    } = opts;

    let (nq, nt) = (q_points.len(), t_points.len());
    assert_eq!(q_vects.len(), nq, "q_points and q_vects must have the same length");
    assert_eq!(t_vects.len(), nt, "t_points and t_vects must have the same length");
    assert!(k >= 1, "k must be at least 1");
    let empty = || (Vec::new(), Vec::new());

    with_pool(threads, move || {
        if nq == 0 || nt == 0 {
            return (vec![-1i64; nq * k], vec![T::from_f64(f64::NEG_INFINITY); nq * k]);
        }

        // One grid spanning both sets, so feature ids are comparable across them.
        let grid = SigGrid::spanning(&[&q_points, &t_points], voxel, n_dirs);
        let sig_q = build_signatures_on(&q_points, &q_vects, &grid, splat);
        let sig_t = build_signatures_on(&t_points, &t_vects, &grid, splat);
        if is_cancelled(cancel) {
            return empty();
        }
        let mut pairs = candidate_pairs_rect(&sig_q, &sig_t, n_candidates, cancel);
        drop((sig_q, sig_t));
        if is_cancelled(cancel) {
            return empty();
        }
        backfill_short_rows_rect(&mut pairs, &q_points, &t_points, k);

        // Both cloud sets are consumed by the build; take the counts first.
        let q_n_pts: Vec<usize> = q_points.iter().map(|c| c.len()).collect();
        let t_n_pts: Vec<usize> = t_points.iter().map(|c| c.len()).collect();
        let idx_bar = progress.then(|| index_bar((nq + nt) as u64));
        let (Some(q_idx), Some(t_idx)) = (
            build_indices(q_points, idx_bar.as_ref(), cancel),
            build_indices(t_points, idx_bar.as_ref(), cancel),
        ) else {
            return empty();
        };
        if let Some(bar) = idx_bar {
            bar.finish();
        }

        let q_self: Vec<f64> = (0..nq)
            .map(|i| match &q_alphas {
                Some(a) => smat.self_hit_alpha(&a[i]),
                None => smat.self_hit(q_n_pts[i]),
            })
            .collect();
        let t_self: Vec<f64> = (0..nt)
            .map(|i| match &t_alphas {
                Some(a) => smat.self_hit_alpha(&a[i]),
                None => smat.self_hit(t_n_pts[i]),
            })
            .collect();

        let two_way = symmetry != Symmetry::Forward;
        let total = pairs.len() as u64 * if two_way { 2 } else { 1 };
        let bar = progress.then(|| scoring_bar(total));

        let mut fwd = vec![0f64; pairs.len()];
        let (ptr, oth, slt) = group_rect(&pairs, nt, true);
        score_groups(
            &ptr, &oth, &slt, &q_idx, &q_vects, &q_alphas, &q_self, &t_idx, &t_vects,
            &t_alphas, normalize, limit_dist, smat, cancel, bar.as_ref(), &mut fwd,
        );

        let mut rev = vec![0f64; pairs.len()];
        if two_way {
            // Same machinery with the sides swapped: the query neuron is now the
            // cloud being descended into, and the target's self-hit normalises.
            let (ptr, oth, slt) = group_rect(&pairs, nq, false);
            score_groups(
                &ptr, &oth, &slt, &t_idx, &t_vects, &t_alphas, &t_self, &q_idx, &q_vects,
                &q_alphas, normalize, limit_dist, smat, cancel, bar.as_ref(), &mut rev,
            );
        }
        match bar {
            Some(b) if !is_cancelled(cancel) => b.finish(),
            Some(b) => b.abandon(),
            None => {}
        }
        if is_cancelled(cancel) {
            return empty();
        }

        // `pairs` is sorted, so each query's candidates are already contiguous.
        let mut row_ptr = vec![0usize; nq + 1];
        for &p in &pairs {
            row_ptr[(p >> 32) as usize + 1] += 1;
        }
        for i in 0..nq {
            row_ptr[i + 1] += row_ptr[i];
        }

        let mut out_idx = vec![-1i64; nq * k];
        let mut out_sc = vec![T::from_f64(f64::NEG_INFINITY); nq * k];
        out_idx
            .par_chunks_mut(k)
            .zip(out_sc.par_chunks_mut(k))
            .enumerate()
            .for_each(|(i, (oi, os))| {
                let (a, b) = (row_ptr[i], row_ptr[i + 1]);
                let score = |s: usize| symmetry.combine(fwd[s], rev[s]);
                let mut ord: Vec<u32> = (0..(b - a) as u32).collect();
                let m = k.min(ord.len());
                if m == 0 {
                    return;
                }
                if ord.len() > m {
                    ord.select_nth_unstable_by(m - 1, |&x, &y| {
                        score(a + y as usize)
                            .partial_cmp(&score(a + x as usize))
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                    ord.truncate(m);
                }
                ord.sort_unstable_by(|&x, &y| {
                    score(a + y as usize)
                        .partial_cmp(&score(a + x as usize))
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                for (c, &o) in ord.iter().enumerate() {
                    oi[c] = (pairs[a + o as usize] & 0xffff_ffff) as i64;
                    os[c] = T::from_f64(score(a + o as usize));
                }
            });

        (out_idx, out_sc)
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
pub(crate) mod tests_support {
    pub(crate) use super::tests::{opts_for, population};
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nblast::{load_smat, nblast_allbyall, Smat};
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    pub(crate) fn opts_for(smat: &Smat) -> Opts<'_> {
        Opts {
            smat,
            normalize: true,
            limit_dist: None,
            threads: None,
            progress: false,
            cancel: None,
        }
    }

    fn knn_opts<'a>(smat: &'a Smat, k: usize, n_candidates: usize) -> KnnOpts<'a> {
        KnnOpts {
            nblast: opts_for(smat),
            k,
            n_candidates,
            voxel: 20.0,
            n_dirs: 3,
            splat: true,
            symmetry: Symmetry::Mean,
        }
    }

    /// `n_clusters` groups of jittered copies of a random walk, so the population
    /// has genuine near-neighbours rather than uniform noise.
    pub(crate) fn population(
        n_clusters: usize,
        per_cluster: usize,
        n_pts: usize,
        seed: u64,
    ) -> (Vec<Vec<[f64; 3]>>, Vec<Vec<[f64; 3]>>) {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut points = Vec::new();
        let mut vects = Vec::new();
        for _ in 0..n_clusters {
            let start: [f64; 3] = std::array::from_fn(|_| rng.gen_range(0.0..100.0));
            let mut walk = Vec::with_capacity(n_pts);
            let mut cur = start;
            let mut dir: [f64; 3] = std::array::from_fn(|_| rng.gen_range(-1.0..1.0));
            for _ in 0..n_pts {
                for d in 0..3 {
                    dir[d] += rng.gen_range(-0.3..0.3);
                }
                let norm = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
                for d in 0..3 {
                    cur[d] += dir[d] / norm;
                }
                walk.push(cur);
            }
            for _ in 0..per_cluster {
                let jitter: [f64; 3] = std::array::from_fn(|_| rng.gen_range(-2.0..2.0));
                let cloud: Vec<[f64; 3]> = walk
                    .iter()
                    .map(|p| std::array::from_fn(|d| p[d] + jitter[d] + rng.gen_range(-0.5..0.5)))
                    .collect();
                // Tangents along the local walk direction, normalised.
                let vect: Vec<[f64; 3]> = (0..cloud.len())
                    .map(|i| {
                        let j = if i + 1 < cloud.len() { i + 1 } else { i - 1 };
                        let mut v: [f64; 3] =
                            std::array::from_fn(|d| cloud[j][d] - cloud[i][d]);
                        let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-12);
                        for d in 0..3 {
                            v[d] /= n;
                        }
                        v
                    })
                    .collect();
                points.push(cloud);
                vects.push(vect);
            }
        }
        (points, vects)
    }

    /// Top-`k` of a dense row-major `n x n` matrix under `sym`, as `(idx, score)`.
    fn dense_topk(dense: &[f64], n: usize, k: usize, sym: Symmetry) -> Vec<Vec<(usize, f64)>> {
        (0..n)
            .map(|i| {
                let mut row: Vec<(usize, f64)> = (0..n)
                    .filter(|&j| j != i)
                    .map(|j| (j, sym.combine(dense[i * n + j], dense[j * n + i])))
                    .collect();
                row.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                row.truncate(k);
                row
            })
            .collect()
    }

    /// A signature grid coarse enough that every neuron shares the single feature,
    /// which makes the candidate stage exhaustive and the result exactly the dense
    /// top-k. This is the correctness anchor for the whole pipeline.
    fn exhaustive(smat: &Smat, k: usize, n: usize, sym: Symmetry) -> KnnOpts<'_> {
        let mut o = knn_opts(smat, k, n - 1);
        o.voxel = 1.0e6;
        o.n_dirs = 1;
        o.splat = false;
        o.symmetry = sym;
        o
    }

    #[test]
    fn exhaustive_candidates_reproduce_dense_topk() {
        let (points, vects) = population(4, 4, 40, 7);
        let n = points.len();
        let smat = load_smat();
        let k = 5;
        let dense: Vec<f64> =
            nblast_allbyall(points.clone(), vects.clone(), None, opts_for(&smat));

        for sym in [Symmetry::Forward, Symmetry::Mean, Symmetry::Min, Symmetry::Max] {
            let (idx, sc): (Vec<i64>, Vec<f64>) = nblast_knn(
                points.clone(),
                vects.clone(),
                None,
                exhaustive(&smat, k, n, sym),
            );
            let want = dense_topk(&dense, n, k, sym);
            for i in 0..n {
                for c in 0..k {
                    let (wi, ws) = want[i][c];
                    assert!(
                        (sc[i * k + c] - ws).abs() < 1e-12,
                        "{sym:?} row {i} slot {c}: score {} vs dense {ws}",
                        sc[i * k + c]
                    );
                    // Indices may legitimately swap on an exact tie; scores may not.
                    if c == 0 || (want[i][c - 1].1 - ws).abs() > 1e-12 {
                        assert_eq!(
                            idx[i * k + c], wi as i64,
                            "{sym:?} row {i} slot {c}: index"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn returned_scores_are_exact_even_when_candidates_are_approximate() {
        // The rerank is lossless by construction: whatever survives candidate
        // generation is scored with the same kernel as the dense path, so every
        // returned score must equal its dense cell even at a tiny candidate budget.
        let (points, vects) = population(4, 4, 40, 11);
        let n = points.len();
        let smat = load_smat();
        let k = 3;
        let dense: Vec<f64> =
            nblast_allbyall(points.clone(), vects.clone(), None, opts_for(&smat));
        let (idx, sc): (Vec<i64>, Vec<f64>) =
            nblast_knn(points, vects, None, knn_opts(&smat, k, 2));

        for i in 0..n {
            for c in 0..k {
                let j = idx[i * k + c];
                if j < 0 {
                    continue;
                }
                let j = j as usize;
                let want = 0.5 * (dense[i * n + j] + dense[j * n + i]);
                assert!(
                    (sc[i * k + c] - want).abs() < 1e-12,
                    "row {i} -> {j}: {} vs dense {want}",
                    sc[i * k + c]
                );
            }
        }
    }

    #[test]
    fn rows_are_sorted_descending_and_exclude_self() {
        let (points, vects) = population(3, 4, 30, 3);
        let n = points.len();
        let smat = load_smat();
        let k = 4;
        let (idx, sc): (Vec<i64>, Vec<f64>) =
            nblast_knn(points, vects, None, knn_opts(&smat, k, 6));
        for i in 0..n {
            for c in 0..k {
                assert_ne!(idx[i * k + c], i as i64, "row {i} contains itself");
                if c > 0 && idx[i * k + c] >= 0 {
                    assert!(
                        sc[i * k + c - 1] >= sc[i * k + c],
                        "row {i} not descending at {c}"
                    );
                }
            }
        }
    }

    #[test]
    fn pads_when_k_exceeds_available_candidates() {
        let (points, vects) = population(2, 3, 25, 5);
        let n = points.len();
        let smat = load_smat();
        let k = n + 4; // more neighbours than neurons exist
        let (idx, sc): (Vec<i64>, Vec<f64>) =
            nblast_knn(points, vects, None, knn_opts(&smat, k, n - 1));
        assert_eq!(idx.len(), n * k);
        for i in 0..n {
            // At most n-1 real neighbours; the tail must be the -1 / -inf padding.
            for c in (n - 1)..k {
                assert_eq!(idx[i * k + c], -1, "row {i} slot {c} should be padding");
                assert!(sc[i * k + c].is_infinite() && sc[i * k + c] < 0.0);
            }
        }
    }

    #[test]
    fn thread_count_does_not_change_the_answer() {
        let (points, vects) = population(3, 4, 30, 9);
        let n = points.len();
        let smat = load_smat();
        let mut o = exhaustive(&smat, 4, n, Symmetry::Mean);
        let (i1, s1): (Vec<i64>, Vec<f64>) =
            nblast_knn(points.clone(), vects.clone(), None, o);
        o.nblast.threads = Some(1);
        let (i2, s2): (Vec<i64>, Vec<f64>) = nblast_knn(points, vects, None, o);
        assert_eq!(i1, i2);
        assert_eq!(s1, s2);
    }

    #[test]
    fn mean_symmetry_fixes_the_containment_asymmetry() {
        // A short neuron lying inside a long one: forward scores it as a great
        // match (all of it is covered), the reverse as a poor one (it covers a
        // fraction of the long neuron). `Mean` must land between the two, which is
        // the whole reason the combine happens before the top-k cut.
        let long: Vec<[f64; 3]> = (0..120).map(|i| [i as f64 * 0.5, 0.0, 0.0]).collect();
        let short: Vec<[f64; 3]> = (0..20).map(|i| [i as f64 * 0.5, 0.0, 0.0]).collect();
        let lv = vec![[1.0, 0.0, 0.0]; long.len()];
        let sv = vec![[1.0, 0.0, 0.0]; short.len()];
        let points = vec![long, short];
        let vects = vec![lv, sv];
        let smat = load_smat();

        let dense: Vec<f64> =
            nblast_allbyall(points.clone(), vects.clone(), None, opts_for(&smat));
        let (long_to_short, short_to_long) = (dense[1], dense[2]);
        assert!(
            short_to_long > long_to_short + 0.2,
            "expected a strong containment asymmetry, got {short_to_long} vs {long_to_short}"
        );

        let (_, fwd): (Vec<i64>, Vec<f64>) = nblast_knn(
            points.clone(),
            vects.clone(),
            None,
            exhaustive(&smat, 1, 2, Symmetry::Forward),
        );
        let (_, mean): (Vec<i64>, Vec<f64>) =
            nblast_knn(points, vects, None, exhaustive(&smat, 1, 2, Symmetry::Mean));

        // Forward keeps each row's own (disagreeing) view ...
        assert!((fwd[0] - long_to_short).abs() < 1e-12);
        assert!((fwd[1] - short_to_long).abs() < 1e-12);
        assert!(fwd[1] - fwd[0] > 0.2, "forward rows should disagree");
        // ... mean gives both rows the same, intermediate value.
        assert!((mean[0] - mean[1]).abs() < 1e-12, "mean rows must agree");
        assert!(mean[0] > long_to_short && mean[0] < short_to_long);
    }

    #[test]
    fn signatures_are_l2_normalised_and_sorted() {
        let (points, vects) = population(3, 3, 30, 13);
        let sig = build_signatures(&points, &vects, 20.0, 3, true);
        assert_eq!(sig.n_rows(), points.len());
        for i in 0..sig.n_rows() {
            let (idx, dat) = sig.row(i);
            assert!(!idx.is_empty(), "row {i} is empty");
            assert!(idx.windows(2).all(|w| w[0] < w[1]), "row {i} not sorted/unique");
            assert!(idx.iter().all(|&f| (f as usize) < sig.n_features));
            let norm: f64 = dat.iter().map(|&w| (w as f64) * (w as f64)).sum::<f64>().sqrt();
            assert!((norm - 1.0).abs() < 1e-5, "row {i} norm {norm}");
        }
    }

    #[test]
    fn candidate_pairs_are_sorted_unique_and_ordered() {
        let (points, vects) = population(4, 4, 30, 17);
        let n = points.len();
        let sig = build_signatures(&points, &vects, 20.0, 3, true);
        let pairs = candidate_pairs(&sig, 5, None);
        assert!(pairs.windows(2).all(|w| w[0] < w[1]), "not sorted/deduped");
        for &p in &pairs {
            let (a, b) = ((p >> 32) as usize, (p & 0xffff_ffff) as usize);
            assert!(a < b, "pair not canonically ordered: ({a}, {b})");
            assert!(b < n, "pair index out of range: ({a}, {b})");
        }
    }

    #[test]
    fn exhaustive_candidates_cover_every_pair() {
        let (points, vects) = population(3, 3, 25, 19);
        let n = points.len();
        // One coarse voxel, no direction bins: every neuron shares the feature.
        let sig = build_signatures(&points, &vects, 1.0e6, 1, false);
        let pairs = candidate_pairs(&sig, n - 1, None);
        assert_eq!(pairs.len(), n * (n - 1) / 2, "expected the complete graph");
    }

    #[test]
    fn empty_input_returns_empty() {
        let smat = load_smat();
        let (idx, sc): (Vec<i64>, Vec<f64>) =
            nblast_knn::<f64, f64>(Vec::new(), Vec::new(), None, knn_opts(&smat, 5, 10));
        assert!(idx.is_empty() && sc.is_empty());
    }

    #[test]
    fn coord_f32_knn_matches_coord_f64() {
        // The k-NN graph is built from the same candidate shortlist at either width
        // - the signature grid stays f64 - so only the exact rerank scores can move,
        // and only by the f32 coordinate resolution. Neighbour *sets* should be
        // essentially identical; ordering within a near-tie may swap.
        let smat = load_smat();
        let (points, vects) = population(6, 5, 120, 42);
        let n = points.len();
        let (k, nc) = (5usize, 12usize);

        let narrow = |cs: &[Vec<[f64; 3]>]| -> Vec<Vec<[f32; 3]>> {
            cs.iter()
                .map(|c| c.iter().map(|p| [p[0] as f32, p[1] as f32, p[2] as f32]).collect())
                .collect()
        };

        let (i64_idx, s64): (Vec<i64>, Vec<f64>) = nblast_knn(
            points.clone(),
            vects.clone(),
            None,
            knn_opts(&smat, k, nc),
        );
        let (i32_idx, s32): (Vec<i64>, Vec<f64>) =
            nblast_knn(narrow(&points), narrow(&vects), None, knn_opts(&smat, k, nc));

        assert_eq!(i32_idx.len(), n * k);
        // Row-wise: the two neighbour sets must overlap in all but at most one slot.
        for i in 0..n {
            let a: std::collections::HashSet<i64> =
                i64_idx[i * k..(i + 1) * k].iter().copied().collect();
            let b: std::collections::HashSet<i64> =
                i32_idx[i * k..(i + 1) * k].iter().copied().collect();
            assert!(
                a.intersection(&b).count() >= k - 1,
                "row {i}: f64 {a:?} vs f32 {b:?}"
            );
        }
        // And the score columns agree: slot j is the j-th best either way.
        for (a, b) in s32.iter().zip(s64.iter()) {
            assert!(
                (a - b).abs() <= 1e-3 * b.abs().max(1.0),
                "f32 coords gave {a}, f64 gave {b}"
            );
        }
    }
}

#[cfg(test)]
mod backfill_tests {
    use super::tests_support::*;
    use super::*;
    use crate::nblast::load_smat;

    #[test]
    fn isolated_neuron_still_gets_k_neighbours() {
        // A tiny fragment parked far from the population shares no signature
        // feature with anything, so candidate generation alone would leave its row
        // short. The backfill must top it up to k with real, exactly-scored pairs.
        let (mut points, mut vects) = population(3, 3, 30, 23);
        let far: Vec<[f64; 3]> = (0..6).map(|i| [9000.0 + i as f64, 9000.0, 9000.0]).collect();
        let far_v = vec![[1.0, 0.0, 0.0]; far.len()];
        points.push(far);
        vects.push(far_v);
        let n = points.len();
        let smat = load_smat();
        let k = 5;

        let sig = build_signatures(&points, &vects, 20.0, 3, true);
        let mut pairs = candidate_pairs(&sig, 4, None);
        let mut counts = vec![0usize; n];
        for &p in &pairs {
            counts[(p >> 32) as usize] += 1;
            counts[(p & 0xffff_ffff) as usize] += 1;
        }
        assert!(counts[n - 1] < k, "the far neuron should start short");
        let fixed = backfill_short_rows(&mut pairs, &points, k);
        assert!(fixed >= 1, "backfill should have touched at least one row");

        let (idx, _): (Vec<i64>, Vec<f64>) = nblast_knn(
            points,
            vects,
            None,
            KnnOpts {
                nblast: opts_for(&smat),
                k,
                n_candidates: 4,
                voxel: 20.0,
                n_dirs: 3,
                splat: true,
                symmetry: Symmetry::Mean,
            },
        );
        for i in 0..n {
            let filled = idx[i * k..(i + 1) * k].iter().filter(|&&v| v >= 0).count();
            assert_eq!(filled, k.min(n - 1), "row {i} short: {filled}");
        }
    }
}

#[cfg(test)]
mod rect_tests {
    use super::tests_support::*;
    use super::*;
    use crate::nblast::{load_smat, nblast_query_target, Smat};

    fn rect_opts<'a>(smat: &'a Smat, k: usize, nc: usize, sym: Symmetry) -> KnnOpts<'a> {
        KnnOpts {
            nblast: opts_for(smat),
            k,
            n_candidates: nc,
            voxel: 20.0,
            n_dirs: 3,
            splat: true,
            symmetry: sym,
        }
    }

    /// A grid so coarse every neuron shares one feature -> exhaustive candidates.
    fn exhaustive_rect<'a>(smat: &'a Smat, k: usize, nt: usize, sym: Symmetry) -> KnnOpts<'a> {
        let mut o = rect_opts(smat, k, nt, sym);
        o.voxel = 1.0e6;
        o.n_dirs = 1;
        o.splat = false;
        o
    }

    /// Top-k of a dense row-major `nq x nt` matrix (and its transpose) under `sym`.
    fn dense_rect_topk(
        fwd: &[f64],
        rev: &[f64],
        nq: usize,
        nt: usize,
        k: usize,
        sym: Symmetry,
    ) -> Vec<Vec<(usize, f64)>> {
        (0..nq)
            .map(|i| {
                let mut row: Vec<(usize, f64)> = (0..nt)
                    .map(|j| (j, sym.combine(fwd[i * nt + j], rev[j * nq + i])))
                    .collect();
                row.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                row.truncate(k);
                row
            })
            .collect()
    }

    fn split(seed: u64) -> (Vec<Vec<[f64; 3]>>, Vec<Vec<[f64; 3]>>, Vec<Vec<[f64; 3]>>, Vec<Vec<[f64; 3]>>) {
        let (points, vects) = population(4, 4, 35, seed);
        let cut = 6;
        let (qp, tp) = (points[..cut].to_vec(), points[cut..].to_vec());
        let (qv, tv) = (vects[..cut].to_vec(), vects[cut..].to_vec());
        (qp, qv, tp, tv)
    }

    #[test]
    fn exhaustive_rect_reproduces_dense_query_target() {
        let (qp, qv, tp, tv) = split(31);
        let (nq, nt) = (qp.len(), tp.len());
        let smat = load_smat();
        let k = 4;
        let fwd: Vec<f64> = nblast_query_target(
            qp.clone(), qv.clone(), None, tp.clone(), tv.clone(), None, opts_for(&smat),
        );
        let rev: Vec<f64> = nblast_query_target(
            tp.clone(), tv.clone(), None, qp.clone(), qv.clone(), None, opts_for(&smat),
        );

        for sym in [Symmetry::Forward, Symmetry::Mean, Symmetry::Min, Symmetry::Max] {
            let (idx, sc): (Vec<i64>, Vec<f64>) = nblast_knn_query_target(
                qp.clone(), qv.clone(), None, tp.clone(), tv.clone(), None,
                exhaustive_rect(&smat, k, nt, sym),
            );
            let want = dense_rect_topk(&fwd, &rev, nq, nt, k, sym);
            for i in 0..nq {
                for c in 0..k {
                    let (wi, ws) = want[i][c];
                    assert!(
                        (sc[i * k + c] - ws).abs() < 1e-12,
                        "{sym:?} row {i} slot {c}: {} vs dense {ws}",
                        sc[i * k + c]
                    );
                    if c == 0 || (want[i][c - 1].1 - ws).abs() > 1e-12 {
                        assert_eq!(idx[i * k + c], wi as i64, "{sym:?} row {i} slot {c}");
                    }
                }
            }
        }
    }

    #[test]
    fn rect_scores_are_exact_even_when_approximate() {
        let (qp, qv, tp, tv) = split(37);
        let (nq, nt) = (qp.len(), tp.len());
        let smat = load_smat();
        let k = 3;
        let fwd: Vec<f64> = nblast_query_target(
            qp.clone(), qv.clone(), None, tp.clone(), tv.clone(), None, opts_for(&smat),
        );
        let rev: Vec<f64> = nblast_query_target(
            tp.clone(), tv.clone(), None, qp.clone(), qv.clone(), None, opts_for(&smat),
        );
        let (idx, sc): (Vec<i64>, Vec<f64>) = nblast_knn_query_target(
            qp, qv, None, tp, tv, None, rect_opts(&smat, k, 2, Symmetry::Mean),
        );
        for i in 0..nq {
            for c in 0..k {
                let j = idx[i * k + c];
                if j < 0 {
                    continue;
                }
                let j = j as usize;
                let want = 0.5 * (fwd[i * nt + j] + rev[j * nq + i]);
                assert!((sc[i * k + c] - want).abs() < 1e-12, "row {i} -> {j}");
            }
        }
    }

    #[test]
    fn rect_keeps_a_neuron_present_in_both_sets() {
        // Unlike the square form, nothing is excluded: a neuron in both sets is a
        // legitimate self-match at 1.0, matching `nblast_query_target`.
        let (points, vects) = population(3, 3, 30, 41);
        let smat = load_smat();
        let (idx, sc): (Vec<i64>, Vec<f64>) = nblast_knn_query_target(
            points.clone(), vects.clone(), None, points.clone(), vects.clone(), None,
            exhaustive_rect(&smat, 1, points.len(), Symmetry::Mean),
        );
        for i in 0..points.len() {
            assert_eq!(idx[i], i as i64, "row {i} should match itself first");
            assert!((sc[i] - 1.0).abs() < 1e-9, "row {i} self score {}", sc[i]);
        }
    }

    #[test]
    fn rect_pads_and_backfills() {
        let (qp, qv, tp, tv) = split(43);
        let nt = tp.len();
        let smat = load_smat();
        let k = nt + 3; // more neighbours than targets exist
        let (idx, sc): (Vec<i64>, Vec<f64>) = nblast_knn_query_target(
            qp.clone(), qv, None, tp, tv, None, rect_opts(&smat, k, nt, Symmetry::Mean),
        );
        for i in 0..qp.len() {
            let filled = idx[i * k..(i + 1) * k].iter().filter(|&&v| v >= 0).count();
            assert_eq!(filled, nt, "row {i} should be full to nt");
            for c in nt..k {
                assert_eq!(idx[i * k + c], -1);
                assert!(sc[i * k + c].is_infinite());
            }
        }
    }

    #[test]
    fn rect_thread_count_does_not_change_the_answer() {
        let (qp, qv, tp, tv) = split(47);
        let smat = load_smat();
        let mut o = exhaustive_rect(&smat, 3, tp.len(), Symmetry::Min);
        let (i1, s1): (Vec<i64>, Vec<f64>) = nblast_knn_query_target(
            qp.clone(), qv.clone(), None, tp.clone(), tv.clone(), None, o,
        );
        o.nblast.threads = Some(1);
        let (i2, s2): (Vec<i64>, Vec<f64>) =
            nblast_knn_query_target(qp, qv, None, tp, tv, None, o);
        assert_eq!((i1, s1), (i2, s2));
    }

    #[test]
    fn rect_empty_sides() {
        let smat = load_smat();
        let (points, vects) = population(2, 2, 20, 53);
        let o = rect_opts(&smat, 5, 4, Symmetry::Mean);
        // No targets: every row is padding, but the shape still holds.
        let (idx, _): (Vec<i64>, Vec<f64>) = nblast_knn_query_target(
            points.clone(), vects.clone(), None, Vec::new(), Vec::new(), None, o,
        );
        assert_eq!(idx.len(), points.len() * 5);
        assert!(idx.iter().all(|&v| v == -1));
        // No queries: empty output.
        let (idx, _): (Vec<i64>, Vec<f64>) =
            nblast_knn_query_target(Vec::new(), Vec::new(), None, points, vects, None, o);
        assert!(idx.is_empty());
    }
}
