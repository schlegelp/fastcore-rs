//! syNBLAST: synapse-based NBLAST.
//!
//! Where NBLAST compares neurons by their *skeleton* points and tangent vectors,
//! syNBLAST compares them by their *synapses* (connectors). For an ordered pair
//! `(query, target)` we take every query connector, find its nearest target
//! connector **of the same type** (e.g. presynapse vs presynapse), and score the
//! euclidean distance through the same lookup matrix NBLAST uses — but with the
//! dot product fixed at `1`, because synapses have no tangent vector (navis'
//! `score_fn(dist, 1)`). The per-point scores are summed and, when `normalize` is
//! set, divided by the query's self-hit so a perfect self-match is `1.0`.
//!
//! Connectors are grouped by an integer `type` id; only like-typed synapses are
//! ever compared. Passing every connector a single type id reproduces navis'
//! `by_type=False` (all synapses in one group). A query synapse whose type is
//! absent in the target is scored at the worst (farthest) distance bin, matching
//! navis' intent for an unmatched type.
//!
//! Like the NBLAST entry points this builds each neuron's per-type index once and
//! scores under `rayon`, releasing peak memory per pair rather than `O(n^2)`.

use aann::PreparedF64;
use rayon::prelude::*;

use crate::nblast::{
    build_index, index_bar, is_cancelled, run_scoring, with_pool, Opts, ScoreOut, Smat,
};

/// Group a neuron's connector points by (integer) `type`, building one `aann`
/// nearest-neighbour index per type.
///
/// The first-seen type order is preserved; the count of distinct types is tiny in
/// practice (usually 1, or pre/post = 2), so a linear scan beats a hash map. Each
/// group's points are triangulated by [`build_index`] exactly like an NBLAST
/// neuron, so tiny groups (fewer than 5 points) fall back to a complete graph and
/// remain exact.
fn build_groups(points: &[[f64; 3]], types: &[i64]) -> Vec<(i64, PreparedF64)> {
    let mut grouped: Vec<(i64, Vec<[f64; 3]>)> = Vec::new();
    for (p, &ty) in points.iter().zip(types.iter()) {
        match grouped.iter_mut().find(|(g, _)| *g == ty) {
            Some((_, pts)) => pts.push(*p),
            None => grouped.push((ty, vec![*p])),
        }
    }
    grouped
        .into_iter()
        .map(|(ty, pts)| (ty, build_index(&pts)))
        .collect()
}

/// Raw (un-normalized) forward syNBLAST score of one query neuron against one
/// target.
///
/// For every query connector we find its nearest target connector *of the same
/// type* (euclidean NN via graph descent) and look the distance up at dot product
/// `1` ([`Smat::syn_score`]); the contributions are summed. Query connectors whose
/// type is absent in the target are each scored at `syn_score(inf)` — the worst,
/// farthest bin — reproducing navis' intended behaviour for an unmatched type.
fn syn_score_pair(
    q_groups: &[(i64, PreparedF64)],
    t_groups: &[(i64, PreparedF64)],
    smat: &Smat,
) -> f64 {
    let mut raw = 0.0;
    for (ty, q_prep) in q_groups {
        match t_groups.iter().find(|(t_ty, _)| t_ty == ty) {
            Some((_, t_prep)) => {
                // Nearest target-of-same-type distance for every query point.
                let (d, _ix) = t_prep.query_prepared(q_prep, None);
                for &dist in d.as_slice().unwrap() {
                    raw += smat.syn_score(dist);
                }
            }
            None => {
                // No target connectors of this type: worst score per query point.
                raw += smat.syn_score(f64::INFINITY) * q_prep.n() as f64;
            }
        }
    }
    raw
}

/// Per-neuron self-hit: `n_connectors * score_fn(0, 1)`, matching navis'
/// `calc_self_hit`. Uses the total connector count (across types), so it is
/// identical whether or not `by_type` grouping is used.
#[inline]
fn self_hit(smat: &Smat, n_connectors: usize) -> f64 {
    smat.syn_score(0.0) * n_connectors as f64
}

/// All-by-all forward syNBLAST over `points` / `types` (one entry per neuron).
///
/// `types[i]` are the per-connector integer type ids of neuron `i` (same length as
/// `points[i]`). Returns a flat row-major `n * n` matrix where cell `[i * n + j]`
/// is the score of query `i` against target `j`; with `opts.normalize` the
/// diagonal is `1.0`. `opts.limit_dist` is not used by syNBLAST. The element type
/// `T` selects the output precision.
pub fn synblast_allbyall<T: ScoreOut>(
    points: Vec<Vec<[f64; 3]>>,
    types: Vec<Vec<i64>>,
    opts: Opts,
) -> Vec<T> {
    let Opts {
        smat,
        normalize,
        threads,
        progress,
        limit_dist: _,
        cancel,
    } = opts;

    with_pool(threads, move || {
        let n = points.len();

        // Build every neuron's per-type indices once, in parallel.
        let idx_bar = progress.then(|| index_bar(n as u64));
        let neurons: Option<Vec<Vec<(i64, PreparedF64)>>> = match &idx_bar {
            Some(bar) => points
                .par_iter()
                .zip(types.par_iter())
                .map(|(p, t)| {
                    if is_cancelled(cancel) {
                        return None;
                    }
                    let g = build_groups(p, t);
                    bar.inc(1);
                    Some(g)
                })
                .collect(),
            None => points
                .par_iter()
                .zip(types.par_iter())
                .map(|(p, t)| {
                    if is_cancelled(cancel) {
                        return None;
                    }
                    Some(build_groups(p, t))
                })
                .collect(),
        };
        let Some(neurons) = neurons else {
            return Vec::new(); // interrupted mid-build; caller discards the result
        };
        if let Some(bar) = idx_bar {
            bar.finish();
        }
        let self_hits: Vec<f64> = points.iter().map(|p| self_hit(smat, p.len())).collect();

        let mut scores: Vec<T> = vec![T::from_f64(0.0); n * n];

        let compute = |(k, out): (usize, &mut T)| {
            let i = k / n; // query neuron
            let j = k % n; // target neuron
            let s = if i == j {
                // Self-match: every connector matches itself at distance 0.
                if normalize {
                    1.0
                } else {
                    self_hits[i]
                }
            } else {
                let raw = syn_score_pair(&neurons[i], &neurons[j], smat);
                if normalize {
                    raw / self_hits[i]
                } else {
                    raw
                }
            };
            *out = T::from_f64(s);
        };

        run_scoring(&mut scores, (n * n) as u64, progress, cancel, compute);
        scores
    })
}

/// Forward syNBLAST of every query neuron against every target neuron.
///
/// Returns a flat row-major `n_query * n_target` matrix where cell
/// `[qi * n_target + tj]` is the score of query `qi` against target `tj`. Unlike
/// the all-by-all entry point there is no diagonal short-cut: query and target are
/// distinct neuron sets. The element type `T` selects the output precision.
#[allow(clippy::too_many_arguments)]
pub fn synblast_query_target<T: ScoreOut>(
    q_points: Vec<Vec<[f64; 3]>>,
    q_types: Vec<Vec<i64>>,
    t_points: Vec<Vec<[f64; 3]>>,
    t_types: Vec<Vec<i64>>,
    opts: Opts,
) -> Vec<T> {
    let Opts {
        smat,
        normalize,
        threads,
        progress,
        limit_dist: _,
        cancel,
    } = opts;

    with_pool(threads, move || {
        let nq = q_points.len();
        let nt = t_points.len();

        // One shared bar spanning both index builds so it reads as a single phase.
        let idx_bar = progress.then(|| index_bar((nq + nt) as u64));
        let build_all =
            |pts: &[Vec<[f64; 3]>], tys: &[Vec<i64>]| -> Option<Vec<Vec<(i64, PreparedF64)>>> {
                match &idx_bar {
                    Some(bar) => pts
                        .par_iter()
                        .zip(tys.par_iter())
                        .map(|(p, t)| {
                            if is_cancelled(cancel) {
                                return None;
                            }
                            let g = build_groups(p, t);
                            bar.inc(1);
                            Some(g)
                        })
                        .collect(),
                    None => pts
                        .par_iter()
                        .zip(tys.par_iter())
                        .map(|(p, t)| {
                            if is_cancelled(cancel) {
                                return None;
                            }
                            Some(build_groups(p, t))
                        })
                        .collect(),
                }
            };
        let (Some(q_neurons), Some(t_neurons)) =
            (build_all(&q_points, &q_types), build_all(&t_points, &t_types))
        else {
            return Vec::new(); // interrupted mid-build; caller discards the result
        };
        if let Some(bar) = idx_bar {
            bar.finish();
        }
        let self_hits: Vec<f64> = q_points.iter().map(|p| self_hit(smat, p.len())).collect();

        let n_cells = nq * nt;
        let mut scores: Vec<T> = vec![T::from_f64(0.0); n_cells];

        let compute = |(k, out): (usize, &mut T)| {
            let qi = k / nt;
            let tj = k % nt;
            let raw = syn_score_pair(&q_neurons[qi], &t_neurons[tj], smat);
            *out = T::from_f64(if normalize {
                raw / self_hits[qi]
            } else {
                raw
            });
        };

        run_scoring(&mut scores, n_cells as u64, progress, cancel, compute);
        scores
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nblast::load_smat;

    fn opts(smat: &Smat) -> Opts<'_> {
        Opts {
            smat,
            normalize: true,
            limit_dist: None,
            threads: None,
            progress: false,
            cancel: None,
        }
    }

    /// `syn_score` reads the last dot-product column (dot fixed at 1).
    #[test]
    fn syn_score_uses_last_dot_column() {
        let smat = load_smat();
        let ncols = smat.values.ncols();
        // Distance 0 -> first dist row, dot 1 -> last dot col.
        assert_eq!(smat.syn_score(0.0), smat.values[(0, ncols - 1)]);
        // A huge distance clamps to the last (worst) dist bin.
        let nrows = smat.values.nrows();
        assert_eq!(smat.syn_score(f64::INFINITY), smat.values[(nrows - 1, ncols - 1)]);
        // The worst bin is negative for FCWB; the self bin is positive.
        assert!(smat.syn_score(0.0) > 0.0);
        assert!(smat.syn_score(f64::INFINITY) < 0.0);
    }

    #[test]
    fn allbyall_diagonal_is_one_when_normalized() {
        // Two well-separated synapse clouds, all one type.
        let a = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 0.5],
        ];
        let b: Vec<[f64; 3]> = a.iter().map(|p| [p[0] + 50.0, p[1], p[2]]).collect();
        let points = vec![a, b];
        let types = vec![vec![0i64; 5], vec![0i64; 5]];
        let smat = load_smat();

        let m: Vec<f64> = synblast_allbyall(points, types, opts(&smat));
        assert_eq!(m.len(), 4);
        assert!((m[0] - 1.0).abs() < 1e-9, "diag[0] = {}", m[0]);
        assert!((m[3] - 1.0).abs() < 1e-9, "diag[1] = {}", m[3]);
        // Far-apart clouds score below a self-match.
        assert!(m[1] < m[0]);
        assert!(m.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn query_target_matches_allbyall_diagonal_block() {
        let a = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 0.5],
        ];
        let b: Vec<[f64; 3]> = a.iter().map(|p| [p[0] + 3.0, p[1], p[2]]).collect();
        let points = vec![a, b];
        let types = vec![vec![0i64; 5], vec![0i64; 5]];
        let smat = load_smat();

        let aba: Vec<f64> = synblast_allbyall(points.clone(), types.clone(), opts(&smat));
        // Query-target of the same set (no diagonal short-cut) must match all-by-all.
        let qt: Vec<f64> = synblast_query_target(
            points.clone(),
            types.clone(),
            points,
            types,
            opts(&smat),
        );
        for (x, y) in aba.iter().zip(qt.iter()) {
            assert!((x - y).abs() < 1e-9, "{x} vs {y}");
        }
    }

    /// With `by_type`, a query type absent in the target is scored at the worst bin;
    /// matching-type synapses at distance 0 recover the self-hit.
    #[test]
    fn by_type_matches_within_type_only() {
        // Query type 0 exists in target; query type 1 does not.
        let q_pts = vec![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [10.0, 10.0, 10.0]];
        let q_types = vec![0i64, 0, 1];
        // Target has only type-0 connectors, co-located with the query's type-0 ones.
        let t_pts = vec![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]];
        let t_types = vec![0i64, 0];
        let smat = load_smat();

        let raw: Vec<f64> = synblast_query_target(
            vec![q_pts],
            vec![q_types],
            vec![t_pts],
            vec![t_types],
            Opts {
                normalize: false,
                ..opts(&smat)
            },
        );
        // Two type-0 points match at distance 0 (self bin) + one unmatched type-1
        // point at the worst bin.
        let expect = 2.0 * smat.syn_score(0.0) + smat.syn_score(f64::INFINITY);
        assert!((raw[0] - expect).abs() < 1e-9, "{} vs {}", raw[0], expect);
    }
}
