//! Primitives on raw 3D point clouds.
//!
//! [`dotprops`] turns a point cloud into the *dotprops* representation NBLAST consumes: a unit
//! tangent vector and an anisotropy value `alpha` per point, both derived from the point's local
//! `k`-neighbourhood. `nblast` has always taken those arrays as given, leaving callers to
//! produce them with `scipy.spatial.cKDTree` plus `N` 3x3 SVDs — which is single-threaded, the
//! only reason `navis_fastcore` would need scipy at all, and, once the rest of a skeletonisation
//! pipeline is in Rust, the dominant remaining cost.
//!
//! Both halves are cheap done properly: an exact k-d tree k-NN ([`crate::kdtree`]) and a
//! closed-ish-form symmetric 3x3 eigendecomposition, over points that never interact, so the
//! whole thing is one embarrassingly parallel pass.

use crate::kdtree::KdTree;
use crate::nblast::with_pool;
use ndarray::{Array1, Array2, ArrayView2};
use rayon::prelude::*;

/// Eigenvalues (descending) and eigenvectors (columns, matching order) of a symmetric 3x3
/// matrix, by cyclic Jacobi rotation.
///
/// Jacobi rather than the trigonometric closed form: the closed form loses most of its
/// precision exactly where it matters here — a near-degenerate neighbourhood, where two
/// eigenvalues nearly coincide and the eigenvector is what we are after. Jacobi is
/// unconditionally stable, and on a 3x3 it converges in a handful of sweeps, which is noise next
/// to the k-NN that produced the matrix.
fn eigh3(mut a: [[f64; 3]; 3]) -> ([f64; 3], [[f64; 3]; 3]) {
    let mut v = [[0.0f64; 3]; 3];
    for (i, row) in v.iter_mut().enumerate() {
        row[i] = 1.0;
    }

    // Convergence is quadratic once the off-diagonal is small; the cap is a backstop, not the
    // expected exit.
    for _ in 0..24 {
        let off = a[0][1].abs() + a[0][2].abs() + a[1][2].abs();
        let diag = a[0][0].abs() + a[1][1].abs() + a[2][2].abs();
        // Scale-relative: an all-zero matrix (coincident points) exits on the first check.
        if off <= f64::EPSILON * diag {
            break;
        }
        for (p, q) in [(0usize, 1usize), (0, 2), (1, 2)] {
            let apq = a[p][q];
            if apq == 0.0 {
                continue;
            }
            // The rotation that zeroes a[p][q]. `t` is the smaller of the two roots, which is
            // what keeps the rotation angle under 45 degrees and the iteration contracting.
            let theta = (a[q][q] - a[p][p]) / (2.0 * apq);
            let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
            let c = 1.0 / (t * t + 1.0).sqrt();
            let s = t * c;

            let (app, aqq) = (a[p][p], a[q][q]);
            a[p][p] = app - t * apq;
            a[q][q] = aqq + t * apq;
            a[p][q] = 0.0;
            a[q][p] = 0.0;
            // The one remaining index: rotate its two off-diagonal entries.
            let r = 3 - p - q;
            let (arp, arq) = (a[r][p], a[r][q]);
            a[r][p] = c * arp - s * arq;
            a[p][r] = a[r][p];
            a[r][q] = s * arp + c * arq;
            a[q][r] = a[r][q];

            for row in v.iter_mut() {
                let (vp, vq) = (row[p], row[q]);
                row[p] = c * vp - s * vq;
                row[q] = s * vp + c * vq;
            }
        }
    }

    // Sort descending by eigenvalue. Three elements — a hand-rolled sort of the (value, column)
    // pairs beats setting up any general machinery, and keeps the columns attached.
    let mut order = [0usize, 1, 2];
    order.sort_by(|&i, &j| a[j][j].total_cmp(&a[i][i]));
    let vals = [
        a[order[0]][order[0]],
        a[order[1]][order[1]],
        a[order[2]][order[2]],
    ];
    let mut vecs = [[0.0f64; 3]; 3];
    for (col, &src) in order.iter().enumerate() {
        for row in 0..3 {
            vecs[row][col] = v[row][src];
        }
    }
    (vals, vecs)
}

/// Tangent vectors and alpha values for a point cloud.
///
/// For each point, takes its `k` nearest neighbours (**the point itself included**, matching
/// `scipy.spatial.cKDTree.query`, which returns the query point as its own first neighbour),
/// forms the scatter matrix of that neighbourhood about its centroid, and returns the principal
/// direction plus a measure of how elongated the neighbourhood is.
///
/// Arguments
/// ---------
/// - `points`: (N, 3) point coordinates.
/// - `k`: Number of nearest neighbours *including the point itself* — `k = 20` uses 19 other
///   points. Clamped to `N`.
/// - `threads`: Size of the rayon pool. `None` => the global pool.
///
/// Returns
/// -------
/// - `vect`: (N, 3) principal direction of each neighbourhood, unit length.
/// - `alpha`: (N,) `(l1 - l2) / (l1 + l2 + l3)` for scatter-matrix eigenvalues
///   `l1 >= l2 >= l3`: 0 for an isotropic neighbourhood, 1 for a perfectly collinear one.
///
/// Conventions
/// -----------
/// - **The sign of `vect` is arbitrary** — an eigenvector is only defined up to sign, and NBLAST
///   scores on `|dot|` so it never observes the choice. It is nonetheless made deterministic
///   here (the largest-magnitude component is forced positive), so results are reproducible run
///   to run and across thread counts.
/// - The scatter matrix is un-normalised (`cptᵀ cpt`, no `/k`), matching navis. `alpha` is a
///   ratio of eigenvalues so the normalisation cancels, and `vect` is unaffected either way.
/// - **Degenerate neighbourhoods** — all-coincident points, or `N < 2` — have `l1 = l2 = l3 = 0`
///   and hence no defined direction and an `alpha` of `0/0`. They get `alpha = 0` and the unit
///   vector `[1, 0, 0]`, rather than the `NaN`s that would silently poison every downstream
///   NBLAST score.
///
/// # Panics
///
/// If `points` is not `(N, 3)`, or holds a non-finite coordinate (which would make the
/// neighbour ordering meaningless rather than merely wrong).
pub fn dotprops(
    points: ArrayView2<f64>,
    k: usize,
    threads: Option<usize>,
) -> (Array2<f64>, Array1<f64>) {
    assert_eq!(points.ncols(), 3, "`points` must have shape (N, 3)");
    let n = points.nrows();

    let pts: Vec<[f64; 3]> = points
        .rows()
        .into_iter()
        .map(|p| {
            let p = [p[0], p[1], p[2]];
            assert!(
                p.iter().all(|c| c.is_finite()),
                "`points` must be finite, got {p:?}"
            );
            p
        })
        .collect();

    let mut vect = Array2::<f64>::zeros((n, 3));
    let mut alpha = Array1::<f64>::zeros(n);
    if n == 0 {
        return (vect, alpha);
    }
    // scipy clamps rather than erroring, and so must we: a cloud with fewer points than `k` is
    // ordinary, not exceptional.
    let k = k.clamp(1, n);

    let tree = KdTree::build(pts.iter().copied().zip(0..).collect());

    with_pool(threads, || {
        let vflat = vect.as_slice_mut().expect("freshly allocated, contiguous");
        let aflat = alpha.as_slice_mut().expect("freshly allocated, contiguous");

        // One chunk per worker with one reusable neighbour buffer each, rather than
        // `par_iter().map_init(..)`, which rayon calls once per work-split and not once per
        // thread — the same trap `mesh::geodesic_matrix_impl` documents.
        let n_chunks = rayon::current_num_threads().max(1);
        let chunk = n.div_ceil(n_chunks).max(1);

        vflat
            .par_chunks_mut(chunk * 3)
            .zip(aflat.par_chunks_mut(chunk))
            .zip(pts.par_chunks(chunk))
            .for_each(|((vblock, ablock), qs)| {
                let mut nbrs: Vec<(f64, u32)> = Vec::with_capacity(k);

                for ((v, a), q) in vblock.chunks_mut(3).zip(ablock.iter_mut()).zip(qs) {
                    tree.knn(q, k, &mut nbrs);

                    // Centroid, then the scatter matrix about it. Accumulating both in one pass
                    // over the (tiny, cache-resident) neighbourhood would need the centroid up
                    // front; two passes over ~20 points is cheaper than the numerically shakier
                    // sum-of-squares-minus-square-of-sums alternative.
                    let inv = 1.0 / nbrs.len() as f64;
                    let mut c = [0.0f64; 3];
                    for &(_, i) in nbrs.iter() {
                        let p = &pts[i as usize];
                        for d in 0..3 {
                            c[d] += p[d];
                        }
                    }
                    for x in c.iter_mut() {
                        *x *= inv;
                    }

                    let mut m = [[0.0f64; 3]; 3];
                    for &(_, i) in nbrs.iter() {
                        let p = &pts[i as usize];
                        let d = [p[0] - c[0], p[1] - c[1], p[2] - c[2]];
                        for r in 0..3 {
                            for s in r..3 {
                                m[r][s] += d[r] * d[s];
                            }
                        }
                    }
                    // Mirror the upper triangle down. Spelled out rather than looped: three
                    // assignments, and the eigensolver needs the full matrix.
                    m[1][0] = m[0][1];
                    m[2][0] = m[0][2];
                    m[2][1] = m[1][2];

                    let (vals, vecs) = eigh3(m);
                    // The scatter matrix is positive semi-definite, so a negative eigenvalue is
                    // pure round-off; clamping keeps `alpha` inside [0, 1] without a second
                    // clamp on the ratio.
                    let (l1, l2, l3) = (vals[0].max(0.0), vals[1].max(0.0), vals[2].max(0.0));
                    let sum = l1 + l2 + l3;
                    *a = if sum > 0.0 { (l1 - l2) / sum } else { 0.0 };

                    let mut e = [vecs[0][0], vecs[1][0], vecs[2][0]];
                    if sum == 0.0 {
                        // No direction exists; pick one rather than emit NaN or an arbitrary
                        // leftover rotation.
                        e = [1.0, 0.0, 0.0];
                    }
                    let norm = (e[0] * e[0] + e[1] * e[1] + e[2] * e[2]).sqrt();
                    if norm > 0.0 {
                        for x in e.iter_mut() {
                            *x /= norm;
                        }
                    }
                    // Fix the sign: largest-magnitude component positive, first one winning a
                    // magnitude tie. Arbitrary, but stable.
                    let big = (0..3)
                        .max_by(|&i, &j| e[i].abs().total_cmp(&e[j].abs()))
                        .unwrap();
                    if e[big] < 0.0 {
                        for x in e.iter_mut() {
                            *x = -*x;
                        }
                    }
                    v.copy_from_slice(&e);
                }
            });
    });

    (vect, alpha)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array2};

    fn cloud(n: usize, seed: u64) -> Array2<f64> {
        let mut s = seed;
        let mut next = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f64) / ((1u64 << 31) as f64)
        };
        Array2::from_shape_fn((n, 3), |_| next())
    }

    /// The scatter matrix of an explicit neighbourhood, the naive way.
    fn scatter(pts: &[[f64; 3]]) -> [[f64; 3]; 3] {
        let inv = 1.0 / pts.len() as f64;
        let mut c = [0.0; 3];
        for p in pts {
            for d in 0..3 {
                c[d] += p[d] * inv;
            }
        }
        let mut m = [[0.0; 3]; 3];
        for p in pts {
            let d = [p[0] - c[0], p[1] - c[1], p[2] - c[2]];
            for r in 0..3 {
                for s in 0..3 {
                    m[r][s] += d[r] * d[s];
                }
            }
        }
        m
    }

    #[test]
    fn eigh3_reproduces_a_known_decomposition() {
        // Diagonal with a known spread, rotated: eigenvalues must survive the rotation.
        let (c, s) = (0.6f64, 0.8f64); // exact 3-4-5 rotation, so no fuzz from the setup
        let r = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]];
        let d = [3.0f64, 2.0, 5.0];
        let mut a = [[0.0f64; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                for t in 0..3 {
                    a[i][j] += r[i][t] * d[t] * r[j][t];
                }
            }
        }
        let (vals, vecs) = eigh3(a);
        assert!((vals[0] - 5.0).abs() < 1e-12);
        assert!((vals[1] - 3.0).abs() < 1e-12);
        assert!((vals[2] - 2.0).abs() < 1e-12);

        // A * v == l * v for each column.
        for col in 0..3 {
            let v = [vecs[0][col], vecs[1][col], vecs[2][col]];
            for i in 0..3 {
                let av: f64 = (0..3).map(|j| a[i][j] * v[j]).sum();
                assert!((av - vals[col] * v[i]).abs() < 1e-11);
            }
        }
    }

    #[test]
    fn a_line_is_perfectly_anisotropic() {
        // Points along x: alpha == 1 (l2 = l3 = 0) and the tangent is +x.
        let pts = Array2::from_shape_fn((50, 3), |(i, d)| if d == 0 { i as f64 } else { 0.0 });
        let (vect, alpha) = dotprops(pts.view(), 5, Some(1));
        for i in 0..50 {
            assert!((alpha[i] - 1.0).abs() < 1e-9, "alpha[{i}] = {}", alpha[i]);
            assert!((vect[[i, 0]].abs() - 1.0).abs() < 1e-9);
            assert!(vect[[i, 1]].abs() < 1e-9 && vect[[i, 2]].abs() < 1e-9);
        }
    }

    #[test]
    fn a_regular_grid_is_isotropic() {
        // A 3D lattice: every neighbourhood is symmetric, so l1 == l2 and alpha == 0.
        let mut pts = Vec::new();
        for i in 0..5 {
            for j in 0..5 {
                for l in 0..5 {
                    pts.push([i as f64, j as f64, l as f64]);
                }
            }
        }
        let arr = Array2::from_shape_vec((125, 3), pts.concat()).unwrap();
        let (_, alpha) = dotprops(arr.view(), 7, None);
        // The interior point (2,2,2) sits at index 2*25 + 2*5 + 2.
        assert!(alpha[62] < 1e-9, "interior alpha = {}", alpha[62]);
    }

    #[test]
    fn matches_a_brute_force_neighbourhood_decomposition() {
        // The oracle: for every point, find its k neighbours by exhaustive search and decompose
        // that neighbourhood directly. This is the same computation navis does with
        // `cKDTree.query` + `np.linalg.svd`, only spelled out.
        let n = 200;
        let arr = cloud(n, 42);
        let pts: Vec<[f64; 3]> = (0..n)
            .map(|i| [arr[[i, 0]], arr[[i, 1]], arr[[i, 2]]])
            .collect();
        let k = 12;
        let (vect, alpha) = dotprops(arr.view(), k, None);

        for i in 0..n {
            let mut order: Vec<(f64, usize)> = (0..n)
                .map(|j| {
                    let d = [
                        pts[j][0] - pts[i][0],
                        pts[j][1] - pts[i][1],
                        pts[j][2] - pts[i][2],
                    ];
                    (d[0] * d[0] + d[1] * d[1] + d[2] * d[2], j)
                })
                .collect();
            order.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let nbhd: Vec<[f64; 3]> = order[..k].iter().map(|&(_, j)| pts[j]).collect();

            let (vals, vecs) = eigh3(scatter(&nbhd));
            let want_alpha = (vals[0] - vals[1]) / (vals[0] + vals[1] + vals[2]);
            assert!(
                (alpha[i] - want_alpha).abs() < 1e-9,
                "point {i}: alpha {} vs {want_alpha}",
                alpha[i]
            );

            // Direction up to sign.
            let dot: f64 = (0..3).map(|d| vect[[i, d]] * vecs[d][0]).sum();
            assert!(dot.abs() > 1.0 - 1e-7, "point {i}: |dot| = {}", dot.abs());
        }
    }

    #[test]
    fn vectors_are_unit_length_and_sign_stable() {
        let arr = cloud(300, 9);
        let (v1, a1) = dotprops(arr.view(), 20, Some(1));
        let (v4, a4) = dotprops(arr.view(), 20, Some(4));
        assert_eq!(v1, v4, "thread count must not change the result");
        assert_eq!(a1, a4);

        for i in 0..300 {
            let n: f64 = (0..3).map(|d| v1[[i, d]] * v1[[i, d]]).sum::<f64>().sqrt();
            assert!((n - 1.0).abs() < 1e-12, "point {i}: |v| = {n}");
            // The sign convention: the largest-magnitude component is positive.
            let big = (0..3)
                .max_by(|&x, &y| v1[[i, x]].abs().total_cmp(&v1[[i, y]].abs()))
                .unwrap();
            assert!(v1[[i, big]] > 0.0);
        }
    }

    #[test]
    fn k_is_clamped_and_includes_the_point_itself() {
        // Three collinear points with k = 2: each point's neighbourhood is itself plus its one
        // nearest neighbour, which is still a line, so alpha stays 1.
        let pts = array![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
        let (_, alpha) = dotprops(pts.view(), 2, None);
        assert!(alpha.iter().all(|&a| (a - 1.0).abs() < 1e-12));

        // k = 99 clamps to 3 and must not panic.
        let (_, alpha) = dotprops(pts.view(), 99, None);
        assert!(alpha.iter().all(|&a| (a - 1.0).abs() < 1e-12));
    }

    #[test]
    fn degenerate_clouds_give_zero_alpha_not_nan() {
        // All-coincident points: no direction exists.
        let pts = Array2::<f64>::zeros((5, 3));
        let (vect, alpha) = dotprops(pts.view(), 3, None);
        assert!(alpha.iter().all(|&a| a == 0.0));
        for i in 0..5 {
            assert_eq!([vect[[i, 0]], vect[[i, 1]], vect[[i, 2]]], [1.0, 0.0, 0.0]);
        }

        // A single point — k clamps to 1, so the neighbourhood is the point alone.
        let one = array![[3.0, 4.0, 5.0]];
        let (vect, alpha) = dotprops(one.view(), 20, None);
        assert_eq!(alpha[0], 0.0);
        assert_eq!([vect[[0, 0]], vect[[0, 1]], vect[[0, 2]]], [1.0, 0.0, 0.0]);

        // No points at all.
        let (vect, alpha) = dotprops(Array2::<f64>::zeros((0, 3)).view(), 5, None);
        assert_eq!(vect.shape(), &[0, 3]);
        assert_eq!(alpha.len(), 0);
    }
}
