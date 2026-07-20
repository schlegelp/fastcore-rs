//! Thin-plate spline (TPS) transforms: fitting a spline to landmark pairs and
//! applying it to points.
//!
//! A thin-plate spline interpolates one set of landmarks onto another while
//! minimising the integral bending norm (Bookstein 1989). In connectomics it is how
//! *Drosophila* template spaces are bridged and mirrored when no image registration
//! exists — [`navis-flybrains`](https://github.com/navis-org/navis-flybrains) ships
//! dozens of these, with 70 to ~3400 landmarks each.
//!
//! - [`TpsTransform::fit`] solves for the spline coefficients from source/target
//!   landmarks. Done once per registration.
//! - [`TpsTransform::xform`] applies it to points. Done many times, and dominates.
//!
//! # Why this module exists in Rust
//!
//! The reference implementation (`morphops` + numpy, via `navis.transforms.TPStransform`)
//! evaluates the warp as `P @ A + U @ W`, where `U` is the `(N, M)` matrix of distances
//! from every point to every landmark. That matrix is the whole problem: at 1M points and
//! 3400 landmarks it is **27 GB**, which is why the caller is forced to batch, and why
//! even batched the cost is memory traffic rather than arithmetic — `U` is written out to
//! DRAM by `cdist` and immediately read back by the matmul.
//!
//! `U` never needs to exist. Each output row depends only on its own row of `U`, so the
//! distance and its contribution to the output can be fused into one pass:
//!
//! ```text
//! out_i = A_0 + x_i·A_1 + y_i·A_2 + z_i·A_3  +  Σ_j ||p_i - X_j|| · W_j
//! ```
//!
//! Three accumulators in registers, no intermediate allocation, and the loop over points
//! is embarrassingly parallel. Landmarks and weights are stored **structure-of-arrays** so
//! the inner loop over `j` vectorises without shuffles.
//!
//! # Fidelity
//!
//! `fit` reproduces `morphops.tps_coefs`: the same saddle-point system solved by Gaussian
//! elimination with partial pivoting, which is what LAPACK's `dgesv` (behind
//! `numpy.linalg.solve`) does. Agreement with navis is checked in the test module.

use std::fmt;
use std::sync::atomic::AtomicBool;

use ndarray::{Array2, ArrayView2};
use rayon::prelude::*;

use crate::nblast::{is_cancelled, with_pool};

/// Points per parallel work unit. Large enough that the per-chunk bookkeeping and the
/// cancellation check disappear against `CHUNK * M` inner iterations, small enough that
/// 14 cores still get even work at the smallest sizes we care about.
const CHUNK: usize = 256;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TpsError {
    /// Source and target landmark counts differ.
    CountMismatch { source: usize, target: usize },
    /// A landmark or point array was not `(N, 3)`.
    NotThreeD { got: usize },
    /// Fewer than 4 landmarks: the affine part alone is underdetermined.
    TooFewLandmarks { got: usize },
    /// The interpolation system is singular — almost always duplicate or coplanar
    /// landmarks.
    Singular,
}

impl fmt::Display for TpsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TpsError::CountMismatch { source, target } => write!(
                f,
                "number of source landmarks ({source}) must match number of target landmarks ({target})"
            ),
            TpsError::NotThreeD { got } => {
                write!(f, "expected an (N, 3) array, got (N, {got})")
            }
            TpsError::TooFewLandmarks { got } => {
                write!(f, "need at least 4 landmarks to fit a 3D thin-plate spline, got {got}")
            }
            TpsError::Singular => write!(
                f,
                "thin-plate spline system is singular; check for duplicate or coplanar landmarks"
            ),
        }
    }
}

impl std::error::Error for TpsError {}

/// A fitted thin-plate spline, ready to apply to points.
///
/// Holds the source landmarks and the coefficients — the target landmarks are not needed
/// after the fit.
#[derive(Debug, Clone)]
pub struct TpsTransform {
    /// Source landmarks, structure-of-arrays for the inner loop.
    sx: Vec<f64>,
    sy: Vec<f64>,
    sz: Vec<f64>,
    /// Non-affine weights `W`, `(M, 3)`, structure-of-arrays.
    wx: Vec<f64>,
    wy: Vec<f64>,
    wz: Vec<f64>,
    /// Affine part `A`, `(4, 3)`: row 0 is the translation, rows 1-3 the linear map.
    a: [[f64; 3]; 4],
}

impl TpsTransform {
    /// Fit the spline that maps `source` onto `target`.
    ///
    /// Both must be `(M, 3)` with the same `M`. This is the expensive-but-once half:
    /// it solves an `(M+4)` square system, so it is cubic in the landmark count.
    pub fn fit(source: ArrayView2<f64>, target: ArrayView2<f64>) -> Result<Self, TpsError> {
        if source.ncols() != 3 {
            return Err(TpsError::NotThreeD { got: source.ncols() });
        }
        if target.ncols() != 3 {
            return Err(TpsError::NotThreeD { got: target.ncols() });
        }
        if source.nrows() != target.nrows() {
            return Err(TpsError::CountMismatch {
                source: source.nrows(),
                target: target.nrows(),
            });
        }
        let m = source.nrows();
        if m < 4 {
            return Err(TpsError::TooFewLandmarks { got: m });
        }

        let (sx, sy, sz) = split_xyz(source);

        // The saddle-point system L = [[K, P], [P^T, 0]] with K_ij = ||X_i - X_j||
        // (the 3D radial basis; in 2D it would be r^2 log r^2) and P = [1, x, y, z].
        let n = m + 4;
        let mut l = vec![0.0_f64; n * n];
        for i in 0..m {
            let (xi, yi, zi) = (sx[i], sy[i], sz[i]);
            let row = &mut l[i * n..i * n + n];
            for j in 0..m {
                let dx = xi - sx[j];
                let dy = yi - sy[j];
                let dz = zi - sz[j];
                row[j] = (dx * dx + dy * dy + dz * dz).sqrt();
            }
            row[m] = 1.0;
            row[m + 1] = xi;
            row[m + 2] = yi;
            row[m + 3] = zi;
            // P^T block, written transposed into the lower-left.
            l[m * n + i] = 1.0;
            l[(m + 1) * n + i] = xi;
            l[(m + 2) * n + i] = yi;
            l[(m + 3) * n + i] = zi;
        }

        // Right-hand side [Y; 0], solved for all three coordinate columns at once.
        let mut rhs = vec![0.0_f64; n * 3];
        for i in 0..m {
            rhs[i * 3] = target[[i, 0]];
            rhs[i * 3 + 1] = target[[i, 1]];
            rhs[i * 3 + 2] = target[[i, 2]];
        }

        let piv = lu_factor(&mut l, n)?;
        lu_solve(&l, n, &piv, &mut rhs, 3);

        let mut wx = Vec::with_capacity(m);
        let mut wy = Vec::with_capacity(m);
        let mut wz = Vec::with_capacity(m);
        for i in 0..m {
            wx.push(rhs[i * 3]);
            wy.push(rhs[i * 3 + 1]);
            wz.push(rhs[i * 3 + 2]);
        }
        let mut a = [[0.0_f64; 3]; 4];
        for (r, row) in a.iter_mut().enumerate() {
            for (c, v) in row.iter_mut().enumerate() {
                *v = rhs[(m + r) * 3 + c];
            }
        }

        Ok(TpsTransform {
            sx,
            sy,
            sz,
            wx,
            wy,
            wz,
            a,
        })
    }

    /// Build a transform from coefficients that were fitted elsewhere.
    ///
    /// Exists so a caller holding `W`/`A` from another implementation can use the fast
    /// `xform` without refitting.
    pub fn from_coefs(
        source: ArrayView2<f64>,
        w: ArrayView2<f64>,
        a: ArrayView2<f64>,
    ) -> Result<Self, TpsError> {
        if source.ncols() != 3 {
            return Err(TpsError::NotThreeD { got: source.ncols() });
        }
        if w.ncols() != 3 {
            return Err(TpsError::NotThreeD { got: w.ncols() });
        }
        if source.nrows() != w.nrows() {
            return Err(TpsError::CountMismatch {
                source: source.nrows(),
                target: w.nrows(),
            });
        }
        if a.nrows() != 4 || a.ncols() != 3 {
            return Err(TpsError::NotThreeD { got: a.ncols() });
        }
        let (sx, sy, sz) = split_xyz(source);
        let (wx, wy, wz) = split_xyz(w);
        let mut arr = [[0.0_f64; 3]; 4];
        for (r, row) in arr.iter_mut().enumerate() {
            for (c, v) in row.iter_mut().enumerate() {
                *v = a[[r, c]];
            }
        }
        Ok(TpsTransform {
            sx,
            sy,
            sz,
            wx,
            wy,
            wz,
            a: arr,
        })
    }

    /// Number of landmarks the spline was fitted on.
    pub fn n_landmarks(&self) -> usize {
        self.sx.len()
    }

    /// The source landmarks, as an `(M, 3)` array.
    pub fn source(&self) -> Array2<f64> {
        join_xyz(&self.sx, &self.sy, &self.sz)
    }

    /// The non-affine weights `W`, as an `(M, 3)` array.
    ///
    /// Together with [`TpsTransform::affine_coefs`] this is the whole fit, so a caller can
    /// serialise it and rebuild via [`TpsTransform::from_coefs`] instead of paying the
    /// cubic solve again.
    pub fn weights(&self) -> Array2<f64> {
        join_xyz(&self.wx, &self.wy, &self.wz)
    }

    /// The affine coefficients `A`, as a `(4, 3)` array: row 0 is the translation.
    pub fn affine_coefs(&self) -> Array2<f64> {
        Array2::from_shape_fn((4, 3), |(r, c)| self.a[r][c])
    }

    /// The affine part as a 4x4 homogeneous matrix (last row `[0, 0, 0, 1]`).
    ///
    /// This is the transform the spline converges to far from the landmarks.
    pub fn matrix_affine(&self) -> [[f64; 4]; 4] {
        let mut m = [[0.0_f64; 4]; 4];
        for r in 0..3 {
            for c in 0..3 {
                // `a` is (4, 3) with rows 1..4 holding the linear map column-wise.
                m[r][c] = self.a[c + 1][r];
            }
            m[r][3] = self.a[0][r];
        }
        m[3] = [0.0, 0.0, 0.0, 1.0];
        m
    }

    /// Apply the spline to `points`, an `(N, 3)` array. Returns an `(N, 3)` array.
    ///
    /// `threads` caps the rayon pool (`None` uses the global one). `cancel` is polled once
    /// per chunk; when it trips the remaining rows are left as written so far and the
    /// caller is expected to discard the result.
    pub fn xform(
        &self,
        points: ArrayView2<f64>,
        threads: Option<usize>,
        cancel: Option<&AtomicBool>,
    ) -> Result<Array2<f64>, TpsError> {
        if points.ncols() != 3 {
            return Err(TpsError::NotThreeD { got: points.ncols() });
        }
        let n = points.nrows();
        let mut out = Array2::<f64>::zeros((n, 3));
        if n == 0 {
            return Ok(out);
        }

        // A non-contiguous view (e.g. a sliced numpy array) would make the inner loop
        // stride unpredictably; pay one copy instead.
        let owned;
        let pts: &[f64] = match points.as_slice() {
            Some(s) => s,
            None => {
                owned = points.to_owned();
                owned.as_slice().expect("freshly owned array is contiguous")
            }
        };

        let out_slice = out.as_slice_mut().expect("freshly allocated array is contiguous");

        with_pool(threads, || {
            pts.par_chunks(CHUNK * 3)
                .zip(out_slice.par_chunks_mut(CHUNK * 3))
                .for_each(|(pin, pout)| {
                    if is_cancelled(cancel) {
                        return;
                    }
                    self.xform_chunk(pin, pout);
                });
        });

        Ok(out)
    }

    /// The fused inner loop: distance and its contribution to the output in one pass, so
    /// the `(chunk, M)` distance matrix is never materialised.
    #[inline]
    fn xform_chunk(&self, pin: &[f64], pout: &mut [f64]) {
        let (sx, sy, sz) = (&self.sx[..], &self.sy[..], &self.sz[..]);
        let m = sx.len();
        // Bind the weight slices to the same length so the bounds checks in the hot loop
        // fold away.
        let (wx, wy, wz) = (&self.wx[..m], &self.wy[..m], &self.wz[..m]);
        let (sy, sz) = (&sy[..m], &sz[..m]);
        let a = &self.a;

        for (p, o) in pin.chunks_exact(3).zip(pout.chunks_exact_mut(3)) {
            let (px, py, pz) = (p[0], p[1], p[2]);

            // Affine part first, then accumulate the non-affine sum into the same
            // registers.
            let mut ax = a[0][0] + px * a[1][0] + py * a[2][0] + pz * a[3][0];
            let mut ay = a[0][1] + px * a[1][1] + py * a[2][1] + pz * a[3][1];
            let mut az = a[0][2] + px * a[1][2] + py * a[2][2] + pz * a[3][2];

            for j in 0..m {
                let dx = px - sx[j];
                let dy = py - sy[j];
                let dz = pz - sz[j];
                // U(r) = r in 3D.
                let r = (dx * dx + dy * dy + dz * dz).sqrt();
                ax += r * wx[j];
                ay += r * wy[j];
                az += r * wz[j];
            }

            o[0] = ax;
            o[1] = ay;
            o[2] = az;
        }
    }
}

/// Reassemble three coordinate vectors into an `(N, 3)` array.
fn join_xyz(x: &[f64], y: &[f64], z: &[f64]) -> Array2<f64> {
    Array2::from_shape_fn((x.len(), 3), |(i, c)| match c {
        0 => x[i],
        1 => y[i],
        _ => z[i],
    })
}

/// Split an `(N, 3)` view into three contiguous coordinate vectors.
fn split_xyz(a: ArrayView2<f64>) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = a.nrows();
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut z = Vec::with_capacity(n);
    for row in a.rows() {
        x.push(row[0]);
        y.push(row[1]);
        z.push(row[2]);
    }
    (x, y, z)
}

/// Columns per panel in the blocked factorisation.
///
/// This is what turns the cubic term from a rank-1 update (one pass over the whole
/// trailing matrix per column, purely memory-bound) into a matrix multiply that reuses a
/// panel held in cache. At `n ~ 3400` the matrix is ~92 MB, so the unblocked form moves
/// ~200 GB and runs at DRAM speed; blocking cuts that by roughly this factor. 128 keeps
/// the `128 x n` panel comfortably inside L2.
const PANEL: usize = 128;

/// Elements of trailing submatrix below which the factorisation stays on one thread.
///
/// The updates inside `lu_factor` are dispatched once per column (panel) or once per panel
/// (trailing), so a small system spends more time in rayon's scheduler than in arithmetic —
/// at 500 landmarks, unconditionally parallelising made the fit ~7x *slower* than running
/// it serially. Below this the work simply is not worth splitting.
const PAR_MIN_WORK: usize = 48 * 1024;

/// LU-factorise `a` (`n x n`, row-major) in place with partial pivoting.
///
/// On return the unit-lower-triangular `L` occupies the strict lower triangle and `U` the
/// diagonal and above. The returned vector holds, for each step `k`, the row that was
/// swapped into position `k`.
///
/// This is the same factorisation LAPACK's `dgetrf` performs — right-looking, blocked,
/// partial pivoting — so results track `numpy.linalg.solve` to rounding.
fn lu_factor(a: &mut [f64], n: usize) -> Result<Vec<usize>, TpsError> {
    let mut piv = vec![0usize; n];

    let mut k = 0;
    while k < n {
        let nb = PANEL.min(n - k);

        // --- Panel factorisation: unblocked LU on columns k..k+nb, full rows below k.
        // Narrow, so the memory-bound rank-1 updates here are cheap; row swaps are applied
        // across the full width immediately to keep the rest of the matrix consistent.
        for jj in k..(k + nb) {
            let mut pivot = jj;
            let mut best = a[jj * n + jj].abs();
            for i in (jj + 1)..n {
                let v = a[i * n + jj].abs();
                if v > best {
                    best = v;
                    pivot = i;
                }
            }
            if best == 0.0 || !best.is_finite() {
                return Err(TpsError::Singular);
            }
            piv[jj] = pivot;
            if pivot != jj {
                for j in 0..n {
                    a.swap(jj * n + j, pivot * n + j);
                }
            }

            let inv = 1.0 / a[jj * n + jj];
            let (head, tail) = a.split_at_mut((jj + 1) * n);
            let prow = &head[jj * n..jj * n + n];
            // Only the panel's own columns are updated here; the rest of the trailing
            // matrix is handled by the block update below.
            let update = |row: &mut [f64]| {
                let f = row[jj] * inv;
                row[jj] = f;
                if f == 0.0 {
                    return;
                }
                for j in (jj + 1)..(k + nb) {
                    row[j] -= f * prow[j];
                }
            };
            if (n - jj - 1) * nb >= PAR_MIN_WORK {
                tail.par_chunks_mut(n).for_each(update);
            } else {
                tail.chunks_mut(n).for_each(update);
            }
        }

        let rest = k + nb;
        if rest >= n {
            break;
        }

        // --- Triangular solve: A12 <- L11^-1 A12, with L11 unit lower triangular.
        for jj in 0..nb {
            let (head, tail) = a.split_at_mut((k + jj + 1) * n);
            let prow = &head[(k + jj) * n..(k + jj) * n + n];
            for ii in (jj + 1)..nb {
                let row = &mut tail[(ii - jj - 1) * n..(ii - jj) * n];
                let f = row[k + jj];
                if f == 0.0 {
                    continue;
                }
                for j in rest..n {
                    row[j] -= f * prow[j];
                }
            }
        }

        // --- Trailing update: A22 <- A22 - A21 A12. The cubic term, now a matrix
        // multiply: the `nb x (n-rest)` panel A12 stays in cache and is reused by every
        // row of A22, which are independent and so distribute across cores.
        let (head, tail) = a.split_at_mut(rest * n);
        let panel = &head[k * n..rest * n];
        let update = |row: &mut [f64]| {
            for p in 0..nb {
                let f = row[k + p];
                if f == 0.0 {
                    continue;
                }
                let prow = &panel[p * n..p * n + n];
                for j in rest..n {
                    row[j] -= f * prow[j];
                }
            }
        };
        if (n - rest) * (n - rest) >= PAR_MIN_WORK {
            tail.par_chunks_mut(n).for_each(update);
        } else {
            tail.chunks_mut(n).for_each(update);
        }

        k = rest;
    }

    Ok(piv)
}

/// Solve `A X = B` given the factorisation from [`lu_factor`].
///
/// `rhs` is `n x nrhs` row-major and is overwritten with `X`. Only quadratic work, so it
/// stays serial.
fn lu_solve(a: &[f64], n: usize, piv: &[usize], rhs: &mut [f64], nrhs: usize) {
    // Replay the row interchanges.
    for k in 0..n {
        let p = piv[k];
        if p != k {
            for c in 0..nrhs {
                rhs.swap(k * nrhs + c, p * nrhs + c);
            }
        }
    }
    // Forward-substitute through L (unit diagonal).
    for k in 0..n {
        for j in 0..k {
            let f = a[k * n + j];
            if f == 0.0 {
                continue;
            }
            for c in 0..nrhs {
                rhs[k * nrhs + c] -= f * rhs[j * nrhs + c];
            }
        }
    }
    // Back-substitute through U.
    for k in (0..n).rev() {
        let diag = a[k * n + k];
        for c in 0..nrhs {
            let mut acc = rhs[k * nrhs + c];
            for j in (k + 1)..n {
                acc -= a[k * n + j] * rhs[j * nrhs + c];
            }
            rhs[k * nrhs + c] = acc / diag;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// The landmarks are reproduced exactly: a TPS interpolates its own control points.
    #[test]
    fn interpolates_landmarks() {
        let src = array![
            [0.0, 0.0, 0.0],
            [10.0, 10.0, 10.0],
            [100.0, 100.0, 100.0],
            [80.0, 10.0, 30.0],
            [5.0, 60.0, 12.0],
        ];
        let trg = array![
            [1.0, 15.0, 5.0],
            [9.0, 18.0, 21.0],
            [80.0, 99.0, 120.0],
            [5.0, 10.0, 80.0],
            [7.0, 55.0, 3.0],
        ];
        let tps = TpsTransform::fit(src.view(), trg.view()).unwrap();
        let out = tps.xform(src.view(), None, None).unwrap();
        for i in 0..src.nrows() {
            for c in 0..3 {
                assert!(
                    (out[[i, c]] - trg[[i, c]]).abs() < 1e-8,
                    "landmark {i} coord {c}: {} != {}",
                    out[[i, c]],
                    trg[[i, c]]
                );
            }
        }
    }

    /// Matches the doctest in `navis.transforms.thinplate.TPStransform`.
    #[test]
    fn matches_navis_doctest() {
        let src = array![
            [0.0, 0.0, 0.0],
            [10.0, 10.0, 10.0],
            [100.0, 100.0, 100.0],
            [80.0, 10.0, 30.0]
        ];
        let trg = array![
            [1.0, 15.0, 5.0],
            [9.0, 18.0, 21.0],
            [80.0, 99.0, 120.0],
            [5.0, 10.0, 80.0]
        ];
        let tps = TpsTransform::fit(src.view(), trg.view()).unwrap();
        let pts = array![[0.0, 0.0, 0.0], [50.0, 50.0, 50.0]];
        let out = tps.xform(pts.view(), None, None).unwrap();
        let want = [[1.0, 15.0, 5.0], [40.55555556, 54.0, 65.0]];
        for i in 0..2 {
            for c in 0..3 {
                assert!(
                    (out[[i, c]] - want[i][c]).abs() < 1e-6,
                    "point {i} coord {c}: {} != {}",
                    out[[i, c]],
                    want[i][c]
                );
            }
        }
    }

    /// An affine landmark set must produce a pure affine, with zero bending.
    #[test]
    fn recovers_affine() {
        let src = array![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
        ];
        // Scale by 2 and translate by (3, 4, 5).
        let trg = src.map(|v| v * 2.0)
            + ndarray::Array2::from_shape_fn((6, 3), |(_, c)| [3.0, 4.0, 5.0][c]);
        let tps = TpsTransform::fit(src.view(), trg.view()).unwrap();

        let pts = array![[7.0, -2.0, 0.5]];
        let out = tps.xform(pts.view(), None, None).unwrap();
        let want = [7.0 * 2.0 + 3.0, -2.0 * 2.0 + 4.0, 0.5 * 2.0 + 5.0];
        for c in 0..3 {
            assert!(
                (out[[0, c]] - want[c]).abs() < 1e-7,
                "coord {c}: {} != {}",
                out[[0, c]],
                want[c]
            );
        }
        // The non-affine weights should all be ~0 for a pure affine.
        for j in 0..tps.n_landmarks() {
            assert!(tps.wx[j].abs() < 1e-9 && tps.wy[j].abs() < 1e-9 && tps.wz[j].abs() < 1e-9);
        }
    }

    #[test]
    fn rejects_bad_shapes() {
        let a = array![[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]];
        let b = array![[0.0, 0.0, 0.0]];
        assert!(matches!(
            TpsTransform::fit(a.view(), b.view()),
            Err(TpsError::CountMismatch { .. })
        ));
        let flat = array![[0.0, 0.0], [1.0, 1.0]];
        assert!(matches!(
            TpsTransform::fit(flat.view(), flat.view()),
            Err(TpsError::NotThreeD { .. })
        ));
        assert!(matches!(
            TpsTransform::fit(a.view(), a.view()),
            Err(TpsError::TooFewLandmarks { .. })
        ));
    }

    /// Duplicate landmarks make the system singular rather than silently wrong.
    #[test]
    fn detects_singular() {
        let src = array![
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let trg = array![
            [0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let got = TpsTransform::fit(src.view(), trg.view());
        assert!(got.is_err(), "duplicate landmarks should not fit cleanly");
    }

    /// Chunk boundaries must not change results.
    #[test]
    fn chunking_is_transparent() {
        let src = array![
            [0.0, 0.0, 0.0],
            [10.0, 10.0, 10.0],
            [100.0, 100.0, 100.0],
            [80.0, 10.0, 30.0],
        ];
        let trg = array![
            [1.0, 15.0, 5.0],
            [9.0, 18.0, 21.0],
            [80.0, 99.0, 120.0],
            [5.0, 10.0, 80.0],
        ];
        let tps = TpsTransform::fit(src.view(), trg.view()).unwrap();

        let n = CHUNK * 2 + 7;
        let pts = Array2::from_shape_fn((n, 3), |(i, c)| (i as f64) * 0.37 + (c as f64) * 11.0);
        let all = tps.xform(pts.view(), None, None).unwrap();
        let one = tps.xform(pts.view(), Some(1), None).unwrap();
        for i in 0..n {
            for c in 0..3 {
                assert_eq!(all[[i, c]], one[[i, c]], "row {i} coord {c}");
            }
        }
    }
}
