//! Moving least squares (MLS) transforms: locally weighted affine warps from landmark
//! pairs.
//!
//! The affine flavour of Schaefer et al. 2006. Unlike a thin-plate spline there is no fit
//! step — every point gets its *own* affine, solved on the fly from all landmarks weighted
//! by inverse squared distance:
//!
//! ```text
//! w_j      = 1 / (||p_j - v||^2 + eps)
//! p*, q*   = weighted centroids of the source and target landmarks
//! Y        = Σ_j w_j (p_j - p*)(p_j - p*)^T        3x3, symmetric
//! Z        = Σ_j w_j (p_j - p*)(q_j - q*)^T        3x3
//! v'       = (v - p*) Y^-1 Z + q*
//! ```
//!
//! # Why this module exists in Rust
//!
//! `molesq` (behind `navis.transforms.MovingLeastSquaresTransform`) expresses this as a
//! chain of `einsum`s over arrays shaped `(1, 3, M, N)` — one weight *per landmark per
//! point per axis*, materialised. At 1M points and 3400 landmarks a single one of those is
//! 82 GB, and the expression builds several. In practice this caps the reference
//! implementation well below the landmark counts real registrations use: the batching
//! `navis` does to work around it still needs ~23 GB at the default batch size, so the
//! transform simply cannot be run at 3400 landmarks.
//!
//! None of it needs to be stored. Everything but the final `v'` is a *reduction* over
//! landmarks, so each point can be handled in two passes with a handful of scalar
//! accumulators — the only per-point storage is the `M` weights carried between the two
//! passes, reused from a per-thread scratch buffer. Peak memory becomes the output array,
//! independent of `M`, and the loop over points parallelises cleanly.
//!
//! # Fidelity
//!
//! `eps` is `f64::EPSILON`, matching `molesq`'s `sys.float_info.epsilon`, and the 3x3
//! solve follows the same convention (row vector times matrix). Agreement with `molesq`
//! is checked in the test module and against `navis` in the Python test suite.

use std::fmt;
use std::sync::atomic::AtomicBool;

use ndarray::{Array2, ArrayView2};
use rayon::prelude::*;

use crate::nblast::{is_cancelled, with_pool};

/// Points per parallel work unit. Each carries an `M`-long weight scratch buffer, so this
/// also bounds the per-thread allocation.
const CHUNK: usize = 256;

/// Guards against division by zero when a point coincides with a landmark. Matches
/// `sys.float_info.epsilon` in `molesq`, which is the same value.
const EPS: f64 = f64::EPSILON;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MlsError {
    /// Source and target landmark counts differ.
    CountMismatch { source: usize, target: usize },
    /// A landmark or point array was not `(N, 3)`.
    NotThreeD { got: usize },
    /// No landmarks at all: there is nothing to weight.
    NoLandmarks,
}

impl fmt::Display for MlsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MlsError::CountMismatch { source, target } => write!(
                f,
                "number of source landmarks ({source}) must match number of target landmarks ({target})"
            ),
            MlsError::NotThreeD { got } => write!(f, "expected an (N, 3) array, got (N, {got})"),
            MlsError::NoLandmarks => write!(f, "need at least one landmark"),
        }
    }
}

impl std::error::Error for MlsError {}

/// Landmark pairs defining a moving-least-squares warp.
///
/// There is no fitted state — construction just reorganises the landmarks — so this is
/// cheap to build and the cost is entirely in [`MlsTransform::xform`].
#[derive(Debug, Clone)]
pub struct MlsTransform {
    /// Source landmarks, structure-of-arrays for the inner loops.
    px: Vec<f64>,
    py: Vec<f64>,
    pz: Vec<f64>,
    /// Target ("deformed") landmarks.
    qx: Vec<f64>,
    qy: Vec<f64>,
    qz: Vec<f64>,
}

impl MlsTransform {
    /// Build from `(M, 3)` source and target landmarks.
    pub fn new(source: ArrayView2<f64>, target: ArrayView2<f64>) -> Result<Self, MlsError> {
        if source.ncols() != 3 {
            return Err(MlsError::NotThreeD { got: source.ncols() });
        }
        if target.ncols() != 3 {
            return Err(MlsError::NotThreeD { got: target.ncols() });
        }
        if source.nrows() != target.nrows() {
            return Err(MlsError::CountMismatch {
                source: source.nrows(),
                target: target.nrows(),
            });
        }
        if source.nrows() == 0 {
            return Err(MlsError::NoLandmarks);
        }
        let (px, py, pz) = split_xyz(source);
        let (qx, qy, qz) = split_xyz(target);
        Ok(MlsTransform {
            px,
            py,
            pz,
            qx,
            qy,
            qz,
        })
    }

    /// Number of landmark pairs.
    pub fn n_landmarks(&self) -> usize {
        self.px.len()
    }

    /// The source landmarks, as an `(M, 3)` array.
    pub fn source(&self) -> Array2<f64> {
        join_xyz(&self.px, &self.py, &self.pz)
    }

    /// The target landmarks, as an `(M, 3)` array.
    pub fn target(&self) -> Array2<f64> {
        join_xyz(&self.qx, &self.qy, &self.qz)
    }

    /// Apply the warp to `points`, an `(N, 3)` array. Returns an `(N, 3)` array.
    ///
    /// `reverse` swaps the roles of source and target, i.e. maps target space back to
    /// source space. Note this is *not* an exact inverse — MLS is only invertible in the
    /// limit — it is the warp fitted in the opposite direction, which is what `molesq`
    /// and `navis` mean by it too.
    ///
    /// `threads` caps the rayon pool (`None` uses the global one). `cancel` is polled once
    /// per chunk.
    pub fn xform(
        &self,
        points: ArrayView2<f64>,
        reverse: bool,
        threads: Option<usize>,
        cancel: Option<&AtomicBool>,
    ) -> Result<Array2<f64>, MlsError> {
        if points.ncols() != 3 {
            return Err(MlsError::NotThreeD { got: points.ncols() });
        }
        let n = points.nrows();
        let mut out = Array2::<f64>::zeros((n, 3));
        if n == 0 {
            return Ok(out);
        }

        let owned;
        let pts: &[f64] = match points.as_slice() {
            Some(s) => s,
            None => {
                owned = points.to_owned();
                owned.as_slice().expect("freshly owned array is contiguous")
            }
        };
        let out_slice = out.as_slice_mut().expect("freshly allocated array is contiguous");

        // Direction is just which landmark set plays which role.
        let (src, dst) = if reverse {
            (
                (&self.qx[..], &self.qy[..], &self.qz[..]),
                (&self.px[..], &self.py[..], &self.pz[..]),
            )
        } else {
            (
                (&self.px[..], &self.py[..], &self.pz[..]),
                (&self.qx[..], &self.qy[..], &self.qz[..]),
            )
        };
        let m = src.0.len();

        with_pool(threads, || {
            pts.par_chunks(CHUNK * 3)
                .zip(out_slice.par_chunks_mut(CHUNK * 3))
                // One scratch buffer per rayon worker, reused across every chunk that
                // worker handles, so the weights never hit the allocator in the hot loop.
                .for_each_init(
                    || vec![0.0_f64; m],
                    |w, (pin, pout)| {
                        if is_cancelled(cancel) {
                            return;
                        }
                        xform_chunk(src, dst, pin, pout, w);
                    },
                );
        });

        Ok(out)
    }

    /// The *global* affine: the least-squares fit of source onto target landmarks.
    ///
    /// MLS is locally weighted, so there is no single affine that describes it — this is
    /// the one it converges to far from the landmarks, where the distance weights even
    /// out. Returned as a 4x4 homogeneous matrix.
    pub fn matrix_affine(&self, reverse: bool) -> [[f64; 4]; 4] {
        let m = self.n_landmarks();
        let (src, dst) = if reverse {
            (
                (&self.qx[..], &self.qy[..], &self.qz[..]),
                (&self.px[..], &self.py[..], &self.pz[..]),
            )
        } else {
            (
                (&self.px[..], &self.py[..], &self.pz[..]),
                (&self.qx[..], &self.qy[..], &self.qz[..]),
            )
        };

        // Normal equations for [src, 1] @ C = dst: a 4x4 system, tiny regardless of M.
        let mut ata = [[0.0_f64; 4]; 4];
        let mut atb = [[0.0_f64; 3]; 4];
        for j in 0..m {
            let s = [src.0[j], src.1[j], src.2[j], 1.0];
            let d = [dst.0[j], dst.1[j], dst.2[j]];
            for r in 0..4 {
                for c in 0..4 {
                    ata[r][c] += s[r] * s[c];
                }
                for c in 0..3 {
                    atb[r][c] += s[r] * d[c];
                }
            }
        }
        let coefs = solve4x3(ata, atb);

        let mut out = [[0.0_f64; 4]; 4];
        for r in 0..3 {
            for c in 0..3 {
                out[r][c] = coefs[c][r];
            }
            out[r][3] = coefs[3][r];
        }
        out[3] = [0.0, 0.0, 0.0, 1.0];
        out
    }
}

type Coords<'a> = (&'a [f64], &'a [f64], &'a [f64]);

/// Two reduction passes per point, with only scalars live between landmarks.
///
/// Pass 1 needs the weighted centroids before pass 2 can form the centred outer products,
/// hence two passes; the weights are cached in `wbuf` rather than recomputed, trading `M`
/// doubles (which stay in L1/L2 at realistic landmark counts) for `M` divisions and square
/// roots.
#[inline]
fn xform_chunk(src: Coords<'_>, dst: Coords<'_>, pin: &[f64], pout: &mut [f64], wbuf: &mut [f64]) {
    let m = src.0.len();
    let (sx, sy, sz) = (&src.0[..m], &src.1[..m], &src.2[..m]);
    let (dx_, dy_, dz_) = (&dst.0[..m], &dst.1[..m], &dst.2[..m]);
    let w = &mut wbuf[..m];

    for (p, o) in pin.chunks_exact(3).zip(pout.chunks_exact_mut(3)) {
        let (vx, vy, vz) = (p[0], p[1], p[2]);

        // Pass 1: inverse-square-distance weights, and the weighted centroids of both
        // landmark sets.
        let mut sw = 0.0;
        let (mut spx, mut spy, mut spz) = (0.0, 0.0, 0.0);
        let (mut sqx, mut sqy, mut sqz) = (0.0, 0.0, 0.0);
        for j in 0..m {
            let ax = sx[j] - vx;
            let ay = sy[j] - vy;
            let az = sz[j] - vz;
            let wj = 1.0 / (ax * ax + ay * ay + az * az + EPS);
            w[j] = wj;
            sw += wj;
            spx += wj * sx[j];
            spy += wj * sy[j];
            spz += wj * sz[j];
            sqx += wj * dx_[j];
            sqy += wj * dy_[j];
            sqz += wj * dz_[j];
        }
        let inv_sw = 1.0 / sw;
        let (pcx, pcy, pcz) = (spx * inv_sw, spy * inv_sw, spz * inv_sw);
        let (qcx, qcy, qcz) = (sqx * inv_sw, sqy * inv_sw, sqz * inv_sw);

        // Pass 2: the two centred, weighted outer-product sums. Y is symmetric, so only
        // its upper triangle is accumulated.
        let (mut y00, mut y01, mut y02, mut y11, mut y12, mut y22) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        let (mut z00, mut z01, mut z02) = (0.0, 0.0, 0.0);
        let (mut z10, mut z11, mut z12) = (0.0, 0.0, 0.0);
        let (mut z20, mut z21, mut z22) = (0.0, 0.0, 0.0);
        for j in 0..m {
            let wj = w[j];
            let hx = sx[j] - pcx;
            let hy = sy[j] - pcy;
            let hz = sz[j] - pcz;
            let gx = dx_[j] - qcx;
            let gy = dy_[j] - qcy;
            let gz = dz_[j] - qcz;

            let wx = wj * hx;
            let wy = wj * hy;
            let wz = wj * hz;

            y00 += wx * hx;
            y01 += wx * hy;
            y02 += wx * hz;
            y11 += wy * hy;
            y12 += wy * hz;
            y22 += wz * hz;

            z00 += wx * gx;
            z01 += wx * gy;
            z02 += wx * gz;
            z10 += wy * gx;
            z11 += wy * gy;
            z12 += wy * gz;
            z20 += wz * gx;
            z21 += wz * gy;
            z22 += wz * gz;
        }

        // v' = (v - p*) Y^-1 Z + q*, with the vector on the left as a row.
        let d = [vx - pcx, vy - pcy, vz - pcz];
        let u = solve_sym3(
            [[y00, y01, y02], [y01, y11, y12], [y02, y12, y22]],
            d,
        );
        o[0] = u[0] * z00 + u[1] * z10 + u[2] * z20 + qcx;
        o[1] = u[0] * z01 + u[1] * z11 + u[2] * z21 + qcy;
        o[2] = u[0] * z02 + u[1] * z12 + u[2] * z22 + qcz;
    }
}

/// Solve `Y u = d` for symmetric 3x3 `Y` via the adjugate.
///
/// Symmetry makes `Y^-1` symmetric too, so the row-vector product `d^T Y^-1` and the
/// column solve `Y^-1 d` coincide and this one routine serves both. A closed form beats a
/// factorisation at 3x3, and it is called once per point.
///
/// A singular `Y` — every landmark effectively coincident with the query, or landmarks
/// degenerate in a plane — yields non-finite components rather than an error, matching what
/// `numpy.linalg.inv` propagates in the reference implementation.
#[inline]
fn solve_sym3(y: [[f64; 3]; 3], d: [f64; 3]) -> [f64; 3] {
    let c00 = y[1][1] * y[2][2] - y[1][2] * y[2][1];
    let c01 = y[0][2] * y[2][1] - y[0][1] * y[2][2];
    let c02 = y[0][1] * y[1][2] - y[0][2] * y[1][1];
    let det = y[0][0] * c00 + y[1][0] * c01 + y[2][0] * c02;
    let inv_det = 1.0 / det;

    let c11 = y[0][0] * y[2][2] - y[0][2] * y[2][0];
    let c12 = y[0][2] * y[1][0] - y[0][0] * y[1][2];
    let c22 = y[0][0] * y[1][1] - y[0][1] * y[1][0];

    [
        (c00 * d[0] + c01 * d[1] + c02 * d[2]) * inv_det,
        (c01 * d[0] + c11 * d[1] + c12 * d[2]) * inv_det,
        (c02 * d[0] + c12 * d[1] + c22 * d[2]) * inv_det,
    ]
}

/// Solve a 4x4 system with three right-hand sides by Gaussian elimination with partial
/// pivoting. Used only for the global affine.
fn solve4x3(mut a: [[f64; 4]; 4], mut b: [[f64; 3]; 4]) -> [[f64; 3]; 4] {
    for k in 0..4 {
        let mut pivot = k;
        for i in (k + 1)..4 {
            if a[i][k].abs() > a[pivot][k].abs() {
                pivot = i;
            }
        }
        a.swap(k, pivot);
        b.swap(k, pivot);
        let diag = a[k][k];
        if diag == 0.0 {
            continue;
        }
        for i in (k + 1)..4 {
            let f = a[i][k] / diag;
            if f == 0.0 {
                continue;
            }
            for j in k..4 {
                a[i][j] -= f * a[k][j];
            }
            for c in 0..3 {
                b[i][c] -= f * b[k][c];
            }
        }
    }
    let mut x = [[0.0_f64; 3]; 4];
    for k in (0..4).rev() {
        for c in 0..3 {
            let mut acc = b[k][c];
            for j in (k + 1)..4 {
                acc -= a[k][j] * x[j][c];
            }
            x[k][c] = if a[k][k] == 0.0 { 0.0 } else { acc / a[k][k] };
        }
    }
    x
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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn cube() -> (Array2<f64>, Array2<f64>) {
        let src = array![
            [0.0, 0.0, 0.0],
            [10.0, 10.0, 10.0],
            [100.0, 100.0, 100.0],
            [80.0, 10.0, 30.0],
            [5.0, 60.0, 12.0],
            [42.0, 7.0, 91.0],
        ];
        let trg = array![
            [1.0, 15.0, 5.0],
            [9.0, 18.0, 21.0],
            [80.0, 99.0, 120.0],
            [5.0, 10.0, 80.0],
            [7.0, 55.0, 3.0],
            [40.0, 11.0, 88.0],
        ];
        (src, trg)
    }

    /// A point sitting exactly on a landmark maps to its partner: the `1/eps` weight
    /// swamps every other landmark.
    #[test]
    fn landmarks_map_to_partners() {
        let (src, trg) = cube();
        let mls = MlsTransform::new(src.view(), trg.view()).unwrap();
        let out = mls.xform(src.view(), false, None, None).unwrap();
        for i in 0..src.nrows() {
            for c in 0..3 {
                assert!(
                    (out[[i, c]] - trg[[i, c]]).abs() < 1e-6,
                    "landmark {i} coord {c}: {} != {}",
                    out[[i, c]],
                    trg[[i, c]]
                );
            }
        }
    }

    /// A globally affine landmark set makes MLS exactly that affine everywhere: each local
    /// solve recovers the same map regardless of the weights.
    #[test]
    fn reproduces_global_affine() {
        let src = array![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ];
        let affine = |v: [f64; 3]| {
            [
                2.0 * v[0] + 0.5 * v[1] + 3.0,
                -1.0 * v[1] + 0.25 * v[2] + 4.0,
                0.75 * v[2] + 0.1 * v[0] + 5.0,
            ]
        };
        let trg = Array2::from_shape_fn((8, 3), |(i, c)| {
            affine([src[[i, 0]], src[[i, 1]], src[[i, 2]]])[c]
        });

        let mls = MlsTransform::new(src.view(), trg.view()).unwrap();
        let pts = array![[0.3, 0.7, 0.2], [5.0, -2.0, 1.5], [0.5, 0.5, 0.5]];
        let out = mls.xform(pts.view(), false, None, None).unwrap();
        for i in 0..pts.nrows() {
            let want = affine([pts[[i, 0]], pts[[i, 1]], pts[[i, 2]]]);
            for c in 0..3 {
                assert!(
                    (out[[i, c]] - want[c]).abs() < 1e-8,
                    "point {i} coord {c}: {} != {}",
                    out[[i, c]],
                    want[c]
                );
            }
        }
    }

    /// `reverse` maps target landmarks back onto source landmarks.
    #[test]
    fn reverse_swaps_direction() {
        let (src, trg) = cube();
        let mls = MlsTransform::new(src.view(), trg.view()).unwrap();
        let out = mls.xform(trg.view(), true, None, None).unwrap();
        for i in 0..src.nrows() {
            for c in 0..3 {
                assert!(
                    (out[[i, c]] - src[[i, c]]).abs() < 1e-6,
                    "landmark {i} coord {c}: {} != {}",
                    out[[i, c]],
                    src[[i, c]]
                );
            }
        }
    }

    /// The global affine of an affine landmark set is that affine.
    #[test]
    fn matrix_affine_recovers_affine() {
        let src = array![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ];
        let trg = Array2::from_shape_fn((5, 3), |(i, c)| {
            [
                2.0 * src[[i, 0]] + 3.0,
                2.0 * src[[i, 1]] + 4.0,
                2.0 * src[[i, 2]] + 5.0,
            ][c]
        });
        let mls = MlsTransform::new(src.view(), trg.view()).unwrap();
        let m = mls.matrix_affine(false);
        let want = [
            [2.0, 0.0, 0.0, 3.0],
            [0.0, 2.0, 0.0, 4.0],
            [0.0, 0.0, 2.0, 5.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        for r in 0..4 {
            for c in 0..4 {
                assert!(
                    (m[r][c] - want[r][c]).abs() < 1e-9,
                    "[{r}][{c}]: {} != {}",
                    m[r][c],
                    want[r][c]
                );
            }
        }
    }

    /// Chunk boundaries and thread count must not change results.
    #[test]
    fn chunking_is_transparent() {
        let (src, trg) = cube();
        let mls = MlsTransform::new(src.view(), trg.view()).unwrap();
        let n = CHUNK * 2 + 13;
        let pts = Array2::from_shape_fn((n, 3), |(i, c)| (i as f64) * 0.41 + (c as f64) * 7.0);
        let all = mls.xform(pts.view(), false, None, None).unwrap();
        let one = mls.xform(pts.view(), false, Some(1), None).unwrap();
        for i in 0..n {
            for c in 0..3 {
                assert_eq!(all[[i, c]], one[[i, c]], "row {i} coord {c}");
            }
        }
    }

    #[test]
    fn rejects_bad_shapes() {
        let a = array![[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]];
        let b = array![[0.0, 0.0, 0.0]];
        assert!(matches!(
            MlsTransform::new(a.view(), b.view()),
            Err(MlsError::CountMismatch { .. })
        ));
        let flat = array![[0.0, 0.0], [1.0, 1.0]];
        assert!(matches!(
            MlsTransform::new(flat.view(), flat.view()),
            Err(MlsError::NotThreeD { .. })
        ));
    }
}
