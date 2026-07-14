//! Elastix spatial transforms: reading a `TransformParameters` file and applying it to points.
//!
//! [Elastix](https://github.com/SuperElastix/elastix) registrations bridge the *Drosophila*
//! VNC and whole-CNS template spaces (FANC → JRCVNC2018F, BANC → JRC2018F, …). A
//! `TransformParameters.*.txt` file is a flat `(Key value…)` list that is **already a chain**:
//! `InitialTransformParametersFileName` names a parent file, resolved recursively. BANC's is
//! four deep.
//!
//! - [`ElastixTransform::from_path`] reads one file *and* the chain hanging off it.
//! - [`Chain`] is one or more of those nose-to-tail, each optionally traversed backwards —
//!   the shape a bridging graph needs.
//! - [`transform_points`] / [`inverse_transform_points`] are the batch entry points.
//!
//! # Why this module exists in Rust
//!
//! The alternative — what `navis` does today — is shelling out to the `transformix` binary:
//! write the points to a temp file, `chdir` into a temp directory, spawn a process, parse the
//! text back. That needs Elastix installed and `LD_LIBRARY_PATH`/`DYLD_LIBRARY_PATH` set by
//! hand, and pays process startup plus text I/O on *every* call. Here the file is parsed once
//! and applied many times, in parallel, with no binary at all.
//!
//! It also buys an **inverse**, which Elastix simply does not have (`transformix` can only go
//! forwards). That is why `navis-flybrains` ships two separate registration files per brain
//! pair.
//!
//! # Fidelity to Elastix
//!
//! Validated **against the `transformix` binary itself** (Elastix 5.2.0) on thousands of random
//! points spanning each grid and well beyond it, for all seven registration files
//! `navis-flybrains` ships — including both four-deep BANC chains. Agreement is `5e-7`, which
//! is `transformix`'s own print precision (it writes six decimals), and the sets of points it
//! leaves untouched are reproduced exactly. (`fastcore/testdata/elastix/` pins synthetic
//! fixtures plus real `transformix` output as a checked-in regression, so the suite needs no
//! Elastix install.)
//!
//! Four behaviours are load-bearing and none of them is what you would guess:
//!
//! - **Outside the B-spline grid, the point is returned _unchanged_** — the identity, never
//!   `NaN`. This is the exact opposite of CMTK, which reports such points as `FAILED`. See
//!   [`BSpline::in_region`] and [`OutOfBounds`].
//! - **The `Euler` rotation order is `Rz·Rx·Ry`**, not `Rz·Ry·Rx`, unless `ComputeZYX` is set.
//!   See [`Linear::from_euler`].
//! - **B-spline coefficients are component-major** — every x-displacement, then every y, then
//!   every z — not interleaved triples. See [`BSpline::from_params`].
//! - **`InitialTransformParametersFileName` resolves against the transform file's own
//!   directory**, not the working directory.
//!
//! # Attribution
//!
//! The file format and the transform semantics are those of
//! [Elastix](https://github.com/SuperElastix/elastix) and [ITK](https://itk.org) (both
//! Apache-2.0); this is an independent implementation derived from their documented behaviour
//! and from measuring the `transformix` binary — no code was copied. The approach of driving
//! elastix transforms from Python was pioneered by
//! [pytransformix](https://github.com/jasper-tms/pytransformix) (Jasper Phelps), which
//! `navis.transforms.elastix` is based on.

use std::collections::HashSet;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;

use ndarray::{Array2, ArrayView2};
use rayon::prelude::*;

use crate::nblast::{is_cancelled, make_bar, with_pool};

/// Rows per parallel chunk. Sized so the cancellation poll and the progress-bar tick amortise
/// over real work: an inverse point costs ~5-20 µs, so a chunk is ~10 ms.
const CHUNK: usize = 1024;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ElastixError {
    Io {
        path: String,
        msg: String,
    },
    /// No `(Transform "...")` key. Also catches files that merely *look* like transforms —
    /// `navis-flybrains` ships a `template_to_BANC.txt` that contains nothing but a filename.
    NotElastix {
        path: String,
    },
    UnsupportedTransform {
        kind: String,
    },
    /// `UseBinaryFormatForTransformationParameters "true"`: the coefficients live in a
    /// separate `.dat` file, which we do not read.
    BinaryParameters {
        path: String,
    },
    /// ITK allows spline orders 0-5; elastix writes 3 and that is all we implement.
    UnsupportedSplineOrder {
        got: i64,
    },
    ParamCount {
        kind: String,
        got: usize,
        want: usize,
    },
    MissingKey {
        path: String,
        key: &'static str,
    },
    BadValue {
        key: String,
        msg: String,
    },
    /// `InitialTransformParametersFileName` chains back onto itself.
    CircularChain {
        path: String,
    },
    SingularLinear,
    /// A `HowToCombineTransforms "Add"` step. `T(x) = T_initial(x) + T_this(x) - x` does not
    /// decompose into invertible hops, so we refuse rather than approximate. No registration
    /// in the wild uses it.
    NotInvertible,
    EmptyChain,
    BadShape {
        got: Vec<usize>,
    },
    /// `initial_guess` was not one point per input point.
    GuessLen {
        got: usize,
        want: usize,
    },
    /// `invert` was not one flag per transform in the chain.
    InvertLen {
        got: usize,
        want: usize,
    },
    Cancelled,
}

impl fmt::Display for ElastixError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ElastixError::Io { path, msg } => write!(f, "could not read {path}: {msg}"),
            ElastixError::NotElastix { path } => write!(
                f,
                "{path} is not an elastix transform file (no `(Transform \"...\")` key)"
            ),
            ElastixError::UnsupportedTransform { kind } => {
                write!(f, "unsupported transform type `{kind}`")
            }
            ElastixError::BinaryParameters { path } => write!(
                f,
                "{path} stores its parameters in binary form, which is not supported"
            ),
            ElastixError::UnsupportedSplineOrder { got } => {
                write!(f, "only cubic B-splines (order 3) are supported, got {got}")
            }
            ElastixError::ParamCount { kind, got, want } => write!(
                f,
                "`{kind}` needs {want} transform parameters, got {got}"
            ),
            ElastixError::MissingKey { path, key } => {
                write!(f, "{path} is missing the `{key}` key")
            }
            ElastixError::BadValue { key, msg } => write!(f, "bad value for `{key}`: {msg}"),
            ElastixError::CircularChain { path } => {
                write!(f, "initial-transform chain loops back onto {path}")
            }
            ElastixError::SingularLinear => write!(f, "linear transform is not invertible"),
            ElastixError::NotInvertible => write!(
                f,
                "an `Add` transform cannot be inverted (only `Compose` chains can)"
            ),
            ElastixError::EmptyChain => write!(f, "chain holds no transforms"),
            ElastixError::BadShape { got } => {
                write!(f, "expected an (N, 3) array of points, got {got:?}")
            }
            ElastixError::GuessLen { got, want } => write!(
                f,
                "`initial_guess` must hold one point per input point: expected {want}, got {got}"
            ),
            ElastixError::InvertLen { got, want } => write!(
                f,
                "`invert` must hold one flag per transform: expected {want}, got {got}"
            ),
            ElastixError::Cancelled => write!(f, "interrupted"),
        }
    }
}

impl std::error::Error for ElastixError {}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

/// The `(Key value…)` entries of one file, in order. Keys may repeat; the last wins, which is
/// what elastix itself does.
#[derive(Debug, Clone, Default)]
struct Params {
    path: String,
    entries: Vec<(String, Vec<String>)>,
}

impl Params {
    /// Scan `(Key value…)` groups, honouring `//` comments and `"quoted strings"`.
    fn parse(text: &str, path: &str) -> Params {
        Params::scan(text, path, false)
    }

    /// As [`Params::parse`], but do not tokenise the coefficient array.
    ///
    /// `(TransformParameters …)` is a *single line* holding every coefficient — 5.5 MB of it in
    /// FANC's warp, 56 MB in BANC's — and tokenising it allocates a `String` per number. Every
    /// other key, including the two that decide invertibility, is a handful of bytes on the
    /// lines around it. Skipping just that one line's values is the whole trick behind
    /// [`probe_invertible`].
    fn parse_headers(text: &str, path: &str) -> Params {
        Params::scan(text, path, true)
    }

    fn scan(text: &str, path: &str, skip_coefficients: bool) -> Params {
        let mut entries: Vec<(String, Vec<String>)> = Vec::new();

        for raw in text.lines() {
            let line = match raw.find("//") {
                Some(i) => &raw[..i],
                None => raw,
            };
            let line = line.trim();
            if !line.starts_with('(') || !line.ends_with(')') {
                continue;
            }
            let body = &line[1..line.len() - 1];

            // The one line worth not looking at. Record the key so `parse_headers` still sees a
            // well-formed file; the values stay unread.
            if skip_coefficients && body.starts_with("TransformParameters ") {
                entries.push(("TransformParameters".to_string(), Vec::new()));
                continue;
            }

            // Split on whitespace, but keep quoted runs together.
            let mut toks: Vec<String> = Vec::new();
            let mut cur = String::new();
            let mut in_quotes = false;
            for ch in body.chars() {
                match ch {
                    '"' => in_quotes = !in_quotes,
                    c if c.is_whitespace() && !in_quotes => {
                        if !cur.is_empty() {
                            toks.push(std::mem::take(&mut cur));
                        }
                    }
                    c => cur.push(c),
                }
            }
            if !cur.is_empty() {
                toks.push(cur);
            }
            if toks.is_empty() {
                continue;
            }
            let key = toks.remove(0);
            entries.push((key, toks));
        }

        Params {
            path: path.to_string(),
            entries,
        }
    }

    fn get(&self, key: &str) -> Option<&[String]> {
        self.entries
            .iter()
            .rev()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_slice())
    }

    fn require(&self, key: &'static str) -> Result<&[String], ElastixError> {
        self.get(key).ok_or(ElastixError::MissingKey {
            path: self.path.clone(),
            key,
        })
    }

    fn str_or(&self, key: &str, default: &str) -> String {
        self.get(key)
            .and_then(|v| v.first())
            .cloned()
            .unwrap_or_else(|| default.to_string())
    }

    fn bool_or(&self, key: &str, default: bool) -> bool {
        match self.get(key).and_then(|v| v.first()) {
            Some(s) => s.eq_ignore_ascii_case("true"),
            None => default,
        }
    }

    fn floats(&self, key: &'static str) -> Result<Vec<f64>, ElastixError> {
        self.require(key)?
            .iter()
            .map(|s| {
                s.parse::<f64>().map_err(|_| ElastixError::BadValue {
                    key: key.to_string(),
                    msg: format!("`{s}` is not a number"),
                })
            })
            .collect()
    }

    fn ints(&self, key: &'static str) -> Result<Vec<i64>, ElastixError> {
        self.require(key)?
            .iter()
            .map(|s| {
                s.parse::<i64>().map_err(|_| ElastixError::BadValue {
                    key: key.to_string(),
                    msg: format!("`{s}` is not an integer"),
                })
            })
            .collect()
    }

    fn triple(&self, key: &'static str) -> Result<[f64; 3], ElastixError> {
        let v = self.floats(key)?;
        if v.len() != 3 {
            return Err(ElastixError::BadValue {
                key: key.to_string(),
                msg: format!("expected 3 values, got {}", v.len()),
            });
        }
        Ok([v[0], v[1], v[2]])
    }

    /// `CenterOfRotationPoint`, defaulting to the origin (translations and B-splines have none).
    fn center(&self) -> [f64; 3] {
        self.triple("CenterOfRotationPoint").unwrap_or([0.0; 3])
    }
}

// ---------------------------------------------------------------------------
// Linear
// ---------------------------------------------------------------------------

/// `T(x) = A(x - c) + c + t`.
///
/// Affine, Euler, Similarity and Translation are all this — they differ only in how `A` is
/// built from the parameter vector, so they collapse into one type.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Linear {
    pub a: [[f64; 3]; 3],
    pub t: [f64; 3],
    pub c: [f64; 3],
}

impl Linear {
    pub fn identity() -> Self {
        Linear {
            a: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            t: [0.0; 3],
            c: [0.0; 3],
        }
    }

    /// 12 parameters: a row-major 3×3 matrix, then the translation.
    fn from_affine(p: &[f64], c: [f64; 3]) -> Result<Self, ElastixError> {
        if p.len() != 12 {
            return Err(ElastixError::ParamCount {
                kind: "AffineTransform".into(),
                got: p.len(),
                want: 12,
            });
        }
        Ok(Linear {
            a: [
                [p[0], p[1], p[2]],
                [p[3], p[4], p[5]],
                [p[6], p[7], p[8]],
            ],
            t: [p[9], p[10], p[11]],
            c,
        })
    }

    fn from_translation(p: &[f64]) -> Result<Self, ElastixError> {
        if p.len() != 3 {
            return Err(ElastixError::ParamCount {
                kind: "TranslationTransform".into(),
                got: p.len(),
                want: 3,
            });
        }
        Ok(Linear {
            t: [p[0], p[1], p[2]],
            ..Linear::identity()
        })
    }

    /// 6 parameters: three Euler angles, then the translation.
    ///
    /// **The rotation order is `Rz·Rx·Ry`**, matching ITK's `Euler3DTransform` with its default
    /// `ComputeZYX = false` (which is quaternion-like, not the ZYX its name suggests). Only when
    /// the file explicitly sets `ComputeZYX "true"` is it `Rz·Ry·Rx`. Both orders are pinned by
    /// `transformix`-generated golden data — do not "simplify" this.
    fn from_euler(p: &[f64], c: [f64; 3], compute_zyx: bool) -> Result<Self, ElastixError> {
        if p.len() != 6 {
            return Err(ElastixError::ParamCount {
                kind: "EulerTransform".into(),
                got: p.len(),
                want: 6,
            });
        }
        let (rx, ry, rz) = (rot_x(p[0]), rot_y(p[1]), rot_z(p[2]));
        let a = if compute_zyx {
            mat3_mul(&rz, &mat3_mul(&ry, &rx))
        } else {
            mat3_mul(&rz, &mat3_mul(&rx, &ry))
        };
        Ok(Linear {
            a,
            t: [p[3], p[4], p[5]],
            c,
        })
    }

    /// 7 parameters: the vector part of a versor (unit quaternion), the translation, the scale.
    fn from_similarity(p: &[f64], c: [f64; 3]) -> Result<Self, ElastixError> {
        if p.len() != 7 {
            return Err(ElastixError::ParamCount {
                kind: "SimilarityTransform".into(),
                got: p.len(),
                want: 7,
            });
        }
        let (x, y, z) = (p[0], p[1], p[2]);
        let w = (1.0 - x * x - y * y - z * z).max(0.0).sqrt();
        let s = p[6];
        let r = [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ];
        let mut a = [[0.0f64; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                a[i][j] = s * r[i][j];
            }
        }
        Ok(Linear {
            a,
            t: [p[3], p[4], p[5]],
            c,
        })
    }

    #[inline]
    pub fn apply(&self, p: [f64; 3]) -> [f64; 3] {
        let d = [p[0] - self.c[0], p[1] - self.c[1], p[2] - self.c[2]];
        let mut out = [0.0f64; 3];
        for (i, o) in out.iter_mut().enumerate() {
            *o = self.a[i][0] * d[0]
                + self.a[i][1] * d[1]
                + self.a[i][2] * d[2]
                + self.c[i]
                + self.t[i];
        }
        out
    }

    /// `x = A⁻¹(y - c - t) + c`, re-expressed in the same `A'(x - c') + c' + t'` form.
    pub fn inverse(&self) -> Option<Linear> {
        let ai = mat3_inv(&self.a)?;
        let mut t = [0.0f64; 3];
        for i in 0..3 {
            t[i] = -(ai[i][0] * self.t[0] + ai[i][1] * self.t[1] + ai[i][2] * self.t[2]);
        }
        Some(Linear {
            a: ai,
            t,
            c: self.c,
        })
    }

    /// The equivalent 4×4 homogeneous matrix, for the bindings.
    pub fn as_array(&self) -> Array2<f64> {
        let mut m = Array2::<f64>::eye(4);
        for i in 0..3 {
            for j in 0..3 {
                m[[i, j]] = self.a[i][j];
            }
            // offset = c + t - A·c
            let ac: f64 = (0..3).map(|k| self.a[i][k] * self.c[k]).sum();
            m[[i, 3]] = self.c[i] + self.t[i] - ac;
        }
        m
    }
}

fn rot_x(a: f64) -> [[f64; 3]; 3] {
    let (s, c) = a.sin_cos();
    [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]
}

fn rot_y(a: f64) -> [[f64; 3]; 3] {
    let (s, c) = a.sin_cos();
    [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]
}

fn rot_z(a: f64) -> [[f64; 3]; 3] {
    let (s, c) = a.sin_cos();
    [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
}

fn mat3_mul(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut m = [[0.0f64; 3]; 3];
    for (i, row) in m.iter_mut().enumerate() {
        for (j, v) in row.iter_mut().enumerate() {
            *v = (0..3).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    m
}

fn mat3_inv(a: &[[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
    if det.abs() < 1e-12 {
        return None;
    }
    let inv_det = 1.0 / det;
    let mut m = [[0.0f64; 3]; 3];
    m[0][0] = (a[1][1] * a[2][2] - a[1][2] * a[2][1]) * inv_det;
    m[0][1] = (a[0][2] * a[2][1] - a[0][1] * a[2][2]) * inv_det;
    m[0][2] = (a[0][1] * a[1][2] - a[0][2] * a[1][1]) * inv_det;
    m[1][0] = (a[1][2] * a[2][0] - a[1][0] * a[2][2]) * inv_det;
    m[1][1] = (a[0][0] * a[2][2] - a[0][2] * a[2][0]) * inv_det;
    m[1][2] = (a[0][2] * a[1][0] - a[0][0] * a[1][2]) * inv_det;
    m[2][0] = (a[1][0] * a[2][1] - a[1][1] * a[2][0]) * inv_det;
    m[2][1] = (a[0][1] * a[2][0] - a[0][0] * a[2][1]) * inv_det;
    m[2][2] = (a[0][0] * a[1][1] - a[0][1] * a[1][0]) * inv_det;
    Some(m)
}

// ---------------------------------------------------------------------------
// B-spline
// ---------------------------------------------------------------------------

/// Cubic B-spline basis, for the four taps at offsets -1, 0, 1, 2.
#[inline]
fn bspline_weights(t: f64) -> [f64; 4] {
    let t2 = t * t;
    let t3 = t2 * t;
    [
        (1.0 - 3.0 * t + 3.0 * t2 - t3) / 6.0,
        (4.0 - 6.0 * t2 + 3.0 * t3) / 6.0,
        (1.0 + 3.0 * t + 3.0 * t2 - 3.0 * t3) / 6.0,
        t3 / 6.0,
    ]
}

/// Derivative of [`bspline_weights`] with respect to `t`.
#[inline]
fn bspline_weights_deriv(t: f64) -> [f64; 4] {
    let t2 = t * t;
    [
        (-3.0 + 6.0 * t - 3.0 * t2) / 6.0,
        (-12.0 * t + 9.0 * t2) / 6.0,
        (3.0 + 6.0 * t - 9.0 * t2) / 6.0,
        3.0 * t2 / 6.0,
    ]
}

/// A cubic B-spline **displacement** field on a regular control-point grid.
///
/// `T(x) = x + Σ w·c` inside the valid region, and `T(x) = x` outside it.
#[derive(Debug, Clone, PartialEq)]
pub struct BSpline {
    pub size: [usize; 3],
    /// `GridIndex`. Zero in every file seen in the wild, but it shifts the valid region and the
    /// coefficient lookup, so it is parsed rather than assumed.
    pub index: [i64; 3],
    pub origin: [f64; 3],
    pub spacing: [f64; 3],
    /// Inverse of `GridDirection`, precomputed — it is in the inner loop.
    pub dir_inv: [[f64; 3]; 3],
    /// `prod(size)` control points, **x fastest**: `idx = ix + nx * (iy + ny * iz)`.
    pub coefficients: Vec<[f64; 3]>,
}

impl BSpline {
    /// Elastix stores the coefficients **component-major**: every x-displacement, then every y,
    /// then every z — *not* interleaved triples. Within each block the grid is rastered x-fastest,
    /// which is the order we keep, so the i-th control point is `(par[i], par[n+i], par[2n+i])`.
    fn from_params(p: &Params, par: &[f64]) -> Result<Self, ElastixError> {
        let order = p.ints("BSplineTransformSplineOrder").map(|v| v[0]).unwrap_or(3);
        if order != 3 {
            return Err(ElastixError::UnsupportedSplineOrder { got: order });
        }

        let gs = p.ints("GridSize")?;
        if gs.len() != 3 || gs.iter().any(|&v| v < 4) {
            return Err(ElastixError::BadValue {
                key: "GridSize".into(),
                msg: format!("expected 3 values of at least 4, got {gs:?}"),
            });
        }
        let size = [gs[0] as usize, gs[1] as usize, gs[2] as usize];

        let gi = p.ints("GridIndex").unwrap_or_else(|_| vec![0, 0, 0]);
        let index = [gi[0], gi[1], gi[2]];

        let n = size[0] * size[1] * size[2];
        if par.len() != 3 * n {
            return Err(ElastixError::ParamCount {
                kind: "BSplineTransform".into(),
                got: par.len(),
                want: 3 * n,
            });
        }
        let coefficients = (0..n).map(|i| [par[i], par[n + i], par[2 * n + i]]).collect();

        let dir = p.floats("GridDirection")?;
        if dir.len() != 9 {
            return Err(ElastixError::BadValue {
                key: "GridDirection".into(),
                msg: format!("expected 9 values, got {}", dir.len()),
            });
        }
        let d = [
            [dir[0], dir[1], dir[2]],
            [dir[3], dir[4], dir[5]],
            [dir[6], dir[7], dir[8]],
        ];
        let dir_inv = mat3_inv(&d).ok_or(ElastixError::BadValue {
            key: "GridDirection".into(),
            msg: "matrix is singular".into(),
        })?;

        Ok(BSpline {
            size,
            index,
            origin: p.triple("GridOrigin")?,
            spacing: p.triple("GridSpacing")?,
            dir_inv,
            coefficients,
        })
    }

    /// Continuous grid index: `u = GridDirection⁻¹·(p − GridOrigin) / GridSpacing`.
    #[inline]
    fn continuous_index(&self, p: [f64; 3]) -> [f64; 3] {
        let d = [
            p[0] - self.origin[0],
            p[1] - self.origin[1],
            p[2] - self.origin[2],
        ];
        let mut u = [0.0f64; 3];
        for (j, uj) in u.iter_mut().enumerate() {
            let v: f64 = (0..3).map(|k| self.dir_inv[j][k] * d[k]).sum();
            *uj = v / self.spacing[j];
        }
        u
    }

    /// ITK's valid region for an **odd** spline order: `[first + 1, first + size - 2)` per axis.
    ///
    /// Verified against `transformix` with zero disagreements on 4000 points spanning three
    /// times the grid extent. Note the upper bound is **exclusive** and sits two cells inside
    /// the grid, not one: a cubic tap needs a neighbour on each side plus one more.
    #[inline]
    pub fn in_region(&self, p: [f64; 3]) -> bool {
        let u = self.continuous_index(p);
        (0..3).all(|j| {
            let lo = self.index[j] as f64 + 1.0;
            let hi = self.index[j] as f64 + self.size[j] as f64 - 2.0;
            u[j] >= lo && u[j] < hi
        })
    }

    /// The forward transform. **Outside the valid region the point is returned unchanged** —
    /// that is what ITK does, and therefore what `transformix` prints.
    #[inline]
    pub fn apply(&self, p: [f64; 3]) -> [f64; 3] {
        if !self.in_region(p) {
            return p;
        }
        self.eval_extrapolated(p)
    }

    /// The spline evaluated **without** the region test, clamping the tap indices to the grid.
    ///
    /// This is deliberately *not* the forward transform: it is smooth everywhere, which is what
    /// the inverse solver needs to descend. The true forward map's identity-outside branch makes
    /// it discontinuous at the boundary, and LM cannot work with that.
    #[inline]
    fn eval_extrapolated(&self, p: [f64; 3]) -> [f64; 3] {
        let u = self.continuous_index(p);
        let mut fl = [0i64; 3];
        let mut w = [[0.0f64; 4]; 3];
        for j in 0..3 {
            let f = u[j].floor();
            fl[j] = f as i64;
            w[j] = bspline_weights(u[j] - f);
        }

        let mut d = [0.0f64; 3];
        for k in 0..4 {
            let iz = self.tap(2, fl[2], k);
            for j in 0..4 {
                let iy = self.tap(1, fl[1], j);
                let wyz = w[1][j] * w[2][k];
                let base = self.size[0] * (iy + self.size[1] * iz);
                for (i, &wx) in w[0].iter().enumerate() {
                    let ix = self.tap(0, fl[0], i);
                    let ww = wx * wyz;
                    let c = &self.coefficients[ix + base];
                    d[0] += ww * c[0];
                    d[1] += ww * c[1];
                    d[2] += ww * c[2];
                }
            }
        }
        [p[0] + d[0], p[1] + d[1], p[2] + d[2]]
    }

    /// The `t`-th tap along axis `j`, clamped into the grid. Clamping *is* the extrapolation
    /// policy, and it is only ever reached from the solver — inside the valid region every tap
    /// is in range by construction.
    #[inline]
    fn tap(&self, j: usize, fl: i64, t: usize) -> usize {
        let i = fl - 1 - self.index[j] + t as i64;
        i.clamp(0, self.size[j] as i64 - 1) as usize
    }

    /// [`Self::eval_extrapolated`] plus its analytic 3×3 Jacobian.
    ///
    /// The spline is a *displacement*, so the Jacobian of the map `x ↦ x + disp(x)` carries an
    /// **identity term**: `I + ∂disp/∂x`. Forgetting it is a classic bug — the solver still
    /// roughly converges, which is what makes it so easy to miss.
    #[inline]
    fn eval_with_jacobian(&self, p: [f64; 3]) -> ([f64; 3], [[f64; 3]; 3]) {
        let u = self.continuous_index(p);
        let mut fl = [0i64; 3];
        let mut w = [[0.0f64; 4]; 3];
        let mut dw = [[0.0f64; 4]; 3];
        for j in 0..3 {
            let f = u[j].floor();
            fl[j] = f as i64;
            let t = u[j] - f;
            w[j] = bspline_weights(t);
            dw[j] = bspline_weights_deriv(t);
        }

        let mut d = [0.0f64; 3];
        // ju[a][j] = ∂disp[a] / ∂u[j]
        let mut ju = [[0.0f64; 3]; 3];
        for k in 0..4 {
            let iz = self.tap(2, fl[2], k);
            for j in 0..4 {
                let iy = self.tap(1, fl[1], j);
                let base = self.size[0] * (iy + self.size[1] * iz);
                for i in 0..4 {
                    let ix = self.tap(0, fl[0], i);
                    let c = &self.coefficients[ix + base];
                    let (wx, wy, wz) = (w[0][i], w[1][j], w[2][k]);
                    let ww = wx * wy * wz;
                    let g = [
                        dw[0][i] * wy * wz,
                        wx * dw[1][j] * wz,
                        wx * wy * dw[2][k],
                    ];
                    for a in 0..3 {
                        d[a] += ww * c[a];
                        for (jj, gj) in g.iter().enumerate() {
                            ju[a][jj] += gj * c[a];
                        }
                    }
                }
            }
        }

        // ∂u/∂x = diag(1/spacing) · dir_inv, so ∂disp/∂x = ju · diag(1/spacing) · dir_inv.
        let mut jac = [[0.0f64; 3]; 3];
        for (a, row) in jac.iter_mut().enumerate() {
            for (b, v) in row.iter_mut().enumerate() {
                *v = (0..3)
                    .map(|j| ju[a][j] / self.spacing[j] * self.dir_inv[j][b])
                    .sum::<f64>()
                    + if a == b { 1.0 } else { 0.0 };
            }
        }
        ([p[0] + d[0], p[1] + d[1], p[2] + d[2]], jac)
    }

    /// Find `x` with `apply(x) == target`.
    ///
    /// The forward map is `x + disp(x)` inside the grid and `x` outside, so it is
    /// **discontinuous at the boundary and hence non-injective** — a target can have a preimage
    /// on each branch. Three things follow, and all three are load-bearing:
    ///
    /// 1. The solve runs against [`Self::eval_extrapolated`], never the real forward map. Seeding
    ///    LM on the true map is *actively wrong*: whenever `target` lies outside the grid,
    ///    `apply(target) == target`, so the residual is 0 on the first evaluation and the solver
    ///    "converges" instantly on the identity — even when the real preimage is 200 µm away.
    /// 2. The seed is a fixed-point iteration `x ← target − disp(x)`, the standard way to invert
    ///    a displacement field. Starting from `target` alone leaves points unconverged where the
    ///    deformation is large (FANC's reaches 200 µm, ~20 grid cells).
    /// 3. The root is only accepted if it lands **inside** the region — only there does the true
    ///    forward map use the warp. Otherwise the identity branch applies, which is legitimate
    ///    only if `target` is itself outside.
    ///
    /// When **both** branches are valid the target has two preimages and there is no fact of the
    /// matter about which one you meant. A caller who supplied `x0` has told us; otherwise we
    /// take the warp — the registration doing its job — over the identity, which is merely
    /// territory it does not cover.
    ///
    /// `None` (→ a `NaN` row) when neither branch holds.
    pub fn solve_inverse(
        &self,
        target: [f64; 3],
        x0: Option<[f64; 3]>,
        lattice: Option<&SeedLattice>,
        opts: &InverseOpts,
    ) -> Option<[f64; 3]> {
        // Starts to try, in order, stopping at the first that lands a genuine preimage. All three
        // are needed, and they fail in different places:
        //
        // - The **fixed-point seed** carries large deformations (FANC's reach ~20 grid cells). But
        //   it is a contraction only where `|∂disp| < 1`, and a strong warp breaks that locally —
        //   there it can throw the start further away than it began.
        // - The **bare target** is the better start exactly where the seed diverges, i.e. where the
        //   deformation is mild.
        // - The **seed lattice** is the global fallback for the rest: a coarse grid of points whose
        //   forward images are known, so we can start from the one that already lands near the
        //   target. Without it, BANC's warp (median displacement 163 µm — ten grid cells) leaves
        //   ~2.6% of points unconverged; with it, ~0.6%.
        //
        // Each start after the first is only paid for by a point that has already failed, which on
        // every real registration is a small minority.
        let seed = self.seed(target, opts.seed_iter);
        let warp = x0
            .into_iter()
            .chain([seed, target])
            .chain(lattice.map(|l| l.nearest_source(target)))
            .find_map(|start| {
                // A root is only a real preimage if it lands where the forward map actually warps.
                self.lm(target, start, opts).filter(|&x| self.in_region(x))
            });
        // ...and `target` is its own preimage, but only out where the forward map is the identity.
        let identity = (!self.in_region(target) && opts.out_of_bounds == OutOfBounds::Identity)
            .then_some(target);

        match (warp, identity) {
            (Some(w), Some(i)) => Some(match x0 {
                Some(g) if sq_dist(i, g) < sq_dist(w, g) => i,
                _ => w,
            }),
            (Some(w), None) => Some(w),
            (None, other) => other,
        }
    }

    /// A coarse grid of points spanning the valid region, paired with their forward images.
    ///
    /// Built once per inverse call (a few thousand spline evaluations, ~1 ms) and consulted only
    /// by points that the cheap starts already failed on.
    pub fn seed_lattice(&self, target_points: usize) -> SeedLattice {
        // Shape the lattice like the control-point grid, scaled to roughly `target_points`.
        let span = [
            (self.size[0] - 3).max(1) as f64,
            (self.size[1] - 3).max(1) as f64,
            (self.size[2] - 3).max(1) as f64,
        ];
        let scale = (target_points as f64 / (span[0] * span[1] * span[2])).cbrt();
        let n: Vec<usize> = span
            .iter()
            .map(|s| ((s * scale).round() as usize).clamp(2, 96))
            .collect();

        let lo = [
            self.index[0] as f64 + 1.0,
            self.index[1] as f64 + 1.0,
            self.index[2] as f64 + 1.0,
        ];
        // Just inside the (exclusive) upper bound.
        let hi = [
            self.index[0] as f64 + self.size[0] as f64 - 2.0 - 1e-6,
            self.index[1] as f64 + self.size[1] as f64 - 2.0 - 1e-6,
            self.index[2] as f64 + self.size[2] as f64 - 2.0 - 1e-6,
        ];

        let mut src = Vec::with_capacity(n[0] * n[1] * n[2]);
        let mut img = Vec::with_capacity(n[0] * n[1] * n[2]);
        for iz in 0..n[2] {
            for iy in 0..n[1] {
                for ix in 0..n[0] {
                    let u = [
                        lerp(lo[0], hi[0], ix, n[0]),
                        lerp(lo[1], hi[1], iy, n[1]),
                        lerp(lo[2], hi[2], iz, n[2]),
                    ];
                    let p = self.world_from_index(u);
                    src.push(p);
                    img.push(self.eval_extrapolated(p));
                }
            }
        }
        SeedLattice { src, img }
    }

    /// World coordinates from a continuous grid index — the inverse of [`Self::continuous_index`].
    #[inline]
    fn world_from_index(&self, u: [f64; 3]) -> [f64; 3] {
        // `continuous_index` is `u = dir_inv · (p − origin) / spacing`, so `p = dir · (u * spacing)
        // + origin`. We hold only `dir_inv`, so invert it back.
        let dir = mat3_inv(&self.dir_inv).expect("dir_inv came from an invertible matrix");
        let s = [
            u[0] * self.spacing[0],
            u[1] * self.spacing[1],
            u[2] * self.spacing[2],
        ];
        let mut p = [0.0f64; 3];
        for (j, pj) in p.iter_mut().enumerate() {
            *pj = (0..3).map(|k| dir[j][k] * s[k]).sum::<f64>() + self.origin[j];
        }
        p
    }

    /// Fixed-point iteration `x ← target − disp(x)`.
    #[inline]
    fn seed(&self, target: [f64; 3], iters: usize) -> [f64; 3] {
        let mut x = target;
        for _ in 0..iters {
            let f = self.eval_extrapolated(x);
            // disp(x) = f - x
            for j in 0..3 {
                x[j] = target[j] - (f[j] - x[j]);
            }
        }
        x
    }

    /// Levenberg-Marquardt against the analytic Jacobian. `None` if the residual never comes
    /// within `opts.accuracy`.
    fn lm(&self, target: [f64; 3], x0: [f64; 3], opts: &InverseOpts) -> Option<[f64; 3]> {
        let mut x = x0;
        let mut cost = sq_dist(self.eval_extrapolated(x), target);
        let mut lambda = 1e-6f64;

        for _ in 0..opts.max_iter {
            if cost == 0.0 {
                break;
            }
            let (f, j) = self.eval_with_jacobian(x);
            let r = [f[0] - target[0], f[1] - target[1], f[2] - target[2]];

            // Normal equations: (JᵀJ + λ·diag(JᵀJ)) d = Jᵀr.
            let mut a = [[0.0f64; 3]; 3];
            let mut g = [0.0f64; 3];
            for c in 0..3 {
                for c2 in 0..3 {
                    a[c][c2] = (0..3).map(|k| j[k][c] * j[k][c2]).sum();
                }
                g[c] = (0..3).map(|k| j[k][c] * r[k]).sum();
            }
            // Marquardt's diagonal scaling rather than `+ λI`: the grid spacings differ by up to
            // 2.5× between axes, so isotropic damping is misscaled.
            for (c, arow) in a.iter_mut().enumerate() {
                arow[c] += lambda * arow[c].abs().max(1e-12);
            }

            // A singular normal matrix means this step is undefined -- but the iterate we already
            // have may be perfectly good, so stop and let the residual check below decide. (`?`
            // here would throw away a converged answer, which near a fold is exactly where the
            // matrix goes singular.)
            let d = match solve3x3(&a, &g) {
                Some(d) => d,
                None => break,
            };
            let x_new = [x[0] - d[0], x[1] - d[1], x[2] - d[2]];
            let cost_new = sq_dist(self.eval_extrapolated(x_new), target);

            if cost_new < cost {
                let step = (0..3).map(|i| d[i].abs()).fold(0.0f64, f64::max);
                x = x_new;
                cost = cost_new;
                lambda = (lambda * 0.3).max(1e-12);
                if step < opts.tolerance {
                    break;
                }
            } else {
                lambda *= 10.0;
                if lambda > 1e12 {
                    break;
                }
            }
        }

        (cost.sqrt() <= opts.accuracy).then_some(x)
    }
}

/// A coarse grid of points and their forward images, used to start the inverse solver globally
/// when the local starts have failed. See [`BSpline::seed_lattice`].
#[derive(Debug, Clone, PartialEq)]
pub struct SeedLattice {
    src: Vec<[f64; 3]>,
    img: Vec<[f64; 3]>,
}

impl SeedLattice {
    /// The lattice point whose forward image is nearest `target`.
    ///
    /// A linear scan. That sounds careless for a few thousand entries, but only points that have
    /// *already* failed both cheap starts ever get here — a small minority even on the worst real
    /// registration — so a spatial index would add a data structure to save time nobody spends.
    fn nearest_source(&self, target: [f64; 3]) -> [f64; 3] {
        let mut best = 0usize;
        let mut best_d = f64::INFINITY;
        for (i, &q) in self.img.iter().enumerate() {
            let d = sq_dist(q, target);
            if d < best_d {
                best_d = d;
                best = i;
            }
        }
        self.src[best]
    }

    pub fn len(&self) -> usize {
        self.src.len()
    }

    pub fn is_empty(&self) -> bool {
        self.src.is_empty()
    }
}

#[inline]
fn lerp(lo: f64, hi: f64, i: usize, n: usize) -> f64 {
    if n <= 1 {
        return lo;
    }
    lo + (hi - lo) * (i as f64) / ((n - 1) as f64)
}

#[inline]
fn sq_dist(a: [f64; 3], b: [f64; 3]) -> f64 {
    (0..3).map(|i| (a[i] - b[i]).powi(2)).sum()
}

/// Cramer's rule. `None` if singular.
#[inline]
fn solve3x3(a: &[[f64; 3]; 3], g: &[f64; 3]) -> Option<[f64; 3]> {
    let inv = mat3_inv(a)?;
    let mut out = [0.0f64; 3];
    for (i, o) in out.iter_mut().enumerate() {
        *o = (0..3).map(|k| inv[i][k] * g[k]).sum();
    }
    Some(out)
}

// ---------------------------------------------------------------------------
// Transform & chain
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub enum Transform {
    Linear(Linear),
    BSpline(BSpline),
}

impl Transform {
    /// `None` only when `out_of_bounds` is [`OutOfBounds::Nan`] and the point falls outside a
    /// B-spline's valid region.
    #[inline]
    fn apply(&self, p: [f64; 3], oob: OutOfBounds) -> Option<[f64; 3]> {
        match self {
            Transform::Linear(l) => Some(l.apply(p)),
            Transform::BSpline(b) => {
                if oob == OutOfBounds::Nan && !b.in_region(p) {
                    None
                } else {
                    Some(b.apply(p))
                }
            }
        }
    }

    pub fn kind(&self) -> &'static str {
        match self {
            Transform::Linear(_) => "linear",
            Transform::BSpline(_) => "bspline",
        }
    }
}

/// How a transform combines with the initial transform preceding it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Combine {
    /// `T(x) = T_this(T_initial(x))`.
    Compose,
    /// `T(x) = T_initial(x) + T_this(x) − x`. Note both are evaluated at the *original* `x`.
    Add,
}

/// One `TransformParameters` file **and the initial-transform chain hanging off it**, ordered
/// initial-first.
///
/// The `Combine` of the first step is meaningless (it has no initial transform) and is ignored.
#[derive(Debug, Clone, PartialEq)]
pub struct ElastixTransform {
    pub steps: Vec<(Transform, Combine)>,
    pub source: PathBuf,
}

/// Resolve an `InitialTransformParametersFileName` against the file that named it.
///
/// Two rules, in order:
///
/// 1. **As recorded**, relative to the naming file's own directory. This is what `transformix`
///    does, and it is why `navis` has to copy files into a temp dir and `chdir` there.
/// 2. If that does not exist, **its basename, in the naming file's directory.**
///
/// Rule 2 is the whole of `navis`'s `copy_files`, done properly. Elastix records the initial
/// transform's path *as it was on the machine that ran the registration* — routinely an absolute
/// path like `/home/someone/scratch/TransformParameters.0.txt`, which of course is not there when
/// you receive the files. `transformix` simply fails; `navis` works around it by copying every
/// file into one directory and rewriting nothing, which only works because the basename is then
/// findable. We just look for the basename.
///
/// This can only *rescue* a lookup that would otherwise fail — rule 1 still wins whenever it
/// resolves, so a file that loads today loads identically tomorrow.
fn resolve_initial(named: &str, from: &Path) -> PathBuf {
    let dir = from.parent().unwrap_or(Path::new("."));
    let as_recorded = {
        let p = PathBuf::from(named);
        if p.is_absolute() {
            p
        } else {
            dir.join(p)
        }
    };
    if as_recorded.exists() {
        return as_recorded;
    }
    match Path::new(named).file_name() {
        Some(base) => dir.join(base),
        None => as_recorded,
    }
}

/// Walk a file and the `InitialTransformParametersFileName` chain hanging off it, innermost last.
///
/// Shared by [`ElastixTransform::from_path`] and [`probe_invertible`] so the two cannot drift
/// apart on cycle detection, path resolution, or what counts as a loadable file. `headers_only`
/// skips tokenising the coefficient array — see [`Params::parse_headers`].
fn walk_chain(
    path: &Path,
    headers_only: bool,
    visit: &mut dyn FnMut(&Params) -> Result<(), ElastixError>,
) -> Result<(), ElastixError> {
    let mut seen: HashSet<PathBuf> = HashSet::new();
    let mut cur = path.to_path_buf();

    loop {
        let canon = cur.canonicalize().unwrap_or_else(|_| cur.clone());
        if !seen.insert(canon) {
            return Err(ElastixError::CircularChain {
                path: cur.display().to_string(),
            });
        }

        let text = std::fs::read_to_string(&cur).map_err(|e| ElastixError::Io {
            path: cur.display().to_string(),
            msg: e.to_string(),
        })?;
        let name = cur.display().to_string();
        let p = if headers_only {
            Params::parse_headers(&text, &name)
        } else {
            Params::parse(&text, &name)
        };
        visit(&p)?;

        let init = p.str_or("InitialTransformParametersFileName", "NoInitialTransform");
        if init == "NoInitialTransform" || init.is_empty() {
            return Ok(());
        }
        cur = resolve_initial(&init, &cur);
    }
}

/// Whether [`ElastixTransform::from_path`] would yield an invertible transform — **without
/// reading the coefficients.**
///
/// A `TransformParameters` file is not invertible exactly when some step in its chain combines
/// via `Add`: `T(x) = T_initial(x) + T_this(x) - x` does not decompose into invertible hops.
/// That fact lives in a six-byte key, but it sits *after* a coefficient array that can run to
/// 56 MB — so answering it honestly used to mean parsing the whole file.
///
/// This reads the same chain, applies the same validation, and skips only the coefficients. It
/// exists for callers that must decide something about *many* files up front — a bridging graph
/// with 52 registrations in it cannot afford 52 full parses just to label its edges.
///
/// Errors for anything that would not load at all (missing, not elastix, unsupported kind,
/// binary parameters, circular chain), so `Ok(true)` is a real promise, not an optimistic guess.
pub fn probe_invertible(path: &Path) -> Result<bool, ElastixError> {
    let mut combines: Vec<Combine> = Vec::new();
    walk_chain(path, /* headers_only = */ true, &mut |p| {
        combines.push(parse_header(p)?.combine);
        Ok(())
    })?;
    combines.reverse(); // initial first, matching `ElastixTransform::steps`
                        // The first step's `Combine` is meaningless — it has nothing to combine
                        // *with* — exactly as in `ElastixTransform::has_add`.
    Ok(!combines.iter().skip(1).any(|c| *c == Combine::Add))
}

impl ElastixTransform {
    /// Read a file, following `InitialTransformParametersFileName` to the root of the chain.
    pub fn from_path(path: &Path) -> Result<Self, ElastixError> {
        let mut steps: Vec<(Transform, Combine)> = Vec::new();
        walk_chain(path, /* headers_only = */ false, &mut |p| {
            steps.push(parse_transform(p)?);
            Ok(())
        })?;
        steps.reverse(); // initial first
        Ok(ElastixTransform {
            steps,
            source: path.to_path_buf(),
        })
    }

    /// `None` only under [`OutOfBounds::Nan`], when a point leaves a B-spline's valid region.
    #[inline]
    fn apply(&self, p: [f64; 3], oob: OutOfBounds) -> Option<[f64; 3]> {
        let mut cur = p;
        for (i, (tr, combine)) in self.steps.iter().enumerate() {
            cur = if i == 0 || *combine == Combine::Compose {
                tr.apply(cur, oob)?
            } else {
                // Add: T(x) = T_initial(x) + T_this(x) - x, both evaluated at the ORIGINAL x.
                let t = tr.apply(p, oob)?;
                [
                    cur[0] + t[0] - p[0],
                    cur[1] + t[1] - p[1],
                    cur[2] + t[2] - p[2],
                ]
            };
        }
        Some(cur)
    }

    /// The value entering each forward step, for a point `p`. `prefix[0] == p`.
    ///
    /// This is how an `initial_guess` -- which the caller gives in *source* space -- is turned
    /// into a seed for a B-spline that actually lives further down the chain, behind an affine.
    /// Seeding the spline with the raw source-space guess would be quietly wrong.
    ///
    /// Only called on `Compose` chains (an `Add` chain is refused before we get here), so the
    /// steps really do apply one after another.
    fn forward_prefix(&self, p: [f64; 3]) -> Vec<[f64; 3]> {
        let mut out = Vec::with_capacity(self.steps.len());
        let mut cur = p;
        for (tr, _) in &self.steps {
            out.push(cur);
            // `Identity` never fails, so the unwrap cannot fire.
            cur = tr
                .apply(cur, OutOfBounds::Identity)
                .expect("Identity never fails");
        }
        out
    }

    pub fn n_steps(&self) -> usize {
        self.steps.len()
    }

    /// Does this transform carry an `Add` step? If so it cannot be undone step-by-step, so the
    /// whole chain refuses to invert. (The first step's `Combine` is meaningless — it has no
    /// initial transform to combine with — so it is skipped.)
    pub fn has_add(&self) -> bool {
        self.steps.iter().skip(1).any(|(_, c)| *c == Combine::Add)
    }

    /// The first `Linear` step, if any — the affine most files start from.
    pub fn linear(&self) -> Option<&Linear> {
        self.steps.iter().find_map(|(t, _)| match t {
            Transform::Linear(l) => Some(l),
            _ => None,
        })
    }

    /// Every B-spline step, outermost last.
    pub fn splines(&self) -> Vec<&BSpline> {
        self.steps
            .iter()
            .filter_map(|(t, _)| match t {
                Transform::BSpline(b) => Some(b),
                _ => None,
            })
            .collect()
    }
}

/// The transform kinds we implement. The single source of truth for the `(Transform …)` value —
/// both the full parse and the header-only probe resolve through here, so they cannot disagree
/// about what is loadable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Affine,
    Translation,
    Euler,
    Similarity,
    BSpline,
}

fn kind_of(s: &str) -> Result<Kind, ElastixError> {
    Ok(match s {
        "AffineTransform" | "AdvancedAffineTransform" | "MatrixOffsetTransformBase" => Kind::Affine,
        "TranslationTransform" | "AdvancedTranslationTransform" => Kind::Translation,
        "EulerTransform" | "AdvancedEulerTransform" => Kind::Euler,
        "SimilarityTransform" | "AdvancedSimilarityTransform" => Kind::Similarity,
        "BSplineTransform" | "RecursiveBSplineTransform" | "AdvancedBSplineTransform" => {
            Kind::BSpline
        }
        other => {
            return Err(ElastixError::UnsupportedTransform {
                kind: other.to_string(),
            })
        }
    })
}

/// Everything about a transform that can be known *without* reading its coefficients.
///
/// Every rejection the full parse makes on grounds other than the numbers themselves — not an
/// elastix file, binary parameters, an unsupported kind, a non-cubic spline — is made here, so a
/// probe that succeeds is a real promise that the file loads.
struct Header {
    kind: Kind,
    combine: Combine,
}

fn parse_header(p: &Params) -> Result<Header, ElastixError> {
    let kind = p.get("Transform").and_then(|v| v.first()).ok_or_else(|| {
        ElastixError::NotElastix {
            path: p.path.clone(),
        }
    })?;
    let kind = kind_of(kind)?;

    if p.bool_or("UseBinaryFormatForTransformationParameters", false) {
        return Err(ElastixError::BinaryParameters {
            path: p.path.clone(),
        });
    }

    let order = p.ints("BSplineTransformSplineOrder").map(|v| v[0]).unwrap_or(3);
    if kind == Kind::BSpline && order != 3 {
        return Err(ElastixError::UnsupportedSplineOrder { got: order });
    }

    let combine = match p.str_or("HowToCombineTransforms", "Compose").as_str() {
        "Add" => Combine::Add,
        _ => Combine::Compose,
    };
    Ok(Header { kind, combine })
}

fn parse_transform(p: &Params) -> Result<(Transform, Combine), ElastixError> {
    let Header { kind, combine } = parse_header(p)?;

    let par = p.floats("TransformParameters")?;
    let c = p.center();

    let tr = match kind {
        Kind::Affine => Transform::Linear(Linear::from_affine(&par, c)?),
        Kind::Translation => Transform::Linear(Linear::from_translation(&par)?),
        Kind::Euler => Transform::Linear(Linear::from_euler(
            &par,
            c,
            p.bool_or("ComputeZYX", false),
        )?),
        Kind::Similarity => Transform::Linear(Linear::from_similarity(&par, c)?),
        Kind::BSpline => Transform::BSpline(BSpline::from_params(p, &par)?),
    };
    Ok((tr, combine))
}

/// One or more [`ElastixTransform`]s applied nose-to-tail.
///
/// A chain holds only the *parse*. Which way each hop is travelled is a property of the
/// traversal, not of the transform — see [`XformOpts::invert`] — so one `Chain` serves every
/// direction, and an inverted view costs nothing to obtain. That matters here: BANC's
/// `BANC_to_template.txt` is 56 MB, and re-reading it just to walk it backwards would be absurd.
#[derive(Debug, Clone, PartialEq)]
pub struct Chain {
    pub xforms: Vec<ElastixTransform>,
}

impl Chain {
    pub fn new(xforms: Vec<ElastixTransform>) -> Result<Self, ElastixError> {
        if xforms.is_empty() {
            return Err(ElastixError::EmptyChain);
        }
        Ok(Chain { xforms })
    }

    pub fn from_paths(paths: &[PathBuf]) -> Result<Self, ElastixError> {
        let xforms = paths
            .iter()
            .map(|p| ElastixTransform::from_path(p))
            .collect::<Result<Vec<_>, _>>()?;
        Chain::new(xforms)
    }

    pub fn n_transforms(&self) -> usize {
        self.xforms.len()
    }

    /// Whether [`inverse_transform_points`] can run — i.e. whether every hop it would have to
    /// traverse backwards is free of `Add` steps.
    ///
    /// `invert` is the same per-hop flag set the traversal will use: a hop already flagged
    /// `invert` is the one a whole-chain inversion ends up running *forwards*, and forwards,
    /// `Add` is fine. `None` (all-forward) is the question the bindings actually ask, and the
    /// one worth reasoning about: "can I invert this parse at all?"
    ///
    /// Exists so the bindings can raise a decent error *before* calling in: extendr cannot carry a
    /// Rust panic's message across to R, so R has to ask first.
    pub fn is_invertible(&self, invert: Option<&[bool]>) -> bool {
        let Ok(flags) = self.flags(invert) else {
            return false;
        };
        self.xforms
            .iter()
            .zip(flags)
            .all(|(x, inv)| inv || !x.has_add())
    }

    /// Resolve the per-hop direction flags, defaulting to all-forward.
    fn flags(&self, invert: Option<&[bool]>) -> Result<Vec<bool>, ElastixError> {
        match invert {
            None => Ok(vec![false; self.xforms.len()]),
            Some(f) if f.len() == self.xforms.len() => Ok(f.to_vec()),
            Some(f) => Err(ElastixError::InvertLen {
                got: f.len(),
                want: self.xforms.len(),
            }),
        }
    }
}

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

/// What to do with points that fall outside a B-spline's valid region.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutOfBounds {
    /// Return them unchanged. **This is what `transformix` does**, so it is the default: swapping
    /// this module in for the binary must not silently change anyone's results.
    ///
    /// The catch is that it is *silent* — a neuron straddling the grid edge comes back partly
    /// transformed and looks perfectly fine. Use [`OutOfBounds::Nan`] when you would rather see
    /// the boundary than trust it.
    Identity,
    /// Return `NaN`, the way [`crate::cmtk`] reports points outside its domain.
    Nan,
}

#[derive(Clone, Copy)]
pub struct XformOpts<'a> {
    pub out_of_bounds: OutOfBounds,
    /// Traverse transform `i` backwards. `None` (the default) is all-forward.
    ///
    /// This is *per hop*, and it is not the same knob as [`inverse_transform_points`], which
    /// inverts the whole composition (reversing the order **and** flipping every hop). A
    /// bridging graph routes through transforms in whichever direction each edge was found, so
    /// a chain may need some hops forwards and others backwards — which no whole-chain
    /// inversion can express.
    pub invert: Option<&'a [bool]>,
    pub threads: Option<usize>,
    pub progress: bool,
    pub cancel: Option<&'a AtomicBool>,
}

impl Default for XformOpts<'_> {
    fn default() -> Self {
        XformOpts {
            out_of_bounds: OutOfBounds::Identity,
            invert: None,
            threads: None,
            progress: false,
            cancel: None,
        }
    }
}

#[derive(Clone, Copy)]
pub struct InverseOpts<'a> {
    pub out_of_bounds: OutOfBounds,
    pub max_iter: usize,
    /// Rounds of the fixed-point pre-seed `x ← target − disp(x)`. Zero starts LM at the target,
    /// which fails wherever the deformation is large — see [`BSpline::solve_inverse`].
    pub seed_iter: usize,
    /// Stop once the step falls below this.
    pub tolerance: f64,
    /// Accept a root only if the residual is within this, in world units; otherwise `NaN`.
    pub accuracy: f64,
    /// Roughly how many points to put in the global seed lattice, the last-resort start for points
    /// the cheap seeds fail on. Built once per call. Zero disables it — which costs BANC's warp
    /// four times as many unconverged points. See [`BSpline::seed_lattice`].
    pub lattice_points: usize,
    /// The same per-hop flags as [`XformOpts::invert`], composed with the whole-chain inversion:
    /// hop `i` runs forwards here exactly when `invert[i]` is set.
    pub invert: Option<&'a [bool]>,
    pub threads: Option<usize>,
    pub progress: bool,
    pub cancel: Option<&'a AtomicBool>,
}

impl Default for InverseOpts<'_> {
    fn default() -> Self {
        InverseOpts {
            out_of_bounds: OutOfBounds::Identity,
            max_iter: 50,
            seed_iter: 8,
            tolerance: 1e-9,
            accuracy: 1e-3,
            lattice_points: 16_000,
            invert: None,
            threads: None,
            progress: false,
            cancel: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Batch drivers
// ---------------------------------------------------------------------------

/// One step of an *inverted* transform, with the linear inverses and the seed lattice already
/// computed — both once per call rather than once per point.
enum InvStep<'a> {
    Linear(Linear),
    BSpline {
        spline: &'a BSpline,
        /// Where this spline sits in the *forward* step list — which is what says how far an
        /// `initial_guess` has to be pushed forward before it means anything to this solver.
        fwd_index: usize,
        lattice: SeedLattice,
    },
}

/// One hop of a chain, resolved for the direction we are actually travelling.
///
/// Either way we keep the whole [`ElastixTransform`]: forwards because `Add` needs the point that
/// entered *that* transform, backwards because an `initial_guess` has to be pushed through the
/// forward steps preceding each spline.
enum Hop<'a> {
    Forward(&'a ElastixTransform),
    Inverse {
        xf: &'a ElastixTransform,
        steps: Vec<InvStep<'a>>,
    },
}

/// Resolve `chain` into ordered hops. `reverse` walks it backwards and flips every hop — i.e. it
/// inverts the whole composition. Linear inverses and seed lattices are computed here, once per
/// call rather than once per point.
fn plan_hops<'a>(
    chain: &'a Chain,
    invert: Option<&[bool]>,
    reverse: bool,
    lattice_points: usize,
) -> Result<Vec<Hop<'a>>, ElastixError> {
    let flags = chain.flags(invert)?;
    let n = chain.xforms.len();
    let order: Vec<usize> = if reverse {
        (0..n).rev().collect()
    } else {
        (0..n).collect()
    };

    let mut hops = Vec::with_capacity(n);
    for i in order {
        let xf = &chain.xforms[i];
        let backwards = flags[i] != reverse; // XOR

        if !backwards {
            hops.push(Hop::Forward(xf));
            continue;
        }

        // Inverting means undoing each step in reverse order -- which only works for `Compose`.
        if xf.has_add() {
            return Err(ElastixError::NotInvertible);
        }
        let mut steps = Vec::with_capacity(xf.steps.len());
        for (j, (tr, _)) in xf.steps.iter().enumerate().rev() {
            steps.push(match tr {
                Transform::Linear(l) => {
                    InvStep::Linear(l.inverse().ok_or(ElastixError::SingularLinear)?)
                }
                Transform::BSpline(b) => InvStep::BSpline {
                    spline: b,
                    fwd_index: j,
                    lattice: b.seed_lattice(lattice_points),
                },
            });
        }
        hops.push(Hop::Inverse { xf, steps });
    }
    Ok(hops)
}

/// Push one point through every hop.
///
/// `guess` seeds the first *inverted* hop only — past that we are in an intermediate space the
/// caller knows nothing about, so there is nothing sensible to seed with. Within that hop it
/// seeds every spline, each in the right coordinate frame (see [`ElastixTransform::forward_prefix`]).
#[inline]
fn run_hops(
    hops: &[Hop],
    p: [f64; 3],
    oob: OutOfBounds,
    iopts: &InverseOpts,
    mut guess: Option<[f64; 3]>,
) -> Option<[f64; 3]> {
    let mut cur = p;
    for hop in hops {
        cur = match hop {
            Hop::Forward(xf) => xf.apply(cur, oob)?,
            Hop::Inverse { xf, steps } => {
                let prefix = guess.take().map(|g| xf.forward_prefix(g));
                let mut c = cur;
                for s in steps {
                    c = match s {
                        InvStep::Linear(l) => l.apply(c),
                        InvStep::BSpline {
                            spline,
                            fwd_index,
                            lattice,
                        } => {
                            let x0 = prefix.as_ref().map(|pre| pre[*fwd_index]);
                            spline.solve_inverse(c, x0, Some(lattice), iopts)?
                        }
                    };
                }
                c
            }
        };
    }
    Some(cur)
}

fn points_to_vec(points: ArrayView2<f64>) -> Result<Vec<[f64; 3]>, ElastixError> {
    if points.ncols() != 3 {
        return Err(ElastixError::BadShape {
            got: points.shape().to_vec(),
        });
    }
    Ok(points.rows().into_iter().map(|r| [r[0], r[1], r[2]]).collect())
}

fn finish(res: Vec<[f64; 3]>, cancel: Option<&AtomicBool>) -> Result<Array2<f64>, ElastixError> {
    if is_cancelled(cancel) {
        return Err(ElastixError::Cancelled);
    }
    let n = res.len();
    let flat: Vec<f64> = res.into_iter().flatten().collect();
    Ok(Array2::from_shape_vec((n, 3), flat).expect("(n, 3) from 3n values"))
}

/// Drive `points` through `hops` in parallel, writing `NaN` for rows that fail.
#[allow(clippy::too_many_arguments)]
fn drive(
    hops: &[Hop],
    pts: &[[f64; 3]],
    oob: OutOfBounds,
    iopts: &InverseOpts,
    guesses: Option<&[[f64; 3]]>,
    threads: Option<usize>,
    progress: bool,
    cancel: Option<&AtomicBool>,
) -> Vec<[f64; 3]> {
    let n = pts.len();
    let mut res: Vec<[f64; 3]> = vec![[f64::NAN; 3]; n];
    let bar = progress.then(|| make_bar("Elastix", n as u64));

    with_pool(threads, || {
        res.par_chunks_mut(CHUNK).enumerate().for_each(|(ci, out)| {
            if is_cancelled(cancel) {
                return;
            }
            let start = ci * CHUNK;
            for (k, dst) in out.iter_mut().enumerate() {
                let i = start + k;
                let guess = guesses.map(|g| g[i]);
                if let Some(v) = run_hops(hops, pts[i], oob, iopts, guess) {
                    *dst = v;
                }
            }
            if let Some(b) = &bar {
                b.inc(out.len() as u64);
            }
        });
    });

    if let Some(b) = bar {
        b.finish_and_clear();
    }
    res
}

/// Forward-transform `points` (an `(N, 3)` array) through `chain`.
///
/// Under [`OutOfBounds::Identity`] — the default, and what `transformix` does — no row can fail.
pub fn transform_points(
    chain: &Chain,
    points: ArrayView2<f64>,
    opts: XformOpts,
) -> Result<Array2<f64>, ElastixError> {
    let pts = points_to_vec(points)?;
    // Only consulted if the chain traverses a hop backwards; a purely forward chain holds no
    // iterative steps, and then the lattice is never built either.
    let iopts = InverseOpts {
        out_of_bounds: opts.out_of_bounds,
        ..InverseOpts::default()
    };
    let hops = plan_hops(chain, opts.invert, false, iopts.lattice_points)?;
    let res = drive(
        &hops,
        &pts,
        opts.out_of_bounds,
        &iopts,
        None,
        opts.threads,
        opts.progress,
        opts.cancel,
    );
    finish(res, opts.cancel)
}

/// Invert `chain` on `points` — something Elastix itself cannot do.
///
/// The guarantee is **forward-consistency**: the point returned satisfies
/// `transform_points(chain, x) == points` to within `opts.accuracy`. It is *not* guaranteed to be
/// the point you started from, because a B-spline warp need not be injective — a strongly folded
/// registration maps several points to the same place, and no inverse can recover which one you
/// meant. Rows with no preimage at all come back `NaN`.
pub fn inverse_transform_points(
    chain: &Chain,
    points: ArrayView2<f64>,
    initial_guess: Option<ArrayView2<f64>>,
    opts: InverseOpts,
) -> Result<Array2<f64>, ElastixError> {
    let pts = points_to_vec(points)?;
    let guesses = initial_guess.map(points_to_vec).transpose()?;
    if let Some(g) = &guesses {
        if g.len() != pts.len() {
            return Err(ElastixError::GuessLen {
                got: g.len(),
                want: pts.len(),
            });
        }
    }
    let hops = plan_hops(chain, opts.invert, true, opts.lattice_points)?;
    let res = drive(
        &hops,
        &pts,
        opts.out_of_bounds,
        &opts,
        guesses.as_deref(),
        opts.threads,
        opts.progress,
        opts.cancel,
    );
    finish(res, opts.cancel)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::atomic::Ordering;

    const DATA: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/elastix");

    /// Measured margin against `transformix` is 5e-7 -- its own print precision. If a change
    /// lands at 1e-5, something is subtly wrong; do not loosen this.
    const ATOL: f64 = 1e-4;

    fn load(name: &str) -> ElastixTransform {
        ElastixTransform::from_path(Path::new(&format!("{DATA}/{name}"))).unwrap()
    }

    fn chain_of(name: &str) -> Chain {
        Chain::new(vec![load(name)]).unwrap()
    }

    /// `transformix`'s own output, keyed by case.
    fn golden() -> HashMap<String, Vec<[f64; 3]>> {
        let mut rdr = csv::Reader::from_path(format!("{DATA}/transformix_golden.csv")).unwrap();
        let mut out: HashMap<String, Vec<(usize, [f64; 3])>> = HashMap::new();
        for rec in rdr.records() {
            let r = rec.unwrap();
            let i: usize = r[1].parse().unwrap();
            let p = [
                r[2].parse().unwrap(),
                r[3].parse().unwrap(),
                r[4].parse().unwrap(),
            ];
            out.entry(r[0].to_string()).or_default().push((i, p));
        }
        out.into_iter()
            .map(|(k, mut v)| {
                v.sort_by_key(|(i, _)| *i);
                (k, v.into_iter().map(|(_, p)| p).collect())
            })
            .collect()
    }

    fn arr(pts: &[[f64; 3]]) -> Array2<f64> {
        Array2::from_shape_vec((pts.len(), 3), pts.iter().flatten().copied().collect()).unwrap()
    }

    fn assert_close(got: &Array2<f64>, want: &[[f64; 3]], atol: f64, what: &str) {
        assert_eq!(got.nrows(), want.len(), "{what}: row count");
        for (i, w) in want.iter().enumerate() {
            for j in 0..3 {
                let g = got[[i, j]];
                assert!(
                    (g - w[j]).abs() <= atol,
                    "{what}: row {i} col {j}: got {g}, want {}, |d| = {}",
                    w[j],
                    (g - w[j]).abs()
                );
            }
        }
    }

    fn xform(name: &str, pts: &[[f64; 3]]) -> Array2<f64> {
        transform_points(&chain_of(name), arr(pts).view(), XformOpts::default()).unwrap()
    }

    // ---- against transformix ----

    /// Every fixture, forward, against the real binary's output. This is the headline test:
    /// it pins the Euler order, the component-major coefficient layout, the valid-region rule,
    /// the identity-outside rule, and both combination modes, all at once.
    #[test]
    fn forward_matches_transformix() {
        let g = golden();
        let input = &g["input"];
        for case in [
            "affine",
            "translation",
            "euler",
            "euler_zyx",
            "similarity",
            "bspline",
            "add",
        ] {
            let got = xform(&format!("{case}.txt"), input);
            assert_close(&got, &g[case], ATOL, case);
        }
    }

    /// `euler.txt` and `euler_zyx.txt` carry identical angles and differ only in `ComputeZYX`.
    /// If the two rotation orders were the same -- or if we picked the intuitive `Rz·Ry·Rx` for
    /// the default -- this would not be able to tell them apart.
    #[test]
    fn compute_zyx_selects_a_different_rotation_order() {
        let g = golden();
        let input = &g["input"];
        let a = xform("euler.txt", input);
        let b = xform("euler_zyx.txt", input);
        let spread = (0..a.nrows())
            .flat_map(|i| (0..3).map(move |j| (i, j)))
            .map(|(i, j)| (a[[i, j]] - b[[i, j]]).abs())
            .fold(0.0f64, f64::max);
        assert!(spread > 1.0, "the two orders barely differ ({spread})");

        let Transform::Linear(l) = &load("euler.txt").steps[0].0 else {
            panic!("expected a Linear")
        };
        let (rx, ry, rz) = (rot_x(0.12), rot_y(-0.2), rot_z(0.31));
        let want = mat3_mul(&rz, &mat3_mul(&rx, &ry)); // Rz·Rx·Ry, NOT Rz·Ry·Rx
        for (got_row, want_row) in l.a.iter().zip(want.iter()) {
            for (g, w) in got_row.iter().zip(want_row.iter()) {
                assert!((g - w).abs() < 1e-12);
            }
        }
    }

    // ---- parser ----

    #[test]
    fn parser_handles_comments_and_quotes() {
        let p = Params::parse(
            "// leading comment\n(Transform \"AffineTransform\")  // trailing\n\
             (NumberOfParameters 12)\n\nnot a group\n(Empty)\n",
            "x",
        );
        assert_eq!(p.get("Transform").unwrap(), &["AffineTransform".to_string()]);
        assert_eq!(p.get("NumberOfParameters").unwrap(), &["12".to_string()]);
        assert!(p.get("Empty").unwrap().is_empty());
        assert!(p.get("nope").is_none());
    }

    /// Repeated keys: the last wins, as elastix itself does.
    #[test]
    fn repeated_keys_take_the_last() {
        let p = Params::parse("(Spacing 1 1 1)\n(Spacing 2 2 2)\n", "x");
        assert_eq!(p.triple("Spacing").unwrap(), [2.0, 2.0, 2.0]);
    }

    #[test]
    fn non_elastix_file_is_rejected() {
        // `navis-flybrains` ships a `template_to_BANC.txt` that holds nothing but a filename.
        let p = Params::parse("3_elastix_Bspline_fine.txt\n", "pointer.txt");
        assert!(matches!(
            parse_transform(&p),
            Err(ElastixError::NotElastix { .. })
        ));
    }

    #[test]
    fn binary_parameters_are_rejected() {
        let p = Params::parse(
            "(Transform \"AffineTransform\")\n(TransformParameters 1 0 0 0 1 0 0 0 1 0 0 0)\n\
             (UseBinaryFormatForTransformationParameters \"true\")\n",
            "b.txt",
        );
        assert!(matches!(
            parse_transform(&p),
            Err(ElastixError::BinaryParameters { .. })
        ));
    }

    #[test]
    fn unsupported_transform_type_is_rejected() {
        let p = Params::parse(
            "(Transform \"SplineKernelTransform\")\n(TransformParameters 1 2 3)\n",
            "k.txt",
        );
        assert!(matches!(
            parse_transform(&p),
            Err(ElastixError::UnsupportedTransform { kind }) if kind == "SplineKernelTransform"
        ));
    }

    #[test]
    fn wrong_parameter_count_is_rejected() {
        let p = Params::parse(
            "(Transform \"AffineTransform\")\n(TransformParameters 1 0 0)\n",
            "a.txt",
        );
        assert!(matches!(
            parse_transform(&p),
            Err(ElastixError::ParamCount { got: 3, want: 12, .. })
        ));
    }

    #[test]
    fn non_cubic_spline_order_is_rejected() {
        let p = Params::parse(
            "(Transform \"BSplineTransform\")\n(TransformParameters 0 0 0)\n\
             (BSplineTransformSplineOrder 2)\n(GridSize 4 4 4)\n",
            "s.txt",
        );
        assert!(matches!(
            parse_transform(&p),
            Err(ElastixError::UnsupportedSplineOrder { got: 2 })
        ));
    }

    #[test]
    fn missing_file_is_an_io_error() {
        assert!(matches!(
            ElastixTransform::from_path(Path::new("/nonexistent/nope.txt")),
            Err(ElastixError::Io { .. })
        ));
    }

    // ---- chain ----

    /// `bspline.txt` names `affine.txt` by a bare relative filename. Resolving it against the
    /// transform file's own directory (not the CWD) is what transformix does.
    #[test]
    fn initial_transform_resolves_against_the_files_own_directory() {
        let xf = load("bspline.txt");
        assert_eq!(xf.n_steps(), 2);
        assert!(matches!(xf.steps[0].0, Transform::Linear(_)), "initial first");
        assert!(matches!(xf.steps[1].0, Transform::BSpline(_)));
        assert_eq!(xf.steps[1].1, Combine::Compose);
        assert_eq!(xf.splines().len(), 1);
        assert!(xf.linear().is_some());
    }

    /// The B-spline hop is `spline(affine(p))` -- so applying the two fixtures by hand, in that
    /// order, must reproduce it.
    #[test]
    fn compose_is_spline_of_affine() {
        let g = golden();
        let mid = xform("affine.txt", &g["input"]);
        let mid: Vec<[f64; 3]> = mid.rows().into_iter().map(|r| [r[0], r[1], r[2]]).collect();

        let spline = load("bspline.txt");
        let Transform::BSpline(b) = &spline.steps[1].0 else {
            panic!("expected a BSpline")
        };
        let by_hand: Vec<[f64; 3]> = mid.iter().map(|&p| b.apply(p)).collect();
        assert_close(&arr(&by_hand), &g["bspline"], ATOL, "compose");
    }

    /// `add.txt`: `T(x) = T_initial(x) + T_this(x) − x`, with **both** evaluated at the original
    /// `x`. Composing them instead gives a visibly different answer.
    #[test]
    fn add_is_not_compose() {
        let g = golden();
        let got = xform("add.txt", &g["input"]);
        assert_close(&got, &g["add"], ATOL, "add");

        let composed = {
            let xf = load("add.txt");
            let (Transform::Linear(a), Transform::Linear(b)) =
                (&xf.steps[0].0, &xf.steps[1].0)
            else {
                panic!("expected two Linears")
            };
            let v: Vec<[f64; 3]> = g["input"].iter().map(|&p| b.apply(a.apply(p))).collect();
            arr(&v)
        };
        let spread = (0..got.nrows())
            .flat_map(|i| (0..3).map(move |j| (i, j)))
            .map(|(i, j)| (got[[i, j]] - composed[[i, j]]).abs())
            .fold(0.0f64, f64::max);
        assert!(spread > 1.0, "Add and Compose agree ({spread}) -- test is blind");
    }

    // ---- header-only probe ----

    /// The probe and the full parse must never disagree. They share the scanner, the kind list,
    /// the validation and the chain walk precisely so that this holds by construction — this
    /// test is the guard on that.
    #[test]
    fn probe_invertible_agrees_with_the_full_parse_on_every_fixture() {
        for name in [
            "affine.txt",
            "translation.txt",
            "euler.txt",
            "euler_zyx.txt",
            "similarity.txt",
            "bspline.txt",
            "add.txt",
        ] {
            let path = PathBuf::from(format!("{DATA}/{name}"));
            let probed = probe_invertible(&path).unwrap();
            let parsed = !load(name).has_add();
            assert_eq!(probed, parsed, "{name}: probe {probed}, full parse {parsed}");
        }
    }

    #[test]
    fn probe_invertible_spots_an_add_chain_without_reading_the_coefficients() {
        assert!(probe_invertible(&PathBuf::from(format!("{DATA}/bspline.txt"))).unwrap());
        assert!(!probe_invertible(&PathBuf::from(format!("{DATA}/add.txt"))).unwrap());
    }

    /// `Ok(true)` has to be a real promise: everything the full parse would reject on grounds
    /// other than the numbers themselves is rejected here too.
    #[test]
    fn probe_invertible_rejects_what_would_not_load() {
        let dir = std::env::temp_dir().join("fastcore_elastix_probe_reject");
        std::fs::create_dir_all(&dir).unwrap();

        std::fs::write(dir.join("not_elastix.txt"), "just some text\n").unwrap();
        assert!(matches!(
            probe_invertible(&dir.join("not_elastix.txt")),
            Err(ElastixError::NotElastix { .. })
        ));

        std::fs::write(
            dir.join("unsupported.txt"),
            "(Transform \"SplineKernelTransform\")\n(TransformParameters 1 2 3)\n",
        )
        .unwrap();
        assert!(matches!(
            probe_invertible(&dir.join("unsupported.txt")),
            Err(ElastixError::UnsupportedTransform { .. })
        ));

        std::fs::write(
            dir.join("order5.txt"),
            "(Transform \"BSplineTransform\")\n(BSplineTransformSplineOrder 5)\n\
             (TransformParameters 1 2 3)\n",
        )
        .unwrap();
        assert!(matches!(
            probe_invertible(&dir.join("order5.txt")),
            Err(ElastixError::UnsupportedSplineOrder { got: 5 })
        ));

        assert!(matches!(
            probe_invertible(&dir.join("missing.txt")),
            Err(ElastixError::Io { .. })
        ));
        std::fs::remove_dir_all(&dir).ok();
    }

    // ---- initial-transform path resolution ----

    /// Elastix records the initial transform's path as it was on the machine that ran the
    /// registration — routinely an absolute path that does not exist when you receive the files.
    /// We fall back to its basename in the naming file's own directory, which is exactly what
    /// `navis`'s `copy_files` achieves by copying everything into one directory.
    #[test]
    fn a_stale_absolute_initial_path_falls_back_to_its_basename() {
        let dir = std::env::temp_dir().join("fastcore_elastix_stale_abs");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("parent.txt"),
            "(Transform \"TranslationTransform\")\n(TransformParameters 1 2 3)\n",
        )
        .unwrap();
        std::fs::write(
            dir.join("child.txt"),
            "(Transform \"TranslationTransform\")\n(TransformParameters 10 20 30)\n\
             (InitialTransformParametersFileName \"/nowhere/on/this/machine/parent.txt\")\n",
        )
        .unwrap();

        let xf = ElastixTransform::from_path(&dir.join("child.txt")).unwrap();
        assert_eq!(xf.steps.len(), 2, "the initial transform must have been found");
        // initial first: translate by (1,2,3), then by (10,20,30).
        assert_eq!(xf.apply([0.0, 0.0, 0.0], OutOfBounds::Identity), Some([11.0, 22.0, 33.0]));

        // The probe walks the same chain, so it resolves the same way.
        assert!(probe_invertible(&dir.join("child.txt")).unwrap());
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The fallback may only ever *rescue* a lookup that would otherwise fail. A recorded path
    /// that resolves still wins, so a file that loads today loads identically tomorrow.
    #[test]
    fn a_resolvable_recorded_path_still_wins_over_the_basename() {
        let dir = std::env::temp_dir().join("fastcore_elastix_pref");
        let real = dir.join("real");
        std::fs::create_dir_all(&real).unwrap();

        // Two different `parent.txt`: one beside the child, one where the child actually points.
        std::fs::write(
            dir.join("parent.txt"),
            "(Transform \"TranslationTransform\")\n(TransformParameters 1 2 3)\n",
        )
        .unwrap();
        std::fs::write(
            real.join("parent.txt"),
            "(Transform \"TranslationTransform\")\n(TransformParameters 100 200 300)\n",
        )
        .unwrap();
        std::fs::write(
            dir.join("child.txt"),
            format!(
                "(Transform \"TranslationTransform\")\n(TransformParameters 0 0 0)\n\
                 (InitialTransformParametersFileName \"{}\")\n",
                real.join("parent.txt").display()
            ),
        )
        .unwrap();

        let xf = ElastixTransform::from_path(&dir.join("child.txt")).unwrap();
        assert_eq!(
            xf.apply([0.0, 0.0, 0.0], OutOfBounds::Identity),
            Some([100.0, 200.0, 300.0]),
            "the recorded path resolves, so it must be preferred over the neighbouring basename"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn circular_chain_is_caught() {
        let dir = std::env::temp_dir().join("fastcore_elastix_cycle");
        std::fs::create_dir_all(&dir).unwrap();
        for (name, init) in [("a.txt", "b.txt"), ("b.txt", "a.txt")] {
            std::fs::write(
                dir.join(name),
                format!(
                    "(Transform \"TranslationTransform\")\n(TransformParameters 1 2 3)\n\
                     (InitialTransformParametersFileName \"{init}\")\n"
                ),
            )
            .unwrap();
        }
        assert!(matches!(
            ElastixTransform::from_path(&dir.join("a.txt")),
            Err(ElastixError::CircularChain { .. })
        ));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn chain_of_two_equals_manual_double_application() {
        let g = golden();
        let once = xform("affine.txt", &g["input"]);
        let once: Vec<[f64; 3]> = once.rows().into_iter().map(|r| [r[0], r[1], r[2]]).collect();
        let twice = xform("affine.txt", &once);

        let chain = Chain::new(vec![load("affine.txt"), load("affine.txt")]).unwrap();
        let got = transform_points(&chain, arr(&g["input"]).view(), XformOpts::default()).unwrap();
        let want: Vec<[f64; 3]> = twice.rows().into_iter().map(|r| [r[0], r[1], r[2]]).collect();
        assert_close(&got, &want, 1e-9, "chain of two");
    }

    /// A hop marked `invert` is traversed backwards, so it undoes the forward one -- and must
    /// agree with `inverse_transform_points` exactly. One parse serves both.
    #[test]
    fn invert_flag_reverses_a_hop() {
        let g = golden();
        let chain = chain_of("bspline.txt");
        let fwd = xform("bspline.txt", &g["input"]);

        let got = transform_points(
            &chain,
            fwd.view(),
            XformOpts {
                invert: Some(&[true]),
                ..Default::default()
            },
        )
        .unwrap();
        let want =
            inverse_transform_points(&chain, fwd.view(), None, InverseOpts::default()).unwrap();
        assert_eq!(got, want);
    }

    #[test]
    fn invert_must_have_one_flag_per_transform() {
        let chain = chain_of("affine.txt");
        let err = transform_points(
            &chain,
            arr(&[[0.0, 0.0, 0.0]]).view(),
            XformOpts {
                invert: Some(&[true, false]),
                ..Default::default()
            },
        )
        .unwrap_err();
        assert!(matches!(err, ElastixError::InvertLen { got: 2, want: 1 }));
    }

    // ---- linear ----

    #[test]
    fn linear_inverse_round_trips() {
        for name in ["affine.txt", "euler.txt", "similarity.txt", "translation.txt"] {
            let Transform::Linear(l) = &load(name).steps[0].0 else {
                panic!("expected a Linear")
            };
            let inv = l.inverse().expect("invertible");
            for p in [[0.0, 0.0, 0.0], [50.0, 50.0, 40.0], [-13.0, 210.0, 7.5]] {
                let back = inv.apply(l.apply(p));
                for j in 0..3 {
                    assert!((back[j] - p[j]).abs() < 1e-10, "{name}: {back:?} vs {p:?}");
                }
            }
        }
    }

    #[test]
    fn singular_linear_has_no_inverse() {
        let l = Linear {
            a: [[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [1.0, 1.0, 1.0]], // row 2 = 2 * row 1
            t: [0.0; 3],
            c: [0.0; 3],
        };
        assert!(l.inverse().is_none());
    }

    /// The 4×4 the bindings hand out must agree with `apply`.
    #[test]
    fn as_array_agrees_with_apply() {
        let Transform::Linear(l) = &load("affine.txt").steps[0].0 else {
            panic!("expected a Linear")
        };
        let m = l.as_array();
        let p = [17.0, -4.0, 31.0];
        let got = l.apply(p);
        for i in 0..3 {
            let want: f64 = (0..3).map(|j| m[[i, j]] * p[j]).sum::<f64>() + m[[i, 3]];
            assert!((got[i] - want).abs() < 1e-12);
        }
    }

    // ---- b-spline ----

    fn fixture_spline() -> BSpline {
        let xf = load("bspline.txt");
        let Transform::BSpline(b) = &xf.steps[1].0 else {
            panic!("expected a BSpline")
        };
        b.clone()
    }

    /// The valid region is `[first + 1, first + size − 2)` -- half-open, and two cells short of
    /// the far edge, not one. Probed on each axis from either side.
    #[test]
    fn valid_region_is_one_in_and_two_short() {
        let b = fixture_spline();
        let mid = [
            b.origin[0] + 4.0 * b.spacing[0],
            b.origin[1] + 4.0 * b.spacing[1],
            b.origin[2] + 3.0 * b.spacing[2],
        ];
        for axis in 0..3 {
            let at = |u: f64| {
                let mut p = mid;
                p[axis] = b.origin[axis] + u * b.spacing[axis];
                p
            };
            let hi = b.size[axis] as f64 - 2.0;
            assert!(!b.in_region(at(1.0 - 1e-6)), "axis {axis}: below 1 is outside");
            assert!(b.in_region(at(1.0 + 1e-6)), "axis {axis}: above 1 is inside");
            assert!(b.in_region(at(hi - 1e-6)), "axis {axis}: below size-2 is inside");
            assert!(!b.in_region(at(hi + 1e-6)), "axis {axis}: at size-2 is outside");
            assert!(
                !b.in_region(at(b.size[axis] as f64 - 1.0)),
                "axis {axis}: size-1 is outside"
            );
        }
    }

    /// Outside the region the point comes back **unchanged** -- ITK's rule, and the exact
    /// opposite of CMTK's `FAILED`. Under `OutOfBounds::Nan` it comes back `NaN` instead.
    #[test]
    fn outside_the_region_is_the_identity_not_nan() {
        let b = fixture_spline();
        let far = [-500.0, -500.0, -500.0];
        assert!(!b.in_region(far));
        assert_eq!(b.apply(far), far);

        let pts = [far];
        let out = transform_points(&chain_of("bspline.txt"), arr(&pts).view(), XformOpts::default())
            .unwrap();
        assert!(out.iter().all(|v| v.is_finite()));

        let out = transform_points(
            &chain_of("bspline.txt"),
            arr(&pts).view(),
            XformOpts {
                out_of_bounds: OutOfBounds::Nan,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(out.iter().all(|v| v.is_nan()), "Nan mode should fail the row");
    }

    /// The analytic Jacobian, against central differences.
    #[test]
    fn jacobian_matches_finite_differences() {
        let b = fixture_spline();
        let p = [30.0, 25.0, 20.0];
        assert!(b.in_region(p));
        let (_, jac) = b.eval_with_jacobian(p);

        let h = 1e-5;
        for j in 0..3 {
            let (mut lo, mut hi) = (p, p);
            lo[j] -= h;
            hi[j] += h;
            let (f_lo, f_hi) = (b.eval_extrapolated(lo), b.eval_extrapolated(hi));
            for i in 0..3 {
                let fd = (f_hi[i] - f_lo[i]) / (2.0 * h);
                assert!(
                    (jac[i][j] - fd).abs() < 1e-6,
                    "J[{i}][{j}]: analytic {}, fd {fd}",
                    jac[i][j]
                );
            }
        }
    }

    /// The spline is a *displacement*, so `d/dx (x + disp(x)) = I + J`. Dropping the identity
    /// term still roughly converges, which is exactly what makes it easy to miss.
    #[test]
    fn jacobian_carries_the_identity_term() {
        let mut b = fixture_spline();
        b.coefficients.iter_mut().for_each(|c| *c = [0.0; 3]); // zero warp => J must be exactly I
        let (f, jac) = b.eval_with_jacobian([30.0, 25.0, 20.0]);
        assert_eq!(f, [30.0, 25.0, 20.0]);
        for (i, row) in jac.iter().enumerate() {
            for (j, v) in row.iter().enumerate() {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((v - want).abs() < 1e-12, "J[{i}][{j}] = {v}");
            }
        }
    }

    // ---- inverse ----

    /// The guarantee: whatever we hand back really is a preimage.
    #[test]
    fn inverse_is_forward_consistent() {
        let g = golden();
        let chain = chain_of("bspline.txt");
        let fwd = transform_points(&chain, arr(&g["input"]).view(), XformOpts::default()).unwrap();
        let back =
            inverse_transform_points(&chain, fwd.view(), None, InverseOpts::default()).unwrap();
        assert!(back.iter().all(|v| v.is_finite()), "no row should fail");

        let refwd = transform_points(&chain, back.view(), XformOpts::default()).unwrap();
        for i in 0..fwd.nrows() {
            for j in 0..3 {
                assert!(
                    (refwd[[i, j]] - fwd[[i, j]]).abs() < 1e-6,
                    "row {i}: f(inv(y)) = {} != y = {}",
                    refwd[[i, j]],
                    fwd[[i, j]]
                );
            }
        }
    }

    /// Which of the golden points enter the fixture's spline (i.e. are inside its grid *after*
    /// the initial affine)? Only these have a unique warp preimage -- see the ambiguity test.
    fn source_in_grid(pts: &[[f64; 3]]) -> Vec<bool> {
        let xf = load("bspline.txt");
        let (Transform::Linear(a), Transform::BSpline(b)) = (&xf.steps[0].0, &xf.steps[1].0) else {
            panic!("expected linear + bspline")
        };
        pts.iter().map(|&p| b.in_region(a.apply(p))).collect()
    }

    /// For a point whose source is inside the grid, the warp preimage is unique (the fixture is
    /// injective: det(I + J) > 0 throughout), so the inverse recovers it exactly.
    #[test]
    fn inverse_round_trips_for_sources_inside_the_grid() {
        let g = golden();
        let chain = chain_of("bspline.txt");
        let inside = source_in_grid(&g["input"]);
        assert!(inside.iter().filter(|&&b| b).count() > 20, "fixture is blind");

        let fwd = transform_points(&chain, arr(&g["input"]).view(), XformOpts::default()).unwrap();
        let back =
            inverse_transform_points(&chain, fwd.view(), None, InverseOpts::default()).unwrap();

        for (i, &is_in) in inside.iter().enumerate() {
            if !is_in {
                continue;
            }
            for j in 0..3 {
                assert!(
                    (back[[i, j]] - g["input"][i][j]).abs() < 1e-6,
                    "row {i}: {} != {}",
                    back[[i, j]],
                    g["input"][i][j]
                );
            }
        }
    }

    /// The forward map is `p + disp(p)` inside the grid and `p` outside, so it is discontinuous
    /// at the boundary and **not injective**: a target can have a preimage on each branch. We
    /// prefer the warp branch -- the registration doing its job -- so for such a point the
    /// inverse returns a *different* point than the one you started from.
    ///
    /// That is not a defect and cannot be fixed; it is what a 2-to-1 map means. What we do
    /// guarantee is that the answer is a genuine preimage, and this pins that distinction.
    #[test]
    fn a_target_can_have_two_preimages_and_we_return_the_warped_one() {
        let g = golden();
        let chain = chain_of("bspline.txt");
        let inside = source_in_grid(&g["input"]);

        let fwd = transform_points(&chain, arr(&g["input"]).view(), XformOpts::default()).unwrap();
        let back =
            inverse_transform_points(&chain, fwd.view(), None, InverseOpts::default()).unwrap();
        let refwd = transform_points(&chain, back.view(), XformOpts::default()).unwrap();

        // Exactly one golden point has an out-of-grid source that ALSO has an in-grid preimage.
        let ambiguous: Vec<usize> = (0..fwd.nrows())
            .filter(|&i| {
                !inside[i] && (0..3).any(|j| (back[[i, j]] - g["input"][i][j]).abs() > 1e-6)
            })
            .collect();
        assert_eq!(ambiguous.len(), 1, "fixture should carry exactly one such point");

        // ...and what we returned for it really does map to the target.
        let i = ambiguous[0];
        for j in 0..3 {
            assert!(
                (refwd[[i, j]] - fwd[[i, j]]).abs() < 1e-6,
                "row {i} is not a preimage after all"
            );
        }
    }

    /// The fixed-point seed lands materially closer to the preimage than the bare target does.
    ///
    /// **This test does not prove the seed is necessary, and it is not meant to.** On a grid this
    /// small the deformation cannot get large enough to strand LM on the flat extrapolation
    /// plateau *and* stay injective, so `seed_iter = 0` converges here just as well (checked: 20k
    /// points, identical to 1e-14). The seed earns its keep on real registrations, whose
    /// displacements reach ~20 grid cells -- FANC's leaves 3% of points unconverged without it.
    /// That is guarded by `test_seeding_rescues_points_the_naive_start_loses` in the Python
    /// suite, which runs against the real file.
    ///
    /// What this pins is the *mechanism*: that `seed` actually moves the start towards the
    /// answer, so a regression that breaks it is caught rather than silently tolerated.
    #[test]
    fn the_fixed_point_seed_moves_towards_the_preimage() {
        let b = fixture_spline();
        let p = [97.4, 63.8, 15.5]; // in-grid; the valid box is [0,98]x[0,98]x[0,65]
        assert!(b.in_region(p));
        let y = b.apply(p);
        assert!(!b.in_region(y), "the warp should drag this one out of the grid");

        let d_target = sq_dist(y, p).sqrt();
        let d_seed = sq_dist(b.seed(y, 8), p).sqrt();
        assert!(
            d_seed < 0.1 * d_target,
            "seed is {d_seed} from the answer, the bare target {d_target}"
        );
    }

    /// Turning the seed off must not change an answer the solver already gets right.
    #[test]
    fn seed_iter_does_not_change_a_converged_answer() {
        let g = golden();
        let chain = chain_of("bspline.txt");
        let fwd = transform_points(&chain, arr(&g["input"]).view(), XformOpts::default()).unwrap();
        let seeded =
            inverse_transform_points(&chain, fwd.view(), None, InverseOpts::default()).unwrap();
        let bare = inverse_transform_points(
            &chain,
            fwd.view(),
            None,
            InverseOpts {
                seed_iter: 0,
                ..InverseOpts::default()
            },
        )
        .unwrap();
        for i in 0..seeded.nrows() {
            for j in 0..3 {
                assert!((seeded[[i, j]] - bare[[i, j]]).abs() < 1e-6);
            }
        }
    }

    /// An `Add` chain cannot be undone step-by-step, so we refuse rather than approximate.
    #[test]
    fn add_chains_are_not_invertible() {
        let chain = chain_of("add.txt");
        let pts = [[10.0, 10.0, 10.0]];
        assert!(matches!(
            inverse_transform_points(&chain, arr(&pts).view(), None, InverseOpts::default()),
            Err(ElastixError::NotInvertible)
        ));
        // ...but it still transforms forwards.
        assert!(transform_points(&chain, arr(&pts).view(), XformOpts::default()).is_ok());
    }

    /// Handed the exact answer as a guess, the solver must land on it -- for **every** row,
    /// including the ambiguous one, where the guess is the only thing that can break the tie.
    ///
    /// This is what catches the coordinate-frame bug: the caller's guess is in *source* space,
    /// but the spline lives behind the initial affine and solves in *post-affine* space. Seeding
    /// it with the raw guess is quietly wrong -- quietly, because the fixed-point seed then
    /// rescues most rows anyway and only the ambiguous one gives it away.
    #[test]
    fn initial_guess_is_pushed_into_the_splines_own_coordinate_frame() {
        let g = golden();
        let chain = chain_of("bspline.txt");
        let fwd = transform_points(&chain, arr(&g["input"]).view(), XformOpts::default()).unwrap();
        let guess = arr(&g["input"]);
        let back =
            inverse_transform_points(&chain, fwd.view(), Some(guess.view()), InverseOpts::default())
                .unwrap();
        assert_close(&back, &g["input"], 1e-6, "guided inverse");

        // The initial affine is not the identity, so a source-space guess really is in a
        // different frame from the spline's input -- otherwise this test proves nothing.
        let xf = load("bspline.txt");
        let prefix = xf.forward_prefix(g["input"][0]);
        assert_ne!(prefix[0], prefix[1], "the affine must actually move the guess");
    }

    #[test]
    fn wrong_length_initial_guess_errors() {
        let chain = chain_of("bspline.txt");
        let pts = arr(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let guess = arr(&[[1.0, 2.0, 3.0]]);
        assert!(matches!(
            inverse_transform_points(&chain, pts.view(), Some(guess.view()), InverseOpts::default()),
            Err(ElastixError::GuessLen { got: 1, want: 2 })
        ));
    }

    // ---- drivers ----

    /// Every point is independent, so the answer must not depend on how many threads saw it.
    #[test]
    fn thread_count_does_not_change_the_answer() {
        let g = golden();
        let chain = chain_of("bspline.txt");
        let fwd = transform_points(&chain, arr(&g["input"]).view(), XformOpts::default()).unwrap();

        let one = inverse_transform_points(
            &chain,
            fwd.view(),
            None,
            InverseOpts {
                threads: Some(1),
                ..Default::default()
            },
        )
        .unwrap();
        let many =
            inverse_transform_points(&chain, fwd.view(), None, InverseOpts::default()).unwrap();
        assert_eq!(one, many, "results must be bit-identical across thread counts");
    }

    #[test]
    fn cancel_flag_short_circuits() {
        let cancel = AtomicBool::new(true);
        let pts = arr(&[[1.0, 2.0, 3.0]]);
        let err = transform_points(
            &chain_of("affine.txt"),
            pts.view(),
            XformOpts {
                cancel: Some(&cancel),
                ..Default::default()
            },
        );
        assert!(matches!(err, Err(ElastixError::Cancelled)));
        cancel.store(false, Ordering::Relaxed);
    }

    #[test]
    fn bad_shape_errors() {
        let pts = Array2::<f64>::zeros((4, 2));
        assert!(matches!(
            transform_points(&chain_of("affine.txt"), pts.view(), XformOpts::default()),
            Err(ElastixError::BadShape { .. })
        ));
    }

    #[test]
    fn empty_chain_errors() {
        assert!(matches!(
            Chain::new(vec![]),
            Err(ElastixError::EmptyChain)
        ));
    }

    /// `is_invertible` is a question about the *parse*, and the bindings ask it with the flags
    /// they will actually traverse with. All-forward is the question that matters: "can I run
    /// `xform_inv` on this at all?"
    #[test]
    fn is_invertible_reports_on_the_hops_it_would_have_to_reverse() {
        assert!(chain_of("bspline.txt").is_invertible(None));
        assert!(!chain_of("add.txt").is_invertible(None));
        // ...but a hop already flagged `invert` is the one a whole-chain inversion runs
        // *forwards*, and forwards, `Add` is fine.
        assert!(chain_of("add.txt").is_invertible(Some(&[true])));
    }
}
