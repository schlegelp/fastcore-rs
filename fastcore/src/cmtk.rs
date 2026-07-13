//! CMTK spatial transforms: reading a registration and applying it to points.
//!
//! [CMTK](https://www.nitrc.org/projects/cmtk) registrations are how *Drosophila*
//! connectomics bridges between template brain spaces (JFRC2 → FCWB and friends). A
//! registration is a 12-DOF **affine**, optionally followed by a cubic B-spline
//! **free-form deformation** on a regular control-point lattice.
//!
//! - [`Registration::from_path`] reads a `*.list` directory (or a bare `registration`
//!   file, plain or gzipped) written in CMTK's TypedStream format.
//! - [`Chain`] is one or more registrations applied nose-to-tail, each optionally
//!   traversed backwards — the shape a bridging graph needs.
//! - [`transform_points`] / [`inverse_transform_points`] are the batch entry points.
//!
//! # Why this module exists in Rust
//!
//! The alternative is shelling out to CMTK's `streamxform` binary once per call, which
//! means installing CMTK and paying process-startup plus text I/O on every transform.
//! Here a registration is parsed once and applied many times, in parallel.
//!
//! The forward warp is a 4×4×4 tensor-product sum — cheap. The **inverse** has no closed
//! form and dominates: it is a per-point nonlinear solve. We run Levenberg–Marquardt
//! against the *analytic* Jacobian (see [`SplineWarp::apply_with_jacobian`]) rather than
//! finite differences, and parallelise over points, which is what makes it fast.
//!
//! # Fidelity to CMTK
//!
//! Validated **against the `streamxform` binary itself** (CMTK 3.3.1) on 5000 random points
//! spanning the domain and well beyond it, for a real JFRC2 → FCWB registration. All four
//! paths — affine and warp, forward and inverse — agree to `5e-7`, which is `streamxform`'s
//! own print precision, and the sets of points it reports as `FAILED` are reproduced
//! **exactly**. (`fastcore/testdata/streamxform_golden.csv` pins 5 of those points as a
//! checked-in regression, so the suite needs no CMTK install.)
//!
//! Three behaviours are load-bearing and non-obvious. The reference implementation this was
//! ported from gets the last two wrong:
//!
//! - **The affine is not applied in the warp path.** See [`SplineWarp::affine`].
//! - **The domain is the world box `[0, domain]`, not the control-point lattice's extent.**
//!   The lattice is padded one cell outside the domain, so the two differ by a whole
//!   control-point spacing. See [`SplineWarp::in_domain`].
//! - **A point CMTK cannot transform yields `NaN`, never a plausible-looking number.**
//!   Outside the domain box, `streamxform` prints `FAILED`; so do we, rather than
//!   extrapolating a warp that was never fitted there. Likewise the inverse iterate is
//!   confined to the domain box and its residual is checked before the answer is accepted.
//!   See [`XformOpts::allow_extrapolation`] and [`InverseOpts::clamp_to_domain`].
//!
//! # Attribution
//!
//! Ported from [`cmtk_apply`](https://github.com/schlegelp/cmtk_apply) (MIT, © 2026 Philipp
//! Schlegel), itself informed by the `nat` R package's CMTK I/O. Relicensed to GPL-3 as part
//! of this crate, as the MIT terms permit.

use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;

use ndarray::{Array2, ArrayView2};
use rayon::prelude::*;

use crate::nblast::{is_cancelled, make_bar, with_pool};

/// Rows per parallel chunk. Sized so the cancellation poll and the progress-bar tick
/// amortise over real work: an inverse point costs ~5-15 µs, so a chunk is ~10 ms.
const CHUNK: usize = 1024;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CmtkError {
    /// The path could not be read, or a `*.list` directory held no `registration`.
    Io { path: String, msg: String },
    /// Missing the leading `! TYPEDSTREAM` header.
    NotTypedStream { path: String },
    /// The file parsed, but held no top-level `registration` block.
    NoRegistrationBlock,
    /// A block was missing a key it cannot do without.
    MissingKey { block: &'static str, key: &'static str },
    BadValue { key: String, msg: String },
    /// `coefficients` appeared before `dims`, so its length was unknowable.
    CoefficientsBeforeDims,
    CoefficientCount { got: usize, want: usize },
    /// Two `affine_xform` (or `spline_warp`) blocks at one level. The reference silently
    /// hands the second one to a function expecting the first; we refuse instead.
    DuplicateBlock { key: &'static str },
    /// An affine was needed (`transform="affine"`, no spline present, or a fallback) but
    /// the registration has none.
    NoAffine,
    /// The affine matrix is not invertible.
    SingularAffine,
    /// A registration with neither an affine nor a spline.
    NoTransform,
    EmptyChain,
    BadShape { got: Vec<usize> },
    /// `initial_guess` was not one point per input point.
    GuessLen { got: usize, want: usize },
    Cancelled,
}

impl fmt::Display for CmtkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CmtkError::Io { path, msg } => write!(f, "could not read {path}: {msg}"),
            CmtkError::NotTypedStream { path } => write!(
                f,
                "{path} is not a CMTK TypedStream file (no `! TYPEDSTREAM` header)"
            ),
            CmtkError::NoRegistrationBlock => {
                write!(f, "no `registration` block found in the TypedStream")
            }
            CmtkError::MissingKey { block, key } => {
                write!(f, "`{block}` block is missing the `{key}` key")
            }
            CmtkError::BadValue { key, msg } => write!(f, "bad value for `{key}`: {msg}"),
            CmtkError::CoefficientsBeforeDims => write!(
                f,
                "`coefficients` appeared before `dims`; its length cannot be determined"
            ),
            CmtkError::CoefficientCount { got, want } => write!(
                f,
                "`coefficients` has {got} values, but `dims` implies {want}"
            ),
            CmtkError::DuplicateBlock { key } => {
                write!(f, "more than one `{key}` block at the same level")
            }
            CmtkError::NoAffine => write!(
                f,
                "this registration has no affine transform, so `transform='affine'` \
                 (or a fallback to it) is not possible"
            ),
            CmtkError::SingularAffine => {
                write!(f, "the affine matrix is singular and cannot be inverted")
            }
            CmtkError::NoTransform => {
                write!(f, "this registration has neither an affine nor a spline warp")
            }
            CmtkError::EmptyChain => write!(f, "a registration chain must not be empty"),
            CmtkError::BadShape { got } => {
                write!(f, "`points` must have shape (N, 3), got {got:?}")
            }
            CmtkError::GuessLen { got, want } => write!(
                f,
                "`initial_guess` must have one point per input point: got {got}, want {want}"
            ),
            CmtkError::Cancelled => write!(f, "interrupted"),
        }
    }
}

impl std::error::Error for CmtkError {}

// ---------------------------------------------------------------------------
// TypedStream
// ---------------------------------------------------------------------------

/// A parsed TypedStream value.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Bool(bool),
    /// One or more whitespace-separated numbers. A scalar is a `Numbers` of length 1.
    Numbers(Vec<f64>),
    Str(String),
    Block(Block),
}

/// A `label { ... }` block. Keys may repeat, so entries are an ordered list rather than a
/// map — `get` takes the first, `get_all` takes them all.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Block {
    entries: Vec<(String, Value)>,
}

impl Block {
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.entries.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    pub fn get_all(&self, key: &str) -> Vec<&Value> {
        self.entries
            .iter()
            .filter(|(k, _)| k == key)
            .map(|(_, v)| v)
            .collect()
    }

    /// The sole sub-block under `key`, erroring if there is more than one.
    fn block(&self, key: &'static str) -> Result<Option<&Block>, CmtkError> {
        let found: Vec<&Block> = self
            .entries
            .iter()
            .filter(|(k, _)| k == key)
            .filter_map(|(_, v)| match v {
                Value::Block(b) => Some(b),
                _ => None,
            })
            .collect();
        match found.len() {
            0 => Ok(None),
            1 => Ok(Some(found[0])),
            _ => Err(CmtkError::DuplicateBlock { key }),
        }
    }

    fn numbers(&self, block: &'static str, key: &'static str) -> Result<&[f64], CmtkError> {
        match self.get(key) {
            Some(Value::Numbers(n)) => Ok(n),
            Some(_) => Err(CmtkError::BadValue {
                key: key.to_string(),
                msg: "expected numbers".to_string(),
            }),
            None => Err(CmtkError::MissingKey { block, key }),
        }
    }

    fn triple(&self, block: &'static str, key: &'static str) -> Result<[f64; 3], CmtkError> {
        let n = self.numbers(block, key)?;
        if n.len() != 3 {
            return Err(CmtkError::BadValue {
                key: key.to_string(),
                msg: format!("expected 3 values, got {}", n.len()),
            });
        }
        Ok([n[0], n[1], n[2]])
    }

    fn bool_or(&self, key: &str, default: bool) -> bool {
        match self.get(key) {
            Some(Value::Bool(b)) => *b,
            _ => default,
        }
    }
}

/// Parse a TypedStream document. Returns the root block and the version string from the
/// header (`! TYPEDSTREAM 1.1` → `"1.1"`).
fn parse_typedstream(text: &str, path: &str) -> Result<(Block, String), CmtkError> {
    let mut lines = text.lines();

    // Header. Blank lines before it are tolerated; anything else is not a TypedStream.
    let version = loop {
        match lines.next() {
            Some(l) if l.trim().is_empty() => continue,
            Some(l) if l.trim_start().starts_with("! TYPEDSTREAM") => {
                break l.split_whitespace().next_back().unwrap_or("").to_string();
            }
            _ => {
                return Err(CmtkError::NotTypedStream {
                    path: path.to_string(),
                })
            }
        }
    };

    // Everything after the header, as a cursor we can advance from inside the recursion
    // (continuation lines for `coefficients`/`active` are consumed mid-entry).
    let rest: Vec<&str> = lines.collect();
    let mut cur = 0usize;
    let mut root = Block::default();
    parse_entries(&rest, &mut cur, &mut root, /* depth = */ 0)?;
    Ok((root, version))
}

/// Fill `block` with entries from `lines[*cur..]`, stopping at the `}` that closes it (or
/// at EOF for the root).
fn parse_entries(
    lines: &[&str],
    cur: &mut usize,
    block: &mut Block,
    depth: usize,
) -> Result<(), CmtkError> {
    // `dims` is needed to size `coefficients`, and it always precedes it in practice.
    let mut dims: Option<[usize; 3]> = None;

    while *cur < lines.len() {
        let line = lines[*cur].trim();
        *cur += 1;

        if line.is_empty() {
            continue;
        }
        if line == "}" {
            if depth == 0 {
                continue; // stray close at the root: ignore rather than fail
            }
            return Ok(());
        }

        // `label {` opens a nested block.
        if let Some(label) = line.strip_suffix('{') {
            let key = label.trim().to_string();
            let mut inner = Block::default();
            parse_entries(lines, cur, &mut inner, depth + 1)?;
            block.entries.push((key, Value::Block(inner)));
            continue;
        }

        let mut it = line.splitn(2, char::is_whitespace);
        let key = it.next().unwrap_or("").to_string();
        let rest = it.next().unwrap_or("").trim();

        match key.as_str() {
            // Spills across continuation lines. Consume greedily: a continuation is any
            // following line whose every token parses as a number. The next real entry
            // (`active …`) and the closing `}` both fail that test, so this terminates.
            "coefficients" => {
                let dims = dims.ok_or(CmtkError::CoefficientsBeforeDims)?;
                let want = dims[0] * dims[1] * dims[2] * 3;
                let mut vals: Vec<f64> = Vec::with_capacity(want);
                push_numbers(rest, &mut vals)?;
                while *cur < lines.len() {
                    let l = lines[*cur].trim();
                    if l.is_empty() || !is_all_numeric(l) {
                        break;
                    }
                    push_numbers(l, &mut vals)?;
                    *cur += 1;
                }
                if vals.len() != want {
                    return Err(CmtkError::CoefficientCount {
                        got: vals.len(),
                        want,
                    });
                }
                block.entries.push((key, Value::Numbers(vals)));
            }
            // One bit per degree of freedom. CMTK's own `streamxform` ignores it and so do
            // we — but its continuation lines must still be *consumed*, or the `}` that
            // closes the enclosing block gets read as an entry.
            "active" => {
                while *cur < lines.len() {
                    let l = lines[*cur].trim();
                    if l.is_empty() || !l.chars().all(|c| c == '0' || c == '1') {
                        break;
                    }
                    *cur += 1;
                }
            }
            _ => {
                let value = parse_value(rest);
                if key == "dims" {
                    if let Value::Numbers(n) = &value {
                        if n.len() == 3 && n.iter().all(|v| *v >= 4.0 && v.fract() == 0.0) {
                            dims = Some([n[0] as usize, n[1] as usize, n[2] as usize]);
                        } else {
                            return Err(CmtkError::BadValue {
                                key: "dims".to_string(),
                                msg: format!(
                                    "expected 3 integers >= 4 (cubic B-splines need a 4-wide \
                                     support), got {n:?}"
                                ),
                            });
                        }
                    }
                }
                block.entries.push((key, value));
            }
        }
    }
    Ok(())
}

fn parse_value(rest: &str) -> Value {
    if rest == "yes" {
        return Value::Bool(true);
    }
    if rest == "no" {
        return Value::Bool(false);
    }
    if let Some(inner) = rest.strip_prefix('"') {
        // Quoted string; take everything up to the closing quote.
        let end = inner.find('"').unwrap_or(inner.len());
        return Value::Str(inner[..end].to_string());
    }
    if !rest.is_empty() && is_all_numeric(rest) {
        let nums: Vec<f64> = rest
            .split_whitespace()
            .filter_map(|t| t.parse::<f64>().ok())
            .collect();
        return Value::Numbers(nums);
    }
    Value::Str(rest.to_string())
}

/// True iff every whitespace-separated token parses as a number (and there is at least one).
fn is_all_numeric(s: &str) -> bool {
    let mut any = false;
    for tok in s.split_whitespace() {
        if tok.parse::<f64>().is_err() {
            return false;
        }
        any = true;
    }
    any
}

fn push_numbers(s: &str, out: &mut Vec<f64>) -> Result<(), CmtkError> {
    for tok in s.split_whitespace() {
        out.push(tok.parse::<f64>().map_err(|e| CmtkError::BadValue {
            key: "coefficients".to_string(),
            msg: format!("{tok:?}: {e}"),
        })?);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Affine
// ---------------------------------------------------------------------------

/// A 4×4 homogeneous affine, row-major (`m[row][col]`). The bottom row is always
/// `[0, 0, 0, 1]` by construction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Affine {
    pub m: [[f64; 4]; 4],
}

impl Affine {
    pub fn identity() -> Self {
        let mut m = [[0.0; 4]; 4];
        for (i, row) in m.iter_mut().enumerate() {
            row[i] = 1.0;
        }
        Affine { m }
    }

    /// Compose CMTK's five parameter triples into a matrix.
    ///
    /// `rotate` is in **degrees**. `legacy` selects the pre-2.4.0 composition (see
    /// [`Registration::legacy`]).
    ///
    /// The rotation block below is written **transposed** relative to the usual row-major
    /// convention (note `m[1][0] = -cos1*sin2`, where you would expect `+`). That is what
    /// CMTK does, and it is what reproduces `streamxform`. Do not "fix" it.
    pub fn from_params(
        xlate: [f64; 3],
        rotate: [f64; 3],
        scale: [f64; 3],
        shear: [f64; 3],
        center: [f64; 3],
        legacy: bool,
    ) -> Self {
        let [sx, sy, sz] = scale;
        let [shx, shy, shz] = shear;

        let (alpha, theta, phi) = (
            rotate[0].to_radians(),
            rotate[1].to_radians(),
            rotate[2].to_radians(),
        );
        let (sin0, cos0) = alpha.sin_cos();
        let (sin1, cos1) = theta.sin_cos();
        let (sin2, cos2) = phi.sin_cos();
        let sin0xsin1 = sin0 * sin1;
        let cos0xsin1 = cos0 * sin1;

        let mut m = Affine::identity().m;
        m[0][0] = cos1 * cos2;
        m[1][0] = -cos1 * sin2;
        m[2][0] = -sin1;
        m[0][1] = sin0xsin1 * cos2 + cos0 * sin2;
        m[1][1] = -sin0xsin1 * sin2 + cos0 * cos2;
        m[2][1] = sin0 * cos1;
        m[0][2] = cos0xsin1 * cos2 - sin0 * sin2;
        m[1][2] = -cos0xsin1 * sin2 - sin0 * cos2;
        m[2][2] = cos0 * cos1;

        if legacy {
            // Scale the columns in place, then left-multiply three single-element shear
            // matrices in the order i = 2, 1, 0.
            for row in m.iter_mut().take(3) {
                row[0] *= sx;
                row[1] *= sy;
                row[2] *= sz;
            }
            const CELL: [(usize, usize); 3] = [(1, 0), (2, 0), (2, 1)];
            for i in (0..3).rev() {
                let mut s = Affine::identity().m;
                let (r, c) = CELL[i];
                s[r][c] = [shx, shy, shz][i];
                m = mat_mul(&s, &m);
            }
        } else {
            // One upper-triangular scale/shear matrix, right-multiplied.
            let mut ss = [[0.0f64; 4]; 4];
            ss[0][0] = sx;
            ss[1][1] = sy;
            ss[2][2] = sz;
            ss[3][3] = 1.0;
            ss[0][1] = shx;
            ss[0][2] = shy;
            ss[1][2] = shz;
            m = mat_mul(&m, &ss);
        }

        // Translation about the rotation centre.
        for i in 0..3 {
            let c_m = center[0] * m[i][0] + center[1] * m[i][1] + center[2] * m[i][2];
            m[i][3] = xlate[i] - c_m + center[i];
        }
        Affine { m }
    }

    #[inline]
    pub fn apply(&self, p: [f64; 3]) -> [f64; 3] {
        let m = &self.m;
        [
            m[0][0] * p[0] + m[0][1] * p[1] + m[0][2] * p[2] + m[0][3],
            m[1][0] * p[0] + m[1][1] * p[1] + m[1][2] * p[2] + m[1][3],
            m[2][0] * p[0] + m[2][1] * p[1] + m[2][2] * p[2] + m[2][3],
        ]
    }

    /// Exact inverse. `None` iff the linear block is singular.
    ///
    /// The bottom row is `[0, 0, 0, 1]`, so this inverts the 3×3 block by cofactors and
    /// back-substitutes the translation — no general 4×4 elimination needed.
    pub fn inverse(&self) -> Option<Affine> {
        let a = &self.m;
        let det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
            - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
            + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
        if det.abs() < 1e-12 || !det.is_finite() {
            return None;
        }
        let inv_det = 1.0 / det;
        let mut m = Affine::identity().m;
        m[0][0] = (a[1][1] * a[2][2] - a[1][2] * a[2][1]) * inv_det;
        m[0][1] = (a[0][2] * a[2][1] - a[0][1] * a[2][2]) * inv_det;
        m[0][2] = (a[0][1] * a[1][2] - a[0][2] * a[1][1]) * inv_det;
        m[1][0] = (a[1][2] * a[2][0] - a[1][0] * a[2][2]) * inv_det;
        m[1][1] = (a[0][0] * a[2][2] - a[0][2] * a[2][0]) * inv_det;
        m[1][2] = (a[0][2] * a[1][0] - a[0][0] * a[1][2]) * inv_det;
        m[2][0] = (a[1][0] * a[2][1] - a[1][1] * a[2][0]) * inv_det;
        m[2][1] = (a[0][1] * a[2][0] - a[0][0] * a[2][1]) * inv_det;
        m[2][2] = (a[0][0] * a[1][1] - a[0][1] * a[1][0]) * inv_det;
        // t' = -A^-1 t
        for row in m.iter_mut().take(3) {
            row[3] = -(row[0] * a[0][3] + row[1] * a[1][3] + row[2] * a[2][3]);
        }
        Some(Affine { m })
    }

    /// The matrix as a `(4, 4)` array, for the bindings.
    pub fn as_array(&self) -> Array2<f64> {
        Array2::from_shape_fn((4, 4), |(i, j)| self.m[i][j])
    }

    fn from_block(b: &Block, legacy: bool) -> Result<Affine, CmtkError> {
        Ok(Affine::from_params(
            b.triple("affine_xform", "xlate")?,
            b.triple("affine_xform", "rotate")?,
            b.triple("affine_xform", "scale")?,
            b.triple("affine_xform", "shear")?,
            b.triple("affine_xform", "center")?,
            legacy,
        ))
    }
}

fn mat_mul(a: &[[f64; 4]; 4], b: &[[f64; 4]; 4]) -> [[f64; 4]; 4] {
    let mut out = [[0.0f64; 4]; 4];
    for (i, orow) in out.iter_mut().enumerate() {
        for (j, o) in orow.iter_mut().enumerate() {
            *o = (0..4).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Spline warp
// ---------------------------------------------------------------------------

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

#[inline]
fn bspline_weights_deriv(t: f64) -> [f64; 4] {
    let t2 = t * t;
    [
        (-3.0 + 6.0 * t - 3.0 * t2) / 6.0,
        (-12.0 * t + 9.0 * t2) / 6.0,
        (3.0 + 6.0 * t - 9.0 * t2) / 6.0,
        (3.0 * t2) / 6.0,
    ]
}

/// A cubic B-spline free-form deformation on a regular control-point lattice.
#[derive(Debug, Clone, PartialEq)]
pub struct SplineWarp {
    pub dims: [usize; 3],
    pub domain: [f64; 3],
    pub origin: [f64; 3],
    /// Derived as `domain[i] / (dims[i] - 3)`. Cached because it is in the inner loop.
    pub spacing: [f64; 3],
    /// `prod(dims)` control points, **x fastest**: `idx = ix + nx * (iy + ny * iz)`.
    pub coefficients: Vec<[f64; 3]>,
    /// `absolute yes` → the coefficients *are* target positions (the affine is already
    /// baked into them), so the result is `spline(p)`. `no` → they are displacements and
    /// the result is `p + spline(p)`.
    pub absolute: bool,
    /// The `affine_xform` nested inside the `spline_warp` block.
    ///
    /// **This is parsed and exposed, but never applied.** CMTK's `streamxform` does not
    /// apply it in the warp path either — with `absolute yes` the affine is already baked
    /// into the coefficients, so applying it again would double-count. Every reader of this
    /// code tries to "fix" that. Don't.
    pub affine: Option<Affine>,
}

impl SplineWarp {
    /// Is `p` inside the domain this warp is actually defined over?
    ///
    /// CMTK's test, verified against `streamxform` on 3000 points: the **world-space domain
    /// box** `[0, domain]`. Note this is *not* the control-point lattice's extent — the
    /// lattice is padded one cell outside the domain (`origin == -spacing`), so testing
    /// `u ∈ [0, dims-3)` gives a box shifted by one spacing that CMTK agrees with nowhere.
    /// The reference implementation makes exactly that mistake.
    ///
    /// This is the same box the inverse solver confines its iterate to, which is not a
    /// coincidence: it is the region where the warp means anything.
    #[inline]
    pub fn in_domain(&self, p: [f64; 3]) -> bool {
        (0..3).all(|i| p[i] >= 0.0 && p[i] <= self.domain[i])
    }

    /// The raw B-spline sum at `p` — *not* yet interpreted as absolute or relative.
    ///
    /// `None` when `!allow_extrapolation` and `p` is outside [`in_domain`]. When
    /// extrapolation *is* allowed, the neighbour indices are clamped to `[0, n-1]`, so the
    /// outermost control points extend outward and every point gets an answer.
    ///
    /// [`in_domain`]: SplineWarp::in_domain
    #[inline]
    fn eval(&self, p: [f64; 3], allow_extrapolation: bool) -> Option<[f64; 3]> {
        if !allow_extrapolation && !self.in_domain(p) {
            return None;
        }
        let [nx, ny, nz] = self.dims;

        let mut base = [0isize; 3];
        let mut w = [[0.0f64; 4]; 3];
        for i in 0..3 {
            let u = (p[i] - self.origin[i]) / self.spacing[i];
            let fl = u.floor();
            base[i] = fl as isize - 1;
            w[i] = bspline_weights(u - fl);
        }

        let mut acc = [0.0f64; 3];
        for dx in 0..4 {
            let ix = clamp_idx(base[0] + dx as isize, nx);
            for dy in 0..4 {
                let iy = clamp_idx(base[1] + dy as isize, ny);
                for dz in 0..4 {
                    let iz = clamp_idx(base[2] + dz as isize, nz);
                    let wt = w[0][dx] * w[1][dy] * w[2][dz];
                    let c = self.coefficients[ix + nx * (iy + ny * iz)];
                    acc[0] += wt * c[0];
                    acc[1] += wt * c[1];
                    acc[2] += wt * c[2];
                }
            }
        }
        Some(acc)
    }

    /// The deformed position of `p`.
    #[inline]
    pub fn apply(&self, p: [f64; 3], allow_extrapolation: bool) -> Option<[f64; 3]> {
        let s = self.eval(p, allow_extrapolation)?;
        Some(if self.absolute {
            s
        } else {
            [p[0] + s[0], p[1] + s[1], p[2] + s[2]]
        })
    }

    /// The deformed position of `p` together with the analytic Jacobian `d(out)/d(in)`.
    ///
    /// Same 4×4×4 loop, with one basis function swapped for its derivative and divided by
    /// that axis's spacing. In relative mode the transform is `p + spline(p)`, so the
    /// identity is added — the reference implementation omits this, which makes its
    /// analytic-Jacobian solver wrong for relative-mode warps.
    #[inline]
    pub fn apply_with_jacobian(
        &self,
        p: [f64; 3],
        allow_extrapolation: bool,
    ) -> Option<([f64; 3], [[f64; 3]; 3])> {
        if !allow_extrapolation && !self.in_domain(p) {
            return None;
        }
        let [nx, ny, nz] = self.dims;

        let mut base = [0isize; 3];
        let mut w = [[0.0f64; 4]; 3];
        let mut dw = [[0.0f64; 4]; 3];
        for i in 0..3 {
            let u = (p[i] - self.origin[i]) / self.spacing[i];
            let fl = u.floor();
            base[i] = fl as isize - 1;
            let t = u - fl;
            w[i] = bspline_weights(t);
            dw[i] = bspline_weights_deriv(t);
        }

        let mut acc = [0.0f64; 3];
        // jac[r][c] = d(out_r) / d(in_c)
        let mut jac = [[0.0f64; 3]; 3];
        for dx in 0..4 {
            let ix = clamp_idx(base[0] + dx as isize, nx);
            for dy in 0..4 {
                let iy = clamp_idx(base[1] + dy as isize, ny);
                for dz in 0..4 {
                    let iz = clamp_idx(base[2] + dz as isize, nz);
                    let c = self.coefficients[ix + nx * (iy + ny * iz)];
                    let wt = w[0][dx] * w[1][dy] * w[2][dz];
                    let gx = dw[0][dx] * w[1][dy] * w[2][dz] / self.spacing[0];
                    let gy = w[0][dx] * dw[1][dy] * w[2][dz] / self.spacing[1];
                    let gz = w[0][dx] * w[1][dy] * dw[2][dz] / self.spacing[2];
                    for r in 0..3 {
                        acc[r] += wt * c[r];
                        jac[r][0] += gx * c[r];
                        jac[r][1] += gy * c[r];
                        jac[r][2] += gz * c[r];
                    }
                }
            }
        }

        if self.absolute {
            Some((acc, jac))
        } else {
            for r in 0..3 {
                acc[r] += p[r];
                jac[r][r] += 1.0;
            }
            Some((acc, jac))
        }
    }

    /// CMTK's domain box, `[0, 0, 0] ..= domain`. The region the inverse iterate is
    /// confined to — see [`InverseOpts::clamp_to_domain`].
    #[inline]
    pub fn domain_box(&self) -> ([f64; 3], [f64; 3]) {
        ([0.0, 0.0, 0.0], self.domain)
    }

    /// Find `x` such that `apply(x) ≈ target`, by damped Gauss-Newton (Levenberg-Marquardt).
    ///
    /// `None` when the residual does not come within `opts.accuracy` — which the caller
    /// turns into a `NaN` row, exactly as `streamxform` prints `FAILED`.
    pub fn solve_inverse(
        &self,
        target: [f64; 3],
        x0: [f64; 3],
        opts: &InverseOpts,
    ) -> Option<[f64; 3]> {
        let (lo, hi) = if opts.clamp_to_domain {
            self.domain_box()
        } else {
            ([f64::NEG_INFINITY; 3], [f64::INFINITY; 3])
        };

        let mut x = clamp_pt(x0, lo, hi);
        // The residual is always evaluated with extrapolation allowed: the iterate may
        // wander outside the lattice on its way to a solution that is inside it.
        let mut cost = sq_dist(self.apply(x, true)?, target);
        let mut lambda = 1e-6f64;

        for _ in 0..opts.max_iter {
            if cost == 0.0 {
                break;
            }
            let (f, j) = self.apply_with_jacobian(x, true)?;
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
            // Marquardt's diagonal scaling, not `+ λI`: the axis spacings differ by ~50%
            // and the coefficient magnitudes by more, so isotropic damping is misscaled.
            for (c, arow) in a.iter_mut().enumerate() {
                arow[c] += lambda * arow[c].abs().max(1e-12);
            }

            let d = match solve3x3(&a, &g) {
                Some(d) => d,
                None => break,
            };
            let x_new = clamp_pt([x[0] - d[0], x[1] - d[1], x[2] - d[2]], lo, hi);
            let f_new = match self.apply(x_new, true) {
                Some(v) => v,
                None => break,
            };
            let cost_new = sq_dist(f_new, target);

            if cost_new < cost {
                // Measure the step *after* clamping — a clamped iterate may have moved far
                // less than `d` suggests, and that is what convergence should see.
                let step = (0..3)
                    .map(|i| (x_new[i] - x[i]).abs())
                    .fold(0.0f64, f64::max);
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

    fn from_block(b: &Block, legacy: bool) -> Result<SplineWarp, CmtkError> {
        let dims_f = b.triple("spline_warp", "dims")?;
        let dims = [dims_f[0] as usize, dims_f[1] as usize, dims_f[2] as usize];
        let domain = b.triple("spline_warp", "domain")?;
        let origin = b.triple("spline_warp", "origin")?;
        let absolute = b.bool_or("absolute", false);

        let mut spacing = [0.0f64; 3];
        for i in 0..3 {
            spacing[i] = domain[i] / (dims[i] - 3) as f64;
            if !spacing[i].is_finite() || spacing[i] == 0.0 {
                return Err(CmtkError::BadValue {
                    key: "domain".to_string(),
                    msg: format!("implies a zero or non-finite spacing on axis {i}"),
                });
            }
        }

        let flat = b.numbers("spline_warp", "coefficients")?;
        let want = dims[0] * dims[1] * dims[2] * 3;
        if flat.len() != want {
            return Err(CmtkError::CoefficientCount {
                got: flat.len(),
                want,
            });
        }
        let coefficients: Vec<[f64; 3]> = flat.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();

        let affine = b
            .block("affine_xform")?
            .map(|ab| Affine::from_block(ab, legacy))
            .transpose()?;

        Ok(SplineWarp {
            dims,
            domain,
            origin,
            spacing,
            coefficients,
            absolute,
            affine,
        })
    }
}

/// Clamp a control-point index into `[0, n-1]`. Within the domain box no clamping is ever
/// needed; outside it, this is what "extrapolation" means — the boundary control points
/// extend outward.
#[inline]
fn clamp_idx(i: isize, n: usize) -> usize {
    (i.max(0) as usize).min(n - 1)
}

#[inline]
fn clamp_pt(p: [f64; 3], lo: [f64; 3], hi: [f64; 3]) -> [f64; 3] {
    [
        p[0].clamp(lo[0], hi[0]),
        p[1].clamp(lo[1], hi[1]),
        p[2].clamp(lo[2], hi[2]),
    ]
}

#[inline]
fn sq_dist(a: [f64; 3], b: [f64; 3]) -> f64 {
    (0..3).map(|i| (a[i] - b[i]).powi(2)).sum()
}

/// Solve `A d = g` for a 3×3 `A` by cofactors. `None` if `A` is singular.
#[inline]
fn solve3x3(a: &[[f64; 3]; 3], g: &[f64; 3]) -> Option<[f64; 3]> {
    let det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
    if !det.is_finite() || det.abs() < 1e-300 {
        return None;
    }
    let inv_det = 1.0 / det;
    let mut out = [0.0f64; 3];
    for (c, o) in out.iter_mut().enumerate() {
        // Cramer: replace column `c` of A with g.
        let mut b = *a;
        for (r, brow) in b.iter_mut().enumerate() {
            brow[c] = g[r];
        }
        let d = b[0][0] * (b[1][1] * b[2][2] - b[1][2] * b[2][1])
            - b[0][1] * (b[1][0] * b[2][2] - b[1][2] * b[2][0])
            + b[0][2] * (b[1][0] * b[2][1] - b[1][1] * b[2][0]);
        *o = d * inv_det;
    }
    out.iter().all(|v| v.is_finite()).then_some(out)
}

// ---------------------------------------------------------------------------
// Registration & chain
// ---------------------------------------------------------------------------

/// One CMTK registration: an affine, a spline warp, or both.
#[derive(Debug, Clone, PartialEq)]
pub struct Registration {
    pub affine: Option<Affine>,
    pub spline: Option<SplineWarp>,
    /// The TypedStream version as written in the header, e.g. `"1.1"`.
    pub version: String,
    /// `version < 2.4`, which selects the pre-2.4.0 affine composition. Getting this wrong
    /// is not subtle: the bundled v1.1 registration is off by ~26 units under the modern
    /// composition.
    pub legacy: bool,
    /// The `registration` file this came from (not the `.list` directory).
    pub source: Option<PathBuf>,
}

impl Registration {
    /// Read a registration from a `*.list` directory (resolving `registration`, then
    /// `registration.gz`) or from a path to the file itself.
    pub fn from_path(path: &Path) -> Result<Self, CmtkError> {
        let file = resolve_registration_path(path)?;
        let disp = file.display().to_string();
        let raw = std::fs::read(&file).map_err(|e| CmtkError::Io {
            path: disp.clone(),
            msg: e.to_string(),
        })?;

        // Gzip is detected by magic bytes, not by extension: registrations are found
        // gzipped under a bare `registration` name in the wild.
        let text = if raw.starts_with(&[0x1f, 0x8b]) {
            use std::io::Read;
            let mut s = String::new();
            flate2::read::GzDecoder::new(&raw[..])
                .read_to_string(&mut s)
                .map_err(|e| CmtkError::Io {
                    path: disp.clone(),
                    msg: format!("gzip: {e}"),
                })?;
            s
        } else {
            String::from_utf8_lossy(&raw).into_owned()
        };

        let mut reg = Registration::from_typedstream(&text, &disp)?;
        reg.source = Some(file);
        Ok(reg)
    }

    /// Parse an already-decoded TypedStream document.
    pub fn from_typedstream(text: &str, path: &str) -> Result<Self, CmtkError> {
        let (root, version) = parse_typedstream(text, path)?;
        let reg = root
            .block("registration")?
            .ok_or(CmtkError::NoRegistrationBlock)?;

        let legacy = is_legacy_version(&version);
        let affine = reg
            .block("affine_xform")?
            .map(|b| Affine::from_block(b, legacy))
            .transpose()?;
        let spline = reg
            .block("spline_warp")?
            .map(|b| SplineWarp::from_block(b, legacy))
            .transpose()?;

        if affine.is_none() && spline.is_none() {
            return Err(CmtkError::NoTransform);
        }
        Ok(Registration {
            affine,
            spline,
            version,
            legacy,
            source: None,
        })
    }
}

/// `version < 2.4`. Unparseable versions are treated as modern, matching the reference.
fn is_legacy_version(version: &str) -> bool {
    let parts: Vec<u32> = version.split('.').map(|p| p.parse::<u32>()).try_fold(
        Vec::new(),
        |mut acc, p| match p {
            Ok(v) => {
                acc.push(v);
                Ok(acc)
            }
            Err(e) => Err(e),
        },
    ).unwrap_or_default();
    if parts.is_empty() {
        return false;
    }
    // Lexicographic compare against [2, 4].
    let target = [2u32, 4u32];
    for i in 0..target.len().max(parts.len()) {
        let a = parts.get(i).copied().unwrap_or(0);
        let b = target.get(i).copied().unwrap_or(0);
        if a != b {
            return a < b;
        }
    }
    false // equal => 2.4 => modern
}

fn resolve_registration_path(path: &Path) -> Result<PathBuf, CmtkError> {
    if path.is_dir() {
        let plain = path.join("registration");
        if plain.is_file() {
            return Ok(plain);
        }
        let gz = path.join("registration.gz");
        if gz.is_file() {
            return Ok(gz);
        }
        return Err(CmtkError::Io {
            path: path.display().to_string(),
            msg: "directory holds neither `registration` nor `registration.gz`".to_string(),
        });
    }
    if path.is_file() {
        return Ok(path.to_path_buf());
    }
    Err(CmtkError::Io {
        path: path.display().to_string(),
        msg: "no such file or directory".to_string(),
    })
}

/// One or more registrations applied nose-to-tail.
///
/// `invert[i]` traverses registration `i` backwards. A bridging graph routes through
/// registrations in whichever direction the edge was found, so this is not a luxury.
#[derive(Debug, Clone, PartialEq)]
pub struct Chain {
    pub regs: Vec<Registration>,
    pub invert: Vec<bool>,
}

impl Chain {
    pub fn new(regs: Vec<Registration>, invert: Vec<bool>) -> Result<Self, CmtkError> {
        if regs.is_empty() {
            return Err(CmtkError::EmptyChain);
        }
        let invert = if invert.is_empty() {
            vec![false; regs.len()]
        } else {
            invert
        };
        if invert.len() != regs.len() {
            return Err(CmtkError::BadValue {
                key: "invert".to_string(),
                msg: format!("expected {} flags, got {}", regs.len(), invert.len()),
            });
        }
        Ok(Chain { regs, invert })
    }

    pub fn from_paths(paths: &[PathBuf], invert: &[bool]) -> Result<Self, CmtkError> {
        let regs = paths
            .iter()
            .map(|p| Registration::from_path(p))
            .collect::<Result<Vec<_>, _>>()?;
        Chain::new(regs, invert.to_vec())
    }

    pub fn n_registrations(&self) -> usize {
        self.regs.len()
    }
}

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// Affine only, ignoring any spline warp.
    Affine,
    /// The full warp. Falls back to the affine for registrations that have no spline.
    Warp,
}

#[derive(Clone, Copy)]
pub struct XformOpts<'a> {
    pub mode: Mode,
    /// Evaluate points outside the domain box by clamping to the outermost control points,
    /// instead of failing them.
    ///
    /// **Defaults to `false`, which is what CMTK does**: `streamxform` reports a point
    /// outside `[0, domain]` as `FAILED`, and we return `NaN`. Turning this on extends the
    /// boundary control points outward so every point gets *an* answer — but that answer is
    /// a clamped-boundary extrapolation of a warp that was never fitted there, and it
    /// silently disagrees with every other CMTK-based tool. The reference implementation
    /// defaults it on; we do not.
    pub allow_extrapolation: bool,
    /// Replace failed rows with the affine result rather than `NaN`. Only reachable when
    /// `allow_extrapolation` is false, since extrapolation otherwise never fails.
    pub fallback_to_affine: bool,
    pub threads: Option<usize>,
    pub progress: bool,
    pub cancel: Option<&'a AtomicBool>,
}

impl Default for XformOpts<'_> {
    fn default() -> Self {
        XformOpts {
            mode: Mode::Warp,
            allow_extrapolation: false,
            fallback_to_affine: false,
            threads: None,
            progress: false,
            cancel: None,
        }
    }
}

#[derive(Clone, Copy)]
pub struct InverseOpts<'a> {
    pub mode: Mode,
    pub max_iter: usize,
    /// Stop once the (post-clamp) step falls below this.
    pub tolerance: f64,
    /// Accept the solution only if the final residual is within this of the target;
    /// otherwise the row is `NaN`. `streamxform` behaves the same way, printing `FAILED`.
    pub accuracy: f64,
    /// Confine the iterate to the spline's domain box.
    ///
    /// **This is what makes the inverse agree with `streamxform`.** Turning it off finds
    /// the "pure" preimage even when it lies outside the image domain — which means
    /// returning a finite answer where CMTK reports failure. Leave it on unless you know
    /// you want to disagree with CMTK.
    pub clamp_to_domain: bool,
    pub threads: Option<usize>,
    pub progress: bool,
    pub cancel: Option<&'a AtomicBool>,
}

impl Default for InverseOpts<'_> {
    fn default() -> Self {
        InverseOpts {
            mode: Mode::Warp,
            max_iter: 50,
            tolerance: 1e-9,
            accuracy: 1e-3,
            clamp_to_domain: true,
            threads: None,
            progress: false,
            cancel: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Batch drivers
// ---------------------------------------------------------------------------

/// One hop of a chain, resolved for the direction we are actually travelling.
///
/// Building this list up front means the affine inverses are computed once per call rather
/// than once per point, and the forward and inverse drivers share one kernel.
enum Step<'a> {
    Affine(Affine),
    Warp {
        spline: &'a SplineWarp,
        /// The affine to fall back to when the spline yields nothing.
        fallback: Option<Affine>,
    },
    WarpInverse(&'a SplineWarp),
}

/// Resolve `chain` into the ordered steps for a traversal. `reverse` walks the chain
/// backwards and flips every hop's direction — i.e. it inverts the whole composition.
fn plan_steps<'a>(
    chain: &'a Chain,
    mode: Mode,
    fallback_to_affine: bool,
    reverse: bool,
) -> Result<Vec<Step<'a>>, CmtkError> {
    let n = chain.regs.len();
    let order: Vec<usize> = if reverse {
        (0..n).rev().collect()
    } else {
        (0..n).collect()
    };

    let mut steps = Vec::with_capacity(n);
    for i in order {
        let reg = &chain.regs[i];
        let backwards = chain.invert[i] != reverse; // XOR

        let use_affine = mode == Mode::Affine || reg.spline.is_none();
        if use_affine {
            let aff = reg.affine.ok_or(CmtkError::NoAffine)?;
            let aff = if backwards {
                aff.inverse().ok_or(CmtkError::SingularAffine)?
            } else {
                aff
            };
            steps.push(Step::Affine(aff));
        } else {
            let spline = reg.spline.as_ref().unwrap();
            if backwards {
                steps.push(Step::WarpInverse(spline));
            } else {
                let fallback = if fallback_to_affine {
                    Some(reg.affine.ok_or(CmtkError::NoAffine)?)
                } else {
                    None
                };
                steps.push(Step::Warp { spline, fallback });
            }
        }
    }
    Ok(steps)
}

/// Push one point through every step. `guess` seeds the first iterative step only — beyond
/// the first hop we are in an intermediate space we know nothing about, so the target is
/// the only sensible starting point.
#[inline]
fn run_steps(
    steps: &[Step],
    p: [f64; 3],
    allow_extrapolation: bool,
    iopts: &InverseOpts,
    mut guess: Option<[f64; 3]>,
) -> Option<[f64; 3]> {
    let mut cur = p;
    for step in steps {
        cur = match step {
            Step::Affine(a) => a.apply(cur),
            Step::Warp { spline, fallback } => match spline.apply(cur, allow_extrapolation) {
                Some(v) => v,
                None => fallback.as_ref()?.apply(cur),
            },
            Step::WarpInverse(spline) => {
                let x0 = guess.take().unwrap_or(cur);
                spline.solve_inverse(cur, x0, iopts)?
            }
        };
    }
    Some(cur)
}

fn points_to_vec(points: ArrayView2<f64>) -> Result<Vec<[f64; 3]>, CmtkError> {
    if points.ncols() != 3 {
        return Err(CmtkError::BadShape {
            got: points.shape().to_vec(),
        });
    }
    Ok(points.rows().into_iter().map(|r| [r[0], r[1], r[2]]).collect())
}

fn finish(res: Vec<[f64; 3]>, cancel: Option<&AtomicBool>) -> Result<Array2<f64>, CmtkError> {
    if is_cancelled(cancel) {
        return Err(CmtkError::Cancelled);
    }
    let n = res.len();
    let flat: Vec<f64> = res.into_iter().flatten().collect();
    Ok(Array2::from_shape_vec((n, 3), flat).expect("(n, 3) from 3n values"))
}

/// Drive `points` through `steps` in parallel, writing `NaN` for rows that fail.
#[allow(clippy::too_many_arguments)]
fn drive(
    steps: &[Step],
    pts: &[[f64; 3]],
    allow_extrapolation: bool,
    iopts: &InverseOpts,
    guesses: Option<&[[f64; 3]]>,
    threads: Option<usize>,
    progress: bool,
    cancel: Option<&AtomicBool>,
) -> Vec<[f64; 3]> {
    let n = pts.len();
    let mut res: Vec<[f64; 3]> = vec![[f64::NAN; 3]; n];
    let bar = progress.then(|| make_bar("CMTK", n as u64));

    with_pool(threads, || {
        res.par_chunks_mut(CHUNK)
            .enumerate()
            .for_each(|(ci, out)| {
                if is_cancelled(cancel) {
                    return;
                }
                let start = ci * CHUNK;
                for (k, dst) in out.iter_mut().enumerate() {
                    let i = start + k;
                    let guess = guesses.map(|g| g[i]);
                    if let Some(v) = run_steps(steps, pts[i], allow_extrapolation, iopts, guess) {
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
/// Rows that cannot be transformed — outside the lattice with `allow_extrapolation` off and
/// no affine fallback — come back as `NaN`.
pub fn transform_points(
    chain: &Chain,
    points: ArrayView2<f64>,
    opts: XformOpts,
) -> Result<Array2<f64>, CmtkError> {
    let pts = points_to_vec(points)?;
    let steps = plan_steps(chain, opts.mode, opts.fallback_to_affine, false)?;
    // Only reachable if the chain traverses a hop backwards; forward chains hold no
    // iterative steps, and the defaults are then never consulted.
    let iopts = InverseOpts::default();
    let res = drive(
        &steps,
        &pts,
        opts.allow_extrapolation,
        &iopts,
        None,
        opts.threads,
        opts.progress,
        opts.cancel,
    );
    finish(res, opts.cancel)
}

/// Inverse-transform `points` through `chain` — i.e. find the inputs that
/// [`transform_points`] would map onto them.
///
/// The affine part is inverted exactly. The spline part has no closed-form inverse and is
/// solved per point; rows that do not converge come back as `NaN`, which is what CMTK's
/// `streamxform` reports as `FAILED`.
///
/// `initial_guess` (one point per input point, if given) seeds the first iterative solve.
pub fn inverse_transform_points(
    chain: &Chain,
    points: ArrayView2<f64>,
    initial_guess: Option<ArrayView2<f64>>,
    opts: InverseOpts,
) -> Result<Array2<f64>, CmtkError> {
    let pts = points_to_vec(points)?;
    let guesses = initial_guess.map(points_to_vec).transpose()?;
    if let Some(g) = &guesses {
        if g.len() != pts.len() {
            return Err(CmtkError::GuessLen {
                got: g.len(),
                want: pts.len(),
            });
        }
    }
    // `reverse = true`: walk the chain backwards, flipping every hop.
    let steps = plan_steps(chain, opts.mode, false, true)?;
    let res = drive(
        &steps,
        &pts,
        /* allow_extrapolation = */ true,
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

    const DATA: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/testdata");

    fn jfrc2_fcwb() -> Registration {
        Registration::from_path(Path::new(&format!("{DATA}/JFRC2_FCWB.list"))).unwrap()
    }

    fn tiny() -> Registration {
        Registration::from_path(Path::new(&format!("{DATA}/tiny_warp.list"))).unwrap()
    }

    fn chain_of(reg: Registration) -> Chain {
        Chain::new(vec![reg], vec![false]).unwrap()
    }

    /// The `streamxform`-generated golden data, keyed by case.
    fn golden() -> HashMap<String, Vec<[f64; 3]>> {
        let mut rdr = csv::Reader::from_path(format!("{DATA}/streamxform_golden.csv")).unwrap();
        let mut out: HashMap<String, Vec<[f64; 3]>> = HashMap::new();
        for rec in rdr.records() {
            let r = rec.unwrap();
            let case = r[0].to_string();
            let p = [
                r[2].parse::<f64>().unwrap(),
                r[3].parse::<f64>().unwrap(),
                r[4].parse::<f64>().unwrap(),
            ];
            out.entry(case).or_default().push(p);
        }
        out
    }

    fn arr(pts: &[[f64; 3]]) -> Array2<f64> {
        Array2::from_shape_fn((pts.len(), 3), |(i, j)| pts[i][j])
    }

    /// Compare against golden, treating NaN as a value that must also be NaN.
    fn assert_close(got: &Array2<f64>, want: &[[f64; 3]], atol: f64, what: &str) {
        assert_eq!(got.nrows(), want.len(), "{what}: row count");
        for i in 0..want.len() {
            for j in 0..3 {
                let (g, w) = (got[[i, j]], want[i][j]);
                if w.is_nan() {
                    assert!(g.is_nan(), "{what}: row {i} col {j} should be NaN, got {g}");
                } else {
                    assert!(
                        (g - w).abs() <= atol,
                        "{what}: row {i} col {j}: got {g}, want {w} (|Δ| = {})",
                        (g - w).abs()
                    );
                }
            }
        }
    }

    // ---- parser ----

    #[test]
    fn header_rejected_when_not_typedstream() {
        let err = Registration::from_typedstream("registration {\n}\n", "x").unwrap_err();
        assert!(matches!(err, CmtkError::NotTypedStream { .. }));
    }

    #[test]
    fn parses_nested_blocks_bools_and_quoted_strings() {
        let reg = tiny();
        assert_eq!(reg.version, "2.4");
        assert!(!reg.legacy);
        let s = reg.spline.as_ref().unwrap();
        assert!(!s.absolute, "`absolute no` must parse as false");
        assert!(s.affine.is_some(), "the nested affine_xform must be parsed");
        assert!(reg.affine.is_some());
    }

    #[test]
    fn quoted_strings_keep_their_spaces() {
        let text = std::fs::read_to_string(format!("{DATA}/tiny_warp.list/registration")).unwrap();
        let (root, _) = parse_typedstream(&text, "x").unwrap();
        let reg = root.block("registration").unwrap().unwrap();
        assert_eq!(
            reg.get("reference_study"),
            Some(&Value::Str("images/tiny reference.nrrd".to_string()))
        );
    }

    #[test]
    fn repeated_keys_accumulate() {
        let text = std::fs::read_to_string(format!("{DATA}/tiny_warp.list/registration")).unwrap();
        let (root, _) = parse_typedstream(&text, "x").unwrap();
        let reg = root.block("registration").unwrap().unwrap();
        assert_eq!(reg.get_all("comment").len(), 2);
        assert_eq!(reg.get("comment"), Some(&Value::Str("first".to_string())));
    }

    #[test]
    fn coefficients_span_continuation_lines() {
        let s = tiny().spline.unwrap();
        assert_eq!(s.coefficients.len(), 125);
        // x fastest: idx = ix + nx*(iy + ny*iz); the fixture stores (0.1*ix, 0.2*iy, 0.3*iz).
        assert_eq!(s.coefficients[0], [0.0, 0.0, 0.0]);
        assert_eq!(s.coefficients[1], [0.1, 0.0, 0.0]); // +1 in x
        assert_eq!(s.coefficients[5], [0.0, 0.2, 0.0]); // +1 in y (nx = 5)
        assert_eq!(s.coefficients[25], [0.0, 0.0, 0.3]); // +1 in z (nx*ny = 25)
    }

    #[test]
    fn active_field_is_consumed_and_ignored() {
        // `active`'s continuation lines must not be mistaken for entries — if they were,
        // the closing `}` would be mis-parsed and `dims`/`coefficients` would be lost.
        let reg = tiny();
        let s = reg.spline.as_ref().unwrap();
        assert_eq!(s.dims, [5, 5, 5]);
        assert_eq!(s.coefficients.len(), 125);
        // And the outer block still closed properly, so the top-level affine is intact.
        assert!(reg.affine.is_some());
    }

    #[test]
    fn coefficients_before_dims_errors() {
        let text = "! TYPEDSTREAM 1.1\n\nregistration {\n\tspline_warp {\n\t\tcoefficients 1 2 3\n\t\tdims 5 5 5\n\t}\n}\n";
        assert_eq!(
            Registration::from_typedstream(text, "x").unwrap_err(),
            CmtkError::CoefficientsBeforeDims
        );
    }

    #[test]
    fn coefficient_count_mismatch_errors() {
        let text = "! TYPEDSTREAM 1.1\n\nregistration {\n\tspline_warp {\n\t\tdims 4 4 4\n\t\tcoefficients 1 2 3\n\t}\n}\n";
        assert_eq!(
            Registration::from_typedstream(text, "x").unwrap_err(),
            CmtkError::CoefficientCount { got: 3, want: 192 }
        );
    }

    #[test]
    fn duplicate_block_errors() {
        let text = "! TYPEDSTREAM 1.1\n\nregistration {\n\taffine_xform {\n\t\txlate 0 0 0\n\t}\n\taffine_xform {\n\t\txlate 1 1 1\n\t}\n}\n";
        assert_eq!(
            Registration::from_typedstream(text, "x").unwrap_err(),
            CmtkError::DuplicateBlock {
                key: "affine_xform"
            }
        );
    }

    #[test]
    fn no_registration_block_errors() {
        let text = "! TYPEDSTREAM 1.1\n\nsomething_else {\n\tfoo 1\n}\n";
        assert_eq!(
            Registration::from_typedstream(text, "x").unwrap_err(),
            CmtkError::NoRegistrationBlock
        );
    }

    #[test]
    fn gzip_is_detected_by_magic_bytes() {
        // The bundled fixture is gzipped under a `.gz` name, but the magic-byte test is
        // what actually drives the decision, so a plain file must still work too.
        let gz = jfrc2_fcwb();
        let plain = tiny();
        assert_eq!(gz.spline.as_ref().unwrap().dims, [59, 27, 11]);
        assert_eq!(plain.spline.as_ref().unwrap().dims, [5, 5, 5]);
    }

    #[test]
    fn missing_path_errors() {
        let err = Registration::from_path(Path::new("/nonexistent/xyz.list")).unwrap_err();
        assert!(matches!(err, CmtkError::Io { .. }));
    }

    #[test]
    fn version_gate() {
        assert!(is_legacy_version("1.1"));
        assert!(is_legacy_version("2.3.9"));
        assert!(is_legacy_version("2.0"));
        assert!(!is_legacy_version("2.4"));
        assert!(!is_legacy_version("2.4.1"));
        assert!(!is_legacy_version("3.3.1"));
        assert!(!is_legacy_version("garbage")); // matches the reference: unparseable => modern
    }

    #[test]
    fn loads_bundled_registration() {
        let reg = jfrc2_fcwb();
        assert_eq!(reg.version, "1.1");
        assert!(reg.legacy);
        let s = reg.spline.as_ref().unwrap();
        assert_eq!(s.dims, [59, 27, 11]);
        assert!(s.absolute);
        assert_eq!(s.coefficients.len(), 59 * 27 * 11);
        assert_eq!(s.coefficients.len(), 17_523);
        for (i, want) in [11.3642, 13.2453, 16.8741].iter().enumerate() {
            assert!((s.spacing[i] - want).abs() < 1e-3, "spacing {i}");
        }
        // The CMTK lattice is padded one cell outside the image domain.
        for i in 0..3 {
            assert!((s.origin[i] + s.spacing[i]).abs() < 1e-6, "origin == -spacing");
        }
    }

    // ---- affine ----

    #[test]
    fn affine_forward_matches_streamxform() {
        let g = golden();
        let chain = chain_of(jfrc2_fcwb());
        let out = transform_points(
            &chain,
            arr(&g["input"]).view(),
            XformOpts {
                mode: Mode::Affine,
                ..Default::default()
            },
        )
        .unwrap();
        assert_close(&out, &g["affine_forward"], 1e-4, "affine forward");
    }

    #[test]
    fn affine_inverse_matches_streamxform() {
        let g = golden();
        let chain = chain_of(jfrc2_fcwb());
        let out = inverse_transform_points(
            &chain,
            arr(&g["input"]).view(),
            None,
            InverseOpts {
                mode: Mode::Affine,
                ..Default::default()
            },
        )
        .unwrap();
        assert_close(&out, &g["affine_inverse"], 1e-4, "affine inverse");
    }

    #[test]
    fn affine_inverse_is_exact() {
        let a = jfrc2_fcwb().affine.unwrap();
        let inv = a.inverse().unwrap();
        let prod = mat_mul(&a.m, &inv.m);
        for i in 0..4 {
            for j in 0..4 {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((prod[i][j] - want).abs() < 1e-12, "M @ M^-1 at ({i},{j})");
            }
        }
        let p = [123.0, 45.0, 67.0];
        let back = inv.apply(a.apply(p));
        for i in 0..3 {
            assert!((back[i] - p[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn modern_affine_path_differs_from_legacy() {
        // There is no golden file for a >= 2.4 registration, so pin the modern composition
        // against a hand-computed matrix: no rotation, so the result is exactly the
        // upper-triangular scale/shear, with the translation taken about the centre.
        let a = Affine::from_params(
            [1.0, 2.0, 3.0],
            [0.0, 0.0, 0.0],
            [2.0, 3.0, 4.0],
            [0.5, 0.25, 0.125],
            [0.0, 0.0, 0.0],
            false,
        );
        let want = [
            [2.0, 0.5, 0.25, 1.0],
            [0.0, 3.0, 0.125, 2.0],
            [0.0, 0.0, 4.0, 3.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        for i in 0..4 {
            for j in 0..4 {
                assert!(
                    (a.m[i][j] - want[i][j]).abs() < 1e-12,
                    "modern ({i},{j}): got {}, want {}",
                    a.m[i][j],
                    want[i][j]
                );
            }
        }
        // The legacy composition of the same parameters is a genuinely different matrix —
        // which is why the version gate matters.
        let legacy = Affine::from_params(
            [1.0, 2.0, 3.0],
            [0.0, 0.0, 0.0],
            [2.0, 3.0, 4.0],
            [0.5, 0.25, 0.125],
            [0.0, 0.0, 0.0],
            true,
        );
        assert!(legacy != a);
    }

    #[test]
    fn singular_affine_has_no_inverse() {
        let a = Affine::from_params(
            [0.0; 3],
            [0.0; 3],
            [0.0, 1.0, 1.0], // zero scale on x
            [0.0; 3],
            [0.0; 3],
            true,
        );
        assert!(a.inverse().is_none());
    }

    // ---- spline forward ----

    #[test]
    fn warp_forward_matches_streamxform() {
        let g = golden();
        let chain = chain_of(jfrc2_fcwb());
        let out =
            transform_points(&chain, arr(&g["input"]).view(), XformOpts::default()).unwrap();
        assert_close(&out, &g["warp_forward"], 1e-4, "warp forward");
    }

    #[test]
    fn jacobian_matches_central_differences() {
        let s = jfrc2_fcwb().spline.unwrap();
        let h = 1e-5;
        for p in [
            [50.0, 50.0, 50.0],
            [100.0, 100.0, 20.0],
            [250.0, 150.0, 60.0],
            [300.0, 200.0, 80.0],
            [420.0, 90.0, 30.0],
        ] {
            let (_, j) = s.apply_with_jacobian(p, true).unwrap();
            for c in 0..3 {
                let (mut a, mut b) = (p, p);
                a[c] += h;
                b[c] -= h;
                let fa = s.apply(a, true).unwrap();
                let fb = s.apply(b, true).unwrap();
                for r in 0..3 {
                    let fd = (fa[r] - fb[r]) / (2.0 * h);
                    assert!(
                        (j[r][c] - fd).abs() < 1e-6,
                        "J[{r}][{c}] at {p:?}: analytic {}, central-diff {fd}",
                        j[r][c]
                    );
                }
            }
        }
    }

    #[test]
    fn jacobian_adds_identity_when_relative() {
        // The reference returns the raw spline Jacobian even in relative mode, where the
        // transform is `p + spline(p)` and the true Jacobian is `I + J`. Verify against
        // central differences on an `absolute no` warp.
        let s = tiny().spline.unwrap();
        assert!(!s.absolute);
        let p = [1.0, 1.5, 2.0];
        let (_, j) = s.apply_with_jacobian(p, true).unwrap();
        let h = 1e-6;
        for c in 0..3 {
            let (mut a, mut b) = (p, p);
            a[c] += h;
            b[c] -= h;
            let fa = s.apply(a, true).unwrap();
            let fb = s.apply(b, true).unwrap();
            for r in 0..3 {
                let fd = (fa[r] - fb[r]) / (2.0 * h);
                assert!(
                    (j[r][c] - fd).abs() < 1e-6,
                    "relative J[{r}][{c}]: analytic {}, central-diff {fd}",
                    j[r][c]
                );
            }
        }
        // And the diagonal really is ~1 + small, not ~small.
        for r in 0..3 {
            assert!(j[r][r] > 0.9, "diagonal {r} should carry the identity");
        }
    }

    #[test]
    fn extrapolation_rejected_when_disallowed() {
        let s = jfrc2_fcwb().spline.unwrap();
        assert!(s.apply([-500.0, -500.0, -500.0], false).is_none());
        // Just outside on one axis only is still outside.
        assert!(s.apply([-0.001, 100.0, 50.0], false).is_none());
        assert!(s.apply([100.0, 100.0, s.domain[2] + 0.001], false).is_none());
        // Just inside is fine.
        assert!(s.apply([0.0, 0.0, 0.0], false).is_some());
        assert!(s.apply([100.0, 100.0, s.domain[2] - 0.001], false).is_some());
    }

    #[test]
    fn domain_is_the_world_box_not_the_lattice_extent() {
        // Verified against the real `streamxform` on 3000 points: CMTK accepts exactly
        // `0 <= p <= domain`. The control-point lattice is padded one cell *outside* that
        // (origin == -spacing), so the lattice-parameter test `u in [0, dims-3)` describes a
        // box shifted by one spacing — which is what the reference implementation uses, and
        // it agrees with CMTK nowhere. Guard against anyone "simplifying" back to it.
        let s = jfrc2_fcwb().spline.unwrap();

        assert!(s.in_domain([0.0, 0.0, 0.0]));
        assert!(s.in_domain(s.domain));
        assert!(!s.in_domain([-0.001, 0.0, 0.0]));
        assert!(!s.in_domain([0.0, 0.0, s.domain[2] + 0.001]));

        // A point the *lattice* box would accept but CMTK does not: u just below 1 on x,
        // i.e. world x just below 0.
        let p = [s.origin[0] + 0.5 * s.spacing[0], 100.0, 50.0]; // u_x = 0.5 -> in [0, dims-3)
        let u_x = (p[0] - s.origin[0]) / s.spacing[0];
        assert!((0.0..(s.dims[0] - 3) as f64).contains(&u_x), "lattice box would accept it");
        assert!(p[0] < 0.0, "but it is outside the world domain box");
        assert!(!s.in_domain(p));
        assert!(s.apply(p, false).is_none());
    }

    #[test]
    fn clamping_is_the_extrapolation_policy() {
        let s = jfrc2_fcwb().spline.unwrap();
        let far = [-500.0, -500.0, -500.0];
        let v = s.apply(far, true).expect("extrapolation yields a value");
        assert!(v.iter().all(|x| x.is_finite()));
    }

    // ---- inverse ----

    #[test]
    fn warp_inverse_matches_streamxform() {
        // The headline test: all five golden rows, *including the two NaNs* that
        // `streamxform` reports as FAILED. Solved cold — no seeding from the answer.
        let g = golden();
        let chain = chain_of(jfrc2_fcwb());
        let out = inverse_transform_points(
            &chain,
            arr(&g["input"]).view(),
            None,
            InverseOpts::default(),
        )
        .unwrap();
        assert_close(&out, &g["warp_inverse"], 1e-4, "warp inverse");
        assert!(out[[0, 0]].is_nan() && out[[4, 0]].is_nan());
    }

    #[test]
    fn warp_inverse_roundtrip() {
        let chain = chain_of(jfrc2_fcwb());
        let pts = [
            [50.0, 50.0, 50.0],
            [100.0, 100.0, 20.0],
            [250.0, 150.0, 60.0],
            [300.0, 120.0, 70.0],
        ];
        let fwd = transform_points(&chain, arr(&pts).view(), XformOpts::default()).unwrap();
        let back =
            inverse_transform_points(&chain, fwd.view(), None, InverseOpts::default()).unwrap();
        for i in 0..pts.len() {
            for j in 0..3 {
                assert!(
                    (back[[i, j]] - pts[i][j]).abs() < 1e-6,
                    "roundtrip row {i} col {j}: {} vs {}",
                    back[[i, j]],
                    pts[i][j]
                );
            }
        }
    }

    #[test]
    fn diverging_point_yields_nan_not_infinity() {
        // Undamped Gauss-Newton runs away to ~-2e16 on this point. LM + the domain clamp
        // must instead reject it cleanly.
        let s = jfrc2_fcwb().spline.unwrap();
        let target = [444.617126, 248.87099, 66.5127818]; // golden warp_forward[4]
        // Solving for the *preimage of the forward image* converges...
        assert!(s.solve_inverse(target, target, &InverseOpts::default()).is_some());
        // ...but the raw input point [500, 250, 100] has no preimage in the domain.
        let hopeless = [500.0, 250.0, 100.0];
        assert!(s
            .solve_inverse(hopeless, hopeless, &InverseOpts::default())
            .is_none());
    }

    #[test]
    fn clamp_to_domain_off_finds_preimages_cmtk_rejects() {
        // The points CMTK calls FAILED *do* have preimages — they just lie outside the
        // image domain. Lifting the box finds them; that is the whole point of the flag,
        // and the reason it defaults to on.
        let g = golden();
        let s = jfrc2_fcwb().spline.unwrap();
        let target = g["input"][0]; // [0, 0, 0] — streamxform: FAILED

        assert!(s
            .solve_inverse(target, target, &InverseOpts::default())
            .is_none());

        let loose = InverseOpts {
            clamp_to_domain: false,
            ..Default::default()
        };
        let x = s
            .solve_inverse(target, target, &loose)
            .expect("the preimage exists once the domain box is lifted");
        // It really is a preimage...
        let fwd = s.apply(x, true).unwrap();
        assert!(sq_dist(fwd, target).sqrt() < 1e-6);
        // ...and it lies outside the domain box, which is exactly why CMTK rejects it.
        let (lo, hi) = s.domain_box();
        assert!(
            (0..3).any(|i| x[i] < lo[i] || x[i] > hi[i]),
            "expected an out-of-domain preimage, got {x:?}"
        );
    }

    #[test]
    fn inverse_is_thread_count_invariant() {
        let g = golden();
        let chain = chain_of(jfrc2_fcwb());
        let one = inverse_transform_points(
            &chain,
            arr(&g["input"]).view(),
            None,
            InverseOpts {
                threads: Some(1),
                ..Default::default()
            },
        )
        .unwrap();
        let many =
            inverse_transform_points(&chain, arr(&g["input"]).view(), None, InverseOpts::default())
                .unwrap();
        for i in 0..one.nrows() {
            for j in 0..3 {
                let (a, b) = (one[[i, j]], many[[i, j]]);
                assert!(
                    (a.is_nan() && b.is_nan()) || a == b,
                    "row {i} col {j} not thread-invariant: {a} vs {b}"
                );
            }
        }
    }

    // ---- chain ----

    #[test]
    fn chain_of_two_equals_manual_double_application() {
        let one = chain_of(jfrc2_fcwb());
        let two = Chain::new(vec![jfrc2_fcwb(), jfrc2_fcwb()], vec![false, false]).unwrap();
        let pts = [[50.0, 50.0, 50.0], [100.0, 100.0, 20.0]];
        let once = transform_points(&one, arr(&pts).view(), XformOpts::default()).unwrap();
        let manual = transform_points(&one, once.view(), XformOpts::default()).unwrap();
        let chained = transform_points(&two, arr(&pts).view(), XformOpts::default()).unwrap();
        for i in 0..pts.len() {
            for j in 0..3 {
                assert!((chained[[i, j]] - manual[[i, j]]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn chain_inverse_reverses_order() {
        // A chain of [reg, reg] inverted must undo both hops, back to the input.
        let two = Chain::new(vec![jfrc2_fcwb(), jfrc2_fcwb()], vec![false, false]).unwrap();
        let pts = [[100.0, 100.0, 20.0], [250.0, 150.0, 60.0]];
        let fwd = transform_points(&two, arr(&pts).view(), XformOpts::default()).unwrap();
        let back = inverse_transform_points(&two, fwd.view(), None, InverseOpts::default()).unwrap();
        for i in 0..pts.len() {
            for j in 0..3 {
                assert!(
                    (back[[i, j]] - pts[i][j]).abs() < 1e-4,
                    "row {i} col {j}: {} vs {}",
                    back[[i, j]],
                    pts[i][j]
                );
            }
        }
    }

    #[test]
    fn invert_flag_undoes_the_forward_transform() {
        let fwd = chain_of(jfrc2_fcwb());
        let bwd = Chain::new(vec![jfrc2_fcwb()], vec![true]).unwrap();
        let pts = [[100.0, 100.0, 20.0], [250.0, 150.0, 60.0]];
        let out = transform_points(&fwd, arr(&pts).view(), XformOpts::default()).unwrap();
        // Traversing the same registration backwards is the inverse transform.
        let back = transform_points(&bwd, out.view(), XformOpts::default()).unwrap();
        for i in 0..pts.len() {
            for j in 0..3 {
                assert!((back[[i, j]] - pts[i][j]).abs() < 1e-4);
            }
        }
    }

    #[test]
    fn fallback_to_affine_fills_failed_rows() {
        let chain = chain_of(jfrc2_fcwb());
        let pts = [[-5000.0, -5000.0, -5000.0], [100.0, 100.0, 20.0]];
        // Without extrapolation the far point fails...
        let strict = transform_points(
            &chain,
            arr(&pts).view(),
            XformOpts {
                allow_extrapolation: false,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(strict[[0, 0]].is_nan());
        assert!(!strict[[1, 0]].is_nan());
        // ...and with the fallback it takes the affine result instead.
        let filled = transform_points(
            &chain,
            arr(&pts).view(),
            XformOpts {
                allow_extrapolation: false,
                fallback_to_affine: true,
                ..Default::default()
            },
        )
        .unwrap();
        let aff = transform_points(
            &chain,
            arr(&pts).view(),
            XformOpts {
                mode: Mode::Affine,
                ..Default::default()
            },
        )
        .unwrap();
        for j in 0..3 {
            assert!((filled[[0, j]] - aff[[0, j]]).abs() < 1e-12);
        }
    }

    #[test]
    fn cancel_flag_short_circuits() {
        let chain = chain_of(jfrc2_fcwb());
        let cancel = AtomicBool::new(true);
        let pts = [[50.0, 50.0, 50.0]];
        let err = transform_points(
            &chain,
            arr(&pts).view(),
            XformOpts {
                cancel: Some(&cancel),
                ..Default::default()
            },
        )
        .unwrap_err();
        assert_eq!(err, CmtkError::Cancelled);
    }

    #[test]
    fn bad_shape_errors() {
        let chain = chain_of(jfrc2_fcwb());
        let pts = Array2::<f64>::zeros((4, 2));
        let err = transform_points(&chain, pts.view(), XformOpts::default()).unwrap_err();
        assert_eq!(err, CmtkError::BadShape { got: vec![4, 2] });
    }

    #[test]
    fn empty_chain_errors() {
        assert_eq!(Chain::new(vec![], vec![]).unwrap_err(), CmtkError::EmptyChain);
    }
}
