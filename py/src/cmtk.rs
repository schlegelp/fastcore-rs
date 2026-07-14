use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::nblast::run_interruptible;
use fastcore::cmtk::{
    inverse_transform_points, transform_points, Chain, CmtkError, Fallback, InverseOpts, Mode,
    Registration, XformOpts,
};

fn to_py_err(e: CmtkError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

fn mode_of(s: &str) -> PyResult<Mode> {
    match s {
        "warp" => Ok(Mode::Warp),
        "affine" => Ok(Mode::Affine),
        other => Err(PyValueError::new_err(format!(
            "`transform` must be 'warp' or 'affine', got {other:?}"
        ))),
    }
}

/// The Python wrapper has already turned `False`/`True`/`"chain"`/`"hop"` into one of these.
fn fallback_of(s: &str) -> PyResult<Fallback> {
    match s {
        "none" => Ok(Fallback::None),
        "chain" => Ok(Fallback::Chain),
        "hop" => Ok(Fallback::Hop),
        other => Err(PyValueError::new_err(format!(
            "`fallback_to_affine` must be False, True, 'chain' or 'hop', got {other:?}"
        ))),
    }
}

/// A CMTK registration (or a chain of them), parsed once and applied many times.
///
/// This is the crate's only `#[pyclass]`, and it earns it: a registration is ~17.5k control
/// points read from a 760 KB file, and the calling pattern (transform every neuron in a
/// dataset) applies it hundreds of times per load. A stateless `f(path, points)` would
/// re-read and re-parse the file on every call.
///
/// `frozen` is required, not cosmetic: the module is `gil_used = false`, so the class must
/// be `Sync`. It also buys `&self` methods with no borrow-flag check, which matters in the
/// `xform` loop.
#[pyclass(frozen, name = "CmtkRegistration", module = "navis_fastcore._fastcore")]
pub struct PyCmtkRegistration {
    /// `Arc` so the worker closure in `run_interruptible` can capture the chain without
    /// cloning ~420 KB of coefficients per call.
    inner: Arc<Chain>,
    paths: Vec<String>,
}

#[pymethods]
impl PyCmtkRegistration {
    /// Load one or more CMTK registrations.
    ///
    /// The object holds only the *parse*. Direction is chosen per call (`xform`/`xform_inv`, and
    /// the `invert` argument on each), so one instance serves every direction.
    ///
    /// Arguments:
    /// - `paths`: one or more `*.list` directories (or `registration` files, plain or
    ///   gzipped), applied nose-to-tail.
    #[new]
    #[pyo3(signature = (paths))]
    fn new(paths: Vec<String>) -> PyResult<Self> {
        let pbs: Vec<PathBuf> = paths.iter().map(PathBuf::from).collect();
        let regs = pbs
            .iter()
            .map(|p| Registration::from_path(p))
            .collect::<Result<Vec<_>, _>>()
            .map_err(to_py_err)?;
        let chain = Chain::new(regs).map_err(to_py_err)?;
        Ok(PyCmtkRegistration {
            inner: Arc::new(chain),
            paths,
        })
    }

    /// Transform points forward through the registration.
    ///
    /// Arguments:
    /// - `points`: `(N, 3)` array of coordinates.
    /// - `transform`: `"warp"` (default) or `"affine"`.
    /// - `allow_extrapolation`: evaluate points outside the domain box by clamping to the
    ///   outermost control points, instead of failing them. Defaults to `False`, which is
    ///   what CMTK does — `streamxform` reports such a point as `FAILED`.
    /// - `fallback_to_affine`: `"none"`, `"chain"` (re-run the whole chain affine-only from the
    ///   original point, as nat/navis do) or `"hop"` (swap the affine in for only the hop that
    ///   failed, keeping the warps that succeeded).
    /// - `invert`: per-hop direction flags, one per path. `True` traverses that registration
    ///   backwards. Not the same as `xform_inv`, which reverses the whole composition.
    /// - `n_cores`: cap the thread pool. `None` uses all cores.
    ///
    /// Returns:
    /// An `(N, 3)` array. Rows that could not be transformed are `NaN`.
    #[pyo3(signature = (points, transform="warp", allow_extrapolation=false,
                        fallback_to_affine="none", invert=None, n_cores=None, progress=false))]
    #[allow(clippy::too_many_arguments)]
    fn xform<'py>(
        &self,
        py: Python<'py>,
        points: PyReadonlyArray2<'py, f64>,
        transform: &str,
        allow_extrapolation: bool,
        fallback_to_affine: &str,
        invert: Option<Vec<bool>>,
        n_cores: Option<usize>,
        progress: bool,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let mode = mode_of(transform)?;
        let fallback = fallback_of(fallback_to_affine)?;
        // Materialise before detaching: the borrow cannot outlive the GIL release.
        let pts = points.as_array().to_owned();
        let chain = Arc::clone(&self.inner);
        let cancel = AtomicBool::new(false);

        let out: Array2<f64> = run_interruptible(py, &cancel, || {
            let opts = XformOpts {
                mode,
                allow_extrapolation,
                fallback,
                invert: invert.as_deref(),
                threads: n_cores,
                progress,
                cancel: Some(&cancel),
            };
            transform_points(&chain, pts.view(), opts)
        })?
        .map_err(to_py_err)?;

        Ok(out.into_pyarray(py))
    }

    /// Transform points *backwards* through the registration.
    ///
    /// The affine part is inverted exactly. The spline part has no closed-form inverse and
    /// is solved per point by damped Gauss-Newton; points that do not converge come back as
    /// `NaN`, which is what CMTK's `streamxform` reports as `FAILED`.
    ///
    /// Arguments:
    /// - `points`: `(N, 3)` array of coordinates.
    /// - `transform`: `"warp"` (default) or `"affine"`.
    /// - `initial_guess`: `(N, 3)` starting points for the solver. Defaults to `points`.
    /// - `max_iter` / `tolerance`: solver budget and step-size convergence threshold.
    /// - `accuracy`: accept a solution only if its residual is within this of the target.
    /// - `clamp_to_domain`: confine the iterate to the spline's domain box. This is what
    ///   makes the result agree with `streamxform`; turning it off finds preimages outside
    ///   the image domain, where CMTK reports failure.
    /// - `fallback_to_affine`: as for `xform`, in the inverse direction.
    /// - `invert`: as for `xform`, composed with the whole-chain inversion.
    /// - `n_cores`: cap the thread pool. `None` uses all cores.
    ///
    /// Returns:
    /// An `(N, 3)` array. Rows that did not converge are `NaN`.
    #[pyo3(signature = (points, transform="warp", initial_guess=None, max_iter=50,
                        tolerance=1e-9, accuracy=1e-3, clamp_to_domain=true,
                        fallback_to_affine="none", invert=None, n_cores=None, progress=false))]
    #[allow(clippy::too_many_arguments)]
    fn xform_inv<'py>(
        &self,
        py: Python<'py>,
        points: PyReadonlyArray2<'py, f64>,
        transform: &str,
        initial_guess: Option<PyReadonlyArray2<'py, f64>>,
        max_iter: usize,
        tolerance: f64,
        accuracy: f64,
        clamp_to_domain: bool,
        fallback_to_affine: &str,
        invert: Option<Vec<bool>>,
        n_cores: Option<usize>,
        progress: bool,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let mode = mode_of(transform)?;
        let fallback = fallback_of(fallback_to_affine)?;
        let pts = points.as_array().to_owned();
        let guess = initial_guess.map(|g| g.as_array().to_owned());
        let chain = Arc::clone(&self.inner);
        let cancel = AtomicBool::new(false);

        let out: Array2<f64> = run_interruptible(py, &cancel, || {
            let opts = InverseOpts {
                mode,
                max_iter,
                tolerance,
                accuracy,
                clamp_to_domain,
                fallback,
                invert: invert.as_deref(),
                threads: n_cores,
                progress,
                cancel: Some(&cancel),
            };
            inverse_transform_points(&chain, pts.view(), guess.as_ref().map(|g| g.view()), opts)
        })?
        .map_err(to_py_err)?;

        Ok(out.into_pyarray(py))
    }

    /// The TypedStream version of each registration in the chain.
    #[getter]
    fn version(&self) -> Vec<String> {
        self.inner.regs.iter().map(|r| r.version.clone()).collect()
    }

    #[getter]
    fn n_registrations(&self) -> usize {
        self.inner.n_registrations()
    }

    /// Whether each registration in the chain carries a spline warp (vs. affine only).
    #[getter]
    fn has_spline(&self) -> Vec<bool> {
        self.inner.regs.iter().map(|r| r.spline.is_some()).collect()
    }

    /// The control-point lattice dimensions of each spline warp, `(k, 3)`. `None` if no
    /// registration in the chain has a spline.
    #[getter]
    fn dims<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<i64>>> {
        let rows: Vec<[usize; 3]> = self
            .inner
            .regs
            .iter()
            .filter_map(|r| r.spline.as_ref().map(|s| s.dims))
            .collect();
        (!rows.is_empty()).then(|| {
            Array2::from_shape_fn((rows.len(), 3), |(i, j)| rows[i][j] as i64).into_pyarray(py)
        })
    }

    /// The control-point spacing of each spline warp, `(k, 3)`.
    #[getter]
    fn spacing<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<f64>>> {
        let rows: Vec<[f64; 3]> = self
            .inner
            .regs
            .iter()
            .filter_map(|r| r.spline.as_ref().map(|s| s.spacing))
            .collect();
        (!rows.is_empty()).then(|| {
            Array2::from_shape_fn((rows.len(), 3), |(i, j)| rows[i][j]).into_pyarray(py)
        })
    }

    /// The 4x4 affine matrix of the first registration in the chain.
    #[getter]
    fn affine<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<f64>>> {
        self.inner.regs[0]
            .affine
            .map(|a| a.as_array().into_pyarray(py))
    }

    #[getter]
    fn paths(&self) -> Vec<String> {
        self.paths.clone()
    }

    fn __repr__(&self) -> String {
        let n = self.inner.n_registrations();
        let kinds: Vec<&str> = self
            .inner
            .regs
            .iter()
            .map(|r| if r.spline.is_some() { "warp" } else { "affine" })
            .collect();
        format!(
            "CmtkRegistration({n} registration{}: {}, paths={:?})",
            if n == 1 { "" } else { "s" },
            kinds.join(" -> "),
            self.paths
        )
    }
}
