use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::nblast::run_interruptible;
use fastcore::elastix::{
    inverse_transform_points, probe_invertible, transform_points, Chain, ElastixError,
    ElastixTransform, InverseOpts, OutOfBounds, XformOpts,
};

fn to_py_err(e: ElastixError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// Whether an Elastix transform can be inverted — **without reading its coefficients.**
///
/// A file is not invertible exactly when some step in its chain combines via `Add`. That fact
/// lives in one short key, but it sits *after* a coefficient array that can run to 56 MB — so
/// answering it used to cost a full parse. This reads the same chain and skips only the numbers:
/// about **20x** faster, and 200x on the big ones.
///
/// For callers that must label many files up front — a bridging graph with dozens of
/// registrations cannot afford a full parse per edge just to decide which way each one goes.
///
/// Raises `ValueError` for anything that would not load at all (missing, not elastix, unsupported
/// transform kind, binary parameters, circular chain), so `True` is a promise rather than a guess.
#[pyfunction]
#[pyo3(name = "probe_elastix_invertible", signature = (path))]
pub fn py_probe_elastix_invertible(path: &str) -> PyResult<bool> {
    probe_invertible(std::path::Path::new(path)).map_err(to_py_err)
}

fn oob_of(s: &str) -> PyResult<OutOfBounds> {
    match s {
        "identity" => Ok(OutOfBounds::Identity),
        "nan" => Ok(OutOfBounds::Nan),
        other => Err(PyValueError::new_err(format!(
            "`out_of_bounds` must be 'identity' or 'nan', got {other:?}"
        ))),
    }
}

/// An Elastix transform (or a chain of them), parsed once and applied many times.
///
/// Worth a `#[pyclass]` for the same reason `CmtkRegistration` is: a real registration is
/// hundreds of thousands of coefficients read from a file that can reach 56 MB (BANC's), and the
/// calling pattern — transform every neuron in a dataset — applies it many times per load.
///
/// `frozen` is required, not cosmetic: the module is `gil_used = false`, so the class must be
/// `Sync`. It also buys `&self` methods with no borrow-flag check, which matters in the `xform`
/// loop.
#[pyclass(frozen, name = "ElastixTransform", module = "navis_fastcore._fastcore")]
pub struct PyElastixTransform {
    /// `Arc` so the worker closure in `run_interruptible` can capture the chain without cloning
    /// the coefficients per call.
    inner: Arc<Chain>,
    paths: Vec<String>,
}

#[pymethods]
impl PyElastixTransform {
    /// Load one or more Elastix `TransformParameters` files.
    ///
    /// Each file is *already* a chain: its `InitialTransformParametersFileName` is followed
    /// recursively, resolved relative to that file's own directory. So there is no need to copy
    /// supplementary files anywhere, which is what `navis` has to do today.
    ///
    /// The object holds only the *parse*. Direction is chosen per call (`xform`/`xform_inv`, and
    /// the `invert` argument on each), so one instance serves every direction — worth caring about
    /// when BANC's warp is 56 MB.
    ///
    /// Arguments:
    /// - `paths`: one or more `TransformParameters.*.txt` files, applied nose-to-tail.
    #[new]
    #[pyo3(signature = (paths))]
    fn new(paths: Vec<String>) -> PyResult<Self> {
        let pbs: Vec<PathBuf> = paths.iter().map(PathBuf::from).collect();
        let xforms = pbs
            .iter()
            .map(|p| ElastixTransform::from_path(p))
            .collect::<Result<Vec<_>, _>>()
            .map_err(to_py_err)?;
        let chain = Chain::new(xforms).map_err(to_py_err)?;
        Ok(PyElastixTransform {
            inner: Arc::new(chain),
            paths,
        })
    }

    /// Transform points forward.
    ///
    /// Arguments:
    /// - `points`: `(N, 3)` array of coordinates.
    /// - `out_of_bounds`: what to do with points outside a B-spline's control-point grid.
    ///   `"identity"` (default) returns them unchanged, which is exactly what `transformix`
    ///   does. `"nan"` returns `NaN` instead, so you can see the boundary rather than trust it.
    /// - `invert`: per-hop direction flags, one per path. `True` traverses that transform
    ///   backwards. Not the same as `xform_inv`, which reverses the whole composition.
    /// - `n_cores`: cap the thread pool. `None` uses all cores.
    ///
    /// Returns:
    /// An `(N, 3)` array.
    #[pyo3(signature = (points, out_of_bounds="identity", invert=None, n_cores=None,
                        progress=false))]
    fn xform<'py>(
        &self,
        py: Python<'py>,
        points: PyReadonlyArray2<'py, f64>,
        out_of_bounds: &str,
        invert: Option<Vec<bool>>,
        n_cores: Option<usize>,
        progress: bool,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let oob = oob_of(out_of_bounds)?;
        // Materialise before detaching: the borrow cannot outlive the GIL release.
        let pts = points.as_array().to_owned();
        let chain = Arc::clone(&self.inner);
        let cancel = AtomicBool::new(false);

        let out: Array2<f64> = run_interruptible(py, &cancel, || {
            let opts = XformOpts {
                out_of_bounds: oob,
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

    /// Transform points *backwards* — something Elastix itself cannot do.
    ///
    /// The linear parts are inverted exactly. Each B-spline warp has no closed-form inverse and
    /// is solved per point by damped Gauss-Newton against the analytic Jacobian.
    ///
    /// The guarantee is **forward-consistency**: `xform(xform_inv(y)) == y`. It is *not*
    /// guaranteed that `xform_inv(xform(p)) == p`, because a B-spline warp need not be
    /// injective — a strongly folded registration maps several points to the same place, and no
    /// inverse can recover which one you meant. Points with no preimage at all come back `NaN`.
    ///
    /// Arguments:
    /// - `points`: `(N, 3)` array of coordinates.
    /// - `out_of_bounds`: as for `xform`.
    /// - `initial_guess`: `(N, 3)` starting points for the solver. Rarely needed — the solver
    ///   seeds itself with a fixed-point iteration.
    /// - `max_iter` / `tolerance`: solver budget and step-size convergence threshold.
    /// - `seed_iter`: rounds of the fixed-point pre-seed. Zero starts at the target, which
    ///   fails wherever the deformation is large.
    /// - `accuracy`: accept a solution only if its residual is within this of the target.
    /// - `lattice_points`: size of the global seed lattice, the last-resort start for points the
    ///   cheap seeds fail on. Built once per call; only failed points consult it.
    /// - `invert`: as for `xform`, composed with the whole-chain inversion.
    /// - `n_cores`: cap the thread pool. `None` uses all cores.
    ///
    /// Returns:
    /// An `(N, 3)` array. Rows with no preimage are `NaN`.
    #[pyo3(signature = (points, out_of_bounds="identity", initial_guess=None, max_iter=50,
                        seed_iter=8, tolerance=1e-9, accuracy=1e-3, lattice_points=16_000,
                        invert=None, n_cores=None, progress=false))]
    #[allow(clippy::too_many_arguments)]
    fn xform_inv<'py>(
        &self,
        py: Python<'py>,
        points: PyReadonlyArray2<'py, f64>,
        out_of_bounds: &str,
        initial_guess: Option<PyReadonlyArray2<'py, f64>>,
        max_iter: usize,
        seed_iter: usize,
        tolerance: f64,
        accuracy: f64,
        lattice_points: usize,
        invert: Option<Vec<bool>>,
        n_cores: Option<usize>,
        progress: bool,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let oob = oob_of(out_of_bounds)?;
        let pts = points.as_array().to_owned();
        let guess = initial_guess.map(|g| g.as_array().to_owned());
        let chain = Arc::clone(&self.inner);
        let cancel = AtomicBool::new(false);

        let out: Array2<f64> = run_interruptible(py, &cancel, || {
            let opts = InverseOpts {
                out_of_bounds: oob,
                max_iter,
                seed_iter,
                tolerance,
                accuracy,
                lattice_points,
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

    #[getter]
    fn n_transforms(&self) -> usize {
        self.inner.n_transforms()
    }

    /// Whether `xform_inv` can run at all. False only for a chain carrying an `Add` step, which
    /// does not decompose into invertible hops.
    #[getter]
    fn invertible(&self) -> bool {
        self.inner.is_invertible(None)
    }

    /// The resolved step kinds of each transform in the chain, initial first — e.g.
    /// `[["linear", "bspline"]]` for a B-spline composed over an affine.
    #[getter]
    fn kinds(&self) -> Vec<Vec<String>> {
        self.inner
            .xforms
            .iter()
            .map(|x| x.steps.iter().map(|(t, _)| t.kind().to_string()).collect())
            .collect()
    }

    /// The control-point grid size of every B-spline in the chain, `(k, 3)`. `None` if there is
    /// no B-spline (a purely linear chain).
    #[getter]
    fn grid_size<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<i64>>> {
        let rows: Vec<[usize; 3]> = self
            .inner
            .xforms
            .iter()
            .flat_map(|x| x.splines())
            .map(|s| s.size)
            .collect();
        (!rows.is_empty()).then(|| {
            Array2::from_shape_fn((rows.len(), 3), |(i, j)| rows[i][j] as i64).into_pyarray(py)
        })
    }

    /// The control-point spacing of every B-spline in the chain, `(k, 3)`.
    #[getter]
    fn grid_spacing<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<f64>>> {
        let rows: Vec<[f64; 3]> = self
            .inner
            .xforms
            .iter()
            .flat_map(|x| x.splines())
            .map(|s| s.spacing)
            .collect();
        (!rows.is_empty())
            .then(|| Array2::from_shape_fn((rows.len(), 3), |(i, j)| rows[i][j]).into_pyarray(py))
    }

    /// The control-point grid origin of every B-spline in the chain, `(k, 3)`.
    #[getter]
    fn grid_origin<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<f64>>> {
        let rows: Vec<[f64; 3]> = self
            .inner
            .xforms
            .iter()
            .flat_map(|x| x.splines())
            .map(|s| s.origin)
            .collect();
        (!rows.is_empty())
            .then(|| Array2::from_shape_fn((rows.len(), 3), |(i, j)| rows[i][j]).into_pyarray(py))
    }

    /// The 4x4 matrix of the first linear step of the first transform, if it has one.
    #[getter]
    fn affine<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<f64>>> {
        self.inner.xforms[0]
            .linear()
            .map(|l| l.as_array().into_pyarray(py))
    }

    #[getter]
    fn paths(&self) -> Vec<String> {
        self.paths.clone()
    }

    fn __repr__(&self) -> String {
        let n = self.inner.n_transforms();
        let kinds: Vec<String> = self
            .inner
            .xforms
            .iter()
            .map(|x| {
                x.steps
                    .iter()
                    .map(|(t, _)| t.kind())
                    .collect::<Vec<_>>()
                    .join("+")
            })
            .collect();
        format!(
            "ElastixTransform({n} transform{}: {}, paths={:?})",
            if n == 1 { "" } else { "s" },
            kinds.join(" -> "),
            self.paths
        )
    }
}
