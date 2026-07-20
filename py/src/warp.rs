//! Python bindings for the landmark-based warps: thin-plate spline and moving least
//! squares.
//!
//! Both are `#[pyclass]`es rather than free functions because the calling pattern is
//! "build once from a registration's landmarks, apply to every neuron in a dataset". For
//! TPS that matters a great deal — the fit is cubic in the landmark count and would
//! otherwise be repeated on every call.

use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::nblast::run_interruptible;
use fastcore::mls::{MlsError, MlsTransform};
use fastcore::tps::{TpsError, TpsTransform};

fn tps_err(e: TpsError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

fn mls_err(e: MlsError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// Turn a 4x4 array of `f64` into a numpy array.
fn mat4_to_py(py: Python<'_>, m: [[f64; 4]; 4]) -> Bound<'_, PyArray2<f64>> {
    let flat: Vec<f64> = m.iter().flat_map(|r| r.iter().copied()).collect();
    Array2::from_shape_vec((4, 4), flat)
        .expect("4x4 always fits")
        .into_pyarray(py)
}

/// A fitted thin-plate spline.
///
/// `frozen` is load-bearing, not cosmetic: the module is `gil_used = false`, so the class
/// must be `Sync`.
#[pyclass(frozen, name = "TpsTransform", module = "navis_fastcore._fastcore")]
pub struct PyTpsTransform {
    /// `Arc` so the worker closure in `run_interruptible` captures the transform without
    /// cloning the landmarks and weights on every call.
    inner: Arc<TpsTransform>,
}

#[pymethods]
impl PyTpsTransform {
    /// Fit the spline mapping `source` onto `target`. Both must be `(M, 3)`.
    #[new]
    fn new(
        py: Python<'_>,
        source: PyReadonlyArray2<'_, f64>,
        target: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Self> {
        // Materialise before detaching: the borrow cannot outlive the GIL release.
        let src = source.as_array().to_owned();
        let trg = target.as_array().to_owned();
        // The fit is an (M+4) cubic solve — seconds at a few thousand landmarks — so it
        // releases the GIL and stays interruptible just like `xform` does.
        let cancel = AtomicBool::new(false);
        let fitted = run_interruptible(py, &cancel, || {
            TpsTransform::fit(src.view(), trg.view())
        })?
        .map_err(tps_err)?;
        Ok(PyTpsTransform {
            inner: Arc::new(fitted),
        })
    }

    /// Rebuild from coefficients that were fitted elsewhere, skipping the cubic solve.
    ///
    /// This is what unpickling uses — refitting in every `multiprocessing` worker would
    /// cost more than the transform itself — and it also lets a caller fit with LAPACK
    /// (`numpy.linalg.solve`) and still get the fast `xform`.
    #[staticmethod]
    fn from_coefs(
        source: PyReadonlyArray2<'_, f64>,
        w: PyReadonlyArray2<'_, f64>,
        a: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Self> {
        let inner =
            TpsTransform::from_coefs(source.as_array(), w.as_array(), a.as_array()).map_err(tps_err)?;
        Ok(PyTpsTransform {
            inner: Arc::new(inner),
        })
    }

    /// Number of landmarks the spline was fitted on.
    #[getter]
    fn n_landmarks(&self) -> usize {
        self.inner.n_landmarks()
    }

    /// The source landmarks, as an `(M, 3)` array.
    #[getter]
    fn source<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.inner.source().into_pyarray(py)
    }

    /// The non-affine weights `W`, as an `(M, 3)` array.
    #[getter]
    fn weights<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.inner.weights().into_pyarray(py)
    }

    /// The affine coefficients `A`, as a `(4, 3)` array.
    #[getter]
    fn affine_coefs<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.inner.affine_coefs().into_pyarray(py)
    }

    /// The affine part as a `(4, 4)` homogeneous matrix.
    #[getter]
    fn matrix_affine<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        mat4_to_py(py, self.inner.matrix_affine())
    }

    /// Transform an `(N, 3)` array of points. Returns an `(N, 3)` array.
    ///
    /// Unlike the reference implementation there is no `batch_size`: the distance matrix
    /// is never materialised, so peak memory is the output array regardless of how many
    /// points or landmarks are involved.
    #[pyo3(signature = (points, n_cores=None))]
    fn xform<'py>(
        &self,
        py: Python<'py>,
        points: PyReadonlyArray2<'py, f64>,
        n_cores: Option<usize>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let pts = points.as_array().to_owned();
        let tps = Arc::clone(&self.inner);
        let cancel = AtomicBool::new(false);

        let out: Array2<f64> = run_interruptible(py, &cancel, || {
            tps.xform(pts.view(), n_cores, Some(&cancel))
        })?
        .map_err(tps_err)?;

        Ok(out.into_pyarray(py))
    }
}

/// A moving-least-squares warp defined by landmark pairs.
#[pyclass(frozen, name = "MlsTransform", module = "navis_fastcore._fastcore")]
pub struct PyMlsTransform {
    inner: Arc<MlsTransform>,
}

#[pymethods]
impl PyMlsTransform {
    /// Build from `(M, 3)` source and target landmarks. Cheap — MLS has no fit step.
    #[new]
    fn new(
        source: PyReadonlyArray2<'_, f64>,
        target: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Self> {
        let src = source.as_array();
        let trg = target.as_array();
        let inner = MlsTransform::new(src.view(), trg.view()).map_err(mls_err)?;
        Ok(PyMlsTransform {
            inner: Arc::new(inner),
        })
    }

    /// Number of landmark pairs.
    #[getter]
    fn n_landmarks(&self) -> usize {
        self.inner.n_landmarks()
    }

    /// The source landmarks, as an `(M, 3)` array.
    #[getter]
    fn source<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.inner.source().into_pyarray(py)
    }

    /// The target landmarks, as an `(M, 3)` array.
    #[getter]
    fn target<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.inner.target().into_pyarray(py)
    }

    /// The *global* affine as a `(4, 4)` homogeneous matrix — the least-squares fit of
    /// source onto target landmarks, which is what the warp converges to far from them.
    #[pyo3(signature = (reverse=false))]
    fn matrix_affine<'py>(&self, py: Python<'py>, reverse: bool) -> Bound<'py, PyArray2<f64>> {
        mat4_to_py(py, self.inner.matrix_affine(reverse))
    }

    /// Transform an `(N, 3)` array of points. Returns an `(N, 3)` array.
    ///
    /// `reverse` fits the warp in the opposite direction (target space back to source
    /// space). As in the reference implementation this is not an exact inverse.
    #[pyo3(signature = (points, reverse=false, n_cores=None))]
    fn xform<'py>(
        &self,
        py: Python<'py>,
        points: PyReadonlyArray2<'py, f64>,
        reverse: bool,
        n_cores: Option<usize>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let pts = points.as_array().to_owned();
        let mls = Arc::clone(&self.inner);
        let cancel = AtomicBool::new(false);

        let out: Array2<f64> = run_interruptible(py, &cancel, || {
            mls.xform(pts.view(), reverse, n_cores, Some(&cancel))
        })?
        .map_err(mls_err)?;

        Ok(out.into_pyarray(py))
    }
}
