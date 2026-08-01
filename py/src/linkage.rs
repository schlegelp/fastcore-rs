use ndarray::{Array2, ArrayView2, ArrayViewMut2};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2, PyReadwriteArray1, PyReadwriteArray2,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::sync::atomic::AtomicBool;

use crate::nblast::run_interruptible;
use fastcore::linkage::{
    check_finite, condense, leaf_order, linkage, linkage_from_scores, observations_from_condensed,
    symmetrize, Dissim, LinkageError, Method, Symmetry, Transform,
};

/// The score matrix, at whatever width it already is.
///
/// As in `matches`, `PyReadonlyArray2` extracts only on an *exact* dtype match, so
/// this can never silently copy: a 40 GB (or `np.memmap`'d) matrix is borrowed, not
/// materialised. `float16` is deliberately absent — see [`Dissim`].
#[derive(FromPyObject)]
pub enum ScoresIn<'py> {
    F32(PyReadonlyArray2<'py, f32>),
    F64(PyReadonlyArray2<'py, f64>),
}

/// A square score matrix borrowed **mutably**, for the in-place symmetrise.
#[derive(FromPyObject)]
pub enum ScoresMut<'py> {
    F32(PyReadwriteArray2<'py, f32>),
    F64(PyReadwriteArray2<'py, f64>),
}

/// A caller-supplied condensed distance vector, borrowed **mutably**.
///
/// Linkage consumes its input as scratch. Taking it mutably here is what keeps the
/// pipeline copy-free; the Python wrapper is responsible for having copied first if
/// the caller asked it to.
#[derive(FromPyObject)]
pub enum CondensedIn<'py> {
    F32(PyReadwriteArray1<'py, f32>),
    F64(PyReadwriteArray1<'py, f64>),
}

fn to_py_err(e: LinkageError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

fn method_of(name: &str) -> PyResult<Method> {
    Method::from_name(name).ok_or_else(|| {
        PyValueError::new_err(format!(
            "unknown `method` {name:?}; expected one of single, complete, average, \
             weighted, ward, centroid, median"
        ))
    })
}

fn symmetry_of(name: &str) -> PyResult<Symmetry> {
    Symmetry::from_name(name).ok_or_else(|| {
        PyValueError::new_err(format!(
            "unknown `symmetry` {name:?}; expected one of none, mean, min, max"
        ))
    })
}

fn transform_of(name: &str) -> PyResult<Transform> {
    Transform::from_name(name).ok_or_else(|| {
        PyValueError::new_err(format!(
            "unknown `transform` {name:?}; expected one of one_minus, none"
        ))
    })
}

// ---------------------------------------------------------------------------
// condensed_distances
// ---------------------------------------------------------------------------

/// Fused symmetrise + similarity→distance + condense, in one pass.
///
/// @param scores Square (n, n) float32 or float64 score matrix.
/// @param symmetry How to combine `M[i,j]` with `M[j,i]`: "none", "mean", "min", "max".
/// @param transform "one_minus" for `1 - score`, or "none" if already distances.
/// @param n_cores Thread cap; `None` uses the global pool.
/// @return 1-D condensed distance vector of length n(n-1)/2, same dtype as `scores`.
#[pyfunction]
#[pyo3(
    name = "condensed_distances",
    signature = (scores, symmetry="mean", transform="one_minus", n_cores=None)
)]
pub fn condensed_distances_py<'py>(
    py: Python<'py>,
    scores: ScoresIn<'py>,
    symmetry: &str,
    transform: &str,
    n_cores: Option<usize>,
) -> PyResult<Bound<'py, PyAny>> {
    let sym = symmetry_of(symmetry)?;
    let tf = transform_of(transform)?;
    let cancel = AtomicBool::new(false);

    fn run<T: Dissim>(
        view: ArrayView2<T>,
        sym: Symmetry,
        tf: Transform,
        n_cores: Option<usize>,
        cancel: &AtomicBool,
    ) -> Result<Vec<T>, LinkageError> {
        condense(view, sym, tf, n_cores, Some(cancel))
    }

    match scores {
        ScoresIn::F32(a) => {
            let v = a.as_array();
            let out = run_interruptible(py, &cancel, || run(v, sym, tf, n_cores, &cancel))?;
            Ok(PyArray1::from_vec(py, out.map_err(to_py_err)?).into_any())
        }
        ScoresIn::F64(a) => {
            let v = a.as_array();
            let out = run_interruptible(py, &cancel, || run(v, sym, tf, n_cores, &cancel))?;
            Ok(PyArray1::from_vec(py, out.map_err(to_py_err)?).into_any())
        }
    }
}

// ---------------------------------------------------------------------------
// symmetrize
// ---------------------------------------------------------------------------

/// Make a square score matrix symmetric, in place and without allocating.
///
/// Replaces `(M + M.T) / 2`, which builds two full n x n temporaries — and even
/// `np.add(M, M.T, out=M)` still costs one, because numpy sees the output
/// overlapping `M.T` and defensively copies.
///
/// @param scores Square (n, n) float32 or float64 matrix, modified in place.
/// @param symmetry "mean", "min", "max", or "none" to mirror the upper triangle.
/// @param n_cores Thread cap; `None` uses the global pool.
#[pyfunction]
#[pyo3(name = "symmetrize", signature = (scores, symmetry="mean", n_cores=None))]
pub fn symmetrize_py(
    py: Python<'_>,
    scores: ScoresMut<'_>,
    symmetry: &str,
    n_cores: Option<usize>,
) -> PyResult<()> {
    let sym = symmetry_of(symmetry)?;
    let cancel = AtomicBool::new(false);

    fn run<T: Dissim>(
        view: ArrayViewMut2<T>,
        sym: Symmetry,
        n_cores: Option<usize>,
        cancel: &AtomicBool,
    ) -> Result<(), LinkageError> {
        symmetrize(view, sym, n_cores, Some(cancel))
    }

    match scores {
        ScoresMut::F32(mut a) => {
            let v = a.as_array_mut();
            run_interruptible(py, &cancel, || run(v, sym, n_cores, &cancel))?
        }
        ScoresMut::F64(mut a) => {
            let v = a.as_array_mut();
            run_interruptible(py, &cancel, || run(v, sym, n_cores, &cancel))?
        }
    }
    .map_err(to_py_err)
}

// ---------------------------------------------------------------------------
// linkage from a square score matrix
// ---------------------------------------------------------------------------

/// Cluster a square score matrix, fusing the whole pipeline.
///
/// The condensed matrix is built and consumed internally, so peak memory is the
/// caller's matrix plus one n(n-1)/2 buffer and nothing else.
///
/// @param scores Square (n, n) float32 or float64 score matrix.
/// @param method Linkage method (single, complete, average, weighted, ward, centroid, median).
/// @param symmetry How to combine `M[i,j]` with `M[j,i]`: "none", "mean", "min", "max".
/// @param transform "one_minus" for `1 - score`, or "none" if already distances.
/// @param n_cores Thread cap for the condensing pass; `None` uses the global pool.
/// @return SciPy-compatible (n-1, 4) float64 linkage matrix.
#[pyfunction]
#[pyo3(
    name = "linkage_from_scores",
    signature = (scores, method="ward", symmetry="mean", transform="one_minus", n_cores=None)
)]
pub fn linkage_from_scores_py<'py>(
    py: Python<'py>,
    scores: ScoresIn<'py>,
    method: &str,
    symmetry: &str,
    transform: &str,
    n_cores: Option<usize>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let m = method_of(method)?;
    let sym = symmetry_of(symmetry)?;
    let tf = transform_of(transform)?;
    let cancel = AtomicBool::new(false);

    fn run<T: Dissim>(
        view: ArrayView2<T>,
        m: Method,
        sym: Symmetry,
        tf: Transform,
        n_cores: Option<usize>,
        cancel: &AtomicBool,
    ) -> Result<Array2<f64>, LinkageError> {
        linkage_from_scores(view, m, sym, tf, n_cores, Some(cancel))
    }

    let z = match scores {
        ScoresIn::F32(a) => {
            let v = a.as_array();
            run_interruptible(py, &cancel, || run(v, m, sym, tf, n_cores, &cancel))?
        }
        ScoresIn::F64(a) => {
            let v = a.as_array();
            run_interruptible(py, &cancel, || run(v, m, sym, tf, n_cores, &cancel))?
        }
    };
    Ok(z.map_err(to_py_err)?.into_pyarray(py))
}

// ---------------------------------------------------------------------------
// linkage from a condensed vector
// ---------------------------------------------------------------------------

/// Cluster an existing condensed distance vector, **in place**.
///
/// `condensed` is used as scratch and is left in an arbitrary state on return.
///
/// @param condensed 1-D float32 or float64 vector of length n(n-1)/2.
/// @param method Linkage method (single, complete, average, weighted, ward, centroid, median).
/// @param n_cores Thread cap for the finiteness check; `None` uses the global pool.
/// @return SciPy-compatible (n-1, 4) float64 linkage matrix.
#[pyfunction]
#[pyo3(name = "linkage_condensed", signature = (condensed, method="ward", n_cores=None))]
pub fn linkage_condensed_py<'py>(
    py: Python<'py>,
    condensed: CondensedIn<'py>,
    method: &str,
    n_cores: Option<usize>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let m = method_of(method)?;
    let cancel = AtomicBool::new(false);

    fn run<T: Dissim>(
        buf: &mut [T],
        m: Method,
        n_cores: Option<usize>,
    ) -> Result<Array2<f64>, LinkageError> {
        let n = observations_from_condensed(buf.len())
            .ok_or(LinkageError::BadCondensedLen { len: buf.len() })?;
        check_finite(buf, n_cores)?;
        linkage(buf, n, m)
    }

    let z = match condensed {
        CondensedIn::F32(mut a) => {
            let buf = a
                .as_slice_mut()
                .map_err(|_| to_py_err(LinkageError::NotContiguous))?;
            run_interruptible(py, &cancel, || run(buf, m, n_cores))?
        }
        CondensedIn::F64(mut a) => {
            let buf = a
                .as_slice_mut()
                .map_err(|_| to_py_err(LinkageError::NotContiguous))?;
            run_interruptible(py, &cancel, || run(buf, m, n_cores))?
        }
    };
    Ok(z.map_err(to_py_err)?.into_pyarray(py))
}

// ---------------------------------------------------------------------------
// leaf_order
// ---------------------------------------------------------------------------

/// The order to place the leaves in so a dendrogram draws without crossings.
///
/// SciPy's `leaves_list`, for callers who have no scipy.
///
/// @param z SciPy-compatible (n-1, 4) float64 linkage matrix.
/// @return (n,) int64 array of observation indices, left to right.
#[pyfunction]
#[pyo3(name = "leaf_order")]
pub fn leaf_order_py<'py>(
    py: Python<'py>,
    z: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let view = z.as_array();
    if view.ncols() != 4 {
        return Err(PyValueError::new_err(format!(
            "`z` must be a (n - 1, 4) linkage matrix; got {} columns",
            view.ncols()
        )));
    }
    let n = view.nrows() + 1;

    // The walk descends into whatever cluster a row names, so an id naming a cluster
    // formed at or after that row would either index off the end of `z` or loop
    // forever. SciPy's labelling rules that out; check it rather than trust it, since
    // the matrix reaching us need not be one we produced.
    for (step, row) in view.rows().into_iter().enumerate() {
        for &c in [row[0], row[1]].iter() {
            if !(c >= 0.0 && c < (n + step) as f64) {
                return Err(PyValueError::new_err(format!(
                    "row {step} of `z` merges cluster {c}, which is neither an \
                     observation (0..{n}) nor a cluster formed before it"
                )));
            }
        }
    }

    // `to_owned` copies (n-1) x 4 f64 - 3 MB at 100k observations, against the tens of
    // seconds the linkage that produced it took. Not worth widening the core signature
    // to a view, which the R bindings also call.
    let order: Vec<i64> = leaf_order(&view.to_owned(), n)
        .into_iter()
        .map(|i| i as i64)
        .collect();

    Ok(order.into_pyarray(py))
}
