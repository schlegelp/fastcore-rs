use half::f16;
use ndarray::{Array2, ArrayView2};
use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::sync::atomic::AtomicBool;

use crate::nblast::run_interruptible;
use fastcore::matches::{
    count_matches, matches_above, top_matches, Criterion, MatchAxis, MatchError, MatchOpts, Score,
};

/// The score matrix, at whatever width it already is.
///
/// `PyReadonlyArray2::<f32>` extracts only on an *exact* dtype match — it downcasts, it does
/// not convert — so this can never silently copy. That is the whole point: the wrapper does
/// no coercion either, `PyReadonlyArray2` borrows, and a 40 GB (or `np.memmap`'d) matrix
/// therefore reaches the kernel without being materialised anywhere.
#[derive(FromPyObject)]
pub enum ScoresIn<'py> {
    F32(PyReadonlyArray2<'py, f32>),
    F64(PyReadonlyArray2<'py, f64>),
    F16(PyReadonlyArray2<'py, f16>),
}

fn to_py_err(e: MatchError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

fn axis_of(axis: u8) -> PyResult<MatchAxis> {
    match axis {
        0 => Ok(MatchAxis::Rows),
        1 => Ok(MatchAxis::Cols),
        _ => Err(PyValueError::new_err(format!(
            "`axis` must be 0 or 1, got {axis}"
        ))),
    }
}

fn criterion_of(threshold: Option<f64>, percentage: Option<f64>) -> PyResult<Criterion> {
    match (threshold, percentage) {
        (Some(t), None) => Ok(Criterion::Threshold(t)),
        (None, Some(p)) => Ok(Criterion::Percentage(p)),
        _ => Err(PyValueError::new_err(
            "provide exactly one of `threshold` or `percentage`",
        )),
    }
}

/// Normalise the memory layout so the core only ever sees a C-contiguous matrix.
///
/// An F-contiguous `(R, C)` array is bit-for-bit a C-contiguous `(C, R)` array viewed
/// transposed, so transposing the view and flipping the axis alongside it puts the request
/// back in the caller's frame — group ids and match indices come out unchanged. That makes
/// `scores.T` free rather than a 40 GB copy, and means the "walk the columns" case never
/// needs a strided path at all.
///
/// Anything else (a genuinely strided slice) is refused rather than copied.
fn normalize<'a, T>(
    view: ArrayView2<'a, T>,
    axis: MatchAxis,
) -> PyResult<(ArrayView2<'a, T>, MatchAxis)> {
    if view.is_standard_layout() {
        Ok((view, axis))
    } else if view.t().is_standard_layout() {
        Ok((view.reversed_axes(), axis.flip()))
    } else {
        Err(to_py_err(MatchError::NotContiguous))
    }
}

fn opts<'a>(
    axis: MatchAxis,
    distances: bool,
    skip: Option<&'a [i64]>,
    max_matches: Option<u64>,
    n_cores: Option<usize>,
    progress: bool,
    cancel: &'a AtomicBool,
) -> MatchOpts<'a> {
    MatchOpts {
        axis,
        distances,
        skip,
        max_matches,
        threads: n_cores,
        progress,
        cancel: Some(cancel),
    }
}

/// `skip` arrives as a C-contiguous int64 array (or not at all).
fn skip_slice<'a>(skip: &'a Option<PyReadonlyArray1<'a, i64>>) -> PyResult<Option<&'a [i64]>> {
    match skip {
        None => Ok(None),
        Some(a) => a
            .as_slice()
            .map(Some)
            .map_err(|_| PyValueError::new_err("`skip` must be a C-contiguous int64 array")),
    }
}

/// Run `top_matches` for one concrete element width and hand back `(indices, values)`.
#[allow(clippy::too_many_arguments)]
fn topn_for<'py, T: Score + numpy::Element>(
    py: Python<'py>,
    arr: PyReadonlyArray2<'py, T>,
    n: usize,
    axis: MatchAxis,
    distances: bool,
    skip: Option<&[i64]>,
    n_cores: Option<usize>,
    progress: bool,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    let (view, axis) = normalize(arr.as_array(), axis)?;
    let cancel = AtomicBool::new(false);
    let o = opts(axis, distances, skip, None, n_cores, progress, &cancel);

    let out = run_interruptible(py, &cancel, move || top_matches(view, n, o))?.map_err(to_py_err)?;

    let shape = (out.n_groups, out.n);
    let idx = Array2::from_shape_vec(shape, out.indices)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let val = Array2::from_shape_vec(shape, out.values)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((idx.into_pyarray(py).into_any(), val.into_pyarray(py).into_any()))
}

/// Run `matches_above` for one concrete element width -> `(offsets, indices, values)`.
#[allow(clippy::too_many_arguments)]
fn ragged_for<'py, T: Score + numpy::Element>(
    py: Python<'py>,
    arr: PyReadonlyArray2<'py, T>,
    crit: Criterion,
    axis: MatchAxis,
    distances: bool,
    skip: Option<&[i64]>,
    max_matches: Option<u64>,
    n_cores: Option<usize>,
    progress: bool,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    let (view, axis) = normalize(arr.as_array(), axis)?;
    let cancel = AtomicBool::new(false);
    let o = opts(axis, distances, skip, max_matches, n_cores, progress, &cancel);

    let out =
        run_interruptible(py, &cancel, move || matches_above(view, crit, o))?.map_err(to_py_err)?;

    Ok((
        out.offsets.into_pyarray(py).into_any(),
        out.indices.into_pyarray(py).into_any(),
        out.values.into_pyarray(py).into_any(),
    ))
}

fn counts_for<'py, T: Score + numpy::Element>(
    py: Python<'py>,
    arr: PyReadonlyArray2<'py, T>,
    crit: Criterion,
    axis: MatchAxis,
    distances: bool,
    skip: Option<&[i64]>,
    n_cores: Option<usize>,
) -> PyResult<Bound<'py, PyAny>> {
    let (view, axis) = normalize(arr.as_array(), axis)?;
    let cancel = AtomicBool::new(false);
    let o = opts(axis, distances, skip, None, n_cores, false, &cancel);

    let out =
        run_interruptible(py, &cancel, move || count_matches(view, crit, o))?.map_err(to_py_err)?;
    Ok(out.into_pyarray(py).into_any())
}

/// Dispatch on the input width. Every arm is the same call at a different `T`.
macro_rules! by_dtype {
    ($scores:expr, |$arr:ident| $body:expr) => {
        match $scores {
            ScoresIn::F32($arr) => $body,
            ScoresIn::F64($arr) => $body,
            ScoresIn::F16($arr) => $body,
        }
    };
}

/// The `n` best matches per group. Returns `(indices, values)`, both `(n_groups, n)` and
/// ordered best-first; `indices` is -1 (and `values` NaN) where a group held fewer than `n`
/// valid cells.
#[pyfunction]
#[pyo3(
    signature = (scores, n, axis=0, distances=false, skip=None, n_cores=None, progress=false),
    name = "top_matches"
)]
#[allow(clippy::too_many_arguments)]
pub fn top_matches_py<'py>(
    py: Python<'py>,
    scores: ScoresIn<'py>,
    n: usize,
    axis: u8,
    distances: bool,
    skip: Option<PyReadonlyArray1<'py, i64>>,
    n_cores: Option<usize>,
    progress: bool,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    let axis = axis_of(axis)?;
    let sk = skip_slice(&skip)?;
    by_dtype!(scores, |a| topn_for(
        py, a, n, axis, distances, sk, n_cores, progress
    ))
}

/// Every match clearing the cutoff, CSR-style: `(offsets, indices, values)`, where group `g`
/// occupies `indices[offsets[g]:offsets[g + 1]]`, best first.
#[pyfunction]
#[pyo3(
    signature = (
        scores, threshold=None, percentage=None, axis=0, distances=false, skip=None,
        max_matches=None, n_cores=None, progress=false
    ),
    name = "matches_above"
)]
#[allow(clippy::too_many_arguments)]
pub fn matches_above_py<'py>(
    py: Python<'py>,
    scores: ScoresIn<'py>,
    threshold: Option<f64>,
    percentage: Option<f64>,
    axis: u8,
    distances: bool,
    skip: Option<PyReadonlyArray1<'py, i64>>,
    max_matches: Option<u64>,
    n_cores: Option<usize>,
    progress: bool,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    let axis = axis_of(axis)?;
    let crit = criterion_of(threshold, percentage)?;
    let sk = skip_slice(&skip)?;
    by_dtype!(scores, |a| ragged_for(
        py, a, crit, axis, distances, sk, max_matches, n_cores, progress
    ))
}

/// How many matches each group *would* yield, without materialising them.
#[pyfunction]
#[pyo3(
    signature = (scores, threshold=None, percentage=None, axis=0, distances=false, skip=None, n_cores=None),
    name = "count_matches"
)]
#[allow(clippy::too_many_arguments)]
pub fn count_matches_py<'py>(
    py: Python<'py>,
    scores: ScoresIn<'py>,
    threshold: Option<f64>,
    percentage: Option<f64>,
    axis: u8,
    distances: bool,
    skip: Option<PyReadonlyArray1<'py, i64>>,
    n_cores: Option<usize>,
) -> PyResult<Bound<'py, PyAny>> {
    let axis = axis_of(axis)?;
    let crit = criterion_of(threshold, percentage)?;
    let sk = skip_slice(&skip)?;
    by_dtype!(scores, |a| counts_for(
        py, a, crit, axis, distances, sk, n_cores
    ))
}
