use ndarray::Array2;
use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::prelude::{PyResult, Python};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, RecvTimeoutError};
use std::time::Duration;

use fastcore::nblast::{
    load_smat, load_smat_alpha, nblast_allbyall, nblast_pairs, nblast_query_target, Opts, Smat,
};
use fastcore::nblast_knn::{nblast_knn, nblast_knn_query_target, KnnOpts, Symmetry};
use fastcore::synblast::{synblast_allbyall, synblast_query_target};

/// Run a GIL-releasing NBLAST `compute` so a `KeyboardInterrupt` (Ctrl-C, or the
/// Jupyter interrupt button) can stop it.
///
/// The heavy work runs on a spawned worker thread while *this* — the calling
/// thread that entered the `#[pyfunction]`, i.e. Python's main thread — polls the
/// interpreter's signal state. `check_signals` only observes signals from the main
/// thread, so the compute has to run off it. On an interrupt we flip `cancel` (which
/// the NBLAST loops poll and short-circuit) and re-raise the caught exception; the
/// worker's partially-filled, discarded result never reaches Python.
pub(crate) fn run_interruptible<T, F>(py: Python<'_>, cancel: &AtomicBool, compute: F) -> PyResult<T>
where
    T: Send,
    F: FnOnce() -> T + Send,
{
    py.detach(|| {
        let (tx, rx) = channel::<()>();
        let mut pending: Option<PyErr> = None;
        let result = std::thread::scope(|s| {
            let handle = s.spawn(move || {
                let out = compute();
                let _ = tx.send(()); // wake the poller the instant we finish
                out
            });
            // Poll ~every 50ms until the worker reports done (or drops the sender).
            while let Err(RecvTimeoutError::Timeout) = rx.recv_timeout(Duration::from_millis(50)) {
                if let Err(err) = Python::attach(|py| py.check_signals()) {
                    if pending.is_none() {
                        pending = Some(err);
                    }
                    cancel.store(true, Ordering::Relaxed);
                }
            }
            handle.join().expect("NBLAST worker thread panicked")
        });
        match pending {
            Some(err) => Err(err),
            None => Ok(result),
        }
    })
}

// Convert a list of (N, 3) numpy arrays into owned point clouds. Done under the
// GIL, before any GIL-releasing call, so no numpy borrow is held across threads.
fn to_clouds(arrays: &[PyReadonlyArray2<f64>]) -> Vec<Vec<[f64; 3]>> {
    arrays
        .iter()
        .map(|a| {
            a.as_array()
                .rows()
                .into_iter()
                .map(|r| [r[0], r[1], r[2]])
                .collect()
        })
        .collect()
}

// Convert an optional list of per-point alpha arrays into owned Vecs. `None`
// (use_alpha off) stays `None`; otherwise one `Vec<f64>` per neuron.
fn to_alphas(arrays: Option<Vec<PyReadonlyArray1<f64>>>) -> Option<Vec<Vec<f64>>> {
    arrays.map(|v| v.iter().map(|a| a.as_array().to_vec()).collect())
}

// Convert a list of per-connector integer type arrays into owned Vecs (one per
// neuron). Used by syNBLAST to group connectors by type. Done under the GIL.
fn to_types(arrays: &[PyReadonlyArray1<i64>]) -> Vec<Vec<i64>> {
    arrays.iter().map(|a| a.as_array().to_vec()).collect()
}

// Build a scoring matrix from Python-supplied parts, or fall back to an embedded
// FCWB matrix when they are not provided. Without an explicit matrix, `use_alpha`
// selects the alpha-calibrated default (matching navis' `smat="auto"`).
fn build_smat(
    values: Option<PyReadonlyArray2<f64>>,
    dist_edges: Option<PyReadonlyArray1<f64>>,
    dot_edges: Option<PyReadonlyArray1<f64>>,
    use_alpha: bool,
) -> Smat {
    match (values, dist_edges, dot_edges) {
        (Some(v), Some(de), Some(ve)) => {
            let arr = v.as_array();
            let (nrows, ncols) = (arr.nrows(), arr.ncols());
            // `.iter()` yields row-major logical order regardless of memory layout.
            let flat: Vec<f64> = arr.iter().copied().collect();
            Smat::from_parts(flat, nrows, ncols, de.as_array().to_vec(), ve.as_array().to_vec())
        }
        _ if use_alpha => load_smat_alpha(),
        _ => load_smat(),
    }
}

/// The `limit_dist="auto"` value for a scoring matrix (embedded FCWB by default,
/// or the supplied one): `1.05 *` the left edge of the last distance bin.
#[pyfunction]
#[pyo3(
    signature = (smat_values=None, dist_edges=None, dot_edges=None, use_alpha=false),
    name = "smat_auto_limit"
)]
pub fn smat_auto_limit_py(
    smat_values: Option<PyReadonlyArray2<f64>>,
    dist_edges: Option<PyReadonlyArray1<f64>>,
    dot_edges: Option<PyReadonlyArray1<f64>>,
    use_alpha: bool,
) -> PyResult<f64> {
    Ok(build_smat(smat_values, dist_edges, dot_edges, use_alpha).auto_limit())
}

/// All-by-all forward NBLAST. Returns an (n, n) matrix (float32 or float64 per
/// `precision`); cell [i, j] is query i against target j (diagonal 1.0 when
/// normalized).
#[pyfunction]
#[pyo3(
    signature = (
        points, vects, alphas=None, smat_values=None, dist_edges=None, dot_edges=None,
        normalize=true, limit_dist=None, n_cores=None, precision=32, progress=false
    ),
    name = "nblast_allbyall"
)]
#[allow(clippy::too_many_arguments)]
pub fn nblast_allbyall_py<'py>(
    py: Python<'py>,
    points: Vec<PyReadonlyArray2<f64>>,
    vects: Vec<PyReadonlyArray2<f64>>,
    alphas: Option<Vec<PyReadonlyArray1<f64>>>,
    smat_values: Option<PyReadonlyArray2<f64>>,
    dist_edges: Option<PyReadonlyArray1<f64>>,
    dot_edges: Option<PyReadonlyArray1<f64>>,
    normalize: bool,
    limit_dist: Option<f64>,
    n_cores: Option<usize>,
    precision: u8,
    progress: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let clouds = to_clouds(&points);
    let vecs = to_clouds(&vects);
    let alpha_vecs = to_alphas(alphas);
    let smat = build_smat(smat_values, dist_edges, dot_edges, alpha_vecs.is_some());
    let n = clouds.len();
    let cancel = AtomicBool::new(false);
    let opts = Opts {
        smat: &smat,
        normalize,
        limit_dist,
        threads: n_cores,
        progress,
        cancel: Some(&cancel),
    };

    let out = match precision {
        32 => {
            let scores: Vec<f32> =
                run_interruptible(py, &cancel, move || nblast_allbyall(clouds, vecs, alpha_vecs, opts))?;
            Array2::from_shape_vec((n, n), scores)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
                .into_pyarray(py)
                .into_any()
        }
        64 => {
            let scores: Vec<f64> =
                run_interruptible(py, &cancel, move || nblast_allbyall(clouds, vecs, alpha_vecs, opts))?;
            Array2::from_shape_vec((n, n), scores)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
                .into_pyarray(py)
                .into_any()
        }
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "precision must be 32 or 64",
            ))
        }
    };
    Ok(out)
}

/// Forward NBLAST for a set of `(query, target)` index pairs. `q_idx[k]` / `t_idx[k]`
/// select the k-th pair; returns a 1-D array (float32 or float64 per `precision`) of
/// length `len(q_idx)`, aligned to the pairs. Used by the two-pass "smart" NBLAST for
/// its full-resolution pass over a sparse candidate set.
#[pyfunction]
#[pyo3(
    signature = (
        q_points, q_vects, t_points, t_vects, q_idx, t_idx, q_alphas=None, t_alphas=None,
        smat_values=None, dist_edges=None, dot_edges=None,
        normalize=true, limit_dist=None, n_cores=None, precision=32, progress=false
    ),
    name = "nblast_pairs"
)]
#[allow(clippy::too_many_arguments)]
pub fn nblast_pairs_py<'py>(
    py: Python<'py>,
    q_points: Vec<PyReadonlyArray2<f64>>,
    q_vects: Vec<PyReadonlyArray2<f64>>,
    t_points: Vec<PyReadonlyArray2<f64>>,
    t_vects: Vec<PyReadonlyArray2<f64>>,
    q_idx: PyReadonlyArray1<i64>,
    t_idx: PyReadonlyArray1<i64>,
    q_alphas: Option<Vec<PyReadonlyArray1<f64>>>,
    t_alphas: Option<Vec<PyReadonlyArray1<f64>>>,
    smat_values: Option<PyReadonlyArray2<f64>>,
    dist_edges: Option<PyReadonlyArray1<f64>>,
    dot_edges: Option<PyReadonlyArray1<f64>>,
    normalize: bool,
    limit_dist: Option<f64>,
    n_cores: Option<usize>,
    precision: u8,
    progress: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let qp = to_clouds(&q_points);
    let qv = to_clouds(&q_vects);
    let tp = to_clouds(&t_points);
    let tv = to_clouds(&t_vects);
    let qa = to_alphas(q_alphas);
    let ta = to_alphas(t_alphas);
    let smat = build_smat(smat_values, dist_edges, dot_edges, qa.is_some());

    // Zip the parallel index arrays into (query, target) pairs under the GIL.
    let qi = q_idx.as_array();
    let ti = t_idx.as_array();
    if qi.len() != ti.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "q_idx and t_idx must have the same length",
        ));
    }
    let pairs: Vec<(usize, usize)> = qi
        .iter()
        .zip(ti.iter())
        .map(|(&a, &b)| (a as usize, b as usize))
        .collect();

    let cancel = AtomicBool::new(false);
    let opts = Opts {
        smat: &smat,
        normalize,
        limit_dist,
        threads: n_cores,
        progress,
        cancel: Some(&cancel),
    };

    let out = match precision {
        32 => {
            let scores: Vec<f32> =
                run_interruptible(py, &cancel, move || nblast_pairs(qp, qv, qa, tp, tv, ta, pairs, opts))?;
            scores.into_pyarray(py).into_any()
        }
        64 => {
            let scores: Vec<f64> =
                run_interruptible(py, &cancel, move || nblast_pairs(qp, qv, qa, tp, tv, ta, pairs, opts))?;
            scores.into_pyarray(py).into_any()
        }
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "precision must be 32 or 64",
            ))
        }
    };
    Ok(out)
}

/// All-by-all forward syNBLAST over synapse clouds. `points[i]` are neuron `i`'s
/// connector coordinates and `types[i]` their per-connector integer type ids (same
/// length). Returns an (n, n) matrix (float32 or float64 per `precision`); cell
/// [i, j] is query i against target j (diagonal 1.0 when normalized).
#[pyfunction]
#[pyo3(
    signature = (
        points, types, smat_values=None, dist_edges=None, dot_edges=None,
        normalize=true, n_cores=None, precision=32, progress=false
    ),
    name = "synblast_allbyall"
)]
#[allow(clippy::too_many_arguments)]
pub fn synblast_allbyall_py<'py>(
    py: Python<'py>,
    points: Vec<PyReadonlyArray2<f64>>,
    types: Vec<PyReadonlyArray1<i64>>,
    smat_values: Option<PyReadonlyArray2<f64>>,
    dist_edges: Option<PyReadonlyArray1<f64>>,
    dot_edges: Option<PyReadonlyArray1<f64>>,
    normalize: bool,
    n_cores: Option<usize>,
    precision: u8,
    progress: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let clouds = to_clouds(&points);
    let tys = to_types(&types);
    // syNBLAST never uses alpha; the plain FCWB matrix is navis' `smat="auto"`.
    let smat = build_smat(smat_values, dist_edges, dot_edges, false);
    let n = clouds.len();
    let cancel = AtomicBool::new(false);
    let opts = Opts {
        smat: &smat,
        normalize,
        limit_dist: None,
        threads: n_cores,
        progress,
        cancel: Some(&cancel),
    };

    let out = match precision {
        32 => {
            let scores: Vec<f32> =
                run_interruptible(py, &cancel, move || synblast_allbyall(clouds, tys, opts))?;
            Array2::from_shape_vec((n, n), scores)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
                .into_pyarray(py)
                .into_any()
        }
        64 => {
            let scores: Vec<f64> =
                run_interruptible(py, &cancel, move || synblast_allbyall(clouds, tys, opts))?;
            Array2::from_shape_vec((n, n), scores)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
                .into_pyarray(py)
                .into_any()
        }
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "precision must be 32 or 64",
            ))
        }
    };
    Ok(out)
}

/// Forward syNBLAST of every query neuron against every target neuron. Returns an
/// (n_query, n_target) matrix (float32 or float64 per `precision`); cell [qi, tj]
/// is query qi against target tj.
#[pyfunction]
#[pyo3(
    signature = (
        q_points, q_types, t_points, t_types, smat_values=None, dist_edges=None, dot_edges=None,
        normalize=true, n_cores=None, precision=32, progress=false
    ),
    name = "synblast"
)]
#[allow(clippy::too_many_arguments)]
pub fn synblast_py<'py>(
    py: Python<'py>,
    q_points: Vec<PyReadonlyArray2<f64>>,
    q_types: Vec<PyReadonlyArray1<i64>>,
    t_points: Vec<PyReadonlyArray2<f64>>,
    t_types: Vec<PyReadonlyArray1<i64>>,
    smat_values: Option<PyReadonlyArray2<f64>>,
    dist_edges: Option<PyReadonlyArray1<f64>>,
    dot_edges: Option<PyReadonlyArray1<f64>>,
    normalize: bool,
    n_cores: Option<usize>,
    precision: u8,
    progress: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let qp = to_clouds(&q_points);
    let qt = to_types(&q_types);
    let tp = to_clouds(&t_points);
    let tt = to_types(&t_types);
    let smat = build_smat(smat_values, dist_edges, dot_edges, false);
    let (nq, nt) = (qp.len(), tp.len());
    let cancel = AtomicBool::new(false);
    let opts = Opts {
        smat: &smat,
        normalize,
        limit_dist: None,
        threads: n_cores,
        progress,
        cancel: Some(&cancel),
    };

    let out = match precision {
        32 => {
            let scores: Vec<f32> =
                run_interruptible(py, &cancel, move || synblast_query_target(qp, qt, tp, tt, opts))?;
            Array2::from_shape_vec((nq, nt), scores)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
                .into_pyarray(py)
                .into_any()
        }
        64 => {
            let scores: Vec<f64> =
                run_interruptible(py, &cancel, move || synblast_query_target(qp, qt, tp, tt, opts))?;
            Array2::from_shape_vec((nq, nt), scores)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
                .into_pyarray(py)
                .into_any()
        }
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "precision must be 32 or 64",
            ))
        }
    };
    Ok(out)
}

/// Forward NBLAST of every query neuron against every target neuron. Returns an
/// (n_query, n_target) matrix (float32 or float64 per `precision`); cell [qi, tj]
/// is query qi against target tj.
#[pyfunction]
#[pyo3(
    signature = (
        q_points, q_vects, t_points, t_vects, q_alphas=None, t_alphas=None,
        smat_values=None, dist_edges=None, dot_edges=None,
        normalize=true, limit_dist=None, n_cores=None, precision=32, progress=false
    ),
    name = "nblast"
)]
#[allow(clippy::too_many_arguments)]
pub fn nblast_py<'py>(
    py: Python<'py>,
    q_points: Vec<PyReadonlyArray2<f64>>,
    q_vects: Vec<PyReadonlyArray2<f64>>,
    t_points: Vec<PyReadonlyArray2<f64>>,
    t_vects: Vec<PyReadonlyArray2<f64>>,
    q_alphas: Option<Vec<PyReadonlyArray1<f64>>>,
    t_alphas: Option<Vec<PyReadonlyArray1<f64>>>,
    smat_values: Option<PyReadonlyArray2<f64>>,
    dist_edges: Option<PyReadonlyArray1<f64>>,
    dot_edges: Option<PyReadonlyArray1<f64>>,
    normalize: bool,
    limit_dist: Option<f64>,
    n_cores: Option<usize>,
    precision: u8,
    progress: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let qp = to_clouds(&q_points);
    let qv = to_clouds(&q_vects);
    let tp = to_clouds(&t_points);
    let tv = to_clouds(&t_vects);
    let qa = to_alphas(q_alphas);
    let ta = to_alphas(t_alphas);
    let smat = build_smat(smat_values, dist_edges, dot_edges, qa.is_some());
    let (nq, nt) = (qp.len(), tp.len());
    let cancel = AtomicBool::new(false);
    let opts = Opts {
        smat: &smat,
        normalize,
        limit_dist,
        threads: n_cores,
        progress,
        cancel: Some(&cancel),
    };

    let out = match precision {
        32 => {
            let scores: Vec<f32> =
                run_interruptible(py, &cancel, move || nblast_query_target(qp, qv, qa, tp, tv, ta, opts))?;
            Array2::from_shape_vec((nq, nt), scores)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
                .into_pyarray(py)
                .into_any()
        }
        64 => {
            let scores: Vec<f64> =
                run_interruptible(py, &cancel, move || nblast_query_target(qp, qv, qa, tp, tv, ta, opts))?;
            Array2::from_shape_vec((nq, nt), scores)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
                .into_pyarray(py)
                .into_any()
        }
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "precision must be 32 or 64",
            ))
        }
    };
    Ok(out)
}

/// Map the Python `symmetry` spelling onto the core's [`Symmetry`].
fn parse_symmetry(name: &str) -> PyResult<Symmetry> {
    match name {
        "forward" => Ok(Symmetry::Forward),
        "mean" => Ok(Symmetry::Mean),
        "min" => Ok(Symmetry::Min),
        "max" => Ok(Symmetry::Max),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown symmetry {other:?}; expected 'forward', 'mean', 'min' or 'max'"
        ))),
    }
}

/// k nearest neighbours under NBLAST, without building the score matrix.
///
/// With `t_points` / `t_vects` supplied this is the rectangular query -> target
/// form and `idx` indexes the *target* list; without them it is the all-by-all
/// form over `points` and self-matches are excluded. Returns `(idx, scores)`,
/// both `(n_query, k)`, rows in descending score order and padded with `-1` /
/// `-inf` when fewer than `k` candidates exist. Scores are exact NBLAST values
/// combined per `symmetry`; only which neurons make the shortlist is approximate.
#[pyfunction]
#[pyo3(
    signature = (
        points, vects, alphas=None, t_points=None, t_vects=None, t_alphas=None,
        k=20, n_candidates=200, symmetry="mean",
        voxel=20.0, n_dirs=3, splat=true,
        smat_values=None, dist_edges=None, dot_edges=None,
        normalize=true, limit_dist=None, n_cores=None, precision=32, progress=false
    ),
    name = "nblast_knn"
)]
#[allow(clippy::too_many_arguments)]
pub fn nblast_knn_py<'py>(
    py: Python<'py>,
    points: Vec<PyReadonlyArray2<f64>>,
    vects: Vec<PyReadonlyArray2<f64>>,
    alphas: Option<Vec<PyReadonlyArray1<f64>>>,
    t_points: Option<Vec<PyReadonlyArray2<f64>>>,
    t_vects: Option<Vec<PyReadonlyArray2<f64>>>,
    t_alphas: Option<Vec<PyReadonlyArray1<f64>>>,
    k: usize,
    n_candidates: usize,
    symmetry: &str,
    voxel: f64,
    n_dirs: usize,
    splat: bool,
    smat_values: Option<PyReadonlyArray2<f64>>,
    dist_edges: Option<PyReadonlyArray1<f64>>,
    dot_edges: Option<PyReadonlyArray1<f64>>,
    normalize: bool,
    limit_dist: Option<f64>,
    n_cores: Option<usize>,
    precision: u8,
    progress: bool,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    if k < 1 {
        return Err(pyo3::exceptions::PyValueError::new_err("k must be >= 1"));
    }
    if voxel <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "voxel must be positive",
        ));
    }
    if points.len() != vects.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "points and vects must have the same length",
        ));
    }
    if t_points.is_some() != t_vects.is_some() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "t_points and t_vects must be given together",
        ));
    }
    if let (Some(tp), Some(tv)) = (&t_points, &t_vects) {
        if tp.len() != tv.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "t_points and t_vects must have the same length",
            ));
        }
    }
    let sym = parse_symmetry(symmetry)?;
    let clouds = to_clouds(&points);
    let vecs = to_clouds(&vects);
    let alpha_vecs = to_alphas(alphas);
    let targets = t_points
        .as_ref()
        .map(|tp| (to_clouds(tp), to_clouds(t_vects.as_ref().unwrap()), to_alphas(t_alphas)));
    let smat = build_smat(smat_values, dist_edges, dot_edges, alpha_vecs.is_some());
    let nq = clouds.len();
    let cancel = AtomicBool::new(false);
    let opts = KnnOpts {
        nblast: Opts {
            smat: &smat,
            normalize,
            limit_dist,
            threads: n_cores,
            progress,
            cancel: Some(&cancel),
        },
        k,
        n_candidates,
        voxel,
        n_dirs,
        splat,
        symmetry: sym,
    };

    // One closure per precision, dispatching square vs rectangular inside, so the
    // GIL-releasing call happens exactly once either way.
    macro_rules! run {
        ($ty:ty) => {{
            let (i, s): (Vec<i64>, Vec<$ty>) = run_interruptible(py, &cancel, move || {
                match targets {
                    Some((tp, tv, ta)) => {
                        nblast_knn_query_target(clouds, vecs, alpha_vecs, tp, tv, ta, opts)
                    }
                    None => nblast_knn(clouds, vecs, alpha_vecs, opts),
                }
            })?;
            (i, s)
        }};
    }

    let shape_err = |e: ndarray::ShapeError| pyo3::exceptions::PyValueError::new_err(e.to_string());
    let (idx, scores) = match precision {
        32 => {
            let (i, s) = run!(f32);
            (
                Array2::from_shape_vec((nq, k), i).map_err(shape_err)?,
                Array2::from_shape_vec((nq, k), s)
                    .map_err(shape_err)?
                    .into_pyarray(py)
                    .into_any(),
            )
        }
        64 => {
            let (i, s) = run!(f64);
            (
                Array2::from_shape_vec((nq, k), i).map_err(shape_err)?,
                Array2::from_shape_vec((nq, k), s)
                    .map_err(shape_err)?
                    .into_pyarray(py)
                    .into_any(),
            )
        }
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "precision must be 32 or 64",
            ))
        }
    };
    Ok((idx.into_pyarray(py).into_any(), scores))
}
