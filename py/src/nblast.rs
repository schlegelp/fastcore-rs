use ndarray::Array2;
use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::prelude::{PyResult, Python};

use fastcore::nblast::{load_smat, load_smat_alpha, nblast_allbyall, nblast_query_target, Opts, Smat};

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
    let opts = Opts {
        smat: &smat,
        normalize,
        limit_dist,
        threads: n_cores,
        progress,
    };

    let out = match precision {
        32 => {
            let scores: Vec<f32> =
                py.detach(move || nblast_allbyall(clouds, vecs, alpha_vecs, opts));
            Array2::from_shape_vec((n, n), scores)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
                .into_pyarray(py)
                .into_any()
        }
        64 => {
            let scores: Vec<f64> =
                py.detach(move || nblast_allbyall(clouds, vecs, alpha_vecs, opts));
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
    let opts = Opts {
        smat: &smat,
        normalize,
        limit_dist,
        threads: n_cores,
        progress,
    };

    let out = match precision {
        32 => {
            let scores: Vec<f32> =
                py.detach(move || nblast_query_target(qp, qv, qa, tp, tv, ta, opts));
            Array2::from_shape_vec((nq, nt), scores)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
                .into_pyarray(py)
                .into_any()
        }
        64 => {
            let scores: Vec<f64> =
                py.detach(move || nblast_query_target(qp, qv, qa, tp, tv, ta, opts));
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
