use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

use fastcore::points::dotprops;

/// Tangent vectors and alpha values for a point cloud.
///
/// For each point, takes its `k` nearest neighbours (the point itself included, matching
/// `scipy.spatial.cKDTree.query`), forms the scatter matrix of that neighbourhood about its
/// centroid, and returns the principal direction plus how elongated the neighbourhood is.
///
/// Arguments
/// ---------
/// - `points`:  (N, 3) float64 point coordinates.
/// - `k`:       Number of nearest neighbours *including the point itself*. Clamped to `N`.
/// - `threads`: Size of the thread pool, or `None` for all cores.
///
/// Returns
/// -------
/// `(vect, alpha)`: an (N, 3) float64 array of unit tangent vectors and an (N, ) float64 array
/// of `(l1 - l2) / (l1 + l2 + l3)` values. The sign of each vector is arbitrary but
/// deterministic; degenerate neighbourhoods give `alpha = 0` and `[1, 0, 0]` rather than NaN.
#[pyfunction]
#[pyo3(name = "dotprops", signature = (points, k=20, threads=None))]
pub fn dotprops_py<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<f64>,
    k: usize,
    threads: Option<usize>,
) -> (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>) {
    let (vect, alpha) = dotprops(points.as_array(), k, threads);
    (vect.into_pyarray(py), alpha.into_pyarray(py))
}
