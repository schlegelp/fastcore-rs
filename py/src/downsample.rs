use ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use fastcore::downsample::{
    downsample_skeleton, resample_skeleton, simplify_rdp, simplify_vw, smooth_skeleton,
    smooth_skeleton_gaussian,
};

/// What all three node-dropping methods hand back.
///
/// Spelled out once because the three signatures are otherwise identical noise, and
/// because the shape matches `simplify_skeleton`: a caller can swap one for another
/// without touching the unpacking.
type DroppedOut<'py> = (
    Bound<'py, PyArray1<i32>>,
    Bound<'py, PyArray1<i32>>,
    Option<Bound<'py, PyArray1<f32>>>,
    Bound<'py, PyArray1<i32>>,
);

/// What `resample_skeleton` hands back: parents, coordinates, the `(source, alpha)` pair
/// that says where each output node came from, and the `node_map` that says where each
/// input node went. Aliased for the same reason as `DroppedOut` — five arrays of four
/// different shapes is past the point where the signature reads.
type ResampledOut<'py> = (
    Bound<'py, PyArray1<i32>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<i32>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<i32>>,
);

fn dropped_to_py(
    py: Python<'_>,
    out: (Vec<i32>, Array1<i32>, Option<Vec<f32>>, Array1<i32>),
) -> DroppedOut<'_> {
    let (kept, new_parents, new_weights, node_map) = out;
    (
        Array1::from(kept).into_pyarray(py),
        new_parents.into_pyarray(py),
        new_weights.map(|w| Array1::from(w).into_pyarray(py)),
        node_map.into_pyarray(py),
    )
}

/// Keep every `factor`-th node of every segment, plus everything that carries topology.
///
/// Arguments:
///
/// - `parents`:  array of parent indices
/// - `factor`:   keep one node in every `factor`, counting from each segment's distal end
/// - `preserve`: optional (N, ) bool array of extra nodes that must survive
/// - `weights`:  optional per-node length of the child -> parent edge
///
/// Returns:
///
/// `(kept, new_parents, new_weights, node_map)`; `new_parents` indexes into `kept`, and
/// `node_map` is (N, ): for each input node, the surviving node its data belongs to now.
///
#[pyfunction]
#[pyo3(name = "downsample_skeleton", signature = (parents, factor, preserve=None, weights=None))]
pub fn downsample_skeleton_py<'py>(
    py: Python<'py>,
    parents: PyReadonlyArray1<i32>,
    factor: usize,
    preserve: Option<PyReadonlyArray1<bool>>,
    weights: Option<PyReadonlyArray1<f32>>,
) -> DroppedOut<'py> {
    let preserve: Option<Array1<bool>> = preserve.map(|p| p.as_array().to_owned());
    let weights: Option<Array1<f32>> = weights.map(|w| w.as_array().to_owned());
    dropped_to_py(
        py,
        downsample_skeleton(&parents.as_array(), factor, &preserve, &weights),
    )
}

/// Drop the nodes that do not bend a neurite, by Ramer-Douglas-Peucker.
///
/// Arguments:
///
/// - `parents`:  array of parent indices
/// - `coords`:   (N, 3) float64 node coordinates
/// - `epsilon`:  how far the simplified path may stray from the original
/// - `preserve`: optional (N, ) bool array of extra nodes that must survive
/// - `weights`:  optional per-node length of the child -> parent edge
/// - `threads`:  size of the thread pool, or `None` for all cores
///
/// Returns:
///
/// `(kept, new_parents, new_weights, node_map)`; `new_parents` indexes into `kept`, and
/// `node_map` is (N, ): for each input node, the surviving node its data belongs to now.
///
#[pyfunction]
#[pyo3(name = "simplify_rdp", signature = (parents, coords, epsilon, preserve=None, weights=None, threads=None))]
pub fn simplify_rdp_py<'py>(
    py: Python<'py>,
    parents: PyReadonlyArray1<i32>,
    coords: PyReadonlyArray2<f64>,
    epsilon: f64,
    preserve: Option<PyReadonlyArray1<bool>>,
    weights: Option<PyReadonlyArray1<f32>>,
    threads: Option<usize>,
) -> DroppedOut<'py> {
    let preserve: Option<Array1<bool>> = preserve.map(|p| p.as_array().to_owned());
    let weights: Option<Array1<f32>> = weights.map(|w| w.as_array().to_owned());
    dropped_to_py(
        py,
        simplify_rdp(
            &parents.as_array(),
            &coords.as_array(),
            epsilon,
            &preserve,
            &weights,
            threads,
        ),
    )
}

/// Drop the nodes that contribute least area, by Visvalingam-Whyatt.
///
/// Arguments:
///
/// - `parents`:  array of parent indices
/// - `coords`:   (N, 3) float64 node coordinates
/// - `min_area`: remove a node while its triangle is smaller than this
/// - `preserve`: optional (N, ) bool array of extra nodes that must survive
/// - `weights`:  optional per-node length of the child -> parent edge
/// - `threads`:  size of the thread pool, or `None` for all cores
///
/// Returns:
///
/// `(kept, new_parents, new_weights, node_map)`; `new_parents` indexes into `kept`, and
/// `node_map` is (N, ): for each input node, the surviving node its data belongs to now.
///
#[pyfunction]
#[pyo3(name = "simplify_vw", signature = (parents, coords, min_area, preserve=None, weights=None, threads=None))]
pub fn simplify_vw_py<'py>(
    py: Python<'py>,
    parents: PyReadonlyArray1<i32>,
    coords: PyReadonlyArray2<f64>,
    min_area: f64,
    preserve: Option<PyReadonlyArray1<bool>>,
    weights: Option<PyReadonlyArray1<f32>>,
    threads: Option<usize>,
) -> DroppedOut<'py> {
    let preserve: Option<Array1<bool>> = preserve.map(|p| p.as_array().to_owned());
    let weights: Option<Array1<f32>> = weights.map(|w| w.as_array().to_owned());
    dropped_to_py(
        py,
        simplify_vw(
            &parents.as_array(),
            &coords.as_array(),
            min_area,
            &preserve,
            &weights,
            threads,
        ),
    )
}

/// Place nodes at a fixed spacing along every neurite.
///
/// Arguments:
///
/// - `parents`: array of parent indices
/// - `coords`:  (N, 3) float64 node coordinates
/// - `spacing`: target distance between adjacent nodes
/// - `threads`: size of the thread pool, or `None` for all cores
///
/// Returns:
///
/// `(new_parents, new_coords, source, alpha, node_map)`. `source` is (M, 2): the input node
/// indices of the edge each output node sits on, child then parent. `alpha` is how far along
/// that edge it lies. `node_map` is (N, ): the output node nearest each input node. The first
/// rows are the input's roots, branch points and leafs, in input order and unmoved.
///
#[pyfunction]
#[pyo3(name = "resample_skeleton", signature = (parents, coords, spacing, threads=None))]
pub fn resample_skeleton_py<'py>(
    py: Python<'py>,
    parents: PyReadonlyArray1<i32>,
    coords: PyReadonlyArray2<f64>,
    spacing: f64,
    threads: Option<usize>,
) -> ResampledOut<'py> {
    let out = resample_skeleton(&parents.as_array(), &coords.as_array(), spacing, threads);
    (
        out.parents.into_pyarray(py),
        out.coords.into_pyarray(py),
        out.source.into_pyarray(py),
        out.alpha.into_pyarray(py),
        out.node_map.into_pyarray(py),
    )
}

/// Smooth a skeleton with a moving average along each neurite.
///
/// Arguments:
///
/// - `parents`: array of parent indices
/// - `coords`:  (N, 3) float64 node coordinates
/// - `window`:  nodes in the window, counting the node itself
/// - `threads`: size of the thread pool, or `None` for all cores
///
/// Returns:
///
/// An (N, 3) array of new coordinates, in the input's node order.
///
#[pyfunction]
#[pyo3(name = "smooth_skeleton", signature = (parents, coords, window, threads=None))]
pub fn smooth_skeleton_py<'py>(
    py: Python<'py>,
    parents: PyReadonlyArray1<i32>,
    coords: PyReadonlyArray2<f64>,
    window: usize,
    threads: Option<usize>,
) -> Bound<'py, PyArray2<f64>> {
    let out: Array2<f64> =
        smooth_skeleton(&parents.as_array(), &coords.as_array(), window, threads);
    out.into_pyarray(py)
}

/// Smooth a skeleton with a Gaussian kernel along each neurite.
///
/// Arguments:
///
/// - `parents`:  array of parent indices
/// - `coords`:   (N, 3) float64 node coordinates
/// - `sigma`:    kernel width, as a distance along the neurite
/// - `truncate`: how many `sigma` out to keep summing
/// - `threads`:  size of the thread pool, or `None` for all cores
///
/// Returns:
///
/// An (N, 3) array of new coordinates, in the input's node order.
///
#[pyfunction]
#[pyo3(name = "smooth_skeleton_gaussian", signature = (parents, coords, sigma, truncate=4.0, threads=None))]
pub fn smooth_skeleton_gaussian_py<'py>(
    py: Python<'py>,
    parents: PyReadonlyArray1<i32>,
    coords: PyReadonlyArray2<f64>,
    sigma: f64,
    truncate: f64,
    threads: Option<usize>,
) -> Bound<'py, PyArray2<f64>> {
    let out: Array2<f64> = smooth_skeleton_gaussian(
        &parents.as_array(),
        &coords.as_array(),
        sigma,
        truncate,
        threads,
    );
    out.into_pyarray(py)
}
