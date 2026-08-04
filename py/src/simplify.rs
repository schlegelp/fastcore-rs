use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use fastcore::simplify::{simplify_mesh, simplify_mesh_lossless, Simplified, Target};

use crate::mesh::as_opt_flags;

/// The three arrays every entry point here returns.
///
/// Spelled out once because both functions hand back the same shape, and because the
/// order matters: vertices first, matching `pyfqmr`'s `getMesh` and trimesh, so that a
/// caller who writes `verts, faces, vmap = ...` gets what they expect. Two `(N, 3)`
/// arrays the other way round would unpack silently and be wrong much later.
type MeshOut<'py> = (
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<u32>>,
    Bound<'py, PyArray1<i32>>,
);

fn into_py(py: Python<'_>, out: Simplified) -> MeshOut<'_> {
    (
        out.vertices.into_pyarray(py),
        out.faces.into_pyarray(py),
        out.vertex_map.into_pyarray(py),
    )
}

/// Select the `Target` variant, as `criterion_of` does for `matches::Criterion`.
///
/// What a ratio *means* — the rounding, the floor of one face — lives in the core,
/// so this only has to pick which of the two the caller named.
fn target_of(ratio: Option<f64>, n_faces: Option<usize>) -> PyResult<Target> {
    match (ratio, n_faces) {
        (Some(r), None) => Ok(Target::Ratio(r)),
        (None, Some(n)) => Ok(Target::Faces(n)),
        _ => Err(PyValueError::new_err(
            "provide exactly one of `ratio` or `n_faces`",
        )),
    }
}

/// Simplify a triangle mesh down to a target face count.
///
/// Arguments
/// ---------
/// - `faces`:           (F, 3) uint32 array of triangular faces (vertex indices).
/// - `vertices`:        (V, 3) float64 vertex positions.
/// - `ratio`:           Fraction of the faces to keep, in (0, 1].
/// - `n_faces`:         Absolute number of faces to keep. Give exactly one of these.
/// - `aggressiveness`:  Exponent of the error-threshold sweep; 7.0 is upstream's default.
/// - `preserve_border`: Freeze every vertex on a mesh boundary.
/// - `lock`:            Optional (V, ) bool array of vertices that must survive unmoved.
///
/// Returns
/// -------
/// A 3-tuple `(vertices, faces, vertex_map)`. `vertices` is (V', 3) float64 and `faces`
/// is (F', 3) uint32. `vertex_map` is (V, ) int32: for each *input* vertex, the index of
/// the *output* vertex it ended up in, or -1 if it did not survive.
#[pyfunction]
#[pyo3(
    name = "simplify_mesh",
    signature = (faces, vertices, ratio=None, n_faces=None, aggressiveness=7.0, preserve_border=false, lock=None)
)]
// The two ways of naming a face budget are mutually exclusive, so they read as one
// argument from Python even though they are two here.
#[allow(clippy::too_many_arguments)]
pub fn simplify_mesh_py<'py>(
    py: Python<'py>,
    faces: PyReadonlyArray2<u32>,
    vertices: PyReadonlyArray2<f64>,
    ratio: Option<f64>,
    n_faces: Option<usize>,
    aggressiveness: f64,
    preserve_border: bool,
    lock: Option<PyReadonlyArray1<bool>>,
) -> PyResult<MeshOut<'py>> {
    let target = target_of(ratio, n_faces)?;
    let (faces, vertices) = (faces.as_array(), vertices.as_array());
    let lock = as_opt_flags(&lock, "lock", vertices.nrows())?;

    // Off the GIL: this is a single-threaded sweep that runs for hundreds of
    // milliseconds on a million-face mesh, and holding the GIL through it would
    // serialise any caller simplifying several meshes from a thread pool. The
    // views are taken first because the `PyReadonlyArray` guards carry a Python
    // token and so cannot cross the boundary; the views themselves are plain data.
    let out = py.detach(|| {
        simplify_mesh(
            faces,
            vertices,
            target,
            aggressiveness,
            preserve_border,
            lock,
        )
    });
    Ok(into_py(py, out))
}

/// Simplify a triangle mesh without changing its shape.
///
/// Arguments
/// ---------
/// - `faces`:           (F, 3) uint32 array of triangular faces (vertex indices).
/// - `vertices`:        (V, 3) float64 vertex positions.
/// - `epsilon`:         Quadric error below which an edge may collapse.
/// - `max_iterations`:  Cap on the number of passes.
/// - `preserve_border`: Freeze every vertex on a mesh boundary.
/// - `lock`:            Optional (V, ) bool array of vertices that must survive unmoved.
///
/// Returns
/// -------
/// A 3-tuple `(vertices, faces, vertex_map)`, as for `simplify_mesh`.
#[pyfunction]
#[pyo3(
    name = "simplify_mesh_lossless",
    signature = (faces, vertices, epsilon=1e-3, max_iterations=9999, preserve_border=false, lock=None)
)]
pub fn simplify_mesh_lossless_py<'py>(
    py: Python<'py>,
    faces: PyReadonlyArray2<u32>,
    vertices: PyReadonlyArray2<f64>,
    epsilon: f64,
    max_iterations: usize,
    preserve_border: bool,
    lock: Option<PyReadonlyArray1<bool>>,
) -> PyResult<MeshOut<'py>> {
    let (faces, vertices) = (faces.as_array(), vertices.as_array());
    let lock = as_opt_flags(&lock, "lock", vertices.nrows())?;

    // As above: views out first, then run the sweep off the GIL.
    let out = py.detach(|| {
        simplify_mesh_lossless(
            faces,
            vertices,
            epsilon,
            max_iterations,
            preserve_border,
            lock,
        )
    });
    Ok(into_py(py, out))
}
