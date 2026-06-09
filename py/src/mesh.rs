use numpy::{IntoPyArray, PyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use fastcore::mesh::mesh_connected_components;

/// Find connected components of a triangle mesh.
///
/// Arguments
/// ---------
/// - `faces`:      (N, 3) uint32 array of triangular faces (vertex indices).
/// - `n_vertices`: Total number of vertices.
///
/// Returns
/// -------
/// A 1-D uint32 array of length `n_vertices` where each entry contains the
/// root-vertex index of the component the vertex belongs to.
#[pyfunction]
#[pyo3(name = "mesh_connected_components")]
pub fn mesh_connected_components_py<'py>(
    py: Python<'py>,
    faces: PyReadonlyArray2<u32>,
    n_vertices: usize,
) -> Bound<'py, PyArray1<u32>> {
    let result = mesh_connected_components(faces.as_array(), n_vertices);
    result.into_pyarray(py)
}
