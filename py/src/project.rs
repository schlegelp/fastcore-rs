use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use fastcore::project::project_mesh_2d;

/// What `project_mesh_2d` hands back: rings, bounding box, face indices, and the two
/// optional halves.
type ProjectionOut<'py> = (
    Bound<'py, PyArray3<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<i64>>,
    Option<Bound<'py, PyArray1<f64>>>,
    Option<Bound<'py, PyArray2<f64>>>,
);

/// Project a mesh into a view plane: cull, sort and lay out, in one pass.
///
/// Arguments
/// ---------
/// - `vertices`: (V, 3) float64 vertex positions.
/// - `faces`:    (F, 3) uint32 array of triangular faces (vertex indices).
/// - `xy_ix`:    the two coordinate columns that make up the picture.
/// - `depth_ix`: the remaining, into-the-screen column.
/// - `front`:    1 or -1, the direction along `depth_ix` that points at the viewer.
/// - `order`:    sort furthest-first and return the depths.
/// - `normals`:  return unit face normals.
/// - `threads`:  Size of the thread pool, or `None` for all cores.
///
/// Returns
/// -------
/// A 5-tuple `(rings, bbox, ix, depth, normals)`. `rings` is (K, 4, 2) float64, each
/// triangle closed by a repeat of its first corner; `bbox` is `[xmin, ymin, xmax, ymax]`
/// over those rings. `depth` and `normals` are `None` unless asked for.
#[pyfunction]
#[pyo3(
    name = "project_mesh_2d",
    signature = (vertices, faces, xy_ix, depth_ix, front, order=true, normals=false, threads=None)
)]
#[allow(clippy::too_many_arguments)]
pub fn project_mesh_2d_py<'py>(
    py: Python<'py>,
    vertices: PyReadonlyArray2<f64>,
    faces: PyReadonlyArray2<u32>,
    xy_ix: (usize, usize),
    depth_ix: usize,
    front: i8,
    order: bool,
    normals: bool,
    threads: Option<usize>,
) -> PyResult<ProjectionOut<'py>> {
    // Cheap, and the alternative is an assert inside the core coming back as a
    // PanicException with a traceback pointing at nothing the caller wrote.
    let mut axes = [xy_ix.0, xy_ix.1, depth_ix];
    axes.sort_unstable();
    if axes != [0, 1, 2] {
        return Err(PyValueError::new_err(format!(
            "`xy_ix` and `depth_ix` must together be 0, 1 and 2 in some order, \
             got {xy_ix:?} and {depth_ix}"
        )));
    }
    if front != 1 && front != -1 {
        return Err(PyValueError::new_err(format!(
            "`front` must be 1 or -1, got {front}"
        )));
    }

    let vertices = vertices.as_array();
    let faces = faces.as_array();

    // Off the GIL: one parallel pass per stage over arrays of tens of millions of rows,
    // and the core takes the rayon pool for the duration. The face index range is checked
    // in there, in parallel - see the note in `project_mesh_2d`.
    let out = py.detach(|| {
        project_mesh_2d(
            vertices, faces, xy_ix, depth_ix, front, order, normals, threads,
        )
    });

    Ok((
        out.rings.into_pyarray(py),
        out.bbox.to_vec().into_pyarray(py),
        out.ix.into_pyarray(py),
        out.depth.map(|d| d.into_pyarray(py)),
        out.normals.map(|n| n.into_pyarray(py)),
    ))
}
