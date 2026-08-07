use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use fastcore::caps::{
    boundary_halfedges, check_rings, exposed_halfedges, trace_loops, triangulate_rings,
};

use crate::mesh::as_slice;

/// What `trace_loops` hands back: the rings, flat, and the offsets that cut them up.
type RingsOut<'py> = (Bound<'py, PyArray1<u32>>, Bound<'py, PyArray1<i64>>);

/// Find every edge of a mesh that has only one face on it.
///
/// Arguments
/// ---------
/// - `faces`:   (F, 3) uint32 array of triangular faces (vertex indices).
/// - `threads`: Size of the thread pool, or `None` for all cores.
///
/// Returns
/// -------
/// A (K, 2) uint32 array of directed half-edges, wound the way their one remaining
/// face winds them.
#[pyfunction]
#[pyo3(name = "boundary_halfedges", signature = (faces, threads=None))]
pub fn boundary_halfedges_py<'py>(
    py: Python<'py>,
    faces: PyReadonlyArray2<u32>,
    threads: Option<usize>,
) -> Bound<'py, PyArray2<u32>> {
    let faces = faces.as_array();
    // Off the GIL: one parallel sort of 3F keys plus two passes over the faces, and the
    // core takes the rayon pool for the duration.
    let out = py.detach(|| boundary_halfedges(faces, threads));
    out.into_pyarray(py)
}

/// Find the edges a subset is about to expose. Takes the faces *before* subsetting.
///
/// Arguments
/// ---------
/// - `faces`:   (F, 3) uint32 array of faces before subsetting.
/// - `dropped`: (V, ) bool array — for each vertex, whether the subset drops it.
/// - `threads`: Size of the thread pool, or `None` for all cores.
///
/// Returns
/// -------
/// A (K, 2) uint32 array of directed half-edges, indices into the *original* vertices.
#[pyfunction]
#[pyo3(name = "exposed_halfedges", signature = (faces, dropped, threads=None))]
pub fn exposed_halfedges_py<'py>(
    py: Python<'py>,
    faces: PyReadonlyArray2<u32>,
    dropped: PyReadonlyArray1<bool>,
    threads: Option<usize>,
) -> PyResult<Bound<'py, PyArray2<u32>>> {
    let faces = faces.as_array();
    // The face index range is the wrapper's job, as it is for `smooth_mesh_py`: it has
    // already checked `faces.max()` against `len(dropped)`, vectorised. Repeating it here
    // would be a *serial* pass over all 3F indices — 1 ms of the 1.6 ms this call takes on
    // a 578k-face mesh — to re-answer a question already answered. A mask that slips
    // through anyway indexes out of bounds in the core, which is a panic, not a wrong
    // answer.
    let dropped = as_slice(&dropped, "dropped")?;

    let out = py.detach(|| exposed_halfedges(faces, dropped, threads));
    Ok(out.into_pyarray(py))
}

/// Walk directed half-edges into closed rings.
///
/// Arguments
/// ---------
/// - `halfedges`: (K, 2) uint32 array of directed half-edges.
///
/// Returns
/// -------
/// A 2-tuple `(rings, offsets)` in CSR form: ring `i` is `rings[offsets[i]:offsets[i+1]]`,
/// so `offsets` (int64) has one more entry than there are rings.
#[pyfunction]
#[pyo3(name = "trace_loops")]
pub fn trace_loops_py<'py>(
    py: Python<'py>,
    halfedges: PyReadonlyArray2<u32>,
) -> RingsOut<'py> {
    let halfedges = halfedges.as_array();
    // Sequential, but proportional to the boundary rather than the mesh — and it builds a
    // CSR of it first, so it is still worth not holding the GIL through.
    let (rings, offsets) = py.detach(|| trace_loops(halfedges));
    (rings.into_pyarray(py), offsets.into_pyarray(py))
}

/// Triangulate boundary rings, wound against the direction they run in.
///
/// Arguments
/// ---------
/// - `rings`, `offsets`: boundary rings in the CSR form `trace_loops` returns.
/// - `vertices`:         (V, 3) float64 vertex positions.
/// - `threads`:          Size of the thread pool, or `None` for all cores.
///
/// Returns
/// -------
/// An (M, 3) uint32 array of new faces, indices into `vertices`.
#[pyfunction]
#[pyo3(name = "triangulate_rings", signature = (rings, offsets, vertices, threads=None))]
pub fn triangulate_rings_py<'py>(
    py: Python<'py>,
    rings: PyReadonlyArray1<u32>,
    offsets: PyReadonlyArray1<i64>,
    vertices: PyReadonlyArray2<f64>,
    threads: Option<usize>,
) -> PyResult<Bound<'py, PyArray2<u32>>> {
    let rings = as_slice(&rings, "rings")?;
    let offsets = as_slice(&offsets, "offsets")?;
    let vertices = vertices.as_array();

    // Shapes and the ring index range are the wrapper's job — `triangulate_rings` in
    // `navis_fastcore/caps.py` has forced the dtypes and checked `rings.max()`, vectorised,
    // before this is reached. What is left is the pair's internal consistency, which the
    // core defines so that this binding and R's agree on it.
    check_rings(rings, offsets).map_err(PyValueError::new_err)?;

    let out = py.detach(|| triangulate_rings(rings, offsets, vertices, threads));
    Ok(out.into_pyarray(py))
}
