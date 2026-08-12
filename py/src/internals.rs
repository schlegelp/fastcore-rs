use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use fastcore::internals::{drop_internals, openness};

use crate::mesh::as_opt_flags;

/// What `drop_internals` hands back: the repaired mesh, the vertex map, and the per-pass log.
type RepairOut<'py> = (
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<u32>>,
    Bound<'py, PyArray1<u32>>,
    Vec<Bound<'py, PyDict>>,
);

/// Fraction of rays leaving each face that escape the mesh.
///
/// For every face, fire a cosine-weighted spray of rays into the hemisphere above it and count
/// how many get away. Outer membrane lands at 0.5-1.0, the wall of an invagination at 0; the
/// distribution is sharply bimodal, so the threshold that separates them is not really a
/// tuning parameter.
///
/// Arguments
/// ---------
/// - `vertices`: (V, 3) float64 array of vertex positions.
/// - `faces`:    (F, 3) uint32 array of triangular faces (vertex indices).
/// - `mask`:     (F, ) bool array, optional. Cast only from these faces. The whole mesh still
///   blocks rays - only the *sources* are restricted - so each value is exactly the one a full
///   sweep would have produced. (The prototype called this `where`, which is a Rust keyword
///   and so cannot be a pyo3 argument name.)
/// - `n_rays`:   Rays per face. 16 is plenty; 8 halves the cost for no measured difference.
/// - `offset`:   How far off the surface to start each ray. `None` is 5% of the median edge
///   length, which keeps this scale-free.
/// - `seed`:     Fixes the spray. Hashed per (face, ray), so the answer does not depend on
///   thread count or on whether this is a full sweep or a partial one.
/// - `threads`:  Size of the thread pool, or `None` for all cores.
///
/// Returns
/// -------
/// A (F, ) float64 array of fractions in [0, 1] - or one value per selected face, in face
/// order, if `mask` was given.
#[pyfunction]
#[pyo3(
    name = "openness",
    signature = (vertices, faces, mask=None, n_rays=16, offset=None, seed=1985, threads=None)
)]
#[allow(clippy::too_many_arguments)]
pub fn openness_py<'py>(
    py: Python<'py>,
    vertices: PyReadonlyArray2<f64>,
    faces: PyReadonlyArray2<u32>,
    mask: Option<PyReadonlyArray1<bool>>,
    n_rays: u32,
    offset: Option<f64>,
    seed: u64,
    threads: Option<usize>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let v = vertices.as_array();
    let f = faces.as_array();
    // Shapes, dtypes and `n_rays` are the wrapper's job and the core's assert; what is left
    // here is the borrow, and the length check that lives next to it.
    let mask = as_opt_flags(&mask, "mask", f.nrows())?;

    // Off the GIL: this is the whole cost of the pipeline it belongs to - a BVH build and
    // n_rays traversals per face, all of it on the rayon pool.
    let out = py.detach(|| openness(v, f, mask, n_rays, offset, seed, threads));
    Ok(PyArray1::from_vec(py, out))
}

/// Strip invaginated surface from a mesh and close it back up.
///
/// An invagination is a piece of membrane that bulges *into* the cell - typically the boundary
/// of an organelle touching it from inside. They are what a wavefront skeletonisation trips
/// over: on a heavily invaginated neuron they account for 3,076 of the mesh's handles, and
/// removing them takes the skeleton from 3,227 leafs to ~500 without losing a real branch.
///
/// The cycle is: score every face by `openness`, smooth that field over the faces, drop what
/// falls below `threshold`, drop the components left enclosing no volume, cap the holes - then
/// repeat, since capping a pocket mouth turns a partially-open neighbour into a fully buried
/// one. Passes converge fast: 18.7% of faces buried, then 0.5%, then 0.2%.
///
/// Arguments
/// ---------
/// - `vertices`:   (V, 3) float64 array of vertex positions.
/// - `faces`:      (F, 3) uint32 array of triangular faces (vertex indices).
/// - `threshold`:  Faces whose smoothed openness falls below this are cut. 0.05-0.10 is the
///   operating range; above ~0.1 the cut starts eating real membrane.
/// - `n_rays`:     Rays per face - see `openness`.
/// - `smooth`:     Diffusion rounds applied to the field before thresholding. Thresholding a
///   raw field cuts along a ragged contour; this shortens it several fold without moving where
///   the cut sits.
/// - `iterations`: Passes of the whole cycle.
/// - `hops`:       How far past the previous pass's caps to re-cast. `None` re-casts
///   everything, which costs about twice as much and moves no mesh metric more than a percent.
///   Worth keeping above `smooth`, which is how far a stale value at the rim can diffuse in.
/// - `seed`:       Fixes the ray spray.
/// - `threads`:    Size of the thread pool, or `None` for all cores.
///
/// Returns
/// -------
/// `(vertices, faces, keep, passes)`. `keep` is a (V', ) uint32 array giving each surviving
/// vertex's index in the input - the repair only ever *removes* vertices, since caps re-use
/// the ones already on the boundary, so that is all a caller needs to carry a vertex map,
/// connectors or any per-vertex annotation across. `passes` is one dict per pass run, with
/// `faces_before`, `buried`, `recast`, `capped` and `faces_after`.
#[pyfunction]
#[pyo3(
    name = "drop_internals",
    signature = (vertices, faces, threshold=0.05, n_rays=16, smooth=10, iterations=3, hops=20, seed=1985, threads=None)
)]
#[allow(clippy::too_many_arguments)]
pub fn drop_internals_py<'py>(
    py: Python<'py>,
    vertices: PyReadonlyArray2<f64>,
    faces: PyReadonlyArray2<u32>,
    threshold: f64,
    n_rays: u32,
    smooth: u32,
    iterations: u32,
    hops: Option<u32>,
    seed: u64,
    threads: Option<usize>,
) -> PyResult<RepairOut<'py>> {
    let v = vertices.as_array();
    let f = faces.as_array();

    let (nv, nf, keep, passes) = py.detach(|| {
        drop_internals(
            v, f, threshold, n_rays, smooth, iterations, hops, seed, threads,
        )
    });

    let log = passes
        .iter()
        .map(|p| {
            let d = PyDict::new(py);
            d.set_item("faces_before", p.faces_before)?;
            d.set_item("buried", p.buried)?;
            d.set_item("recast", p.recast)?;
            d.set_item("capped", p.capped)?;
            d.set_item("faces_after", p.faces_after)?;
            Ok(d)
        })
        .collect::<PyResult<Vec<_>>>()?;

    Ok((
        nv.into_pyarray(py),
        nf.into_pyarray(py),
        keep.into_pyarray(py),
        log,
    ))
}
