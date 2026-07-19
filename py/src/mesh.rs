use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use fastcore::mesh::{
    geodesic_farthest_mesh, geodesic_matrix_graph, geodesic_matrix_mesh, geodesic_nearest_mesh,
    mesh_connected_components, unique_edges,
};

/// Borrow an index array as a contiguous slice.
///
/// The borrow has to be bound to a local by the caller: `x.as_ref().map(|a| a.as_slice())`
/// chained off a temporary would not live long enough for the slice we hand to the core.
fn as_slice<'a>(a: &'a PyReadonlyArray1<u32>, what: &str) -> PyResult<&'a [u32]> {
    a.as_slice()
        .map_err(|_| PyValueError::new_err(format!("`{what}` must be C-contiguous")))
}

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

/// Unique undirected edges of a triangle mesh (trimesh `edges_unique` equivalent).
///
/// Arguments
/// ---------
/// - `faces`:          (F, 3) uint32 array of triangular faces (vertex indices).
/// - `coords`:         (V, 3) float64 vertex positions; when given, also return
///   each unique edge's euclidean length (trimesh's `edges_unique_length`).
/// - `return_index`:   Also return each unique edge's first occurrence in the
///   per-face edge list (trimesh's `edges_unique_idx`).
/// - `return_inverse`: Also return, for each of the 3F per-face edges, the row of
///   its unique edge (trimesh's `edges_unique_inverse`; reshape to (F, 3) for
///   `faces_unique_edges`).
/// - `threads`:        Size of the thread pool, or `None` for all cores.
///
/// Returns
/// -------
/// A 4-tuple `(edges, index, inverse, lengths)`: `edges` is (n_unique, 2) int64
/// with rows `[min, max]` ordered ascending by (max, min) — identical to trimesh;
/// the other three are parallel arrays or `None` when not requested.
#[pyfunction]
#[pyo3(
    name = "unique_edges",
    signature = (faces, coords=None, return_index=false, return_inverse=false, threads=None)
)]
#[allow(clippy::type_complexity)]
pub fn unique_edges_py<'py>(
    py: Python<'py>,
    faces: PyReadonlyArray2<u32>,
    coords: Option<PyReadonlyArray2<f64>>,
    return_index: bool,
    return_inverse: bool,
    threads: Option<usize>,
) -> (
    Bound<'py, PyArray2<i64>>,
    Option<Bound<'py, PyArray1<i64>>>,
    Option<Bound<'py, PyArray1<i64>>>,
    Option<Bound<'py, PyArray1<f64>>>,
) {
    let (edges, index, inverse, lengths) = unique_edges(
        faces.as_array(),
        coords.as_ref().map(|c| c.as_array()),
        return_index,
        return_inverse,
        threads,
    );
    (
        edges.into_pyarray(py),
        index.map(|a| a.into_pyarray(py)),
        inverse.map(|a| a.into_pyarray(py)),
        lengths.map(|a| a.into_pyarray(py)),
    )
}

/// Pairwise geodesic ("along-the-mesh-edge") distances on a triangle mesh.
///
/// Arguments
/// ---------
/// - `faces`:      (F, 3) uint32 array of triangular faces (vertex indices).
/// - `n_vertices`: Total number of vertices.
/// - `coords`:     (n_vertices, 3) float64 vertex positions, or `None` for hop counts.
/// - `sources`:    uint32 source vertex indices, or `None` for all.
/// - `targets`:    uint32 target vertex indices, or `None` for all.
/// - `limit`:      Prune the search at this distance (inclusive), or `None`.
/// - `threads`:    Size of the thread pool, or `None` for all cores.
///
/// Returns
/// -------
/// A (len(sources), len(targets)) float32 matrix; `-1` where unreachable.
#[pyfunction]
#[pyo3(
    name = "geodesic_matrix_mesh",
    signature = (faces, n_vertices, coords=None, sources=None, targets=None, limit=None, threads=None)
)]
#[allow(clippy::too_many_arguments)]
pub fn geodesic_matrix_mesh_py<'py>(
    py: Python<'py>,
    faces: PyReadonlyArray2<u32>,
    n_vertices: usize,
    coords: Option<PyReadonlyArray2<f64>>,
    sources: Option<PyReadonlyArray1<u32>>,
    targets: Option<PyReadonlyArray1<u32>>,
    limit: Option<f32>,
    threads: Option<usize>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let src = match sources.as_ref() {
        Some(a) => Some(as_slice(a, "sources")?),
        None => None,
    };
    let tgt = match targets.as_ref() {
        Some(a) => Some(as_slice(a, "targets")?),
        None => None,
    };
    let dists = geodesic_matrix_mesh(
        faces.as_array(),
        n_vertices,
        coords.as_ref().map(|c| c.as_array()),
        src,
        tgt,
        limit,
        threads,
    );
    Ok(dists.into_pyarray(py))
}

/// Pairwise geodesic distances over an arbitrary undirected graph given as an edge list.
///
/// The general form of `geodesic_matrix_mesh`: unlike the `dag` geodesic functions, this makes
/// no tree assumption, so cycles are fine.
///
/// Arguments
/// ---------
/// - `edges`:   (E, 2) uint32 array of undirected edges (node indices).
/// - `n_nodes`: Total number of nodes.
/// - `weights`:  (E, ) float32 edge lengths, or `None` for hop counts.
/// - `directed`: If true, an edge (u, v) may only be traversed from u to v.
/// - `sources`, `targets`, `limit`, `threads`: as `geodesic_matrix_mesh`.
///
/// Returns
/// -------
/// A (len(sources), len(targets)) float32 matrix; `-1` where unreachable.
#[pyfunction]
#[pyo3(
    name = "geodesic_matrix_graph",
    signature = (edges, n_nodes, weights=None, directed=false, sources=None, targets=None, limit=None, threads=None)
)]
#[allow(clippy::too_many_arguments)]
pub fn geodesic_matrix_graph_py<'py>(
    py: Python<'py>,
    edges: PyReadonlyArray2<u32>,
    n_nodes: usize,
    weights: Option<PyReadonlyArray1<f32>>,
    directed: bool,
    sources: Option<PyReadonlyArray1<u32>>,
    targets: Option<PyReadonlyArray1<u32>>,
    limit: Option<f32>,
    threads: Option<usize>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let src = match sources.as_ref() {
        Some(a) => Some(as_slice(a, "sources")?),
        None => None,
    };
    let tgt = match targets.as_ref() {
        Some(a) => Some(as_slice(a, "targets")?),
        None => None,
    };
    let w = weights.as_ref().map(|w| w.as_array());
    let dists = geodesic_matrix_graph(
        edges.as_array(),
        n_nodes,
        w.as_ref(),
        directed,
        src,
        tgt,
        limit,
        threads,
    );
    Ok(dists.into_pyarray(py))
}

/// For each source, the distance to its nearest target and that target's vertex index.
///
/// O(sources) memory instead of O(sources x targets) — the only thing that scales on a large
/// mesh. Sources are matched to a *distinct* target, never to themselves; `-1` / `-1` when no
/// target is reachable.
#[pyfunction]
#[pyo3(
    name = "geodesic_nearest_mesh",
    signature = (faces, n_vertices, coords=None, sources=None, targets=None, limit=None, threads=None)
)]
#[allow(clippy::too_many_arguments)]
pub fn geodesic_nearest_mesh_py<'py>(
    py: Python<'py>,
    faces: PyReadonlyArray2<u32>,
    n_vertices: usize,
    coords: Option<PyReadonlyArray2<f64>>,
    sources: Option<PyReadonlyArray1<u32>>,
    targets: Option<PyReadonlyArray1<u32>>,
    limit: Option<f32>,
    threads: Option<usize>,
) -> PyResult<(Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<i32>>)> {
    let src = match sources.as_ref() {
        Some(a) => Some(as_slice(a, "sources")?),
        None => None,
    };
    let tgt = match targets.as_ref() {
        Some(a) => Some(as_slice(a, "targets")?),
        None => None,
    };
    let (dists, nodes) = geodesic_nearest_mesh(
        faces.as_array(),
        n_vertices,
        coords.as_ref().map(|c| c.as_array()),
        src,
        tgt,
        limit,
        threads,
    );
    Ok((dists.into_pyarray(py), nodes.into_pyarray(py)))
}

/// For each source, the distance to its farthest target and that target's vertex index.
///
/// The mirror of `geodesic_nearest_mesh`, with the same conventions.
#[pyfunction]
#[pyo3(
    name = "geodesic_farthest_mesh",
    signature = (faces, n_vertices, coords=None, sources=None, targets=None, limit=None, threads=None)
)]
#[allow(clippy::too_many_arguments)]
pub fn geodesic_farthest_mesh_py<'py>(
    py: Python<'py>,
    faces: PyReadonlyArray2<u32>,
    n_vertices: usize,
    coords: Option<PyReadonlyArray2<f64>>,
    sources: Option<PyReadonlyArray1<u32>>,
    targets: Option<PyReadonlyArray1<u32>>,
    limit: Option<f32>,
    threads: Option<usize>,
) -> PyResult<(Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<i32>>)> {
    let src = match sources.as_ref() {
        Some(a) => Some(as_slice(a, "sources")?),
        None => None,
    };
    let tgt = match targets.as_ref() {
        Some(a) => Some(as_slice(a, "targets")?),
        None => None,
    };
    let (dists, nodes) = geodesic_farthest_mesh(
        faces.as_array(),
        n_vertices,
        coords.as_ref().map(|c| c.as_array()),
        src,
        tgt,
        limit,
        threads,
    );
    Ok((dists.into_pyarray(py), nodes.into_pyarray(py)))
}
