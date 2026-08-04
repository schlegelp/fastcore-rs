use std::sync::RwLock;

use ndarray::{ArrayView1, ArrayView2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use fastcore::mesh::{
    bridges, connected_components_graph, contract_vertices, geodesic_clusters,
    geodesic_farthest_mesh, geodesic_matrix_graph, geodesic_matrix_mesh, geodesic_mst_graph,
    geodesic_mst_mesh, geodesic_nearest_mesh, geodesic_path_graph, geodesic_predecessors_graph,
    level_set_components, mesh_connected_components, minimum_spanning_tree, parents_from_edges,
    unique_edges, GeodesicGraph, Weight,
};

/// Edge weights, at whatever width the caller already has them in.
///
/// As in `linkage`, `PyReadonlyArray1` extracts only on an *exact* dtype match, so this can
/// never silently copy or cast — the array that arrives is the array the search runs on, and its
/// dtype is what the answer comes back as. `float16` is deliberately absent: Dijkstra
/// accumulates one addition per hop and `f16` runs out of mantissa within a handful of them.
#[derive(FromPyObject)]
pub enum WeightsIn<'py> {
    F32(PyReadonlyArray1<'py, f32>),
    F64(PyReadonlyArray1<'py, f64>),
}

/// The width to answer an *unweighted* query at.
///
/// Hop counts are integers and exact at either width, so this changes nothing about the numbers
/// — only the dtype the caller gets back, which matters when the result is about to be combined
/// with something else. Whenever `weights` is given its own dtype decides and this is not
/// consulted; the Python wrapper is what turns a `dtype=` argument into the pair.
type Float64 = bool;

/// Borrow a 1-D array as a contiguous slice.
///
/// The borrow has to be bound to a local by the caller: `x.as_ref().map(|a| a.as_slice())`
/// chained off a temporary would not live long enough for the slice we hand to the core.
fn as_slice<'a, T: numpy::Element>(a: &'a PyReadonlyArray1<T>, what: &str) -> PyResult<&'a [T]> {
    a.as_slice()
        .map_err(|_| PyValueError::new_err(format!("`{what}` must be C-contiguous")))
}

/// The optional form of [`as_slice`]: `None` passes straight through.
///
/// Every geodesic entry point takes two or three optional index arrays, so without this each
/// one grows a four-line `match` per argument that says nothing the type does not.
fn as_opt_slice<'a, T: numpy::Element>(
    a: &'a Option<PyReadonlyArray1<T>>,
    what: &str,
) -> PyResult<Option<&'a [T]>> {
    a.as_ref().map(|a| as_slice(a, what)).transpose()
}

/// Borrow a per-item boolean mask, checking it is the length the graph expects.
///
/// A wrong-length mask is the easy mistake to make with these — it silently means something
/// else rather than failing — so the check lives next to the borrow.
fn as_flags<'a>(a: &'a PyReadonlyArray1<bool>, what: &str, n_items: usize) -> PyResult<&'a [bool]> {
    let s = as_slice(a, what)?;
    if s.len() != n_items {
        return Err(PyValueError::new_err(format!(
            "`{what}` must have one flag per item: got {}, expected {n_items}",
            s.len()
        )));
    }
    Ok(s)
}

/// The optional form of [`as_flags`], as [`as_opt_slice`] is to [`as_slice`].
pub(crate) fn as_opt_flags<'a>(
    a: &'a Option<PyReadonlyArray1<bool>>,
    what: &str,
    n_items: usize,
) -> PyResult<Option<&'a [bool]>> {
    a.as_ref().map(|a| as_flags(a, what, n_items)).transpose()
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
/// A 4-tuple `(edges, index, inverse, lengths)`: `edges` is (n_unique, 2) uint32
/// with rows `[min, max]` ordered ascending by (max, min) — identical to trimesh;
/// the other three are parallel arrays or `None` when not requested. `index` and
/// `inverse` are int64: they are positions in the 3F edge list, not node ids.
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
    Bound<'py, PyArray2<u32>>,
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
/// - `float64`:    Accumulate and return distances in float64 rather than float32. `coords` is
///   float64 either way — that is the *coordinates'* precision, and each edge length is
///   computed from them at that width and rounded once on the way into the graph.
///
/// Returns
/// -------
/// A (len(sources), len(targets)) float32 (or float64) matrix; `-1` where unreachable.
#[pyfunction]
#[pyo3(
    name = "geodesic_matrix_mesh",
    signature = (faces, n_vertices, coords=None, sources=None, targets=None, limit=None, threads=None, float64=false)
)]
#[allow(clippy::too_many_arguments)]
pub fn geodesic_matrix_mesh_py<'py>(
    py: Python<'py>,
    faces: PyReadonlyArray2<u32>,
    n_vertices: usize,
    coords: Option<PyReadonlyArray2<f64>>,
    sources: Option<PyReadonlyArray1<u32>>,
    targets: Option<PyReadonlyArray1<u32>>,
    limit: Option<f64>,
    threads: Option<usize>,
    float64: Float64,
) -> PyResult<Bound<'py, PyAny>> {
    let src = as_opt_slice(&sources, "sources")?;
    let tgt = as_opt_slice(&targets, "targets")?;
    let (f, c) = (faces.as_array(), coords.as_ref().map(|c| c.as_array()));
    Ok(if float64 {
        geodesic_matrix_mesh::<f64>(f, n_vertices, c, src, tgt, limit, threads)
            .into_pyarray(py)
            .into_any()
    } else {
        geodesic_matrix_mesh::<f32>(
            f,
            n_vertices,
            c,
            src,
            tgt,
            limit.map(f32::from_f64),
            threads,
        )
        .into_pyarray(py)
        .into_any()
    })
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
/// - `weights`:  (E, ) float32 *or* float64 edge lengths, or `None` for hop counts.
/// - `directed`: If true, an edge (u, v) may only be traversed from u to v.
/// - `sources`, `targets`, `limit`, `threads`: as `geodesic_matrix_mesh`.
/// - `float64`: The width for the unweighted case; see [`Float64`].
///
/// Returns
/// -------
/// A (len(sources), len(targets)) matrix in the dtype of `weights`; `-1` where unreachable.
#[pyfunction]
#[pyo3(
    name = "geodesic_matrix_graph",
    signature = (edges, n_nodes, weights=None, directed=false, sources=None, targets=None, limit=None, threads=None, float64=false)
)]
#[allow(clippy::too_many_arguments)]
pub fn geodesic_matrix_graph_py<'py>(
    py: Python<'py>,
    edges: PyReadonlyArray2<u32>,
    n_nodes: usize,
    weights: Option<WeightsIn<'py>>,
    directed: bool,
    sources: Option<PyReadonlyArray1<u32>>,
    targets: Option<PyReadonlyArray1<u32>>,
    limit: Option<f64>,
    threads: Option<usize>,
    float64: Float64,
) -> PyResult<Bound<'py, PyAny>> {
    let src = as_opt_slice(&sources, "sources")?;
    let tgt = as_opt_slice(&targets, "targets")?;

    #[allow(clippy::too_many_arguments)]
    fn run<'py, W: Weight + numpy::Element>(
        py: Python<'py>,
        edges: ArrayView2<u32>,
        n_nodes: usize,
        weights: Option<&ArrayView1<W>>,
        directed: bool,
        sources: Option<&[u32]>,
        targets: Option<&[u32]>,
        limit: Option<f64>,
        threads: Option<usize>,
    ) -> Bound<'py, PyAny> {
        geodesic_matrix_graph(
            edges,
            n_nodes,
            weights,
            directed,
            sources,
            targets,
            limit.map(W::from_f64),
            threads,
        )
        .into_pyarray(py)
        .into_any()
    }

    let e = edges.as_array();
    Ok(match weights {
        Some(WeightsIn::F32(w)) => {
            let w = w.as_array();
            run(py, e, n_nodes, Some(&w), directed, src, tgt, limit, threads)
        }
        Some(WeightsIn::F64(w)) => {
            let w = w.as_array();
            run(py, e, n_nodes, Some(&w), directed, src, tgt, limit, threads)
        }
        None if float64 => run::<f64>(py, e, n_nodes, None, directed, src, tgt, limit, threads),
        None => run::<f32>(py, e, n_nodes, None, directed, src, tgt, limit, threads),
    })
}

/// For each source, the distance to its nearest target and that target's vertex index.
///
/// O(sources) memory instead of O(sources x targets) — the only thing that scales on a large
/// mesh. Sources are matched to a *distinct* target, never to themselves; `-1` / `-1` when no
/// target is reachable.
#[pyfunction]
#[pyo3(
    name = "geodesic_nearest_mesh",
    signature = (faces, n_vertices, coords=None, sources=None, targets=None, limit=None, threads=None, float64=false)
)]
#[allow(clippy::too_many_arguments)]
pub fn geodesic_nearest_mesh_py<'py>(
    py: Python<'py>,
    faces: PyReadonlyArray2<u32>,
    n_vertices: usize,
    coords: Option<PyReadonlyArray2<f64>>,
    sources: Option<PyReadonlyArray1<u32>>,
    targets: Option<PyReadonlyArray1<u32>>,
    limit: Option<f64>,
    threads: Option<usize>,
    float64: Float64,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyArray1<i32>>)> {
    let src = as_opt_slice(&sources, "sources")?;
    let tgt = as_opt_slice(&targets, "targets")?;
    let (f, c) = (faces.as_array(), coords.as_ref().map(|c| c.as_array()));
    Ok(if float64 {
        let (d, n) = geodesic_nearest_mesh::<f64>(f, n_vertices, c, src, tgt, limit, threads);
        (d.into_pyarray(py).into_any(), n.into_pyarray(py))
    } else {
        let (d, n) = geodesic_nearest_mesh::<f32>(
            f,
            n_vertices,
            c,
            src,
            tgt,
            limit.map(f32::from_f64),
            threads,
        );
        (d.into_pyarray(py).into_any(), n.into_pyarray(py))
    })
}

/// Connected components of an undirected graph given as an edge list.
///
/// The edge-list counterpart of `mesh_connected_components`.
///
/// Arguments
/// ---------
/// - `edges`:   (E, 2) uint32 array of undirected edges (node indices).
/// - `n_nodes`: Total number of nodes.
///
/// Returns
/// -------
/// A 1-D uint32 array holding, per node, the smallest node index in its component.
#[pyfunction]
#[pyo3(name = "connected_components_graph")]
pub fn connected_components_graph_py<'py>(
    py: Python<'py>,
    edges: PyReadonlyArray2<u32>,
    n_nodes: usize,
) -> Bound<'py, PyArray1<u32>> {
    connected_components_graph(edges.as_array(), n_nodes).into_pyarray(py)
}

/// Connected components of every level set at once.
///
/// Finds the connected components of each subgraph induced by the nodes sharing a label, for
/// all labels in one `O(E)` pass — no per-level subgraph construction.
///
/// Arguments
/// ---------
/// - `edges`:   (E, 2) uint32 array of undirected edges (node indices).
/// - `n_nodes`: Total number of nodes.
/// - `labels`:  (n_nodes, ) int64 label per node. Negative labels mark excluded nodes,
///   which join no component and come back as `-1`.
///
/// Returns
/// -------
/// `(ids, n_components)`: `ids` is a 1-D int32 array of contiguous component ids in
/// `[0, n_components)`, or `-1` for excluded nodes.
#[pyfunction]
#[pyo3(name = "level_set_components")]
pub fn level_set_components_py<'py>(
    py: Python<'py>,
    edges: PyReadonlyArray2<u32>,
    n_nodes: usize,
    labels: PyReadonlyArray1<i64>,
) -> (Bound<'py, PyArray1<i32>>, usize) {
    let (ids, n) = level_set_components(edges.as_array(), n_nodes, labels.as_array());
    (ids.into_pyarray(py), n)
}

/// Contract nodes onto new ids, returning the simplified edge list.
///
/// igraph's `contract_vertices()` + `simplify()`, fused: both endpoints are pushed through
/// `mapping`, self-loops are dropped and the rest deduplicated.
///
/// Arguments
/// ---------
/// - `edges`:   (E, 2) uint32 array of undirected edges (node indices).
/// - `mapping`: (n_old, ) uint32 new id per old node.
/// - `threads`: Size of the thread pool, or `None` for all cores.
///
/// Returns
/// -------
/// An (n_unique, 2) uint32 array of `[min, max]` rows, ordered as `unique_edges`.
#[pyfunction]
#[pyo3(name = "contract_vertices", signature = (edges, mapping, threads=None))]
pub fn contract_vertices_py<'py>(
    py: Python<'py>,
    edges: PyReadonlyArray2<u32>,
    mapping: PyReadonlyArray1<u32>,
    threads: Option<usize>,
) -> Bound<'py, PyArray2<u32>> {
    contract_vertices(edges.as_array(), mapping.as_array(), threads).into_pyarray(py)
}

/// Minimum (or maximum) spanning forest of an undirected graph.
///
/// Kruskal's algorithm. Disconnected input yields one tree per component.
///
/// Arguments
/// ---------
/// - `edges`:    (E, 2) uint32 array of undirected edges (node indices).
/// - `n_nodes`:  Total number of nodes.
/// - `weights`:  (E, ) float32 or float64 weights, or `None` to treat every edge as equal. Must
///   be finite; negative weights are allowed.
/// - `maximize`: Return the maximum spanning forest instead.
/// - `threads`:  Size of the thread pool, or `None` for all cores.
///
/// Returns
/// -------
/// A 1-D int64 array of row indices into `edges`, ordered by weight. The dtype does not depend
/// on the weights' — these are positions in the caller's array — but the *order* can, where two
/// weights are close enough to compare equal at float32 and not at float64.
#[pyfunction]
#[pyo3(
    name = "minimum_spanning_tree",
    signature = (edges, n_nodes, weights=None, maximize=false, threads=None)
)]
pub fn minimum_spanning_tree_py<'py>(
    py: Python<'py>,
    edges: PyReadonlyArray2<u32>,
    n_nodes: usize,
    weights: Option<WeightsIn<'py>>,
    maximize: bool,
    threads: Option<usize>,
) -> Bound<'py, PyArray1<i64>> {
    let e = edges.as_array();
    match weights {
        Some(WeightsIn::F32(w)) => {
            minimum_spanning_tree(e, n_nodes, Some(&w.as_array()), maximize, threads)
        }
        Some(WeightsIn::F64(w)) => {
            minimum_spanning_tree(e, n_nodes, Some(&w.as_array()), maximize, threads)
        }
        // Unweighted: every edge is equal and the order is the input's, so the width is
        // unobservable here — unlike the geodesic drivers, nothing is accumulated.
        None => minimum_spanning_tree::<f32>(e, n_nodes, None, maximize, threads),
    }
    .into_pyarray(py)
}

/// Which edges are bridges — the ones whose removal would disconnect their component.
///
/// Tarjan's algorithm, one depth-first sweep. Parallel edges are honoured (two nodes joined
/// twice are joined by a cycle, so neither edge is a bridge) and self-loops are never bridges.
///
/// Arguments
/// ---------
/// - `edges`:   (E, 2) uint32 array of undirected edges (node indices).
/// - `n_nodes`: Total number of nodes.
///
/// Returns
/// -------
/// A 1-D bool array with one flag per input edge, `True` for a bridge.
#[pyfunction]
#[pyo3(name = "bridges")]
pub fn bridges_py<'py>(
    py: Python<'py>,
    edges: PyReadonlyArray2<u32>,
    n_nodes: usize,
) -> Bound<'py, PyArray1<bool>> {
    bridges(edges.as_array(), n_nodes).into_pyarray(py)
}

/// Orient a graph into a rooted spanning forest — one parent per node, `-1` at the roots.
///
/// Cycles are broken; each component contributes a spanning tree of itself. One search covers
/// the whole graph however finely it is fragmented.
///
/// Arguments
/// ---------
/// - `edges`:   (E, 2) uint32 array of undirected edges (node indices).
/// - `n_nodes`: Total number of nodes.
/// - `weights`: (E, ) float32 or float64 edge lengths, or `None` for hop counts (breadth-first
///   tree).
/// - `roots`:   (R, ) uint32 nodes to root at, or `None` for the lowest node index in each
///   component. Components holding none of `roots` fall back to that.
///
/// Returns
/// -------
/// `(parents, order)`: `parents` is a (n_nodes, ) int32 array of parent indices (`-1` at a
/// root); `order` is a (n_nodes, ) uint32 topological order in which every node follows its
/// parent. Neither dtype depends on the weights', but which tree comes out can.
#[pyfunction]
#[pyo3(name = "parents_from_edges", signature = (edges, n_nodes, weights=None, roots=None))]
pub fn parents_from_edges_py<'py>(
    py: Python<'py>,
    edges: PyReadonlyArray2<u32>,
    n_nodes: usize,
    weights: Option<WeightsIn<'py>>,
    roots: Option<PyReadonlyArray1<u32>>,
) -> PyResult<(Bound<'py, PyArray1<i32>>, Bound<'py, PyArray1<u32>>)> {
    let r = as_opt_slice(&roots, "roots")?;
    let e = edges.as_array();
    let (parents, order) = match weights {
        Some(WeightsIn::F32(w)) => parents_from_edges(e, n_nodes, Some(&w.as_array()), r),
        Some(WeightsIn::F64(w)) => parents_from_edges(e, n_nodes, Some(&w.as_array()), r),
        // Unweighted: the breadth-first tree, which hop counts pin down at either width.
        None => parents_from_edges::<f32>(e, n_nodes, None, r),
    };
    Ok((parents.into_pyarray(py), order.into_pyarray(py)))
}

/// Minimum spanning tree over a subset of mesh vertices, weighted by geodesic distance.
///
/// Never forms the `k x k` distance matrix: one multi-source sweep partitions the mesh by
/// nearest chosen vertex, and each edge straddling two cells offers one candidate.
///
/// Arguments
/// ---------
/// - `faces`:      (F, 3) uint32 array of triangular faces (vertex indices).
/// - `n_vertices`: Total number of vertices.
/// - `nodes`:      (K, ) uint32 vertices to span. Must be distinct.
/// - `coords`:     (n_vertices, 3) float64 positions for euclidean edge weights, or `None`
///   for hop counts.
/// - `limit`:      Do not join vertices farther apart than this.
/// - `threads`:    Size of the thread pool, or `None` for all cores.
/// - `float64`:    Accumulate and return distances in float64; as `geodesic_matrix_mesh`.
///
/// Returns
/// -------
/// `(edges, weights)`: `edges` is an (M, 2) int64 array of *positions in `nodes`*, ascending by
/// weight; `weights` is the (M, ) float32 (or float64) geodesic distance across each.
#[pyfunction]
#[pyo3(
    name = "geodesic_mst_mesh",
    signature = (faces, n_vertices, nodes, coords=None, limit=None, threads=None, float64=false)
)]
#[allow(clippy::too_many_arguments)]
pub fn geodesic_mst_mesh_py<'py>(
    py: Python<'py>,
    faces: PyReadonlyArray2<u32>,
    n_vertices: usize,
    nodes: PyReadonlyArray1<u32>,
    coords: Option<PyReadonlyArray2<f64>>,
    limit: Option<f64>,
    threads: Option<usize>,
    float64: Float64,
) -> PyResult<(Bound<'py, PyArray2<i64>>, Bound<'py, PyAny>)> {
    let nodes = as_slice(&nodes, "nodes")?;
    let (f, c) = (faces.as_array(), coords.as_ref().map(|c| c.as_array()));
    Ok(if float64 {
        let (e, w) = geodesic_mst_mesh::<f64>(f, n_vertices, c, nodes, limit, threads);
        (e.into_pyarray(py), w.into_pyarray(py).into_any())
    } else {
        let (e, w) =
            geodesic_mst_mesh::<f32>(f, n_vertices, c, nodes, limit.map(f32::from_f64), threads);
        (e.into_pyarray(py), w.into_pyarray(py).into_any())
    })
}

/// Minimum spanning tree over a subset of graph nodes, weighted by geodesic distance.
///
/// The edge-list form of `geodesic_mst_mesh`; always undirected. The returned distances are in
/// the dtype of `weights`.
#[pyfunction]
#[pyo3(
    name = "geodesic_mst_graph",
    signature = (edges, n_nodes, nodes, weights=None, limit=None, threads=None, float64=false)
)]
#[allow(clippy::too_many_arguments)]
pub fn geodesic_mst_graph_py<'py>(
    py: Python<'py>,
    edges: PyReadonlyArray2<u32>,
    n_nodes: usize,
    nodes: PyReadonlyArray1<u32>,
    weights: Option<WeightsIn<'py>>,
    limit: Option<f64>,
    threads: Option<usize>,
    float64: Float64,
) -> PyResult<(Bound<'py, PyArray2<i64>>, Bound<'py, PyAny>)> {
    let nodes = as_slice(&nodes, "nodes")?;

    fn run<'py, W: Weight + numpy::Element>(
        py: Python<'py>,
        edges: ArrayView2<u32>,
        n_nodes: usize,
        weights: Option<&ArrayView1<W>>,
        nodes: &[u32],
        limit: Option<f64>,
        threads: Option<usize>,
    ) -> (Bound<'py, PyArray2<i64>>, Bound<'py, PyAny>) {
        let (e, w) = geodesic_mst_graph(
            edges,
            n_nodes,
            weights,
            nodes,
            limit.map(W::from_f64),
            threads,
        );
        (e.into_pyarray(py), w.into_pyarray(py).into_any())
    }

    let e = edges.as_array();
    Ok(match weights {
        Some(WeightsIn::F32(w)) => run(py, e, n_nodes, Some(&w.as_array()), nodes, limit, threads),
        Some(WeightsIn::F64(w)) => run(py, e, n_nodes, Some(&w.as_array()), nodes, limit, threads),
        None if float64 => run::<f64>(py, e, n_nodes, None, nodes, limit, threads),
        None => run::<f32>(py, e, n_nodes, None, nodes, limit, threads),
    })
}

/// For each source, the distance to its farthest target and that target's vertex index.
///
/// The mirror of `geodesic_nearest_mesh`, with the same conventions.
#[pyfunction]
#[pyo3(
    name = "geodesic_farthest_mesh",
    signature = (faces, n_vertices, coords=None, sources=None, targets=None, limit=None, threads=None, float64=false)
)]
#[allow(clippy::too_many_arguments)]
pub fn geodesic_farthest_mesh_py<'py>(
    py: Python<'py>,
    faces: PyReadonlyArray2<u32>,
    n_vertices: usize,
    coords: Option<PyReadonlyArray2<f64>>,
    sources: Option<PyReadonlyArray1<u32>>,
    targets: Option<PyReadonlyArray1<u32>>,
    limit: Option<f64>,
    threads: Option<usize>,
    float64: Float64,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyArray1<i32>>)> {
    let src = as_opt_slice(&sources, "sources")?;
    let tgt = as_opt_slice(&targets, "targets")?;
    let (f, c) = (faces.as_array(), coords.as_ref().map(|c| c.as_array()));
    Ok(if float64 {
        let (d, n) = geodesic_farthest_mesh::<f64>(f, n_vertices, c, src, tgt, limit, threads);
        (d.into_pyarray(py).into_any(), n.into_pyarray(py))
    } else {
        let (d, n) = geodesic_farthest_mesh::<f32>(
            f,
            n_vertices,
            c,
            src,
            tgt,
            limit.map(f32::from_f64),
            threads,
        );
        (d.into_pyarray(py).into_any(), n.into_pyarray(py))
    })
}

/// Shortest-path trees over a graph — distances *and* the route to each node.
///
/// The predecessor-returning counterpart of `geodesic_matrix_graph`.
///
/// Arguments
/// ---------
/// - `edges`:    (E, 2) uint32 array of edges (node indices).
/// - `n_nodes`:  Total number of nodes.
/// - `weights`:  (E, ) float32 or float64 edge lengths, or `None` for hop counts. Zero weights
///   are allowed.
/// - `directed`: If `True`, an edge `(u, v)` may only be traversed from `u` to `v`.
/// - `sources`:  (S, ) uint32 source nodes, or `None` for all nodes.
/// - `limit`:    Prune the search at this distance.
/// - `threads`:  Size of the thread pool, or `None` for all cores.
/// - `float64`:  The width for the unweighted case; see [`Float64`].
///
/// Returns
/// -------
/// `(distances, predecessors)`: a (S, n_nodes) matrix in the dtype of `weights`, `-1` where
/// unreachable, and a (S, n_nodes) int32 matrix holding the node before each node on its
/// shortest path back to that row's source (`-1` for the source itself and for unreachable
/// nodes).
#[pyfunction]
#[pyo3(
    name = "geodesic_predecessors",
    signature = (edges, n_nodes, weights=None, directed=false, sources=None, limit=None, threads=None, float64=false)
)]
#[allow(clippy::too_many_arguments)]
pub fn geodesic_predecessors_py<'py>(
    py: Python<'py>,
    edges: PyReadonlyArray2<u32>,
    n_nodes: usize,
    weights: Option<WeightsIn<'py>>,
    directed: bool,
    sources: Option<PyReadonlyArray1<u32>>,
    limit: Option<f64>,
    threads: Option<usize>,
    float64: Float64,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyArray2<i32>>)> {
    let src = as_opt_slice(&sources, "sources")?;

    #[allow(clippy::too_many_arguments)]
    fn run<'py, W: Weight + numpy::Element>(
        py: Python<'py>,
        edges: ArrayView2<u32>,
        n_nodes: usize,
        weights: Option<&ArrayView1<W>>,
        directed: bool,
        sources: Option<&[u32]>,
        limit: Option<f64>,
        threads: Option<usize>,
    ) -> (Bound<'py, PyAny>, Bound<'py, PyArray2<i32>>) {
        let (d, p) = geodesic_predecessors_graph(
            edges,
            n_nodes,
            weights,
            directed,
            sources,
            limit.map(W::from_f64),
            threads,
        );
        (d.into_pyarray(py).into_any(), p.into_pyarray(py))
    }

    let e = edges.as_array();
    Ok(match weights {
        Some(WeightsIn::F32(w)) => {
            let w = w.as_array();
            run(py, e, n_nodes, Some(&w), directed, src, limit, threads)
        }
        Some(WeightsIn::F64(w)) => {
            let w = w.as_array();
            run(py, e, n_nodes, Some(&w), directed, src, limit, threads)
        }
        None if float64 => run::<f64>(py, e, n_nodes, None, directed, src, limit, threads),
        None => run::<f32>(py, e, n_nodes, None, directed, src, limit, threads),
    })
}

/// Node sequences of the shortest paths from one source to each target.
///
/// One search, with the predecessor chains walked in Rust — the per-call overhead this exists
/// to remove. Also stops as soon as the last target settles.
///
/// Arguments
/// ---------
/// - `edges`, `n_nodes`, `weights`, `directed`: as `geodesic_predecessors`.
/// - `source`:  Source node index.
/// - `targets`: (T, ) uint32 target node indices.
///
/// Returns
/// -------
/// A list of `T` 1-D uint32 arrays, ordered source-first / target-last. An unreachable target
/// gives an empty array.
#[pyfunction]
#[pyo3(
    name = "geodesic_path",
    signature = (edges, n_nodes, source, targets, weights=None, directed=false)
)]
pub fn geodesic_path_py<'py>(
    py: Python<'py>,
    edges: PyReadonlyArray2<u32>,
    n_nodes: usize,
    source: u32,
    targets: PyReadonlyArray1<u32>,
    weights: Option<WeightsIn<'py>>,
    directed: bool,
) -> PyResult<Vec<Bound<'py, PyArray1<u32>>>> {
    let tgt = as_slice(&targets, "targets")?;
    let e = edges.as_array();
    // Node ids, so the width is invisible in the output — but not in *which* route wins, which
    // is why this takes the weights at their own width rather than casting to f32.
    let paths = match weights {
        Some(WeightsIn::F32(w)) => {
            geodesic_path_graph(e, n_nodes, Some(&w.as_array()), directed, source, tgt)
        }
        Some(WeightsIn::F64(w)) => {
            geodesic_path_graph(e, n_nodes, Some(&w.as_array()), directed, source, tgt)
        }
        None => geodesic_path_graph::<f32>(e, n_nodes, None, directed, source, tgt),
    };
    Ok(paths.into_iter().map(|p| p.into_pyarray(py)).collect())
}

/// Greedily partition nodes into connected clusters of bounded geodesic radius.
///
/// Each cluster is a ball of radius `max_dist` around its seed, minus whatever earlier
/// clusters already claimed. The radius is the true geodesic distance from the seed, not the
/// length of the walk that reached it.
///
/// Arguments
/// ---------
/// - `edges`:    (E, 2) uint32 array of undirected edges (node indices).
/// - `n_nodes`:  Total number of nodes.
/// - `max_dist`: Maximum distance from a cluster's seed.
/// - `weights`:  (E, ) float32 or float64 edge lengths, or `None` for hop counts.
/// - `seeds`:    (S, ) uint32 preferred seeds, in order of preference. Any node still
///   unassigned afterwards seeds a cluster of its own, in ascending index order.
///
/// Returns
/// -------
/// `(labels, n_clusters)`: `labels` is a 1-D int32 array of contiguous cluster ids in
/// `[0, n_clusters)`, numbered in the order the clusters were grown. The width shows up in
/// *which* nodes fall inside a ball, not in the dtype.
#[pyfunction]
#[pyo3(
    name = "geodesic_clusters",
    signature = (edges, n_nodes, max_dist, weights=None, seeds=None)
)]
pub fn geodesic_clusters_py<'py>(
    py: Python<'py>,
    edges: PyReadonlyArray2<u32>,
    n_nodes: usize,
    max_dist: f64,
    weights: Option<WeightsIn<'py>>,
    seeds: Option<PyReadonlyArray1<u32>>,
) -> PyResult<(Bound<'py, PyArray1<i32>>, usize)> {
    let sd = as_opt_slice(&seeds, "seeds")?;
    let e = edges.as_array();
    let (labels, n) = match weights {
        Some(WeightsIn::F32(w)) => {
            geodesic_clusters(e, n_nodes, f32::from_f64(max_dist), Some(&w.as_array()), sd)
        }
        Some(WeightsIn::F64(w)) => geodesic_clusters(e, n_nodes, max_dist, Some(&w.as_array()), sd),
        None => geodesic_clusters::<f32>(e, n_nodes, f32::from_f64(max_dist), None, sd),
    };
    Ok((labels.into_pyarray(py), n))
}

/// A graph prepared once for many geodesic queries.
///
/// A `#[pyclass]` for the same reason `TpsTransform` is one: the expensive part is the
/// preparation, and the calling pattern is "build once, query thousands of times". Routing
/// `grow` through a free function would rebuild the adjacency — O(E) over the whole graph — on
/// every call, for a query that only ever explores a small ball. On a 160k-vertex mesh tiled
/// into fragments of 64 that is ~2500 full rebuilds against ~2500 tiny searches, which inverts
/// the cost of the algorithm entirely.
///
/// `frozen` is load-bearing rather than cosmetic: the module is `gil_used = false`, so the
/// class must be `Sync`. The search scratch genuinely is mutable state, so it lives behind a
/// lock — uncontended, and nanoseconds against a search. An `RwLock` rather than a `Mutex`
/// because only `grow`, `farthest_seed` and the two component queries actually mutate: the
/// matrix-style queries borrow the graph immutably, and under a free-threaded interpreter
/// there is no reason for two of those to block each other.
#[pyclass(frozen, name = "GeodesicGraph", module = "navis_fastcore._fastcore")]
pub struct PyGeodesicGraph {
    inner: RwLock<GeodesicGraph>,
}

#[pymethods]
impl PyGeodesicGraph {
    /// Prepare a graph. See the Python wrapper for the argument semantics.
    #[new]
    #[pyo3(signature = (edges, n_nodes, weights=None, directed=false, item_nodes=None))]
    fn new(
        edges: PyReadonlyArray2<u32>,
        n_nodes: usize,
        weights: Option<PyReadonlyArray1<f32>>,
        directed: bool,
        item_nodes: Option<PyReadonlyArray1<u32>>,
    ) -> PyResult<Self> {
        let items = as_opt_slice(&item_nodes, "item_nodes")?;
        let w = weights.as_ref().map(|w| w.as_array());
        let inner = GeodesicGraph::new(edges.as_array(), n_nodes, w.as_ref(), directed, items);
        Ok(PyGeodesicGraph {
            inner: RwLock::new(inner),
        })
    }

    /// Number of nodes in the graph.
    #[getter]
    fn n_nodes(&self) -> usize {
        self.inner.read().expect("poisoned").n_nodes()
    }

    /// Number of items. Equals `n_nodes` unless `item_nodes` was given.
    #[getter]
    fn n_items(&self) -> usize {
        self.inner.read().expect("poisoned").n_items()
    }

    /// Grow a connected region of up to `size` items outwards from item `seed`.
    ///
    /// `return_distances` additionally hands back each item's distance to the seed — free from
    /// the search, and what a caller needs to thin the region by radius.
    ///
    /// The GIL is deliberately *not* released: a single call is a small ball, the caller is a
    /// tight Python loop that reacquires immediately, and `forbidden` is borrowed straight out
    /// of numpy — releasing would force a copy of it on every call to buy nothing.
    #[pyo3(signature = (seed, size, forbidden=None, return_distances=false))]
    fn grow<'py>(
        &self,
        py: Python<'py>,
        seed: u32,
        size: usize,
        forbidden: Option<PyReadonlyArray1<bool>>,
        return_distances: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let mut g = self.inner.write().expect("poisoned");
        if (seed as usize) >= g.n_items() {
            return Err(PyValueError::new_err(format!(
                "`seed` is item {seed}, but there are {} items",
                g.n_items()
            )));
        }
        let n_items = g.n_items();
        let f = forbidden
            .as_ref()
            .map(|a| as_flags(a, "forbidden", n_items))
            .transpose()?;
        let (idx, dists) = g.grow(seed, size, f);
        let idx = idx.into_pyarray(py);
        if return_distances {
            Ok((idx, dists.into_pyarray(py)).into_pyobject(py)?.into_any())
        } else {
            Ok(idx.into_any())
        }
    }

    /// Every node within `max_dist` of any source, its distance, and its nearest source.
    ///
    /// As `grow`, the GIL is held: the call is a bounded ball, `sources` is borrowed straight
    /// out of numpy, and the caller is a loop that would reacquire immediately.
    fn ball<'py>(
        &self,
        py: Python<'py>,
        sources: PyReadonlyArray1<u32>,
        max_dist: f32,
    ) -> PyResult<(
        Bound<'py, PyArray1<u32>>,
        Bound<'py, PyArray1<f32>>,
        Bound<'py, PyArray1<u32>>,
    )> {
        // `sources` and `max_dist` are validated by the Python wrapper, as they are for
        // `distances` and `path`, and asserted by the core behind it.
        let mut g = self.inner.write().expect("poisoned");
        let (nodes, dists, srcs) = g.ball(as_slice(&sources, "sources")?, max_dist);
        Ok((
            nodes.into_pyarray(py),
            dists.into_pyarray(py),
            srcs.into_pyarray(py),
        ))
    }

    /// Re-weight edges in place. See the Python wrapper for the semantics.
    fn set_weights(
        &self,
        edges: PyReadonlyArray2<u32>,
        weights: PyReadonlyArray1<f32>,
    ) -> PyResult<()> {
        // Every caller-reachable mistake comes back as one error type, detected in the pass the
        // writes need anyway — so there is nothing to pre-check here. Shape and length are the
        // Python wrapper's job, as everywhere else in this class.
        let mut g = self.inner.write().expect("poisoned");
        g.set_weights(edges.as_array(), weights.as_array())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// The next farthest-point seed. See the Python wrapper for the semantics.
    fn farthest_seed(&self, done: PyReadonlyArray1<bool>) -> PyResult<Option<u32>> {
        let mut g = self.inner.write().expect("poisoned");
        let n_items = g.n_items();
        Ok(g.farthest_seed(as_flags(&done, "done", n_items)?))
    }

    /// Component label of each item.
    fn item_components<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        let mut g = self.inner.write().expect("poisoned");
        g.item_components().into_pyarray(py)
    }

    /// Component label of each node.
    fn components<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        let mut g = self.inner.write().expect("poisoned");
        g.components().into_pyarray(py)
    }

    /// The node each item sits on.
    #[getter]
    fn item_nodes<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        let g = self.inner.read().expect("poisoned");
        g.item_nodes().to_vec().into_pyarray(py)
    }

    /// Pairwise geodesic distances, as `geodesic_matrix_graph`.
    #[pyo3(signature = (sources=None, targets=None, limit=None, threads=None))]
    fn distances<'py>(
        &self,
        py: Python<'py>,
        sources: Option<PyReadonlyArray1<u32>>,
        targets: Option<PyReadonlyArray1<u32>>,
        limit: Option<f32>,
        threads: Option<usize>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let src = as_opt_slice(&sources, "sources")?;
        let tgt = as_opt_slice(&targets, "targets")?;
        let g = self.inner.read().expect("poisoned");
        Ok(g.distances(src, tgt, limit, threads).into_pyarray(py))
    }

    /// Nearest target per source, as `geodesic_nearest_mesh`.
    #[pyo3(signature = (sources=None, targets=None, limit=None, threads=None))]
    fn nearest<'py>(
        &self,
        py: Python<'py>,
        sources: Option<PyReadonlyArray1<u32>>,
        targets: Option<PyReadonlyArray1<u32>>,
        limit: Option<f32>,
        threads: Option<usize>,
    ) -> PyResult<(Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<i32>>)> {
        let src = as_opt_slice(&sources, "sources")?;
        let tgt = as_opt_slice(&targets, "targets")?;
        let g = self.inner.read().expect("poisoned");
        let (d, i) = g.nearest(src, tgt, limit, threads);
        Ok((d.into_pyarray(py), i.into_pyarray(py)))
    }

    /// Farthest target per source, as `geodesic_farthest_mesh`.
    #[pyo3(signature = (sources=None, targets=None, limit=None, threads=None))]
    fn farthest<'py>(
        &self,
        py: Python<'py>,
        sources: Option<PyReadonlyArray1<u32>>,
        targets: Option<PyReadonlyArray1<u32>>,
        limit: Option<f32>,
        threads: Option<usize>,
    ) -> PyResult<(Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<i32>>)> {
        let src = as_opt_slice(&sources, "sources")?;
        let tgt = as_opt_slice(&targets, "targets")?;
        let g = self.inner.read().expect("poisoned");
        let (d, i) = g.farthest(src, tgt, limit, threads);
        Ok((d.into_pyarray(py), i.into_pyarray(py)))
    }

    /// Shortest-path trees, as `geodesic_predecessors`.
    #[pyo3(signature = (sources=None, limit=None, threads=None))]
    fn predecessors<'py>(
        &self,
        py: Python<'py>,
        sources: Option<PyReadonlyArray1<u32>>,
        limit: Option<f32>,
        threads: Option<usize>,
    ) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<i32>>)> {
        let src = as_opt_slice(&sources, "sources")?;
        let g = self.inner.read().expect("poisoned");
        let (d, p) = g.predecessors(src, limit, threads);
        Ok((d.into_pyarray(py), p.into_pyarray(py)))
    }

    /// Shortest-path node sequences, as `geodesic_path`.
    fn path<'py>(
        &self,
        py: Python<'py>,
        source: u32,
        targets: PyReadonlyArray1<u32>,
    ) -> PyResult<Vec<Bound<'py, PyArray1<u32>>>> {
        let tgt = as_slice(&targets, "targets")?;
        let g = self.inner.read().expect("poisoned");
        Ok(g.path(source, tgt)
            .into_iter()
            .map(|p| p.into_pyarray(py))
            .collect())
    }

    /// Radius-bounded clustering, as `geodesic_clusters`.
    #[pyo3(signature = (max_dist, seeds=None))]
    fn clusters<'py>(
        &self,
        py: Python<'py>,
        max_dist: f32,
        seeds: Option<PyReadonlyArray1<u32>>,
    ) -> PyResult<(Bound<'py, PyArray1<i32>>, usize)> {
        let sd = as_opt_slice(&seeds, "seeds")?;
        let g = self.inner.read().expect("poisoned");
        let (labels, n) = g.clusters(max_dist, sd);
        Ok((labels.into_pyarray(py), n))
    }

    /// The induced subgraph on `nodes`, plus the original index of each surviving item.
    fn subset<'py>(
        &self,
        py: Python<'py>,
        nodes: PyReadonlyArray1<u32>,
    ) -> PyResult<(Self, Bound<'py, PyArray1<u32>>)> {
        let keep = as_slice(&nodes, "nodes")?;
        let g = self.inner.read().expect("poisoned");
        // Range and distinctness are checked by `_prep_indices(unique=True)` in the Python
        // wrapper, as for every other node subset this package takes.
        let (sub, kept) = g.subset(keep);
        Ok((
            PyGeodesicGraph {
                inner: RwLock::new(sub),
            },
            kept.into_pyarray(py),
        ))
    }

    fn __repr__(&self) -> String {
        let g = self.inner.read().expect("poisoned");
        format!(
            "GeodesicGraph(n_nodes={}, n_items={})",
            g.n_nodes(),
            g.n_items()
        )
    }
}
