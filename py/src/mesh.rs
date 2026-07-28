use std::sync::RwLock;

use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use fastcore::mesh::{
    connected_components_graph, contract_vertices, geodesic_clusters, geodesic_farthest_mesh,
    geodesic_matrix_graph, geodesic_matrix_mesh, geodesic_nearest_mesh, geodesic_path_graph,
    geodesic_predecessors_graph, level_set_components, mesh_connected_components,
    minimum_spanning_tree, unique_edges, GeodesicGraph,
};

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
    let src = as_opt_slice(&sources, "sources")?;
    let tgt = as_opt_slice(&targets, "targets")?;
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
    let src = as_opt_slice(&sources, "sources")?;
    let tgt = as_opt_slice(&targets, "targets")?;
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
    let src = as_opt_slice(&sources, "sources")?;
    let tgt = as_opt_slice(&targets, "targets")?;
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
/// An (n_unique, 2) int64 array of `[min, max]` rows, ordered as `unique_edges`.
#[pyfunction]
#[pyo3(name = "contract_vertices", signature = (edges, mapping, threads=None))]
pub fn contract_vertices_py<'py>(
    py: Python<'py>,
    edges: PyReadonlyArray2<u32>,
    mapping: PyReadonlyArray1<u32>,
    threads: Option<usize>,
) -> Bound<'py, PyArray2<i64>> {
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
/// - `weights`:  (E, ) float32 weights, or `None` to treat every edge as equal. Must be
///   finite; negative weights are allowed.
/// - `maximize`: Return the maximum spanning forest instead.
/// - `threads`:  Size of the thread pool, or `None` for all cores.
///
/// Returns
/// -------
/// A 1-D int64 array of row indices into `edges`, ordered by weight.
#[pyfunction]
#[pyo3(
    name = "minimum_spanning_tree",
    signature = (edges, n_nodes, weights=None, maximize=false, threads=None)
)]
pub fn minimum_spanning_tree_py<'py>(
    py: Python<'py>,
    edges: PyReadonlyArray2<u32>,
    n_nodes: usize,
    weights: Option<PyReadonlyArray1<f32>>,
    maximize: bool,
    threads: Option<usize>,
) -> Bound<'py, PyArray1<i64>> {
    let w = weights.as_ref().map(|w| w.as_array());
    minimum_spanning_tree(edges.as_array(), n_nodes, w.as_ref(), maximize, threads).into_pyarray(py)
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
    let src = as_opt_slice(&sources, "sources")?;
    let tgt = as_opt_slice(&targets, "targets")?;
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

/// Shortest-path trees over a graph — distances *and* the route to each node.
///
/// The predecessor-returning counterpart of `geodesic_matrix_graph`.
///
/// Arguments
/// ---------
/// - `edges`:    (E, 2) uint32 array of edges (node indices).
/// - `n_nodes`:  Total number of nodes.
/// - `weights`:  (E, ) float32 edge lengths, or `None` for hop counts. Zero weights are
///   allowed.
/// - `directed`: If `True`, an edge `(u, v)` may only be traversed from `u` to `v`.
/// - `sources`:  (S, ) uint32 source nodes, or `None` for all nodes.
/// - `limit`:    Prune the search at this distance.
/// - `threads`:  Size of the thread pool, or `None` for all cores.
///
/// Returns
/// -------
/// `(distances, predecessors)`: a (S, n_nodes) float32 matrix, `-1` where unreachable, and a
/// (S, n_nodes) int32 matrix holding the node before each node on its shortest path back to
/// that row's source (`-1` for the source itself and for unreachable nodes).
#[pyfunction]
#[pyo3(
    name = "geodesic_predecessors",
    signature = (edges, n_nodes, weights=None, directed=false, sources=None, limit=None, threads=None)
)]
#[allow(clippy::too_many_arguments)]
pub fn geodesic_predecessors_py<'py>(
    py: Python<'py>,
    edges: PyReadonlyArray2<u32>,
    n_nodes: usize,
    weights: Option<PyReadonlyArray1<f32>>,
    directed: bool,
    sources: Option<PyReadonlyArray1<u32>>,
    limit: Option<f32>,
    threads: Option<usize>,
) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<i32>>)> {
    let src = as_opt_slice(&sources, "sources")?;
    let w = weights.as_ref().map(|w| w.as_array());
    let (dists, preds) = geodesic_predecessors_graph(
        edges.as_array(),
        n_nodes,
        w.as_ref(),
        directed,
        src,
        limit,
        threads,
    );
    Ok((dists.into_pyarray(py), preds.into_pyarray(py)))
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
    weights: Option<PyReadonlyArray1<f32>>,
    directed: bool,
) -> PyResult<Vec<Bound<'py, PyArray1<u32>>>> {
    let tgt = as_slice(&targets, "targets")?;
    let w = weights.as_ref().map(|w| w.as_array());
    let paths = geodesic_path_graph(edges.as_array(), n_nodes, w.as_ref(), directed, source, tgt);
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
/// - `weights`:  (E, ) float32 edge lengths, or `None` for hop counts.
/// - `seeds`:    (S, ) uint32 preferred seeds, in order of preference. Any node still
///   unassigned afterwards seeds a cluster of its own, in ascending index order.
///
/// Returns
/// -------
/// `(labels, n_clusters)`: `labels` is a 1-D int32 array of contiguous cluster ids in
/// `[0, n_clusters)`, numbered in the order the clusters were grown.
#[pyfunction]
#[pyo3(
    name = "geodesic_clusters",
    signature = (edges, n_nodes, max_dist, weights=None, seeds=None)
)]
pub fn geodesic_clusters_py<'py>(
    py: Python<'py>,
    edges: PyReadonlyArray2<u32>,
    n_nodes: usize,
    max_dist: f32,
    weights: Option<PyReadonlyArray1<f32>>,
    seeds: Option<PyReadonlyArray1<u32>>,
) -> PyResult<(Bound<'py, PyArray1<i32>>, usize)> {
    let sd = as_opt_slice(&seeds, "seeds")?;
    let w = weights.as_ref().map(|w| w.as_array());
    let (labels, n) = geodesic_clusters(edges.as_array(), n_nodes, max_dist, w.as_ref(), sd);
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
        // Checked here rather than left to the core's assert so a caller mistake surfaces as
        // a `ValueError` like every other argument error on this class, not a panic.
        let mut seen = vec![false; g.n_nodes()];
        for &v in keep {
            if (v as usize) >= g.n_nodes() {
                return Err(PyValueError::new_err(format!(
                    "`nodes` contains node {v}, but n_nodes = {}",
                    g.n_nodes()
                )));
            }
            if std::mem::replace(&mut seen[v as usize], true) {
                return Err(PyValueError::new_err(format!(
                    "`nodes` contains node {v} more than once"
                )));
            }
        }
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
