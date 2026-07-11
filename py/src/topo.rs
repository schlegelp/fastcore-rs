use ndarray::{Array1, Array2, ArrayView2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::Python;
use pyo3::prelude::*;

use fastcore::topo::{reroot_rewire, stitch_fragments};

/// Compute minimal-length bridge edges to reconnect skeleton fragments.
///
/// Arguments:
///
/// - `coords`: `(N, 3)` array of node coordinates.
/// - `components`: `(N, )` array of connected-component labels per node.
/// - `mask`: optional `(N, )` boolean array marking nodes eligible as bridge
///           endpoints. If not provided all nodes are eligible.
/// - `max_dist`: upper bound on the length of any single bridge (use `inf` for
///               no bound).
///
/// Returns:
///
/// A tuple `(edges, distances)` where `edges` is an `(M, 2)` int32 array of node
/// index pairs and `distances` is an `(M, )` float32 array of bridge lengths.
#[pyfunction]
#[pyo3(name = "stitch_fragments", signature = (coords, components, mask, max_dist))]
pub fn stitch_fragments_py<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<f64>,
    components: PyReadonlyArray1<i32>,
    mask: Option<PyReadonlyArray1<bool>>,
    max_dist: f64,
) -> (Bound<'py, PyArray2<i32>>, Bound<'py, PyArray1<f32>>) {
    let mask: Option<Array1<bool>> = mask.map(|m| m.as_array().to_owned());

    let bridges = stitch_fragments(
        &coords.as_array(),
        &components.as_array(),
        &mask,
        max_dist,
    );

    // Split the (a, b, dist) tuples into an (M, 2) edge array and an (M,) dist array.
    let m = bridges.len();
    let mut edges: Array2<i32> = Array2::zeros((m, 2));
    let mut dists: Array1<f32> = Array1::zeros(m);
    for (i, (a, b, d)) in bridges.into_iter().enumerate() {
        edges[[i, 0]] = a;
        edges[[i, 1]] = b;
        dists[i] = d;
    }

    (edges.into_pyarray(py), dists.into_pyarray(py))
}

/// Regenerate a parent array after adding undirected edges (rewire + reroot).
///
/// Arguments:
///
/// - `parents`: `(N, )` array of original parent indices (roots are negative).
/// - `new_edges`: `(M, 2)` int32 array of undirected edges to add.
/// - `root`: preferred root node index; use a negative value to auto-pick.
///
/// Returns:
///
/// A `(N, )` int32 array of new parent indices (roots negative).
#[pyfunction]
#[pyo3(name = "reroot_rewire", signature = (parents, new_edges, root))]
pub fn reroot_rewire_py<'py>(
    py: Python<'py>,
    parents: PyReadonlyArray1<i32>,
    new_edges: PyReadonlyArray2<i32>,
    root: i32,
) -> Bound<'py, PyArray1<i32>> {
    let edges: ArrayView2<i32> = new_edges.as_array();
    let new_parents = reroot_rewire(&parents.as_array(), &edges, root);
    new_parents.into_pyarray(py)
}
