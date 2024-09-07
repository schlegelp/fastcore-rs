use ndarray::{Array, Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::prelude::Python;
use pyo3::prelude::*;
use std::collections::HashMap;

use fastcore::dag::{
    all_dists_to_root, break_segments, dist_to_root, generate_segments,
    geodesic_distances_all_by_all, geodesic_distances_partial, synapse_flow_centrality,
    connected_components, classify_nodes, strahler_index, prune_twigs
};

/// For each node ID in `parents` find its index in `nodes`.
///
/// Notes:
///
/// - negative IDs (= parents of root nodes) will be passed through
/// - there is no check whether all IDs in `parents` actually exist in `nodes`:
///   if an ID in `parents` does not exist in `nodes` it gets a negative index
///
/// Arguments:
///
/// - `nodes`: array of node IDs
/// - `parents`: array of parent IDs
///
/// Returns:
///
/// An array of indices for each node indicating the index of the parent node.
///
#[pyfunction]
pub fn node_indices_64<'py>(
    py: Python<'py>,
    nodes: PyReadonlyArray1<i64>,
    parents: PyReadonlyArray1<i64>,
) -> &'py PyArray1<i32> {
    let mut indices: Vec<i32> = vec![-1; nodes.len().expect("Failed to get length of nodes")];
    let x_nodes = nodes.as_array();
    let x_parents = parents.as_array();

    // Create a HashMap where the keys are nodes and the values are indices
    let node_to_index: HashMap<_, _> = x_nodes
        .iter()
        .enumerate()
        .map(|(index, node)| (*node, index as i32))
        .collect();

    for (i, parent) in x_parents.iter().enumerate() {
        if *parent < 0 {
            indices[i] = -1;
            continue;
        }
        // Use the HashMap to find the index of the parent node
        if let Some(index) = node_to_index.get(parent) {
            indices[i] = *index;
        }
    }

    indices.into_pyarray(py)
}


#[pyfunction]
pub fn node_indices_32<'py>(
    py: Python<'py>,
    nodes: PyReadonlyArray1<i32>,
    parents: PyReadonlyArray1<i32>,
) -> &'py PyArray1<i32> {
    let mut indices: Vec<i32> = vec![-1; nodes.len().expect("Failed to get length of nodes")];
    let x_nodes = nodes.as_array();
    let x_parents = parents.as_array();

    // Create a HashMap where the keys are nodes and the values are indices
    let node_to_index: HashMap<_, _> = x_nodes
        .iter()
        .enumerate()
        .map(|(index, node)| (*node, index as i32))
        .collect();

    for (i, parent) in x_parents.iter().enumerate() {
        if *parent < 0 {
            indices[i] = -1;
            continue;
        }
        // Use the HashMap to find the index of the parent node
        if let Some(index) = node_to_index.get(parent) {
            indices[i] = *index;
        }
    }

    indices.into_pyarray(py)
}

/// Generate linear segments while maximizing segment lengths.
///
/// Arguments:
///
/// - `parents`: array of parent IDs
/// - `weights`: optional array of weights for each node; if not provided all nodes are assumed
///              to have a weight of 1
///
/// Returns:
///
/// A vector of vectors where each vector contains the nodes of a segment.
///
#[pyfunction]
#[pyo3(name = "generate_segments")]
pub fn generate_segments_py(
    parents: PyReadonlyArray1<i32>,
    weights: Option<PyReadonlyArray1<f32>>,
) -> Vec<Vec<i32>> {
    let weights: Option<Array1<f32>> = if weights.is_some() {
        Some(weights.unwrap().as_array().to_owned())
    } else {
        None
    };

    let all_segments = generate_segments(&parents.as_array(), weights);

    all_segments
}

#[pyfunction]
#[pyo3(name = "break_segments")]
pub fn break_segments_py(parents: PyReadonlyArray1<i32>) -> Vec<Vec<i32>> {
    let all_segments = break_segments(&parents.as_array());

    all_segments
}

/// Return path length from each node to the root node.
///
/// This function wrangles the Python arrays into Rust arrays.
///
/// Arguments:
///
/// - `parents`: array of parent IDs
/// - `sources`: optional array of source IDs
/// - `weights`: optional array of weights for each node
///
/// Returns:
///
/// A 1D array of f32 values indicating the distance between each node and the root.
///
#[pyfunction]
#[pyo3(name = "all_dists_to_root")]
pub fn all_dists_to_root_py<'py>(
    py: Python<'py>,
    parents: PyReadonlyArray1<i32>,
    sources: Option<PyReadonlyArray1<i32>>,
    weights: Option<PyReadonlyArray1<f32>>,
) -> &'py PyArray1<f32> {
    let x_sources: Array1<i32>;
    // If no sources, use all nodes as sources
    if sources.is_none() {
        x_sources =
            Array::from_iter(0..parents.len().expect("Failed to get length of parents") as i32);
    } else {
        x_sources = sources.unwrap().as_array().to_owned();
    }
    let weights: Option<Array1<f32>> = if weights.is_some() {
        Some(weights.unwrap().as_array().to_owned())
    } else {
        None
    };

    let dists: Vec<f32> = all_dists_to_root(&parents.as_array(), &Some(x_sources), &weights);
    dists.into_pyarray(py)
}

/// Return path length from a single node to the root.
///
/// Arguments:
///
/// - `parents`: array of parent IDs
/// - `node`: node ID
///
/// Returns:
///
/// A f32 value indicating the distance between the node and the root.
#[pyfunction]
#[pyo3(name = "dist_to_root")]
pub fn dist_to_root_py(parents: PyReadonlyArray1<i32>, node: i32) -> f32 {
    let dist = dist_to_root(&parents.as_array(), node);
    dist
}

/// Compute geodesic distances along the tree.
///
/// This function wrangles the Python arrays into Rust arrays and then calls the
/// appropriate geodesic distance function.
///
/// Arguments:
///
/// - `parents`: array of parent IDs
/// - `sources`: optional array of source IDs
/// - `targets`: optional array of target IDs
/// - `weights`: optional array of weights for each node
/// - `directed`: boolean indicating whether to return only the directed (child -> parent) distances
///
/// Returns:
///
/// A 2D array of f32 values indicating the distances between sources and targets.
///
#[pyfunction]
#[pyo3(name = "geodesic_distances", signature = (parents, sources, targets, weights, directed=false))]
pub fn geodesic_distances_py<'py>(
    py: Python<'py>,
    parents: PyReadonlyArray1<i32>,
    sources: Option<PyReadonlyArray1<i32>>,
    targets: Option<PyReadonlyArray1<i32>>,
    weights: Option<PyReadonlyArray1<f32>>,
    directed: bool,
) -> &'py PyArray2<f32> {
    let weights: Option<Array1<f32>> = if weights.is_some() {
        Some(weights.unwrap().as_array().to_owned())
    } else {
        None
    };

    // Unwrap sources and targets and rewrap them as Option
    let sources = if sources.is_some() {
        Some(sources.unwrap().as_array().to_owned())
    } else {
        None
    };
    let targets = if targets.is_some() {
        Some(targets.unwrap().as_array().to_owned())
    } else {
        None
    };

    let dists: Array2<f32>;
    // If no sources and targets, use the more efficient full implementation
    if sources.is_none() && targets.is_none() {
        dists = geodesic_distances_all_by_all(&parents.as_array(), &weights, directed);
    // If sources and/or targets use the partial implementation
    } else {
        dists =
            geodesic_distances_partial(&parents.as_array(), &sources, &targets, &weights, directed);
    }
    dists.into_pyarray(py)
}

/// Compute synapse flow centrality for each node.
///
/// Arguments:
///
/// - `parents`: array of parent IDs
/// - `presynapses`: array of i32 indicating how many presynapses are associated with
///                  a given node
/// - `postsynapses`: array of i32 indicating how many postsynapses are associated with
///                  a given node
///
/// Returns:
///
/// An array of u32 values indicating the flow centrality for each node.
///
#[pyfunction]
#[pyo3(name = "synapse_flow_centrality")]
pub fn synapse_flow_centrality_py<'py>(
    py: Python<'py>,
    parents: PyReadonlyArray1<i32>,
    presynapses: PyReadonlyArray1<u32>,
    postsynapses: PyReadonlyArray1<u32>,
    mode: String,
) -> &'py PyArray1<u32> {
    let flow: Array1<u32> = synapse_flow_centrality(
        &parents.as_array(),
        &presynapses.as_array(),
        &postsynapses.as_array(),
        mode,
    );
    flow.into_pyarray(py)
}

/// Find connected components in tree.
///
/// Arguments:
///
/// - `parents`: array of parent IDs
///
/// Returns:
///
/// An i32 array of the same length as `parents` where each node is assigned
/// the ID of its root node.
///
#[pyfunction]
#[pyo3(name = "connected_components")]
pub fn connected_components_py<'py>(
    py: Python<'py>,
    parents: PyReadonlyArray1<i32>,
) -> &'py PyArray1<i32> {
    let cc: Array1<i32> = connected_components(&parents.as_array());
    cc.into_pyarray(py)
}

/// Prune terminal twigs below a given size threshold.
///
/// Returns the indices of nodes to keep.
#[pyfunction]
#[pyo3(name = "prune_twigs")]
pub fn prune_twigs_py(
    parents: PyReadonlyArray1<i32>,
    threshold: f32,
    weights: Option<PyReadonlyArray1<f32>>,
) -> Vec<i32> {
    let weights: Option<Array1<f32>> = if weights.is_some() {
        Some(weights.unwrap().as_array().to_owned())
    } else {
        None
    };

    prune_twigs(&parents.as_array(), threshold, &weights)
}

/// Calculate Strahler Index.
///
/// This function wrangles the Python arrays into Rust arrays.
///
/// Arguments:
///
/// - `parents`: array of parent IDs
/// - `method`: "greedy" | "standard"
/// - `to_ignore`: optional array of node IDs to ignore (must be leafs)
/// - `min_twig_size`: optional integer indicating the minimum twig size
///
/// Returns:
///
/// A 1D array of i32 values indicating the Strahler Index for each node.
///
#[pyfunction]
#[pyo3(name = "strahler_index")]
pub fn strahler_index_py<'py>(
    py: Python<'py>,
    parents: PyReadonlyArray1<i32>,
    method: String,
    to_ignore: Option<PyReadonlyArray1<i32>>,
    min_twig_size: Option<i32>,
) -> &'py PyArray1<i32> {
    if method != "standard" && method != "greedy" {
        panic!(
            "Invalid method: {}. Must be either 'standard' or 'greedy'",
            method
        );
    }

    let to_ignore: Option<Vec<i32>> = if to_ignore.is_some() {
        Some(to_ignore.unwrap().as_array().to_owned().to_vec())
    } else {
        None
    };

    let min_twig_size: Option<i32> = if min_twig_size.is_some() {
        Some(min_twig_size.unwrap())
    } else {
        None
    };

    let strahler: Array1<i32> = strahler_index(
        &parents.as_array(),
        method == "greedy",
        &to_ignore,
        &min_twig_size,
    );
    strahler.into_pyarray(py)
}

/// Classify nodes into roots, leaves, branch points and slabs.
///
/// This function wrangles the Python arrays into Rust arrays.
///
/// Arguments:
///
/// - `parents`: array of parent IDs
///
/// Returns:
///
/// A 1D array of integer values indicating the node type:
/// - 0: root
/// - 1: leaf
/// - 2: branch point
/// - 3: slab
///
#[pyfunction]
#[pyo3(name = "classify_nodes")]
pub fn classify_nodes_py<'py>(
    py: Python<'py>,
    parents: PyReadonlyArray1<i32>,
) -> &'py PyArray1<i32> {
    let node_types: Array1<i32> = classify_nodes(&parents.as_array());
    node_types.into_pyarray(py)
}
