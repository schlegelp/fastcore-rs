use ndarray::{Array, Array1, Ix1, ArrayView1};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayDyn, PyReadonlyArrayDyn, PyReadonlyArray1};
use pyo3::prelude::{pymodule, PyModule, PyResult, Python};
use pyo3::prelude::*;
use std::collections::HashMap;
use std::collections::HashSet;

// For each node ID in A find its index in B.
//
// Typically `A` will be parent IDs and `B` will be node IDs.
// Negative IDs (= parents of root nodes) will be passed through.
//
// Note that there is no check whether all IDs in A actually exist in B. If
// an ID in A does not exist in B it gets a negative index (i.e. like roots).
#[pyfunction]
fn _node_indices<'py>(py: Python<'py>,
                      nodes: PyReadonlyArray1<i64>,
                      parents: PyReadonlyArray1<i64>) -> &'py PyArray1<i32> {
    let mut indices: Vec<i32> = vec![-1; nodes.len()];
    let x_nodes = nodes.as_array();
    let x_parents = parents.as_array();

    // Create a HashMap where the keys are nodes and the values are indices
    let node_to_index: HashMap<_, _> = x_nodes.iter().enumerate().map(|(index, node)| (*node, index as i32)).collect();

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

// Generate linear segments while maximizing segment lengths (sort of).
// `parents` contains the index of the parent node for each node.
#[pyfunction]
fn _generate_segments(parents: PyReadonlyArray1<i32>) -> Vec<Vec<i32>>  {
    let x_parents = parents.as_array();
    let x_parents_set: HashSet<_> = x_parents.iter().cloned().collect();
    let nodes: Vec<i32> = (0..parents.len() as i32).collect();
    let nodes = Array::from(nodes);
    let mut all_segments: Vec<Vec<i32>> = vec![];
    let mut current_segment = Array::from_elem(x_parents.len(), -1i32);
    let mut seen = Array::from_elem(x_parents.len(), false);
    let mut i: usize;
    let mut node: i32;

    let leafs: Vec<_> = nodes.iter()
        .filter(|&node| !x_parents_set.contains(node))
        .cloned()
        .collect();
    // println!("Found {} leafs among the {} nodes", leafs.len(), nodes.len());

    for (idx, leaf) in leafs.iter().enumerate() {
        // Reset current_segment and counter
        i = 0;
        current_segment.fill(-1);
        // Start with this leaf node
        node = *leaf;

        // println!("Starting segment {} at node {}", idx, node);
        // Iterate until we reach the root node
        while node >= 0 {
            // Add the current node to the current segment
            current_segment[i] = node;

            // Increment counter
            i += 1;

            // Stop if this node has already been seen
            if seen[node as usize] {
                // println!("Stopping segment {} at node {} ({})", idx, node, i);
                break;
            }

            // Mark the current node as seen
            seen[node as usize] = true;

            // Get the parent of the current node
            node = x_parents[node as usize];

        // Keep track of the current segment
        // Note that we're truncating it to exclude -1 values (i.e. empties)
        all_segments.push(current_segment.iter().filter(|&&x| x >= 0).cloned().collect());
        }
    }
    all_segments.sort_by(|a, b| b.len().cmp(&a.len()));
    all_segments
}


// Return path length from each node to the root node.
#[pyfunction]
fn all_dists_to_root<'py>(py: Python<'py>,
                          parents: PyReadonlyArray1<i32>,
                          sources: Option<PyReadonlyArray1<i32>>) -> &'py PyArray1<f32> {
    let x_sources: Array1<i32>;
    // If no sources, use all nodes as sources
    if sources.is_none() {
        x_sources = Array::from_iter(0..parents.len() as i32);
    } else {
        x_sources = sources.unwrap().to_owned_array().into_dimensionality::<Ix1>().unwrap();
    }
    let dists: Vec<f32> = _all_dists_to_root(&parents.as_array().to_owned(), &Some(x_sources));
    dists.into_pyarray(py)
}


// Return path length from each node to the root node.
// This is the pure rust implementation for internal use.
fn _all_dists_to_root(parents: &Array1<i32>,
                      sources: &Option<Array1<i32>>) -> Vec<f32> {
    let x_sources: Array1<i32>;
    // If no sources, use all nodes as sources
    if sources.is_none() {
        x_sources = Array::from_iter(0..parents.len() as i32);
    } else {
        x_sources = sources.as_ref().unwrap().clone();
    }

    let mut node: i32;
    let mut dists: Vec<f32> = vec![0.; x_sources.len()];

    for i in 0..x_sources.len() {
        node = x_sources[i];
        while node >= 0 {
            dists[i] += 1.;
            node = parents[node as usize];
        }
    }
    dists
}


#[pymodule]
#[pyo3(name = "_fastcore")]
fn fastcore(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_node_indices, m)?)?;
    m.add_function(wrap_pyfunction!(_generate_segments, m)?)?;

    Ok(())
}