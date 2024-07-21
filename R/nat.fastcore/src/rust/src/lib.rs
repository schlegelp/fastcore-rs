use extendr_api::prelude::*;
use ndarray::Array1;
use std::collections::HashMap;


/// For each node ID in `parents` find its index in `nodes`.
///
/// Importantly this is 0-indexed to match indexing in Rust.
/// Roots will have parent index -1.
///
/// @export
#[extendr]
pub fn node_indices(
    nodes: Vec<i32>,
    parents: Vec<i32>,
) -> Vec<i32> {
    let mut indices: Vec<i32> = vec![-1; nodes.len()];

    // Create a HashMap where the keys are nodes and the values are indices
    let node_to_index: HashMap<_, _> = nodes
        .iter()
        .enumerate()
        .map(|(index, node)| (*node, index as i32))
        .collect();

    for (i, parent) in parents.iter().enumerate() {
        if *parent < 0 {
            indices[i] = -1;
            continue;
        }
        // Use the HashMap to find the index of the parent node
        if let Some(index) = node_to_index.get(parent) {
            indices[i] = *index;
        }
    }

    indices
}

/// Calculate child -> parent distances.
/// @export
#[extendr]
pub fn child_to_parent_dists(
    parents: Vec<i32>,
    x: Vec<f64>,
    y: Vec<f64>,
    z: Vec<f64>,
) -> Vec<f64> {
    let mut dists: Vec<f64> = vec![0.0; parents.len()];

    for (i, parent) in parents.iter().enumerate() {
        if *parent < 0 {
            continue;
        }
        let dx = x[i] - x[*parent as usize];
        let dy = y[i] - y[*parent as usize];
        let dz = z[i] - z[*parent as usize];
        dists[i] = (dx * dx + dy * dy + dz * dz).sqrt();
    }
    dists
}

/// Compute all distances to root.
/// @export
#[extendr]
pub fn all_dists_to_root(
    parents: Vec<i32>,
    sources: Option<Vec<i32>>,
    weights: Option<Vec<f64>>,  // f64 is used to match R's numeric type
) -> Vec<f32> {
    let parents = Array1::from_vec(parents);
    let sources: Option<Array1<i32>> = if sources.is_none() {
        None
    } else {
        Some(Array1::from_vec(sources.unwrap()))
    };

    let weights: Option<Array1<f32>> = if weights.is_none() {
        None
    } else {
        // Convert f64 to f32
        Some(Array1::from_vec(weights.unwrap().iter().map(|x| *x as f32).collect()))
    };

    fastcore::dag::all_dists_to_root(&parents.view(), &sources, &weights)
}

/// Geodesic distances between nodes.
/// @export
#[extendr]
pub fn geodesic_distances(
    parents: Vec<i32>,
    sources: Option<Vec<i32>>,
    targets: Option<Vec<i32>>,
    weights: Option<Vec<f64>>,
    directed: bool,
) -> Robj {
    let parents = Array1::from_vec(parents);

    let weights: Option<Array1<f64>> = if weights.is_none() {
        None
    } else {
        Some(Array1::from_vec(weights.unwrap()))
    };

    // Unwrap sources and targets and rewrap them as Option
    let sources: Option<Array1<i32>> = if sources.is_none() {
        None
    } else {
        Some(Array1::from_vec(sources.unwrap()))
    };
    let targets: Option<Array1<i32>> = if targets.is_none() {
        None
    } else {
        Some(Array1::from_vec(targets.unwrap()))
    };

    let dists: Array2<f64> = if sources.is_none() && targets.is_none() {
        // If no sources and targets, use the more efficient full implementation
        fastcore::dag::geodesic_distances_all_by_all(&parents.view(), &weights, directed)
    // If sources and/or targets use the partial implementation
    } else {
        fastcore::dag::geodesic_distances_partial(&parents.view(), &sources, &targets, &weights, directed)
    };

    // Note: below conversion to R matrix takes up more than 50% of the time
    dists.try_into().unwrap()
}

/// Calculate Strahler Index.
/// @export
#[extendr]
pub fn strahler_index(
    parents: Vec<i32>,
    greedy: bool,
    to_ignore: Option<Vec<i32>>,
    min_twig_size: Option<i32>,
) -> Robj {
    let parents = Array1::from_vec(parents);
    fastcore::dag::strahler_index(&parents.view(), greedy, &to_ignore, &min_twig_size).try_into().unwrap()
}

/// Connected components.
/// @export
#[extendr]
pub fn connected_components(
    parents: Vec<i32>,
) -> Robj {
    let parents = Array1::from_vec(parents);
    fastcore::dag::connected_components(&parents.view()).try_into().unwrap()
}

/// Prune twigs below given threshold.
///
/// Returns indices of nodes to keep.
///
/// @export
#[extendr]
pub fn prune_twigs(
    parents: Vec<i32>,
    threshold: f64,
    weights: Option<Vec<f64>>,
) -> Vec<i32> {
    let parents = Array1::from_vec(parents);

    let weights: Option<Array1<f64>> = if weights.is_none() {
        None
    } else {
        Some(Array1::from_vec(weights.unwrap()))
    };

    fastcore::dag::prune_twigs(&parents.view(), threshold as f32, &weights)
}

// Macro to generate exports.
// This ensures exported functions are registered with R.
// See corresponding C code in `entrypoint.c`.
extendr_module! {
    mod nat_fastcore;
    fn all_dists_to_root;
    fn node_indices;
    fn geodesic_distances;
    fn strahler_index;
    fn connected_components;
    fn prune_twigs;
    fn child_to_parent_dists;
}
