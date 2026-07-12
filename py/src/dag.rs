use ndarray::{Array, Array1, Array2, ArrayView1};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::prelude::Python;
use pyo3::prelude::*;
use rustc_hash::FxHashMap;

use fastcore::dag::{
    all_dists_to_root, break_segments, classify_nodes, connected_components, dist_to_root,
    generate_segments, geodesic_distances_all_by_all, geodesic_distances_partial,
    geodesic_farthest, geodesic_nearest, geodesic_pairs, prune_twigs, strahler_index,
    subtree_height, synapse_flow_centrality, has_cycles,
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
/// Shared implementation behind `node_indices_{16,32,64}`.
///
/// This runs on nearly every public call (the Python layer maps IDs to indices before it
/// touches the core), so it is worth more than a `HashMap`. Two strategies:
///
/// - **Dense**: SWC files number their nodes `1..N`, so the ID range is contiguous. A
///   direct lookup table indexed by `id - min` beats any hash and is the common case.
/// - **Sparse**: CATMAID-style arbitrary 64-bit IDs. Fall back to a hash map, but with a
///   fast integer hasher -- the std hasher is SipHash, which we would pay 2N times.
fn node_indices_impl<T>(nodes: &ArrayView1<T>, to_map: &ArrayView1<T>) -> Vec<i32>
where
    T: Copy + Into<i64>,
{
    let n = nodes.len();
    let mut indices: Vec<i32> = vec![-1; to_map.len()];
    if n == 0 {
        return indices;
    }

    let mut min_id = i64::MAX;
    let mut max_id = i64::MIN;
    for &node in nodes.iter() {
        let v: i64 = node.into();
        min_id = min_id.min(v);
        max_id = max_id.max(v);
    }

    // Use i128 so a min/max spanning the whole i64 range cannot overflow. Cap the table
    // at 4x the node count so that one outlier ID can't blow the table up.
    let span = (max_id as i128) - (min_id as i128) + 1;

    if span <= (n as i128) * 4 {
        let mut lut: Vec<i32> = vec![-1; span as usize];
        for (index, &node) in nodes.iter().enumerate() {
            let v: i64 = node.into();
            // Duplicate IDs: last one wins, matching the HashMap this replaces.
            lut[(v - min_id) as usize] = index as i32;
        }
        for (i, &id) in to_map.iter().enumerate() {
            let v: i64 = id.into();
            // Negative IDs (parents of roots) and IDs outside the range stay at -1.
            if v >= 0 && v >= min_id && v <= max_id {
                indices[i] = lut[(v - min_id) as usize];
            }
        }
        return indices;
    }

    let mut node_to_index: FxHashMap<i64, i32> =
        FxHashMap::with_capacity_and_hasher(n, Default::default());
    for (index, &node) in nodes.iter().enumerate() {
        node_to_index.insert(node.into(), index as i32);
    }
    for (i, &id) in to_map.iter().enumerate() {
        let v: i64 = id.into();
        if v < 0 {
            continue;
        }
        if let Some(&index) = node_to_index.get(&v) {
            indices[i] = index;
        }
    }
    indices
}

#[pyfunction]
pub fn node_indices_64<'py>(
    py: Python<'py>,
    nodes: PyReadonlyArray1<i64>,
    to_map: PyReadonlyArray1<i64>,
) -> Bound<'py, PyArray1<i32>> {
    node_indices_impl(&nodes.as_array(), &to_map.as_array()).into_pyarray(py)
}

#[pyfunction]
pub fn node_indices_32<'py>(
    py: Python<'py>,
    nodes: PyReadonlyArray1<i32>,
    to_map: PyReadonlyArray1<i32>,
) -> Bound<'py, PyArray1<i32>> {
    node_indices_impl(&nodes.as_array(), &to_map.as_array()).into_pyarray(py)
}

#[pyfunction]
pub fn node_indices_16<'py>(
    py: Python<'py>,
    nodes: PyReadonlyArray1<i16>,
    to_map: PyReadonlyArray1<i16>,
) -> Bound<'py, PyArray1<i32>> {
    node_indices_impl(&nodes.as_array(), &to_map.as_array()).into_pyarray(py)
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
) -> (Vec<Vec<i32>>, Option<Vec<f32>>) {
    let weights: Option<Array1<f32>> = if weights.is_some() {
        Some(weights.unwrap().as_array().to_owned())
    } else {
        None
    };

    let (all_segments, lengths) = generate_segments(&parents.as_array(), weights);

    (all_segments, lengths)
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
) -> Bound<'py, PyArray1<f32>> {
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
) -> Bound<'py, PyArray2<f32>> {
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

/// Compute the distance to the nearest target for each source.
///
/// This is a memory-efficient companion to `geodesic_distances` that never materialises the
/// full distance matrix, so it scales to several 100k sources/targets.
///
/// Arguments:
///
/// - `parents`: array of parent IDs
/// - `sources`: optional array of source IDs
/// - `targets`: optional array of target IDs
/// - `weights`: optional array of weights for each node
/// - `directed`: boolean indicating whether to only consider targets towards the root
///
/// Returns:
///
/// A tuple `(distances, nearest)` of 1D arrays: the distance to and node index of the nearest
/// target for each source. Sources without a reachable target get `-1.0` / `-1`.
///
#[pyfunction]
#[pyo3(name = "geodesic_nearest", signature = (parents, sources, targets, weights, directed=false))]
pub fn geodesic_nearest_py<'py>(
    py: Python<'py>,
    parents: PyReadonlyArray1<i32>,
    sources: Option<PyReadonlyArray1<i32>>,
    targets: Option<PyReadonlyArray1<i32>>,
    weights: Option<PyReadonlyArray1<f32>>,
    directed: bool,
) -> (Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<i32>>) {
    let weights: Option<Array1<f32>> = if weights.is_some() {
        Some(weights.unwrap().as_array().to_owned())
    } else {
        None
    };

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

    let (dists, nearest) =
        geodesic_nearest(&parents.as_array(), &sources, &targets, &weights, directed);

    (dists.into_pyarray(py), nearest.into_pyarray(py))
}

/// Compute the distance to the farthest target for each source.
///
/// The mirror image of `geodesic_nearest`: same linear-time algorithm, but it keeps the farthest
/// rather than the nearest target. Never materialises the full distance matrix.
///
/// Arguments:
///
/// - `parents`: array of parent IDs
/// - `sources`: optional array of source IDs
/// - `targets`: optional array of target IDs
/// - `weights`: optional array of weights for each node
/// - `directed`: boolean indicating whether to only consider targets towards the root
///
/// Returns:
///
/// A tuple `(distances, farthest)` of 1D arrays: the distance to and node index of the farthest
/// target for each source. Sources without a reachable target get `-1.0` / `-1`.
///
#[pyfunction]
#[pyo3(name = "geodesic_farthest", signature = (parents, sources, targets, weights, directed=false))]
pub fn geodesic_farthest_py<'py>(
    py: Python<'py>,
    parents: PyReadonlyArray1<i32>,
    sources: Option<PyReadonlyArray1<i32>>,
    targets: Option<PyReadonlyArray1<i32>>,
    weights: Option<PyReadonlyArray1<f32>>,
    directed: bool,
) -> (Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<i32>>) {
    let weights: Option<Array1<f32>> = if weights.is_some() {
        Some(weights.unwrap().as_array().to_owned())
    } else {
        None
    };

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

    let (dists, farthest) =
        geodesic_farthest(&parents.as_array(), &sources, &targets, &weights, directed);

    (dists.into_pyarray(py), farthest.into_pyarray(py))
}

/// Compute geodesic distances along the tree for pairs of nodes.
///
/// This function wrangles the Python arrays into Rust arrays and then calls the
/// appropriate geodesic distance function.
///
/// Arguments:
///
/// - `parents`: array of parent indices
/// - `sources`: array of source indices for pairs
/// - `targets`: array of target indices for pairs
/// - `weights`: optional array of weights for each node
/// - `directed`: boolean indicating whether to return only the directed (child -> parent) distances
///
/// Returns:
///
/// A 1D array of f32 values indicating the distances between the pairs of nodes.
///
#[pyfunction]
#[pyo3(name = "geodesic_pairs", signature = (parents, pairs_source, pairs_target, weights, directed=false))]
pub fn geodesic_pairs_py<'py>(
    py: Python<'py>,
    parents: PyReadonlyArray1<i32>,
    pairs_source: PyReadonlyArray1<i32>,
    pairs_target: PyReadonlyArray1<i32>,
    weights: Option<PyReadonlyArray1<f32>>,
    directed: bool,
) -> Bound<'py, PyArray1<f32>> {
    let weights: Option<Array1<f32>> = if weights.is_some() {
        Some(weights.unwrap().as_array().to_owned())
    } else {
        None
    };

    let dists = geodesic_pairs(
        &parents.as_array(),
        &pairs_source.as_array(),
        &pairs_target.as_array(),
        &weights,
        directed,
    );
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
) -> Bound<'py, PyArray1<u32>> {
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
) -> Bound<'py, PyArray1<i32>> {
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
    mask: Option<PyReadonlyArray1<bool>>,
) -> Vec<i32> {
    let weights: Option<Array1<f32>> = if weights.is_some() {
        Some(weights.unwrap().as_array().to_owned())
    } else {
        None
    };
    let mask: Option<Array1<bool>> = if mask.is_some() {
        Some(mask.unwrap().as_array().to_owned())
    } else {
        None
    };


    prune_twigs(&parents.as_array(), threshold, &weights, &mask)
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
) -> Bound<'py, PyArray1<i32>> {
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

/// Calculate the height of the subtree below each node.
///
/// This function wrangles the Python arrays into Rust arrays.
///
/// Arguments:
///
/// - `parents`: array of parent IDs
/// - `weights`: optional array of weights for each child -> parent connection
///
/// Returns:
///
/// A 1D array of f32 values with the height of each node (leafs are 0).
///
#[pyfunction]
#[pyo3(name = "subtree_height", signature = (parents, weights=None))]
pub fn subtree_height_py<'py>(
    py: Python<'py>,
    parents: PyReadonlyArray1<i32>,
    weights: Option<PyReadonlyArray1<f32>>,
) -> Bound<'py, PyArray1<f32>> {
    let weights: Option<Array1<f32>> = weights.map(|w| w.as_array().to_owned());

    let height: Array1<f32> = subtree_height(&parents.as_array(), &weights);
    height.into_pyarray(py)
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
) -> Bound<'py, PyArray1<i32>> {
    let node_types: Array1<i32> = classify_nodes(&parents.as_array());
    node_types.into_pyarray(py)
}

/// Check for cycles in a tree.
///
/// Arguments:
///
/// - `parents`: array of parent IDs
///
/// Returns:
///
/// A boolean indicating whether the tree has cycles.
///
#[pyfunction]
#[pyo3(name = "has_cycles")]
pub fn has_cycles_py(parents: PyReadonlyArray1<i32>) -> bool {
    has_cycles(&parents.as_array())
}