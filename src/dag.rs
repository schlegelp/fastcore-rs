use ndarray::{Array, Array1, Array2, Ix1, ArrayView1};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::prelude::Python;
use std::collections::HashMap;
use std::collections::HashSet;

/// For each node ID in `parents` find its index in `nodes`.
///  - negative IDs (= parents of root nodes) will be passed through
///  - there is no check whether all IDs in `parents` actually exist in `nodes`:
///    if an ID in `parents` does not exist in `nodes` it gets a negative index
///
/// # Arguments
///  * nodes - array of node IDs
///  * parents - array of parent IDs
#[pyfunction]
pub fn node_indices<'py>(
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

// Extract leafs from parents
fn find_leafs(parents: &Array1<i32>, sort_by_dist: bool) -> Vec<i32> {
    let parents_set: HashSet<_> = parents.iter().cloned().collect();
    let nodes: Vec<i32> = (0..parents.len() as i32).collect();
    let nodes = Array::from(nodes);
    // Find all leaf nodes
    let leafs: Vec<_> = nodes
        .iter()
        .filter(|&node| !parents_set.contains(node))
        .cloned()
        .collect();

    if sort_by_dist {
        // Get the distance from each leaf node to the root node
        let dists = all_dists_to_root(&parents, &Some(Array::from(leafs.clone())));

        // Sort `leafs` by `dists` in descending order
        let mut leafs: Vec<_> = leafs.iter().cloned().zip(dists).collect();
        leafs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let leafs: Vec<_> = leafs.iter().map(|(leaf_id, _dist)| *leaf_id).collect();
        leafs
    } else {
        leafs
    }
}

/// Generate linear segments while maximizing segment lengths.
/// `parents` contains the index of the parent node for each node.
#[pyfunction]
pub fn generate_segments(
    parents: PyReadonlyArray1<i32>,
    weights: Option<PyReadonlyArray1<f32>>,
) -> Vec<Vec<i32>> {
    let x_parents = parents.as_array();
    let mut all_segments: Vec<Vec<i32>> = vec![];
    let mut current_segment = Array::from_elem(x_parents.len(), -1i32);
    let mut seen = Array::from_elem(x_parents.len(), false);
    let mut i: usize;
    let mut node: i32;

    let leafs = find_leafs(&x_parents.to_owned(), true);

    // println!("Found {} leafs among the {} nodes", leafs.len(), x_parents.len());
    // println!("Starting with {} segments", all_segments.len());
    for (_idx, leaf) in leafs.iter().enumerate() {
        // Reset current_segment and counter
        i = 0;
        // current_segment.fill(-1);
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
        }

        // Keep track of the current segment
        // Note that we're truncating it to exclude -1 values (i.e. empties)
        // by keeping only the first `i` elements
        all_segments.push(current_segment.slice(s![..i]).iter().cloned().collect());
    }
    // println!("Found {} segments", all_segments.len());
    all_segments.sort_by(|a, b| b.len().cmp(&a.len()));
    all_segments
}

/// Return path length from each node to the root node.
#[pyfunction]
#[pyo3(name = "all_dists_to_root")]
pub fn all_dists_to_root_py<'py>(
    py: Python<'py>,
    parents: PyReadonlyArray1<i32>,
    sources: Option<PyReadonlyArray1<i32>>,
) -> &'py PyArray1<f32> {
    let x_sources: Array1<i32>;
    // If no sources, use all nodes as sources
    if sources.is_none() {
        x_sources =
            Array::from_iter(0..parents.len().expect("Failed to get length of parents") as i32);
    } else {
        x_sources = sources.unwrap().as_array().to_owned();
    }
    let dists: Vec<f32> = all_dists_to_root(&parents.as_array().to_owned(), &Some(x_sources));
    dists.into_pyarray(py)
}

/// Return path length from each node to the root node.
/// This is the pure rust implementation for internal use.
fn all_dists_to_root(parents: &Array1<i32>, sources: &Option<Array1<i32>>) -> Vec<f32> {
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

// Return path length from a single node to the root.
#[pyfunction]
#[pyo3(name = "dist_to_root")]
pub fn dist_to_root_py(parents: PyReadonlyArray1<i32>, node: i32) -> f32 {
    let dist = dist_to_root(&parents.as_array().to_owned(), node);
    dist
}

// Return path length from a single node to the root.
// This is the pure rust implementation for internal use.
fn dist_to_root(parents: &Array1<i32>, node: i32) -> f32 {
    let mut dist = 0.;
    let mut node = node;
    while node >= 0 {
        dist += 1.;
        node = parents[node as usize];
    }
    dist
}

// Compute geodesic distances between all pairs of nodes
#[pyfunction]
#[pyo3(name = "geodesic_distances")]
pub fn geodesic_distances_py<'py>(
    py: Python<'py>,
    parents: PyReadonlyArray1<i32>,
    sources: Option<PyReadonlyArray1<i32>>,
    weights: Option<PyReadonlyArray1<f32>>,
) -> &'py PyArray2<f32> {
    let x_sources: Array1<i32>;
    // If no sources, use all nodes as sources
    if sources.is_none() {
        x_sources =
            Array::from_iter(0..parents.len().expect("Failed to get lenght of parents") as i32);
    } else {
        x_sources = sources
            .unwrap()
            .to_owned_array()
            .into_dimensionality::<Ix1>()
            .unwrap();
    }
    let weights: Option<Array1<f32>> = if weights.is_some() {
        Some(
            weights
                .unwrap()
                .to_owned_array()
                .into_dimensionality::<Ix1>()
                .unwrap(),
        )
    } else {
        None
    };
    let dists: Array2<f32> =
        geodesic_distances(&parents.as_array().to_owned(), &Some(x_sources), &weights);
    dists.into_pyarray(py)
}

// Compute geodesic distances between all pairs of nodes
// This is the pure rust implementation for internal use.
fn geodesic_distances(
    parents: &Array1<i32>,
    sources: &Option<Array1<i32>>,
    weights: &Option<Array1<f32>>,
) -> Array2<f32> {
/*     let x_sources: Array1<i32>;
    // If no sources, use all nodes as sources
    if sources.is_none() {
        x_sources = Array::from_iter(0..parents.len() as i32);
    } else {
        x_sources = sources.as_ref().unwrap().clone();
    }
 */
    let mut node: usize;
    let mut d: f32;
    let mut dists: Array2<f32> = Array::from_elem((parents.len(), parents.len()), -1.0);

    // Extract leafs from parents
    let leafs = find_leafs(&parents, true);

    // Walk from each node to the root node
    for idx1 in 0..parents.len() {
        node = idx1; // start with the distance between the node and itself
        d = 0.0;
        while dists[[idx1, node]] < 0.0 {
            dists[[idx1, node]] = d;
            dists[[node, idx1]] = d;

            // Track distance travelled
            d += if weights.is_some() {
                weights.as_ref().unwrap()[node]
            } else {
                1.0
            };
            if parents[node] < 0 {
                break;
            }

            node = parents[node] as usize;
        }
    }

    // Above, we calculated the "forward" distances but we're still missing
    // the distances between nodes on separate branches. Our approach now is:
    // (1) Go over all pairs of leafs, (2) find the first common branch point and
    // (3) use their respective distances to that branch point to fill in the
    // missing values in the matrix. We'll be using several stop conditions to
    // avoid doing the same work twice!
    // One important note:
    // We can't use threading here because filling the matrix in parallel
    // messes with our stop conditions!
    let mut l1: usize;
    let mut l2: usize;
    let mut node1: usize;
    let mut node2: usize;
    for idx1 in 0..leafs.len() {
        l1 = leafs[idx1] as usize;
        for idx2 in idx1 + 1..leafs.len() {
            l2 = leafs[idx2] as usize;

            // Skip if we already have a value for this pair
            if dists[[l1, l2]] >= 0.0 {
                continue;
            }

            // Find the first common branch point
            node = l2;
            while parents[node] >= 0 && dists[[l1, node]] < 0.0 {
                node = parents[node] as usize;
            }

            // If the dist is still <0 then these two leafs never converge
            if dists[[l1, node]] < 0.0 {
                continue;
            }
            // Now walk towards the common branch point for both leafs and
            // sum up the respective distances to the root nodes
            node1 = l1;
            node2 = l2;
            while node1 != node {
                // Stop early if we meet a node pair we've already visited
                if dists[[node1, node2]] >= 0.0 {
                    break;
                }
                while node2 != node {
                    // Stop early if we meet a node pair we've already visited
                    if dists[[node1, node2]] >= 0.0 {
                        break;
                    }
                    d = dists[[node1, node]] + dists[[node2, node]];
                    dists[[node1, node2]] = d;
                    dists[[node2, node1]] = d;

                    if parents[node2] < 0 {
                        break;
                    }

                    // Move one down on branch 2
                    node2 = parents[node2] as usize;
                }
                if parents[node1] < 0 {
                    break;
                }
                // Move one down on branch 1
                node1 = parents[node1] as usize;
                // Reset position on branch 2
                node2 = l2;
            }
        }
    }

    dists
}

/// Calculate synapse flow centrality for each node
/// This works by, for each connected component:
/// 1. Walk from each node with a presynapse to the root node and for each node
///    along the way count the number of presynapses distal to them; from that we
///    can also infer the number of presynapses proximal to them.
/// 2. Do the same for nodes with postsynapses.
/// 3. For each node, multiply the number of proximal presynapses with the number
///    of distal postsynapses and vice versa. That's the flow centrality for the
///    segment between that node and its parent.
///
/// # Arguments
///  * parents - array of parent IDs
///  * presynapses - array of i32 indicating how many presynapses a given node
///    is associated with
///  * postsynapses - array of i32 indicating how many postsynapses a given node
///    is a associated with
fn synapse_flow_centrality(
    parents: &ArrayView1<i32>,
    presynapses: &ArrayView1<u32>,
    postsynapses: &ArrayView1<u32>,
) -> Array1<u32> {
    let mut node: usize;
    let mut n_pre: u32;
    let mut n_post: u32;
    let mut proximal_presynapses: Array1<u32> = Array::from_elem(parents.len(), 0);
    let mut distal_presynapses: Array1<u32> = Array::from_elem(parents.len(), 0);
    let mut proximal_postsynapses: Array1<u32> = Array::from_elem(parents.len(), 0);
    let mut distal_postsynapses: Array1<u32> = Array::from_elem(parents.len(), 0);
    let mut flow_centrality: Array1<u32> = Array::from_elem(parents.len(), 0);

    // Walk from each node to the root node
    for idx in 0..parents.len() {
        n_pre = presynapses[idx]; // number of presynapses for this node
        n_post = postsynapses[idx]; // number of postsynapses for this node

        // Skip if this node has no presynapses or postsynapses
        if n_pre == 0 && n_post == 0 {
            continue;
        }

        // Walk from this node to the root node and increment the number of
        // presynapses and postsynapses for each node along the way
        node = idx;
        loop {
            distal_presynapses[node] += n_pre;
            distal_postsynapses[node] += n_post;

            // Stop if we reached the root node
            if parents[node] < 0 {
                break;
            }

            node = parents[node] as usize;
        }
    }

    // To account for the fact that the neuron may consist of multiple
    // connected components, we need to calculate the number of total
    // presynapses and postsynapses on each connected component.
    let cc = connected_components(parents);
    let mut cc_presynapses: HashMap<i32, u32> = HashMap::new();
    let mut cc_postsynapses: HashMap<i32, u32> = HashMap::new();

    // Go over nodes and add up presynapses and postsynapses for each
    // connected component
    for idx in 0..parents.len() {
        let cc_id = cc[idx];
        n_pre = presynapses[idx];
        n_post = postsynapses[idx];
        if cc_presynapses.contains_key(&cc_id) {
            cc_presynapses.insert(cc_id, cc_presynapses[&cc_id] + n_pre);
        } else {
            cc_presynapses.insert(cc_id, n_pre);
        }
        if cc_postsynapses.contains_key(&cc_id) {
            cc_postsynapses.insert(cc_id, cc_postsynapses[&cc_id] + n_post);
        } else {
            cc_postsynapses.insert(cc_id, n_post);
        }
    }

    // Next calculate proximal pre- and postsynapses per connected component
    let mut pre_total: u32;
    let mut post_total: u32;
    for idx in 0..parents.len() {
        pre_total = cc_presynapses[&cc[idx]];
        post_total = cc_postsynapses[&cc[idx]];
        proximal_presynapses[idx] = pre_total - distal_presynapses[idx];
        proximal_postsynapses[idx] = post_total - distal_postsynapses[idx];
    }

    // Calculate flow centrality for each node
    for idx in 0..parents.len() {
        flow_centrality[idx] += proximal_presynapses[idx] * distal_postsynapses[idx];
        flow_centrality[idx] += proximal_postsynapses[idx] * distal_presynapses[idx];
    }

    flow_centrality
}

/// Compute synapse flow centrality for each node
#[pyfunction]
#[pyo3(name = "synapse_flow_centrality")]
pub fn synapse_flow_centrality_py<'py>(
    py: Python<'py>,
    parents: PyReadonlyArray1<i32>,
    presynapses: PyReadonlyArray1<u32>,
    postsynapses: PyReadonlyArray1<u32>,
) -> &'py PyArray1<u32> {
    let flow: Array1<u32> = synapse_flow_centrality(
        &parents.as_array(),
        &presynapses.as_array(),
        &postsynapses.as_array(),
    );
    flow.into_pyarray(py)
}

/// Find connected components in graph. Returns an array of the same length as
/// `parents` where each node is assigned the ID of its root node.
fn connected_components(parents: &ArrayView1<i32>) -> Array1<i32> {
    let mut node: usize;
    let mut component: Array1<bool> = Array::from_elem(parents.len(), false);
    let mut components: Array1<i32> = Array::from_elem(parents.len(), 0);
    let mut seen: Array1<bool> = Array::from_elem(parents.len(), false);

    // Walk from each node to the root node
    for idx in 0..parents.len() {
        // Skip if this node has already been seen
        if seen[idx] {
            continue;
        }

        // Reset `component` to all false values
        component.fill(false);

        node = idx;
        loop {
            // Track this node both globally as well as for this component
            seen[node] = true;  // global (not reset)
            component[node] = true; // local (reset at every iteration)

            // Stop if we reached the root node
            if parents[node] < 0 {
                // Set all nodes in this componenent the root node
                for i in 0..parents.len() {
                    if component[i] {
                        components[i] = node as i32;
                    }
                }
                // Stop looping
                break;
            }

            node = parents[node] as usize;
        }
    }

    components
}

// Compute synapse flow centrality for each node
#[pyfunction]
#[pyo3(name = "connected_components")]
pub fn connected_components_py<'py>(
    py: Python<'py>,
    parents: PyReadonlyArray1<i32>
) -> &'py PyArray1<i32> {
    let cc: Array1<i32> =
        connected_components(&parents.as_array());
    cc.into_pyarray(py)
}
