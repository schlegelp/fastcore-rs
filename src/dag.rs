use itertools::Itertools;
use ndarray::{s, Array, Array1, Array2, ArrayView1};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::prelude::Python;
use pyo3::prelude::*;
use std::collections::HashMap;
use std::collections::HashSet;

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

/// Extract leafs from parents.
///
/// Arguments:
///
/// - `parents`: array of parent IDs
/// - `sort_by_dist`: boolean indicating whether to sort the leafs by distance to the root node
/// - `weights`: optional array of weights for each node; if not provided all nodes are assumed
///              to have a weight of 1
///
/// Returns:
///
/// A vector of leaf nodes.
///
fn find_leafs(
    parents: &Array1<i32>,
    sort_by_dist: bool,
    weights: &Option<Array1<f32>>,
) -> Vec<i32> {
    let parents_set: HashSet<_> = parents.iter().cloned().collect();
    let nodes: Vec<i32> = (0..parents.len() as i32).collect();
    let nodes = Array::from(nodes);
    // Find all leaf nodes (i.e. all nodes that are not contained in `parents`)
    let leafs: Vec<_> = nodes
        .iter()
        .filter(|&node| !parents_set.contains(node))
        .cloned()
        .collect();

    if sort_by_dist {
        // Get the distance from each leaf node to the root node
        let dists = all_dists_to_root(&parents, &Some(Array::from(leafs.clone())), weights);

        // Sort `leafs` by `dists` in descending order
        let mut leafs: Vec<_> = leafs.iter().cloned().zip(dists).collect();
        leafs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let leafs: Vec<_> = leafs.iter().map(|(leaf_id, _dist)| *leaf_id).collect();
        leafs
    } else {
        leafs
    }
}

/// Extract branch points from parents.
///
/// Arguments:
///
/// - `parents`: array of parent IDs
///
/// Returns:
///
/// A vector of branch points.
///
fn find_branch_points(parents: &Array1<i32>) -> Vec<i32> {
    let mut branch_points: Vec<i32> = vec![];
    let mut seen: HashSet<i32> = HashSet::new();
    // Go over all nodes and add them to the branch_points vector if they've been seen before
    for node in parents.iter() {
        if seen.contains(node) {
            branch_points.push(*node);
        }
        seen.insert(*node);
    }
    branch_points
}

/// Extract roots from parents.
///
/// Arguments:
///
/// - `parents`: array of parent IDs
///
/// Returns:
///
/// A vector of root nodes.
///
fn find_roots(parents: &Array1<i32>) -> Vec<i32> {
    let mut roots: Vec<i32> = vec![];
    // Go over all nodes and add them to the roots vector
    for (i, parent) in parents.iter().enumerate() {
        if *parent < 0 {
            roots.push(i as i32);
        }
    }
    roots
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

    let weights: Option<Array1<f32>> = if weights.is_some() {
        Some(weights.unwrap().as_array().to_owned())
    } else {
        None
    };

    // Extract leafs from parents
    // N.B. that this also sorts the leafs by distance to the root node!
    let leafs = find_leafs(&x_parents.to_owned(), true, &weights);

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

    let dists: Vec<f32> =
        all_dists_to_root(&parents.as_array().to_owned(), &Some(x_sources), &weights);
    dists.into_pyarray(py)
}

/// Return path length from each node to the root node.
///
/// This is the pure rust implementation for internal use.
///
/// Arguments:
///
/// - `parents`: array of parent IDs
/// - `sources`: optional array of source IDs
/// - `weights`: optional array of weights for each node
///
/// Returns:
///
/// A vector of f32 values indicating the distance between each node and the root.
///
fn all_dists_to_root(
    parents: &Array1<i32>,
    sources: &Option<Array1<i32>>,
    weights: &Option<Array1<f32>>,
) -> Vec<f32> {
    let x_sources: Array1<i32>;
    // If no sources, use all nodes as sources
    if sources.is_none() {
        x_sources = Array::from_iter(0..parents.len() as i32);
    } else {
        x_sources = sources.as_ref().unwrap().clone();
    }

    let mut node: i32;
    let mut dists: Vec<f32> = vec![0.; x_sources.len()];

    if weights.is_none() {
        for i in 0..x_sources.len() {
            node = x_sources[i];
            while node >= 0 {
                dists[i] += 1.;
                node = parents[node as usize];
            }
        }
    } else {
        let weights = weights.as_ref().unwrap();
        for i in 0..x_sources.len() {
            node = x_sources[i];
            while node >= 0 {
                dists[i] += weights[node as usize];
                node = parents[node as usize];
            }
        }
    }

    dists
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

// Return path length from a single node to the root.
// This is the pure rust implementation for internal use.
fn dist_to_root(parents: &ArrayView1<i32>, node: i32) -> f32 {
    let mut dist = 0.;
    let mut node = node;
    while node >= 0 {
        dist += 1.;
        node = parents[node as usize];
    }
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
        dists = geodesic_distances_all_by_all(&parents.as_array().to_owned(), &weights, directed);
    // If sources and/or targets use the partial implementation
    } else {
        dists = geodesic_distances_partial(
            &parents.as_array().to_owned(),
            &sources,
            &targets,
            &weights,
            directed,
        );
    }
    dists.into_pyarray(py)
}

/// Compute geodesic distances between all pairs of nodes
///
/// TODOs:
/// - return condensed distance matrix instead of square matrix when `directed=False`?
/// - use threading
///
/// Arguments:
///
/// - `parents`: array of parent IDs
/// - `weights`: optional array of weights for each node
/// - `directed`: boolean indicating whether to return only the directed (child -> parent) distances
///
/// Returns:
///
/// A 2D array of f32 values indicating the distances between all pairs of nodes.
///
fn geodesic_distances_all_by_all(
    parents: &Array1<i32>,
    weights: &Option<Array1<f32>>,
    directed: bool,
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

    // Walk from each node to the root node.
    // We're basically brute forcing the "forward" (child -> parent) distances here
    // In theory we could be a bit more efficient by using leaf nodes as seeds,
    // tracking the distances as we go along and then filling the missing values.
    // This requires a lot more book keeping though and I'm not (yet) sure that's worth it.
    for idx1 in 0..parents.len() {
        node = idx1; // start with the distance between the node and itself
        d = 0.0;
        // Keep going until we hit the root
        loop {
            dists[[idx1, node]] = d;

            if !directed {
                dists[[node, idx1]] = d; // symmetric matrix
            }

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

    // Return early if we only wanted the directed distances
    if directed {
        return dists;
    }

    // Extract leafs from parents
    let leafs = find_leafs(&parents, true, &None);

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

    // Use permutations to go over all pairs of leafs while avoiding duplicates
    for perm in (0..leafs.len()).permutations(2) {
        l1 = leafs[perm[0]] as usize;
        l2 = leafs[perm[1]] as usize;

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

    dists
}

/// Calculate geodesic distances between a set of sources and target.
///
///  Forward distances:
///   1. From each leaf walk towards the root
///   2. When encountering a source or target node, track the distance
///   3. Use the tracked distances to calculate the distance between sources and targets
///  Reverse distances:
///   4. Starting at the root(s), walk up the tree
///   5. When encountering a branch point get the distances of sources and targets on each of the
///      branches
///   6. Calculate the distances between sources and targets on different branches
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
fn geodesic_distances_partial(
    parents: &Array1<i32>,
    sources: &Option<Array1<i32>>,
    targets: &Option<Array1<i32>>,
    weights: &Option<Array1<f32>>,
    directed: bool,
) -> Array2<f32> {
    // If no sources, use all nodes as sources
    let sources = if sources.is_none() {
        Array::from_iter(0..parents.len() as i32)
    } else {
        sources.as_ref().unwrap().clone()
    };
    // If not targets, use all nodes as targets
    let targets = if targets.is_none() {
        Array::from_iter(0..parents.len() as i32)
    } else {
        targets.as_ref().unwrap().clone()
    };
    // Map row/col indices in `dists` to parent node indices
    // Importantly, we need to do this before turning `sources` and `targets`
    // into a HashSet because that might reorder the nodes
    let source_to_index: HashMap<_, _> = sources
        .iter()
        .enumerate()
        .map(|(i, node)| (*node, i))
        .collect();
    let target_to_index: HashMap<_, _> = targets
        .iter()
        .enumerate()
        .map(|(i, node)| (*node, i))
        .collect();

    // Generate two arrays indicating whether a given node is in sources or targets
    let mut is_source: Array1<bool> = Array::from_elem(parents.len(), false);
    for &node in sources.iter() {
        is_source[node as usize] = true;
    }
    let mut is_target: Array1<bool> = Array::from_elem(parents.len(), false);
    for &node in targets.iter() {
        is_target[node as usize] = true;
    }

    // Prepare the distance matrix filled with -1
    let mut dists: Array2<f32> = Array::from_elem((sources.len(), targets.len()), -1.0);

    // Get a list of leafs
    let leafs = find_leafs(&parents, true, weights);

    // Prepare some more variables
    let mut node: usize;
    let mut d: f32;

    // From each leaf walk towards the root and track the forward distances
    // N.B. We could be a bit more clever by using stop conditions to avoid going over
    // the same part of the tree twice but so far this part doesn't seem to be the bottle neck
    for leaf in leafs.iter() {
        node = *leaf as usize; // start with the distance between the node and itself
        d = 0.0;

        // Prepare vector to track (node, distance) tuples
        let mut this_sources: Vec<(usize, f32)> = vec![];
        let mut this_targets: Vec<(usize, f32)> = vec![];

        // Walk towards the root
        loop {
            // If this node is a source, track the distance
            if is_source[node] {
                this_sources.push((node, d));
            }
            // If this node is a target, track the distance
            if is_target[node] {
                this_targets.push((node, d));
            }

            // Track distance travelled
            d += if weights.is_some() {
                weights.as_ref().unwrap()[node]
            } else {
                1.0
            };

            // Stop if we reached the root
            if parents[node] < 0 {
                break;
            }

            node = parents[node] as usize;
        }

        // Fill in the forward distances
        for (source, d1) in this_sources.iter() {
            for (target, d2) in this_targets.iter() {
                // The distance between the two is the absolute value of the difference
                dists[[
                    source_to_index[&(*source as i32)],
                    target_to_index[&(*target as i32)],
                ]] = (d1 - d2).abs();
            }
        }
    }

    // Return early if we only wanted the directed distances
    if directed {
        return dists;
    }

    // Now the reverse distances

    // Calculate parent -> children direction for each node
    // This is a vector of [(child1, child2, ...), ...]
    let children = extract_parent_child(parents);

    // From each root start walking up the tree and calculate the reverse distances
    for root in find_roots(parents) {
        let (_, _) = walk_up_and_count_recursively(
            root,
            0.0,
            &children,
            &is_target,
            &is_source,
            &weights,
            &mut dists,
            &source_to_index,
            &target_to_index,
        );
    }

    return dists;
}

/// Function to recursively walk up the tree from a given node.
///
/// At each branchpoint we recurse into its children.
///
/// Arguments:
///
/// - `start_node`: the node to start from
/// - `start_dist`: the distance from the start node to its parent (usually the branch point)
/// - `children`: a vector of vectors where each vector contains the children of a node
/// - `is_target`: a boolean array indicating whether a node is a target
/// - `is_source`: a boolean array indicating whether a node is a source
/// - `weights`: an optional array of weights for each node
/// - `dists`: the distance matrix to fill
/// - `source_to_index`: a hashmap mapping node IDs to indices in the distance matrix
/// - `target_to_index`: a hashmap mapping node IDs to indices in the distance matrix
///
/// Returns:
///
/// A tuple of two vectors: the first contains the distances from sources to the root node
/// and the second contains the distances from targets to the root node. The distances are
/// tuples of (index in the distance matrix, distance).
///
fn walk_up_and_count_recursively(
    start_node: i32,
    start_dist: f32,
    children: &Vec<Vec<i32>>,
    is_target: &Array1<bool>,
    is_source: &Array1<bool>,
    weights: &Option<Array1<f32>>,
    dists: &mut Array2<f32>,
    source_to_index: &HashMap<i32, usize>,
    target_to_index: &HashMap<i32, usize>,
) -> (Vec<(usize, f32)>, Vec<(usize, f32)>) {
    // Track the distance and the source/target distances on this subtree
    // These vectors contain (node index into dists array, distance to root)
    let mut source_dists: Vec<(usize, f32)> = vec![];
    let mut target_dists: Vec<(usize, f32)> = vec![];

    // Walk up the tree and track the distances
    let mut node = start_node as usize;
    let mut d: f32 = start_dist;
    loop {
        // If this node is a source, track the distance
        if is_source[node] {
            source_dists.push((source_to_index[&(node as i32)], d));
        }
        // If this node is a target, track the distance
        if is_target[node] {
            target_dists.push((target_to_index[&(node as i32)], d));
        }

        // If this node is a leaf node, we can stop moving up
        if children[node].len() == 0 {
            break;
        }
        // If this node is a branch point, we need to recurse into the children
        if children[node].len() > 1 {
            // We need to keep track of the distances from each of the branches
            let mut branch_sources_dists: Vec<Vec<(usize, f32)>> = vec![];
            let mut branch_targets_dists: Vec<Vec<(usize, f32)>> = vec![];

            for child in children[node].iter() {
                let child_dist = if weights.is_some() {
                    weights.as_ref().unwrap()[*child as usize]
                } else {
                    1.0
                };
                let (child_sources, child_targets) = walk_up_and_count_recursively(
                    *child,
                    child_dist,
                    children,
                    is_target,
                    is_source,
                    weights,
                    dists,
                    source_to_index,
                    target_to_index,
                );
                branch_sources_dists.push(child_sources);
                branch_targets_dists.push(child_targets);
            }

            // Now we know, for each of the branches, which sources and targets are on it and at what distance
            // We can now calculate the distances between all sources and targets on different branches and
            // write that to `dists`
            for (i, branch1) in branch_sources_dists.iter().enumerate() {
                for (j, branch2) in branch_targets_dists.iter().enumerate() {
                    // Skip sources/targets on the same branch
                    if i == j {
                        continue;
                    }
                    for (source_ix, d1) in branch1.iter() {
                        for (target_ix, d2) in branch2.iter() {
                            dists[[*source_ix, *target_ix]] = d1 + d2;
                        }
                    }
                }
            }

            // Next, we need to add these distances to the local source/target distances
            // AND add the current distance
            for (i, branch) in branch_sources_dists.iter().enumerate() {
                for (source_ix, d1) in branch.iter() {
                    source_dists.push((*source_ix, d1 + d));
                }
            }
            for (i, branch) in branch_targets_dists.iter().enumerate() {
                for (target_ix, d2) in branch.iter() {
                    target_dists.push((*target_ix, d2 + d));
                }
            }

            break;
        }

        node = children[node][0] as usize;

        // Track distance travelled
        // N.B. that here we're incrementing the distance AFTER going to the next node
        d += if weights.is_some() {
            weights.as_ref().unwrap()[node]
        } else {
            1.0
        };
    }

    // println!("Start Node: {} (at dist {})", start_node, start_dist);
    // println!(" Finished at: {}", node);
    // println!(" Sources: {:?}", source_dists);
    // println!(" Targets: {:?}", target_dists);

    (source_dists, target_dists)
}

/// Calculate synapse flow centrality for each node.
///
/// This works by, for each connected component:
///  1. Walk from each node with a presynapse to the root node and for each node
///     along the way count the number of presynapses distal to them; from that we
///     can also infer the number of presynapses proximal to them.
///  2. Do the same for nodes with postsynapses.
///  3. For each node, multiply the number of proximal presynapses with the number
///     of distal postsynapses and vice versa. That's the flow centrality for the
///     segment between that node and its parent.
///
/// Arguments:
///
/// - `parents`: array of parent IDs
/// - `presynapses`: array of i32 indicating how many presynapses a given node
///                  is associated with
/// - `postsynapses`: array of i32 indicating how many postsynapses a given node
///                   is a associated with
///
/// Returns:
///
/// An array of u32 values indicating the flow centrality for each node.
///
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
) -> &'py PyArray1<u32> {
    let flow: Array1<u32> = synapse_flow_centrality(
        &parents.as_array(),
        &presynapses.as_array(),
        &postsynapses.as_array(),
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
            seen[node] = true; // global (not reset)
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

/// Compute parent -> childs mapping.
///
/// Arguments:
///
/// - `parents`: array of parent IDs
///
/// Returns:
///
/// A vector of vectors where each vector contains the children of a node.
/// For example parent array `[-1, 0, 1, 1]` would return `[[1], [2, 3], [], []]`.
///
fn extract_parent_child(parents: &Array1<i32>) -> Vec<Vec<i32>> {
    let mut parent_child: Vec<Vec<i32>> = vec![vec![]; parents.len()];
    for (i, parent) in parents.iter().enumerate() {
        if *parent >= 0 {
            parent_child[*parent as usize].push(i as i32);
        }
    }
    parent_child
}
