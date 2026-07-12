use itertools::Itertools;
use ndarray::parallel::prelude::*;
use ndarray::{s, Array, Array1, Array2, ArrayView1};
use num::Float;
use rayon::slice::ParallelSlice;
use std::collections::HashMap;
use std::collections::HashSet;
use std::ops::AddAssign;

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
fn find_leafs<T>(
    parents: &ArrayView1<i32>,
    sort_by_dist: bool,
    weights: &Option<Array1<T>>,
) -> Vec<i32>
where
    T: Float + AddAssign,
{
    let parents_set: HashSet<_> = parents.iter().cloned().collect();
    let nodes: Array1<i32> = Array::from_iter(0..parents.len() as i32);
    // Find all leaf nodes (i.e. all nodes that are not contained in `parents`)
    let leafs: Vec<i32> = nodes
        .iter()
        .filter(|&node| !parents_set.contains(node))
        .cloned()
        .collect();

    if sort_by_dist {
        // Get the distance from each leaf node to the root node
        let dists = all_dists_to_root(parents, &Some(Array::from(leafs.clone())), weights);

        // Sort `leafs` by `dists` in descending order
        let mut leafs: Vec<(i32, T)> = leafs.iter().cloned().zip(dists).collect();
        leafs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let leafs: Vec<i32> = leafs.iter().map(|(leaf_id, _dist)| *leaf_id).collect();
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
fn find_branch_points(parents: &ArrayView1<i32>) -> Vec<i32> {
    let mut branch_points: Vec<i32> = vec![];
    let mut seen: Array1<bool> = Array::from_elem(parents.len(), false);
    // Go over all nodes and add them to the branch_points vector if they've been seen before
    for node in parents.iter() {
        // Skip root nodes
        if *node < 0 {
            continue;
        }

        if seen[*node as usize] {
            branch_points.push(*node);
        } else {
            seen[*node as usize] = true;
        }
    }
    branch_points
}

/// Count the number of children per node.
///
/// This will tell us whether a node is a leaf, a branch point or something else.
fn number_of_children(parents: &ArrayView1<i32>) -> Array1<i32> {
    let mut n_children: Array1<i32> = Array::from_elem(parents.len(), 0);
    // Go over all nodes and add them to the branch_points vector if they've been seen before
    for node in parents.iter() {
        // Skip root nodes
        if *node < 0 {
            continue;
        }

        n_children[*node as usize] += 1;
    }
    n_children
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
fn find_roots(parents: &ArrayView1<i32>) -> Vec<i32> {
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
pub fn generate_segments<T>(parents: &ArrayView1<i32>, weights: Option<Array1<T>>) -> (Vec<Vec<i32>>, Option<Vec<T>>)
where
    T: Float + AddAssign + std::iter::Sum + std::fmt::Debug,
{
    let mut all_segments: Vec<Vec<i32>> = vec![];
    let mut current_segment = Array::from_elem(parents.len(), -1i32);
    let mut seen = Array::from_elem(parents.len(), false);
    let mut i: usize;
    let mut node: i32;
    let mut lengths: Option<Vec<T>> = None;

    let weights: Option<Array1<T>> = if weights.is_some() {
        Some(weights.unwrap().to_owned())
    } else {
        None
    };

    // Extract leafs from parents
    // N.B. that this also sorts the leafs by distance to the root node!
    let leafs = find_leafs(parents, true, &weights);

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
            node = parents[node as usize];
        }

        // Keep track of the current segment
        // Note that we're truncating it to exclude -1 values (i.e. empties)
        // by keeping only the first `i` elements
        all_segments.push(current_segment.slice(s![..i]).iter().cloned().collect());
    }
    // println!("Found {} segments", all_segments.len());

    // If no weights, we can just sort by length
    if weights.is_none() {
        all_segments.sort_by(|a, b| b.len().cmp(&a.len()));
    } else {
        // If weights are provided we need to sort by the sum of the weights
        let weights = weights.unwrap();
        let unsorted: Vec<T> = all_segments
            .iter()
            .map(|segment| {
                segment
                    .iter()
                    .map(|&node| weights[node as usize])
                    .sum::<T>()
            })
            .collect();

        // Generate indices for sorting
        let mut indices: Vec<usize> = (0..all_segments.len()).collect();
        // Sort indices by the lengths, longest first
        indices.sort_by(|a, b| unsorted[*b].partial_cmp(&unsorted[*a]).unwrap());

        // Permute segments AND lengths by the same indices. These have to move together:
        // returning `unsorted` as-is (as we used to) leaves `lengths[i]` describing some
        // other segment than `all_segments[i]`.
        all_segments = indices.iter().map(|&i| all_segments[i].clone()).collect();
        lengths = Some(indices.iter().map(|&i| unsorted[i]).collect());
    }
    (all_segments, lengths)
}

/// Break neuron into linear segments connecting leafs, branch points and root(s).
///
pub fn break_segments(parents: &ArrayView1<i32>) -> Vec<Vec<i32>> {
    let mut all_segments: Vec<Vec<i32>> = vec![];
    let mut current_segment = Array::from_elem(parents.len(), -1i32);
    let mut i: usize;
    let mut node: i32;

    // First, figure out which nodes are leafs, branch points and roots
    let mut is_branch_leaf = Array::from_elem(parents.len(), false);

    for leaf in find_leafs(parents, false, &None::<Array1<f32>>) {
        is_branch_leaf[leaf as usize] = true;
    }
    for branch in find_branch_points(parents) {
        is_branch_leaf[branch as usize] = true;
    }

    for idx in 0..parents.len() {
        // If this node is neither a leaf nor a branch point, skip it
        if !is_branch_leaf[idx] {
            continue;
        }
        // Same if this is a root node
        if parents[idx] < 0 {
            continue;
        }

        // Reset current_segment and counter
        i = 0;

        // Iterate until we reach the next branch point or the root node
        node = idx as i32;
        while node >= 0 {
            // Add the current node to the current segment
            current_segment[i] = node;

            // Increment counter
            i += 1;

            // Stop if this node is a branch point and
            // we're not at the start of the segment
            if is_branch_leaf[node as usize] && i > 1 {
                break;
            }

            // Get the parent of the current node
            node = parents[node as usize];
        }

        // Keep track of the current segment
        // Note that we're truncating it to exclude -1 values (i.e. empties)
        // by keeping only the first `i` elements
        all_segments.push(current_segment.slice(s![..i]).iter().cloned().collect());
    }

    all_segments
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
pub fn all_dists_to_root<T>(
    parents: &ArrayView1<i32>,
    sources: &Option<Array1<i32>>,
    weights: &Option<Array1<T>>,
) -> Vec<T>
where
    T: Float + AddAssign,
{
    let x_sources: Array1<i32>;
    // If no sources, use all nodes as sources
    if sources.is_none() {
        x_sources = Array::from_iter(0..parents.len() as i32);
    } else {
        x_sources = sources.as_ref().unwrap().to_owned();
    }

    let n = parents.len();
    let mut dists: Vec<T> = vec![T::zero(); x_sources.len()];

    // Memoize the distance for every node we touch. Without this, sources sharing a
    // path to the root each re-walk that path: on a neuron (a long backbone with many
    // twigs hanging off it) that is O(N * depth), i.e. quadratic. Walking up until we
    // hit an already-solved node and then unwinding makes the total O(nodes touched),
    // and it stays cheap when `sources` is only a small subset.
    // This is the same path-unwind trick `connected_components` uses below.
    let mut memo: Vec<T> = vec![T::zero(); n];
    let mut solved: Vec<bool> = vec![false; n];
    // Node indices are i32, so u32 is plenty and halves this buffer. It only ever grows to
    // the length of the longest unsolved chain, and is reused across sources.
    let mut path: Vec<u32> = Vec::new();

    for i in 0..x_sources.len() {
        let source = x_sources[i] as usize;
        let mut node = source;

        path.clear();
        while !solved[node] && parents[node] >= 0 {
            path.push(node as u32);
            node = parents[node] as usize;
            // Cycles are malformed input (see `has_cycles`) and used to hang here
            // forever. A valid path visits each node at most once, so overshooting `n`
            // means we are going round in circles: bail out rather than spin.
            if path.len() > n {
                break;
            }
        }

        // We stopped at either a root or an already-solved node. A root is distance
        // zero from itself, which is what `memo` is already initialised to.
        solved[node] = true;

        // Unwind, filling in every node on the way back down.
        // `d[node] = d[parent] + w(node)` -- the weight of an edge is stored on its
        // child, which is what the original leaf-to-root accumulation did.
        let mut acc = memo[node];
        match weights.as_ref() {
            None => {
                for &nd in path.iter().rev() {
                    let nd = nd as usize;
                    acc += T::one();
                    memo[nd] = acc;
                    solved[nd] = true;
                }
            }
            Some(w) => {
                for &nd in path.iter().rev() {
                    let nd = nd as usize;
                    acc += w[nd];
                    memo[nd] = acc;
                    solved[nd] = true;
                }
            }
        }

        dists[i] = memo[source];
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
/// A f32 value indicating the distance between the node and the root. This counts edges,
/// so a root is at distance 0 -- matching `all_dists_to_root`.
pub fn dist_to_root(parents: &ArrayView1<i32>, node: i32) -> f32 {
    let mut dist = 0.;
    let mut node = node;
    // N.B. the condition is on the PARENT: we count edges traversed, not nodes visited.
    // This previously read `while node >= 0`, which counted the start node itself and so
    // reported a root as 1.0 and disagreed with `all_dists_to_root` by exactly one.
    while node >= 0 && parents[node as usize] >= 0 {
        dist += 1.;
        node = parents[node as usize];
    }
    dist
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
pub fn geodesic_distances_all_by_all<T>(
    parents: &ArrayView1<i32>,
    weights: &Option<Array1<T>>,
    directed: bool,
) -> Array2<T>
where
    T: Float + AddAssign,
{
    let mut node: usize;
    let mut d: T;
    let mut dists: Array2<T> =
        Array::from_elem((parents.len(), parents.len()), T::from(-1.0).unwrap());

    // Walk from each node to the root node.
    // We're basically brute forcing the "forward" (child -> parent) distances here
    // In theory we could be a bit more efficient by using leaf nodes as seeds,
    // tracking the distances as we go along and then filling the missing values.
    // This requires a lot more bookkeeping though and I'm not (yet) sure that's worth it.
    for idx1 in 0..parents.len() {
        node = idx1; // start with the distance between the node and itself
        d = T::zero();
        // Keep going until we hit the root
        loop {
            dists[[idx1, node]] = d;

            if !directed {
                dists[[node, idx1]] = d; // symmetric matrix
            }

            // Track distance travelled
            d += if let Some(w) = weights {
                w[node]
            } else {
                T::one()
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
    let leafs = find_leafs(parents, true, weights);

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
        while parents[node] >= 0 && dists[[l1, node]] < T::zero() {
            node = parents[node] as usize;
        }

        // If the dist is still <0 then these two leafs never converge
        if dists[[l1, node]] < T::zero() {
            continue;
        }
        // Now walk towards the common branch point for both leafs and
        // sum up the respective distances to the root nodes
        node1 = l1;
        node2 = l2;
        while node1 != node {
            // Stop early if we meet a node pair we've already visited
            if dists[[node1, node2]] >= T::zero() {
                break;
            }
            while node2 != node {
                // Stop early if we meet a node pair we've already visited
                if dists[[node1, node2]] >= T::zero() {
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
pub fn geodesic_distances_partial<T>(
    parents: &ArrayView1<i32>,
    sources: &Option<Array1<i32>>,
    targets: &Option<Array1<i32>>,
    weights: &Option<Array1<T>>,
    directed: bool,
) -> Array2<T>
where
    T: Float + AddAssign,
{
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
    let mut dists: Array2<T> =
        Array::from_elem((sources.len(), targets.len()), T::from(-1.0).unwrap());

    // Both passes below walk parent -> child, so build the child lists once and share them.
    let children = extract_parent_child(parents);
    let roots = find_roots(parents);
    let ctx = PartialCtx {
        children: &children,
        is_source: &is_source,
        is_target: &is_target,
        weights,
        source_to_index: &source_to_index,
        target_to_index: &target_to_index,
    };

    // --- Pass 1: pairs where one node lies on the other's path to the root ---------------
    //
    // Two nodes that share a root-path are necessarily ancestor and descendant, so their
    // distance is simply the difference of their depths.
    //
    // This used to walk from EVERY leaf all the way up to the root, re-treading the shared
    // trunk once per leaf -- O(leafs * depth), which on a neuron (a long backbone with a twig
    // every few nodes) is quadratic. The old comment here conceded as much: "We could be a
    // bit more clever by using stop conditions to avoid going over the same part of the tree
    // twice". It also asked `find_leafs` to sort the leafs by distance, an ordering nothing
    // downstream used.
    //
    // Instead: one depth-first walk per root, carrying the sources and targets seen so far on
    // the current root-path. Each node is visited once and each matrix cell is written once,
    // so the cost is O(N + cells written) -- and the cells are the output, so that is optimal.

    // Sources/targets on the path from the root down to wherever we currently are, with their
    // depth. Both stay sorted by depth (weights are distances, so never negative), which the
    // `directed` early-out in `visit_forward` relies on.
    let mut active_sources: Vec<(usize, Depth)> = vec![];
    let mut active_targets: Vec<(usize, Depth)> = vec![];

    // Explicit DFS stack of (node, next child to descend into, depth from root). Recursing
    // here would blow the stack on a deep neuron, exactly as `walk_up_and_count` used to.
    let mut stack: Vec<(u32, u32, Depth)> = vec![];

    for &root in roots.iter() {
        let root = root as usize;
        visit_forward(
            &ctx,
            root,
            0.0,
            directed,
            &mut active_sources,
            &mut active_targets,
            &mut dists,
        );
        stack.push((root as u32, 0, 0.0));

        while !stack.is_empty() {
            let top = stack.len() - 1;
            let (node, next_child, depth) = stack[top];
            let siblings = ctx.children.children(node as usize);

            if (next_child as usize) < siblings.len() {
                let child = siblings[next_child as usize] as usize;
                stack[top].1 += 1;

                // An edge's weight is stored on its child, so depth(child) = depth(parent) + w.
                let child_depth = depth + edge_weight(&ctx, child);

                visit_forward(
                    &ctx,
                    child,
                    child_depth,
                    directed,
                    &mut active_sources,
                    &mut active_targets,
                    &mut dists,
                );
                stack.push((child as u32, 0, child_depth));
            } else {
                stack.pop();
                // Leaving this node: it is no longer on the path, so drop whatever it added.
                let node = node as usize;
                if is_source[node] {
                    active_sources.pop();
                }
                if is_target[node] {
                    active_targets.pop();
                }
            }
        }
    }

    // Return early if we only wanted the directed distances
    if directed {
        return dists;
    }

    // --- Pass 2: pairs that meet at a branch point (their lowest common ancestor) ---------
    for &root in roots.iter() {
        walk_up_and_count(&ctx, root, &mut dists);
    }

    dists
}

/// Record one node during the forward walk: pair it with every source/target already on the
/// current root-path, then add it to the path itself.
#[allow(clippy::too_many_arguments)]
fn visit_forward<T>(
    ctx: &PartialCtx<T>,
    node: usize,
    depth: Depth,
    directed: bool,
    active_sources: &mut Vec<(usize, Depth)>,
    active_targets: &mut Vec<(usize, Depth)>,
    dists: &mut Array2<T>,
) where
    T: Float + AddAssign,
{
    let node_is_source = ctx.is_source[node];
    let node_is_target = ctx.is_target[node];
    if !node_is_source && !node_is_target {
        return;
    }

    let row = if node_is_source {
        Some(ctx.source_to_index[&(node as i32)])
    } else {
        None
    };
    let col = if node_is_target {
        Some(ctx.target_to_index[&(node as i32)])
    } else {
        None
    };

    // This node as a source, against every target above it. Those are all strict ancestors,
    // so the target is upstream of the source and `directed` accepts every one of them --
    // no filtering needed, and every iteration writes a cell.
    if let Some(row) = row {
        for (target_ix, target_depth) in active_targets.iter() {
            dists[[row, *target_ix]] = narrow((depth - *target_depth).abs());
        }
    }

    // This node as a target, against every source above it. Now the roles are reversed: the
    // sources are *upstream* of the target, which is the direction `directed` rejects.
    // `active_sources` is sorted by depth, so walking it backwards lets us stop at the first
    // strictly shallower source rather than scanning the whole path (which would put the
    // quadratic straight back in).
    if let Some(col) = col {
        if directed {
            for (source_ix, source_depth) in active_sources.iter().rev() {
                if *source_depth < depth {
                    break;
                }
                dists[[*source_ix, col]] = narrow((*source_depth - depth).abs());
            }
        } else {
            for (source_ix, source_depth) in active_sources.iter() {
                dists[[*source_ix, col]] = narrow((*source_depth - depth).abs());
            }
        }
    }

    // A node that is both a source and a target is zero away from itself.
    if let (Some(row), Some(col)) = (row, col) {
        dists[[row, col]] = T::zero();
    }

    if let Some(row) = row {
        active_sources.push((row, depth));
    }
    if let Some(col) = col {
        active_targets.push((col, depth));
    }
}

/// Calculate the distance to the nearest target for each source.
///
/// This is a memory-efficient companion to `geodesic_distances_partial`: instead of
/// materialising the full `sources x targets` distance matrix (which is infeasible for
/// several 100k sources/targets) it only keeps, for each source, the distance to and the
/// node index of the *nearest* target.
///
/// The implementation is a linear-time (O(N)) rerooting tree DP, so it scales to very large
/// neurons in both time and memory. A source that is itself a target is matched to the
/// nearest *other* (distinct) target, never to itself.
///
/// Arguments:
///
/// - `parents`: array of parent IDs
/// - `sources`: optional array of source IDs (defaults to all nodes)
/// - `targets`: optional array of target IDs (defaults to all nodes)
/// - `weights`: optional array of weights for each child -> parent connection
/// - `directed`: if `true` only consider targets reachable by walking towards the root
///               (i.e. proper ancestors of the source)
///
/// Returns:
///
/// A tuple `(distances, nearest)` where `distances[i]` is the distance from source `i` to its
/// nearest target and `nearest[i]` is that target's node index. Sources without any reachable
/// (distinct) target get `-1.0` / `-1` respectively. Both are ordered to match the order of
/// `sources`. Ties are broken arbitrarily but deterministically.
///
pub fn geodesic_nearest<T>(
    parents: &ArrayView1<i32>,
    sources: &Option<Array1<i32>>,
    targets: &Option<Array1<i32>>,
    weights: &Option<Array1<T>>,
    directed: bool,
) -> (Array1<T>, Array1<i32>)
where
    T: Float + AddAssign,
{
    geodesic_extreme(parents, sources, targets, weights, directed, false)
}

/// Calculate the distance to the farthest target for each source.
///
/// This is the mirror image of `geodesic_nearest`: same linear-time (O(N)) rerooting tree DP,
/// but it keeps, for each source, the distance to and the node index of the *farthest* target
/// instead of the nearest one. A source that is itself a target is matched to the farthest
/// *other* (distinct) target, never to itself.
///
/// Arguments:
///
/// - `parents`: array of parent IDs
/// - `sources`: optional array of source IDs (defaults to all nodes)
/// - `targets`: optional array of target IDs (defaults to all nodes)
/// - `weights`: optional array of weights for each child -> parent connection
/// - `directed`: if `true` only consider targets reachable by walking towards the root (i.e.
///               proper ancestors of the source); with non-negative weights the farthest such
///               target is the target ancestor closest to the root
///
/// Returns:
///
/// A tuple `(distances, farthest)` where `distances[i]` is the distance from source `i` to its
/// farthest target and `farthest[i]` is that target's node index. Sources without any reachable
/// (distinct) target get `-1.0` / `-1` respectively. Both are ordered to match the order of
/// `sources`. Ties are broken arbitrarily but deterministically.
///
pub fn geodesic_farthest<T>(
    parents: &ArrayView1<i32>,
    sources: &Option<Array1<i32>>,
    targets: &Option<Array1<i32>>,
    weights: &Option<Array1<T>>,
    directed: bool,
) -> (Array1<T>, Array1<i32>)
where
    T: Float + AddAssign,
{
    geodesic_extreme(parents, sources, targets, weights, directed, true)
}

/// Shared implementation behind `geodesic_nearest` and `geodesic_farthest`.
///
/// The two are the very same DP and differ only in which end of the distance spectrum we keep,
/// so the objective is parametrised by:
///
/// - a `sentinel` that is, by construction, the *worst* possible value ( `+inf` when minimising,
///   `-inf` when maximising). Plain comparisons against it therefore work unchanged and "did we
///   find a target?" is simply `x.is_finite()`.
/// - a `better(a, b)` comparison (`a < b` when minimising, `a > b` when maximising).
///
/// Careful: wherever "the parent is itself a target" competes with an already-known candidate it
/// must be *merged*, not blindly assigned. A target sitting on the node is at distance zero -
/// always the best candidate when minimising, always the worst one when maximising.
///
/// Arguments:
///
/// - `parents`: array of parent IDs
/// - `sources`: optional array of source IDs (defaults to all nodes)
/// - `targets`: optional array of target IDs (defaults to all nodes)
/// - `weights`: optional array of weights for each child -> parent connection
/// - `directed`: if `true` only consider targets reachable by walking towards the root
/// - `farthest`: if `true` keep the farthest target instead of the nearest one
///
/// Returns:
///
/// A tuple `(distances, target)` ordered like `sources`. Sources without any reachable (distinct)
/// target get `-1.0` / `-1`.
///
fn geodesic_extreme<T>(
    parents: &ArrayView1<i32>,
    sources: &Option<Array1<i32>>,
    targets: &Option<Array1<i32>>,
    weights: &Option<Array1<T>>,
    directed: bool,
    farthest: bool,
) -> (Array1<T>, Array1<i32>)
where
    T: Float + AddAssign,
{
    let n = parents.len();

    // If no sources, use all nodes as sources
    let sources = if sources.is_none() {
        Array::from_iter(0..n as i32)
    } else {
        sources.as_ref().unwrap().clone()
    };
    // If no targets, use all nodes as targets
    let targets = if targets.is_none() {
        Array::from_iter(0..n as i32)
    } else {
        targets.as_ref().unwrap().clone()
    };

    // Boolean mask indicating whether a node is a target
    let mut is_target: Array1<bool> = Array::from_elem(n, false);
    for &node in targets.iter() {
        is_target[node as usize] = true;
    }

    // The sentinel is the worst possible value for our objective, so a value is "real" (i.e. we
    // did find a target) exactly when it is finite.
    let sentinel = if farthest {
        T::neg_infinity()
    } else {
        T::infinity()
    };
    let zero = T::zero();
    let better = |a: T, b: T| -> bool {
        if farthest {
            a > b
        } else {
            a < b
        }
    };

    // Edge weight of the connection from `node` to its parent
    let weight = |node: usize| -> T {
        if weights.is_some() {
            weights.as_ref().unwrap()[node]
        } else {
            T::one()
        }
    };

    // Output arrays, ordered like `sources`
    let mut out_dist: Array1<T> = Array::from_elem(sources.len(), T::from(-1.0).unwrap());
    let mut out_tgt: Array1<i32> = Array::from_elem(sources.len(), -1i32);

    // Build a BFS order from the roots so that every parent precedes its children. We use this
    // order for both passes instead of recursion to avoid stack overflows on deep (100k+) chains.
    let children = extract_parent_child(parents);
    let roots = find_roots(parents);
    let mut order: Vec<usize> = Vec::with_capacity(n);
    for &r in roots.iter() {
        order.push(r as usize);
    }
    let mut head = 0;
    while head < order.len() {
        let node = order[head];
        head += 1;
        for &c in children.children(node).iter() {
            order.push(c as usize);
        }
    }

    // ---- Directed case: best target among proper ancestors ----
    if directed {
        let mut ddir_dist: Vec<T> = vec![sentinel; n];
        let mut ddir_tgt: Vec<i32> = vec![-1i32; n];

        // Pre-order: parents before children
        for &node in order.iter() {
            let p = parents[node];
            if p < 0 {
                continue; // root has no ancestors
            }
            let p = p as usize;
            // Best target at `p` or above it. `p` itself has to *compete* here: a target sitting
            // on `p` is at distance zero, which only wins if nothing better is reachable further
            // up (for `nearest` that is always, for `farthest` only if there is nothing at all).
            let mut cd = ddir_dist[p];
            let mut ct = ddir_tgt[p];
            if is_target[p] && better(zero, cd) {
                cd = zero;
                ct = p as i32;
            }
            if cd.is_finite() {
                ddir_dist[node] = cd + weight(node);
                ddir_tgt[node] = ct;
            }
        }

        for (idx, &s) in sources.iter().enumerate() {
            let s = s as usize;
            if ddir_dist[s].is_finite() {
                out_dist[idx] = ddir_dist[s];
                out_tgt[idx] = ddir_tgt[s];
            }
        }
        return (out_dist, out_tgt);
    }

    // ---- Undirected case: rerooting DP ----

    // `down1`: best target within the subtree of a node (including the node itself).
    let mut down1_dist: Vec<T> = vec![sentinel; n];
    let mut down1_tgt: Vec<i32> = vec![-1i32; n];
    // Best/second-best child contribution `down1[child] + w(child)` at each node. We keep the
    // best child's node index so a child can exclude its own contribution when reading siblings.
    let mut cbest_dist: Vec<T> = vec![sentinel; n];
    let mut cbest_tgt: Vec<i32> = vec![-1i32; n];
    let mut cbest_child: Vec<i32> = vec![-1i32; n];
    let mut csec_dist: Vec<T> = vec![sentinel; n];
    let mut csec_tgt: Vec<i32> = vec![-1i32; n];

    // Post-order: children before parents (reverse BFS order)
    for &v in order.iter().rev() {
        let mut d1d = sentinel;
        let mut d1t = -1i32;
        if is_target[v] {
            // `d1d` is still the sentinel here, so this is an initialisation rather than a merge
            // and needs no comparison (unlike the `up` pass below).
            d1d = zero;
            d1t = v as i32;
        }

        let mut cb_d = sentinel;
        let mut cb_t = -1i32;
        let mut cb_c = -1i32;
        let mut cs_d = sentinel;
        let mut cs_t = -1i32;

        for &c in children.children(v).iter() {
            let c = c as usize;
            if !down1_dist[c].is_finite() {
                continue; // no target in this child's subtree
            }
            let val = down1_dist[c] + weight(c);
            let t = down1_tgt[c];
            if better(val, cb_d) {
                cs_d = cb_d;
                cs_t = cb_t;
                cb_d = val;
                cb_t = t;
                cb_c = c as i32;
            } else if better(val, cs_d) {
                cs_d = val;
                cs_t = t;
            }
        }

        // down1 = best(self, best child contribution)
        if better(cb_d, d1d) {
            d1d = cb_d;
            d1t = cb_t;
        }

        down1_dist[v] = d1d;
        down1_tgt[v] = d1t;
        cbest_dist[v] = cb_d;
        cbest_tgt[v] = cb_t;
        cbest_child[v] = cb_c;
        csec_dist[v] = cs_d;
        csec_tgt[v] = cs_t;
    }

    // `up`: best target outside the subtree of a node, reaching it through its parent.
    let mut up_dist: Vec<T> = vec![sentinel; n];
    let mut up_tgt: Vec<i32> = vec![-1i32; n];

    // Pre-order: parents before children
    for &p in order.iter() {
        for &c in children.children(p).iter() {
            let c = c as usize;
            // Distance from `p` to the best target that is not inside the subtree of `c`. Those
            // targets fall into three disjoint groups: outside the subtree of `p`, `p` itself,
            // and `c`'s sibling subtrees. As in the directed pass, `p` itself has to compete
            // rather than overwrite.
            let mut cd = up_dist[p];
            let mut ct = up_tgt[p];
            if is_target[p] && better(zero, cd) {
                cd = zero;
                ct = p as i32;
            }
            // Targets in `p`'s other subtrees: best child contribution that is not `c`
            let (sd, st) = if cbest_child[p] != c as i32 {
                (cbest_dist[p], cbest_tgt[p])
            } else {
                (csec_dist[p], csec_tgt[p])
            };
            if better(sd, cd) {
                cd = sd;
                ct = st;
            }
            if cd.is_finite() {
                up_dist[c] = cd + weight(c);
                up_tgt[c] = ct;
            }
        }
    }

    // Final answer per source: best *other* target = best(best-in-subtree-excl-self, outside).
    // Reading `cbest` (and not `down1`) is what keeps a source that is itself a target from
    // matching itself.
    for (idx, &s) in sources.iter().enumerate() {
        let s = s as usize;
        let mut bd = cbest_dist[s];
        let mut bt = cbest_tgt[s];
        if better(up_dist[s], bd) {
            bd = up_dist[s];
            bt = up_tgt[s];
        }
        if bd.is_finite() {
            out_dist[idx] = bd;
            out_tgt[idx] = bt;
        }
    }

    (out_dist, out_tgt)
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
/// The read-only context shared by every step of the walk below. Bundled into a struct
/// purely to keep the argument list sane.
struct PartialCtx<'a, T> {
    children: &'a ChildList,
    is_source: &'a Array1<bool>,
    is_target: &'a Array1<bool>,
    weights: &'a Option<Array1<T>>,
    source_to_index: &'a HashMap<i32, usize>,
    target_to_index: &'a HashMap<i32, usize>,
}

/// Depths are carried in f64 whatever `T` is, and every distance is derived as a difference of
/// two of them.
///
/// That difference is unavoidable: an algorithm that writes each matrix cell exactly once must
/// reconstruct distances from precomputed depths instead of re-accumulating them per pair --
/// re-accumulating is precisely what made this quadratic. But a difference of two depths is a
/// cancellation, and in f32 that costs real accuracy: deep inside a neuron the depths are large
/// while the distance between two nearby nodes is small, so the leading digits cancel and the
/// answer is left with whatever noise the f32 depths carried. Accumulating and subtracting in
/// f64, then narrowing to `T`, keeps ~9 significant digits in hand -- comfortably enough for the
/// f32 result to come out correctly rounded.
type Depth = f64;

/// One pending branch point on the DFS stack -- i.e. one frame of what used to be a
/// recursive call.
struct Frame {
    /// The branch point this segment ran into.
    branch_node: usize,
    /// Depth of `branch_node`, measured from the root.
    branch_depth: Depth,
    /// Sources/targets sitting on this segment itself, with their depth from the root.
    source_dists: Vec<(usize, Depth)>,
    target_dists: Vec<(usize, Depth)>,
    /// Which child of `branch_node` we descend into next.
    next_child: usize,
    /// Union of the sources/targets of every child subtree finished so far.
    acc_sources: Vec<(usize, Depth)>,
    acc_targets: Vec<(usize, Depth)>,
}

/// Fold `other` into `acc`, keeping whichever is already bigger as the destination.
///
/// This "small to large" rule is what stops the roll-up being quadratic. Because the entries
/// carry depths measured from the *root*, merging is pure concatenation -- nothing has to be
/// rewritten -- so we are free to pick whichever direction moves fewer elements. An entry only
/// ever moves when it is in the smaller half, which at least doubles the size of the list it
/// lands in, so it can move at most log2(N) times in total.
fn merge_small_to_large<T>(acc: &mut Vec<T>, mut other: Vec<T>) {
    if acc.len() < other.len() {
        std::mem::swap(acc, &mut other);
    }
    acc.append(&mut other);
}

/// Walk a single unbranched segment from `start_node` downwards, collecting the sources and
/// targets on it, each tagged with its depth from the root.
///
/// Returns `Some((branch_node, depth))` if the segment ended at a branch point, or `None` if
/// it ran into a leaf.
#[allow(clippy::type_complexity)]
fn walk_segment<T>(
    ctx: &PartialCtx<T>,
    start_node: i32,
    start_depth: Depth,
) -> (
    Vec<(usize, Depth)>,
    Vec<(usize, Depth)>,
    Option<(usize, Depth)>,
)
where
    T: Float + AddAssign,
{
    let mut source_dists: Vec<(usize, Depth)> = vec![];
    let mut target_dists: Vec<(usize, Depth)> = vec![];

    let mut node = start_node as usize;
    let mut d: Depth = start_depth;

    loop {
        if ctx.is_source[node] {
            source_dists.push((ctx.source_to_index[&(node as i32)], d));
        }
        if ctx.is_target[node] {
            target_dists.push((ctx.target_to_index[&(node as i32)], d));
        }

        let n_children = ctx.children.children(node).len();

        // Leaf: nothing below us.
        if n_children == 0 {
            return (source_dists, target_dists, None);
        }
        // Branch point: the caller has to fan out into the children.
        if n_children > 1 {
            return (source_dists, target_dists, Some((node, d)));
        }

        node = ctx.children.children(node)[0] as usize;

        // N.B. we increment the depth AFTER moving, i.e. an edge's weight is stored on
        // the child.
        d += edge_weight(ctx, node);
    }
}

/// Depth contributed by the edge above `node`, in f64. An edge's weight is stored on its child.
#[inline]
fn edge_weight<T>(ctx: &PartialCtx<T>, node: usize) -> Depth
where
    T: Float + AddAssign,
{
    match ctx.weights {
        Some(w) => w[node].to_f64().expect("weight is not representable as f64"),
        None => 1.0,
    }
}

/// Narrow an f64 distance back down to the matrix's element type.
#[inline]
fn narrow<T: Float>(d: Depth) -> T {
    T::from(d).expect("distance is not representable in the output type")
}

/// Absorb one finished child subtree into its branch point.
///
/// Every source in this child and every target in a *previously* absorbed child meet at this
/// branch point, so this is where their distance becomes known -- and the only place it is
/// written. Pairing each child against the running union (rather than against every other
/// child in turn) means the work is proportional to the cells actually written, instead of
/// quadratic in the number of children.
fn absorb_child<T>(
    frame: &mut Frame,
    child_sources: Vec<(usize, Depth)>,
    child_targets: Vec<(usize, Depth)>,
    dists: &mut Array2<T>,
) where
    T: Float + AddAssign,
{
    let branch_depth = frame.branch_depth;

    // Subtract the branch depth from each side *before* adding, rather than computing
    // `depth_s + depth_t - 2 * branch_depth`: each difference is a real path length, so this
    // keeps the magnitudes small and the cancellation as mild as possible.
    for (source_ix, source_depth) in child_sources.iter() {
        for (target_ix, target_depth) in frame.acc_targets.iter() {
            dists[[*source_ix, *target_ix]] =
                narrow((*source_depth - branch_depth) + (*target_depth - branch_depth));
        }
    }
    for (target_ix, target_depth) in child_targets.iter() {
        for (source_ix, source_depth) in frame.acc_sources.iter() {
            dists[[*source_ix, *target_ix]] =
                narrow((*source_depth - branch_depth) + (*target_depth - branch_depth));
        }
    }

    merge_small_to_large(&mut frame.acc_sources, child_sources);
    merge_small_to_large(&mut frame.acc_targets, child_targets);
}

/// Walk the tree below `root` and fill in the distance for every source/target pair whose
/// paths meet at a branch point (their lowest common ancestor).
///
/// This used to recurse once per branch point. A neuron whose backbone branches at nearly
/// every node therefore recursed as deep as it had nodes and **blew the stack** -- a hard
/// segfault, not an error, on inputs of ~45k chained branch points. `geodesic_extreme`
/// already went iterative for exactly this reason; this is the same treatment. The DFS
/// state now lives in an explicit heap stack, so depth costs memory rather than crashing.
///
/// It also used to be quadratic: every source and target was copied into a fresh vector at
/// every branch point on its way to the root, because the distances were stored relative to
/// each segment's start and so had to be rewritten on each hop. Storing depths from the root
/// instead makes the roll-up a plain concatenation, which in turn allows the small-to-large
/// merge in `merge_small_to_large` -- so an entry moves O(log N) times instead of once per
/// branch point above it.
fn walk_up_and_count<T>(ctx: &PartialCtx<T>, root: i32, dists: &mut Array2<T>)
where
    T: Float + AddAssign,
{
    let (source_dists, target_dists, branch) = walk_segment(ctx, root, 0.0);

    // A root whose segment runs straight into a leaf has no branch point, so there is no
    // pair whose LCA lies below it: nothing to do.
    let Some((branch_node, branch_depth)) = branch else {
        return;
    };

    let mut stack: Vec<Frame> = vec![Frame {
        branch_node,
        branch_depth,
        source_dists,
        target_dists,
        next_child: 0,
        acc_sources: vec![],
        acc_targets: vec![],
    }];

    while let Some(frame) = stack.last_mut() {
        let siblings = ctx.children.children(frame.branch_node);

        if frame.next_child < siblings.len() {
            let child = siblings[frame.next_child];
            frame.next_child += 1;

            let child_depth = frame.branch_depth + edge_weight(ctx, child as usize);

            let (source_dists, target_dists, branch) = walk_segment(ctx, child, child_depth);

            match branch {
                // The child's segment ran into a leaf, so its subtree is fully described by
                // what we just collected: fold it straight into the branch point.
                None => {
                    let frame = stack.last_mut().unwrap();
                    absorb_child(frame, source_dists, target_dists, dists);
                }
                // The child branches again: descend. It gets folded into us when its own
                // frame pops, which happens before we move on to the next sibling.
                Some((branch_node, branch_depth)) => stack.push(Frame {
                    branch_node,
                    branch_depth,
                    source_dists,
                    target_dists,
                    next_child: 0,
                    acc_sources: vec![],
                    acc_targets: vec![],
                }),
            }
            continue;
        }

        // Every child of this branch point has been folded in. All that is left is to add the
        // nodes on this frame's own segment. Those sit *above* the branch point, so they are
        // ancestors of everything below it and their pairs were already handled by the forward
        // pass -- they are not cross-paired here, only carried upward.
        let mut frame = stack.pop().unwrap();
        merge_small_to_large(&mut frame.acc_sources, std::mem::take(&mut frame.source_dists));
        merge_small_to_large(&mut frame.acc_targets, std::mem::take(&mut frame.target_dists));

        // Hand the finished subtree to our parent branch point (if any).
        if let Some(parent) = stack.last_mut() {
            absorb_child(parent, frame.acc_sources, frame.acc_targets, dists);
        }
    }
}

/// Compute geodesic distances between pairs of nodes.
///
///
/// Arguments:
///
/// - `parents`: array of parent indices
/// - `pair_source`: array of source indices
/// - `pair_target`: array of target indices
/// - `weights`: optional array of weights for each child -> parent connection
/// - `directed`: boolean indicating whether to return only the directed (child -> parent) distances
///
/// Returns:
///
/// A 1d array of f32/f64 values indicating the distances between the queried pairs.
///
pub fn geodesic_pairs(
    parents: &ArrayView1<i32>,
    pairs_source: &ArrayView1<i32>,
    pairs_target: &ArrayView1<i32>,
    weights: &Option<Array1<f32>>,
    directed: bool,
) -> Array1<f32> {
    // Make sure we have even number of sources/targets
    if pairs_source.len() != pairs_target.len() {
        panic!("Length of sources and targets not matching!");
    }

    // Convert `pairs_source` to a vector for parallel processing
    let pairs_source: Vec<i32> = pairs_source.iter().cloned().collect();
    let pairs_target: Vec<i32> = pairs_target.iter().cloned().collect();

    // Split the pairs into exactly one chunk per worker, and give each chunk one set of
    // scratch buffers that it reuses for every pair it handles.
    //
    // Two traps here. Allocating an N-element array *per pair* (the original) costs O(N) per
    // pair while the walk itself is only O(depth), so on a large neuron the allocation *is*
    // the runtime. But `map_init` is not the fix: rayon calls its initialiser once per
    // work-split, not once per thread, so it quietly keeps far more N-sized buffers alive
    // than there are threads (measured 45 MB vs 17 MB at N=200k). Chunking explicitly bounds
    // the number of live buffers to the thread count.
    let n_chunks = rayon::current_num_threads().max(1);
    let chunk_size = pairs_source.len().div_ceil(n_chunks).max(1);

    let chunks: Vec<Vec<f32>> = pairs_source
        .par_chunks(chunk_size)
        .zip(pairs_target.par_chunks(chunk_size))
        .map(|(sources, targets)| {
            let mut seen = vec![-1.0f32; parents.len()];
            let mut touched: Vec<u32> = Vec::new();

            sources
                .iter()
                .zip(targets.iter())
                .map(|(idx1, idx2)| {
                    geodesic_distances_single_pair(
                        parents,
                        *idx1 as usize,
                        *idx2 as usize,
                        weights,
                        directed,
                        &mut seen,
                        &mut touched,
                    )
                })
                .collect()
        })
        .collect();

    // Convert the vector to an array and return
    Array::from(chunks.concat())
}

/// Distance between a single pair, using caller-owned scratch buffers.
///
/// `seen` must arrive filled with -1.0 and is left that way again on return; `touched`
/// records which entries we dirtied so we only have to reset those.
fn geodesic_distances_single_pair(
    parents: &ArrayView1<i32>,
    idx1: usize,
    idx2: usize,
    weights: &Option<Array1<f32>>,
    directed: bool,
    seen: &mut [f32],
    touched: &mut Vec<u32>,
) -> f32 {
    let dist = walk_pair(parents, idx1, idx2, weights, directed, seen, touched);

    // Hand the buffer back clean for the next pair on this thread. Resetting the
    // O(depth) entries we actually wrote is what makes reusing it sound *and* cheap.
    for &node in touched.iter() {
        seen[node as usize] = -1.0;
    }
    touched.clear();

    dist
}

fn walk_pair(
    parents: &ArrayView1<i32>,
    idx1: usize,
    idx2: usize,
    weights: &Option<Array1<f32>>,
    directed: bool,
    seen: &mut [f32],
    touched: &mut Vec<u32>,
) -> f32 {
    // Walk from idx1 to root node
    let mut node: usize = idx1;
    let mut d: f32 = 0.0;

    loop {
        // If come across the target node, return here
        // (also happens if idx1 == idx2)
        if node == idx2 {
            return d;
        };
        seen[node] = d;
        touched.push(node as u32);

        // Break if we hit the root node
        if parents[node] < 0 {
            // If we reached the root node without finding idx2 and we want only
            // directed distances, then we return -1
            if directed {
                return -1.0;
            };

            // Now do the same for the target node
            node = idx2;
            d = 0.0;

            loop {
                // If this node has already been visited in when walking from idx1 to the root
                // we can just sum up the distances.
                // This also covers cases where `idx2` is upstream `idx1`!
                if seen[node] > -1.0 {
                    return d + seen[node];
                }

                // Track distance travelled
                d += if let Some(w) = weights { w[node] } else { 1.0 };

                // If we hit the root node again, then idx1 and idx2 are on disconnected
                // branches. Report that as -1, the same "unreachable" sentinel used by the
                // directed case above and by `geodesic_distances_all_by_all`. This used to
                // return 1.0, which is indistinguishable from a genuine one-edge distance.
                if parents[node] < 0 {
                    return -1.0;
                }

                node = parents[node] as usize;
            }
        }
        // Track distance travelled
        d += if let Some(w) = weights { w[node] } else { 1.0 };

        node = parents[node] as usize;
    }
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
/// - `mode`: string indicating whether to calculate the "centrifugal", "centripetal"
///           or the "sum" of both flow centrality
///
/// Returns:
///
/// An array of u32 values indicating the flow centrality for each node.
///
pub fn synapse_flow_centrality(
    parents: &ArrayView1<i32>,
    presynapses: &ArrayView1<u32>,
    postsynapses: &ArrayView1<u32>,
    mode: String,
) -> Array1<u32> {
    // Translate `mode` to integer
    let mode_int: u32;
    if mode == "centrifugal" {
        mode_int = 0;
    } else if mode == "centripetal" {
        mode_int = 1;
    } else if mode == "sum" {
        mode_int = 2;
    } else {
        panic!(
            "Invalid mode: {}. Must be either 'centrifugal', 'centripetal' or 'sum'",
            mode
        );
    }

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
    let mut cc_id: i32;
    for idx in 0..parents.len() {
        cc_id = cc[idx];
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
        if mode_int == 0 {
            flow_centrality[idx] = proximal_postsynapses[idx] * distal_presynapses[idx];
        } else if mode_int == 1 {
            flow_centrality[idx] = proximal_presynapses[idx] * distal_postsynapses[idx];
        } else {
            flow_centrality[idx] = proximal_presynapses[idx] * distal_postsynapses[idx]
                + proximal_postsynapses[idx] * distal_presynapses[idx];
        }
    }

    flow_centrality
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
pub fn connected_components(parents: &ArrayView1<i32>) -> Array1<i32> {
    let mut components: Array1<i32> = Array::from_elem(parents.len(), -1i32);

    for idx in 0..parents.len() {
        if components[idx] >= 0 {
            continue;
        }

        let mut path: Vec<usize> = vec![];
        let mut node = idx;

        loop {
            if components[node] >= 0 {
                // Already assigned — propagate its component back down the path
                let cc_id = components[node];
                for &p in path.iter() {
                    components[p] = cc_id;
                }
                break;
            }

            path.push(node);

            if parents[node] < 0 {
                // Hit root — assign all path nodes to this root
                let cc_id = node as i32;
                for &p in path.iter() {
                    components[p] = cc_id;
                }
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
/// A `ChildList` giving `children(node) -> &[i32]`.
/// For example parent array `[-1, 0, 1, 1]` yields `[1]`, `[2, 3]`, `[]`, `[]`.
///
fn extract_parent_child(parents: &ArrayView1<i32>) -> ChildList {
    ChildList::new(parents)
}

/// Children of each node, in one flat allocation (a CSR/adjacency layout).
///
/// The obvious `Vec<Vec<i32>>` costs one heap allocation *per node* -- ~40-50 bytes each
/// once you count the `Vec` header, the allocation itself and allocator overhead, so
/// ~10 MB for a 200k-node neuron to hold 800 KB of actual data -- and it scatters the
/// children across the heap so every lookup is a pointer chase. Two flat vectors hold the
/// same thing in 8 bytes per node, contiguously.
struct ChildList {
    /// `offsets[node]..offsets[node + 1]` is the slice of `flat` holding node's children.
    offsets: Vec<u32>,
    flat: Vec<i32>,
}

impl ChildList {
    fn new(parents: &ArrayView1<i32>) -> Self {
        let n = parents.len();

        // Count children per node, then prefix-sum into offsets.
        let mut offsets: Vec<u32> = vec![0; n + 1];
        for parent in parents.iter() {
            if *parent >= 0 {
                offsets[*parent as usize + 1] += 1;
            }
        }
        for i in 0..n {
            offsets[i + 1] += offsets[i];
        }

        // Scatter the children into place. `cursor` tracks the next free slot per node;
        // children therefore land in ascending node order, matching the old `push` order.
        let mut flat: Vec<i32> = vec![0; offsets[n] as usize];
        let mut cursor: Vec<u32> = offsets[..n].to_vec();
        for (child, parent) in parents.iter().enumerate() {
            if *parent >= 0 {
                let slot = &mut cursor[*parent as usize];
                flat[*slot as usize] = child as i32;
                *slot += 1;
            }
        }

        ChildList { offsets, flat }
    }

    #[inline]
    fn children(&self, node: usize) -> &[i32] {
        &self.flat[self.offsets[node] as usize..self.offsets[node + 1] as usize]
    }
}

/// Prune terminal twigs below a given size threshold.
///
/// Returns the indices of nodes to keep.
pub fn prune_twigs<T>(
    parents: &ArrayView1<i32>,
    threshold: f32,
    weights: &Option<Array1<T>>,
    mask: &Option<Array1<bool>>,
) -> Vec<i32>
where
    T: Float + AddAssign,
{
    let mut d: T;
    let mut node: i32;
    let mut keep: Array1<bool> = Array::from_elem(parents.len(), true);
    let mut twig: Vec<i32> = vec![];
    let threshold = T::from(threshold).unwrap();

    let n_children = number_of_children(parents);

    // Iterate over leaf nodes
    for _node in 0..parents.len() {
        node = _node as i32;
        // Skip if not a leaf node
        if n_children[node as usize] > 0 {
            continue;
        }

        // Skip leaf nodes that are not in the mask
        if mask.is_some() && !mask.as_ref().unwrap()[node as usize] {
            continue;
        }

        // Reset distance and twig
        d = T::zero();
        twig.clear();
        while node >= 0 {
            // Stop if this twig is already above threshold
            if d > threshold {
                break;
            }

            // Stop if this nodes is masked out
            if mask.is_some() && !mask.as_ref().unwrap()[node as usize] {
                break;
            }

            // Stop if this node has more than one child (i.e. it's a branch point)
            if n_children[node as usize] > 1 {
                break;
            }

            // Track this node
            twig.push(node);

            // Move on to the next node
            d += if weights.is_some() {
                weights.as_ref().unwrap()[node as usize]
            } else {
                T::one()
            };
            node = parents[node as usize];
        }

        // Mark twig nodes for removal
        if d <= threshold {
            for node in twig.iter() {
                keep[*node as usize] = false;
            }
        }
    }

    // Return indices of nodes to keep
    let mut keep_indices: Vec<i32> = vec![];
    for (i, k) in keep.iter().enumerate() {
        if *k {
            keep_indices.push(i as i32);
        }
    }
    keep_indices
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
pub fn strahler_index(
    parents: &ArrayView1<i32>,
    greedy: bool,
    to_ignore: &Option<Vec<i32>>,
    min_twig_size: &Option<i32>,
) -> Array1<i32> {
    let n = parents.len();

    // Vector for the Strahler indices
    let mut strahler: Array1<i32> = Array::from_elem(n, 0);

    // Child counts give us both leafs (no children) and branch points (2+ children) in one pass.
    let n_children = number_of_children(parents);
    let is_branch_point = |node: usize| n_children[node] >= 2;

    // Twigs that don't contribute to the index and instead inherit the index of their first
    // branch point. Contains tuples of (leaf, first branch point).
    let mut to_backfill: Vec<(usize, usize)> = vec![];
    // Leafs whose twig is excluded from the calculation.
    let mut is_ignored: Vec<bool> = vec![false; n];

    let ignore_set: Option<HashSet<i32>> =
        to_ignore.as_ref().map(|ids| ids.iter().cloned().collect());

    for leaf in 0..n {
        if n_children[leaf] != 0 {
            continue;
        }

        // Leafs the caller asked us to ignore. Note we deliberately do *not* treat the root as a
        // branch point here: a twig that reaches the root without passing one has nothing to
        // inherit from and simply stays at 0.
        if ignore_set.as_ref().is_some_and(|s| s.contains(&(leaf as i32))) {
            is_ignored[leaf] = true;

            let mut node = leaf;
            while parents[node] >= 0 {
                if is_branch_point(node) {
                    to_backfill.push((leaf, node));
                    break;
                }
                node = parents[node] as usize;
            }
            continue;
        }

        // Twigs shorter than `min_twig_size`. Unlike above, a root *does* count as a branch point
        // here. `d` counts nodes from the leaf up to and including the branch point.
        if let Some(min_size) = min_twig_size {
            let mut node = leaf;
            let mut d: i32 = 1;
            loop {
                if is_branch_point(node) {
                    if d < *min_size {
                        is_ignored[leaf] = true;
                        to_backfill.push((leaf, node));
                    }
                    break;
                }
                if parents[node] < 0 {
                    break;
                }
                node = parents[node] as usize;
                d += 1;
            }
        }
    }

    // A node contributes to the index if at least one contributing leaf sits below it. Every node
    // between a leaf and its first branch point has exactly one child, so the nodes of an ignored
    // twig are precisely the ones this misses -- we get them for free and needn't track them
    // separately. Each node is marked at most once, so this stays O(N) overall.
    let mut contributes: Vec<bool> = vec![false; n];
    for leaf in 0..n {
        if n_children[leaf] != 0 || is_ignored[leaf] {
            continue;
        }
        let mut node = leaf;
        while !contributes[node] {
            contributes[node] = true;
            if parents[node] < 0 {
                break;
            }
            node = parents[node] as usize;
        }
    }

    // Number of contributing children per node; doubles as the countdown telling us when a node
    // has seen all its children and can be finalised.
    let mut pending: Vec<i32> = vec![0; n];
    for node in 0..n {
        if contributes[node] && parents[node] >= 0 {
            pending[parents[node] as usize] += 1;
        }
    }

    // What the children have carried up so far. "standard" needs the highest index among the
    // children and how many of them carry it; "greedy" simply sums them, which is what makes
    // converging branches always increment.
    let mut child_max: Vec<i32> = vec![0; n];
    let mut child_max_count: Vec<i32> = vec![0; n];
    let mut child_sum: Vec<i32> = vec![0; n];

    // Walk the tree bottom-up, children before parents. A contributing node with children always
    // has at least one contributing child, so the only nodes starting with nothing pending are the
    // contributing leafs.
    let mut queue: Vec<usize> = (0..n)
        .filter(|&node| contributes[node] && n_children[node] == 0)
        .collect();

    while let Some(node) = queue.pop() {
        let si = if n_children[node] == 0 {
            1 // leafs start at 1
        } else if greedy {
            child_sum[node]
        } else if child_max_count[node] >= 2 {
            child_max[node] + 1 // two or more children tie for the highest index -> increment
        } else {
            child_max[node] // otherwise the highest index carries through unchanged
        };
        strahler[node] = si;

        if parents[node] < 0 {
            continue;
        }
        let parent = parents[node] as usize;

        if greedy {
            child_sum[parent] += si;
        } else if si > child_max[parent] {
            child_max[parent] = si;
            child_max_count[parent] = 1;
        } else if si == child_max[parent] {
            child_max_count[parent] += 1;
        }

        pending[parent] -= 1;
        if pending[parent] == 0 {
            queue.push(parent);
        }
    }

    // Fill the ignored twigs with the Strahler index of their first branch point.
    for (leaf, bp) in to_backfill.iter() {
        let si = strahler[*bp];

        let mut node = *leaf;
        loop {
            strahler[node] = si;

            if is_branch_point(node) || parents[node] < 0 {
                break;
            }

            node = parents[node] as usize;
        }
    }

    strahler
}

/// Calculate the height of the subtree below each node.
///
/// A node's height is the geodesic distance from it *down* to the farthest leaf below it. Leafs
/// therefore have a height of 0, and a root carries the length of the longest root-to-leaf path in
/// its component.
///
/// This accumulates the height straight from the leafs rather than subtracting two distances to
/// the root (`height = depth(deepest leaf below) - depth(node)`). The direct form needs no root
/// distances at all, and it keeps every addition small: the subtraction cancels two ~1e6 nm depths
/// against each other, which at f32 leaves ~0.1 nm of error in a quantity callers then compare
/// against a threshold.
///
/// Weights are assumed to be non-negative, as everywhere else in this module.
///
/// See also `geodesic_farthest`, which answers a *different* question: its `directed` mode looks
/// towards the root, and its undirected mode can leave the subtree entirely.
///
/// Arguments:
///
/// - `parents`: array of parent IDs
/// - `weights`: optional array of weights for each child -> parent connection; if not provided
///   all connections are assumed to have a weight of 1
///
/// Returns:
///
/// A 1D array with the height of each node.
///
pub fn subtree_height<T>(parents: &ArrayView1<i32>, weights: &Option<Array1<T>>) -> Array1<T>
where
    T: Float + AddAssign,
{
    let n = parents.len();
    let mut height: Array1<T> = Array::from_elem(n, T::zero());

    // The same Kahn-style countdown `strahler_index` uses: a node is ready to be folded into its
    // parent once every one of its children has been folded into *it*, and the nodes that start out
    // ready are exactly the leafs. Iterative rather than recursive because a neuron can be a 100k
    // node chain. Cycle-safe as a side effect: a node on a cycle never reaches a pending count of
    // zero, so it is never visited and keeps its zero instead of hanging the sweep.
    let mut pending: Vec<i32> = number_of_children(parents).to_vec();
    let mut queue: Vec<usize> = (0..n).filter(|&node| pending[node] == 0).collect();

    while let Some(node) = queue.pop() {
        let parent = parents[node];
        if parent < 0 {
            continue; // a root has nothing above it to carry its height into
        }
        let parent = parent as usize;

        // The weight of the node -> parent edge is stored on the *child*, matching
        // `all_dists_to_root`'s `d[node] = d[parent] + w[node]`. Note we never read a root's own
        // weight, which is what makes the NaN `parent_dist` leaves there harmless.
        let candidate = height[node]
            + match weights.as_ref() {
                Some(w) => w[node],
                None => T::one(),
            };

        // `height` starts at zero, so this is really `max(0, max over the children)`. With
        // non-negative weights that is the plain maximum.
        if candidate > height[parent] {
            height[parent] = candidate;
        }

        pending[parent] -= 1;
        if pending[parent] == 0 {
            queue.push(parent);
        }
    }

    height
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
pub fn classify_nodes(parents: &ArrayView1<i32>) -> Array1<i32> {
    // A node's type is fully determined by its number of children plus whether it is a root,
    // so two linear passes suffice. Note that roots take priority: a root can also be a leaf
    // (no children) or a branch point (2+ children) but is always reported as a root.
    let n_children = number_of_children(parents);

    Array1::from_iter(
        parents
            .iter()
            .zip(n_children.iter())
            .map(|(&parent, &n)| match (parent, n) {
                (p, _) if p < 0 => 0, // root
                (_, 0) => 1,          // leaf
                (_, 1) => 3,          // slab
                _ => 2,               // branch point
            }),
    )
}


/// Check if a tree has cycles
///
/// Arguments:
///
/// - `parents`: array of parent IDs
///
/// Returns:
///
/// A boolean indicating whether the tree has cycles
///
pub fn has_cycles(parents: &ArrayView1<i32>) -> bool {
    // Standard three-colour walk:
    //   0 = unvisited, 1 = on the path we are currently walking, 2 = known cycle-free.
    //
    // The state is what makes this linear. Two things used to make it quadratic: a
    // full O(N) reset of the `seen` array once per node (which ran even for nodes that
    // were about to be skipped), and the absence of any stop condition *inside* the
    // walk -- so on a topologically sorted input (i.e. every SWC file) no node was ever
    // pruned and every one of them walked all the way to the root.
    const UNVISITED: u8 = 0;
    const ON_PATH: u8 = 1;
    const CYCLE_FREE: u8 = 2;

    // One byte per node is the entire footprint: because a node has at most one parent, the
    // "path" we are walking is just a chain we can re-walk to mark, so we never have to
    // store it. (The old code kept two full bool arrays.)
    let mut state: Vec<u8> = vec![UNVISITED; parents.len()];

    for idx in 0..parents.len() {
        if state[idx] != UNVISITED {
            continue;
        }

        // Walk up, marking the chain as we go.
        let mut node = idx;
        let mut looped = false;
        loop {
            match state[node] {
                // Walked into a node already on the chain we are building: that is a cycle
                // (a self-loop is just the length-1 case).
                ON_PATH => {
                    looped = true;
                    break;
                }
                // Walked into a node we already cleared, so everything above it is clear
                // too. This is the stop condition the old code lacked, and it is what makes
                // the total work O(N) instead of O(N * depth).
                CYCLE_FREE => break,
                _ => {}
            }

            state[node] = ON_PATH;

            if parents[node] < 0 {
                break;
            }
            node = parents[node] as usize;
        }

        if looped {
            return true;
        }

        // No cycle: re-walk the same chain and promote it to CYCLE_FREE. Each node is
        // ON_PATH exactly once across the whole run, so this second pass is still O(N)
        // overall -- it buys us the whole path buffer for free.
        let mut node = idx;
        while state[node] == ON_PATH {
            state[node] = CYCLE_FREE;
            if parents[node] < 0 {
                break;
            }
            node = parents[node] as usize;
        }
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    /// Guards the one asymmetry between `geodesic_nearest` and `geodesic_farthest`: a target
    /// sitting on the node we walk through is at distance zero, which wins a minimum but *loses*
    /// a maximum. Collapsing the guarded merge in `geodesic_extreme` back into an unconditional
    /// "if the parent is a target, take it" would make this return node 1 at distance 1.
    #[test]
    fn directed_farthest_skips_nearer_target_ancestor() {
        // 0 <- 1 <- 2, every node a target, unit weights
        let parents = arr1(&[-1, 0, 1]);
        let sources = Some(arr1(&[2]));

        let (dist, tgt) = geodesic_farthest::<f32>(&parents.view(), &sources, &None, &None, true);
        assert_eq!(dist[0], 2.0); // node 0, not the closer target ancestor node 1
        assert_eq!(tgt[0], 0);

        let (dist, tgt) = geodesic_nearest::<f32>(&parents.view(), &sources, &None, &None, true);
        assert_eq!(dist[0], 1.0);
        assert_eq!(tgt[0], 1);
    }

    /// A source that is its own only target has no *distinct* target and must report "none".
    #[test]
    fn farthest_excludes_self() {
        let parents = arr1(&[-1, 0, 1]);
        let sources = Some(arr1(&[1]));
        let targets = Some(arr1(&[1]));

        let (dist, tgt) = geodesic_farthest::<f32>(&parents.view(), &sources, &targets, &None, false);
        assert_eq!(dist[0], -1.0);
        assert_eq!(tgt[0], -1);
    }

    /// The undirected rerooting DP has to look *through* the parent, not just into the subtree.
    #[test]
    fn undirected_farthest_reaches_across_the_root() {
        //     0
        //    / \
        //   1   3
        //   |
        //   2
        let parents = arr1(&[-1, 0, 1, 0]);
        let sources = Some(arr1(&[2]));
        let targets = Some(arr1(&[0, 3]));

        let (dist, tgt) = geodesic_farthest::<f32>(&parents.view(), &sources, &targets, &None, false);
        assert_eq!(dist[0], 3.0); // 2 -> 1 -> 0 -> 3
        assert_eq!(tgt[0], 3);
    }

    /// The canonical example from the Python docstring; also the doctest in `dag.py`.
    #[test]
    fn classify_matches_the_documented_example() {
        //   0 - 1 - 2 - 3
        //        \
        //         4 - 5 - 6
        //              \
        //               7
        let parents = arr1(&[-1, 0, 1, 2, 1, 4, 5, 5]);

        let types = classify_nodes(&parents.view());

        assert_eq!(types, arr1(&[0, 2, 3, 1, 3, 2, 1, 1]));
    }

    /// Roots outrank every other class. A root with no children is still a root (not a leaf) and
    /// a root with two children is still a root (not a branch point) -- dropping the `p < 0` arm
    /// to the bottom of the match would silently reclassify both.
    #[test]
    fn classify_gives_roots_priority_over_leafs_and_branch_points() {
        assert_eq!(classify_nodes(&arr1(&[-1]).view()), arr1(&[0])); // isolated root, 0 children
        assert_eq!(classify_nodes(&arr1(&[-1, 0]).view()), arr1(&[0, 1])); // root, 1 child
        assert_eq!(classify_nodes(&arr1(&[-1, 0, 0]).view()), arr1(&[0, 1, 1])); // root, 2 children
    }

    /// Every node of a forest is classified relative to its own root; the components don't
    /// interact.
    #[test]
    fn classify_handles_multi_root_forests() {
        //   1       4 - 6
        //  /       /
        // 0       3
        //  \       \
        //   2       5
        let parents = arr1(&[-1, 0, 0, -1, 3, 3, 4]);

        let types = classify_nodes(&parents.view());

        assert_eq!(types, arr1(&[0, 1, 1, 0, 3, 1, 1]));
    }

    /// Three or more children is still a branch point, not something else.
    #[test]
    fn classify_treats_trifurcations_as_branch_points() {
        //   1        2
        //  /        /
        // 0 - 4 - 3
        //          \
        //           5
        let parents = arr1(&[-1, 0, 4, 4, 0, 4]);

        let types = classify_nodes(&parents.view());

        assert_eq!(types[4], 2); // three children -> branch point
        assert_eq!(types, arr1(&[0, 1, 1, 1, 2, 1]));
    }

    /// "standard" is the textbook Strahler index: a node's index only goes up when two or more of
    /// its children tie for the highest index. The single-child branch below node 1 must *not*
    /// bump it.
    #[test]
    fn strahler_standard_only_increments_on_a_tie() {
        //   0 - 1 - 2 - 3
        //        \
        //         4 - 5 - 6
        //              \
        //               7
        let parents = arr1(&[-1, 0, 1, 2, 1, 4, 5, 5]);

        let si = strahler_index(&parents.view(), false, &None, &None);

        // 5 ties two leafs -> 2, which carries up through 4 to 1; 1 sees {2 (SI 1), 4 (SI 2)},
        // no tie at the max, so it stays 2.
        assert_eq!(si, arr1(&[2, 2, 1, 1, 2, 2, 1, 1]));
    }

    /// "greedy" always increases the index at converging branches, whether or not they tie. That
    /// makes a node's index the number of leafs below it -- note the trifurcation at node 5 gets
    /// 3, which the standard method would only ever call 2.
    #[test]
    fn strahler_greedy_adds_up_converging_branches() {
        //   0 - 1 - 2 (leaf)
        //        \
        //         3 - 4,5,6 (leafs)
        let parents = arr1(&[-1, 0, 1, 1, 3, 3, 3]);

        let greedy = strahler_index(&parents.view(), true, &None, &None);
        let standard = strahler_index(&parents.view(), false, &None, &None);

        assert_eq!(greedy, arr1(&[4, 4, 1, 3, 1, 1, 1])); // 4 leafs below the root
        assert_eq!(standard, arr1(&[2, 2, 1, 2, 1, 1, 1]));
    }

    /// Twigs below `min_twig_size` must not contribute, and instead inherit the index of their
    /// first branch point. Without the twig, node 1 has a single contributing child and stays 1.
    ///
    /// Careful with the threshold: the twig is measured from the leaf up to *and including* its
    /// first branch point, so a one-node twig needs `min_twig_size = 3` to be dropped, not 2. That
    /// is off by one against the documented "twigs with fewer nodes than this", but it is the
    /// long-standing behaviour and callers are calibrated to it -- don't quietly change it.
    #[test]
    fn strahler_min_twig_size_excludes_and_backfills_short_twigs() {
        //   0 - 1 - 2 - 3 (long branch)
        //        \
        //         4 (one-node twig)
        let parents = arr1(&[-1, 0, 1, 2, 1]);

        // No threshold: the twig counts, so node 1 ties two SI-1 children -> 2.
        assert_eq!(
            strahler_index(&parents.view(), false, &None, &None),
            arr1(&[2, 2, 1, 1, 1])
        );

        // 2 is not enough to drop a one-node twig (see above).
        assert_eq!(
            strahler_index(&parents.view(), false, &None, &Some(2)),
            arr1(&[2, 2, 1, 1, 1])
        );

        // 3 drops it: nothing ties at node 1 any more, and the twig inherits node 1's index.
        assert_eq!(
            strahler_index(&parents.view(), false, &None, &Some(3)),
            arr1(&[1, 1, 1, 1, 1])
        );
    }

    /// `to_ignore` drops a twig regardless of its length, backfilling it from its branch point.
    #[test]
    fn strahler_to_ignore_excludes_named_leafs() {
        //   0 - 1 - 2 - 3
        //        \
        //         4 - 5
        let parents = arr1(&[-1, 0, 1, 2, 1, 4]);

        assert_eq!(
            strahler_index(&parents.view(), false, &None, &None),
            arr1(&[2, 2, 1, 1, 1, 1])
        );
        // Ignoring leaf 5 removes the 4->5 branch, so node 1 no longer ties.
        assert_eq!(
            strahler_index(&parents.view(), false, &Some(vec![5]), &None),
            arr1(&[1, 1, 1, 1, 1, 1])
        );
    }

    /// Characterisation test, not a specification. Cyclic input is malformed and no caller should
    /// rely on this, but it used to *hang forever* (the leaf-sorting pass in `find_leafs` walked
    /// each leaf to a root that a cycle never reaches). Classifying by child count terminates
    /// instead. Don't "fix" this back into a leaf-to-root walk.
    #[test]
    fn classify_terminates_on_cyclic_input() {
        let parents = arr1(&[0]); // node 0 is its own parent

        let types = classify_nodes(&parents.view());

        assert_eq!(types, arr1(&[3]));
    }

    /// `dist_to_root` counts EDGES, so a root is at distance zero. It used to count nodes
    /// (root = 1.0), which disagreed with `all_dists_to_root` by exactly one. These two are
    /// the same measurement and must not drift apart again.
    #[test]
    fn dist_to_root_counts_edges_and_agrees_with_all_dists_to_root() {
        //   0 - 1 - 2
        let parents = arr1(&[-1, 0, 1]);

        let all: Vec<f32> = all_dists_to_root(&parents.view(), &None, &None::<Array1<f32>>);
        assert_eq!(all, vec![0.0, 1.0, 2.0]);

        for node in 0..3 {
            assert_eq!(dist_to_root(&parents.view(), node), all[node as usize]);
        }
    }

    /// Memoizing `all_dists_to_root` must not change what it returns -- for a subset of
    /// sources, for weights, or across a multi-root forest where each tree has its own root.
    #[test]
    fn all_dists_to_root_handles_subsets_weights_and_forests() {
        //   0 - 1 - 2      3 - 4      (two separate roots)
        let parents = arr1(&[-1, 0, 1, -1, 3]);

        assert_eq!(
            all_dists_to_root(&parents.view(), &None, &None::<Array1<f32>>),
            vec![0.0, 1.0, 2.0, 0.0, 1.0]
        );

        // A subset of sources must give the same answers, in the order asked for.
        assert_eq!(
            all_dists_to_root(&parents.view(), &Some(arr1(&[4, 2, 0])), &None::<Array1<f32>>),
            vec![1.0, 2.0, 0.0]
        );

        // Weights live on the child, i.e. an edge contributes the weight of the node below it.
        let weights = Some(arr1(&[100.0f32, 2.0, 3.0, 100.0, 5.0]));
        assert_eq!(
            all_dists_to_root(&parents.view(), &None, &weights),
            vec![0.0, 2.0, 5.0, 0.0, 5.0]
        );
    }

    /// `has_cycles` used to be quadratic *and* is easy to get subtly wrong: the walk needs a
    /// stop condition for nodes already proven clean, without that stop leaking across into
    /// "already on the current path" (which is what a cycle actually is).
    #[test]
    fn has_cycles_detects_cycles_without_false_positives_on_forests() {
        // Acyclic: a chain, a forest of two trees, a single root, and a converging tree.
        assert!(!has_cycles(&arr1(&[-1, 0, 1]).view()));
        assert!(!has_cycles(&arr1(&[-1, 0, 1, -1, 3]).view()));
        assert!(!has_cycles(&arr1(&[-1]).view()));
        assert!(!has_cycles(&arr1(&[-1, 0, 0, 1, 1, 2]).view()));
        assert!(!has_cycles(&arr1::<i32>(&[]).view()));

        // Cyclic: a self-loop, a two-node cycle, a three-node cycle with no root at all...
        assert!(has_cycles(&arr1(&[0]).view()));
        assert!(has_cycles(&arr1(&[1, 0]).view()));
        assert!(has_cycles(&arr1(&[1, 2, 0]).view()));

        // ...and the case the `checked`-array pruning could plausibly swallow: a perfectly
        // good tree that is visited FIRST, with a cycle hanging off to the side. Nodes 0-2
        // get marked clean, then the walk from 3 must still find 3 -> 4 -> 3.
        assert!(has_cycles(&arr1(&[-1, 0, 1, 4, 3]).view()));
    }

    /// The weighted branch of `generate_segments` sorts its segments longest-first but used to
    /// return the lengths in the ORIGINAL order, so `lengths[i]` described some other segment
    /// than `segments[i]`. They have to be permuted together.
    #[test]
    fn generate_segments_lengths_line_up_with_their_segments() {
        //   0 - 1 - 2 - 3
        //        \
        //         4 - 5 - 6
        // Weight the shorter branch heavily so that sorting by weight reorders the segments
        // relative to the order the leafs come out in -- otherwise the bug hides.
        let parents = arr1(&[-1, 0, 1, 2, 1, 4, 5]);
        let weights = arr1(&[1.0f32, 1.0, 1.0, 1.0, 50.0, 50.0, 50.0]);

        let (segments, lengths) = generate_segments(&parents.view(), Some(weights.clone()));
        let lengths = lengths.expect("weighted input must return lengths");

        assert_eq!(segments.len(), lengths.len());
        for (segment, reported) in segments.iter().zip(lengths.iter()) {
            let actual: f32 = segment.iter().map(|&n| weights[n as usize]).sum();
            assert_eq!(actual, *reported);
        }

        // And they must still come out longest-first.
        assert!(lengths.windows(2).all(|w| w[0] >= w[1]));
    }

    /// A tree where every backbone node also carries a twig, so the branch points form one
    /// long chain -- the shape that made the old recursive `walk_up_and_count_recursively`
    /// recurse once per branch point. This checks the iterative traversal still gets the
    /// distances right; `partial_survives_a_deep_chain_of_branch_points` checks it survives
    /// a depth that used to segfault.
    #[test]
    fn partial_matches_all_by_all_on_a_chain_of_branch_points() {
        // backbone 0-1-2-3, each with a twig hanging off it
        //   0 - 1 - 2 - 3
        //   |   |   |   |
        //   4   5   6   7
        let parents = arr1(&[-1, 0, 1, 2, 0, 1, 2, 3]);

        let full = geodesic_distances_all_by_all(&parents.view(), &None::<Array1<f32>>, false);

        // Asking for a subset of sources must agree with the full matrix, row for row.
        let sources = Some(arr1(&[4, 7, 0]));
        let partial =
            geodesic_distances_partial(&parents.view(), &sources, &None, &None::<Array1<f32>>, false);

        for (row, &source) in [4usize, 7, 0].iter().enumerate() {
            for target in 0..8usize {
                assert_eq!(
                    partial[[row, target]],
                    full[[source, target]],
                    "source {source} -> target {target}"
                );
            }
        }
    }

    /// A branch point with many children. The cross-pairing at a branch point used to loop over
    /// every ordered pair of children; it now folds each child into a running union instead, so
    /// this is the case that would break if the folding missed a direction (source in an early
    /// child vs target in a later one, or the other way round).
    #[test]
    fn partial_matches_all_by_all_at_a_wide_branch_point() {
        // A root with eight children, each carrying a two-node tail.
        //   0 -> 1..8, and each i -> i+8
        let mut parents: Vec<i32> = vec![-1];
        for _ in 0..8 {
            parents.push(0);
        }
        for i in 1..=8 {
            parents.push(i);
        }
        let parents = Array1::from(parents);
        let n = parents.len();

        let weights = Some(Array1::from(
            (0..n).map(|i| 1.0 + (i as f32) * 0.5).collect::<Vec<f32>>(),
        ));

        let full = geodesic_distances_all_by_all(&parents.view(), &weights, false);

        // Every source subset must reproduce the corresponding rows of the full matrix.
        let sources = Some(arr1(&[9, 16, 0, 4]));
        let partial =
            geodesic_distances_partial(&parents.view(), &sources, &None, &weights, false);

        for (row, &source) in [9usize, 16, 0, 4].iter().enumerate() {
            for target in 0..n {
                assert_eq!(
                    partial[[row, target]],
                    full[[source, target]],
                    "source {source} -> target {target}"
                );
            }
        }
    }

    /// Regression guard for a hard segfault: `walk_up_and_count` used to recurse once per
    /// branch point, so a backbone that branches at every node blew the stack somewhere north
    /// of ~45k branch points. Keep this at a depth that would actually have crashed.
    #[test]
    fn partial_survives_a_deep_chain_of_branch_points() {
        // 60k backbone nodes, each with a twig: 60k branch points, all in one chain.
        const BACKBONE: usize = 60_000;
        let mut parents: Vec<i32> = Vec::with_capacity(BACKBONE * 2);
        parents.push(-1);
        for i in 1..BACKBONE {
            parents.push(i as i32 - 1);
        }
        for b in 0..BACKBONE {
            parents.push(b as i32);
        }
        let parents = Array1::from(parents);

        let sources = Some(arr1(&[0]));
        let targets = Some(arr1(&[BACKBONE as i32 - 1]));
        let dists = geodesic_distances_partial(
            &parents.view(),
            &sources,
            &targets,
            &None::<Array1<f32>>,
            false,
        );

        // Node 0 to the far end of the backbone is BACKBONE - 1 edges.
        assert_eq!(dists[[0, 0]], (BACKBONE - 1) as f32);
    }

    /// Nodes in different connected components have no path between them. That is reported as
    /// -1, the same sentinel `geodesic_distances_all_by_all` uses. It used to be reported as
    /// 1.0, which is indistinguishable from a genuine one-edge distance.
    #[test]
    fn geodesic_pairs_reports_disconnected_as_minus_one() {
        //   0 - 1        2 - 3      (two components)
        let parents = arr1(&[-1, 0, -1, 2]);

        let dists = geodesic_pairs(
            &parents.view(),
            &arr1(&[0, 0, 2]).view(),
            &arr1(&[2, 1, 3]).view(),
            &None,
            false,
        );

        // 0 -> 2 crosses components; the other two are real one-edge distances.
        assert_eq!(dists, arr1(&[-1.0, 1.0, 1.0]));
    }

    /// A node's height is the distance *down* to the farthest leaf below it, so leafs are 0 and a
    /// root carries the longest root-to-leaf path. Node 1 has to take the *longest* of its two
    /// branches, not the first or the last one it happens to see.
    #[test]
    fn subtree_height_measures_the_longest_path_below_each_node() {
        //   0 - 1 - 2 - 3
        //        \
        //         4 - 5 - 6
        //              \
        //               7
        let parents = arr1(&[-1, 0, 1, 2, 1, 4, 5, 5]);

        let height = subtree_height::<f32>(&parents.view(), &None);

        // Node 1 sees a 2-hop branch (2, 3) and a 3-hop one (4, 5, 6) -> 3, not 2.
        assert_eq!(height, arr1(&[4.0, 3.0, 1.0, 0.0, 2.0, 1.0, 0.0, 0.0]));
    }

    /// The weight of an edge lives on its *child*: `weights[node]` is the node -> parent edge, as
    /// in `all_dists_to_root`. Reading it off the parent instead would still produce plausible
    /// numbers, so this pins a tree where the longest branch by hop count is *not* the longest by
    /// weight.
    #[test]
    fn subtree_height_weights_the_child_to_parent_edge() {
        //   0 - 1 - 2 - 3
        //        \
        //         4 - 5 - 6
        let parents = arr1(&[-1, 0, 1, 2, 1, 4, 5]);
        //   node:            0    1    2     3    4    5    6
        let weights = arr1(&[0.0, 1.0, 5.0, 10.0, 1.0, 1.0, 1.0]);

        let height = subtree_height(&parents.view(), &Some(weights));

        // 1 -> 2 -> 3 weighs 5 + 10 = 15; the *longer* 1 -> 4 -> 5 -> 6 only weighs 1 + 1 + 1 = 3.
        assert_eq!(height, arr1(&[16.0f32, 15.0, 10.0, 0.0, 2.0, 1.0, 0.0]));
    }

    /// Every component is measured against its own leafs and the components don't interact --
    /// unlike the depth-subtraction formulation, this needs no root distances to get that right.
    /// An isolated root is both a leaf and a root, and has height 0.
    #[test]
    fn subtree_height_handles_multi_root_forests() {
        //   1        4 - 6      7 (isolated root)
        //  /        /
        // 0        3
        //  \        \
        //   2        5
        let parents = arr1(&[-1, 0, 0, -1, 3, 3, 4, -1]);

        let height = subtree_height::<f32>(&parents.view(), &None);

        assert_eq!(height, arr1(&[1.0, 0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0]));
    }

    /// `parent_dist` leaves NaN in the root's weight slot. We never read a root's own weight, so
    /// that NaN must not leak into any height. A naive `deepest - depth` would propagate it.
    #[test]
    fn subtree_height_ignores_the_roots_nan_weight() {
        //   0 - 1 - 2
        let parents = arr1(&[-1, 0, 1]);
        let weights = arr1(&[f32::NAN, 2.0, 3.0]);

        let height = subtree_height(&parents.view(), &Some(weights));

        assert_eq!(height, arr1(&[5.0f32, 3.0, 0.0]));
    }

    /// A cycle is malformed input (see `has_cycles`) but it must not hang the sweep. Cycle nodes
    /// never reach a pending count of zero, so they are never visited and keep their zero; the rest
    /// of the tree is unaffected. The recursive formulation of this DP would instead spin forever.
    #[test]
    fn subtree_height_does_not_hang_on_a_cycle() {
        // 0 -> 1 -> 2 -> 0 (a 3-cycle), plus an unrelated tree 4 -> 3
        let parents = arr1(&[1, 2, 0, -1, 3]);

        let height = subtree_height::<f32>(&parents.view(), &None);

        assert_eq!(height, arr1(&[0.0, 0.0, 0.0, 1.0, 0.0]));
    }

    /// The sweep is iterative, so an unbranched chain far deeper than the stack must not blow it.
    #[test]
    fn subtree_height_survives_a_very_deep_chain() {
        let n = 200_000;
        // 0 <- 1 <- 2 <- ... : node i's parent is i - 1, so 0 is the root and n - 1 the only leaf.
        let parents = Array1::from_iter((0..n).map(|i| i as i32 - 1));

        let height = subtree_height::<f32>(&parents.view(), &None);

        assert_eq!(height[0], (n - 1) as f32);
        assert_eq!(height[n - 1], 0.0);
    }
}