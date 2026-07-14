use extendr_api::prelude::*;
use ndarray::{Array1, Array2, ArrayView1};
use std::collections::HashMap;

use fastcore::cmtk::{self, Chain, Fallback, InverseOpts, Mode, XformOpts};
use fastcore::elastix::{self, OutOfBounds};
use fastcore::nblast::{load_smat, load_smat_alpha, Opts, Smat};

/// For each node ID in `parents` find its index in `nodes`.
///
/// Importantly this is 0-indexed to match indexing in Rust.
/// Roots will have parent index -1.
///
/// @param nodes Integer vector of node IDs.
/// @param parents Integer vector of parent IDs, one per node; roots use their
///   own ID or a negative value.
/// @return Integer vector of 0-based parent indices (`-1` for roots).
/// @export
#[extendr]
pub fn node_indices(nodes: Vec<i32>, parents: Vec<i32>) -> Vec<i32> {
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
///
/// @param parents Integer vector of 0-based parent indices (roots are `< 0`).
/// @param x,y,z Numeric vectors of node coordinates, one entry per node.
/// @return Numeric vector of Euclidean child-to-parent distances (`0` for roots).
/// @export
#[extendr]
pub fn child_to_parent_dists(parents: Vec<i32>, x: Vec<f64>, y: Vec<f64>, z: Vec<f64>) -> Vec<f64> {
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
///
/// @param parents Integer vector of 0-based parent indices (roots are `< 0`).
/// @param sources Optional integer vector of node indices to measure from;
///   `NULL` uses every node.
/// @param weights Optional numeric vector of child-to-parent edge weights;
///   `NULL` counts edges (hop distance).
/// @return Numeric vector of distances to the root for each requested node.
/// @export
#[extendr]
pub fn all_dists_to_root(
    parents: Vec<i32>,
    sources: Option<Vec<i32>>,
    weights: Option<Vec<f64>>, // f64 is used to match R's numeric type
) -> Vec<f32> {
    let parents = Array1::from_vec(parents);
    let sources: Option<Array1<i32>> = sources.map(Array1::from_vec);
    // Convert f64 to f32
    let weights: Option<Array1<f32>> =
        weights.map(|w| Array1::from_vec(w.iter().map(|x| *x as f32).collect()));

    fastcore::dag::all_dists_to_root(&parents.view(), &sources, &weights)
}

/// Geodesic distances between nodes.
///
/// @param parents Integer vector of 0-based parent indices (roots are `< 0`).
/// @param sources Optional integer vector of source node indices; `NULL` uses
///   every node.
/// @param targets Optional integer vector of target node indices; `NULL` uses
///   every node.
/// @param weights Optional numeric vector of edge weights; `NULL` counts edges.
/// @param directed Logical; if `TRUE` only traverse edges child-to-parent.
/// @return Numeric matrix of geodesic distances (sources in rows, targets in
///   columns).
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
    let weights: Option<Array1<f64>> = weights.map(Array1::from_vec);
    let sources: Option<Array1<i32>> = sources.map(Array1::from_vec);
    let targets: Option<Array1<i32>> = targets.map(Array1::from_vec);

    let dists: Array2<f64> = if sources.is_none() && targets.is_none() {
        // If no sources and targets, use the more efficient full implementation
        fastcore::dag::geodesic_distances_all_by_all(&parents.view(), &weights, directed)
    // If sources and/or targets use the partial implementation
    } else {
        fastcore::dag::geodesic_distances_partial(
            &parents.view(),
            &sources,
            &targets,
            &weights,
            directed,
        )
    };

    array2_to_rmatrix(&dists)
}

/// Calculate Strahler Index.
///
/// @param parents Integer vector of 0-based parent indices (roots are `< 0`).
/// @param greedy Logical; use the greedy variant of the algorithm.
/// @param to_ignore Optional integer vector of node indices to skip.
/// @param min_twig_size Optional integer; ignore twigs shorter than this.
/// @return Integer vector with the Strahler index of each node.
/// @export
#[extendr]
pub fn strahler_index(
    parents: Vec<i32>,
    greedy: bool,
    to_ignore: Option<Vec<i32>>,
    min_twig_size: Option<i32>,
) -> Vec<i32> {
    let parents = Array1::from_vec(parents);
    fastcore::dag::strahler_index(&parents.view(), greedy, &to_ignore, &min_twig_size).to_vec()
}

/// Height of the subtree below each node.
///
/// A node's height is the geodesic distance from it down to the farthest leaf
/// below it; leaves have a height of 0 and a root carries the length of the
/// longest root-to-leaf path in its component.
///
/// @param parents Integer vector of 0-based parent indices (roots are `< 0`).
/// @param weights Optional numeric vector of child-to-parent edge weights;
///   `NULL` counts edges (hop distance).
/// @return Numeric vector with the height of each node.
/// @export
#[extendr]
pub fn subtree_height(parents: Vec<i32>, weights: Option<Vec<f64>>) -> Vec<f64> {
    let parents = Array1::from_vec(parents);
    let weights: Option<Array1<f64>> = weights.map(Array1::from_vec);

    fastcore::dag::subtree_height(&parents.view(), &weights).to_vec()
}

/// Connected components.
///
/// @param parents Integer vector of 0-based parent indices (roots are `< 0`).
/// @return Integer vector assigning each node a component id.
/// @export
#[extendr]
pub fn connected_components(parents: Vec<i32>) -> Vec<i32> {
    let parents = Array1::from_vec(parents);
    fastcore::dag::connected_components(&parents.view()).to_vec()
}

/// Prune twigs below given threshold.
///
/// Returns indices of nodes to keep.
///
/// @param parents Integer vector of 0-based parent indices (roots are `< 0`).
/// @param threshold Numeric length threshold; twigs shorter than this are pruned.
/// @param weights Optional numeric vector of edge weights; `NULL` counts edges.
/// @return Integer vector of node indices to keep.
/// @export
#[extendr]
pub fn prune_twigs(parents: Vec<i32>, threshold: f64, weights: Option<Vec<f64>>) -> Vec<i32> {
    let parents = Array1::from_vec(parents);
    let weights: Option<Array1<f64>> = weights.map(Array1::from_vec);

    // Mask is currently not supported - strangely, extendr does not seem to support Vec<bool>
    fastcore::dag::prune_twigs(&parents.view(), threshold as f32, &weights, &None)
}

/// Return path length from a single node to the root.
///
/// @param parents Integer vector of 0-based parent indices (roots are `< 0`).
/// @param node Integer index of the node to measure from.
/// @return Numeric path length (edge count) from `node` to its root.
/// @export
#[extendr]
pub fn dist_to_root(parents: Vec<i32>, node: i32) -> f64 {
    let parents = Array1::from_vec(parents);
    fastcore::dag::dist_to_root(&parents.view(), node) as f64
}

/// Classify nodes into roots (0), leaves (1), branch points (2) and slabs (3).
///
/// @param parents Integer vector of 0-based parent indices (roots are `< 0`).
/// @return Integer vector: `0` root, `1` leaf, `2` branch point, `3` slab.
/// @export
#[extendr]
pub fn classify_nodes(parents: Vec<i32>) -> Vec<i32> {
    let parents = Array1::from_vec(parents);
    fastcore::dag::classify_nodes(&parents.view()).to_vec()
}

/// Check whether the tree contains cycles.
///
/// @param parents Integer vector of 0-based parent indices (roots are `< 0`).
/// @return Logical; `TRUE` if the parent structure contains a cycle.
/// @export
#[extendr]
pub fn has_cycles(parents: Vec<i32>) -> bool {
    let parents = Array1::from_vec(parents);
    fastcore::dag::has_cycles(&parents.view())
}

/// Geodesic distances for explicit pairs of nodes.
///
/// `sources` and `targets` are parallel arrays of node indices; the returned
/// vector holds the distance between each `(source, target)` pair.
///
/// @param parents Integer vector of 0-based parent indices (roots are `< 0`).
/// @param sources Integer vector of source node indices.
/// @param targets Integer vector of target node indices (same length as
///   `sources`).
/// @param weights Optional numeric vector of edge weights; `NULL` counts edges.
/// @param directed Logical; if `TRUE` only traverse edges child-to-parent.
/// @return Numeric vector with the distance of each `(source, target)` pair.
/// @export
#[extendr]
pub fn geodesic_pairs(
    parents: Vec<i32>,
    sources: Vec<i32>,
    targets: Vec<i32>,
    weights: Option<Vec<f64>>,
    directed: bool,
) -> Vec<f32> {
    let parents = Array1::from_vec(parents);
    let sources = Array1::from_vec(sources);
    let targets = Array1::from_vec(targets);
    let weights: Option<Array1<f32>> =
        weights.map(|w| Array1::from_vec(w.iter().map(|x| *x as f32).collect()));

    fastcore::dag::geodesic_pairs(
        &parents.view(),
        &sources.view(),
        &targets.view(),
        &weights,
        directed,
    )
    .to_vec()
}

/// Distance to the nearest target for each source.
///
/// Memory-efficient companion to `geodesic_distances` that never materialises the
/// full distance matrix. Returns a list with `distances` (distance to the nearest
/// target) and `nearest` (index of that target); sources without a reachable
/// target get `-1`.
///
/// @param parents Integer vector of 0-based parent indices (roots are `< 0`).
/// @param sources Optional integer vector of source node indices; `NULL` uses
///   every node.
/// @param targets Optional integer vector of target node indices; `NULL` uses
///   every node.
/// @param weights Optional numeric vector of edge weights; `NULL` counts edges.
/// @param directed Logical; if `TRUE` only traverse edges child-to-parent.
/// @return List with `distances` (numeric, distance to the nearest target) and
///   `nearest` (integer target index, `-1` when unreachable).
/// @export
#[extendr]
pub fn geodesic_nearest(
    parents: Vec<i32>,
    sources: Option<Vec<i32>>,
    targets: Option<Vec<i32>>,
    weights: Option<Vec<f64>>,
    directed: bool,
) -> Robj {
    let parents = Array1::from_vec(parents);
    let sources = sources.map(Array1::from_vec);
    let targets = targets.map(Array1::from_vec);
    let weights: Option<Array1<f32>> =
        weights.map(|w| Array1::from_vec(w.iter().map(|x| *x as f32).collect()));

    let (dists, nearest) =
        fastcore::dag::geodesic_nearest(&parents.view(), &sources, &targets, &weights, directed);

    list!(distances = dists.to_vec(), nearest = nearest.to_vec()).into()
}

/// Distance to the farthest target for each source.
///
/// The mirror image of `geodesic_nearest`: same linear-time algorithm, but it keeps
/// the farthest rather than the nearest target and never materialises the full
/// distance matrix. Returns a list with `distances` (distance to the farthest
/// target) and `farthest` (index of that target); sources without a reachable
/// target get `-1`. A source that is itself a target is matched to the farthest
/// *other* target, never to itself.
///
/// @param parents Integer vector of 0-based parent indices (roots are `< 0`).
/// @param sources Optional integer vector of source node indices; `NULL` uses
///   every node.
/// @param targets Optional integer vector of target node indices; `NULL` uses
///   every node.
/// @param weights Optional numeric vector of edge weights; `NULL` counts edges.
/// @param directed Logical; if `TRUE` only traverse edges child-to-parent. With
///   non-negative weights the farthest such target is the target ancestor closest
///   to the root.
/// @return List with `distances` (numeric, distance to the farthest target) and
///   `farthest` (integer target index, `-1` when unreachable).
/// @export
#[extendr]
pub fn geodesic_farthest(
    parents: Vec<i32>,
    sources: Option<Vec<i32>>,
    targets: Option<Vec<i32>>,
    weights: Option<Vec<f64>>,
    directed: bool,
) -> Robj {
    let parents = Array1::from_vec(parents);
    let sources = sources.map(Array1::from_vec);
    let targets = targets.map(Array1::from_vec);
    let weights: Option<Array1<f32>> =
        weights.map(|w| Array1::from_vec(w.iter().map(|x| *x as f32).collect()));

    let (dists, farthest) =
        fastcore::dag::geodesic_farthest(&parents.view(), &sources, &targets, &weights, directed);

    list!(distances = dists.to_vec(), farthest = farthest.to_vec()).into()
}

/// Synapse flow centrality for each node.
///
/// `presynapses`/`postsynapses` give the number of pre-/post-synapses at each node.
/// `mode` is one of "centrifugal", "centripetal" or "sum".
///
/// @param parents Integer vector of 0-based parent indices (roots are `< 0`).
/// @param presynapses Integer vector: number of presynapses at each node.
/// @param postsynapses Integer vector: number of postsynapses at each node.
/// @param mode Character; one of `"centrifugal"`, `"centripetal"` or `"sum"`.
/// @return Integer vector with the synapse flow centrality of each node.
/// @export
#[extendr]
pub fn synapse_flow_centrality(
    parents: Vec<i32>,
    presynapses: Vec<i32>,
    postsynapses: Vec<i32>,
    mode: String,
) -> Vec<i32> {
    let parents = Array1::from_vec(parents);
    let presyn: Array1<u32> = Array1::from_vec(presynapses.iter().map(|x| *x as u32).collect());
    let postsyn: Array1<u32> = Array1::from_vec(postsynapses.iter().map(|x| *x as u32).collect());

    let flow = fastcore::dag::synapse_flow_centrality(
        &parents.view(),
        &presyn.view(),
        &postsyn.view(),
        mode,
    );
    flow.iter().map(|&x| x as i32).collect()
}

/// Generate linear segments while maximising segment lengths.
///
/// Returns a list with `segments` (a list of integer vectors, one per segment)
/// and `lengths` (per-segment lengths, or NULL if no weights were supplied).
///
/// @param parents Integer vector of 0-based parent indices (roots are `< 0`).
/// @param weights Optional numeric vector of edge weights; `NULL` returns no
///   `lengths`.
/// @return List with `segments` (list of integer node-index vectors) and
///   `lengths` (numeric per-segment lengths, or `NULL`).
/// @export
#[extendr]
pub fn generate_segments(parents: Vec<i32>, weights: Option<Vec<f64>>) -> Robj {
    let parents = Array1::from_vec(parents);
    let weights: Option<Array1<f32>> =
        weights.map(|w| Array1::from_vec(w.iter().map(|x| *x as f32).collect()));

    let (segments, lengths) = fastcore::dag::generate_segments(&parents.view(), weights);

    let seg_list = List::from_values(segments.into_iter());
    let lengths_robj: Robj = match lengths {
        Some(l) => l.iter().map(|x| *x as f64).collect::<Vec<f64>>().into(),
        None => ().into(),
    };
    list!(segments = seg_list, lengths = lengths_robj).into()
}

/// Break the tree into its linear segments (one integer vector per segment).
///
/// @param parents Integer vector of 0-based parent indices (roots are `< 0`).
/// @return List of integer vectors, one per linear segment.
/// @export
#[extendr]
pub fn break_segments(parents: Vec<i32>) -> Robj {
    let parents = Array1::from_vec(parents);
    let segments = fastcore::dag::break_segments(&parents.view());
    List::from_values(segments.into_iter()).into()
}

// ---------------------------------------------------------------------------
// Topology repair ("healing")
// ---------------------------------------------------------------------------
//
// These mirror the Python bindings in `py/src/topo.rs`. Coordinates are passed as
// separate x/y/z numeric vectors (as in `child_to_parent_dists`) rather than an
// (N, 3) matrix, and node references are 0-based indices throughout, like the rest
// of the DAG family.

/// Assemble an (N, 3) coordinate array from R's separate x/y/z vectors.
fn xyz_to_coords(x: &[f64], y: &[f64], z: &[f64]) -> Array2<f64> {
    let n = x.len();
    assert!(
        y.len() == n && z.len() == n,
        "`x`, `y` and `z` must have the same length"
    );
    Array2::from_shape_fn((n, 3), |(i, j)| match j {
        0 => x[i],
        1 => y[i],
        _ => z[i],
    })
}

/// Convert an optional R logical vector into a boolean mask. `NULL` -> `None`.
///
/// Taken as a bare `Robj` because extendr does not convert R logicals into
/// `Vec<bool>` (see the note on `prune_twigs`); `as_logical_slice` does.
fn robj_to_mask(mask: &Robj, n: usize) -> Option<Array1<bool>> {
    if mask.is_null() {
        return None;
    }
    let values = mask
        .as_logical_slice()
        .expect("`mask` must be a logical vector or NULL");
    assert_eq!(values.len(), n, "`mask` must have one entry per node");
    Some(values.iter().map(|b| b.is_true()).collect())
}

/// Interpret R's `use_radius` (`NULL` / `FALSE` / `TRUE` / a number) as a weight.
/// `None` means "do not use radius"; `Some(w)` scales the radius dimension by `w`.
fn parse_use_radius(use_radius: &Robj) -> Option<f64> {
    if use_radius.is_null() {
        return None;
    }
    if let Some(flag) = use_radius.as_logical() {
        return flag.is_true().then_some(1.0);
    }
    let weight = use_radius
        .as_real()
        .expect("`use_radius` must be TRUE/FALSE, a number or NULL");
    (weight != 0.0).then_some(weight)
}

/// Per-node radius to use as a 4th coordinate, scaled by `weight`.
///
/// We use the mean radius of the linear segment a node belongs to rather than
/// the node's own radius, which is far less noisy. Isolated nodes (a root with
/// no children) form no segment and so fall back to their own radius.
fn segment_radius(parents: &ArrayView1<i32>, radius: &[f64], weight: f64) -> Array1<f64> {
    let n = parents.len();
    assert_eq!(radius.len(), n, "`radius` must have one entry per node");

    // Missing radii would poison the segment mean.
    let clean: Vec<f64> = radius
        .iter()
        .map(|&r| if r.is_finite() { r } else { 0.0 })
        .collect();

    let mut out = Array1::from_vec(clean.clone());
    for seg in fastcore::dag::break_segments(parents) {
        let mean = seg.iter().map(|&i| clean[i as usize]).sum::<f64>() / seg.len() as f64;
        for &i in &seg {
            out[i as usize] = mean;
        }
    }
    out * weight
}

/// Append a scaled segment-radius column to `(N, 3)` coords, giving `(N, 4)`.
fn with_radius_column(coords: Array2<f64>, radius_seg: &Array1<f64>) -> Array2<f64> {
    let n = coords.nrows();
    Array2::from_shape_fn((n, 4), |(i, j)| {
        if j < 3 {
            coords[[i, j]]
        } else {
            radius_seg[i]
        }
    })
}

/// Find the minimal-length edges that reconnect the fragments of a skeleton.
///
/// Given a per-node component label and node coordinates, this returns the set of
/// new edges that would join the fragments into a single tree while minimising the
/// total added length (a minimum spanning tree over the fragments). It does *not*
/// modify the skeleton — see `heal_skeleton` for the one-shot version.
///
/// @param components Integer vector giving each node's connected component, e.g.
///   the output of `connected_components()`. Only equality of labels matters.
/// @param x,y,z Numeric vectors of node coordinates, one entry per node.
/// @param w Optional numeric vector giving a 4th coordinate, one entry per node.
///   The search then happens in 4D, so nodes with similar `w` look closer
///   together; pass a (scaled) radius here to prefer bridging fragments of
///   similar calibre. Note that `max_dist` is then measured in 4D too. `NULL`
///   searches in plain 3D.
/// @param mask Optional logical vector marking the nodes that may be used as
///   endpoints for a new edge; `NULL` allows every node. A fragment without a
///   single eligible node cannot be connected.
/// @param max_dist Optional numeric upper bound on the length of any single new
///   edge; `NULL` means no limit. Fragments whose closest eligible nodes are
///   farther apart than this are left disconnected.
/// @return List with `from` and `to` (integer vectors of 0-based node indices, one
///   pair per new edge) and `dist` (numeric edge lengths). At most
///   `(#fragments - 1)` edges.
/// @export
#[extendr]
pub fn stitch_fragments(
    components: Vec<i32>,
    x: Vec<f64>,
    y: Vec<f64>,
    z: Vec<f64>,
    w: Option<Vec<f64>>,
    mask: Robj,
    max_dist: Option<f64>,
) -> Robj {
    let n = components.len();
    let mut coords = xyz_to_coords(&x, &y, &z);
    assert_eq!(
        coords.nrows(),
        n,
        "`x`, `y` and `z` must have one entry per node"
    );
    if let Some(w) = w {
        assert_eq!(w.len(), n, "`w` must have one entry per node");
        coords = with_radius_column(coords, &Array1::from_vec(w));
    }
    let components = Array1::from_vec(components);
    let mask = robj_to_mask(&mask, n);

    let bridges = fastcore::topo::stitch_fragments(
        &coords.view(),
        &components.view(),
        &mask,
        max_dist.unwrap_or(f64::INFINITY),
    );

    let from: Vec<i32> = bridges.iter().map(|(a, _, _)| *a).collect();
    let to: Vec<i32> = bridges.iter().map(|(_, b, _)| *b).collect();
    let dist: Vec<f64> = bridges.iter().map(|(_, _, d)| *d as f64).collect();

    list!(from = from, to = to, dist = dist).into()
}

/// Regenerate a parent vector after adding a set of undirected edges.
///
/// Turns an edited edge set back into a valid rooted tree: the undirected
/// adjacency is built from the original child -> parent edges plus the new
/// `from`/`to` edges, then oriented away from `root` by breadth-first search.
///
/// @param parents Integer vector of 0-based parent indices (roots are `< 0`).
/// @param from,to Integer vectors of 0-based node indices giving the undirected
///   edges to add, e.g. the `from`/`to` returned by `stitch_fragments()`.
/// @param root Integer 0-based index of the preferred root; its whole component is
///   rooted there. Use a negative value to auto-pick (lowest index per component).
/// @return Integer vector of new 0-based parent indices (roots are `-1`). Any
///   component not reachable from `root` is rooted at its lowest-index node, so the
///   result is valid even when the skeleton could not be fully healed.
/// @export
#[extendr]
pub fn reroot_rewire(parents: Vec<i32>, from: Vec<i32>, to: Vec<i32>, root: i32) -> Vec<i32> {
    assert_eq!(from.len(), to.len(), "`from` and `to` must be the same length");
    let parents = Array1::from_vec(parents);

    let new_edges = Array2::from_shape_fn((from.len(), 2), |(i, j)| if j == 0 { from[i] } else { to[i] });

    fastcore::topo::reroot_rewire(&parents.view(), &new_edges.view(), root).to_vec()
}

/// Heal a fragmented skeleton by reconnecting its fragments.
///
/// Convenience wrapper that finds the minimal-length set of new edges between the
/// skeleton's connected components (see `stitch_fragments()`) and regenerates a
/// single rooted tree from them (see `reroot_rewire()`).
///
/// @param parents Integer vector of 0-based parent indices (roots are `< 0`), e.g.
///   from `node_indices()`.
/// @param x,y,z Numeric vectors of node coordinates, one entry per node.
/// @param method Character; `"ALL"` lets any node form a new edge, `"LEAFS"`
///   restricts new edges to leaf and root nodes (faster, occasionally suboptimal
///   attachment points).
/// @param max_dist Optional numeric maximum length for any single new edge; gaps
///   larger than this are left unhealed, so the result may stay fragmented. `NULL`
///   means no limit.
/// @param min_size Optional integer; fragments with fewer than this many nodes are
///   excluded from healing and stay disconnected. `NULL` heals every fragment.
/// @param mask Optional logical vector restricting which nodes may be used as
///   endpoints for a new edge; combined with `method`. `NULL` allows every node.
/// @param radius Optional numeric vector of node radii, one entry per node. Only
///   required when `use_radius` is set.
/// @param use_radius `TRUE`/`FALSE`, a number, or `NULL`. If set, node radii are
///   taken into account when measuring distances, which prioritises connecting
///   fragments of similar calibre. A number weights the effect: higher values give
///   radius more influence (`TRUE` means 1). To keep this robust we use the mean
///   radius of the segment a node belongs to, not the node's own radius. Note that
///   `max_dist` is then measured in this augmented space too.
/// @return Integer vector of new 0-based parent indices (roots are `-1`). If the
///   skeleton could be fully healed this is a single tree with one root.
/// @export
#[extendr]
pub fn heal_skeleton(
    parents: Vec<i32>,
    x: Vec<f64>,
    y: Vec<f64>,
    z: Vec<f64>,
    method: String,
    max_dist: Option<f64>,
    min_size: Option<i32>,
    mask: Robj,
    radius: Option<Vec<f64>>,
    use_radius: Robj,
) -> Vec<i32> {
    let n = parents.len();
    let mut coords = xyz_to_coords(&x, &y, &z);
    assert_eq!(
        coords.nrows(),
        n,
        "`x`, `y` and `z` must have one entry per node"
    );

    let parents = Array1::from_vec(parents);

    // Optionally augment the coordinates with a scaled segment-radius column.
    if let Some(weight) = parse_use_radius(&use_radius) {
        let radius = radius.expect("`use_radius` requires `radius` to be provided");
        let radius_seg = segment_radius(&parents.view(), &radius, weight);
        coords = with_radius_column(coords, &radius_seg);
    }

    let components = fastcore::dag::connected_components(&parents.view());

    // Build the candidate mask from the various restrictions.
    let mut candidate: Array1<bool> = Array1::from_elem(n, true);

    match method.to_uppercase().as_str() {
        "ALL" => (),
        "LEAFS" => {
            // classify_nodes: 0 = root, 1 = leaf, 2 = branch point, 3 = slab.
            let node_type = fastcore::dag::classify_nodes(&parents.view());
            for i in 0..n {
                candidate[i] &= node_type[i] == 0 || node_type[i] == 1;
            }
        }
        _ => panic!("`method` must be either \"ALL\" or \"LEAFS\""),
    }

    if let Some(mask) = robj_to_mask(&mask, n) {
        for i in 0..n {
            candidate[i] &= mask[i];
        }
    }

    if let Some(min_size) = min_size {
        let mut sizes: HashMap<i32, i32> = HashMap::new();
        for &c in components.iter() {
            *sizes.entry(c).or_insert(0) += 1;
        }
        for i in 0..n {
            candidate[i] &= sizes[&components[i]] >= min_size;
        }
    }

    // 1. Find the bridging edges.
    let bridges = fastcore::topo::stitch_fragments(
        &coords.view(),
        &components.view(),
        &Some(candidate),
        max_dist.unwrap_or(f64::INFINITY),
    );
    let new_edges = Array2::from_shape_fn((bridges.len(), 2), |(i, j)| {
        if j == 0 {
            bridges[i].0
        } else {
            bridges[i].1
        }
    });

    // 2. Regenerate the parent vector. Prefer the existing (first) root so the
    //    healed skeleton keeps its orientation where possible.
    let root = parents.iter().position(|&p| p < 0).map_or(-1, |i| i as i32);

    fastcore::topo::reroot_rewire(&parents.view(), &new_edges.view(), root).to_vec()
}

/// Find connected components of a triangle mesh.
///
/// `faces` is an (N, 3) matrix of vertex indices. Returns an integer vector of
/// length `n_vertices` assigning each vertex the root-vertex index of its
/// component.
///
/// @param faces Integer or numeric `(N, 3)` matrix of triangle vertex indices.
/// @param n_vertices Integer; total number of vertices in the mesh.
/// @return Integer vector of length `n_vertices` giving each vertex the
///   root-vertex index of its component.
/// @export
#[extendr]
pub fn mesh_connected_components(faces: Robj, n_vertices: i32) -> Vec<i32> {
    let faces_u32 = robj_to_faces(&faces);
    fastcore::mesh::mesh_connected_components(faces_u32.view(), n_vertices as usize)
        .iter()
        .map(|&x| x as i32)
        .collect()
}

/// Convert an optional R `(V, 3)` numeric matrix of vertex coordinates.
///
/// R's `NULL` arrives as `Some(Robj::null())`, not as `None` — `NULL` is itself a
/// perfectly good `Robj` — so the null check has to be explicit or we would try to
/// read a matrix out of it.
fn robj_to_coords(vertices: Option<Robj>) -> Option<Array2<f64>> {
    let v = vertices.filter(|v| !v.is_null())?;
    let m = <RMatrix<f64>>::try_from(v).expect("`vertices` must be a numeric (V, 3) matrix");
    let nr = m.nrows();
    let d = m.data();
    Some(Array2::from_shape_fn((nr, 3), |(i, j)| d[j * nr + i]))
}

/// Convert an R `(E, 2)` numeric/integer matrix of edges.
fn robj_to_edges(edges: &Robj) -> Array2<u32> {
    if let Ok(m) = <RMatrix<i32>>::try_from(edges.clone()) {
        let nr = m.nrows();
        let d = m.data();
        Array2::from_shape_fn((nr, 2), |(i, j)| d[j * nr + i] as u32)
    } else if let Ok(m) = <RMatrix<f64>>::try_from(edges.clone()) {
        let nr = m.nrows();
        let d = m.data();
        Array2::from_shape_fn((nr, 2), |(i, j)| d[j * nr + i] as u32)
    } else {
        panic!("`edges` must be a numeric (E, 2) matrix");
    }
}

fn to_u32(v: Option<Vec<i32>>) -> Option<Vec<u32>> {
    v.map(|x| x.iter().map(|&i| i as u32).collect())
}

fn array2_f32_to_rmatrix(arr: &Array2<f32>) -> Robj {
    let (nr, nc) = (arr.nrows(), arr.ncols());
    RArray::new_matrix(nr, nc, |r, c| arr[[r, c]] as f64).into()
}

/// Geodesic ("along-the-mesh-edge") distances on a triangle mesh.
///
/// The mesh counterpart to `geodesic_distances`, which works on skeletons. A
/// skeleton is a tree, so distances there come from walking to the lowest common
/// ancestor; a mesh is a general cyclic graph, so this runs one Dijkstra per source
/// (or a BFS when unweighted), in parallel.
///
/// Note this is the distance *along mesh edges*, not the exact surface geodesic:
/// paths are constrained to run along edges, so on a coarse mesh they overshoot.
///
/// Beware the size of the output: a full `V x V` matrix is ~107 GB at V = 164k. Use
/// `sources` and/or `targets` — unlike `scipy`'s Dijkstra, passing `targets` here
/// means only those columns are ever allocated.
///
/// @param faces Integer or numeric `(F, 3)` matrix of triangle vertex indices
///   (0-based).
/// @param n_vertices Integer; total number of vertices in the mesh.
/// @param vertices Optional numeric `(V, 3)` matrix of vertex coordinates. If
///   given, edges are weighted by their euclidean length; if `NULL`, every edge has
///   weight 1 and the result is a hop count.
/// @param sources Optional integer vector of source vertex indices; `NULL` uses
///   every vertex.
/// @param targets Optional integer vector of target vertex indices; `NULL` uses
///   every vertex.
/// @param limit Optional numeric; ignore vertices further away than this.
/// @param threads Optional integer; number of threads. `NULL` uses all cores.
/// @return Numeric matrix of geodesic distances (sources in rows, targets in
///   columns). Unreachable pairs are `-1`.
/// @export
#[extendr]
pub fn geodesic_matrix_mesh(
    faces: Robj,
    n_vertices: i32,
    vertices: Option<Robj>,
    sources: Option<Vec<i32>>,
    targets: Option<Vec<i32>>,
    limit: Option<f64>,
    threads: Option<i32>,
) -> Robj {
    let faces = robj_to_faces(&faces);
    let coords = robj_to_coords(vertices);
    let sources = to_u32(sources);
    let targets = to_u32(targets);

    let dists = fastcore::mesh::geodesic_matrix_mesh(
        faces.view(),
        n_vertices as usize,
        coords.as_ref().map(|c| c.view()),
        sources.as_deref(),
        targets.as_deref(),
        limit.map(|l| l as f32),
        threads.map(|t| t as usize),
    );

    array2_f32_to_rmatrix(&dists)
}

/// Geodesic distances over an arbitrary graph given as an edge list.
///
/// The general form of `geodesic_matrix_mesh`. Unlike `geodesic_distances`, this
/// makes no tree assumption — cycles are fine.
///
/// @param edges Integer or numeric `(E, 2)` matrix of edges (0-based node indices).
/// @param n_nodes Integer; total number of nodes.
/// @param weights Optional numeric vector with one length per edge; `NULL` counts
///   edges. Must be finite and non-negative. Parallel edges collapse to the
///   shortest.
/// @param directed Logical; if `TRUE` an edge `(u, v)` may only be traversed from
///   `u` to `v`.
/// @param sources Optional integer vector of source node indices; `NULL` uses every
///   node.
/// @param targets Optional integer vector of target node indices; `NULL` uses every
///   node.
/// @param limit Optional numeric; ignore nodes further away than this.
/// @param threads Optional integer; number of threads. `NULL` uses all cores.
/// @return Numeric matrix of geodesic distances (sources in rows, targets in
///   columns). Unreachable pairs are `-1`.
/// @export
#[extendr]
pub fn geodesic_matrix_graph(
    edges: Robj,
    n_nodes: i32,
    weights: Option<Vec<f64>>,
    directed: bool,
    sources: Option<Vec<i32>>,
    targets: Option<Vec<i32>>,
    limit: Option<f64>,
    threads: Option<i32>,
) -> Robj {
    let edges = robj_to_edges(&edges);
    let weights: Option<Array1<f32>> =
        weights.map(|w| Array1::from_vec(w.iter().map(|x| *x as f32).collect()));
    let sources = to_u32(sources);
    let targets = to_u32(targets);

    let dists = fastcore::mesh::geodesic_matrix_graph(
        edges.view(),
        n_nodes as usize,
        weights.as_ref().map(|w| w.view()).as_ref(),
        directed,
        sources.as_deref(),
        targets.as_deref(),
        limit.map(|l| l as f32),
        threads.map(|t| t as usize),
    );

    array2_f32_to_rmatrix(&dists)
}

/// Distance to the nearest target vertex, for each source vertex of a mesh.
///
/// A memory-efficient alternative to `geodesic_matrix_mesh`: it keeps only the
/// nearest target and the distance to it, so the output is `O(sources)` rather than
/// `O(sources * targets)`. It is also faster, because the search stops at the first
/// target it settles rather than exploring the whole component.
///
/// Returns a list with `distances` and `nearest` (the vertex index of that nearest
/// target). Sources with no reachable target get `-1`. A source that is itself a
/// target is matched to its nearest *other* target, never to itself.
///
/// @param faces Integer or numeric `(F, 3)` matrix of triangle vertex indices.
/// @param n_vertices Integer; total number of vertices in the mesh.
/// @param vertices Optional numeric `(V, 3)` matrix of vertex coordinates; `NULL`
///   counts edges.
/// @param sources Optional integer vector of source vertex indices; `NULL` uses
///   every vertex.
/// @param targets Optional integer vector of target vertex indices; `NULL` uses
///   every vertex.
/// @param limit Optional numeric; ignore targets further away than this.
/// @param threads Optional integer; number of threads. `NULL` uses all cores.
/// @return A list with `distances` and `nearest`.
/// @export
#[extendr]
pub fn geodesic_nearest_mesh(
    faces: Robj,
    n_vertices: i32,
    vertices: Option<Robj>,
    sources: Option<Vec<i32>>,
    targets: Option<Vec<i32>>,
    limit: Option<f64>,
    threads: Option<i32>,
) -> Robj {
    let faces = robj_to_faces(&faces);
    let coords = robj_to_coords(vertices);
    let sources = to_u32(sources);
    let targets = to_u32(targets);

    let (dists, nearest) = fastcore::mesh::geodesic_nearest_mesh(
        faces.view(),
        n_vertices as usize,
        coords.as_ref().map(|c| c.view()),
        sources.as_deref(),
        targets.as_deref(),
        limit.map(|l| l as f32),
        threads.map(|t| t as usize),
    );

    list!(distances = dists.to_vec(), nearest = nearest.to_vec()).into()
}

/// Distance to the farthest target vertex, for each source vertex of a mesh.
///
/// The mirror image of `geodesic_nearest_mesh`, with the same `O(sources)` memory
/// footprint. Unlike `nearest`, this cannot stop early — it has to settle every
/// target — but the farthest one then comes for free, because the search settles
/// vertices in increasing order of distance.
///
/// Returns a list with `distances` and `farthest`. Sources with no reachable target
/// get `-1`. A source that is itself a target is matched to a *distinct* target.
///
/// @param faces Integer or numeric `(F, 3)` matrix of triangle vertex indices.
/// @param n_vertices Integer; total number of vertices in the mesh.
/// @param vertices Optional numeric `(V, 3)` matrix of vertex coordinates; `NULL`
///   counts edges.
/// @param sources Optional integer vector of source vertex indices; `NULL` uses
///   every vertex.
/// @param targets Optional integer vector of target vertex indices; `NULL` uses
///   every vertex.
/// @param limit Optional numeric; ignore targets further away than this.
/// @param threads Optional integer; number of threads. `NULL` uses all cores.
/// @return A list with `distances` and `farthest`.
/// @export
#[extendr]
pub fn geodesic_farthest_mesh(
    faces: Robj,
    n_vertices: i32,
    vertices: Option<Robj>,
    sources: Option<Vec<i32>>,
    targets: Option<Vec<i32>>,
    limit: Option<f64>,
    threads: Option<i32>,
) -> Robj {
    let faces = robj_to_faces(&faces);
    let coords = robj_to_coords(vertices);
    let sources = to_u32(sources);
    let targets = to_u32(targets);

    let (dists, farthest) = fastcore::mesh::geodesic_farthest_mesh(
        faces.view(),
        n_vertices as usize,
        coords.as_ref().map(|c| c.view()),
        sources.as_deref(),
        targets.as_deref(),
        limit.map(|l| l as f32),
        threads.map(|t| t as usize),
    );

    list!(distances = dists.to_vec(), farthest = farthest.to_vec()).into()
}

// ---------------------------------------------------------------------------
// NBLAST / synBLAST
// ---------------------------------------------------------------------------
//
// These mirror the Python bindings in `py/src/nblast.rs`. Point/tangent clouds
// are passed from R as *lists of (N, 3) numeric matrices*; per-neuron alphas and
// synapse types as lists of numeric vectors. The scoring matrix can be supplied
// as parts (`smat_values` + `dist_edges` + `dot_edges`) or defaulted to the
// embedded FCWB matrix. Unlike the Python side there is no cooperative Ctrl-C
// cancellation (R's `.Call` blocks until the compute returns); `cancel` is always
// `None`.

/// Convert one R (N, 3) numeric matrix into an owned point cloud.
fn robj_to_cloud(robj: &Robj) -> Vec<[f64; 3]> {
    let m = <RMatrix<f64>>::try_from(robj.clone())
        .expect("each cloud must be a numeric (N, 3) matrix");
    let nr = m.nrows();
    let d = m.data(); // column-major, length nr * ncols
    (0..nr).map(|i| [d[i], d[nr + i], d[2 * nr + i]]).collect()
}

/// Convert an R list of (N, 3) numeric matrices into owned point clouds.
fn to_clouds(list: &List) -> Vec<Vec<[f64; 3]>> {
    list.values().map(|robj| robj_to_cloud(&robj)).collect()
}

/// Convert an optional R list of per-point alpha vectors into owned Vecs. A NULL
/// `robj` (use_alpha off) yields `None`.
fn to_alphas(robj: Robj) -> Option<Vec<Vec<f64>>> {
    if robj.is_null() {
        return None;
    }
    let list = List::try_from(robj).expect("`alphas` must be a list of numeric vectors");
    Some(
        list.values()
            .map(|r| {
                r.as_real_slice()
                    .expect("alphas must be numeric vectors")
                    .to_vec()
            })
            .collect(),
    )
}

/// Convert an R list of per-connector integer type vectors into owned Vecs.
fn to_types(list: &List) -> Vec<Vec<i64>> {
    list.values()
        .map(|robj| {
            if let Some(s) = robj.as_integer_slice() {
                s.iter().map(|&x| x as i64).collect()
            } else if let Some(s) = robj.as_real_slice() {
                s.iter().map(|&x| x as i64).collect()
            } else {
                panic!("`types` must be integer or numeric vectors");
            }
        })
        .collect()
}

/// Convert an R (N, 3) integer/numeric matrix of faces into an owned `Array2<u32>`.
fn robj_to_faces(faces: &Robj) -> Array2<u32> {
    if let Ok(m) = <RMatrix<i32>>::try_from(faces.clone()) {
        let nr = m.nrows();
        let d = m.data();
        Array2::from_shape_fn((nr, 3), |(i, j)| d[j * nr + i] as u32)
    } else if let Ok(m) = <RMatrix<f64>>::try_from(faces.clone()) {
        let nr = m.nrows();
        let d = m.data();
        Array2::from_shape_fn((nr, 3), |(i, j)| d[j * nr + i] as u32)
    } else {
        panic!("`faces` must be a numeric (N, 3) matrix");
    }
}

/// Build a scoring matrix from supplied parts, or fall back to an embedded FCWB
/// matrix (alpha-calibrated when `use_alpha`). A NULL `smat_values` (or missing
/// edges) selects the fallback.
fn build_smat(
    smat_values: Robj,
    dist_edges: Option<Vec<f64>>,
    dot_edges: Option<Vec<f64>>,
    use_alpha: bool,
) -> Smat {
    if !smat_values.is_null() {
        if let (Some(de), Some(ve)) = (dist_edges, dot_edges) {
            let m = <RMatrix<f64>>::try_from(smat_values)
                .expect("`smat_values` must be a numeric matrix");
            let nr = m.nrows();
            let nc = m.ncols();
            let d = m.data(); // column-major
            let mut flat: Vec<f64> = Vec::with_capacity(nr * nc);
            for r in 0..nr {
                for c in 0..nc {
                    flat.push(d[c * nr + r]);
                }
            }
            return Smat::from_parts(flat, nr, nc, de, ve);
        }
    }
    if use_alpha {
        load_smat_alpha()
    } else {
        load_smat()
    }
}

/// Map an optional core count (<= 0 or NULL -> default global pool).
fn to_threads(n_cores: Option<i32>) -> Option<usize> {
    n_cores.and_then(|c| if c > 0 { Some(c as usize) } else { None })
}

/// Build an R numeric matrix from a row-major flat vector.
fn flat_to_rmatrix(flat: &[f64], nrows: usize, ncols: usize) -> Robj {
    RArray::new_matrix(nrows, ncols, |r, c| flat[r * ncols + c]).into()
}

/// Build an R numeric matrix from an `ndarray` `Array2<f64>`.
fn array2_to_rmatrix(arr: &Array2<f64>) -> Robj {
    let (nr, nc) = (arr.nrows(), arr.ncols());
    RArray::new_matrix(nr, nc, |r, c| arr[[r, c]]).into()
}

/// The `limit_dist="auto"` value for a scoring matrix.
///
/// @param smat_values Numeric scoring matrix, or `NULL` for the built-in FCWB
///   matrix.
/// @param dist_edges Numeric vector of distance bin edges for `smat_values`.
/// @param dot_edges Numeric vector of dot-product bin edges for `smat_values`.
/// @param use_alpha Logical; when falling back to the built-in matrix, use the
///   alpha-weighted variant.
/// @return Numeric `limit_dist` value implied by the scoring matrix.
/// @export
#[extendr]
pub fn smat_auto_limit(
    smat_values: Robj,
    dist_edges: Option<Vec<f64>>,
    dot_edges: Option<Vec<f64>>,
    use_alpha: bool,
) -> f64 {
    build_smat(smat_values, dist_edges, dot_edges, use_alpha).auto_limit()
}

/// All-by-all forward NBLAST.
///
/// `points`/`vects` are lists of (N, 3) matrices (one per neuron). Returns an
/// (n, n) score matrix; cell (i, j) is query i against target j.
///
/// @param points List of `(N, 3)` numeric matrices of neuron point coordinates.
/// @param vects List of `(N, 3)` numeric matrices of unit tangent vectors, one
///   per neuron and aligned with `points`.
/// @param alphas Optional list of per-point alpha (anisotropy) vectors; `NULL`
///   disables alpha weighting.
/// @param smat_values Numeric scoring matrix, or `NULL` for the built-in FCWB
///   matrix.
/// @param dist_edges Numeric vector of distance bin edges for `smat_values`.
/// @param dot_edges Numeric vector of dot-product bin edges for `smat_values`.
/// @param normalize Logical; normalise each score by the query self-match score.
/// @param limit_dist Optional numeric distance cut-off; `NULL` disables it.
/// @param n_cores Optional integer thread count; `NULL` or `<= 0` uses all cores.
/// @param precision Integer; compute in 32- or 64-bit floats.
/// @param progress Logical; display a progress bar.
/// @return Numeric `(n, n)` score matrix; cell `(i, j)` is query `i` vs target `j`.
/// @export
#[extendr]
#[allow(clippy::too_many_arguments)]
pub fn nblast_allbyall(
    points: List,
    vects: List,
    alphas: Robj,
    smat_values: Robj,
    dist_edges: Option<Vec<f64>>,
    dot_edges: Option<Vec<f64>>,
    normalize: bool,
    limit_dist: Option<f64>,
    n_cores: Option<i32>,
    precision: i32,
    progress: bool,
) -> Robj {
    let clouds = to_clouds(&points);
    let vecs = to_clouds(&vects);
    let alpha_vecs = to_alphas(alphas);
    let smat = build_smat(smat_values, dist_edges, dot_edges, alpha_vecs.is_some());
    let n = clouds.len();
    let opts = Opts {
        smat: &smat,
        normalize,
        limit_dist,
        threads: to_threads(n_cores),
        progress,
        cancel: None,
    };

    let flat: Vec<f64> = match precision {
        32 => fastcore::nblast::nblast_allbyall::<f32>(clouds, vecs, alpha_vecs, opts)
            .into_iter()
            .map(|x| x as f64)
            .collect(),
        64 => fastcore::nblast::nblast_allbyall::<f64>(clouds, vecs, alpha_vecs, opts),
        _ => panic!("`precision` must be 32 or 64"),
    };
    flat_to_rmatrix(&flat, n, n)
}

/// Forward NBLAST of every query neuron against every target neuron.
///
/// Returns an (n_query, n_target) score matrix.
///
/// @param q_points List of `(N, 3)` numeric matrices of query point coordinates.
/// @param q_vects List of `(N, 3)` numeric matrices of query tangent vectors.
/// @param t_points List of `(N, 3)` numeric matrices of target point coordinates.
/// @param t_vects List of `(N, 3)` numeric matrices of target tangent vectors.
/// @param q_alphas Optional list of per-point alpha vectors for the queries;
///   `NULL` disables alpha weighting.
/// @param t_alphas Optional list of per-point alpha vectors for the targets.
/// @param smat_values Numeric scoring matrix, or `NULL` for the built-in FCWB
///   matrix.
/// @param dist_edges Numeric vector of distance bin edges for `smat_values`.
/// @param dot_edges Numeric vector of dot-product bin edges for `smat_values`.
/// @param normalize Logical; normalise each score by the query self-match score.
/// @param limit_dist Optional numeric distance cut-off; `NULL` disables it.
/// @param n_cores Optional integer thread count; `NULL` or `<= 0` uses all cores.
/// @param precision Integer; compute in 32- or 64-bit floats.
/// @param progress Logical; display a progress bar.
/// @return Numeric `(n_query, n_target)` score matrix.
/// @export
#[extendr]
#[allow(clippy::too_many_arguments)]
pub fn nblast(
    q_points: List,
    q_vects: List,
    t_points: List,
    t_vects: List,
    q_alphas: Robj,
    t_alphas: Robj,
    smat_values: Robj,
    dist_edges: Option<Vec<f64>>,
    dot_edges: Option<Vec<f64>>,
    normalize: bool,
    limit_dist: Option<f64>,
    n_cores: Option<i32>,
    precision: i32,
    progress: bool,
) -> Robj {
    let qp = to_clouds(&q_points);
    let qv = to_clouds(&q_vects);
    let tp = to_clouds(&t_points);
    let tv = to_clouds(&t_vects);
    let qa = to_alphas(q_alphas);
    let ta = to_alphas(t_alphas);
    let smat = build_smat(smat_values, dist_edges, dot_edges, qa.is_some());
    let (nq, nt) = (qp.len(), tp.len());
    let opts = Opts {
        smat: &smat,
        normalize,
        limit_dist,
        threads: to_threads(n_cores),
        progress,
        cancel: None,
    };

    let flat: Vec<f64> = match precision {
        32 => fastcore::nblast::nblast_query_target::<f32>(qp, qv, qa, tp, tv, ta, opts)
            .into_iter()
            .map(|x| x as f64)
            .collect(),
        64 => fastcore::nblast::nblast_query_target::<f64>(qp, qv, qa, tp, tv, ta, opts),
        _ => panic!("`precision` must be 32 or 64"),
    };
    flat_to_rmatrix(&flat, nq, nt)
}

/// Forward NBLAST for a set of `(query, target)` index pairs.
///
/// `q_idx`/`t_idx` are 0-based indices into the query/target lists; element k of
/// the result is query `q_idx[k]` against target `t_idx[k]`.
///
/// @param q_points List of `(N, 3)` numeric matrices of query point coordinates.
/// @param q_vects List of `(N, 3)` numeric matrices of query tangent vectors.
/// @param t_points List of `(N, 3)` numeric matrices of target point coordinates.
/// @param t_vects List of `(N, 3)` numeric matrices of target tangent vectors.
/// @param q_idx Integer vector of 0-based query indices, one per pair.
/// @param t_idx Integer vector of 0-based target indices (same length as `q_idx`).
/// @param q_alphas Optional list of per-point alpha vectors for the queries;
///   `NULL` disables alpha weighting.
/// @param t_alphas Optional list of per-point alpha vectors for the targets.
/// @param smat_values Numeric scoring matrix, or `NULL` for the built-in FCWB
///   matrix.
/// @param dist_edges Numeric vector of distance bin edges for `smat_values`.
/// @param dot_edges Numeric vector of dot-product bin edges for `smat_values`.
/// @param normalize Logical; normalise each score by the query self-match score.
/// @param limit_dist Optional numeric distance cut-off; `NULL` disables it.
/// @param n_cores Optional integer thread count; `NULL` or `<= 0` uses all cores.
/// @param precision Integer; compute in 32- or 64-bit floats.
/// @param progress Logical; display a progress bar.
/// @return Numeric vector of scores, one per `(query, target)` pair.
/// @export
#[extendr]
#[allow(clippy::too_many_arguments)]
pub fn nblast_pairs(
    q_points: List,
    q_vects: List,
    t_points: List,
    t_vects: List,
    q_idx: Vec<i32>,
    t_idx: Vec<i32>,
    q_alphas: Robj,
    t_alphas: Robj,
    smat_values: Robj,
    dist_edges: Option<Vec<f64>>,
    dot_edges: Option<Vec<f64>>,
    normalize: bool,
    limit_dist: Option<f64>,
    n_cores: Option<i32>,
    precision: i32,
    progress: bool,
) -> Vec<f64> {
    let qp = to_clouds(&q_points);
    let qv = to_clouds(&q_vects);
    let tp = to_clouds(&t_points);
    let tv = to_clouds(&t_vects);
    let qa = to_alphas(q_alphas);
    let ta = to_alphas(t_alphas);
    let smat = build_smat(smat_values, dist_edges, dot_edges, qa.is_some());

    if q_idx.len() != t_idx.len() {
        panic!("`q_idx` and `t_idx` must have the same length");
    }
    let pairs: Vec<(usize, usize)> = q_idx
        .iter()
        .zip(t_idx.iter())
        .map(|(&a, &b)| (a as usize, b as usize))
        .collect();

    let opts = Opts {
        smat: &smat,
        normalize,
        limit_dist,
        threads: to_threads(n_cores),
        progress,
        cancel: None,
    };

    match precision {
        32 => fastcore::nblast::nblast_pairs::<f32>(qp, qv, qa, tp, tv, ta, pairs, opts)
            .into_iter()
            .map(|x| x as f64)
            .collect(),
        64 => fastcore::nblast::nblast_pairs::<f64>(qp, qv, qa, tp, tv, ta, pairs, opts),
        _ => panic!("`precision` must be 32 or 64"),
    }
}

/// All-by-all forward syNBLAST over synapse clouds.
///
/// `points` are lists of (N, 3) connector coordinate matrices and `types` the
/// matching per-connector integer type ids. Returns an (n, n) score matrix.
///
/// @param points List of `(N, 3)` numeric matrices of connector coordinates,
///   one per neuron.
/// @param types List of integer vectors of per-connector type ids, aligned with
///   `points`.
/// @param smat_values Numeric scoring matrix, or `NULL` for the built-in FCWB
///   matrix.
/// @param dist_edges Numeric vector of distance bin edges for `smat_values`.
/// @param dot_edges Numeric vector of dot-product bin edges for `smat_values`.
/// @param normalize Logical; normalise each score by the query self-match score.
/// @param n_cores Optional integer thread count; `NULL` or `<= 0` uses all cores.
/// @param precision Integer; compute in 32- or 64-bit floats.
/// @param progress Logical; display a progress bar.
/// @return Numeric `(n, n)` score matrix; cell `(i, j)` is query `i` vs target `j`.
/// @export
#[extendr]
#[allow(clippy::too_many_arguments)]
pub fn synblast_allbyall(
    points: List,
    types: List,
    smat_values: Robj,
    dist_edges: Option<Vec<f64>>,
    dot_edges: Option<Vec<f64>>,
    normalize: bool,
    n_cores: Option<i32>,
    precision: i32,
    progress: bool,
) -> Robj {
    let clouds = to_clouds(&points);
    let tys = to_types(&types);
    let smat = build_smat(smat_values, dist_edges, dot_edges, false);
    let n = clouds.len();
    let opts = Opts {
        smat: &smat,
        normalize,
        limit_dist: None,
        threads: to_threads(n_cores),
        progress,
        cancel: None,
    };

    let flat: Vec<f64> = match precision {
        32 => fastcore::synblast::synblast_allbyall::<f32>(clouds, tys, opts)
            .into_iter()
            .map(|x| x as f64)
            .collect(),
        64 => fastcore::synblast::synblast_allbyall::<f64>(clouds, tys, opts),
        _ => panic!("`precision` must be 32 or 64"),
    };
    flat_to_rmatrix(&flat, n, n)
}

/// Forward syNBLAST of every query neuron against every target neuron.
///
/// Returns an (n_query, n_target) score matrix.
///
/// @param q_points List of `(N, 3)` numeric matrices of query connector
///   coordinates.
/// @param q_types List of integer vectors of query per-connector type ids.
/// @param t_points List of `(N, 3)` numeric matrices of target connector
///   coordinates.
/// @param t_types List of integer vectors of target per-connector type ids.
/// @param smat_values Numeric scoring matrix, or `NULL` for the built-in FCWB
///   matrix.
/// @param dist_edges Numeric vector of distance bin edges for `smat_values`.
/// @param dot_edges Numeric vector of dot-product bin edges for `smat_values`.
/// @param normalize Logical; normalise each score by the query self-match score.
/// @param n_cores Optional integer thread count; `NULL` or `<= 0` uses all cores.
/// @param precision Integer; compute in 32- or 64-bit floats.
/// @param progress Logical; display a progress bar.
/// @return Numeric `(n_query, n_target)` score matrix.
/// @export
#[extendr]
#[allow(clippy::too_many_arguments)]
pub fn synblast(
    q_points: List,
    q_types: List,
    t_points: List,
    t_types: List,
    smat_values: Robj,
    dist_edges: Option<Vec<f64>>,
    dot_edges: Option<Vec<f64>>,
    normalize: bool,
    n_cores: Option<i32>,
    precision: i32,
    progress: bool,
) -> Robj {
    let qp = to_clouds(&q_points);
    let qt = to_types(&q_types);
    let tp = to_clouds(&t_points);
    let tt = to_types(&t_types);
    let smat = build_smat(smat_values, dist_edges, dot_edges, false);
    let (nq, nt) = (qp.len(), tp.len());
    let opts = Opts {
        smat: &smat,
        normalize,
        limit_dist: None,
        threads: to_threads(n_cores),
        progress,
        cancel: None,
    };

    let flat: Vec<f64> = match precision {
        32 => fastcore::synblast::synblast_query_target::<f32>(qp, qt, tp, tt, opts)
            .into_iter()
            .map(|x| x as f64)
            .collect(),
        64 => fastcore::synblast::synblast_query_target::<f64>(qp, qt, tp, tt, opts),
        _ => panic!("`precision` must be 32 or 64"),
    };
    flat_to_rmatrix(&flat, nq, nt)
}

// ---------------------------------------------------------------------------
// CMTK transforms
// ---------------------------------------------------------------------------

/// An `(N, 3)` R matrix -> row-major coordinates. R matrices are column-major.
fn rmatrix_to_coords(m: &RMatrix<f64>, arg: &str) -> Array2<f64> {
    assert!(
        m.ncols() == 3,
        "`{arg}` must be an (N, 3) matrix of 3D coordinates, got {} column(s)",
        m.ncols()
    );
    let nr = m.nrows();
    let d = m.data();
    Array2::from_shape_fn((nr, 3), |(i, j)| d[j * nr + i])
}

fn coords_to_rmatrix(arr: &Array2<f64>) -> Robj {
    RArray::new_matrix(arr.nrows(), arr.ncols(), |r, c| arr[[r, c]]).into()
}

/// The R wrapper has already turned `FALSE`/`TRUE`/`"chain"`/`"hop"` into one of these.
fn cmtk_fallback(fallback: &str) -> Fallback {
    match fallback {
        "none" => Fallback::None,
        "chain" => Fallback::Chain,
        "hop" => Fallback::Hop,
        other => panic!(
            "`fallback_to_affine` must be FALSE, TRUE, \"chain\" or \"hop\", got \"{other}\""
        ),
    }
}

fn cmtk_mode(transform: &str) -> Mode {
    match transform {
        "warp" => Mode::Warp,
        "affine" => Mode::Affine,
        other => panic!("`transform` must be \"warp\" or \"affine\", got \"{other}\""),
    }
}

/// Per-hop direction flags as they cross from R: 0/1 per hop, empty for the all-forward
/// default. An integer vector rather than a logical one because extendr cannot take a
/// `Vec<bool>` as *input*; the R wrappers take a proper logical and convert.
fn invert_flags(invert: Vec<i32>) -> Option<Vec<bool>> {
    if invert.is_empty() {
        return None;
    }
    Some(invert.iter().map(|&i| i != 0).collect())
}

/// A loaded CMTK registration, or a chain of them.
///
/// Held behind an external pointer so the registration is parsed **once** and then applied
/// as often as you like — a real registration is ~17k control points read from a 760 KB
/// file, and `xform_brain`-style code applies it to every neuron in a dataset.
pub struct CmtkRegistration {
    chain: Chain,
    paths: Vec<String>,
}

#[extendr]
impl CmtkRegistration {
    /// Read one or more registrations.
    ///
    /// The pointer holds only the *parse*; direction is passed per call to `xform`/`xform_inv`,
    /// so one object serves every direction.
    ///
    /// NB: extendr 0.7 cannot turn an `Err` into an R condition -- it unwraps it -- and a
    /// panic raised from an *associated* function (unlike a method) loses its payload, so R
    /// would only ever see "User function panicked: load". `cmtk_read()` therefore validates
    /// the paths before we get here; this panic is the backstop for a corrupt file.
    fn load(paths: Vec<String>) -> Self {
        let pbs: Vec<std::path::PathBuf> = paths.iter().map(std::path::PathBuf::from).collect();
        let chain = Chain::from_paths(&pbs).unwrap_or_else(|e| panic!("{e}"));
        CmtkRegistration { chain, paths }
    }

    fn n_registrations(&self) -> i32 {
        self.chain.n_registrations() as i32
    }

    fn paths(&self) -> Vec<String> {
        self.paths.clone()
    }

    fn versions(&self) -> Vec<String> {
        self.chain.regs.iter().map(|r| r.version.clone()).collect()
    }

    fn has_spline(&self) -> Vec<bool> {
        self.chain.regs.iter().map(|r| r.spline.is_some()).collect()
    }

    /// The 4x4 affine of the first registration, or `NULL` if it has none.
    fn affine(&self) -> Robj {
        match self.chain.regs[0].affine {
            Some(a) => coords_to_rmatrix(&a.as_array()),
            None => NULL.into(),
        }
    }

    /// Control-point lattice dimensions of each spline warp, as a `(k, 3)` matrix.
    fn dims(&self) -> Robj {
        let rows: Vec<[usize; 3]> = self
            .chain
            .regs
            .iter()
            .filter_map(|r| r.spline.as_ref().map(|s| s.dims))
            .collect();
        if rows.is_empty() {
            return NULL.into();
        }
        RArray::new_matrix(rows.len(), 3, |r, c| rows[r][c] as f64).into()
    }

    /// Control-point spacing of each spline warp, as a `(k, 3)` matrix.
    fn spacing(&self) -> Robj {
        let rows: Vec<[f64; 3]> = self
            .chain
            .regs
            .iter()
            .filter_map(|r| r.spline.as_ref().map(|s| s.spacing))
            .collect();
        if rows.is_empty() {
            return NULL.into();
        }
        RArray::new_matrix(rows.len(), 3, |r, c| rows[r][c]).into()
    }

    /// The domain box of each spline warp, as a `(k, 3)` matrix. Points outside `[0, domain]`
    /// cannot be transformed — CMTK reports them as FAILED and we return `NaN`.
    fn domain(&self) -> Robj {
        let rows: Vec<[f64; 3]> = self
            .chain
            .regs
            .iter()
            .filter_map(|r| r.spline.as_ref().map(|s| s.domain))
            .collect();
        if rows.is_empty() {
            return NULL.into();
        }
        RArray::new_matrix(rows.len(), 3, |r, c| rows[r][c]).into()
    }

    #[allow(clippy::too_many_arguments)]
    fn xform(
        &self,
        coords: RMatrix<f64>,
        transform: &str,
        allow_extrapolation: bool,
        fallback_to_affine: &str,
        invert: Vec<i32>,
        n_cores: Option<i32>,
        progress: bool,
    ) -> Robj {
        let pts = rmatrix_to_coords(&coords, "xyz");
        let flags = invert_flags(invert);
        let opts = XformOpts {
            mode: cmtk_mode(transform),
            allow_extrapolation,
            fallback: cmtk_fallback(fallback_to_affine),
            invert: flags.as_deref(),
            threads: n_cores.map(|n| n.max(1) as usize),
            progress,
            cancel: None,
        };
        let out = cmtk::transform_points(&self.chain, pts.view(), opts)
            .unwrap_or_else(|e| panic!("{e}"));
        coords_to_rmatrix(&out)
    }

    #[allow(clippy::too_many_arguments)]
    fn xform_inv(
        &self,
        coords: RMatrix<f64>,
        transform: &str,
        initial_guess: Option<Robj>,
        max_iter: i32,
        tolerance: f64,
        accuracy: f64,
        clamp_to_domain: bool,
        fallback_to_affine: &str,
        invert: Vec<i32>,
        n_cores: Option<i32>,
        progress: bool,
    ) -> Robj {
        let pts = rmatrix_to_coords(&coords, "xyz");
        // R's NULL arrives as Some(Robj::null()), not None -- see `robj_to_coords`.
        let guess: Option<Array2<f64>> = initial_guess
            .filter(|g| !g.is_null())
            .map(|g| {
                let m = <RMatrix<f64>>::try_from(g)
                    .expect("`initial_guess` must be a numeric (N, 3) matrix");
                rmatrix_to_coords(&m, "initial_guess")
            });
        let flags = invert_flags(invert);
        let opts = InverseOpts {
            mode: cmtk_mode(transform),
            max_iter: max_iter.max(1) as usize,
            tolerance,
            accuracy,
            clamp_to_domain,
            fallback: cmtk_fallback(fallback_to_affine),
            invert: flags.as_deref(),
            threads: n_cores.map(|n| n.max(1) as usize),
            progress,
            cancel: None,
        };
        let out = cmtk::inverse_transform_points(
            &self.chain,
            pts.view(),
            guess.as_ref().map(|g| g.view()),
            opts,
        )
        .unwrap_or_else(|e| panic!("{e}"));
        coords_to_rmatrix(&out)
    }
}

// ---------------------------------------------------------------------------
// Elastix transforms
// ---------------------------------------------------------------------------

fn elastix_oob(out_of_bounds: &str) -> OutOfBounds {
    match out_of_bounds {
        "identity" => OutOfBounds::Identity,
        "nan" => OutOfBounds::Nan,
        other => panic!("`out_of_bounds` must be \"identity\" or \"nan\", got \"{other}\""),
    }
}

// Whether an Elastix transform can be inverted, without reading its coefficients.
//
// A file is not invertible exactly when some step in its chain combines via `Add`. That key sits
// *after* a coefficient array that can run to 56 MB, so answering it used to cost a full parse.
// This skips only the numbers: ~20x faster, ~200x on the big ones.
//
// Deliberately NOT a `///` doc comment: rextendr turns those into roxygen, which would generate an
// .Rd for an internal function. The exported `elastix_probe_invertible()` wraps this and validates
// the path R-side first (extendr cannot carry a panic's message across to R), so the panic here is
// the backstop for a corrupt file.
#[extendr]
fn probe_invertible_raw(path: &str) -> bool {
    elastix::probe_invertible(std::path::Path::new(path)).unwrap_or_else(|e| panic!("{e}"))
}

/// A loaded Elastix transform, or a chain of them.
///
/// Held behind an external pointer so the file is parsed **once** and then applied as often as
/// you like -- BANC's `BANC_to_template.txt` is 56 MB, and `xform_brain`-style code applies a
/// transform to every neuron in a dataset.
pub struct ElastixTransformPtr {
    chain: elastix::Chain,
    paths: Vec<String>,
}

#[extendr]
impl ElastixTransformPtr {
    /// Read one or more `TransformParameters` files.
    ///
    /// The pointer holds only the *parse*; direction is passed per call to `xform`/`xform_inv`,
    /// so one object serves every direction. That is worth caring about here: BANC's warp is
    /// 56 MB, and re-reading it just to walk it backwards would be absurd.
    ///
    /// NB: extendr 0.7 cannot turn an `Err` into an R condition -- it unwraps it -- and a panic
    /// raised from an *associated* function (unlike a method) loses its payload, so R would only
    /// ever see "User function panicked: load". `elastix_read()` therefore validates the paths
    /// before we get here; this panic is the backstop for a corrupt file.
    fn load(paths: Vec<String>) -> Self {
        let pbs: Vec<std::path::PathBuf> = paths.iter().map(std::path::PathBuf::from).collect();
        let chain = elastix::Chain::from_paths(&pbs).unwrap_or_else(|e| panic!("{e}"));
        ElastixTransformPtr { chain, paths }
    }

    fn n_transforms(&self) -> i32 {
        self.chain.n_transforms() as i32
    }

    fn paths(&self) -> Vec<String> {
        self.paths.clone()
    }

    /// Whether `xform_inv` can run at all. `elastix_xform_inv()` asks before calling in: extendr
    /// cannot carry a Rust panic's message across to R (it arrives as the useless "User function
    /// panicked: xform_inv"), so the check has to happen on the R side to produce a real error.
    fn invertible(&self) -> bool {
        self.chain.is_invertible(None)
    }

    /// The resolved step kinds of each transform, initial first, one string per transform
    /// (e.g. `"linear+bspline"`).
    fn kinds(&self) -> Vec<String> {
        self.chain
            .xforms
            .iter()
            .map(|x| {
                x.steps
                    .iter()
                    .map(|(t, _)| t.kind())
                    .collect::<Vec<_>>()
                    .join("+")
            })
            .collect()
    }

    /// The 4x4 matrix of the first linear step of the first transform, or `NULL`.
    fn affine(&self) -> Robj {
        match self.chain.xforms[0].linear() {
            Some(l) => coords_to_rmatrix(&l.as_array()),
            None => NULL.into(),
        }
    }

    /// Control-point grid size of every B-spline in the chain, as a `(k, 3)` matrix.
    fn grid_size(&self) -> Robj {
        let rows: Vec<[usize; 3]> = self
            .chain
            .xforms
            .iter()
            .flat_map(|x| x.splines())
            .map(|s| s.size)
            .collect();
        if rows.is_empty() {
            return NULL.into();
        }
        RArray::new_matrix(rows.len(), 3, |r, c| rows[r][c] as f64).into()
    }

    /// Control-point spacing of every B-spline in the chain, as a `(k, 3)` matrix.
    fn grid_spacing(&self) -> Robj {
        let rows: Vec<[f64; 3]> = self
            .chain
            .xforms
            .iter()
            .flat_map(|x| x.splines())
            .map(|s| s.spacing)
            .collect();
        if rows.is_empty() {
            return NULL.into();
        }
        RArray::new_matrix(rows.len(), 3, |r, c| rows[r][c]).into()
    }

    /// Control-point grid origin of every B-spline in the chain, as a `(k, 3)` matrix.
    fn grid_origin(&self) -> Robj {
        let rows: Vec<[f64; 3]> = self
            .chain
            .xforms
            .iter()
            .flat_map(|x| x.splines())
            .map(|s| s.origin)
            .collect();
        if rows.is_empty() {
            return NULL.into();
        }
        RArray::new_matrix(rows.len(), 3, |r, c| rows[r][c]).into()
    }

    fn xform(
        &self,
        coords: RMatrix<f64>,
        out_of_bounds: &str,
        invert: Vec<i32>,
        n_cores: Option<i32>,
        progress: bool,
    ) -> Robj {
        let pts = rmatrix_to_coords(&coords, "xyz");
        let flags = invert_flags(invert);
        let opts = elastix::XformOpts {
            out_of_bounds: elastix_oob(out_of_bounds),
            invert: flags.as_deref(),
            threads: n_cores.map(|n| n.max(1) as usize),
            progress,
            cancel: None,
        };
        let out = elastix::transform_points(&self.chain, pts.view(), opts)
            .unwrap_or_else(|e| panic!("{e}"));
        coords_to_rmatrix(&out)
    }

    #[allow(clippy::too_many_arguments)]
    fn xform_inv(
        &self,
        coords: RMatrix<f64>,
        out_of_bounds: &str,
        initial_guess: Option<Robj>,
        max_iter: i32,
        seed_iter: i32,
        tolerance: f64,
        accuracy: f64,
        lattice_points: i32,
        invert: Vec<i32>,
        n_cores: Option<i32>,
        progress: bool,
    ) -> Robj {
        let pts = rmatrix_to_coords(&coords, "xyz");
        // R's NULL arrives as Some(Robj::null()), not None.
        let guess: Option<Array2<f64>> = initial_guess.filter(|g| !g.is_null()).map(|g| {
            let m = <RMatrix<f64>>::try_from(g)
                .expect("`initial_guess` must be a numeric (N, 3) matrix");
            rmatrix_to_coords(&m, "initial_guess")
        });
        let flags = invert_flags(invert);
        let opts = elastix::InverseOpts {
            out_of_bounds: elastix_oob(out_of_bounds),
            max_iter: max_iter.max(1) as usize,
            seed_iter: seed_iter.max(0) as usize,
            tolerance,
            accuracy,
            lattice_points: lattice_points.max(0) as usize,
            invert: flags.as_deref(),
            threads: n_cores.map(|n| n.max(1) as usize),
            progress,
            cancel: None,
        };
        let out = elastix::inverse_transform_points(
            &self.chain,
            pts.view(),
            guess.as_ref().map(|g| g.view()),
            opts,
        )
        .unwrap_or_else(|e| panic!("{e}"));
        coords_to_rmatrix(&out)
    }
}

// Macro to generate exports.
// This ensures exported functions are registered with R.
// See corresponding C code in `entrypoint.c`.
extendr_module! {
    mod nat_fastcore;
    impl CmtkRegistration;
    impl ElastixTransformPtr;
    fn probe_invertible_raw;
    fn all_dists_to_root;
    fn node_indices;
    fn geodesic_distances;
    fn strahler_index;
    fn subtree_height;
    fn connected_components;
    fn prune_twigs;
    fn child_to_parent_dists;
    fn dist_to_root;
    fn classify_nodes;
    fn has_cycles;
    fn geodesic_pairs;
    fn geodesic_nearest;
    fn geodesic_farthest;
    fn synapse_flow_centrality;
    fn generate_segments;
    fn break_segments;
    fn stitch_fragments;
    fn reroot_rewire;
    fn heal_skeleton;
    fn mesh_connected_components;
    fn geodesic_matrix_mesh;
    fn geodesic_matrix_graph;
    fn geodesic_nearest_mesh;
    fn geodesic_farthest_mesh;
    fn smat_auto_limit;
    fn nblast_allbyall;
    fn nblast;
    fn nblast_pairs;
    fn synblast_allbyall;
    fn synblast;
}
