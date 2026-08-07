use extendr_api::prelude::*;
use extendr_api::{AsTypedSlice, ToVectorValue};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut2, ShapeBuilder};
use std::collections::HashMap;

use fastcore::cmtk::{self, Chain, Fallback, InverseOpts, Mode, XformOpts};
use fastcore::mls::MlsTransform;
use fastcore::tps::TpsTransform;
use fastcore::elastix::{self, OutOfBounds};
use fastcore::linkage::{
    condense, leaf_order, linkage as core_linkage, linkage_from_scores,
    observations_from_condensed, symmetrize, Method as LinkageMethod, Symmetry, Transform,
};
use fastcore::mesh::Weight;
use fastcore::nblast::{load_smat, load_smat_alpha, Opts, Smat};
use fastcore::nblast_knn::{KnnOpts, Symmetry as KnnSymmetry};

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

    array2_to_r(&dists, |x| x)
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
/// @param threads Optional integer; number of threads. `NULL` uses all cores.
/// @return Numeric vector with the distance of each `(source, target)` pair.
/// @export
#[extendr]
pub fn geodesic_pairs(
    parents: Vec<i32>,
    sources: Vec<i32>,
    targets: Vec<i32>,
    weights: Option<Vec<f64>>,
    directed: bool,
    #[default = "NULL"] threads: Option<i32>,
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
        threads.map(|t| t as usize),
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

/// Optional per-element lengths as an R numeric vector, or `NULL`.
///
/// extendr maps `f32` straight to a REALSXP, so there is no widening to do here — only
/// the `Option` -> `NULL` mapping, which two functions need and would otherwise spell
/// out identically.
fn opt_lengths(v: Option<Vec<f32>>) -> Robj {
    match v {
        Some(w) => w.into(),
        None => ().into(),
    }
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
    list!(segments = seg_list, lengths = opt_lengths(lengths)).into()
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
// Tree traversal and editing
// ---------------------------------------------------------------------------
//
// These mirror the Python bindings in `py/src/dag.rs`. Each looks like a general
// graph algorithm but is a linear pass over the parent vector, so building a graph
// object to answer it costs more than the answer does. Node references are 0-based
// indices throughout, like the rest of the DAG family, and roots are `< 0`.

/// Coerce optional R weights to the width the search will run at.
fn to_weights<W: Weight>(weights: Option<Vec<f64>>) -> Option<Array1<W>> {
    weights.map(|w| w.into_iter().map(W::from_f64).collect())
}

/// Resolve a `precision` argument (32 or 64) to "run the search in double precision".
///
/// R has no float32 type, so — unlike Python, where the weights array carries its own
/// width and the answer comes back in it — there is nothing here to read the choice
/// off: weights arrive as doubles whatever the caller meant by them, and the result
/// goes back as doubles either way. An explicit argument is what is left, and it
/// mirrors `nblast(precision = )`, which asks the same question of the scoring
/// pipeline.
///
/// 32 stays the default. Distances are accumulated one edge at a time, so 64 is worth
/// asking for when the *paths* are long (tens of thousands of hops) or the weights
/// span a wide dynamic range; for ordinary mesh and skeleton work a 24-bit mantissa
/// resolves a 100 mm neuron to ~6 nm, and the distance matrix is by far the largest
/// thing these functions allocate.
fn wide(precision: i32) -> bool {
    match precision {
        32 => false,
        64 => true,
        _ => panic!("`precision` must be 32 or 64"),
    }
}

/// Turn a list-of-paths result into an R list of integer vectors.
fn paths_to_list<I>(paths: I) -> Robj
where
    I: IntoIterator,
    I::IntoIter: ExactSizeIterator,
    I::Item: Into<Robj>,
{
    List::from_values(paths).into()
}

/// Every node in the sub-tree below each source.
///
/// The replacement for `igraph::subcomponent(mode = "in")`, and what makes "cut the
/// skeleton here" a masking operation rather than a graph rebuild.
///
/// @param parents Integer vector of 0-based parent indices (roots are `< 0`).
/// @param sources Integer vector of 0-based node indices to walk down from.
/// @return List of integer vectors, one per source in `sources` order. Each starts
///   with the source itself and is in depth-first pre-order, so a node always
///   precedes its own descendants. An out-of-range source gives an empty vector.
/// @export
#[extendr]
pub fn descendants(parents: Vec<i32>, sources: Vec<i32>) -> Robj {
    let parents = Array1::from_vec(parents);
    let sources = Array1::from_vec(sources);
    paths_to_list(fastcore::dag::descendants(&parents.view(), &sources.view()))
}

/// The path from each source up to its root.
///
/// The other direction of the same walk as `descendants`.
///
/// @param parents Integer vector of 0-based parent indices (roots are `< 0`).
/// @param sources Integer vector of 0-based node indices to walk up from.
/// @return List of integer vectors, one per source in `sources` order, ordered
///   source-first / root-last. A source that is itself a root gives a
///   single-element vector; an out-of-range source gives an empty one.
/// @export
#[extendr]
pub fn paths_to_root(parents: Vec<i32>, sources: Vec<i32>) -> Robj {
    let parents = Array1::from_vec(parents);
    let sources = Array1::from_vec(sources);
    paths_to_list(fastcore::dag::paths_to_root(&parents.view(), &sources.view()))
}

/// Re-orient a forest so that each of `new_roots` becomes its component's root.
///
/// Only the edges between each new root and its component's old root are reversed;
/// components nobody names come back byte-identical. The general form of
/// `reroot_rewire`, which takes one preferred root plus a set of new edges.
///
/// @param parents Integer vector of 0-based parent indices (roots are `< 0`).
/// @param new_roots Integer vector of 0-based node indices to root at. Two roots in
///   the same component: the last one named wins.
/// @return Integer vector of new 0-based parent indices, aligned with `parents`.
///   Roots are `-1`.
/// @export
#[extendr]
pub fn reroot(parents: Vec<i32>, new_roots: Vec<i32>) -> Vec<i32> {
    let parents = Array1::from_vec(parents);
    let new_roots = Array1::from_vec(new_roots);
    fastcore::dag::reroot(&parents.view(), &new_roots.view()).to_vec()
}

/// Collapse groups of nodes onto a representative and rewire what is left.
///
/// Edges internal to a group are dropped rather than turned into self-loops.
///
/// @param parents Integer vector of 0-based parent indices (roots are `< 0`).
/// @param mapping Integer vector, one entry per node, giving the 0-based index of
///   the node it collapses onto. A node mapped to itself survives.
/// @return List with `nodes` (0-based indices of the surviving nodes, in their
///   original relative order) and `parents` (their new 0-based parent indices,
///   `-1` for roots, indexing *into* `nodes`).
/// @export
#[extendr]
pub fn contract_nodes(parents: Vec<i32>, mapping: Vec<i32>) -> Robj {
    // Everything Rust owns is confined to this block. `throw_r_error` is a longjmp, so
    // it runs no destructors — anything still live at the throw would leak, and here
    // that would be two node-sized arrays.
    let result = {
        let parents = Array1::from_vec(parents);
        let mapping = Array1::from_vec(mapping);
        fastcore::dag::contract_nodes(&parents.view(), &mapping.view())
            .map(|(nodes, new_parents)| -> Robj {
                list!(nodes = nodes, parents = new_parents.to_vec()).into()
            })
            .map_err(|e| e.to_string())
    };
    // A mapping that would close a cycle is refused rather than silently returning a
    // non-forest, so surface it as an R error rather than as extendr's generic
    // "User function panicked", which discards the reason.
    match result {
        Ok(robj) => robj,
        Err(msg) => throw_r_error(msg),
    }
}

/// Keep only roots, leafs and branch points, preserving cable length.
///
/// Each replacement edge carries the summed length of the chain it stands in for, so
/// total cable length is unchanged.
///
/// @param parents Integer vector of 0-based parent indices (roots are `< 0`).
/// @param weights Optional numeric vector of child-to-parent edge weights; `NULL`
///   returns no `weights`.
/// @return List with `nodes` (0-based indices of the surviving nodes, in their
///   original relative order), `parents` (their new 0-based parent indices, `-1` for
///   roots, indexing *into* `nodes`), `weights` (length of each node's edge to its
///   new parent, roots `0`; `NULL` exactly when `weights` was `NULL`) and `node_map`
///   (one entry per *input* node, giving the 0-based index *into* `nodes` of the
///   survivor its data belongs to now: itself if it survived, otherwise whichever end
///   of its chain is nearer, measured in `weights`, ties going towards the root).
/// @export
#[extendr]
pub fn simplify_skeleton(parents: Vec<i32>, #[default = "NULL"] weights: Option<Vec<f64>>) -> Robj {
    let parents = Array1::from_vec(parents);
    let weights = to_weights::<f32>(weights);

    dropped_to_list(fastcore::dag::simplify_skeleton(&parents.view(), &weights))
}

/// Take the parent vector and the three coordinate vectors every geometric entry point
/// opens with, checking they describe the same nodes.
fn parents_and_coords(
    parents: Vec<i32>,
    x: &[f64],
    y: &[f64],
    z: &[f64],
) -> (Array1<i32>, Array2<f64>) {
    let coords = xyz_to_coords(x, y, z);
    assert_eq!(
        coords.nrows(),
        parents.len(),
        "`x`, `y` and `z` must have one entry per node"
    );
    (Array1::from_vec(parents), coords)
}

/// Pack the tuple every node-dropping method returns into an R list.
fn dropped_to_list(out: (Vec<i32>, Array1<i32>, Option<Vec<f32>>, Array1<i32>)) -> Robj {
    let (nodes, new_parents, new_weights, node_map) = out;
    list!(
        nodes = nodes,
        parents = new_parents.to_vec(),
        weights = opt_lengths(new_weights),
        node_map = node_map.to_vec()
    )
    .into()
}

/// Keep every `factor`-th node of every segment, dropping the rest.
///
/// The plain "make this skeleton smaller" operation: it pays no attention to geometry,
/// so reach for it when the skeleton is already evenly sampled and you just want fewer
/// nodes. Roots, branch points and leafs always survive, so the result is still the same
/// neuron — only its unbranched stretches are sampled `factor` times more coarsely. See
/// `simplify_rdp()` and `simplify_vw()` for the geometry-aware alternatives.
///
/// @param parents Integer vector of 0-based parent indices (roots are `< 0`).
/// @param factor Integer; keep one node in every `factor`, counting from each segment's
///   distal end. `1` keeps everything; the useful range starts at 2.
/// @param preserve Optional logical vector, one entry per node, marking extra nodes that
///   must survive; `NULL` for none.
/// @param weights Optional numeric vector of child-to-parent edge weights; `NULL`
///   returns no `weights`.
/// @return List with `nodes` (0-based indices of the surviving nodes, in their original
///   relative order), `parents` (their new 0-based parent indices, `-1` for roots,
///   indexing *into* `nodes`), `weights` (length of each node's edge to its new
///   parent, i.e. the summed length of the chain it replaces; `NULL` exactly when
///   `weights` was `NULL`) and `node_map` (one entry per *input* node, giving the
///   0-based index *into* `nodes` of the survivor its data belongs to now: itself if it
///   survived, otherwise whichever end of its chain is nearer, measured in `weights`,
///   ties going towards the root). Total cable length is preserved.
/// @export
#[extendr]
pub fn downsample_skeleton(
    parents: Vec<i32>,
    factor: i32,
    #[default = "NULL"] preserve: Robj,
    #[default = "NULL"] weights: Option<Vec<f64>>,
) -> Robj {
    assert!(factor >= 1, "`factor` must be >= 1");
    let n = parents.len();
    let parents = Array1::from_vec(parents);
    let preserve = robj_to_mask(&preserve, n);
    let weights = to_weights::<f32>(weights);

    dropped_to_list(fastcore::downsample::downsample_skeleton(
        &parents.view(),
        factor as usize,
        &preserve,
        &weights,
    ))
}

/// Drop the nodes that do not bend a neurite, by Ramer-Douglas-Peucker.
///
/// Where `downsample_skeleton()` thins by counting, this thins by *shape*: a node
/// survives only if removing it would move the traced path by more than `epsilon`. Long
/// straight stretches collapse to their two ends while a tight curve keeps every node it
/// needs, so the same tolerance buys a much better skeleton per node than a fixed factor
/// does.
///
/// Each replacement edge carries the length of the chain it stands in for, so geodesic
/// distances stay right even where the geometry has been cut across.
///
/// @param parents Integer vector of 0-based parent indices (roots are `< 0`).
/// @param x,y,z Numeric vectors of node coordinates, one entry per node.
/// @param epsilon Numeric; how far the simplified path may stray from the original, in
///   the units of the coordinates. `0` still drops nodes that are *exactly* collinear,
///   and nothing else.
/// @param preserve Optional logical vector, one entry per node, marking extra nodes that
///   must survive; `NULL` for none.
/// @param weights Optional numeric vector of child-to-parent edge weights; `NULL`
///   returns no `weights`.
/// @param threads Optional integer; number of threads. `NULL` uses all cores.
/// @return List with `nodes`, `parents`, `weights` and `node_map`, as
///   `downsample_skeleton()`.
/// @export
#[extendr]
pub fn simplify_rdp(
    parents: Vec<i32>,
    x: Vec<f64>,
    y: Vec<f64>,
    z: Vec<f64>,
    epsilon: f64,
    #[default = "NULL"] preserve: Robj,
    #[default = "NULL"] weights: Option<Vec<f64>>,
    #[default = "NULL"] threads: Option<i32>,
) -> Robj {
    let n = parents.len();
    let (parents, coords) = parents_and_coords(parents, &x, &y, &z);
    let preserve = robj_to_mask(&preserve, n);
    let weights = to_weights::<f32>(weights);

    dropped_to_list(fastcore::downsample::simplify_rdp(
        &parents.view(),
        &coords.view(),
        epsilon,
        &preserve,
        &weights,
        threads.map(|t| t as usize),
    ))
}

/// Drop the nodes that contribute least area, by Visvalingam-Whyatt.
///
/// The other geometry-aware thinning. Where `simplify_rdp()` asks how far the path
/// *moves*, this asks how much area each node adds to it and repeatedly removes whichever
/// node adds least. The difference shows under aggressive simplification: RDP will
/// happily keep one spike and flatten everything around it, while Visvalingam-Whyatt
/// sheds detail evenly and so keeps a neurite looking like itself.
///
/// @param parents Integer vector of 0-based parent indices (roots are `< 0`).
/// @param x,y,z Numeric vectors of node coordinates, one entry per node.
/// @param min_area Numeric; remove a node while the triangle it forms with its two
///   surviving neighbours is smaller than this, in the *squared* units of the
///   coordinates. `0` or less is a no-op.
/// @param preserve Optional logical vector, one entry per node, marking extra nodes that
///   must survive; `NULL` for none.
/// @param weights Optional numeric vector of child-to-parent edge weights; `NULL`
///   returns no `weights`.
/// @param threads Optional integer; number of threads. `NULL` uses all cores.
/// @return List with `nodes`, `parents`, `weights` and `node_map`, as
///   `downsample_skeleton()`.
/// @export
#[extendr]
pub fn simplify_vw(
    parents: Vec<i32>,
    x: Vec<f64>,
    y: Vec<f64>,
    z: Vec<f64>,
    min_area: f64,
    #[default = "NULL"] preserve: Robj,
    #[default = "NULL"] weights: Option<Vec<f64>>,
    #[default = "NULL"] threads: Option<i32>,
) -> Robj {
    let n = parents.len();
    let (parents, coords) = parents_and_coords(parents, &x, &y, &z);
    let preserve = robj_to_mask(&preserve, n);
    let weights = to_weights::<f32>(weights);

    dropped_to_list(fastcore::downsample::simplify_vw(
        &parents.view(),
        &coords.view(),
        min_area,
        &preserve,
        &weights,
        threads.map(|t| t as usize),
    ))
}

/// Place nodes at a fixed spacing along every neurite.
///
/// The inverse problem to `downsample_skeleton()`: rather than thinning what is there,
/// this re-samples each segment from scratch, so a skeleton whose node density varies
/// tenfold between neurites comes out evenly sampled throughout. Each segment is divided
/// into `round(length / spacing)` equal parts (at least one), so both of its endpoints
/// land exactly and no runt edge is left over; a segment shorter than `spacing / 2`
/// collapses to a single straight edge.
///
/// @param parents Integer vector of 0-based parent indices (roots are `< 0`).
/// @param x,y,z Numeric vectors of node coordinates, one entry per node.
/// @param spacing Numeric; target distance between adjacent nodes.
/// @param threads Optional integer; number of threads. `NULL` uses all cores.
/// @return List with `parents` (0-based parent index per output node, `-1` for roots),
///   `x`, `y`, `z` (their coordinates), `source_from` and `source_to` (the 0-based
///   *input* node indices of the edge each output node sits on, child then parent) and
///   `alpha` (how far along that edge it lies, from the child end). The input's roots,
///   branch points and leafs come first, in input order and unmoved; they carry their own
///   index in both `source_` columns and an `alpha` of 0, so
///   `attr[source_from + 1] * (1 - alpha) + attr[source_to + 1] * alpha` interpolates any
///   per-node quantity over the whole output. `node_map` points the other way: one entry
///   per *input* node, giving the 0-based index of the output node nearest it along the
///   neurite (ties going towards the root). Use `source_`/`alpha` to carry a per-node
///   column forward and `node_map` to re-home whatever is *attached* to a node.
/// @export
#[extendr]
pub fn resample_skeleton(
    parents: Vec<i32>,
    x: Vec<f64>,
    y: Vec<f64>,
    z: Vec<f64>,
    spacing: f64,
    #[default = "NULL"] threads: Option<i32>,
) -> Robj {
    let (parents, coords) = parents_and_coords(parents, &x, &y, &z);

    let out = fastcore::downsample::resample_skeleton(
        &parents.view(),
        &coords.view(),
        spacing,
        threads.map(|t| t as usize),
    );

    list!(
        parents = out.parents.to_vec(),
        x = out.coords.column(0).to_vec(),
        y = out.coords.column(1).to_vec(),
        z = out.coords.column(2).to_vec(),
        source_from = out.source.column(0).to_vec(),
        source_to = out.source.column(1).to_vec(),
        alpha = out.alpha.to_vec(),
        node_map = out.node_map.to_vec()
    )
    .into()
}

/// Smooth a skeleton with a moving average along each neurite.
///
/// Takes the tracing jitter out of a skeleton without touching its topology or its node
/// count: every node keeps its identity and its parent, and only its coordinates move.
/// Roots, branch points and leafs are pinned — a branch point that drifted would drag
/// three neurites apart — so this is safe to run before measuring angles, tortuosity or
/// tangent vectors, all of which a raw traced skeleton overstates.
///
/// @param parents Integer vector of 0-based parent indices (roots are `< 0`).
/// @param x,y,z Numeric vectors of node coordinates, one entry per node.
/// @param window Integer; nodes in the window, counting the node itself. Even values
///   round down to the odd value below, since the window is symmetric. `0` and `1` are
///   no-ops.
/// @param threads Optional integer; number of threads. `NULL` uses all cores.
/// @return List with `x`, `y` and `z`: the new coordinates, in the input's node order.
/// @export
#[extendr]
pub fn smooth_skeleton(
    parents: Vec<i32>,
    x: Vec<f64>,
    y: Vec<f64>,
    z: Vec<f64>,
    #[default = "5"] window: i32,
    #[default = "NULL"] threads: Option<i32>,
) -> Robj {
    // Guards the `as usize` below; a negative window would wrap past the core's check.
    assert!(window >= 0, "`window` must be non-negative");
    let (parents, coords) = parents_and_coords(parents, &x, &y, &z);

    let out = fastcore::downsample::smooth_skeleton(
        &parents.view(),
        &coords.view(),
        window as usize,
        threads.map(|t| t as usize),
    );

    coords_to_list(&out)
}

/// Smooth a skeleton with a Gaussian kernel along each neurite.
///
/// The same operation as `smooth_skeleton()` with a softer, scale-based kernel: `sigma`
/// is a distance in the units of the coordinates rather than a count of nodes, so the
/// amount of smoothing does not change when the skeleton is resampled. The kernel
/// measures distance *along* the neurite rather than between the points, which would
/// otherwise let the far arm of a hairpin pull on the near one. Segment ends are pinned
/// by reflecting the neurite about them.
///
/// @param parents Integer vector of 0-based parent indices (roots are `< 0`).
/// @param x,y,z Numeric vectors of node coordinates, one entry per node.
/// @param sigma Numeric; kernel width, as a distance along the neurite.
/// @param truncate Numeric; how many `sigma` out to keep summing. 4 covers all but 1e-4
///   of the kernel's mass.
/// @param threads Optional integer; number of threads. `NULL` uses all cores.
/// @return List with `x`, `y` and `z`: the new coordinates, in the input's node order.
/// @export
#[extendr]
pub fn smooth_skeleton_gaussian(
    parents: Vec<i32>,
    x: Vec<f64>,
    y: Vec<f64>,
    z: Vec<f64>,
    sigma: f64,
    #[default = "4.0"] truncate: f64,
    #[default = "NULL"] threads: Option<i32>,
) -> Robj {
    let (parents, coords) = parents_and_coords(parents, &x, &y, &z);

    let out = fastcore::downsample::smooth_skeleton_gaussian(
        &parents.view(),
        &coords.view(),
        sigma,
        truncate,
        threads.map(|t| t as usize),
    );

    coords_to_list(&out)
}

/// Split an `(N, 3)` coordinate array back into the `x`/`y`/`z` list R works in.
fn coords_to_list(coords: &Array2<f64>) -> Robj {
    list!(
        x = coords.column(0).to_vec(),
        y = coords.column(1).to_vec(),
        z = coords.column(2).to_vec()
    )
    .into()
}

/// The skeleton's adjacency matrix, as the three arrays of a CSR matrix.
///
/// Handing back the raw arrays rather than a matrix object keeps this package free of
/// a sparse-matrix dependency; feed them to `Matrix::sparseMatrix(p = indptr, i =
/// indices, x = data, ...)` if you want one.
///
/// @param parents Integer vector of 0-based parent indices (roots are `< 0`).
/// @param weights Optional numeric vector of child-to-parent edge weights; `NULL`
///   gives every edge weight 1.
/// @param directed Logical; if `TRUE` (default) only child-to-parent edges are
///   stored, if `FALSE` both directions are.
/// @param transpose Logical; if `TRUE` store parent-to-child instead. Ignored when
///   `directed` is `FALSE`, where the matrix is symmetric anyway.
/// @return List with `indptr` (integer, `N + 1` entries), `indices` (integer column
///   index per non-zero) and `data` (numeric weight per non-zero). Rows and columns
///   are in node-index order and column indices ascend within each row.
/// @export
#[extendr]
pub fn adjacency(
    parents: Vec<i32>,
    #[default = "NULL"]
    weights: Option<Vec<f64>>,
    #[default = "TRUE"]
    directed: bool,
    #[default = "FALSE"]
    transpose: bool,
) -> Robj {
    let parents = Array1::from_vec(parents);
    let weights = to_weights::<f32>(weights);

    let (indptr, indices, data) =
        fastcore::dag::adjacency(&parents.view(), &weights, directed, transpose);

    list!(indptr = indptr, indices = indices, data = data).into()
}

/// The longest path from any node to its root.
///
/// Not the NP-hard general problem: in a rooted forest every maximal path is fixed by
/// its start node, so this is a distances-to-root question.
///
/// @param parents Integer vector of 0-based parent indices (roots are `< 0`).
/// @param weights Optional numeric vector of child-to-parent edge weights; `NULL`
///   counts edges.
/// @return Integer vector of 0-based node indices along the path, **distal first** —
///   so the first element is the far end and the last is a root. Ties break towards
///   the lowest node index.
/// @export
#[extendr]
pub fn longest_path(parents: Vec<i32>, #[default = "NULL"] weights: Option<Vec<f64>>) -> Vec<i32> {
    let parents = Array1::from_vec(parents);
    let weights = to_weights::<f32>(weights);
    fastcore::dag::longest_path(&parents.view(), &weights)
}

/// The `n` longest paths, each peeled off before the next is sought.
///
/// @param parents Integer vector of 0-based parent indices (roots are `< 0`).
/// @param n Integer; how many paths to take.
/// @param weights Optional numeric vector of child-to-parent edge weights; `NULL`
///   counts edges.
/// @param min_length Optional numeric; stop once the next path is no longer than
///   this. Note it measures the path's whole *catchment* — every edge whose parent
///   lies on the path, so each twig hanging off it contributes its first edge too —
///   and that hitting it **stops** the search rather than skipping one path. Both are
///   inherited from `navis::split_into_fragments`, where they are load-bearing.
/// @return List of up to `n` integer vectors of 0-based node indices, longest first,
///   each distal-first.
/// @export
#[extendr]
pub fn longest_paths(
    parents: Vec<i32>,
    n: i32,
    #[default = "NULL"]
    weights: Option<Vec<f64>>,
    #[default = "NULL"]
    min_length: Option<f64>,
) -> Robj {
    let parents = Array1::from_vec(parents);
    let weights = to_weights::<f32>(weights);
    paths_to_list(fastcore::dag::longest_paths(
        &parents.view(),
        n.max(0) as usize,
        &weights,
        min_length.map(|l| l as f32),
    ))
}

/// Betweenness centrality, in `O(N)` rather than Brandes' `O(V*E)`.
///
/// Shortest paths in a tree are unique, so the count through a node is a closed form:
/// descendants times ancestors when directed, and a sum of products over the parts it
/// separates when not. Pairs are only counted within a connected component.
///
/// Note this is *not* `navis::betweeness_centrality(from_ = ...)`, which despite the
/// name computes a descendant count — see `descendant_counts`.
///
/// @param parents Integer vector of 0-based parent indices (roots are `< 0`).
/// @param directed Logical; if `TRUE` only count paths running child-to-parent.
/// @return Numeric vector of path counts, aligned with `parents`. Returned as double
///   rather than integer because an undirected 100k-node skeleton reaches ~5e9, well
///   past R's 32-bit integer.
/// @export
#[extendr]
pub fn betweenness(parents: Vec<i32>, #[default = "TRUE"] directed: bool) -> Vec<f64> {
    let parents = Array1::from_vec(parents);
    fastcore::dag::betweenness(&parents.view(), directed)
        .iter()
        .map(|&x| x as f64)
        .collect()
}

/// How many nodes lie strictly below each node.
///
/// This is what `navis::betweeness_centrality(from_ = ...)` actually computes, and
/// what `navis::find_main_branchpoint(method = "betweenness")` — its one caller —
/// wants.
///
/// @param parents Integer vector of 0-based parent indices (roots are `< 0`).
/// @param targets Optional integer vector of 0-based node indices; `NULL` counts
///   every node, otherwise only these are counted.
/// @return Numeric vector of counts, aligned with `parents`. A node is never its own
///   descendant, so a leaf scores `0` even when it is itself a target. Double rather
///   than integer for the same reason as `betweenness`.
/// @export
#[extendr]
pub fn descendant_counts(parents: Vec<i32>, #[default = "NULL"] targets: Option<Vec<i32>>) -> Vec<f64> {
    let parents = Array1::from_vec(parents);
    let targets = targets.map(Array1::from_vec);
    fastcore::dag::descendant_counts(&parents.view(), &targets)
        .iter()
        .map(|&x| x as f64)
        .collect()
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
/// @param threads Optional integer; number of threads. `NULL` uses all cores.
///   Worth capping when stitching many skeletons across several processes — see
///   `set_num_threads()`, which is cheaper than passing this on every call.
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
    #[default = "NULL"] threads: Option<i32>,
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
        threads.map(|t| t as usize),
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
/// @param threads Optional integer; number of threads. `NULL` uses all cores.
///   Worth capping when healing many skeletons across several processes — see
///   `set_num_threads()`, which is cheaper than passing this on every call.
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
    #[default = "NULL"] threads: Option<i32>,
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
        threads.map(|t| t as usize),
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

/// Node indices as the core wants them. R hands us `i32`; the core indexes in `u32`.
fn as_u32(v: &[i32]) -> Vec<u32> {
    v.iter().map(|&i| i as u32).collect()
}

/// The optional form of [`as_u32`], for the many arguments that default to `NULL`.
fn to_u32(v: Option<Vec<i32>>) -> Option<Vec<u32>> {
    v.map(|x| as_u32(&x))
}

/// Convert an `(R, C)` ndarray into an R matrix of whatever type `f` yields.
///
/// Not `RArray::new_matrix`: that one builds a `flat_map`, whose `size_hint` is
/// `(0, None)`, so extendr cannot take its `fixed_size_collect` path and stages the
/// entire matrix in a temporary `Vec` before copying it into R — three passes over the
/// data and a transient allocation the size of the result. `arr.t().iter()` is an
/// `ExactSizeIterator` yielding exactly R's column-major order, so this writes once,
/// straight into the R vector.
fn array2_to_r<T: Copy, U>(arr: &Array2<T>, f: impl FnMut(T) -> U) -> Robj
where
    U: ToVectorValue,
    Robj: for<'a> AsTypedSlice<'a, U>,
{
    let (nr, nc) = (arr.nrows(), arr.ncols());
    arr.t()
        .iter()
        .copied()
        .map(f)
        .collect_rarray([nr, nc])
        .expect("dims are the array's own")
        .into()
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
/// @param precision Integer; 32 or 64, the width distances are accumulated at. The
///   result is numeric either way — R has no float32 — so this buys accuracy.
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
    #[default = "32"]
    precision: i32,
) -> Robj {
    let faces = robj_to_faces(&faces);
    let coords = robj_to_coords(vertices);
    let sources = to_u32(sources);
    let targets = to_u32(targets);

    fn run<W: Weight + Into<f64>>(
        faces: ArrayView2<u32>,
        n_vertices: i32,
        coords: Option<ArrayView2<f64>>,
        sources: Option<&[u32]>,
        targets: Option<&[u32]>,
        limit: Option<f64>,
        threads: Option<i32>,
    ) -> Robj {
        let d = fastcore::mesh::geodesic_matrix_mesh::<W>(
            faces,
            n_vertices as usize,
            coords,
            sources,
            targets,
            limit.map(W::from_f64),
            threads.map(|t| t as usize),
        );
        array2_to_r(&d, Into::into)
    }

    let c = coords.as_ref().map(|c| c.view());
    let (src, tgt) = (sources.as_deref(), targets.as_deref());
    if wide(precision) {
        run::<f64>(faces.view(), n_vertices, c, src, tgt, limit, threads)
    } else {
        run::<f32>(faces.view(), n_vertices, c, src, tgt, limit, threads)
    }
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
/// @param precision Integer; 32 or 64, the width distances are accumulated at. The
///   result is numeric either way — R has no float32 — so this buys accuracy.
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
    #[default = "32"]
    precision: i32,
) -> Robj {
    let edges = robj_to_edges(&edges);
    let sources = to_u32(sources);
    let targets = to_u32(targets);

    fn run<W: Weight + Into<f64>>(
        edges: ArrayView2<u32>,
        n_nodes: i32,
        weights: Option<Vec<f64>>,
        directed: bool,
        sources: Option<&[u32]>,
        targets: Option<&[u32]>,
        limit: Option<f64>,
        threads: Option<i32>,
    ) -> Robj {
        let w = to_weights::<W>(weights);
        let d = fastcore::mesh::geodesic_matrix_graph(
            edges,
            n_nodes as usize,
            w.as_ref().map(|w| w.view()).as_ref(),
            directed,
            sources,
            targets,
            limit.map(W::from_f64),
            threads.map(|t| t as usize),
        );
        array2_to_r(&d, Into::into)
    }

    let (src, tgt) = (sources.as_deref(), targets.as_deref());
    if wide(precision) {
        run::<f64>(edges.view(), n_nodes, weights, directed, src, tgt, limit, threads)
    } else {
        run::<f32>(edges.view(), n_nodes, weights, directed, src, tgt, limit, threads)
    }
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
/// @param precision Integer; 32 or 64, the width distances are accumulated at. The
///   result is numeric either way — R has no float32 — so this buys accuracy.
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
    #[default = "32"]
    precision: i32,
) -> Robj {
    let faces = robj_to_faces(&faces);
    let coords = robj_to_coords(vertices);
    let sources = to_u32(sources);
    let targets = to_u32(targets);

    fn run<W: Weight + Into<f64>>(
        faces: ArrayView2<u32>,
        n_vertices: i32,
        coords: Option<ArrayView2<f64>>,
        sources: Option<&[u32]>,
        targets: Option<&[u32]>,
        limit: Option<f64>,
        threads: Option<i32>,
    ) -> (Vec<f64>, Vec<i32>) {
        let (d, i) = fastcore::mesh::geodesic_nearest_mesh::<W>(
            faces,
            n_vertices as usize,
            coords,
            sources,
            targets,
            limit.map(W::from_f64),
            threads.map(|t| t as usize),
        );
        (d.iter().map(|&x| x.into()).collect(), i.to_vec())
    }

    let c = coords.as_ref().map(|c| c.view());
    let (src, tgt) = (sources.as_deref(), targets.as_deref());
    let (dists, nearest) = if wide(precision) {
        run::<f64>(faces.view(), n_vertices, c, src, tgt, limit, threads)
    } else {
        run::<f32>(faces.view(), n_vertices, c, src, tgt, limit, threads)
    };

    list!(distances = dists, nearest = nearest).into()
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
/// @param precision Integer; 32 or 64, the width distances are accumulated at. The
///   result is numeric either way — R has no float32 — so this buys accuracy.
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
    #[default = "32"]
    precision: i32,
) -> Robj {
    let faces = robj_to_faces(&faces);
    let coords = robj_to_coords(vertices);
    let sources = to_u32(sources);
    let targets = to_u32(targets);

    fn run<W: Weight + Into<f64>>(
        faces: ArrayView2<u32>,
        n_vertices: i32,
        coords: Option<ArrayView2<f64>>,
        sources: Option<&[u32]>,
        targets: Option<&[u32]>,
        limit: Option<f64>,
        threads: Option<i32>,
    ) -> (Vec<f64>, Vec<i32>) {
        let (d, i) = fastcore::mesh::geodesic_farthest_mesh::<W>(
            faces,
            n_vertices as usize,
            coords,
            sources,
            targets,
            limit.map(W::from_f64),
            threads.map(|t| t as usize),
        );
        (d.iter().map(|&x| x.into()).collect(), i.to_vec())
    }

    let c = coords.as_ref().map(|c| c.view());
    let (src, tgt) = (sources.as_deref(), targets.as_deref());
    let (dists, farthest) = if wide(precision) {
        run::<f64>(faces.view(), n_vertices, c, src, tgt, limit, threads)
    } else {
        run::<f32>(faces.view(), n_vertices, c, src, tgt, limit, threads)
    };

    list!(distances = dists, farthest = farthest).into()
}

// ---------------------------------------------------------------------------
// Graph primitives
// ---------------------------------------------------------------------------
//
// The handful of traversal operations mesh algorithms actually need, taken straight
// off an edge list. These exist because reaching for a general-purpose graph library
// means paying to *build* a graph object first, which on a real mesh costs more than
// every query you then run against it. Node references are 0-based, as elsewhere.

/// Unique undirected edges of a triangle mesh.
///
/// Each face `(a, b, c)` contributes the edges `(a, b)`, `(b, c)`, `(c, a)`. Edges are
/// undirected, so each pair is normalised to `[min, max]` before dedup; self-loops
/// from degenerate faces are kept.
///
/// @param faces Integer or numeric `(F, 3)` matrix of triangle vertex indices
///   (0-based).
/// @param vertices Optional numeric `(V, 3)` matrix of vertex coordinates. If given,
///   also returns the euclidean length of each unique edge.
/// @param threads Optional integer; number of threads. `NULL` uses all cores.
/// @return List with `edges` (integer `(n_unique, 2)` matrix of `[min, max]` rows,
///   sorted ascending by `(max, min)`) and `lengths` (numeric edge lengths, or `NULL`
///   when `vertices` was `NULL`).
/// @export
#[extendr]
pub fn unique_edges(
    faces: Robj,
    #[default = "NULL"] vertices: Option<Robj>,
    #[default = "NULL"] threads: Option<i32>,
) -> Robj {
    let faces = robj_to_faces(&faces);
    let coords = robj_to_coords(vertices);

    // `return_index`/`return_inverse` are `false`, which selects the core's fast path:
    // neither vector is allocated, so nothing is computed and thrown away. They are
    // additive to the returned list if a caller ever wants them.
    let (edges, _, _, lengths) = fastcore::mesh::unique_edges(
        faces.view(),
        coords.as_ref().map(|c| c.view()),
        false,
        false,
        to_threads(threads),
    );

    let lengths_robj: Robj = match lengths {
        Some(l) => l.to_vec().into(),
        None => ().into(),
    };
    list!(edges = array2_to_r(&edges, |x| x as i32), lengths = lengths_robj).into()
}

/// Connected components of a graph given as an edge list.
///
/// The edge-list counterpart of `mesh_connected_components`, using the same
/// Union-Find: one integer array, no adjacency list.
///
/// @param edges Integer or numeric `(E, 2)` matrix of edges (0-based node indices).
///   Direction is ignored; self-loops and parallel edges are harmless.
/// @param n_nodes Integer; total number of nodes. Nodes named by no edge form
///   components of size one.
/// @return Integer vector giving, per node, the smallest node index in its component.
/// @export
#[extendr]
pub fn connected_components_graph(edges: Robj, n_nodes: i32) -> Vec<i32> {
    let edges = robj_to_edges(&edges);
    fastcore::mesh::connected_components_graph(edges.view(), n_nodes as usize)
        .iter()
        .map(|&x| x as i32)
        .collect()
}

/// Connected components of every level set at once.
///
/// Given a label per node, finds the connected components of each subgraph induced by
/// the nodes sharing a label — all labels in a single pass, by unioning an edge only
/// when its two endpoints agree. This is the inner loop of wavefront-style mesh
/// skeletonization, where the label is a binned geodesic distance and each component
/// is one ring around the structure.
///
/// Done conventionally that loop costs one subgraph construction plus one component
/// search per distinct label; here it is one sweep over the edges.
///
/// @param edges Integer or numeric `(E, 2)` matrix of edges (0-based node indices).
/// @param n_nodes Integer; total number of nodes.
/// @param labels Integer vector with one label per node. **Negative labels mark
///   excluded nodes**: they join no component and come back as `-1`. That is what lets
///   you feed in the output of a search that could not reach everything —
///   `geodesic_matrix_*` returns `-1` for unreachable — rather than lumping every
///   unreachable node into one bogus level.
/// @return List with `ids` (integer component index per node in `[0, n_components)`,
///   or `-1` for excluded nodes; assigned in order of first appearance scanning nodes
///   low to high) and `n_components` (integer).
/// @export
#[extendr]
pub fn level_set_components(edges: Robj, n_nodes: i32, labels: Vec<i32>) -> Robj {
    let edges = robj_to_edges(&edges);
    let labels: Array1<i64> = labels.into_iter().map(i64::from).collect();
    let (ids, n) =
        fastcore::mesh::level_set_components(edges.view(), n_nodes as usize, labels.view());
    list!(ids = ids, n_components = n as i32).into()
}

/// Contract nodes onto new ids, returning the simplified edge list.
///
/// Both endpoints of every edge are pushed through `mapping`; edges that end up with
/// both ends on the same new node are dropped, and the rest deduplicated. This is
/// `igraph::contract_vertices()` followed by `simplify()`, fused — and, unlike that
/// pair, it does not rewrite a graph object in place, so contracting costs no copy.
///
/// @param edges Integer or numeric `(E, 2)` matrix of edges (0-based node indices).
/// @param mapping Integer vector giving the new 0-based id of each old node. Ids need
///   not be contiguous, but the output is only as compact as the ids you supply.
/// @param threads Optional integer; number of threads. `NULL` uses all cores.
/// @return Integer `(n_unique, 2)` matrix of the surviving edges as `[min, max]` rows.
/// @export
#[extendr]
pub fn contract_vertices(edges: Robj, mapping: Vec<i32>, #[default = "NULL"] threads: Option<i32>) -> Robj {
    let edges = robj_to_edges(&edges);
    let mapping: Array1<u32> = Array1::from_vec(as_u32(&mapping));
    let out = fastcore::mesh::contract_vertices(edges.view(), mapping.view(), to_threads(threads));
    array2_to_r(&out, |x| x as i32)
}

/// Minimum (or maximum) spanning forest of an undirected graph.
///
/// Kruskal's algorithm on the same Union-Find as the component search: sort the edges
/// by weight, keep the ones that join two different components. Disconnected input is
/// fine — each component contributes its own tree.
///
/// @param edges Integer or numeric `(E, 2)` matrix of edges (0-based node indices).
/// @param n_nodes Integer; total number of nodes.
/// @param weights Optional numeric vector, one weight per edge; `NULL` treats every
///   edge as equal. Must be finite; negative weights are allowed.
/// @param maximize Logical; return the *maximum* spanning forest instead. This exists
///   so you do not have to pass `1 / weights` to invert the ordering — a transform
///   that both loses precision and blows up on the zero weights that legitimately
///   occur.
/// @param threads Optional integer; number of threads. `NULL` uses all cores.
/// @param precision Integer; 32 or 64, the width the search accumulates at. Nothing
///   here returns a distance, so this changes only which answer close ties resolve to.
/// @return Integer vector of 0-based **row indices into `edges`**, ordered by weight —
///   not the edges themselves, so you can index whatever per-edge data you hold with
///   the same vector. Remember to add 1 before using it to subset an R matrix.
/// @export
#[extendr]
pub fn minimum_spanning_tree(
    edges: Robj,
    n_nodes: i32,
    #[default = "NULL"]
    weights: Option<Vec<f64>>,
    #[default = "FALSE"]
    maximize: bool,
    #[default = "NULL"]
    threads: Option<i32>,
    #[default = "32"]
    precision: i32,
) -> Vec<i32> {
    let edges = robj_to_edges(&edges);

    fn run<W: Weight>(
        edges: ArrayView2<u32>,
        n_nodes: i32,
        weights: Option<Vec<f64>>,
        maximize: bool,
        threads: Option<i32>,
    ) -> Array1<i64> {
        let w = to_weights::<W>(weights);
        fastcore::mesh::minimum_spanning_tree(
            edges,
            n_nodes as usize,
            w.as_ref().map(|w| w.view()).as_ref(),
            maximize,
            to_threads(threads),
        )
    }

    let keep = if wide(precision) {
        run::<f64>(edges.view(), n_nodes, weights, maximize, threads)
    } else {
        run::<f32>(edges.view(), n_nodes, weights, maximize, threads)
    };
    keep.iter().map(|&x| x as i32).collect()
}

/// Orient a graph into a rooted spanning forest — one parent per node.
///
/// `minimum_spanning_tree` picks *which* edges survive; this picks which way they
/// point, which is what turns a bag of undirected edges into something you can walk,
/// root, or write out as SWC. Cycles are fine — each component contributes a spanning
/// tree of itself, so this doubles as a cycle-breaker.
///
/// One search covers the whole graph. The obvious construction — a shortest-path tree
/// per component — costs `O(components * n_nodes)` in output alone, which on a mesh
/// that shatters into thousands of specks is gigabytes to answer a question whose
/// answer is one vector.
///
/// @param edges Integer or numeric `(E, 2)` matrix of edges (0-based node indices).
///   Direction is ignored.
/// @param n_nodes Integer; total number of nodes. Nodes named by no edge are isolated
///   roots.
/// @param weights Optional numeric vector, one length per edge. `NULL` gives the
///   breadth-first tree; weights give the shortest-path tree, which is a different
///   (and generally deeper) spanning tree. Neither is the minimum spanning tree — for
///   that, run `minimum_spanning_tree` first and orient the edges it keeps.
/// @param roots Optional integer vector of 0-based nodes to root at; `NULL` roots each
///   component at its lowest node index. Components holding none of `roots` fall back
///   to that, so the result is always a complete forest. Two roots in the *same*
///   component split it into two trees, each node going to whichever root is nearer.
/// @param precision Integer; 32 or 64, the width the search accumulates at. Nothing
///   here returns a distance, so this changes only which answer close ties resolve to.
/// @return List with `parents` (integer 0-based parent index per node, `-1` at a root)
///   and `order` (integer 0-based node indices in the order they settled — a node
///   always follows its parent, so relabelling by it guarantees parents get lower
///   indices than their children, which is exactly the SWC requirement).
/// @export
#[extendr]
pub fn parents_from_edges(
    edges: Robj,
    n_nodes: i32,
    #[default = "NULL"]
    weights: Option<Vec<f64>>,
    #[default = "NULL"]
    roots: Option<Vec<i32>>,
    #[default = "32"]
    precision: i32,
) -> Robj {
    let edges = robj_to_edges(&edges);
    let roots = to_u32(roots);

    fn run<W: Weight>(
        edges: ArrayView2<u32>,
        n_nodes: i32,
        weights: Option<Vec<f64>>,
        roots: Option<&[u32]>,
    ) -> (Array1<i32>, Array1<u32>) {
        let w = to_weights::<W>(weights);
        fastcore::mesh::parents_from_edges(
            edges,
            n_nodes as usize,
            w.as_ref().map(|w| w.view()).as_ref(),
            roots,
        )
    }

    let (parents, order) = if wide(precision) {
        run::<f64>(edges.view(), n_nodes, weights, roots.as_deref())
    } else {
        run::<f32>(edges.view(), n_nodes, weights, roots.as_deref())
    };

    list!(
        parents = parents.to_vec(),
        order = order.iter().map(|&x| x as i32).collect::<Vec<i32>>()
    )
    .into()
}

/// Which edges are bridges — the ones whose removal would disconnect their component.
///
/// Tarjan's algorithm, one depth-first sweep. The counterpart to
/// `minimum_spanning_tree` rather than a variant of it: the MST asks which edges to
/// *keep* to stay connected, this asks which ones may not be *dropped*. That is the
/// question behind "prune this graph but do not shatter it".
///
/// Parallel edges are honoured — two nodes joined twice are joined by a cycle, so
/// neither of those edges is a bridge. Self-loops are never bridges.
///
/// @param edges Integer or numeric `(E, 2)` matrix of edges (0-based node indices).
///   Direction is ignored.
/// @param n_nodes Integer; total number of nodes.
/// @return Logical vector with one flag per input edge, `TRUE` for a bridge.
/// @export
#[extendr]
pub fn bridges(edges: Robj, n_nodes: i32) -> Vec<bool> {
    let edges = robj_to_edges(&edges);
    fastcore::mesh::bridges(edges.view(), n_nodes as usize).to_vec()
}

/// Package a geodesic MST result for R.
fn mst_result<W: Weight + Into<f64>>(edges: Array2<i64>, weights: Array1<W>) -> Robj {
    let weights: Vec<f64> = weights.iter().map(|&x| x.into()).collect();
    list!(edges = array2_to_r(&edges, |x| x as i32), weights = weights).into()
}

/// Minimum spanning tree over a subset of mesh vertices, by geodesic distance.
///
/// The tree that reconnects a scatter of surviving vertices through the mesh they were
/// carved out of — the last step of a skeletonization, where the mesh has been thinned
/// to a few thousand vertices that must be rejoined along the surface rather than
/// through space.
///
/// The obvious route is to ask for the `k x k` geodesic matrix and hand it to a matrix
/// MST. That materialises `k^2` distances to use `k - 1` of them — 400 MB at
/// `k = 10000`, before the `O(k^2)` MST itself — and needs `k` separate searches to
/// fill. This never forms the matrix: following Mehlhorn's distance-network
/// construction, one multi-source search partitions every vertex by which of `nodes`
/// is nearest, and each mesh edge whose endpoints fall in different cells offers one
/// candidate. An MST over those is an MST of the full distance network, so one sweep
/// and one Kruskal replace `k` searches and a dense matrix.
///
/// @param faces Integer or numeric `(F, 3)` matrix of triangle vertex indices
///   (0-based).
/// @param n_vertices Integer; total number of vertices.
/// @param nodes Integer vector of 0-based vertices to span. Must be distinct.
/// @param vertices Optional numeric `(V, 3)` matrix of vertex coordinates. If given,
///   edges are weighted by their euclidean length; if `NULL`, distances are hop
///   counts.
/// @param limit Optional numeric; do not join vertices further apart than this. The
///   result is then a *forest* when that disconnects the subset. Unlike a matrix
///   route's limit this also prunes the sweep, so it buys time rather than merely
///   discarding results.
/// @param threads Optional integer; number of threads. `NULL` uses all cores.
/// @param precision Integer; 32 or 64, the width distances are accumulated at. The
///   result is numeric either way — R has no float32 — so this buys accuracy.
/// @return List with `edges` (integer `(M, 2)` matrix of **0-based positions in
///   `nodes`**, not vertex indices — so `nodes[edges + 1]` maps back — ascending by
///   weight) and `weights` (numeric geodesic distance across each). The returned
///   weights are exactly the geodesic distances between the pairs they join, so they
///   are usable as lengths and not merely as an ordering.
/// @export
#[extendr]
pub fn geodesic_mst_mesh(
    faces: Robj,
    n_vertices: i32,
    nodes: Vec<i32>,
    #[default = "NULL"]
    vertices: Option<Robj>,
    #[default = "NULL"]
    limit: Option<f64>,
    #[default = "NULL"]
    threads: Option<i32>,
    #[default = "32"]
    precision: i32,
) -> Robj {
    let faces = robj_to_faces(&faces);
    let coords = robj_to_coords(vertices);
    let nodes = as_u32(&nodes);

    fn run<W: Weight + Into<f64>>(
        faces: ArrayView2<u32>,
        n_vertices: i32,
        coords: Option<ArrayView2<f64>>,
        nodes: &[u32],
        limit: Option<f64>,
        threads: Option<i32>,
    ) -> Robj {
        let (e, w) = fastcore::mesh::geodesic_mst_mesh::<W>(
            faces,
            n_vertices as usize,
            coords,
            nodes,
            limit.map(W::from_f64),
            to_threads(threads),
        );
        mst_result(e, w)
    }

    let c = coords.as_ref().map(|c| c.view());
    if wide(precision) {
        run::<f64>(faces.view(), n_vertices, c, &nodes, limit, threads)
    } else {
        run::<f32>(faces.view(), n_vertices, c, &nodes, limit, threads)
    }
}

/// Minimum spanning tree over a subset of graph nodes, by geodesic distance.
///
/// The edge-list form of `geodesic_mst_mesh`, which explains why this never builds the
/// `k x k` distance matrix the question seems to call for. Always undirected — a
/// minimum spanning tree of a directed graph is a different problem (an arborescence)
/// with a different algorithm.
///
/// @param edges Integer or numeric `(E, 2)` matrix of edges (0-based node indices).
/// @param n_nodes Integer; total number of nodes.
/// @param nodes Integer vector of 0-based nodes to span. Must be distinct.
/// @param weights Optional numeric vector, one length per edge; `NULL` counts edges.
/// @param limit Optional numeric; do not join nodes further apart than this.
/// @param threads Optional integer; number of threads. `NULL` uses all cores.
/// @param precision Integer; 32 or 64, the width distances are accumulated at. The
///   result is numeric either way — R has no float32 — so this buys accuracy.
/// @return List with `edges` (integer `(M, 2)` matrix of 0-based positions in `nodes`,
///   ascending by weight) and `weights` (numeric geodesic distance across each).
/// @export
#[extendr]
pub fn geodesic_mst_graph(
    edges: Robj,
    n_nodes: i32,
    nodes: Vec<i32>,
    #[default = "NULL"]
    weights: Option<Vec<f64>>,
    #[default = "NULL"]
    limit: Option<f64>,
    #[default = "NULL"]
    threads: Option<i32>,
    #[default = "32"]
    precision: i32,
) -> Robj {
    let edges = robj_to_edges(&edges);
    let nodes = as_u32(&nodes);

    fn run<W: Weight + Into<f64>>(
        edges: ArrayView2<u32>,
        n_nodes: i32,
        weights: Option<Vec<f64>>,
        nodes: &[u32],
        limit: Option<f64>,
        threads: Option<i32>,
    ) -> Robj {
        let w = to_weights::<W>(weights);
        let (mst, mst_w) = fastcore::mesh::geodesic_mst_graph(
            edges,
            n_nodes as usize,
            w.as_ref().map(|w| w.view()).as_ref(),
            nodes,
            limit.map(W::from_f64),
            to_threads(threads),
        );
        mst_result(mst, mst_w)
    }

    if wide(precision) {
        run::<f64>(edges.view(), n_nodes, weights, &nodes, limit, threads)
    } else {
        run::<f32>(edges.view(), n_nodes, weights, &nodes, limit, threads)
    }
}

/// Shortest path trees over a graph — distances *and* the route to each node.
///
/// The predecessor-returning counterpart to `geodesic_matrix_graph`. Use this when you
/// need the path itself; use `geodesic_matrix_graph` when the distance is enough, and
/// `geodesic_path` when you want the node sequences rather than the raw chains.
///
/// Because this takes a bare edge list there is no index to build or invalidate
/// between calls, which is what algorithms that re-weight the graph every iteration
/// need — TEASAR zeroes the edges along each path it extracts, then searches again.
///
/// @param edges Integer or numeric `(E, 2)` matrix of edges (0-based node indices).
/// @param n_nodes Integer; total number of nodes.
/// @param weights Optional numeric vector, one length per edge; `NULL` counts edges.
///   Must be finite and non-negative. **Zero weights are explicitly allowed** — they
///   are how a penalised-path search makes an already-extracted route free to
///   re-traverse.
/// @param directed Logical; if `TRUE` an edge `(u, v)` may only be traversed from `u`
///   to `v`.
/// @param sources Optional integer vector of 0-based source nodes, one tree each;
///   `NULL` uses every node.
/// @param limit Optional numeric; ignore nodes further away than this.
/// @param threads Optional integer; number of threads. `NULL` uses all cores.
/// @param precision Integer; 32 or 64, the width distances are accumulated at. The
///   result is numeric either way — R has no float32 — so this buys accuracy.
/// @return List with `distances` (numeric `(n_sources, n_nodes)` matrix, `-1` where
///   unreachable) and `predecessors` (integer matrix of the same shape giving, per
///   node, the node before it on the shortest path back to that row's source; `-1` for
///   the source itself and for unreachable nodes, so one `>= 0` test both walks the
///   path and terminates it).
/// @export
#[extendr]
pub fn geodesic_predecessors(
    edges: Robj,
    n_nodes: i32,
    #[default = "NULL"]
    weights: Option<Vec<f64>>,
    #[default = "FALSE"]
    directed: bool,
    #[default = "NULL"]
    sources: Option<Vec<i32>>,
    #[default = "NULL"]
    limit: Option<f64>,
    #[default = "NULL"]
    threads: Option<i32>,
    #[default = "32"]
    precision: i32,
) -> Robj {
    let edges = robj_to_edges(&edges);
    let sources = to_u32(sources);

    fn run<W: Weight + Into<f64>>(
        edges: ArrayView2<u32>,
        n_nodes: i32,
        weights: Option<Vec<f64>>,
        directed: bool,
        sources: Option<&[u32]>,
        limit: Option<f64>,
        threads: Option<i32>,
    ) -> (Robj, Array2<i32>) {
        let w = to_weights::<W>(weights);
        let (d, p) = fastcore::mesh::geodesic_predecessors_graph(
            edges,
            n_nodes as usize,
            w.as_ref().map(|w| w.view()).as_ref(),
            directed,
            sources,
            limit.map(W::from_f64),
            to_threads(threads),
        );
        (array2_to_r(&d, Into::into), p)
    }

    let src = sources.as_deref();
    let (dists, preds) = if wide(precision) {
        run::<f64>(edges.view(), n_nodes, weights, directed, src, limit, threads)
    } else {
        run::<f32>(edges.view(), n_nodes, weights, directed, src, limit, threads)
    };

    list!(distances = dists, predecessors = array2_to_r(&preds, |x| x)).into()
}

/// The shortest route from one source to each target, as node sequences.
///
/// @param edges Integer or numeric `(E, 2)` matrix of edges (0-based node indices).
/// @param n_nodes Integer; total number of nodes.
/// @param source Integer; the 0-based node to start from.
/// @param targets Integer vector of 0-based nodes to reach.
/// @param weights Optional numeric vector, one length per edge; `NULL` counts edges.
/// @param directed Logical; if `TRUE` an edge `(u, v)` may only be traversed from `u`
///   to `v`.
/// @param precision Integer; 32 or 64, the width the search accumulates at. Nothing
///   here returns a distance, so this changes only which answer close ties resolve to.
/// @return List of integer vectors, one per target in `targets` order, each running
///   source-first to target-last. An unreachable target gives an empty vector.
/// @export
#[extendr]
pub fn geodesic_path(
    edges: Robj,
    n_nodes: i32,
    source: i32,
    targets: Vec<i32>,
    #[default = "NULL"]
    weights: Option<Vec<f64>>,
    #[default = "FALSE"]
    directed: bool,
    #[default = "32"]
    precision: i32,
) -> Robj {
    let edges = robj_to_edges(&edges);
    let targets = as_u32(&targets);

    fn run<W: Weight>(
        edges: ArrayView2<u32>,
        n_nodes: i32,
        weights: Option<Vec<f64>>,
        directed: bool,
        source: i32,
        targets: &[u32],
    ) -> Vec<Vec<u32>> {
        let w = to_weights::<W>(weights);
        fastcore::mesh::geodesic_path_graph(
            edges,
            n_nodes as usize,
            w.as_ref().map(|w| w.view()).as_ref(),
            directed,
            source as u32,
            targets,
        )
    }

    let paths = if wide(precision) {
        run::<f64>(edges.view(), n_nodes, weights, directed, source, &targets)
    } else {
        run::<f32>(edges.view(), n_nodes, weights, directed, source, &targets)
    };

    paths_to_list(
        paths
            .into_iter()
            .map(|p| p.into_iter().map(|v| v as i32).collect::<Vec<i32>>()),
    )
}

/// Greedily partition nodes into connected clusters of bounded radius.
///
/// Repeatedly takes an unassigned node as a seed and grows a cluster outwards from it,
/// absorbing any node reachable within `max_dist` that no earlier cluster has already
/// claimed. Collapsing each cluster to its centroid gives a coarser graph whose nodes
/// are spaced by roughly `max_dist`, which is what makes this useful as mesh or
/// skeleton downsampling.
///
/// The radius is the **true geodesic distance from the seed**, not the length of the
/// walk that happened to reach it — so a node close to a seed is never excluded merely
/// because a traversal arrived at it the long way round.
///
/// @param edges Integer or numeric `(E, 2)` matrix of edges (0-based node indices).
///   Treated as undirected.
/// @param n_nodes Integer; total number of nodes. Isolated nodes each become their own
///   cluster.
/// @param max_dist Numeric; maximum distance from a cluster's seed. Must be finite and
///   non-negative.
/// @param weights Optional numeric vector, one length per edge; `NULL` makes
///   `max_dist` a hop count.
/// @param seeds Optional integer vector of 0-based nodes to try as seeds, in order of
///   preference. Any node left unassigned afterwards becomes a seed in ascending index
///   order; `NULL` seeds in ascending index order throughout.
/// @param precision Integer; 32 or 64, the width the search accumulates at. Nothing
///   here returns a distance, so this changes only which answer close ties resolve to.
/// @return List with `labels` (integer cluster index per node, contiguous in
///   `[0, n_clusters)` and numbered in the order the clusters were grown; every node is
///   labelled) and `n_clusters` (integer).
/// @export
#[extendr]
pub fn geodesic_clusters(
    edges: Robj,
    n_nodes: i32,
    max_dist: f64,
    #[default = "NULL"]
    weights: Option<Vec<f64>>,
    #[default = "NULL"]
    seeds: Option<Vec<i32>>,
    #[default = "32"]
    precision: i32,
) -> Robj {
    let edges = robj_to_edges(&edges);
    let seeds = to_u32(seeds);

    fn run<W: Weight>(
        edges: ArrayView2<u32>,
        n_nodes: i32,
        max_dist: f64,
        weights: Option<Vec<f64>>,
        seeds: Option<&[u32]>,
    ) -> (Vec<i32>, usize) {
        let w = to_weights::<W>(weights);
        fastcore::mesh::geodesic_clusters(
            edges,
            n_nodes as usize,
            W::from_f64(max_dist),
            w.as_ref().map(|w| w.view()).as_ref(),
            seeds,
        )
    }

    let sd = seeds.as_deref();
    let (labels, n) = if wide(precision) {
        run::<f64>(edges.view(), n_nodes, max_dist, weights, sd)
    } else {
        run::<f32>(edges.view(), n_nodes, max_dist, weights, sd)
    };
    list!(labels = labels, n_clusters = n as i32).into()
}

// ---------------------------------------------------------------------------
// Mesh simplification
// ---------------------------------------------------------------------------
//
// The one thing here that no other simplifier returns is `vertex_map`, so both
// entry points hand it back unconditionally. Indices stay 0-based, as everywhere
// else in these bindings, and `-1` (not `NA`) marks a vertex that did not survive —
// keeping the sentinel the same value the Rust and Python sides use.

/// Select which of the two ways of naming a face budget the caller used.
///
/// What a ratio *means* — the rounding, the floor of one face — lives in the core,
/// so this only picks the variant and the rule stays defined once for all surfaces.
fn to_target(ratio: Option<f64>, n_faces: Option<i32>) -> fastcore::simplify::Target {
    match (ratio, n_faces) {
        (Some(r), None) => fastcore::simplify::Target::Ratio(r),
        (None, Some(n)) => {
            assert!(n >= 0, "`n_faces` must be non-negative, got {n}");
            fastcore::simplify::Target::Faces(n as usize)
        }
        _ => panic!("provide exactly one of `ratio` or `n_faces`"),
    }
}

/// Shared argument handling for the two simplification entry points.
fn simplify_inputs(
    faces: &Robj,
    vertices: &Robj,
    lock: &Robj,
) -> (Array2<u32>, Array2<f64>, Option<Array1<bool>>) {
    let faces = robj_to_faces(faces);
    let coords = robj_to_coords(Some(vertices.clone()))
        .expect("`vertices` must be a numeric (V, 3) matrix");
    let mask = robj_to_mask(lock, coords.nrows());
    (faces, coords, mask)
}

/// Pack a simplified mesh into the list R sees.
fn simplified_to_r(out: fastcore::simplify::Simplified) -> Robj {
    list!(
        vertices = array2_to_r(&out.vertices, |x| x),
        faces = array2_to_r(&out.faces, |x| x as i32),
        // As a slice, not `to_vec()`: the map is already contiguous, so a copy into
        // an intermediate `Vec` before extendr's own copy into the INTSXP is free
        // to skip.
        vertex_map = out.vertex_map.as_slice().expect("vertex_map is contiguous")
    )
    .into()
}

/// Simplify a triangle mesh, tracking where every vertex went.
///
/// Iteratively contracts the edge whose collapse costs least under the
/// Garland-Heckbert quadric error. Unlike other implementations of this algorithm it
/// also reports, for every vertex of the original mesh, which vertex of the
/// simplified mesh it ended up in - so per-vertex data survives the simplification.
///
/// Non-manifold input is fine: no manifoldness is assumed or checked, and each
/// collapse guard skips what it cannot handle rather than failing.
///
/// @param faces Integer or numeric `(F, 3)` matrix of triangle vertex indices
///   (0-based).
/// @param vertices Numeric `(V, 3)` matrix of vertex coordinates. Must be finite.
/// @param ratio Numeric fraction of the faces to keep, in `(0, 1]`.
/// @param n_faces Integer number of faces to keep. Give exactly one of `ratio` or
///   `n_faces`.
/// @param aggressiveness Numeric exponent of the error-threshold sweep. Higher
///   reaches the target in fewer, coarser passes. Default 7.
/// @param preserve_border Logical; freeze every vertex on a mesh boundary.
/// @param lock Optional logical vector, one entry per vertex. A locked vertex is
///   never merged into another and never moved, so it keeps its exact coordinates.
/// @return List with `vertices` (numeric `(V', 3)` matrix), `faces` (integer
///   `(F', 3)` matrix, 0-based) and `vertex_map` (integer, one entry per *original*
///   vertex giving its 0-based index in `vertices`, or `-1` if it did not survive).
///
///   Being *merged* is not a `-1`: a collapsed vertex carries the index of whatever
///   it merged into, which is the point of the map. An entry is `-1` exactly when
///   the vertex it ended up in is referenced by no surviving face, which takes one
///   of four forms: it was in no face to begin with; it was only ever in zero-area
///   faces, which are dropped on the way in and so reduce to the first case; the
///   whole piece it belonged to was consumed, since nothing is reserved per
///   connected component and a small fragment goes once the target is tight enough;
///   or the input was degenerate throughout and the output mesh is empty. Mask with
///   `vertex_map >= 0` before aggregating.
/// @export
#[extendr]
pub fn simplify_mesh(
    faces: Robj,
    vertices: Robj,
    #[default = "NULL"] ratio: Option<f64>,
    #[default = "NULL"] n_faces: Option<i32>,
    #[default = "7.0"] aggressiveness: f64,
    #[default = "FALSE"] preserve_border: bool,
    #[default = "NULL"] lock: Robj,
) -> Robj {
    let (faces, coords, mask) = simplify_inputs(&faces, &vertices, &lock);

    let out = fastcore::simplify::simplify_mesh(
        faces.view(),
        coords.view(),
        to_target(ratio, n_faces),
        aggressiveness,
        preserve_border,
        mask.as_ref().map(|m| m.as_slice().expect("mask is contiguous")),
    );
    simplified_to_r(out)
}

/// Simplify a triangle mesh without changing its shape.
///
/// Collapses only edges whose quadric error is below `epsilon` and repeats until a
/// whole pass changes nothing. There is no face budget: this sheds over-tessellation
/// - coplanar fans, duplicate vertices, degenerate faces - rather than hitting a
/// target. Use [simplify_mesh()] for that.
///
/// Note that "lossless" is a claim about the surface, not the outline: a quadric
/// measures distance to the planes of the incident faces, and the plane of a flat
/// patch says nothing about where that patch ends. Pass `preserve_border = TRUE` on
/// open meshes.
///
/// @param faces Integer or numeric `(F, 3)` matrix of triangle vertex indices
///   (0-based).
/// @param vertices Numeric `(V, 3)` matrix of vertex coordinates. Must be finite.
/// @param epsilon Numeric quadric error below which an edge may collapse. An
///   *absolute* error with units of squared distance, so it scales with your
///   coordinates. Default 1e-3.
/// @param max_iterations Integer cap on the number of passes. Default 9999.
/// @param preserve_border Logical; freeze every vertex on a mesh boundary.
/// @param lock Optional logical vector, one entry per vertex; as [simplify_mesh()].
/// @return List with `vertices`, `faces` and `vertex_map`, as [simplify_mesh()],
///   including when an entry of `vertex_map` is `-1`. The "whole piece consumed"
///   case arrives by a different route here: there is no face budget, but `epsilon`
///   is an *absolute* error, so a component small enough that all of its edges fall
///   under it collapses away entirely.
/// @export
#[extendr]
pub fn simplify_mesh_lossless(
    faces: Robj,
    vertices: Robj,
    #[default = "1e-3"] epsilon: f64,
    #[default = "9999"] max_iterations: i32,
    #[default = "FALSE"] preserve_border: bool,
    #[default = "NULL"] lock: Robj,
) -> Robj {
    let (faces, coords, mask) = simplify_inputs(&faces, &vertices, &lock);
    // `epsilon` is checked by the core, which is where it means something.
    assert!(
        max_iterations >= 0,
        "`max_iterations` must be non-negative, got {max_iterations}"
    );

    let out = fastcore::simplify::simplify_mesh_lossless(
        faces.view(),
        coords.view(),
        epsilon,
        max_iterations as usize,
        preserve_border,
        mask.as_ref().map(|m| m.as_slice().expect("mask is contiguous")),
    );
    simplified_to_r(out)
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
        32 => fastcore::nblast::nblast_allbyall::<f32, f64>(clouds, vecs, alpha_vecs, opts)
            .into_iter()
            .map(|x| x as f64)
            .collect(),
        64 => fastcore::nblast::nblast_allbyall::<f64, f64>(clouds, vecs, alpha_vecs, opts),
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
        32 => fastcore::nblast::nblast_query_target::<f32, f64>(qp, qv, qa, tp, tv, ta, opts)
            .into_iter()
            .map(|x| x as f64)
            .collect(),
        64 => fastcore::nblast::nblast_query_target::<f64, f64>(qp, qv, qa, tp, tv, ta, opts),
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
        32 => fastcore::nblast::nblast_pairs::<f32, f64>(qp, qv, qa, tp, tv, ta, pairs, opts)
            .into_iter()
            .map(|x| x as f64)
            .collect(),
        64 => fastcore::nblast::nblast_pairs::<f64, f64>(qp, qv, qa, tp, tv, ta, pairs, opts),
        _ => panic!("`precision` must be 32 or 64"),
    }
}

/// Map an R `symmetry` string onto the core's [`Symmetry`].
fn to_symmetry(name: &str) -> KnnSymmetry {
    match name {
        "forward" => KnnSymmetry::Forward,
        "mean" => KnnSymmetry::Mean,
        "min" => KnnSymmetry::Min,
        "max" => KnnSymmetry::Max,
        other => panic!("unknown `symmetry` {other:?}; expected 'forward', 'mean', 'min' or 'max'"),
    }
}

/// Pack a flat row-major k-NN result into R matrices.
///
/// Two translations to R conventions happen here: neighbour indices become
/// **1-based**, and the `-1` / `-Inf` padding the core emits for rows with fewer
/// than `k` candidates becomes `NA` in both matrices, which is what R code will
/// expect to test for.
fn knn_to_r(idx: &[i64], scores: &[f64], nrows: usize, k: usize) -> Robj {
    let idx_m = RArray::new_matrix(nrows, k, |r, c| {
        let v = idx[r * k + c];
        if v < 0 {
            Rint::na()
        } else {
            Rint::from((v + 1) as i32)
        }
    });
    let sc_m = RArray::new_matrix(nrows, k, |r, c| {
        let v = scores[r * k + c];
        if idx[r * k + c] < 0 || !v.is_finite() {
            Rfloat::na()
        } else {
            Rfloat::from(v)
        }
    });
    list!(idx = idx_m, scores = sc_m).into()
}

/// k nearest neighbours under NBLAST, without building the score matrix.
///
/// With `t_points`/`t_vects` supplied this is the query -> target form and the
/// returned indices address the *target* list; otherwise it is the all-by-all
/// form over `points`, with self-matches excluded. Only which neurons make the
/// shortlist is approximate — every returned score is an exact NBLAST value.
///
/// @param points List of `(N, 3)` numeric matrices of query point coordinates.
/// @param vects List of `(N, 3)` numeric matrices of unit tangent vectors, one
///   per neuron and aligned with `points`.
/// @param alphas Optional list of per-point alpha (anisotropy) vectors; `NULL`
///   disables alpha weighting.
/// @param t_points Optional list of `(N, 3)` target point matrices; `NULL` runs
///   the all-by-all form.
/// @param t_vects Optional list of `(N, 3)` target tangent matrices; must be
///   given together with `t_points`.
/// @param t_alphas Optional list of per-point alpha vectors for the targets.
/// @param k Integer; neighbours to return per neuron.
/// @param n_candidates Integer; shortlist size per neuron (the recall/cost knob).
/// @param symmetry One of `"mean"`, `"forward"`, `"min"`, `"max"`; how the two
///   directions of a pair are combined *before* the top-`k` cut.
/// @param voxel Numeric; signature voxel edge in the units of `points`.
/// @param n_dirs Integer; tangent-direction bins for the signature (1 disables).
/// @param splat Logical; trilinearly splat each point over its 8 nearest voxels.
/// @param smat_values Numeric scoring matrix, or `NULL` for the built-in FCWB
///   matrix.
/// @param dist_edges Numeric vector of distance bin edges for `smat_values`.
/// @param dot_edges Numeric vector of dot-product bin edges for `smat_values`.
/// @param normalize Logical; normalise each score by the query self-match score.
/// @param limit_dist Optional numeric distance cut-off; `NULL` disables it.
/// @param n_cores Optional integer thread count; `NULL` or `<= 0` uses all cores.
/// @param precision Integer; compute in 32- or 64-bit floats.
/// @param progress Logical; display a progress bar.
/// @return A list with `idx`, an integer `(n_query, k)` matrix of **1-based**
///   neighbour indices, and `scores`, the matching numeric matrix. Rows with
///   fewer than `k` candidates are padded with `NA` in both.
/// @noRd
#[extendr]
#[allow(clippy::too_many_arguments)]
pub fn nblast_knn_raw(
    points: List,
    vects: List,
    alphas: Robj,
    t_points: Robj,
    t_vects: Robj,
    t_alphas: Robj,
    k: i32,
    n_candidates: i32,
    symmetry: &str,
    voxel: f64,
    n_dirs: i32,
    splat: bool,
    smat_values: Robj,
    dist_edges: Option<Vec<f64>>,
    dot_edges: Option<Vec<f64>>,
    normalize: bool,
    limit_dist: Option<f64>,
    n_cores: Option<i32>,
    precision: i32,
    progress: bool,
) -> Robj {
    if k < 1 {
        panic!("`k` must be >= 1");
    }
    if voxel <= 0.0 {
        panic!("`voxel` must be positive");
    }
    if t_points.is_null() != t_vects.is_null() {
        panic!("`t_points` and `t_vects` must be given together");
    }
    let k = k as usize;
    let clouds = to_clouds(&points);
    let vecs = to_clouds(&vects);
    let alpha_vecs = to_alphas(alphas);
    let smat = build_smat(smat_values, dist_edges, dot_edges, alpha_vecs.is_some());
    let nq = clouds.len();
    let opts = KnnOpts {
        nblast: Opts {
            smat: &smat,
            normalize,
            limit_dist,
            threads: to_threads(n_cores),
            progress,
            cancel: None,
        },
        k,
        n_candidates: n_candidates.max(0) as usize,
        voxel,
        n_dirs: n_dirs.max(1) as usize,
        splat,
        symmetry: to_symmetry(symmetry),
    };

    let targets = if t_points.is_null() {
        None
    } else {
        let tp = <List>::try_from(t_points).expect("`t_points` must be a list of matrices");
        let tv = <List>::try_from(t_vects).expect("`t_vects` must be a list of matrices");
        if tp.len() != tv.len() {
            panic!("`t_points` and `t_vects` must have the same length");
        }
        Some((to_clouds(&tp), to_clouds(&tv), to_alphas(t_alphas)))
    };

    let (idx, scores): (Vec<i64>, Vec<f64>) = match (precision, targets) {
        (32, Some((tp, tv, ta))) => {
            let (i, s) = fastcore::nblast_knn::nblast_knn_query_target::<f32, f64>(
                clouds, vecs, alpha_vecs, tp, tv, ta, opts,
            );
            (i, s.into_iter().map(|x| x as f64).collect())
        }
        (64, Some((tp, tv, ta))) => fastcore::nblast_knn::nblast_knn_query_target::<f64, f64>(
            clouds, vecs, alpha_vecs, tp, tv, ta, opts,
        ),
        (32, None) => {
            let (i, s) =
                fastcore::nblast_knn::nblast_knn::<f32, f64>(clouds, vecs, alpha_vecs, opts);
            (i, s.into_iter().map(|x| x as f64).collect())
        }
        (64, None) => fastcore::nblast_knn::nblast_knn::<f64, f64>(clouds, vecs, alpha_vecs, opts),
        _ => panic!("`precision` must be 32 or 64"),
    };
    knn_to_r(&idx, &scores, nq, k)
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
        32 => fastcore::synblast::synblast_allbyall::<f32, f64>(clouds, tys, opts)
            .into_iter()
            .map(|x| x as f64)
            .collect(),
        64 => fastcore::synblast::synblast_allbyall::<f64, f64>(clouds, tys, opts),
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
        32 => fastcore::synblast::synblast_query_target::<f32, f64>(qp, qt, tp, tt, opts)
            .into_iter()
            .map(|x| x as f64)
            .collect(),
        64 => fastcore::synblast::synblast_query_target::<f64, f64>(qp, qt, tp, tt, opts),
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

// ---------------------------------------------------------------------------
// Landmark transforms: thin-plate spline and moving least squares
// ---------------------------------------------------------------------------

fn xform_threads(n_cores: Option<i32>) -> Option<usize> {
    n_cores.map(|n| n.max(1) as usize)
}

/// A fitted thin-plate spline.
///
/// Held behind an external pointer because the fit is *cubic* in the landmark count -- a
/// few thousand landmarks is a second of work -- while applying it is linear. Refitting per
/// call, which a stateless `f(source, target, points)` would force, would dominate every
/// realistic workload.
///
/// The fit is stored as an `Option` with the error alongside rather than being unwrapped
/// here: extendr 0.7 loses a panic's payload when it is raised from an *associated*
/// function, so a singular system would reach R as the useless "User function panicked:
/// fit". `tps_transform()` reads `error()` immediately and raises a real R condition.
pub struct TpsTransformPtr {
    inner: Option<TpsTransform>,
    error: String,
}

#[extendr]
impl TpsTransformPtr {
    /// Fit the spline mapping `source` onto `target`. Shapes are validated R-side.
    fn fit(source: RMatrix<f64>, target: RMatrix<f64>) -> Self {
        let src = rmatrix_to_coords(&source, "source");
        let trg = rmatrix_to_coords(&target, "target");
        match TpsTransform::fit(src.view(), trg.view()) {
            Ok(t) => TpsTransformPtr {
                inner: Some(t),
                error: String::new(),
            },
            Err(e) => TpsTransformPtr {
                inner: None,
                error: e.to_string(),
            },
        }
    }

    /// Rebuild from coefficients, skipping the fit.
    fn from_coefs(source: RMatrix<f64>, w: RMatrix<f64>, a: RMatrix<f64>) -> Self {
        let src = rmatrix_to_coords(&source, "source");
        let wm = rmatrix_to_coords(&w, "W");
        let am = rmatrix_to_coords(&a, "A");
        match TpsTransform::from_coefs(src.view(), wm.view(), am.view()) {
            Ok(t) => TpsTransformPtr {
                inner: Some(t),
                error: String::new(),
            },
            Err(e) => TpsTransformPtr {
                inner: None,
                error: e.to_string(),
            },
        }
    }

    /// Empty when the fit succeeded; the failure message otherwise.
    fn error(&self) -> String {
        self.error.clone()
    }

    fn n_landmarks(&self) -> i32 {
        self.get().n_landmarks() as i32
    }

    fn source(&self) -> Robj {
        coords_to_rmatrix(&self.get().source())
    }

    fn weights(&self) -> Robj {
        coords_to_rmatrix(&self.get().weights())
    }

    fn affine_coefs(&self) -> Robj {
        coords_to_rmatrix(&self.get().affine_coefs())
    }

    /// The affine part as a 4x4 homogeneous matrix.
    fn matrix_affine(&self) -> Robj {
        let m = self.get().matrix_affine();
        RArray::new_matrix(4, 4, |r, c| m[r][c]).into()
    }

    fn xform(&self, coords: RMatrix<f64>, n_cores: Option<i32>) -> Robj {
        let pts = rmatrix_to_coords(&coords, "xyz");
        let out = self
            .get()
            .xform(pts.view(), xform_threads(n_cores), None)
            .unwrap_or_else(|e| panic!("{e}"));
        coords_to_rmatrix(&out)
    }
}

impl TpsTransformPtr {
    /// `tps_transform()` refuses to build an object around a failed fit, so by the time any
    /// method runs this is always populated.
    fn get(&self) -> &TpsTransform {
        self.inner
            .as_ref()
            .expect("TPS transform was used despite a failed fit")
    }
}

/// Landmark pairs defining a moving-least-squares warp.
///
/// There is no fit to cache -- every point is solved independently -- so this pointer
/// exists only to avoid re-copying the landmarks on every call, and to give R an object
/// with the same shape as the other transforms.
pub struct MlsTransformPtr {
    inner: Option<MlsTransform>,
    error: String,
}

#[extendr]
impl MlsTransformPtr {
    fn build(source: RMatrix<f64>, target: RMatrix<f64>) -> Self {
        let src = rmatrix_to_coords(&source, "source");
        let trg = rmatrix_to_coords(&target, "target");
        match MlsTransform::new(src.view(), trg.view()) {
            Ok(t) => MlsTransformPtr {
                inner: Some(t),
                error: String::new(),
            },
            Err(e) => MlsTransformPtr {
                inner: None,
                error: e.to_string(),
            },
        }
    }

    /// Empty when construction succeeded; the failure message otherwise.
    fn error(&self) -> String {
        self.error.clone()
    }

    fn n_landmarks(&self) -> i32 {
        self.get().n_landmarks() as i32
    }

    fn source(&self) -> Robj {
        coords_to_rmatrix(&self.get().source())
    }

    fn target(&self) -> Robj {
        coords_to_rmatrix(&self.get().target())
    }

    /// The *global* affine as a 4x4 homogeneous matrix.
    fn matrix_affine(&self, reverse: bool) -> Robj {
        let m = self.get().matrix_affine(reverse);
        RArray::new_matrix(4, 4, |r, c| m[r][c]).into()
    }

    fn xform(&self, coords: RMatrix<f64>, reverse: bool, n_cores: Option<i32>) -> Robj {
        let pts = rmatrix_to_coords(&coords, "xyz");
        let out = self
            .get()
            .xform(pts.view(), reverse, xform_threads(n_cores), None)
            .unwrap_or_else(|e| panic!("{e}"));
        coords_to_rmatrix(&out)
    }
}

impl MlsTransformPtr {
    fn get(&self) -> &MlsTransform {
        self.inner
            .as_ref()
            .expect("MLS transform was used despite a failed construction")
    }
}

// ---------------------------------------------------------------------------
// Hierarchical clustering
// ---------------------------------------------------------------------------

fn linkage_method(name: &str) -> LinkageMethod {
    LinkageMethod::from_name(name).unwrap_or_else(|| {
        panic!(
            "unknown `method` '{name}'; expected one of single, complete, average, \
             weighted, ward, centroid, median"
        )
    })
}

fn linkage_symmetry(name: &str) -> Symmetry {
    Symmetry::from_name(name).unwrap_or_else(|| {
        panic!("unknown `symmetry` '{name}'; expected one of none, mean, min, max")
    })
}

fn linkage_transform(name: &str) -> Transform {
    Transform::from_name(name).unwrap_or_else(|| {
        panic!("unknown `transform` '{name}'; expected one of one_minus, none")
    })
}

/// Borrow an R matrix as an `ndarray` view, without copying.
///
/// R matrices are column-major, so this is an F-order view. The clustering kernels
/// take that in their stride: they transpose the view rather than copy it, and
/// symmetrising is invariant under the transpose (`Symmetry::None` compensates by
/// swapping which of the two cells it reads). A 100k score matrix therefore reaches
/// the kernel without being materialised a second time.
fn rmatrix_to_view(m: &RMatrix<f64>) -> ArrayView2<'_, f64> {
    let (nr, nc) = (m.nrows(), m.ncols());
    ArrayView2::from_shape((nr, nc).f(), m.data())
        .unwrap_or_else(|e| panic!("could not view `scores` as a matrix: {e}"))
}

/// Validate an optional `labels` argument against the observation count.
fn check_labels(labels: Robj, n: usize) -> Robj {
    if labels.is_null() {
        return labels;
    }
    let len = labels
        .as_string_vector()
        .unwrap_or_else(|| panic!("`labels` must be a character vector or NULL"))
        .len();
    if len != n {
        panic!("`labels` must have one entry per observation: got {len}, want {n}");
    }
    labels
}

/// Turn a SciPy-style linkage matrix into an R `hclust` object.
///
/// The two labelling schemes differ: SciPy numbers singletons `0..n` and the
/// cluster formed at step `i` as `n + i`, whereas R writes `-j` for observation `j`
/// (1-based) and `+k` for the cluster formed at the earlier step `k` (1-based).
/// Because `kodama` already emits the smaller id first, and singleton ids are all
/// below the merged ones, the negatives land first in each row exactly as R's own
/// `hclust` writes them.
fn hclust_from_z(z: &Array2<f64>, n: usize, method: &str, labels: Robj) -> Robj {
    let k = n - 1;
    let to_r = |c: f64| {
        let c = c as usize;
        if c < n {
            -((c + 1) as i32)
        } else {
            (c - n + 1) as i32
        }
    };

    let merge = RArray::new_matrix(k, 2, |r, c| to_r(z[[r, c]]));
    let height: Vec<f64> = (0..k).map(|r| z[[r, 2]]).collect();
    // `order` must use the same child ordering as `merge`, or the dendrogram draws
    // with crossing branches; both read the one `z`.
    let order: Vec<i32> = leaf_order(z, n)
        .into_iter()
        .map(|i| (i + 1) as i32)
        .collect();

    let mut out: Robj = List::from_names_and_values(
        ["merge", "height", "order", "labels", "method", "dist.method"],
        [
            merge.into_robj(),
            height.into_robj(),
            order.into_robj(),
            labels,
            method.into_robj(),
            r!(NULL),
        ],
    )
    .unwrap_or_else(|e| panic!("could not build hclust object: {e}"))
    .into();
    out.set_class(["hclust"])
        .unwrap_or_else(|e| panic!("could not set class: {e}"));
    out
}

// Hierarchical clustering of a square score matrix, fusing symmetrise + transform +
// condense into one pass and then clustering that buffer in place.
//
// Argument coercion, defaults and the user-facing documentation live in
// R/clustering.R; this is the raw entry point, as with `probe_invertible_raw`.
#[extendr]
fn nblast_hclust_raw(
    scores: RMatrix<f64>,
    method: &str,
    symmetry: &str,
    transform: &str,
    labels: Robj,
    n_cores: Option<i32>,
) -> Robj {
    let view = rmatrix_to_view(&scores);
    let n = view.nrows();
    let labels = check_labels(labels, n);

    let z = linkage_from_scores(
        view,
        linkage_method(method),
        linkage_symmetry(symmetry),
        linkage_transform(transform),
        to_threads(n_cores),
        None,
    )
    .unwrap_or_else(|e| panic!("{e}"));

    hclust_from_z(&z, n, method, labels)
}

// Condensed distances from a square score matrix, as an R `dist` object.
//
// R stores a `dist` as the lower triangle by column, which for a symmetric matrix is
// element-for-element the same sequence as the upper triangle by row that the kernel
// writes — so the fused output needs no rearranging, only its attributes.
//
// Documented R-side in R/clustering.R.
#[extendr]
fn nblast_dist_raw(
    scores: RMatrix<f64>,
    symmetry: &str,
    transform: &str,
    labels: Robj,
    n_cores: Option<i32>,
) -> Robj {
    let view = rmatrix_to_view(&scores);
    let n = view.nrows();
    let labels = check_labels(labels, n);

    let cond = condense(
        view,
        linkage_symmetry(symmetry),
        linkage_transform(transform),
        to_threads(n_cores),
        None,
    )
    .unwrap_or_else(|e| panic!("{e}"));

    let mut out: Robj = cond.into_robj();
    out.set_attrib("Size", (n as i32).into_robj()).unwrap();
    out.set_attrib("Diag", false.into_robj()).unwrap();
    out.set_attrib("Upper", false.into_robj()).unwrap();
    out.set_attrib("Labels", labels).unwrap();
    out.set_class(["dist"]).unwrap();
    out
}

// Hierarchical clustering of an existing condensed distance vector.
//
// Clustering consumes its input as scratch and R's value semantics forbid writing to
// the caller's vector, so `d` is copied once here. Documented R-side in
// R/clustering.R.
#[extendr]
fn fast_hclust_raw(d: Robj, method: &str, labels: Robj) -> Robj {
    let slice = d
        .as_real_slice()
        .unwrap_or_else(|| panic!("`d` must be a `dist` object or a numeric vector"));

    // Prefer the declared Size; fall back to solving n(n-1)/2 = length(d).
    let n = match d.get_attrib("Size").and_then(|s| s.as_integer()) {
        Some(size) if size >= 2 => size as usize,
        _ => observations_from_condensed(slice.len())
            .unwrap_or_else(|| panic!("length {} is not n(n-1)/2 for any n", slice.len())),
    };
    if slice.len() != n * (n - 1) / 2 {
        panic!(
            "`d` has {} entries, but Size = {n} implies {}",
            slice.len(),
            n * (n - 1) / 2
        );
    }

    let labels = if labels.is_null() {
        d.get_attrib("Labels").unwrap_or_else(|| r!(NULL))
    } else {
        labels
    };
    let labels = check_labels(labels, n);

    // The one copy R's semantics force on us; see the note above.
    let mut buf = slice.to_vec();
    let z = core_linkage(&mut buf, n, linkage_method(method))
        .unwrap_or_else(|e| panic!("{e}"));

    hclust_from_z(&z, n, method, labels)
}

// Symmetrise a square score matrix. Documented R-side in R/clustering.R.
//
// R's value semantics forbid writing through to the caller's matrix, so unlike the
// Python binding — which symmetrises in place and allocates nothing — this copies
// once and symmetrises the copy. That is still one `n x n` against the two or three
// `(m + t(m)) / 2` builds on the way to the same answer.
#[extendr]
fn symmetrize_raw(scores: RMatrix<f64>, symmetry: &str, n_cores: Option<i32>) -> Robj {
    let (nr, nc) = (scores.nrows(), scores.ncols());
    let mut data: Vec<f64> = scores.data().to_vec();

    {
        // `.f()`: R matrices are column-major. Viewing them as row-major would
        // transpose, which the combining modes would survive and `"none"` — mirror
        // the upper triangle — would not.
        let view = ArrayViewMut2::from_shape((nr, nc).f(), &mut data)
            .unwrap_or_else(|e| panic!("could not view `scores` as a matrix: {e}"));
        symmetrize(view, linkage_symmetry(symmetry), to_threads(n_cores), None)
            .unwrap_or_else(|e| panic!("{e}"));
    }

    RArray::new_matrix(nr, nc, |r, c| data[c * nr + r]).into()
}

// Dendrogram leaf order for an `hclust`-style merge matrix. Documented R-side in
// R/clustering.R, which validates the matrix first: an entry naming a merge that has
// not happened yet would send the walk into a loop rather than out to the leaves.
#[extendr]
fn leaf_order_raw(merge: RMatrix<i32>) -> Vec<i32> {
    let k = merge.nrows();
    let n = k + 1;
    let d = merge.data();

    // R writes `-j` for observation `j` (1-based) and `+i` for the cluster formed at
    // step `i` (1-based); the core reads SciPy's labelling, where observations are
    // `0..n` and step `i` yields `n + i`. `hclust_from_z` is this map in reverse.
    let z = Array2::from_shape_fn((k, 2), |(r, c)| {
        let v = d[c * k + r];
        if v < 0 {
            (-v - 1) as f64
        } else {
            (n as i32 + v - 1) as f64
        }
    });

    leaf_order(&z, n).into_iter().map(|i| i as i32 + 1).collect()
}

/// Set the number of threads used for parallel work in this session.
///
/// By default nat.fastcore uses every core it can see, which is the right answer
/// for a single call and the wrong one when the *caller* is already spreading
/// work over processes (`parallel::mclapply()`, `future::plan(multisession)`, a
/// cluster job): each worker would claim every core, and the resulting
/// oversubscription can make the whole thing slower than running it on one core.
/// Nothing tells a worker process that it is one of twenty, so it has to be told.
///
/// Call this once, before any other nat.fastcore function. The pool is built at
/// most once per session, by whichever comes first: an earlier
/// `set_num_threads()`, the `RAYON_NUM_THREADS` environment variable, a call to
/// `get_num_threads()`, or simply the first parallel call. Calling it again with
/// the same `n` is a no-op; calling it with a different `n` is an error, as the
/// pool cannot be resized.
///
/// @param n Integer; number of threads. Must be >= 1.
/// @return `NULL`, invisibly. Called for its side effect.
/// @export
#[extendr]
pub fn set_num_threads(n: i32) {
    // `panic!` as everywhere else in this file. Worth knowing what R sees: the
    // condition message is extendr's generic "User function panicked:
    // set_num_threads", and the text below reaches the console on stderr rather
    // than `conditionMessage()`. Returning a `Result` does not improve on that —
    // the generated wrapper unwraps it and panics anyway — and `throw_r_error`
    // longjmps past Rust frames, which is a poor trade for a better string.
    if n < 1 {
        panic!("`n` must be >= 1, got {n}");
    }
    if let Err(e) = fastcore::threads::set_num_threads(n as usize) {
        panic!("{e}");
    }
}

/// Number of threads available for parallel work in this session.
///
/// Note that asking builds the thread pool if it does not exist yet — which is
/// then exactly what makes a subsequent `set_num_threads()` fail. Set first, ask
/// second.
///
/// @return Integer; the number of threads.
/// @export
#[extendr]
pub fn get_num_threads() -> i32 {
    fastcore::threads::num_threads() as i32
}

// Macro to generate exports.
// This ensures exported functions are registered with R.
// See corresponding C code in `entrypoint.c`.
extendr_module! {
    mod nat_fastcore;
    impl CmtkRegistration;
    impl ElastixTransformPtr;
    impl TpsTransformPtr;
    impl MlsTransformPtr;
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
    fn descendants;
    fn paths_to_root;
    fn reroot;
    fn contract_nodes;
    fn simplify_skeleton;
    fn downsample_skeleton;
    fn simplify_rdp;
    fn simplify_vw;
    fn resample_skeleton;
    fn smooth_skeleton;
    fn smooth_skeleton_gaussian;
    fn adjacency;
    fn longest_path;
    fn longest_paths;
    fn betweenness;
    fn descendant_counts;
    fn stitch_fragments;
    fn reroot_rewire;
    fn heal_skeleton;
    fn mesh_connected_components;
    fn geodesic_matrix_mesh;
    fn geodesic_matrix_graph;
    fn geodesic_nearest_mesh;
    fn geodesic_farthest_mesh;
    fn unique_edges;
    fn connected_components_graph;
    fn level_set_components;
    fn contract_vertices;
    fn minimum_spanning_tree;
    fn parents_from_edges;
    fn bridges;
    fn geodesic_mst_mesh;
    fn geodesic_mst_graph;
    fn geodesic_predecessors;
    fn geodesic_path;
    fn geodesic_clusters;
    fn simplify_mesh;
    fn simplify_mesh_lossless;
    fn smat_auto_limit;
    fn nblast_allbyall;
    fn nblast;
    fn nblast_pairs;
    fn nblast_knn_raw;
    fn synblast_allbyall;
    fn synblast;
    fn nblast_hclust_raw;
    fn nblast_dist_raw;
    fn fast_hclust_raw;
    fn symmetrize_raw;
    fn leaf_order_raw;
    fn set_num_threads;
    fn get_num_threads;
}
