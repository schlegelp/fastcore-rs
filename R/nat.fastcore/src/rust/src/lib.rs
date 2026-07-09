use extendr_api::prelude::*;
use ndarray::{Array1, Array2};
use std::collections::HashMap;

use fastcore::nblast::{load_smat, load_smat_alpha, Opts, Smat};

/// For each node ID in `parents` find its index in `nodes`.
///
/// Importantly this is 0-indexed to match indexing in Rust.
/// Roots will have parent index -1.
///
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

/// Connected components.
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
/// @export
#[extendr]
pub fn prune_twigs(parents: Vec<i32>, threshold: f64, weights: Option<Vec<f64>>) -> Vec<i32> {
    let parents = Array1::from_vec(parents);
    let weights: Option<Array1<f64>> = weights.map(Array1::from_vec);

    // Mask is currently not supported - strangely, extendr does not seem to support Vec<bool>
    fastcore::dag::prune_twigs(&parents.view(), threshold as f32, &weights, &None)
}

/// Return path length from a single node to the root.
/// @export
#[extendr]
pub fn dist_to_root(parents: Vec<i32>, node: i32) -> f64 {
    let parents = Array1::from_vec(parents);
    fastcore::dag::dist_to_root(&parents.view(), node) as f64
}

/// Classify nodes into roots (0), leaves (1), branch points (2) and slabs (3).
/// @export
#[extendr]
pub fn classify_nodes(parents: Vec<i32>) -> Vec<i32> {
    let parents = Array1::from_vec(parents);
    fastcore::dag::classify_nodes(&parents.view()).to_vec()
}

/// Check whether the tree contains cycles.
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

/// Synapse flow centrality for each node.
///
/// `presynapses`/`postsynapses` give the number of pre-/post-synapses at each node.
/// `mode` is one of "centrifugal", "centripetal" or "sum".
///
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
/// @export
#[extendr]
pub fn break_segments(parents: Vec<i32>) -> Robj {
    let parents = Array1::from_vec(parents);
    let segments = fastcore::dag::break_segments(&parents.view());
    List::from_values(segments.into_iter()).into()
}

/// Find connected components of a triangle mesh.
///
/// `faces` is an (N, 3) matrix of vertex indices. Returns an integer vector of
/// length `n_vertices` assigning each vertex the root-vertex index of its
/// component.
///
/// @export
#[extendr]
pub fn mesh_connected_components(faces: Robj, n_vertices: i32) -> Vec<i32> {
    let faces_u32 = robj_to_faces(&faces);
    fastcore::mesh::mesh_connected_components(faces_u32.view(), n_vertices as usize)
        .iter()
        .map(|&x| x as i32)
        .collect()
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
    fn dist_to_root;
    fn classify_nodes;
    fn has_cycles;
    fn geodesic_pairs;
    fn geodesic_nearest;
    fn synapse_flow_centrality;
    fn generate_segments;
    fn break_segments;
    fn mesh_connected_components;
    fn smat_auto_limit;
    fn nblast_allbyall;
    fn nblast;
    fn nblast_pairs;
    fn synblast_allbyall;
    fn synblast;
}
