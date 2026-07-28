use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rayon::prelude::*;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::nblast::with_pool;

/// Path-halving find: iterative, no stack allocation.
/// Makes every other node on the path point to its grandparent.
#[inline]
fn find(parent: &mut [u32], mut x: u32) -> u32 {
    loop {
        let p = parent[x as usize];
        if p == x {
            return x;
        }
        // Path-halving: point x to its grandparent
        let gp = parent[p as usize];
        parent[x as usize] = gp;
        x = gp;
    }
}

/// Union two nodes, attaching the larger root to the smaller.
///
/// Keeping the *smaller* index as the root is what makes the component label of a set of
/// vertices independent of the order the edges arrived in — callers rely on the label being
/// the minimum vertex index of the component.
#[inline]
fn union(parent: &mut [u32], a: u32, b: u32) {
    let ra = find(parent, a);
    let rb = find(parent, b);
    if ra != rb {
        if ra < rb {
            parent[rb as usize] = ra;
        } else {
            parent[ra as usize] = rb;
        }
    }
}

/// Find connected components of a triangle mesh.
///
/// Uses Union-Find (DSU) with path-halving. The only extra allocation is a
/// single `Vec<u32>` of length `n_vertices` for the parent array — no
/// adjacency list is built.
///
/// Arguments
/// ---------
/// - `faces`:       (N, 3) array of triangular faces given as vertex indices.
/// - `n_vertices`:  Total number of vertices.
///
/// Returns
/// -------
/// A `Vec<u32>` of length `n_vertices` where each entry contains the
/// root-vertex index of the component the vertex belongs to.
pub fn mesh_connected_components(faces: ArrayView2<u32>, n_vertices: usize) -> Vec<u32> {
    // Each vertex is its own parent initially — the only allocation.
    let mut parent: Vec<u32> = (0..n_vertices as u32).collect();

    // Walk every face and union the three vertices. Unioning a–b and a–c is enough; b–c
    // follows by transitivity.
    for face in faces.rows() {
        union(&mut parent, face[0], face[1]);
        union(&mut parent, face[0], face[2]);
    }

    // Final compression: make every vertex point directly to its root.
    for i in 0..n_vertices {
        parent[i] = find(&mut parent, i as u32);
    }

    parent
}

// ---------------------------------------------------------------------------
// Unique edges
// ---------------------------------------------------------------------------

/// Pack an undirected edge into one sortable integer: larger vertex index in the
/// high 32 bits, smaller in the low 32. Ascending key order is therefore
/// (max, min) — the exact order trimesh's `edges_unique` produces, since its
/// row hash `((b + 2^31) << 32) | (a + 2^31)` only differs by a monotone offset.
#[inline]
fn edge_key(u: u32, v: u32) -> u64 {
    let (lo, hi) = if u <= v { (u, v) } else { (v, u) };
    ((hi as u64) << 32) | (lo as u64)
}

/// Unique undirected edges of a triangle mesh — a drop-in for trimesh's
/// `edges_unique` (plus `edges_unique_idx`, `edges_unique_inverse` and
/// `edges_unique_length`).
///
/// Each face `(a, b, c)` contributes the edges `(a, b), (b, c), (c, a)`, in that
/// order, giving a conceptual `3F`-long edge list. Edges are undirected, so each
/// pair is normalised to `[min, max]` before dedup. Self-loops from degenerate
/// faces are kept, as trimesh does.
///
/// Arguments
/// ---------
/// - `faces`:          (F, 3) array of triangular faces given as vertex indices.
/// - `coords`:         (V, 3) vertex positions; when given, also return the
///   euclidean length of each unique edge (trimesh's `edges_unique_length`).
/// - `return_index`:   Also return, per unique edge, the index of its first
///   occurrence in the `3F` edge list.
/// - `return_inverse`: Also return, per edge in the `3F` list, the row of its
///   unique edge (reshape to `(F, 3)` for per-face edge ids).
/// - `threads`:        Size of the thread pool, or `None` for all cores.
///
/// Returns
/// -------
/// `(edges, index, inverse, lengths)` where `edges` is a `(n_unique, 2)` i64
/// array with rows `[min, max]` sorted ascending by `(max, min)` — byte-for-byte
/// the order, dtype and first-occurrence semantics of trimesh / `np.unique`.
#[allow(clippy::type_complexity)]
pub fn unique_edges(
    faces: ArrayView2<u32>,
    coords: Option<ArrayView2<f64>>,
    return_index: bool,
    return_inverse: bool,
    threads: Option<usize>,
) -> (
    Array2<i64>,
    Option<Array1<i64>>,
    Option<Array1<i64>>,
    Option<Array1<f64>>,
) {
    let n_edges = faces.nrows() * 3;

    // The Python wrapper always hands us C-order (borrowed as-is); a strided
    // view from a Rust caller gets copied into standard layout.
    let storage = faces.as_standard_layout();
    let s: &[u32] = storage.as_slice().expect("standard layout is contiguous");

    with_pool(threads, || {
        let (edges, index, inverse) = if !return_index && !return_inverse {
            // Fast path: sort bare keys, dedup in one scan.
            let mut keys = vec![0u64; n_edges];
            keys.par_chunks_exact_mut(3)
                .zip(s.par_chunks_exact(3))
                .for_each(|(out, f)| {
                    out[0] = edge_key(f[0], f[1]);
                    out[1] = edge_key(f[1], f[2]);
                    out[2] = edge_key(f[2], f[0]);
                });
            keys.par_sort_unstable();

            let n_unique =
                keys.windows(2).filter(|w| w[0] != w[1]).count() + usize::from(!keys.is_empty());
            let mut edges: Vec<i64> = Vec::with_capacity(n_unique * 2);
            let mut prev = None;
            for &k in &keys {
                if prev != Some(k) {
                    edges.push((k & 0xFFFF_FFFF) as i64);
                    edges.push((k >> 32) as i64);
                    prev = Some(k);
                }
            }
            (edges, None, None)
        } else {
            // Full path: fold each edge's position in the 3F list into the low 64
            // bits so one *unstable* integer sort still lands ties in original
            // order — which is exactly np.unique's stable-argsort "first
            // occurrence" semantics.
            let mut packed = vec![0u128; n_edges];
            packed
                .par_chunks_exact_mut(3)
                .zip(s.par_chunks_exact(3))
                .enumerate()
                .for_each(|(i, (out, f))| {
                    let e = (3 * i) as u128;
                    out[0] = ((edge_key(f[0], f[1]) as u128) << 64) | e;
                    out[1] = ((edge_key(f[1], f[2]) as u128) << 64) | (e + 1);
                    out[2] = ((edge_key(f[2], f[0]) as u128) << 64) | (e + 2);
                });
            packed.par_sort_unstable();

            let mut edges: Vec<i64> = Vec::new();
            let mut index: Vec<i64> = Vec::new();
            let mut inverse: Vec<i64> = if return_inverse { vec![0; n_edges] } else { Vec::new() };
            let mut prev: Option<u64> = None;
            let mut slot: i64 = -1;
            for &p in &packed {
                let key = (p >> 64) as u64;
                let orig = p as u64;
                if prev != Some(key) {
                    slot += 1;
                    edges.push((key & 0xFFFF_FFFF) as i64);
                    edges.push((key >> 32) as i64);
                    if return_index {
                        index.push(orig as i64);
                    }
                    prev = Some(key);
                }
                if return_inverse {
                    inverse[orig as usize] = slot;
                }
            }
            (
                edges,
                return_index.then_some(index),
                return_inverse.then_some(inverse),
            )
        };

        let lengths = coords.map(|c| edge_lengths(&edges, c));
        let n = edges.len() / 2;
        (
            Array2::from_shape_vec((n, 2), edges).unwrap(),
            index.map(Array1::from_vec),
            inverse.map(Array1::from_vec),
            lengths,
        )
    })
}

/// Euclidean length of each `[a, b]` edge in a flat pair list.
///
/// Runs on the ambient rayon pool — callers wrap it in `with_pool`.
fn edge_lengths(edges: &[i64], coords: ArrayView2<f64>) -> Array1<f64> {
    let storage = coords.as_standard_layout();
    let c: &[f64] = storage.as_slice().expect("standard layout is contiguous");
    let mut out = vec![0f64; edges.len() / 2];
    out.par_iter_mut()
        .zip(edges.par_chunks_exact(2))
        .for_each(|(o, e)| {
            let a = &c[3 * e[0] as usize..3 * e[0] as usize + 3];
            let b = &c[3 * e[1] as usize..3 * e[1] as usize + 3];
            let dx = a[0] - b[0];
            let dy = a[1] - b[1];
            let dz = a[2] - b[2];
            *o = (dx * dx + dy * dy + dz * dz).sqrt();
        });
    Array1::from_vec(out)
}

// ---------------------------------------------------------------------------
// Graph primitives
// ---------------------------------------------------------------------------

/// Range-check an edge list and return its row count.
fn check_edges(edges: ArrayView2<u32>, n_nodes: usize) -> usize {
    assert_eq!(edges.ncols(), 2, "`edges` must have shape (E, 2)");
    for e in edges.rows() {
        for &v in e {
            assert!(
                (v as usize) < n_nodes,
                "edge references node {v}, but n_nodes = {n_nodes}"
            );
        }
    }
    edges.nrows()
}

/// Connected components of an undirected graph given as an edge list.
///
/// The edge-list counterpart of [`mesh_connected_components`], and the same Union-Find: one
/// `Vec<u32>` of length `n_nodes`, no adjacency list. Use this when the graph is not a
/// triangle mesh (or when you already hold the deduplicated edges and would rather not walk
/// the faces again).
///
/// Arguments
/// ---------
/// - `edges`:   (E, 2) array of edges given as node indices. Direction is ignored;
///   self-loops and parallel edges are harmless.
/// - `n_nodes`: Total number of nodes. Nodes not named by any edge are isolated
///   components of size one.
///
/// Returns
/// -------
/// A `Vec<u32>` of length `n_nodes` holding, per node, the root of its component — which is
/// the smallest node index in that component.
pub fn connected_components_graph(edges: ArrayView2<u32>, n_nodes: usize) -> Vec<u32> {
    check_edges(edges, n_nodes);

    let mut parent: Vec<u32> = (0..n_nodes as u32).collect();
    for e in edges.rows() {
        union(&mut parent, e[0], e[1]);
    }
    for i in 0..n_nodes {
        parent[i] = find(&mut parent, i as u32);
    }
    parent
}

/// Connected components of every level set at once.
///
/// Given a label per node, this finds the connected components of each subgraph induced by
/// the nodes sharing a label — all labels in a single pass, by unioning an edge only when its
/// two endpoints agree. It is the primitive behind "which nodes were reached by the same
/// wavefront and are actually touching", where the label is a (binned) geodesic distance and
/// each component is one ring around the structure.
///
/// The point is that it replaces a *loop*. Done with a general-purpose graph library the same
/// result costs one induced-subgraph construction plus one component search per distinct
/// label, so a mesh with a thousand levels pays a thousand graph builds; here the whole thing
/// is one `O(E α(N))` sweep over the edges, and the only allocations are three `N`-sized
/// integer arrays.
///
/// Arguments
/// ---------
/// - `edges`:   (E, 2) array of edges given as node indices.
/// - `n_nodes`: Total number of nodes.
/// - `labels`:  One label per node. **Negative labels mark excluded nodes**: they join no
///   component and come back as `-1`. That is what makes the output of a search that could
///   not reach everything (`geodesic_matrix_*` returns `-1.0` for unreachable) usable here
///   directly, rather than lumping every unreachable node into one bogus level.
///
/// Returns
/// -------
/// `(ids, n_components)` where `ids[i]` is the component of node `i` in `[0, n_components)`,
/// or `-1` if the node was excluded. Ids are contiguous and assigned in order of first
/// appearance scanning nodes low to high, so they are deterministic and can index straight
/// into a `n_components`-long accumulator — no separate "unique" pass needed.
pub fn level_set_components(
    edges: ArrayView2<u32>,
    n_nodes: usize,
    labels: ArrayView1<i64>,
) -> (Vec<i32>, usize) {
    check_edges(edges, n_nodes);
    assert_eq!(
        labels.len(),
        n_nodes,
        "`labels` must have one entry per node ({n_nodes}), got {}",
        labels.len()
    );

    let mut parent: Vec<u32> = (0..n_nodes as u32).collect();
    for e in edges.rows() {
        let (u, v) = (e[0], e[1]);
        // A negative label excludes the node outright, so testing `>= 0` on one endpoint is
        // enough — if it passes and the labels are equal, neither is negative.
        let lu = labels[u as usize];
        if lu >= 0 && lu == labels[v as usize] {
            union(&mut parent, u, v);
        }
    }

    // Densify. `remap` is indexed by DSU root, which is always a node index, so one N-sized
    // array covers every possible root without a hash map.
    let mut remap: Vec<i32> = vec![-1; n_nodes];
    let mut ids: Vec<i32> = vec![-1; n_nodes];
    let mut n_components: usize = 0;
    for i in 0..n_nodes {
        if labels[i] < 0 {
            continue;
        }
        let root = find(&mut parent, i as u32) as usize;
        if remap[root] < 0 {
            remap[root] = n_components as i32;
            n_components += 1;
        }
        ids[i] = remap[root];
    }

    (ids, n_components)
}

/// Contract nodes onto new ids, returning the simplified edge list.
///
/// Both endpoints of every edge are pushed through `mapping`; edges that end up with both
/// ends on the same new node (self-loops) are dropped, and the rest are deduplicated. This is
/// igraph's `contract_vertices()` followed by `simplify()`, fused — and, unlike igraph's
/// version, it does not rewrite a graph object in place, so contracting does not cost a copy
/// of the graph.
///
/// Arguments
/// ---------
/// - `edges`:   (E, 2) array of edges given as node indices.
/// - `mapping`: New id for each old node, i.e. `mapping[old] = new`. Ids need not be
///   contiguous, but the output is only as compact as the ids you supply.
/// - `threads`: Size of the thread pool, or `None` for all cores.
///
/// Returns
/// -------
/// An `(n_unique, 2)` i64 array of the surviving edges as `[min, max]` rows, sorted ascending
/// by `(max, min)` — the same ordering [`unique_edges`] produces.
pub fn contract_vertices(
    edges: ArrayView2<u32>,
    mapping: ArrayView1<u32>,
    threads: Option<usize>,
) -> Array2<i64> {
    assert_eq!(edges.ncols(), 2, "`edges` must have shape (E, 2)");
    let n_old = mapping.len();
    for e in edges.rows() {
        for &v in e {
            assert!(
                (v as usize) < n_old,
                "edge references node {v}, but `mapping` only covers {n_old} nodes"
            );
        }
    }

    let storage = edges.as_standard_layout();
    let s: &[u32] = storage.as_slice().expect("standard layout is contiguous");
    let map = mapping.as_standard_layout();
    let m: &[u32] = map.as_slice().expect("standard layout is contiguous");

    with_pool(threads, || {
        // Self-loops are dropped *before* the sort, not after: on a contraction that collapses
        // a mesh down to a skeleton the overwhelming majority of edges are internal to a group,
        // so this is what keeps the sort off the discarded bulk.
        let mut keys: Vec<u64> = s
            .par_chunks_exact(2)
            .filter_map(|e| {
                let (a, b) = (m[e[0] as usize], m[e[1] as usize]);
                (a != b).then(|| edge_key(a, b))
            })
            .collect();
        keys.par_sort_unstable();

        let mut out: Vec<i64> = Vec::new();
        let mut prev = None;
        for &k in &keys {
            if prev != Some(k) {
                out.push((k & 0xFFFF_FFFF) as i64);
                out.push((k >> 32) as i64);
                prev = Some(k);
            }
        }
        let n = out.len() / 2;
        Array2::from_shape_vec((n, 2), out).unwrap()
    })
}

/// Minimum (or maximum) spanning forest of an undirected graph.
///
/// Kruskal's algorithm on the same Union-Find as the component search above: sort the edges
/// by weight, keep the ones that join two different components. Disconnected input is fine —
/// each component contributes its own tree, so this is really a spanning *forest*, matching
/// igraph's `spanning_tree()` and scipy's `minimum_spanning_tree`.
///
/// Arguments
/// ---------
/// - `edges`:    (E, 2) array of edges given as node indices.
/// - `n_nodes`:  Total number of nodes.
/// - `weights`:  Weight per edge, or `None` to treat every edge as equal (any spanning
///   forest, edges preferred in input order). Must be finite; negative weights are allowed.
/// - `maximize`: Return the *maximum* spanning forest instead. This exists so callers do not
///   have to pass `1 / weight` to invert the ordering — a transform that both loses precision
///   and blows up on the zero weights that legitimately occur.
/// - `threads`:  Size of the thread pool, or `None` for all cores.
///
/// Returns
/// -------
/// A 1-D i64 array of *row indices into `edges`*, ascending by weight — not the edges
/// themselves, so the caller can index whatever per-edge data it holds (weights, ids,
/// attributes) with the same array. Length is `n_nodes - (number of components)`.
pub fn minimum_spanning_tree(
    edges: ArrayView2<u32>,
    n_nodes: usize,
    weights: Option<&ArrayView1<f32>>,
    maximize: bool,
    threads: Option<usize>,
) -> Array1<i64> {
    let n_edges = check_edges(edges, n_nodes);
    if let Some(w) = weights {
        assert_eq!(
            w.len(),
            n_edges,
            "`weights` must have one entry per edge ({n_edges}), got {}",
            w.len()
        );
        for &x in w {
            assert!(x.is_finite(), "edge weights must be finite, got {x}");
        }
    }

    let order: Vec<u32> = match weights {
        None => (0..n_edges as u32).collect(),
        Some(w) => with_pool(threads, || {
            let mut order: Vec<u32> = (0..n_edges as u32).collect();
            // `total_cmp` gives a total order over all finite floats including negatives, so
            // no bit-pattern trick (which would need non-negative weights) and no
            // `partial_cmp().unwrap()` panic path. Ties break on the edge index, which makes
            // the chosen tree reproducible across runs and thread counts.
            order.par_sort_unstable_by(|&a, &b| {
                w[a as usize]
                    .total_cmp(&w[b as usize])
                    .then_with(|| a.cmp(&b))
            });
            if maximize {
                order.reverse();
            }
            order
        }),
    };

    let mut parent: Vec<u32> = (0..n_nodes as u32).collect();
    // A spanning forest has at most n_nodes - 1 edges, so this never grows past that.
    let mut out: Vec<i64> = Vec::with_capacity(n_nodes.saturating_sub(1));
    for &i in &order {
        let e = edges.row(i as usize);
        let (ru, rv) = (find(&mut parent, e[0]), find(&mut parent, e[1]));
        if ru != rv {
            if ru < rv {
                parent[rv as usize] = ru;
            } else {
                parent[ru as usize] = rv;
            }
            out.push(i as i64);
            // Every accepted edge merges two components; once we are down to one there is
            // nothing left to join and the rest of the sorted list is dead weight.
            if out.len() + 1 == n_nodes {
                break;
            }
        }
    }

    Array1::from_vec(out)
}

// ---------------------------------------------------------------------------
// Adjacency
// ---------------------------------------------------------------------------

/// Undirected adjacency of a mesh (or any graph), in CSR layout.
///
/// Same shape as `dag::ChildList`, for the same reason: `Vec<Vec<u32>>` costs one heap
/// allocation per vertex, so it burns megabytes of allocator overhead on a large mesh *and*
/// scatters the neighbours across the heap, turning every edge relaxation into a pointer
/// chase. Flat vectors hold the same data contiguously, which matters because the row scan
/// is the memory-bound inner loop of both kernels below.
///
/// Neighbours of each vertex are sorted, deduplicated and free of self-loops. Dedup is not
/// optional book-keeping: building naively from faces yields exactly 2x duplicate arcs (every
/// interior edge is shared by two faces), so skipping it would double both the resident size
/// and the bytes touched per relaxation.
///
/// `weights` is `None` for the unweighted (hop-count) case, which lets the BFS kernel avoid
/// touching a weight array it would only ever read 1.0 from.
pub struct Adjacency {
    /// `offsets[v]..offsets[v + 1]` is the slice of `nbrs` holding v's neighbours.
    offsets: Vec<u32>,
    nbrs: Vec<u32>,
    /// Length of each arc, parallel to `nbrs`. `None` => unit weights.
    weights: Option<Vec<f32>>,
    /// Whether an edge was stored as one arc or two. Recorded because the searches are not the
    /// only thing that cares: re-weighting an undirected edge has to move *both* of its arcs to
    /// keep the adjacency symmetric, and only the builder knows there are two.
    directed: bool,
}

impl Adjacency {
    #[inline]
    fn n_nodes(&self) -> usize {
        self.offsets.len() - 1
    }

    #[inline]
    fn row(&self, v: u32) -> std::ops::Range<usize> {
        self.offsets[v as usize] as usize..self.offsets[v as usize + 1] as usize
    }

    /// Overwrite the weight of the edge between `u` and `v`, reporting whether it exists.
    ///
    /// An *edge*, not an arc: on an undirected graph both stored arcs have to move, or the
    /// adjacency stops being symmetric and `d(u, v)` quietly stops equalling `d(v, u)`. That
    /// rule lives here, next to the invariant it protects, rather than in whichever caller
    /// happens to re-weight next — which is also why `directed` is a field of this struct.
    ///
    /// Cannot *add* an edge: growing the CSR would mean rebuilding it, which is exactly the
    /// cost re-weighting in place exists to avoid.
    fn set_edge(&mut self, u: u32, v: u32, w: f32) -> bool {
        // `&&` short-circuits, so the reverse arc is only touched once the forward one is known
        // to exist — a missing edge cannot leave a half-applied edit behind.
        self.set_arc(u, v, w) && (self.directed || self.set_arc(v, u, w))
    }

    /// Overwrite the weight of the arc `u -> v`, reporting whether that arc exists.
    ///
    /// A binary search, not a scan: `compact` leaves every row sorted by neighbour, which is
    /// what makes re-weighting an edge O(log valence) rather than a pass over the edge list.
    /// That is the difference between an algorithm that re-weights as it goes — TEASAR zeroing
    /// each path it extracts so the next one may re-traverse it for free — costing O(path) per
    /// step and O(E) per step.
    ///
    /// Returns `false` for an arc that is not present, including every arc of an unweighted
    /// graph: there is no weight array to write into, and inventing one would silently turn a
    /// BFS graph into a Dijkstra one.
    fn set_arc(&mut self, u: u32, v: u32, w: f32) -> bool {
        let r = self.row(u);
        let Adjacency { nbrs, weights, .. } = self;
        let Some(weights) = weights.as_mut() else {
            return false;
        };
        match nbrs[r.clone()].binary_search(&v) {
            Ok(k) => {
                weights[r.start + k] = w;
                true
            }
            Err(_) => false,
        }
    }

    /// Sort each row, drop duplicates and self-loops, and compact in place.
    ///
    /// Rows are tiny (mesh valence is ~6), so `sort_unstable` on a row degenerates to an
    /// insertion sort — no hashing, no global O(E log E) sort, no second E-sized buffer.
    /// Compaction is safe in place because we only ever *remove* elements, so the write
    /// cursor never overtakes the read cursor.
    ///
    /// `keyed` rows pack (neighbour, payload) into a u64 so that one sort orders by
    /// neighbour first and payload second; see `from_edges`.
    fn compact(offsets: &mut [u32], packed: &mut Vec<u64>, n_nodes: usize) {
        let old: Vec<u32> = offsets.to_vec();
        let mut w: usize = 0;
        for u in 0..n_nodes {
            let lo = old[u] as usize;
            let hi = old[u + 1] as usize;
            debug_assert!(w <= lo);
            packed[lo..hi].sort_unstable();

            offsets[u] = w as u32;
            let mut prev = u64::MAX;
            for k in lo..hi {
                let p = packed[k];
                let v = (p >> 32) as u32;
                // Keep the first entry per neighbour. Because the row is sorted and the
                // payload sits in the low bits, "first" is the *smallest* payload — which is
                // what we want for parallel edges: the shortest one is the only one that can
                // ever be on a shortest path.
                if (prev >> 32) as u32 != v && v as usize != u {
                    packed[w] = p;
                    w += 1;
                    prev = p;
                }
            }
        }
        offsets[n_nodes] = w as u32;
        packed.truncate(w);
        packed.shrink_to_fit();
    }

    /// Build vertex adjacency from a triangle mesh.
    ///
    /// Each face `(a, b, c)` contributes the six arcs a→b, b→a, b→c, c→b, c→a, a→c.
    /// `coords` is `Some` for euclidean edge weights, `None` for hop counts.
    pub fn from_faces(
        faces: ArrayView2<u32>,
        n_nodes: usize,
        coords: Option<ArrayView2<f64>>,
    ) -> Self {
        assert_eq!(faces.ncols(), 3, "`faces` must have shape (F, 3)");
        if let Some(c) = coords.as_ref() {
            assert_eq!(
                c.shape(),
                [n_nodes, 3],
                "`coords` must have shape (n_vertices, 3)"
            );
        }
        let n_arcs = faces.nrows().saturating_mul(6);
        assert!(
            n_arcs <= u32::MAX as usize,
            "too many faces: CSR offsets are u32"
        );

        // Count: every vertex of a face gains exactly two arcs.
        let mut offsets: Vec<u32> = vec![0; n_nodes + 1];
        for face in faces.rows() {
            for &v in face {
                assert!(
                    (v as usize) < n_nodes,
                    "face references vertex {v}, but n_vertices = {n_nodes}"
                );
                offsets[v as usize + 1] += 2;
            }
        }
        for i in 0..n_nodes {
            offsets[i + 1] += offsets[i];
        }

        // Scatter. The payload is unused here (weights come from `coords` after dedup, so we
        // never compute a length we then throw away), but reusing the packed representation
        // lets us share `compact`.
        let mut packed: Vec<u64> = vec![0; offsets[n_nodes] as usize];
        let mut cursor: Vec<u32> = offsets[..n_nodes].to_vec();
        let mut put = |u: u32, v: u32| {
            let slot = &mut cursor[u as usize];
            packed[*slot as usize] = (v as u64) << 32;
            *slot += 1;
        };
        for face in faces.rows() {
            let (a, b, c) = (face[0], face[1], face[2]);
            put(a, b);
            put(b, a);
            put(b, c);
            put(c, b);
            put(c, a);
            put(a, c);
        }
        drop(cursor);

        Self::compact(&mut offsets, &mut packed, n_nodes);
        let nbrs: Vec<u32> = packed.iter().map(|&p| (p >> 32) as u32).collect();

        // Weights last, so we only pay for arcs that survived dedup.
        //
        // d(u,v) and d(v,u) are computed independently but come out bit-identical: the
        // expression squares each delta, and (a-b)^2 == (b-a)^2 exactly in IEEE. The
        // adjacency is therefore *exactly* symmetric — an asymmetric weight would silently
        // break d(s,t) == d(t,s).
        let weights = coords.map(|c| {
            let mut out: Vec<f32> = Vec::with_capacity(nbrs.len());
            for u in 0..n_nodes {
                let (ux, uy, uz) = (c[[u, 0]], c[[u, 1]], c[[u, 2]]);
                for &v in &nbrs[offsets[u] as usize..offsets[u + 1] as usize] {
                    let v = v as usize;
                    let (dx, dy, dz) = (ux - c[[v, 0]], uy - c[[v, 1]], uz - c[[v, 2]]);
                    out.push((dx * dx + dy * dy + dz * dz).sqrt() as f32);
                }
            }
            out
        });

        Adjacency {
            offsets,
            nbrs,
            weights,
            directed: false, // every face contributes both arcs of each of its edges
        }
    }

    /// Build adjacency from an explicit `(E, 2)` edge list.
    ///
    /// `directed` emits only the `u -> v` arc; otherwise both, so the graph is symmetric.
    /// `weights` is `None` for hop counts. Parallel edges collapse to the shortest; negative
    /// weights are rejected (Dijkstra has no answer for them, and the bit-ordered heap key
    /// below assumes non-negative distances).
    pub fn from_edges(
        edges: ArrayView2<u32>,
        n_nodes: usize,
        weights: Option<&ArrayView1<f32>>,
        directed: bool,
    ) -> Self {
        assert_eq!(edges.ncols(), 2, "`edges` must have shape (E, 2)");
        if let Some(w) = weights {
            assert_eq!(
                w.len(),
                edges.nrows(),
                "`weights` must have one entry per edge"
            );
        }
        let per_edge = if directed { 1 } else { 2 };
        let n_arcs = edges.nrows().saturating_mul(per_edge);
        assert!(
            n_arcs <= u32::MAX as usize,
            "too many edges: CSR offsets are u32"
        );

        let mut offsets: Vec<u32> = vec![0; n_nodes + 1];
        for e in edges.rows() {
            for (k, &v) in e.iter().enumerate() {
                assert!(
                    (v as usize) < n_nodes,
                    "edge references node {v}, but n_nodes = {n_nodes}"
                );
                // A directed edge only ever leaves its source, so only that row grows. We still
                // have to range-check the target.
                if !directed || k == 0 {
                    offsets[v as usize + 1] += 1;
                }
            }
        }
        for i in 0..n_nodes {
            offsets[i + 1] += offsets[i];
        }

        // Pack (neighbour, weight-bits) into one u64 so a single sort orders by neighbour and
        // then by weight. That works because a non-negative f32's IEEE bit pattern is
        // monotone when read as a u32 — the same fact the heap key relies on. Sorting ascending
        // therefore puts the *shortest* parallel edge first, and `compact` keeps the first.
        let mut packed: Vec<u64> = vec![0; offsets[n_nodes] as usize];
        let mut cursor: Vec<u32> = offsets[..n_nodes].to_vec();
        {
            let mut put = |u: u32, v: u32, wbits: u32| {
                let slot = &mut cursor[u as usize];
                packed[*slot as usize] = ((v as u64) << 32) | wbits as u64;
                *slot += 1;
            };
            for (i, e) in edges.rows().into_iter().enumerate() {
                let wbits = match weights {
                    Some(w) => {
                        let x = w[i];
                        assert!(
                            x >= 0.0 && x.is_finite(),
                            "edge weights must be finite and non-negative, got {x}"
                        );
                        x.to_bits()
                    }
                    None => 0,
                };
                put(e[0], e[1], wbits);
                if !directed {
                    put(e[1], e[0], wbits);
                }
            }
        }
        drop(cursor);

        Self::compact(&mut offsets, &mut packed, n_nodes);

        let nbrs: Vec<u32> = packed.iter().map(|&p| (p >> 32) as u32).collect();
        let weights = weights.map(|_| {
            packed
                .iter()
                .map(|&p| f32::from_bits(p as u32))
                .collect::<Vec<f32>>()
        });

        Adjacency {
            offsets,
            nbrs,
            weights,
            directed,
        }
    }

    /// The subgraph induced on `keep`, renumbered so new node `i` is old node `keep[i]`.
    ///
    /// Arcs with an endpoint outside `keep` are dropped and the rest carry their weights over.
    /// Rows come out sorted, exactly as `compact` leaves them, so the result is
    /// indistinguishable from an adjacency built afresh from the surviving edges — including
    /// in the order the kernels visit neighbours, which is what makes tie-breaking, and
    /// therefore every result, identical either way.
    ///
    fn induced(&self, keep: &[u32]) -> Adjacency {
        let n_old = self.n_nodes();
        let mut new_id: Vec<u32> = vec![u32::MAX; n_old];
        for (i, &v) in keep.iter().enumerate() {
            assert!(
                (v as usize) < n_old,
                "`nodes` contains node {v}, but n_nodes = {n_old}"
            );
            assert!(
                new_id[v as usize] == u32::MAX,
                "`nodes` contains node {v} more than once"
            );
            new_id[v as usize] = i as u32;
        }

        // Packed (neighbour, weight-bits) rows, as in `from_edges`, so one sort per row orders
        // by the *new* index while keeping each arc's weight welded to it.
        let mut offsets: Vec<u32> = vec![0; keep.len() + 1];
        let mut packed: Vec<u64> = Vec::new();
        for (i, &v) in keep.iter().enumerate() {
            let r = self.row(v);
            let start = packed.len();
            for (k, &n) in self.nbrs[r.clone()].iter().enumerate() {
                let m = new_id[n as usize];
                if m != u32::MAX {
                    let bits = self
                        .weights
                        .as_ref()
                        .map_or(0, |w| w[r.start + k].to_bits());
                    packed.push(((m as u64) << 32) | bits as u64);
                }
            }
            packed[start..].sort_unstable();
            offsets[i + 1] = packed.len() as u32;
        }
        // No dedup or self-loop pass: the source rows have neither, and `new_id` is injective,
        // so neither can appear here.
        let nbrs: Vec<u32> = packed.iter().map(|&p| (p >> 32) as u32).collect();
        let weights = self
            .weights
            .as_ref()
            .map(|_| packed.iter().map(|&p| f32::from_bits(p as u32)).collect());

        Adjacency {
            offsets,
            nbrs,
            weights,
            directed: self.directed,
        }
    }
}

// ---------------------------------------------------------------------------
// Search kernels
// ---------------------------------------------------------------------------

/// Min-heap entry, 8 packed bytes.
///
/// The distance is stored as its raw IEEE bit pattern. For *non-negative* floats — which ours
/// always are, since weights are lengths and we start at 0 — that bit pattern is monotone when
/// compared as a `u32`, so `Ord` on the bits *is* `Ord` on the floats, exactly, `+inf`
/// included. This is not an approximation to be tolerated: it buys a derived `Ord` (hence no
/// `partial_cmp().unwrap()` and no NaN panic path), an integer compare in the sift loop, and an
/// 8-byte POD entry that packs four to a cache line.
///
/// `dist_bits` must stay the first field — the derived `Ord` is lexicographic in declaration
/// order. Tie-breaking on `node` makes the order total, so results are reproducible across
/// runs and thread counts.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct HeapEntry {
    dist_bits: u32,
    node: u32,
}

/// "No predecessor" — the source itself, and any node the search never reached.
///
/// The two are distinguishable via the distance (0.0 vs unreachable), so one sentinel is
/// enough; it surfaces to callers as `-1`, consistent with the rest of the module rather than
/// with scipy's `-9999`.
const NO_PRED: u32 = u32::MAX;

/// Per-worker scratch. Allocated once per rayon chunk and reused across every source in it.
struct Scratch {
    /// Tentative distance per node. `INFINITY` = not reached.
    /// Invariant: all-`INFINITY` on entry to and exit from every search.
    dist: Vec<f32>,
    /// The node before each node on its shortest path back to the source, or [`NO_PRED`].
    /// Empty unless the caller asked for predecessors — an empty `Vec` does not allocate, so
    /// the distance-only drivers pay nothing for it.
    /// Invariant: all-[`NO_PRED`] on entry to and exit from every search.
    pred: Vec<u32>,
    /// Nearest source per node, for the multi-source searches that report one. Written as a
    /// node settles, from the value its predecessor already carries, so it is never read before
    /// it is written within a search and needs no reset. Empty unless a caller asked, as
    /// `pred` is.
    src: Vec<u32>,
    /// Nodes whose `dist` is finite, so the reset walks only what we actually touched.
    touched: Vec<u32>,
    heap: BinaryHeap<Reverse<HeapEntry>>,
    /// BFS ping-pong frontiers. `Vec::new()` does not allocate, so the Dijkstra path pays
    /// nothing for these and the BFS path pays nothing for `heap`.
    cur: Vec<u32>,
    next: Vec<u32>,
}

impl Scratch {
    fn new(n_nodes: usize) -> Self {
        Scratch {
            dist: vec![f32::INFINITY; n_nodes],
            pred: Vec::new(),
            src: Vec::new(),
            touched: Vec::new(),
            heap: BinaryHeap::new(),
            cur: Vec::new(),
            next: Vec::new(),
        }
    }

    /// As `new`, but with the predecessor array allocated — for the `PRED = true` kernels.
    fn with_pred(n_nodes: usize) -> Self {
        Scratch {
            pred: vec![NO_PRED; n_nodes],
            ..Scratch::new(n_nodes)
        }
    }

    /// Make room for `PRED = true` and for [`resolve_sources`](Self::resolve_sources).
    ///
    /// Idempotent, so a reused scratch pays for the two arrays once rather than per search —
    /// which is the point of holding one at all.
    fn enable_sources(&mut self, n_nodes: usize) {
        if self.pred.is_empty() {
            self.pred = vec![NO_PRED; n_nodes];
        }
        if self.src.is_empty() {
            self.src = vec![0; n_nodes];
        }
    }

    /// Label each of `nodes` with the source its shortest path came from.
    ///
    /// `nodes` must be in settle order, as a [`Collect`] visitor records it. That is what makes
    /// this one forward pass rather than a chain-walk per node: a node's predecessor always
    /// settles before the node, so `src[pred[v]]` is already correct by the time we reach `v`.
    /// A node with no predecessor is a source and is its own.
    ///
    /// `src` needs no reset for the same reason — nothing is read before it is written.
    fn resolve_sources(&mut self, nodes: &[u32]) -> Vec<u32> {
        let mut out: Vec<u32> = Vec::with_capacity(nodes.len());
        for &v in nodes {
            let p = self.pred[v as usize];
            let s = if p == NO_PRED {
                v
            } else {
                self.src[p as usize]
            };
            self.src[v as usize] = s;
            out.push(s);
        }
        out
    }

    /// Restore the all-`INFINITY` invariant.
    ///
    /// Walking `touched` is O(work actually done), which is the whole point when `limit` or an
    /// all-targets-settled exit stops the search after a handful of nodes and a blanket `fill`
    /// would cost more than the search itself. But those are *scattered* writes, and once a
    /// decent fraction of the graph has been touched a linear memset is several times faster
    /// per element. So flip over at a threshold; anywhere in 1/8..1/2 behaves the same.
    #[inline]
    fn reset(&mut self) {
        let track_pred = !self.pred.is_empty();
        if self.touched.len() * 4 >= self.dist.len() {
            self.dist.fill(f32::INFINITY);
            if track_pred {
                self.pred.fill(NO_PRED);
            }
        } else {
            for &v in &self.touched {
                self.dist[v as usize] = f32::INFINITY;
                if track_pred {
                    self.pred[v as usize] = NO_PRED;
                }
            }
        }
        self.touched.clear();
        self.heap.clear(); // retains capacity — no realloc churn on the next source
        self.cur.clear();
        self.next.clear();
    }
}

/// What a search should do with a node it has just settled.
///
/// The third state is what makes this an enum rather than the `bool` it replaced: a *wall* is
/// a node the search may look at but must not expand through. Bounded growth needs it — a node
/// whose every item some earlier fragment already claimed must not conduct, or fragments would
/// leak into each other across territory that is no longer theirs to cross.
#[derive(Copy, Clone)]
enum Visit {
    /// Relax this node's neighbours and carry on.
    Expand,
    /// Do not expand this node; carry on with whatever else is on the frontier.
    Wall,
    /// The search has its answer — return immediately.
    Stop,
}

/// The per-node callback a search kernel invokes as it settles nodes, in increasing distance
/// order.
///
/// A trait rather than a closure so the kernels monomorphise over it: [`Targets`] (stop once
/// the interesting nodes have settled) and [`Grow`] (collect items until a budget is spent)
/// each compile to their own specialised loop with the dispatch folded away — the same reason
/// `PRED` is a const parameter rather than a flag.
trait Visitor {
    /// Note that `node` has settled at distance `d`, and say how the search should proceed.
    fn settle(&mut self, node: u32, d: f32) -> Visit;
}

/// Which targets a search is waiting on, and what it learned when it settled them.
///
/// Shared by the matrix, nearest and farthest drivers, because all three want the same thing
/// (stop as early as the question allows) and differ only in what they keep.
struct Targets<'a> {
    /// `None` => every node is a target, so there is nothing to exit early from.
    mask: Option<&'a [bool]>,
    /// Unique targets that must settle before the search can stop.
    remaining: u32,
    /// A node that does not count as a target — the source itself, for nearest/farthest,
    /// which are defined against *distinct* targets. `u32::MAX` = none.
    exclude: u32,
    /// Stop as soon as the first target settles (nearest).
    stop_at_first: bool,
    /// First target settled, i.e. the nearest. Dijkstra settles in increasing distance order.
    first: Option<(u32, f32)>,
    /// Last target settled, i.e. the farthest — free, for the same reason.
    last: Option<(u32, f32)>,
}

/// Targets never wall: a target is something to *find*, not something to route around, so
/// every settled node is expanded until the search is done.
impl Visitor for Targets<'_> {
    #[inline]
    fn settle(&mut self, node: u32, d: f32) -> Visit {
        let is_target = match self.mask {
            Some(m) => m[node as usize],
            None => true,
        };
        if !is_target || node == self.exclude {
            return Visit::Expand;
        }
        if self.first.is_none() {
            self.first = Some((node, d));
            if self.stop_at_first {
                return Visit::Stop;
            }
        }
        self.last = Some((node, d));
        self.remaining = self.remaining.saturating_sub(1);
        if self.remaining == 0 {
            Visit::Stop
        } else {
            Visit::Expand
        }
    }
}

/// Dijkstra's relaxation loop: pop, settle, relax, repeat until the heap runs dry.
///
/// Takes its arrays loose rather than as a [`Scratch`] because callers need it under different
/// framing — [`search_from_many`] relaxes into a scratch, the farthest-point fold into its own
/// persistent field — and a second copy of Dijkstra is not something to keep in step by hand.
/// Everything that is easy to get wrong, from the stale-entry test to tie-breaking, is written
/// once.
///
/// Two things `scipy.sparse.csgraph.dijkstra` structurally cannot do, both here: stop once every
/// target has settled (scipy has no notion of targets — it materialises all N columns and lets
/// you slice afterwards), and prune at *relaxation* on `limit` so the heap never grows past the
/// ball of radius `limit`.
///
/// `PRED` additionally records each node's predecessor in `pred`. It is a const parameter rather
/// than a flag so the branch folds away entirely in the distance-only instantiation, which is
/// the one most drivers use.
///
/// `vis` decides, per settled node, whether to expand it, wall it off or stop — see [`Visitor`].
///
/// `dist` is neither reset nor assumed empty. A *warm* array whose entries are already valid
/// distances turns the ordinary `nd < *slot` relaxation test into a prune, which is exactly
/// what makes the incremental farthest-point fold in [`GeodesicGraph::farthest_seed`] cheap;
/// see its docs for why warm-starting is sound there and not in general.
///
/// `reached` collects every node whose distance goes finite for the first time — the nodes a
/// caller must later reset, and equally the nodes that have just become reachable.
fn dijkstra_drain<const PRED: bool, V: Visitor>(
    adj: &Adjacency,
    dist: &mut [f32],
    pred: &mut [u32],
    reached: &mut Vec<u32>,
    heap: &mut BinaryHeap<Reverse<HeapEntry>>,
    limit: f32,
    vis: &mut V,
) {
    let weights = adj
        .weights
        .as_deref()
        .expect("dijkstra_drain requires a weighted adjacency");

    while let Some(Reverse(HeapEntry { dist_bits, node: u })) = heap.pop() {
        // Stale entry: `u` was relaxed again after this was pushed and has already settled at a
        // smaller distance. `dist[u]` only ever decreases and we only push on a *strict*
        // improvement, so no two live entries for `u` can carry the same bits — "bits still
        // match" is exactly "this is the live entry". Each node therefore settles exactly once
        // and no separate `settled` bitmap is needed.
        if dist_bits != dist[u as usize].to_bits() {
            continue;
        }
        let du = f32::from_bits(dist_bits);

        match vis.settle(u, du) {
            Visit::Stop => return,
            // Settled, but not conducting: leave its neighbours alone and pop the next node.
            // The node keeps its distance and stays in `touched`, so the reset still finds it.
            Visit::Wall => continue,
            Visit::Expand => {}
        }

        let r = adj.row(u);
        for (&v, &w) in adj.nbrs[r.clone()].iter().zip(&weights[r]) {
            // Accumulating in f32 keeps Dijkstra's invariant: w >= 0 and round-to-nearest gives
            // fl(du + w) >= du, so the key never moves backwards.
            let nd = du + w;
            if nd > limit {
                continue; // prune here, not at pop — this is where the memory win lives
            }
            let slot = &mut dist[v as usize];
            if nd < *slot {
                if slot.is_infinite() {
                    reached.push(v);
                }
                *slot = nd;
                // Only on a *strict* improvement, never on a tie. Equal-length paths are
                // therefore resolved towards the predecessor that settled first, which is
                // deterministic (the heap orders on `(distance, node)`, a total order) but is
                // deliberately not "lowest predecessor index": rewriting `pred` on a tie would
                // be unsound with zero-weight edges, which are explicitly allowed here. Two
                // nodes joined by a zero-weight edge have equal distance, so each is a valid
                // predecessor of the other, and a tie-rewrite could point them at each other —
                // a 2-cycle that hangs anything walking the chain.
                if PRED {
                    pred[v as usize] = u;
                }
                heap.push(Reverse(HeapEntry {
                    dist_bits: nd.to_bits(),
                    node: v,
                }));
            }
        }
    }
}

/// BFS's frontier loop — [`dijkstra_drain`]'s unweighted (hop-count) twin, loose arrays and all,
/// and split out for the same reason. `cur` arrives holding the level-0 frontier, however the
/// caller chose to seed it.
///
/// Unit weights make the frontier monotone by construction, so this needs no priority queue at
/// all: two ping-pong frontiers give O(V + E) with no sift, no stale entries and no float
/// compares. Routing the unweighted case through `dijkstra_drain` would be several times slower
/// for no reason. Hop counts are integers and exact in f32 up to 2^24; no mesh has a 16M-hop
/// path.
///
/// `PRED` as there. A node is claimed by whichever frontier member reaches it first, so ties
/// within a level resolve in frontier order — deterministic, and acyclic for free because `dist`
/// strictly increases along the chain.
///
/// As there, `dist` may be warm: the guard is `level < *slot` rather than "unvisited", which on
/// a cold array (everything `INFINITY`) is the same test and on a warm one keeps only genuine
/// improvements. `reached` collects the nodes that go finite for the first time.
#[allow(clippy::too_many_arguments)]
fn bfs_drain<const PRED: bool, V: Visitor>(
    adj: &Adjacency,
    dist: &mut [f32],
    pred: &mut [u32],
    reached: &mut Vec<u32>,
    cur: &mut Vec<u32>,
    next: &mut Vec<u32>,
    limit: f32,
    vis: &mut V,
) {
    // `level` is the depth we are about to emit, so guarding *before* the increment keeps a
    // node at distance exactly `limit` and drops one at `limit + 1` — the same inclusive
    // boundary `dijkstra_drain` has, and the same one scipy has.
    let mut level: f32 = 0.0;
    while !cur.is_empty() && level < limit {
        level += 1.0;
        for &u in cur.iter() {
            let r = adj.row(u);
            for &v in &adj.nbrs[r] {
                let slot = &mut dist[v as usize];
                if level < *slot {
                    if slot.is_infinite() {
                        reached.push(v);
                    }
                    *slot = level;
                    if PRED {
                        pred[v as usize] = u;
                    }
                    match vis.settle(v, level) {
                        Visit::Stop => return,
                        Visit::Wall => {}
                        Visit::Expand => next.push(v),
                    }
                }
            }
        }
        std::mem::swap(cur, next);
        next.clear();
    }
}

/// Observes nothing and never stops — for searches whose entire output is the distance array
/// they leave behind. Monomorphisation erases it completely.
struct NoVisitor;

impl Visitor for NoVisitor {
    #[inline]
    fn settle(&mut self, _node: u32, _d: f32) -> Visit {
        Visit::Expand
    }
}

/// Records every node the search settles, in settle order — i.e. by increasing distance.
///
/// The order is not incidental. It is what lets [`GeodesicGraph::ball`] resolve each node's
/// *nearest source* in one forward pass over the output: a node's predecessor always settles
/// before the node itself, so by the time we reach a node its predecessor's source is already
/// known. Scanning `pred` chains per node instead would be O(depth) apiece.
struct Collect<'a> {
    nodes: &'a mut Vec<u32>,
    dists: &'a mut Vec<f32>,
}

impl Visitor for Collect<'_> {
    #[inline]
    fn settle(&mut self, node: u32, d: f32) -> Visit {
        self.nodes.push(node);
        self.dists.push(d);
        Visit::Expand
    }
}

/// Search outwards from one source — the single-source case of [`search_from_many`].
#[inline]
fn search_from<const PRED: bool, V: Visitor>(
    adj: &Adjacency,
    source: u32,
    limit: f32,
    vis: &mut V,
    scratch: &mut Scratch,
) {
    search_from_many::<PRED, V>(adj, std::slice::from_ref(&source), limit, vis, scratch);
}

/// Seed a search from a set of sources, giving each node its distance to the *nearest* of them.
///
/// Not the same thing as a search per source: one frontier holding all of them settles each
/// node once, at its distance to whichever source is closest, for the cost of a single search.
/// That is the query `scipy.sparse.csgraph.dijkstra(..., min_only=True)` answers, and the
/// reason it is worth having is that a great many "how far is everything from this *set*"
/// questions — a region and its surroundings, an invalidation radius around a path — are
/// phrased against a set that is far cheaper to sweep from than to iterate over.
///
/// Sources may repeat; the duplicates are dropped rather than seeding a second frontier entry.
/// That guard is also what lets [`search_from`] be this function with a one-element slice rather
/// than a second pair of kernels: seeding is the only thing a single-source search does
/// differently, and it does it in the degenerate case of this loop.
///
/// `scratch` must arrive clean, and the caller resets it afterwards — including when `vis` stops
/// the search early and leaves the frontier holding entries.
fn search_from_many<const PRED: bool, V: Visitor>(
    adj: &Adjacency,
    sources: &[u32],
    limit: f32,
    vis: &mut V,
    scratch: &mut Scratch,
) {
    let weighted = adj.weights.is_some();
    let Scratch {
        dist,
        pred,
        touched,
        heap,
        cur,
        next,
        ..
    } = scratch;

    for &s in sources {
        let slot = &mut dist[s as usize];
        if *slot == 0.0 {
            continue; // a repeated source; it is already on the frontier
        }
        debug_assert!(slot.is_infinite(), "scratch was not clean");
        *slot = 0.0;
        touched.push(s);
        if weighted {
            heap.push(Reverse(HeapEntry {
                dist_bits: 0,
                node: s,
            }));
        } else {
            // The two kernels differ in who settles the level-0 frontier: Dijkstra settles on
            // pop, so the sources go through `vis` inside the drain, whereas `bfs_drain` only
            // ever settles nodes it *discovers*. So settle them here — and note the frontier is
            // seeded *after* the visit, which is what gives `Wall` its meaning: a walled source
            // never enters `cur`, so it does not conduct.
            match vis.settle(s, 0.0) {
                Visit::Stop => return,
                Visit::Wall => {}
                Visit::Expand => cur.push(s),
            }
        }
    }

    if weighted {
        dijkstra_drain::<PRED, V>(adj, dist, pred, touched, heap, limit, vis);
    } else {
        bfs_drain::<PRED, V>(adj, dist, pred, touched, cur, next, limit, vis);
    }
}

// ---------------------------------------------------------------------------
// Drivers
// ---------------------------------------------------------------------------

/// Resolve an optional index subset to a slice, defaulting to "all nodes in index order".
fn resolve<'a>(subset: Option<&'a [u32]>, all: &'a [u32], n_nodes: usize, what: &str) -> &'a [u32] {
    let s = subset.unwrap_or(all);
    for &i in s {
        assert!(
            (i as usize) < n_nodes,
            "`{what}` contains node {i}, but n_nodes = {n_nodes}"
        );
    }
    s
}

/// Build the target mask + unique count. `None` mask means "every node is a target", which
/// makes the early exit meaningless — so we skip allocating an N-sized array we would never
/// consult.
fn target_mask(targets: &[u32], n_nodes: usize) -> (Option<Vec<bool>>, u32) {
    let identity =
        targets.len() == n_nodes && targets.iter().enumerate().all(|(i, &t)| i as u32 == t);
    if identity {
        return (None, n_nodes as u32);
    }
    let mut mask = vec![false; n_nodes];
    let mut n = 0u32;
    for &t in targets {
        if !mask[t as usize] {
            mask[t as usize] = true;
            n += 1; // count *unique* targets, or a duplicated id would stall the early exit
        }
    }
    (Some(mask), n)
}

/// Pairwise distances between `sources` and `targets` over a prebuilt adjacency.
fn geodesic_matrix_impl(
    adj: &Adjacency,
    sources: Option<&[u32]>,
    targets: Option<&[u32]>,
    limit: Option<f32>,
    threads: Option<usize>,
) -> Array2<f32> {
    let n_nodes = adj.n_nodes();
    let all: Vec<u32> = if sources.is_none() || targets.is_none() {
        (0..n_nodes as u32).collect()
    } else {
        Vec::new()
    };
    let sources = resolve(sources, &all, n_nodes, "sources");
    let targets = resolve(targets, &all, n_nodes, "targets");

    let (n_rows, n_cols) = (sources.len(), targets.len());
    if n_rows == 0 || n_cols == 0 {
        return Array2::zeros((n_rows, n_cols)); // par_chunks_mut(0) would panic
    }

    let (mask, n_targets) = target_mask(targets, n_nodes);
    let limit = limit.unwrap_or(f32::INFINITY);

    // -1 is the crate's unreachable sentinel (navis maps it to np.inf on receipt). The gather
    // below writes every cell, so the prefill is defence-in-depth — but a memset is noise next
    // to S searches, and a missed cell surfacing as a plausible 0.0 instead of an obvious -1 is
    // not a trade worth making.
    let mut flat: Vec<f32> = vec![-1.0; n_rows * n_cols];

    with_pool(threads, || {
        // One chunk per worker, one set of scratch buffers per chunk.
        //
        // Same trap as `dag::geodesic_pairs` (dag.rs:1418): `par_iter().map_init(..)` looks
        // right, but rayon calls the initialiser once per *work-split*, not once per thread, so
        // it quietly keeps far more N-sized `dist` buffers alive than there are threads
        // (measured 45 MB vs 17 MB at N=200k over there). Chunking explicitly bounds the live
        // buffers to the thread count.
        let n_chunks = rayon::current_num_threads().max(1);
        let chunk = n_rows.div_ceil(n_chunks).max(1);

        flat.par_chunks_mut(chunk * n_cols)
            .zip(sources.par_chunks(chunk))
            .for_each(|(block, srcs)| {
                let mut scratch = Scratch::new(n_nodes);

                for (row, &s) in block.chunks_mut(n_cols).zip(srcs) {
                    let mut tgt = Targets {
                        mask: mask.as_deref(),
                        remaining: n_targets,
                        exclude: u32::MAX,
                        stop_at_first: false,
                        first: None,
                        last: None,
                    };
                    search_from::<false, _>(adj, s, limit, &mut tgt, &mut scratch);

                    // Gather at the end rather than writing cells as targets settle: this
                    // preserves the caller's `targets` order exactly and handles duplicate
                    // target ids for free. It is O(n_cols), which we pay regardless — every
                    // output cell has to be written.
                    match mask {
                        None => {
                            for (cell, &d) in row.iter_mut().zip(scratch.dist.iter()) {
                                *cell = if d.is_finite() { d } else { -1.0 };
                            }
                        }
                        Some(_) => {
                            for (cell, &t) in row.iter_mut().zip(targets) {
                                let d = scratch.dist[t as usize];
                                *cell = if d.is_finite() { d } else { -1.0 };
                            }
                        }
                    }

                    scratch.reset();
                }
            });
    });

    // `from_shape_vec` takes the Vec by value — the reshape is a move, not a copy.
    Array2::from_shape_vec((n_rows, n_cols), flat)
        .expect("shape is n_rows x n_cols by construction")
}

/// Distance to the nearest (or farthest) target, for each source.
fn geodesic_extreme_impl(
    adj: &Adjacency,
    sources: Option<&[u32]>,
    targets: Option<&[u32]>,
    limit: Option<f32>,
    threads: Option<usize>,
    farthest: bool,
) -> (Array1<f32>, Array1<i32>) {
    let n_nodes = adj.n_nodes();
    let all: Vec<u32> = if sources.is_none() || targets.is_none() {
        (0..n_nodes as u32).collect()
    } else {
        Vec::new()
    };
    let sources = resolve(sources, &all, n_nodes, "sources");
    let targets = resolve(targets, &all, n_nodes, "targets");

    let n_rows = sources.len();
    if n_rows == 0 {
        return (Array1::zeros(0), Array1::zeros(0));
    }

    let (mask, n_targets) = target_mask(targets, n_nodes);
    let limit = limit.unwrap_or(f32::INFINITY);

    let mut dists: Vec<f32> = vec![-1.0; n_rows];
    let mut nodes: Vec<i32> = vec![-1; n_rows];

    with_pool(threads, || {
        let n_chunks = rayon::current_num_threads().max(1);
        let chunk = n_rows.div_ceil(n_chunks).max(1);

        dists
            .par_chunks_mut(chunk)
            .zip(nodes.par_chunks_mut(chunk))
            .zip(sources.par_chunks(chunk))
            .for_each(|((dblock, nblock), srcs)| {
                let mut scratch = Scratch::new(n_nodes);

                for ((dcell, ncell), &s) in dblock.iter_mut().zip(nblock.iter_mut()).zip(srcs) {
                    // A source that is itself a target is matched to the nearest/farthest
                    // *other* target, never to itself — matching `dag::geodesic_nearest`.
                    let self_is_target = match mask.as_deref() {
                        Some(m) => m[s as usize],
                        None => true,
                    };
                    let remaining = n_targets.saturating_sub(self_is_target as u32);
                    if remaining == 0 {
                        continue; // no distinct target exists; leave -1 / -1
                    }

                    let mut tgt = Targets {
                        mask: mask.as_deref(),
                        remaining,
                        exclude: s,
                        // Nearest can stop at the first target it settles, because Dijkstra and
                        // BFS both settle in increasing distance order. Farthest cannot — it
                        // has to settle them all, and then the *last* one settled is the
                        // answer, for free.
                        stop_at_first: !farthest,
                        first: None,
                        last: None,
                    };
                    search_from::<false, _>(adj, s, limit, &mut tgt, &mut scratch);

                    if let Some((node, d)) = if farthest { tgt.last } else { tgt.first } {
                        *dcell = d;
                        *ncell = node as i32;
                    }

                    scratch.reset();
                }
            });
    });

    (Array1::from_vec(dists), Array1::from_vec(nodes))
}

/// Full shortest-path *trees*: distances and predecessors, one row per source.
///
/// There is no `targets` argument, unlike the matrix driver. A predecessor row is only useful
/// if it is complete — walking it from a target steps through nodes the caller never asked
/// for — so restricting the columns would not save the search any work, only make the result
/// unusable.
fn geodesic_predecessors_impl(
    adj: &Adjacency,
    sources: Option<&[u32]>,
    limit: Option<f32>,
    threads: Option<usize>,
) -> (Array2<f32>, Array2<i32>) {
    let n_nodes = adj.n_nodes();
    let all: Vec<u32> = if sources.is_none() {
        (0..n_nodes as u32).collect()
    } else {
        Vec::new()
    };
    let sources = resolve(sources, &all, n_nodes, "sources");

    let n_rows = sources.len();
    if n_rows == 0 || n_nodes == 0 {
        return (
            Array2::zeros((n_rows, n_nodes)),
            Array2::zeros((n_rows, n_nodes)),
        );
    }

    let limit = limit.unwrap_or(f32::INFINITY);
    let mut dflat: Vec<f32> = vec![-1.0; n_rows * n_nodes];
    let mut pflat: Vec<i32> = vec![-1; n_rows * n_nodes];

    with_pool(threads, || {
        let n_chunks = rayon::current_num_threads().max(1);
        let chunk = n_rows.div_ceil(n_chunks).max(1);

        dflat
            .par_chunks_mut(chunk * n_nodes)
            .zip(pflat.par_chunks_mut(chunk * n_nodes))
            .zip(sources.par_chunks(chunk))
            .for_each(|((dblock, pblock), srcs)| {
                let mut scratch = Scratch::with_pred(n_nodes);

                for ((drow, prow), &s) in dblock
                    .chunks_mut(n_nodes)
                    .zip(pblock.chunks_mut(n_nodes))
                    .zip(srcs)
                {
                    let mut tgt = Targets {
                        mask: None,
                        remaining: n_nodes as u32,
                        exclude: u32::MAX,
                        stop_at_first: false,
                        first: None,
                        last: None,
                    };
                    search_from::<true, _>(adj, s, limit, &mut tgt, &mut scratch);

                    for (cell, &d) in drow.iter_mut().zip(scratch.dist.iter()) {
                        *cell = if d.is_finite() { d } else { -1.0 };
                    }
                    for (cell, &p) in prow.iter_mut().zip(scratch.pred.iter()) {
                        *cell = if p == NO_PRED { -1 } else { p as i32 };
                    }

                    scratch.reset();
                }
            });
    });

    (
        Array2::from_shape_vec((n_rows, n_nodes), dflat)
            .expect("shape is n_rows x n_nodes by construction"),
        Array2::from_shape_vec((n_rows, n_nodes), pflat)
            .expect("shape is n_rows x n_nodes by construction"),
    )
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Pairwise geodesic ("along-the-mesh-edge") distances on a triangle mesh.
///
/// Note this is the *edge-path* distance, not the exact surface geodesic: paths are constrained
/// to run along mesh edges, so on a coarse mesh they overshoot the true surface distance. This
/// is the same approximation navis makes today.
///
/// Arguments
/// ---------
/// - `faces`: (F, 3) array of triangular faces given as vertex indices.
/// - `n_vertices`: Total number of vertices. May exceed `faces.max() + 1`; isolated
///   vertices simply reach nothing.
/// - `coords`: (n_vertices, 3) vertex positions. `Some` => edges are weighted by their
///   euclidean length; `None` => unit weights (hop count).
/// - `sources`: Source vertex indices. `None` => all vertices, in index order.
/// - `targets`: Target vertex indices. `None` => all vertices, in index order. Order is
///   preserved and duplicates are allowed.
/// - `limit`: Prune the search at this distance. Vertices at exactly `limit` are kept,
///   matching `scipy.sparse.csgraph.dijkstra`.
/// - `threads`: Size of the rayon pool. `None` => the global pool.
///
/// Returns
/// -------
/// A `(sources.len(), targets.len())` f32 matrix. Unreachable pairs — disconnected, or beyond
/// `limit` — are `-1.0`.
pub fn geodesic_matrix_mesh(
    faces: ArrayView2<u32>,
    n_vertices: usize,
    coords: Option<ArrayView2<f64>>,
    sources: Option<&[u32]>,
    targets: Option<&[u32]>,
    limit: Option<f32>,
    threads: Option<usize>,
) -> Array2<f32> {
    let adj = Adjacency::from_faces(faces, n_vertices, coords);
    geodesic_matrix_impl(&adj, sources, targets, limit, threads)
}

/// Pairwise geodesic distances over an arbitrary undirected graph given as an edge list.
///
/// The general form of `geodesic_matrix_mesh`. Unlike `dag::geodesic_distances_*`, this makes no
/// tree assumption — cycles are fine.
///
/// Arguments
/// ---------
/// - `edges`: (E, 2) array of edges given as node indices.
/// - `n_nodes`: Total number of nodes.
/// - `weights`: Length of each edge. `None` => unit weights (hop count). Must be finite
///   and non-negative. Parallel edges collapse to the shortest.
/// - `directed`: If `true`, an edge `(u, v)` may only be traversed from `u` to `v`.
/// - `sources`, `targets`, `limit`, `threads`: as `geodesic_matrix_mesh`.
#[allow(clippy::too_many_arguments)]
pub fn geodesic_matrix_graph(
    edges: ArrayView2<u32>,
    n_nodes: usize,
    weights: Option<&ArrayView1<f32>>,
    directed: bool,
    sources: Option<&[u32]>,
    targets: Option<&[u32]>,
    limit: Option<f32>,
    threads: Option<usize>,
) -> Array2<f32> {
    let adj = Adjacency::from_edges(edges, n_nodes, weights, directed);
    geodesic_matrix_impl(&adj, sources, targets, limit, threads)
}

/// For each source, the distance to its nearest target and that target's index.
///
/// The memory-efficient counterpart to `geodesic_matrix_mesh`: O(sources) output instead of
/// O(sources x targets), which is the only thing that scales on a large mesh — a full V x V
/// matrix is ~107 GB at V = 164k. It is also *faster* than the matrix, because the search stops
/// at the first target it settles rather than exploring the whole component.
///
/// A source that is itself a target is matched to its nearest *distinct* target, never to
/// itself. Sources with no reachable distinct target (disconnected, or beyond `limit`) get
/// `-1.0` / `-1`. Ties break towards the lower vertex index.
pub fn geodesic_nearest_mesh(
    faces: ArrayView2<u32>,
    n_vertices: usize,
    coords: Option<ArrayView2<f64>>,
    sources: Option<&[u32]>,
    targets: Option<&[u32]>,
    limit: Option<f32>,
    threads: Option<usize>,
) -> (Array1<f32>, Array1<i32>) {
    let adj = Adjacency::from_faces(faces, n_vertices, coords);
    geodesic_extreme_impl(&adj, sources, targets, limit, threads, false)
}

/// For each source, the distance to its farthest target and that target's index.
///
/// The mirror of `geodesic_nearest_mesh`. Unlike nearest, this cannot stop early — it has to
/// settle every target — but the farthest one is then free, because both kernels settle nodes
/// in increasing distance order, so it is simply the last one settled.
///
/// Same conventions as `geodesic_nearest_mesh`: distinct targets only, `-1.0` / `-1` when none
/// is reachable.
pub fn geodesic_farthest_mesh(
    faces: ArrayView2<u32>,
    n_vertices: usize,
    coords: Option<ArrayView2<f64>>,
    sources: Option<&[u32]>,
    targets: Option<&[u32]>,
    limit: Option<f32>,
    threads: Option<usize>,
) -> (Array1<f32>, Array1<i32>) {
    let adj = Adjacency::from_faces(faces, n_vertices, coords);
    geodesic_extreme_impl(&adj, sources, targets, limit, threads, true)
}

/// Shortest-path trees over an arbitrary graph — distances *and* the route to every node.
///
/// The predecessor-returning counterpart to `geodesic_matrix_graph`. Use this when the caller
/// needs the path itself rather than its length; `geodesic_matrix_graph` when the distance is
/// enough. Algorithms that repeatedly re-weight the graph (TEASAR zeroes the edges along each
/// path it extracts, then searches again) are the motivating case, which is why this takes a
/// raw edge list — there is no index to build or invalidate between calls.
///
/// Arguments
/// ---------
/// - `edges`, `n_nodes`, `weights`, `directed`, `limit`, `threads`: as `geodesic_matrix_graph`.
///   **Zero-weight edges are explicitly allowed** — they are the mechanism a penalised-path
///   search uses to make an already-extracted route free to re-traverse.
/// - `sources`: Source nodes, one shortest-path tree each. `None` => all nodes.
///
/// Returns
/// -------
/// - distances: `(sources.len(), n_nodes)`, `-1.0` where unreachable — as
///   `geodesic_matrix_graph`.
/// - predecessors: `(sources.len(), n_nodes)`, the node before each node on its shortest path
///   back to that row's source. `-1` for the source itself and for unreachable nodes, so a
///   single `>= 0` test both walks the chain and terminates it.
///
/// Among equal-length paths the predecessor is the one that was reached first, in the kernel's
/// own deterministic order — reproducible run to run and independent of `threads`, since each
/// source is searched in isolation. It is deliberately *not* the lowest-index predecessor: see
/// `dijkstra_drain` for why rewriting on a tie is unsound once zero-weight edges are in play.
pub fn geodesic_predecessors_graph(
    edges: ArrayView2<u32>,
    n_nodes: usize,
    weights: Option<&ArrayView1<f32>>,
    directed: bool,
    sources: Option<&[u32]>,
    limit: Option<f32>,
    threads: Option<usize>,
) -> (Array2<f32>, Array2<i32>) {
    let adj = Adjacency::from_edges(edges, n_nodes, weights, directed);
    geodesic_predecessors_impl(&adj, sources, limit, threads)
}

/// Node sequences of the shortest paths from one `source` to each of `targets`.
///
/// The convenience form of `geodesic_predecessors_graph` for the common single-source case:
/// one search, then the predecessor chains walked here rather than in the caller — which is
/// exactly the per-call overhead the binding exists to remove. Because every target is known
/// up front the search also stops as soon as the last of them settles, so a short path in a
/// large graph costs a ball, not a sweep.
///
/// Returns one path per target, ordered source-first / target-last (so `path[0]` is always
/// `source` and `path.last()` the target). An unreachable target gives an empty path. A target
/// equal to `source` gives the one-element path `[source]`.
pub fn geodesic_path_graph(
    edges: ArrayView2<u32>,
    n_nodes: usize,
    weights: Option<&ArrayView1<f32>>,
    directed: bool,
    source: u32,
    targets: &[u32],
) -> Vec<Vec<u32>> {
    let adj = Adjacency::from_edges(edges, n_nodes, weights, directed);
    geodesic_path_impl(&adj, source, targets)
}

/// `geodesic_path_graph` over a prebuilt adjacency.
fn geodesic_path_impl(adj: &Adjacency, source: u32, targets: &[u32]) -> Vec<Vec<u32>> {
    let n_nodes = adj.n_nodes();
    assert!(
        (source as usize) < n_nodes,
        "`source` is node {source}, but n_nodes = {n_nodes}"
    );
    for &t in targets {
        assert!(
            (t as usize) < n_nodes,
            "`targets` contains node {t}, but n_nodes = {n_nodes}"
        );
    }
    if targets.is_empty() {
        return Vec::new();
    }

    let (mask, n_targets) = target_mask(targets, n_nodes);
    let mut scratch = Scratch::with_pred(n_nodes);
    let mut tgt = Targets {
        mask: mask.as_deref(),
        remaining: n_targets,
        exclude: u32::MAX,
        stop_at_first: false,
        first: None,
        last: None,
    };
    search_from::<true, _>(adj, source, f32::INFINITY, &mut tgt, &mut scratch);

    targets
        .iter()
        .map(|&t| {
            if !scratch.dist[t as usize].is_finite() {
                return Vec::new();
            }
            // Walk back to the source, then reverse. The chain is finite because `dist` never
            // increases along it and every step is a node that settled earlier.
            let mut path = vec![t];
            let mut cur = t;
            while cur != source {
                let p = scratch.pred[cur as usize];
                debug_assert!(p != NO_PRED, "reachable node {cur} has no predecessor");
                path.push(p);
                cur = p;
            }
            path.reverse();
            path
        })
        .collect()
}

/// Greedily partition nodes into connected clusters of bounded geodesic radius.
///
/// Repeatedly takes an unassigned node as a seed and grows a cluster outwards from it,
/// absorbing every node within `max_dist` of that seed which no earlier cluster has already
/// claimed. Useful as mesh or skeleton downsampling: collapsing each cluster to its centroid
/// gives a coarser graph whose nodes are spaced by roughly `max_dist`.
///
/// The radius is the **true geodesic distance from the seed**, not the length of the path a
/// traversal happened to take to get there. That distinction is the whole reason this is a
/// bounded Dijkstra rather than a bounded depth-first walk: a depth-first walk can reach a
/// node the long way round and reject it as too far even though it sits well inside the ball,
/// so its clusters depend on visit order rather than on geometry.
///
/// Expansion runs *through* nodes earlier clusters already claimed — ownership only decides
/// what a cluster keeps, never where it may look — so a cluster is always a ball, though not
/// necessarily a connected one once the earlier claims are removed from it.
///
/// Arguments
/// ---------
/// - `edges`: (E, 2) array of edges given as node indices.
/// - `n_nodes`: Total number of nodes. Isolated nodes each become their own cluster.
/// - `max_dist`: Maximum distance from a cluster's seed. Must be finite and non-negative;
///   `0.0` puts every node in its own cluster (up to zero-weight edges).
/// - `weights`: Length of each edge. `None` => unit weights, i.e. `max_dist` is a hop count.
/// - `seeds`: Nodes to try as seeds, in order of preference. Any node still unassigned once
///   they are exhausted becomes a seed itself, in ascending index order. `None` => seed purely
///   in ascending index order. A seed that an earlier cluster already claimed is skipped.
///
/// Returns
/// -------
/// - `labels`: cluster of each node, contiguous in `[0, n_clusters)` and numbered in the order
///   the clusters were grown. Every node is labelled — the fallback seeding guarantees it.
/// - `n_clusters`.
///
/// This is inherently sequential: cluster *n* depends on everything every earlier cluster
/// claimed, so there is no `threads` argument to give.
pub fn geodesic_clusters(
    edges: ArrayView2<u32>,
    n_nodes: usize,
    max_dist: f32,
    weights: Option<&ArrayView1<f32>>,
    seeds: Option<&[u32]>,
) -> (Vec<i32>, usize) {
    let adj = Adjacency::from_edges(edges, n_nodes, weights, false);
    geodesic_clusters_impl(&adj, max_dist, seeds)
}

/// `geodesic_clusters` over a prebuilt adjacency.
///
/// Unlike the free function this inherits the adjacency's direction: given a directed one it
/// grows out-balls, which is a different (if equally well-defined) partition.
fn geodesic_clusters_impl(
    adj: &Adjacency,
    max_dist: f32,
    seeds: Option<&[u32]>,
) -> (Vec<i32>, usize) {
    let n_nodes = adj.n_nodes();
    assert!(
        max_dist >= 0.0 && max_dist.is_finite(),
        "`max_dist` must be finite and non-negative, got {max_dist}"
    );
    if let Some(s) = seeds {
        for &v in s {
            assert!(
                (v as usize) < n_nodes,
                "`seeds` contains node {v}, but n_nodes = {n_nodes}"
            );
        }
    }
    let mut labels: Vec<i32> = vec![-1; n_nodes];
    if n_nodes == 0 {
        return (labels, 0);
    }

    let mut scratch = Scratch::new(n_nodes);
    let mut n_clusters = 0usize;

    // Preferred seeds first, then every node in index order as a fallback. Chaining rather
    // than a second loop keeps the "skip if already claimed" test in one place; the fallback
    // pass is O(n_nodes) of pure array reads for seeds that were already consumed.
    let preferred = seeds.unwrap_or(&[]).iter().copied();
    for seed in preferred.chain(0..n_nodes as u32) {
        if labels[seed as usize] >= 0 {
            continue;
        }
        let mut tgt = Targets {
            mask: None,
            remaining: n_nodes as u32,
            exclude: u32::MAX,
            stop_at_first: false,
            first: None,
            last: None,
        };
        search_from::<false, _>(adj, seed, max_dist, &mut tgt, &mut scratch);

        // `touched` is exactly the ball: a node lands there when its distance first goes
        // finite, and relaxation prunes anything past `max_dist`. So the claim is O(ball),
        // never O(n_nodes) — which is what keeps the whole loop near-linear when the clusters
        // are small.
        for &v in &scratch.touched {
            if labels[v as usize] < 0 {
                labels[v as usize] = n_clusters as i32;
            }
        }
        n_clusters += 1;
        scratch.reset();
    }

    (labels, n_clusters)
}

// ---------------------------------------------------------------------------
// Reusable graph handle
// ---------------------------------------------------------------------------

/// Why the farthest-point fold may warm-start a search, when nothing else may.
///
/// [`GeodesicGraph::farthest_seed`] keeps a running "distance to the nearest source" field and
/// folds each new batch of sources into it by driving [`dijkstra_drain`] / [`bfs_drain`]
/// directly, with that field in place of `Scratch::dist`. There is no separate kernel and no
/// trick: warm-starting the distance array with the previous field turns the ordinary
/// `nd < dist[v]` relaxation test into a *prune*. A node the new sources cannot improve is
/// never pushed, so the search costs the region those sources actually claim rather than the
/// whole graph — and the naive alternative, a fresh unpruned multi-source sweep per seed, is
/// exactly what makes the usual `scipy`-backed implementation quadratic in the seed count.
///
/// It is sound only because that particular field obeys the triangle inequality
/// `min[w] <= min[v] + d(v, w)` — it is a minimum over shortest-path distances from a source
/// set, and that property is closed under adding sources. So a node that fails to improve
/// cannot carry an improvement to anything behind it either, which is precisely the licence to
/// stop expanding it. Warm-start an array *without* that property and the result is silently
/// wrong, which is why this is documented here rather than offered as a general option.
///
/// Max-heap entry for farthest-point *selection*.
///
/// Distances only ever decrease as sources accumulate, so a stored key is an upper bound on
/// the item's current distance. That is what licenses lazy deletion: pop the largest key, and
/// if it still matches the live value it is genuinely the farthest — every other item's live
/// value is at most its own key, which is at most this one.
///
/// `dist_bits` first so the derived `Ord` orders by distance; `item` wrapped in [`Reverse`] so
/// that ties — which are everywhere on a symmetric mesh — resolve towards the *lowest* item
/// index, matching what an `argmax` scan would have returned.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct FpsEntry {
    dist_bits: u32,
    item: Reverse<u32>,
}

/// Collect the items attached to settled nodes until a budget is spent.
///
/// The counterpart to [`Targets`]: where that one asks "have I found what I was looking
/// for?", this one asks "have I gathered enough?" — the difference between a search bounded by
/// *what* it reaches and one bounded by *how much* it reaches.
struct Grow<'a> {
    /// Item CSR: `offsets[v]..offsets[v + 1]` slices `ids` to the items sitting on node `v`.
    offsets: &'a [u32],
    ids: &'a [u32],
    /// Items an earlier fragment already claimed. `None` => nothing is claimed, and therefore
    /// no walls: growth then explores freely and fragments may overlap.
    forbidden: Option<&'a [bool]>,
    /// Items collected so far, in settle order — i.e. by increasing geodesic distance from the
    /// seed, ties broken by the kernel's own deterministic order.
    out: &'a mut Vec<u32>,
    /// Each collected item's distance to the seed's node, parallel to `out`.
    ///
    /// The search knows this the moment it settles a node, so recording it is free — and it
    /// answers a question the indices alone cannot: how far out a patch actually reaches.
    /// Every item on one node necessarily shares its distance, since an item's position *is*
    /// its node's.
    dists: &'a mut Vec<f32>,
    /// How many items the caller asked for. Never exceeded: a node whose items would overshoot
    /// contributes only as many as still fit.
    size: usize,
    /// The node the search started from. Never a wall, whatever it carries — the caller asked
    /// to grow from here, so refusing to leave would be perverse.
    source: u32,
}

impl Visitor for Grow<'_> {
    #[inline]
    fn settle(&mut self, node: u32, d: f32) -> Visit {
        let r = self.offsets[node as usize] as usize..self.offsets[node as usize + 1] as usize;
        let carries_items = !r.is_empty();
        let mut took_any = false;
        for &i in &self.ids[r] {
            let claimed = self.forbidden.is_some_and(|f| f[i as usize]);
            if !claimed {
                self.out.push(i);
                self.dists.push(d);
                took_any = true;
                if self.out.len() == self.size {
                    return Visit::Stop;
                }
            }
        }
        // A node whose every item is already claimed is a *wall*: growth stops at it, which is
        // what keeps a partition's fragments disjoint *and* connected — without it, growth
        // would tunnel through a finished fragment and come out somewhere unrelated.
        //
        // A node carrying no items at all is the opposite: a pure *conduit*. That asymmetry is
        // the whole point of the item indirection — on a cloud far sparser than the mesh it
        // rides on, the empty vertices in between are exactly what keeps a patch connected.
        if carries_items && !took_any && node != self.source {
            Visit::Wall
        } else {
            Visit::Expand
        }
    }
}

/// Why a [`GeodesicGraph::set_weights`] call could not be applied as asked.
///
/// One type for all of it rather than a mix of panics and returns: every variant is reachable
/// from well-formed input — an edge list that has drifted out of step with the graph, a length
/// computed from a degenerate triangle, a graph built for hop counts — so all of them are the
/// caller's to handle, and each carries the value that offended so the caller can say which.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SetWeightsError {
    /// The graph was built with `weights = None`. There is no weight array to write into, and
    /// materialising one would quietly turn every later search from a BFS into a Dijkstra.
    Unweighted,
    /// `edges` referenced this node, which the graph does not have.
    NodeOutOfRange(u32),
    /// A weight was negative or non-finite. Dijkstra has no answer for either.
    BadWeight(f32),
    /// These two nodes exist but are not joined by an edge. Adding one would mean rebuilding
    /// the CSR, which is the cost re-weighting in place exists to avoid.
    NoSuchEdge(u32, u32),
}

impl std::fmt::Display for SetWeightsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            SetWeightsError::Unweighted => write!(
                f,
                "cannot set weights on a graph built with `weights=None`; \
                 build it with explicit weights instead"
            ),
            SetWeightsError::NodeOutOfRange(v) => {
                write!(
                    f,
                    "`edges` references node {v}, which the graph does not have"
                )
            }
            SetWeightsError::BadWeight(w) => {
                write!(f, "`weights` must be finite and non-negative, got {w}")
            }
            SetWeightsError::NoSuchEdge(u, v) => write!(f, "the graph has no edge {u} - {v}"),
        }
    }
}

impl std::error::Error for SetWeightsError {}

/// A graph prepared once for many geodesic queries.
///
/// The free functions above each build an [`Adjacency`] from a raw edge list, answer one
/// question and throw it away — right for a single all-pairs sweep, wrong for the algorithms
/// that ask thousands of *small* questions of the same graph. Tiling a neuron into fixed-size
/// fragments is the motivating case: it calls [`grow`](Self::grow) once per fragment, and a
/// fragment settles a few hundred nodes while rebuilding the adjacency costs O(E) over the
/// whole mesh. Done the free-function way, the rebuild dwarfs the work by orders of magnitude.
///
/// So this owns the adjacency, the per-search [`Scratch`] and the item index, and every query
/// method pays only for the ball it actually explores.
///
/// Most methods here are the free functions above with the rebuild taken out —
/// [`distances`](Self::distances), [`nearest`](Self::nearest), [`farthest`](Self::farthest),
/// [`predecessors`](Self::predecessors), [`path`](Self::path), [`clusters`](Self::clusters) and
/// [`components`](Self::components) each answer exactly what their counterpart does. The two
/// that have no counterpart, [`grow`](Self::grow) and [`farthest_seed`](Self::farthest_seed),
/// are the ones that only make sense against a graph you keep.
///
/// # Items
///
/// Optionally, each node may carry zero or more **items** — points of a cloud attached to the
/// graph, one entry of a resampled surface say. By default each node is its own single item
/// and the distinction vanishes entirely.
///
/// The two index spaces split cleanly by method, and the rule is short: `grow`,
/// `farthest_seed` and `item_components` count and return **items**; everything else speaks in
/// graph **nodes**, exactly as the free function it mirrors does. So growth follows the graph
/// but is measured in cloud points — which is what keeps a patch of a cloud far sparser than
/// its mesh connected, since the empty nodes in between conduct without contributing — while a
/// distance matrix stays a matrix over the graph. See [`Grow`] for how the two roles interact.
pub struct GeodesicGraph {
    adj: Adjacency,
    /// Item CSR, as consumed by [`Grow`]. Materialised even for the default one-item-per-node
    /// case so the kernel keeps a single path; it costs 8 bytes per node against an adjacency
    /// that is an order of magnitude larger.
    item_offsets: Vec<u32>,
    item_ids: Vec<u32>,
    /// The node each item sits on — the CSR's inverse, needed to turn a seed *item* into the
    /// node a search starts from.
    item_node: Vec<u32>,
    /// Reused across queries. This is the allocation the whole type exists to keep: it is
    /// O(n_nodes) and would otherwise be paid, and zeroed, once per fragment.
    scratch: Scratch,
    /// Farthest-point-sampling state: for each *node*, the distance to the nearest source
    /// folded in so far, `INFINITY` where no source reaches it. Empty until the first
    /// [`farthest_seed`](Self::farthest_seed) call — callers who only ever `grow` pay nothing.
    fps_min: Vec<f32>,
    /// Items already folded into `fps_min` as sources. Parallel to `item_node`.
    fps_seen: Vec<bool>,
    /// Selection heap: every item whose node is reachable from some source, keyed by its
    /// distance at the time it was pushed. Lazily corrected on inspection — see [`FpsEntry`].
    /// This is what keeps `farthest_seed` off an O(n_items) argmax scan per call, which
    /// otherwise dominates everything once the fold itself is pruned.
    fps_heap: BinaryHeap<FpsEntry>,
    /// Nodes that went from unreachable to reachable during the last fold, i.e. whose items
    /// need enrolling in `fps_heap`. A field rather than a local so the run does not allocate
    /// once per seed.
    fps_newly_finite: Vec<u32>,
    /// Component label per node: the smallest node index in its component, matching
    /// [`connected_components_graph`]'s convention. Empty until first needed.
    comp: Vec<u32>,
    /// The distinct component labels, ascending. Lets the fallback below reset and scan its
    /// tallies in O(components) rather than sweeping the whole node-sized label space.
    comp_labels: Vec<u32>,
    /// Reusable per-component tallies for the fallback: how many undone items a component
    /// holds, and the lowest-indexed of them. Both are indexed by label, hence node-sized, but
    /// only ever touched at the `comp_labels` positions. Kept rather than allocated per call
    /// because on a mesh with hundreds of disconnected specks the fallback runs hundreds of
    /// times.
    comp_counts: Vec<u32>,
    comp_first: Vec<u32>,
}

impl GeodesicGraph {
    /// Prepare a graph for repeated queries.
    ///
    /// Arguments
    /// ---------
    /// - `edges`: (E, 2) array of edges given as node indices. Always undirected — every query
    ///   here is about reachability *from* a seed, and a directed graph would make "connected"
    ///   mean two different things depending on which way you asked.
    /// - `n_nodes`: Total number of nodes.
    /// - `weights`: Length of each edge. `None` => unit weights, i.e. distances are hop counts
    ///   and the searches run as BFS rather than Dijkstra. Must be finite and non-negative;
    ///   parallel edges collapse to the shortest.
    /// - `directed`: If `true`, an edge `(u, v)` may only be traversed from `u` to `v`. Every
    ///   method then follows arc direction, so each becomes its "outward from here" reading:
    ///   [`grow`](Self::grow) gathers the out-reachable ball, [`farthest_seed`](Self::farthest_seed)
    ///   measures distance *from* the done set, and [`components`](Self::components) still
    ///   reports *weakly* connected components, since a search has to start somewhere.
    /// - `item_nodes`: The node each item is attached to, as an array of length `n_items`.
    ///   `None` => each node is its own single item, so items and nodes coincide.
    pub fn new(
        edges: ArrayView2<u32>,
        n_nodes: usize,
        weights: Option<&ArrayView1<f32>>,
        directed: bool,
        item_nodes: Option<&[u32]>,
    ) -> Self {
        let adj = Adjacency::from_edges(edges, n_nodes, weights, directed);
        let item_node: Vec<u32> = match item_nodes {
            Some(v) => {
                for &n in v {
                    assert!(
                        (n as usize) < n_nodes,
                        "`item_nodes` contains node {n}, but n_nodes = {n_nodes}"
                    );
                }
                v.to_vec()
            }
            None => (0..n_nodes as u32).collect(),
        };
        Self::from_parts(adj, item_node)
    }

    /// Assemble from an adjacency and an item-to-node map that are already known good.
    ///
    /// The shared tail of [`new`](Self::new) and [`subset`](Self::subset).
    fn from_parts(adj: Adjacency, item_node: Vec<u32>) -> Self {
        let n_nodes = adj.n_nodes();
        assert!(
            item_node.len() <= u32::MAX as usize,
            "too many items: the item CSR is indexed by u32"
        );

        // Counting sort into the CSR. Items stay in ascending index order within each node,
        // which is what makes `grow`'s output reproducible for a given seed and forbidden set.
        let mut item_offsets: Vec<u32> = vec![0; n_nodes + 1];
        for &v in &item_node {
            item_offsets[v as usize + 1] += 1;
        }
        for i in 0..n_nodes {
            item_offsets[i + 1] += item_offsets[i];
        }
        let mut item_ids: Vec<u32> = vec![0; item_node.len()];
        let mut cursor: Vec<u32> = item_offsets[..n_nodes].to_vec();
        for (i, &v) in item_node.iter().enumerate() {
            let slot = &mut cursor[v as usize];
            item_ids[*slot as usize] = i as u32;
            *slot += 1;
        }

        GeodesicGraph {
            scratch: Scratch::new(n_nodes),
            adj,
            item_offsets,
            item_ids,
            item_node,
            // All of these are built on first use; an empty `Vec`/`BinaryHeap` does not
            // allocate, so a caller who only ever calls `grow` pays for none of it.
            fps_min: Vec::new(),
            fps_seen: Vec::new(),
            fps_heap: BinaryHeap::new(),
            fps_newly_finite: Vec::new(),
            comp: Vec::new(),
            comp_labels: Vec::new(),
            comp_counts: Vec::new(),
            comp_first: Vec::new(),
        }
    }

    /// Number of nodes in the graph.
    pub fn n_nodes(&self) -> usize {
        self.adj.n_nodes()
    }

    /// Number of items. Equals `n_nodes` unless `item_nodes` was given.
    pub fn n_items(&self) -> usize {
        self.item_node.len()
    }

    /// The node each item sits on. The identity map unless `item_nodes` was given.
    pub fn item_nodes(&self) -> &[u32] {
        &self.item_node
    }

    /// Pairwise distances between `sources` and `targets`. See [`geodesic_matrix_graph`].
    pub fn distances(
        &self,
        sources: Option<&[u32]>,
        targets: Option<&[u32]>,
        limit: Option<f32>,
        threads: Option<usize>,
    ) -> Array2<f32> {
        geodesic_matrix_impl(&self.adj, sources, targets, limit, threads)
    }

    /// For each source, the distance to its nearest target and that target's index.
    /// See [`geodesic_nearest_mesh`].
    pub fn nearest(
        &self,
        sources: Option<&[u32]>,
        targets: Option<&[u32]>,
        limit: Option<f32>,
        threads: Option<usize>,
    ) -> (Array1<f32>, Array1<i32>) {
        geodesic_extreme_impl(&self.adj, sources, targets, limit, threads, false)
    }

    /// For each source, the distance to its farthest target and that target's index.
    /// See [`geodesic_farthest_mesh`].
    pub fn farthest(
        &self,
        sources: Option<&[u32]>,
        targets: Option<&[u32]>,
        limit: Option<f32>,
        threads: Option<usize>,
    ) -> (Array1<f32>, Array1<i32>) {
        geodesic_extreme_impl(&self.adj, sources, targets, limit, threads, true)
    }

    /// Shortest-path trees: distances *and* the route to every node.
    /// See [`geodesic_predecessors_graph`].
    pub fn predecessors(
        &self,
        sources: Option<&[u32]>,
        limit: Option<f32>,
        threads: Option<usize>,
    ) -> (Array2<f32>, Array2<i32>) {
        geodesic_predecessors_impl(&self.adj, sources, limit, threads)
    }

    /// Node sequences of the shortest paths from `source` to each of `targets`.
    /// See [`geodesic_path_graph`].
    pub fn path(&self, source: u32, targets: &[u32]) -> Vec<Vec<u32>> {
        geodesic_path_impl(&self.adj, source, targets)
    }

    /// Every node within `max_dist` of *any* source, how far it is, and which source is nearest.
    ///
    /// One multi-source search, so this costs the ball it returns rather than one sweep per
    /// source — and the output is the ball, not a node-sized array with the ball buried in it.
    /// Both halves of that matter for the calling pattern this exists for: an algorithm that
    /// walks a graph invalidating a neighbourhood at a time asks this question thousands of
    /// times against radii that cover a fraction of a percent of the graph, and
    /// `scipy.sparse.csgraph.dijkstra(..., min_only=True, limit=...)` — the closest thing
    /// elsewhere — allocates and fills three node-sized arrays per call regardless, which for
    /// a small radius costs more than the search.
    ///
    /// Arguments
    /// ---------
    /// - `sources`: Node indices. May repeat. Empty gives an empty result.
    /// - `max_dist`: Radius, inclusive, in the graph's own metric (hop counts when the graph is
    ///   unweighted). `f32::INFINITY` for no bound, which makes this "nearest source of every
    ///   reachable node".
    ///
    /// Returns `(nodes, distances, sources)`, aligned, in increasing-distance order. Every
    /// source is in `nodes` at distance 0 and is its own nearest source. Nodes farther than
    /// `max_dist`, and nodes no source reaches, are simply absent.
    ///
    /// Ties — a node equidistant from two sources — go to whichever settled first, which is
    /// deterministic but otherwise arbitrary, exactly as it is for `min_only` elsewhere.
    pub fn ball(&mut self, sources: &[u32], max_dist: f32) -> (Vec<u32>, Vec<f32>, Vec<u32>) {
        let n_nodes = self.n_nodes();
        for &s in sources {
            assert!(
                (s as usize) < n_nodes,
                "`sources` contains node {s}, but n_nodes = {n_nodes}"
            );
        }
        assert!(
            max_dist >= 0.0 && !max_dist.is_nan(),
            "`max_dist` must be non-negative, got {max_dist}"
        );
        let mut nodes: Vec<u32> = Vec::new();
        let mut dists: Vec<f32> = Vec::new();
        if sources.is_empty() || n_nodes == 0 {
            return (nodes, dists, Vec::new());
        }

        // The predecessor chain is how a node learns which source it belongs to, so unlike
        // `grow` this search needs `PRED` — and, for the same reason, `src`.
        self.scratch.enable_sources(n_nodes);

        let mut vis = Collect {
            nodes: &mut nodes,
            dists: &mut dists,
        };
        search_from_many::<true, _>(&self.adj, sources, max_dist, &mut vis, &mut self.scratch);

        let srcs = self.scratch.resolve_sources(&nodes);
        self.scratch.reset();
        (nodes, dists, srcs)
    }

    /// Re-weight edges in place, leaving the adjacency otherwise untouched.
    ///
    /// The point is to keep a graph across a run that *changes* it — TEASAR zeroing each path
    /// it extracts so later paths may re-traverse it for free is the motivating case. Rebuilding
    /// the graph after each change costs O(E) against an edit of a few hundred edges, and
    /// undoes the whole reason for holding a prepared graph; this costs O(edits log valence).
    ///
    /// Arguments
    /// ---------
    /// - `edges`: (K, 2) array of edges to re-weight, as node pairs. Each must be an edge the
    ///   graph actually has. Order does not matter and repeats are allowed — the last write
    ///   wins.
    /// - `weights`: The new weight of each, finite and non-negative.
    ///
    /// Every way a caller can get this wrong comes back as an [`SetWeightsError`] rather than a
    /// panic, because all of them are reachable with well-formed input — an edge list that has
    /// drifted out of step with the graph, a weight computed from a degenerate triangle — and
    /// the offending value is what the caller needs to see. They are all detected in the single
    /// pass over `edges` that the writes themselves need, so nothing is checked twice.
    ///
    /// Edits before the first error have been applied. A missing edge is *not* treated as a
    /// request to add one: growing the CSR would mean rebuilding it, which is the cost this
    /// method exists to avoid.
    ///
    /// # Panics
    ///
    /// If the arrays disagree in shape or length — a caller bug rather than bad data, and the
    /// same thing every other entry point here asserts on.
    ///
    /// # Note
    ///
    /// Distances change, so the incremental field behind
    /// [`farthest_seed`](Self::farthest_seed) is discarded here — a min folded under the old
    /// weights cannot be corrected under the new ones. A `farthest_seed` run interleaved with
    /// re-weighting therefore pays a cold start after each edit; one that is not, pays nothing.
    /// Component labels survive, since which edges exist has not changed.
    pub fn set_weights(
        &mut self,
        edges: ArrayView2<u32>,
        weights: ArrayView1<f32>,
    ) -> Result<(), SetWeightsError> {
        assert_eq!(edges.ncols(), 2, "`edges` must have shape (K, 2)");
        assert_eq!(
            weights.len(),
            edges.nrows(),
            "`weights` must have one entry per edge"
        );
        if self.adj.weights.is_none() {
            return Err(SetWeightsError::Unweighted);
        }

        // Distances are about to move, so the incremental field is stale whether or not every
        // edit lands. Discarding it up front keeps an early return from leaving it behind.
        self.fps_min.clear();
        self.fps_seen.clear();
        self.fps_heap.clear();
        self.fps_newly_finite.clear();

        let n_nodes = self.n_nodes() as u32;
        for (e, &w) in edges.rows().into_iter().zip(weights) {
            let (u, v) = (e[0], e[1]);
            if u >= n_nodes || v >= n_nodes {
                return Err(SetWeightsError::NodeOutOfRange(u.max(v)));
            }
            if !w.is_finite() || w < 0.0 {
                return Err(SetWeightsError::BadWeight(w));
            }
            if !self.adj.set_edge(u, v, w) {
                return Err(SetWeightsError::NoSuchEdge(u, v));
            }
        }
        Ok(())
    }

    /// Greedily partition *nodes* into connected clusters of bounded radius.
    /// See [`geodesic_clusters`], whose distance-bounded ball is the sibling of
    /// [`grow`](Self::grow)'s count-bounded one.
    pub fn clusters(&self, max_dist: f32, seeds: Option<&[u32]>) -> (Vec<i32>, usize) {
        geodesic_clusters_impl(&self.adj, max_dist, seeds)
    }

    /// Component label of each *node*: the smallest node index in its component, matching
    /// [`connected_components_graph`]. See [`item_components`](Self::item_components) for the
    /// per-item view.
    pub fn components(&mut self) -> Vec<u32> {
        self.ensure_components();
        self.comp.clone()
    }

    /// The subgraph induced on `nodes`, as a graph in its own right.
    ///
    /// New node `i` is old node `nodes[i]`, so the caller's array *is* the node map back.
    /// Edges with an endpoint outside the subset are dropped, and items land wherever their
    /// node did — an item whose node was dropped goes with it.
    ///
    /// Because the adjacency is carved out of the one already built rather than re-derived
    /// from an edge list, this costs O(V + E) of the *parent* and never returns to the caller's
    /// original input. Restricting to one connected component is the motivating case:
    ///
    /// ```ignore
    /// let comp = g.components();
    /// let biggest: Vec<u32> = (0..g.n_nodes() as u32).filter(|&v| comp[v as usize] == label).collect();
    /// let (sub, kept_items) = g.subset(&biggest);
    /// ```
    ///
    /// Returns the subgraph and the *original* index of each of its items, so results computed
    /// on the subset can be mapped back. `nodes` may be in any order but must not repeat.
    ///
    /// Items are renumbered by walking the new nodes in order and taking each one's items. The
    /// point of that order rather than the simpler "ascending original index" is that it keeps
    /// items and nodes in step: subset a graph that never had items attached and you get one
    /// where item `i` is still node `i`, whatever order `nodes` came in. Sorting by original
    /// index instead would quietly hand back a graph whose item indices are no longer its node
    /// indices, and `grow` would return numbers that look like nodes and are not.
    pub fn subset(&self, nodes: &[u32]) -> (GeodesicGraph, Vec<u32>) {
        let adj = self.adj.induced(nodes);
        let mut kept_items: Vec<u32> = Vec::new();
        let mut item_node: Vec<u32> = Vec::new();
        for (new, &v) in nodes.iter().enumerate() {
            let r =
                self.item_offsets[v as usize] as usize..self.item_offsets[v as usize + 1] as usize;
            for &i in &self.item_ids[r] {
                kept_items.push(i);
                item_node.push(new as u32);
            }
        }
        (Self::from_parts(adj, item_node), kept_items)
    }

    /// Grow a connected region of up to `size` items outwards from item `seed`.
    ///
    /// Dijkstra (or BFS, for unit weights) out from the seed's node, collecting the items on
    /// each node it settles until `size` of them are gathered. Because nodes settle in
    /// increasing distance order, the region is the geodesic *ball* around the seed that
    /// happens to hold `size` items — not whatever a depth-first walk stumbled into — and it
    /// is connected, since every settled node bar the source is reached through one that
    /// settled earlier.
    ///
    /// This is the count-bounded sibling of [`geodesic_clusters`]'s distance-bounded ball, and
    /// the reason it cannot share that implementation: a radius is known before the search
    /// starts and can be pruned at relaxation, whereas a count is only known to be met at the
    /// moment it is met, so the region has to be recorded in settle order as the search runs.
    ///
    /// Arguments
    /// ---------
    /// - `seed`: Item to grow from.
    /// - `size`: How many items to gather. Fewer come back only when the reachable region runs
    ///   out of eligible items.
    /// - `forbidden`: One flag per item, marking those an earlier fragment already claimed.
    ///   Claimed items are never collected, and a node whose items are *all* claimed becomes a
    ///   wall the growth will not cross — which is what makes repeated calls carve a graph
    ///   into disjoint *connected* fragments. `None` => nothing is claimed, so successive
    ///   calls are independent and may overlap.
    ///
    /// Returns the items seed-first, in increasing-distance order, together with each one's
    /// distance to the seed's node.
    ///
    /// The distances come free — the search settles nodes in distance order, so it holds the
    /// number already — and they are what makes a *non-uniform* patch possible: thinning a
    /// grown region by radius needs to know the radius. Items sharing a node share a distance,
    /// exactly, since an item's position is its node's.
    pub fn grow(
        &mut self,
        seed: u32,
        size: usize,
        forbidden: Option<&[bool]>,
    ) -> (Vec<u32>, Vec<f32>) {
        assert!(
            (seed as usize) < self.item_node.len(),
            "`seed` is item {seed}, but there are {} items",
            self.item_node.len()
        );
        if let Some(f) = forbidden {
            assert_eq!(
                f.len(),
                self.item_node.len(),
                "`forbidden` must have one flag per item"
            );
        }
        let mut out: Vec<u32> = Vec::new();
        let mut dists: Vec<f32> = Vec::new();
        if size == 0 {
            return (out, dists);
        }
        out.reserve(size);
        dists.reserve(size);

        let source = self.item_node[seed as usize];
        let mut vis = Grow {
            offsets: &self.item_offsets,
            ids: &self.item_ids,
            forbidden,
            out: &mut out,
            dists: &mut dists,
            size,
            source,
        };
        // No `limit`: the budget is a count, so there is no radius to prune at. The search is
        // bounded instead by `Visit::Stop` the moment the budget is spent.
        search_from::<false, _>(
            &self.adj,
            source,
            f32::INFINITY,
            &mut vis,
            &mut self.scratch,
        );
        self.scratch.reset();
        (out, dists)
    }

    /// The next farthest-point seed: the undone item geodesically farthest from everything
    /// already done.
    ///
    /// Repeatedly calling this with a growing `done` set spreads seeds evenly over the graph —
    /// farthest-point sampling — which is what you want when placing patches, landmarks or
    /// cluster centres that should not clump. Pair it with [`grow`](Self::grow): seed, grow,
    /// mark what you covered, seed again.
    ///
    /// # Reachability
    ///
    /// Only items *reachable* from something in `done` are candidates. Unreachable ones are
    /// infinitely far and would otherwise win every time, so a graph with many small
    /// components would seed every speck before returning to the main body. When the reachable
    /// frontier is exhausted — or `done` is empty — the search jumps to a fresh component,
    /// largest first, and takes its lowest-index undone item. Ties go to the lower item index.
    ///
    /// # Cost
    ///
    /// Two things keep a long run of seeds off a quadratic path. The distance field is
    /// maintained incrementally — each call folds in only the sources `done` has *gained*, and
    /// that fold is pruned against the running field (see the note on warm-starting above [`GeodesicGraph`]), so it costs
    /// the region the new sources actually claim rather than a sweep of the graph. And the
    /// winner comes off a lazily-corrected heap (see [`FpsEntry`]) rather than an `argmax`
    /// scan, which matters because once the fold is pruned the scan is what dominates.
    ///
    /// What remains per call is one pass over `done` itself, to spot the sources it gained.
    /// That is the floor for a mask-shaped API — anything cheaper would mean making the caller
    /// report what changed — and in practice it is the bulk of the cost.
    ///
    /// `done` is expected to only ever *grow* between calls. It may shrink — the field is
    /// rebuilt from scratch when that is detected, so the answer stays correct — but the
    /// incremental saving is lost for that call.
    ///
    /// Returns `None` only when every item is already done.
    pub fn farthest_seed(&mut self, done: &[bool]) -> Option<u32> {
        assert_eq!(
            done.len(),
            self.item_node.len(),
            "`done` must have one flag per item"
        );
        if done.iter().any(|&d| d) {
            self.fps_fold(done);
            if let Some(i) = self.fps_select(done) {
                return Some(i);
            }
        }
        self.largest_unset(done)
    }

    /// The farthest reachable undone item, via lazy deletion on [`fps_heap`](Self::fps_heap).
    ///
    /// Deliberately *peeks* rather than pops the winner: the answer must be a pure function of
    /// `done`, so asking twice without marking anything must give the same item twice. Only
    /// entries that are stale (re-pushed at their current distance) or spent (already done)
    /// leave the heap.
    fn fps_select(&mut self, done: &[bool]) -> Option<u32> {
        let Self {
            fps_heap,
            fps_min,
            item_node,
            ..
        } = self;
        loop {
            let mut top = fps_heap.peek_mut()?;
            let i = top.item.0;
            if done[i as usize] {
                std::collections::binary_heap::PeekMut::pop(top);
                continue; // spent: a done item is never a candidate again
            }
            let live = fps_min[item_node[i as usize] as usize].to_bits();
            if live == top.dist_bits {
                return Some(i);
            }
            // Stale: the entry was pushed before some source moved this item closer. Correct
            // it in place — a `PeekMut` that was mutably dereferenced sifts down once when it
            // drops, where pop-then-push would sift twice. This terminates because each pass
            // strictly lowers the key it rewrites, and keys are bounded below by zero.
            top.dist_bits = live;
        }
    }

    /// Component label of each item: the smallest *node* index in its component.
    ///
    /// Labels are node indices rather than a contiguous range, matching
    /// [`connected_components_graph`]. Exposed because the useful thing to do with a component
    /// is pick from it — a random seed drawn from the largest component, say, which
    /// [`farthest_seed`](Self::farthest_seed) deliberately does not do for you: it would mean
    /// owning a random number generator, and a caller who cares about reproducibility wants
    /// that to be theirs.
    pub fn item_components(&mut self) -> Vec<u32> {
        self.ensure_components();
        self.item_node
            .iter()
            .map(|&v| self.comp[v as usize])
            .collect()
    }

    /// Fold every source `done` has gained since the last call into `fps_min`.
    fn fps_fold(&mut self, done: &[bool]) {
        if self.fps_min.is_empty() {
            self.fps_min = vec![f32::INFINITY; self.adj.n_nodes()];
            self.fps_seen = vec![false; self.item_node.len()];
        }

        let Self {
            fps_min,
            fps_seen,
            fps_newly_finite,
            fps_heap,
            item_node,
            item_offsets,
            item_ids,
            scratch,
            adj,
            ..
        } = self;
        let weighted = adj.weights.is_some();
        fps_newly_finite.clear();

        // Enrol every source `done` has gained, and notice in the same pass if it has *lost*
        // one. The incremental field is only valid while `done` grows — a min cannot be
        // un-folded — so a cleared flag means the field no longer describes `done`. There is
        // nothing to repair in that case, only to discard: wipe the state and go round again,
        // which with `fps_seen` all-false cannot trip the check a second time. Skipping the
        // check altogether would return confidently wrong seeds.
        let mut any_new = false;
        'enrol: loop {
            for (i, &d) in done.iter().enumerate() {
                if fps_seen[i] {
                    if d {
                        continue;
                    }
                    fps_min.fill(f32::INFINITY);
                    fps_seen.fill(false);
                    fps_newly_finite.clear();
                    fps_heap.clear();
                    scratch.heap.clear();
                    scratch.cur.clear();
                    scratch.next.clear();
                    any_new = false;
                    continue 'enrol;
                }
                if !d {
                    continue;
                }
                fps_seen[i] = true;
                let v = item_node[i];
                // Several items can share a node, and a node may already be a source from an
                // earlier fold; either way it is already at distance zero and re-seeding it
                // would only add a redundant frontier entry.
                if fps_min[v as usize] != 0.0 {
                    if fps_min[v as usize].is_infinite() {
                        fps_newly_finite.push(v);
                    }
                    fps_min[v as usize] = 0.0;
                    if weighted {
                        scratch.heap.push(Reverse(HeapEntry {
                            dist_bits: 0,
                            node: v,
                        }));
                    } else {
                        scratch.cur.push(v);
                    }
                    any_new = true;
                }
            }
            break;
        }

        if !any_new {
            return;
        }
        // The ordinary search kernels, driven over the persistent field instead of `Scratch`'s
        // — see the note above them on why warm-starting is sound here. `PRED` is off, so the
        // empty predecessor slice is never indexed.
        let no_pred: &mut [u32] = &mut [];
        if weighted {
            dijkstra_drain::<false, _>(
                adj,
                fps_min,
                no_pred,
                fps_newly_finite,
                &mut scratch.heap,
                f32::INFINITY,
                &mut NoVisitor,
            );
        } else {
            bfs_drain::<false, _>(
                adj,
                fps_min,
                no_pred,
                fps_newly_finite,
                &mut scratch.cur,
                &mut scratch.next,
                f32::INFINITY,
                &mut NoVisitor,
            );
        }
        // Both kernels drain what they were given, so `Scratch` is left as they found it —
        // `dist` in particular is untouched, since the fold relaxes into `fps_min` instead.
        debug_assert!(scratch.heap.is_empty() && scratch.cur.is_empty());

        // Enrol the items that have just become candidates. A node contributes its items
        // exactly once, the first time anything reaches it, so the heap holds one entry per
        // reachable item over the whole run — never a per-call rebuild.
        for &v in fps_newly_finite.iter() {
            let r = item_offsets[v as usize] as usize..item_offsets[v as usize + 1] as usize;
            let bits = fps_min[v as usize].to_bits();
            for &i in &item_ids[r] {
                fps_heap.push(FpsEntry {
                    dist_bits: bits,
                    item: Reverse(i),
                });
            }
        }
    }

    /// Lowest-index undone item of the component holding the most undone items.
    ///
    /// One pass over `done` does both halves of the job: count the undone items per component
    /// and remember the first one seen in each. Because `i` ascends, "first seen" *is* the
    /// lowest index, so the winner needs no second search. Everything else is keyed on the
    /// component list rather than the node-sized label space, which matters because this runs
    /// once per component — a mesh riddled with specks would otherwise pay O(components x
    /// nodes) to place a handful of seeds.
    fn largest_unset(&mut self, done: &[bool]) -> Option<u32> {
        self.ensure_components();
        for &label in &self.comp_labels {
            self.comp_counts[label as usize] = 0;
            self.comp_first[label as usize] = u32::MAX;
        }
        for (i, &d) in done.iter().enumerate() {
            if !d {
                let label = self.comp[self.item_node[i] as usize] as usize;
                self.comp_counts[label] += 1;
                if self.comp_first[label] == u32::MAX {
                    self.comp_first[label] = i as u32;
                }
            }
        }
        // `comp_labels` ascends, and `>` keeps the first maximum, so ties go to the lowest
        // label — the same answer the old full-width scan gave.
        let mut best: Option<(u32, u32)> = None;
        for &label in &self.comp_labels {
            let c = self.comp_counts[label as usize];
            if c > 0 && best.is_none_or(|(_, b)| c > b) {
                best = Some((label, c));
            }
        }
        let (label, _) = best?; // every item done => no component has any left
        Some(self.comp_first[label as usize])
    }

    /// Label each node with the smallest node index in its component, if not done already.
    ///
    /// A plain BFS over the adjacency rather than the union-find in
    /// [`connected_components_graph`], because the adjacency is already built and walking it
    /// is cheaper than a second pass over the edge list. Starting nodes are visited in
    /// ascending order, so the first node of each component *is* its minimum index, which
    /// reproduces that function's labelling.
    fn ensure_components(&mut self) {
        if !self.comp.is_empty() {
            return;
        }
        let n = self.adj.n_nodes();
        self.comp = vec![u32::MAX; n];
        self.comp_counts = vec![0; n];
        self.comp_first = vec![u32::MAX; n];
        let mut stack: Vec<u32> = Vec::new();
        for start in 0..n as u32 {
            if self.comp[start as usize] != u32::MAX {
                continue;
            }
            // Each `start` that survives the check *is* a component label, and they arrive in
            // ascending order — so the label list falls out of the walk for free.
            self.comp_labels.push(start);
            self.comp[start as usize] = start;
            stack.push(start);
            while let Some(u) = stack.pop() {
                for &v in &self.adj.nbrs[self.adj.row(u)] {
                    if self.comp[v as usize] == u32::MAX {
                        self.comp[v as usize] = start;
                        stack.push(v);
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array2};

    /// A regular `n x n` grid of vertices, triangulated by splitting each cell along its
    /// (0,0)->(1,1) diagonal.
    ///
    /// This has a closed-form metric, which makes it an oracle with no external dependency:
    /// the diagonal edge advances both coordinates at once, so from (0,0) to (i,j)
    ///   - hop distance     = max(i, j)
    ///   - weighted distance = s * (sqrt(2) * min(i, j) + |i - j|)   at grid spacing `s`
    /// (sqrt(2) < 2, so it is always worth taking the diagonal while both coords still differ.)
    fn grid(n: usize, s: f64) -> (Array2<u32>, Array2<f64>) {
        let id = |i: usize, j: usize| (i * n + j) as u32;
        let mut faces: Vec<u32> = Vec::new();
        for i in 0..n - 1 {
            for j in 0..n - 1 {
                // Split along the (i,j)-(i+1,j+1) diagonal so that diagonal edge exists.
                faces.extend_from_slice(&[id(i, j), id(i + 1, j), id(i + 1, j + 1)]);
                faces.extend_from_slice(&[id(i, j), id(i + 1, j + 1), id(i, j + 1)]);
            }
        }
        let n_faces = faces.len() / 3;
        let faces = Array2::from_shape_vec((n_faces, 3), faces).unwrap();

        let mut coords: Vec<f64> = Vec::new();
        for i in 0..n {
            for j in 0..n {
                coords.extend_from_slice(&[i as f64 * s, j as f64 * s, 0.0]);
            }
        }
        let coords = Array2::from_shape_vec((n * n, 3), coords).unwrap();
        (faces, coords)
    }

    #[test]
    fn adjacency_dedups_and_drops_self_loops() {
        // Two triangles sharing edge 1-2.
        let faces = array![[0u32, 1, 2], [1, 2, 3]];
        let adj = Adjacency::from_faces(faces.view(), 4, None);

        // Shared edge 1-2 appears in both faces; without dedup vertex 1 would list 2 twice.
        assert_eq!(&adj.nbrs[adj.row(0)], &[1, 2]);
        assert_eq!(&adj.nbrs[adj.row(1)], &[0, 2, 3]);
        assert_eq!(&adj.nbrs[adj.row(2)], &[0, 1, 3]);
        assert_eq!(&adj.nbrs[adj.row(3)], &[1, 2]);

        // 5 undirected edges -> 10 arcs. The naive build would have produced 12.
        assert_eq!(adj.nbrs.len(), 10);
    }

    #[test]
    fn degenerate_face_produces_no_self_loop() {
        // Face (0, 0, 1) is degenerate: it would union 0 with itself.
        let faces = array![[0u32, 0, 1]];
        let adj = Adjacency::from_faces(faces.view(), 2, None);
        assert_eq!(&adj.nbrs[adj.row(0)], &[1]);
        assert_eq!(&adj.nbrs[adj.row(1)], &[0]);
    }

    #[test]
    fn arc_weights_are_exactly_symmetric() {
        // An asymmetric weight would silently break d(s,t) == d(t,s), so assert *bit*
        // equality, not approximate equality.
        let (faces, coords) = grid(6, 0.7);
        let adj = Adjacency::from_faces(faces.view(), 36, Some(coords.view()));
        let w = adj.weights.as_ref().unwrap();

        for u in 0..36u32 {
            for (k, &v) in adj.nbrs[adj.row(u)].iter().enumerate() {
                let w_uv = w[adj.row(u).start + k];
                let back = adj.nbrs[adj.row(v)].iter().position(|&x| x == u).unwrap();
                let w_vu = w[adj.row(v).start + back];
                assert_eq!(w_uv.to_bits(), w_vu.to_bits(), "arc {u}->{v} is asymmetric");
            }
        }
    }

    #[test]
    fn weighted_distances_match_the_grid_closed_form() {
        let n = 12;
        let s = 0.3f64;
        let (faces, coords) = grid(n, s);
        let d = geodesic_matrix_mesh(
            faces.view(),
            n * n,
            Some(coords.view()),
            Some(&[0]),
            None,
            None,
            None,
        );

        for i in 0..n {
            for j in 0..n {
                let expect =
                    s * (2f64.sqrt() * i.min(j) as f64 + (i as isize - j as isize).abs() as f64);
                let got = d[[0, i * n + j]];
                assert!(
                    (got as f64 - expect).abs() < 1e-4,
                    "({i},{j}): got {got}, expected {expect}"
                );
            }
        }
    }

    #[test]
    fn unweighted_distances_match_the_grid_closed_form() {
        let n = 12;
        let (faces, _) = grid(n, 1.0);
        let d = geodesic_matrix_mesh(faces.view(), n * n, None, Some(&[0]), None, None, None);

        for i in 0..n {
            for j in 0..n {
                let expect = i.max(j) as f32;
                assert_eq!(d[[0, i * n + j]], expect, "({i},{j})");
            }
        }
    }

    #[test]
    fn disconnected_components_are_minus_one() {
        // Two disjoint triangles.
        let faces = array![[0u32, 1, 2], [3, 4, 5]];
        let coords = array![
            [0.0f64, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [10.0, 0.0, 0.0],
            [11.0, 0.0, 0.0],
            [10.0, 1.0, 0.0],
        ];
        let d = geodesic_matrix_mesh(faces.view(), 6, Some(coords.view()), None, None, None, None);

        for i in 0..6 {
            for j in 0..6 {
                let same_component = (i < 3) == (j < 3);
                if same_component {
                    assert!(d[[i, j]] >= 0.0, "({i},{j}) should be reachable");
                } else {
                    assert_eq!(d[[i, j]], -1.0, "({i},{j}) should be unreachable");
                }
            }
        }
    }

    #[test]
    fn isolated_vertex_reaches_only_itself() {
        // Vertex 3 is counted but appears in no face.
        let faces = array![[0u32, 1, 2]];
        let d = geodesic_matrix_mesh(faces.view(), 4, None, None, None, None, None);
        assert_eq!(d[[3, 3]], 0.0);
        for j in 0..3 {
            assert_eq!(d[[3, j]], -1.0);
            assert_eq!(d[[j, 3]], -1.0);
        }
    }

    #[test]
    fn full_matrix_is_exactly_symmetric() {
        let (faces, coords) = grid(9, 1.3);
        let d = geodesic_matrix_mesh(
            faces.view(),
            81,
            Some(coords.view()),
            None,
            None,
            None,
            None,
        );
        for i in 0..81 {
            for j in 0..81 {
                // Bit equality: the adjacency is exactly symmetric and Dijkstra is
                // deterministic, so there is no excuse for a wobble here.
                assert_eq!(
                    d[[i, j]].to_bits(),
                    d[[j, i]].to_bits(),
                    "({i},{j}) vs ({j},{i})"
                );
            }
        }
    }

    #[test]
    fn subsetting_agrees_with_slicing_the_full_matrix() {
        // The cheapest way to catch index-mapping bugs, and it needs no external oracle.
        let (faces, coords) = grid(10, 0.9);
        let full = geodesic_matrix_mesh(
            faces.view(),
            100,
            Some(coords.view()),
            None,
            None,
            None,
            None,
        );

        let sources = [7u32, 0, 93, 42];
        let targets = [11u32, 99, 3];
        let sub = geodesic_matrix_mesh(
            faces.view(),
            100,
            Some(coords.view()),
            Some(&sources),
            Some(&targets),
            None,
            None,
        );

        for (i, &s) in sources.iter().enumerate() {
            for (j, &t) in targets.iter().enumerate() {
                assert_eq!(sub[[i, j]], full[[s as usize, t as usize]], "({i},{j})");
            }
        }
    }

    #[test]
    fn duplicate_targets_do_not_stall_the_early_exit() {
        // The early exit counts *unique* targets; a duplicated id would otherwise leave
        // `remaining` permanently above zero and quietly disable the exit.
        let (faces, coords) = grid(8, 1.0);
        let targets = [5u32, 5, 5, 20];
        let d = geodesic_matrix_mesh(
            faces.view(),
            64,
            Some(coords.view()),
            Some(&[0]),
            Some(&targets),
            None,
            None,
        );
        assert_eq!(d.ncols(), 4);
        assert_eq!(d[[0, 0]], d[[0, 1]]);
        assert_eq!(d[[0, 1]], d[[0, 2]]);
        assert!(d[[0, 3]] > 0.0);
    }

    #[test]
    fn limit_boundary_is_inclusive() {
        // Match scipy: a node at distance exactly `limit` is kept, one just beyond is not.
        let (faces, coords) = grid(10, 1.0);
        let n = 100;
        let full = geodesic_matrix_mesh(
            faces.view(),
            n,
            Some(coords.view()),
            Some(&[0]),
            None,
            None,
            None,
        );

        // Vertex (0, 3) sits at distance exactly 3.0 from vertex 0 (three axis hops).
        let exact = full[[0, 3]];
        assert!((exact - 3.0).abs() < 1e-6, "fixture assumption: {exact}");

        let at = geodesic_matrix_mesh(
            faces.view(),
            n,
            Some(coords.view()),
            Some(&[0]),
            None,
            Some(exact),
            None,
        );
        assert_eq!(at[[0, 3]], exact, "distance == limit must be kept");

        let just_under = geodesic_matrix_mesh(
            faces.view(),
            n,
            Some(coords.view()),
            Some(&[0]),
            None,
            Some(exact - 1e-3),
            None,
        );
        assert_eq!(just_under[[0, 3]], -1.0, "distance > limit must be dropped");

        // Everything inside the limit must be untouched by the pruning.
        for j in 0..n {
            if full[[0, j]] >= 0.0 && full[[0, j]] <= exact {
                assert_eq!(
                    at[[0, j]],
                    full[[0, j]],
                    "vertex {j} inside the limit moved"
                );
            } else {
                assert_eq!(at[[0, j]], -1.0, "vertex {j} outside the limit survived");
            }
        }
    }

    #[test]
    fn results_do_not_depend_on_thread_count() {
        // The real race detector.
        let (faces, coords) = grid(11, 0.6);
        let reference = geodesic_matrix_mesh(
            faces.view(),
            121,
            Some(coords.view()),
            None,
            None,
            None,
            Some(1),
        );
        for n in [2usize, 3, 7, 16] {
            let got = geodesic_matrix_mesh(
                faces.view(),
                121,
                Some(coords.view()),
                None,
                None,
                None,
                Some(n),
            );
            assert_eq!(got, reference, "thread count {n} changed the result");
        }
    }

    #[test]
    fn edge_list_graph_handles_cycles_and_parallel_edges() {
        // A triangle (a cycle — which every dag.rs geodesic function would reject) plus a
        // parallel edge 0-1 that is *longer* than the direct one, so it must be discarded.
        let edges = array![[0u32, 1], [1, 2], [2, 0], [0, 1]];
        let weights = ndarray::arr1(&[1.0f32, 1.0, 5.0, 9.0]);
        let d = geodesic_matrix_graph(
            edges.view(),
            3,
            Some(&weights.view()),
            false,
            None,
            None,
            None,
            None,
        );

        assert_eq!(d[[0, 1]], 1.0); // direct edge wins over the parallel 9.0
        assert_eq!(d[[1, 2]], 1.0);
        assert_eq!(d[[0, 2]], 2.0); // 0->1->2 beats the direct 5.0
        assert_eq!(d[[2, 0]], 2.0);
    }

    #[test]
    fn directed_edges_are_one_way() {
        // A path 0 -> 1 -> 2. Undirected you can walk back; directed you cannot.
        let edges = array![[0u32, 1], [1, 2]];
        let weights = ndarray::arr1(&[1.0f32, 1.0]);

        let dir = geodesic_matrix_graph(
            edges.view(),
            3,
            Some(&weights.view()),
            true,
            None,
            None,
            None,
            None,
        );
        assert_eq!(dir[[0, 2]], 2.0);
        assert_eq!(dir[[2, 0]], -1.0, "cannot walk against a directed edge");

        let undir = geodesic_matrix_graph(
            edges.view(),
            3,
            Some(&weights.view()),
            false,
            None,
            None,
            None,
            None,
        );
        assert_eq!(undir[[2, 0]], 2.0);
    }

    #[test]
    fn nearest_and_farthest_exclude_the_source_itself() {
        let (faces, coords) = grid(7, 1.0);
        let n = 49;
        // Vertex 0 is both a source and a target: it must match a *distinct* target.
        let sources = [0u32];
        let targets = [0u32, 1, 48];

        let (dn, nn) = geodesic_nearest_mesh(
            faces.view(),
            n,
            Some(coords.view()),
            Some(&sources),
            Some(&targets),
            None,
            None,
        );
        assert_eq!(nn[0], 1, "nearest distinct target should be vertex 1");
        assert!((dn[0] - 1.0).abs() < 1e-6);

        let (df, nf) = geodesic_farthest_mesh(
            faces.view(),
            n,
            Some(coords.view()),
            Some(&sources),
            Some(&targets),
            None,
            None,
        );
        assert_eq!(nf[0], 48, "farthest target should be the opposite corner");
        assert!(df[0] > dn[0]);
    }

    #[test]
    fn nearest_agrees_with_the_full_matrix() {
        let (faces, coords) = grid(9, 1.1);
        let n = 81;
        let targets = [3u32, 40, 77];

        let full = geodesic_matrix_mesh(
            faces.view(),
            n,
            Some(coords.view()),
            None,
            Some(&targets),
            None,
            None,
        );
        let (dn, _) = geodesic_nearest_mesh(
            faces.view(),
            n,
            Some(coords.view()),
            None,
            Some(&targets),
            None,
            None,
        );

        for s in 0..n {
            // Reproduce the "distinct target" rule when reducing the matrix by hand.
            let best = targets
                .iter()
                .enumerate()
                .filter(|(_, &t)| t as usize != s)
                .map(|(j, _)| full[[s, j]])
                .filter(|&d| d >= 0.0)
                .fold(f32::INFINITY, f32::min);
            let expect = if best.is_finite() { best } else { -1.0 };
            assert!(
                (dn[s] - expect).abs() < 1e-5,
                "source {s}: nearest {} vs matrix {expect}",
                dn[s]
            );
        }
    }

    #[test]
    fn nearest_with_no_reachable_target_is_minus_one() {
        // Two disjoint triangles; the only target lives in the other component.
        let faces = array![[0u32, 1, 2], [3, 4, 5]];
        let (d, n) =
            geodesic_nearest_mesh(faces.view(), 6, None, Some(&[0, 1]), Some(&[4]), None, None);
        assert_eq!(d.to_vec(), vec![-1.0, -1.0]);
        assert_eq!(n.to_vec(), vec![-1, -1]);
    }

    #[test]
    fn unique_edges_matches_trimesh_convention() {
        // Two triangles sharing edge 1-2. The 3F edge list is
        //   face 0: (0,1) (1,2) (2,0)   -> indices 0, 1, 2
        //   face 1: (1,2) (2,3) (3,1)   -> indices 3, 4, 5
        let faces = array![[0u32, 1, 2], [1, 2, 3]];
        let (edges, index, inverse, lengths) = unique_edges(faces.view(), None, true, true, None);

        // Rows [min, max], ascending by (max, min) — trimesh's exact order.
        assert_eq!(edges, array![[0i64, 1], [0, 2], [1, 2], [1, 3], [2, 3]]);
        // First occurrence of each unique edge in the 3F list.
        assert_eq!(index.unwrap().to_vec(), vec![0i64, 2, 1, 5, 4]);
        // Slot of every 3F edge in the unique list; reshape (F, 3) gives
        // trimesh's faces_unique_edges.
        assert_eq!(inverse.unwrap().to_vec(), vec![0i64, 2, 1, 2, 4, 3]);
        assert!(lengths.is_none());

        // Fast path must agree with the full path on the edges themselves.
        let (fast, i, v, l) = unique_edges(faces.view(), None, false, false, None);
        assert_eq!(fast, edges);
        assert!(i.is_none() && v.is_none() && l.is_none());
    }

    #[test]
    fn unique_edges_lengths() {
        // Unit square split along the (1)-(2) diagonal.
        let faces = array![[0u32, 1, 2], [1, 2, 3]];
        let coords = array![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ];

        // Both paths must produce lengths, parallel to the unique edge rows
        // [0,1], [0,2], [1,2], [1,3], [2,3].
        let expect = [1.0, 1.0, 2f64.sqrt(), 1.0, 1.0];
        for (ri, rv) in [(false, false), (true, true)] {
            let (_, _, _, lengths) = unique_edges(faces.view(), Some(coords.view()), ri, rv, None);
            let lengths = lengths.unwrap();
            assert_eq!(lengths.len(), 5);
            for (got, want) in lengths.iter().zip(expect) {
                assert!((got - want).abs() < 1e-12, "{got} vs {want}");
            }
        }
    }

    #[test]
    fn unique_edges_keeps_degenerate_self_loops() {
        // trimesh does NOT filter self-loops from degenerate faces.
        let faces = array![[0u32, 0, 1]];
        let (edges, _, _, _) = unique_edges(faces.view(), None, false, false, None);
        assert_eq!(edges, array![[0i64, 0], [0, 1]]);
    }

    #[test]
    fn connected_components_graph_agrees_with_the_mesh_version() {
        // Whatever the face-based DSU says about a mesh, the edge-based one must say about
        // that mesh's unique edges — same labels, not merely the same partition.
        let faces = array![[0u32, 1, 2], [1, 2, 3], [4, 5, 6]];
        let (edges, _, _, _) = unique_edges(faces.view(), None, false, false, None);
        let edges: Array2<u32> = edges.mapv(|v| v as u32);

        let from_faces = mesh_connected_components(faces.view(), 8);
        let from_edges = connected_components_graph(edges.view(), 8);

        assert_eq!(from_edges, from_faces);
        // Labels are the minimum index of each component; vertex 7 is isolated.
        assert_eq!(from_edges, vec![0, 0, 0, 0, 4, 4, 4, 7]);
    }

    #[test]
    fn level_sets_split_a_ring_that_is_disconnected_at_the_same_level() {
        // Path 0-1-2-3-4 plus an isolated pair 5-6. Label by "distance parity" so that
        // level 0 = {0, 2, 4, 6} and level 1 = {1, 3, 5}: no two same-label nodes touch, so
        // every node must end up in its own component.
        let edges = array![[0u32, 1], [1, 2], [2, 3], [3, 4], [5, 6]];
        let labels = array![0i64, 1, 0, 1, 0, 1, 0];
        let (ids, n) = level_set_components(edges.view(), 7, labels.view());
        assert_eq!(n, 7);
        assert_eq!(ids, vec![0, 1, 2, 3, 4, 5, 6]);

        // Now label the path 0,0,0,1,1 — one level-0 run and one level-1 run — while the
        // pair 5-6 shares level 0 but is *not* adjacent to the path, so it is a third
        // component even though its label matches the first.
        let labels = array![0i64, 0, 0, 1, 1, 0, 0];
        let (ids, n) = level_set_components(edges.view(), 7, labels.view());
        assert_eq!(n, 3);
        assert_eq!(ids, vec![0, 0, 0, 1, 1, 2, 2]);
    }

    #[test]
    fn negative_labels_are_excluded_not_grouped() {
        // The unreachable marker `geodesic_matrix_*` emits is -1. Those nodes must not fuse
        // into one giant phantom level just because they share the sentinel — even when they
        // are adjacent to each other.
        let edges = array![[0u32, 1], [1, 2], [2, 3]];
        let labels = array![-1i64, -1, 5, 5];
        let (ids, n) = level_set_components(edges.view(), 4, labels.view());
        assert_eq!(n, 1);
        assert_eq!(ids, vec![-1, -1, 0, 0]);
    }

    #[test]
    fn level_set_components_match_a_per_level_subgraph_search() {
        // The oracle: do it the slow way — for each distinct label, induce the subgraph on
        // those nodes and find its components — and check the fused pass agrees.
        let n = 9;
        let (faces, _) = grid(n, 1.0);
        let n_nodes = n * n;
        let (edges, _, _, _) = unique_edges(faces.view(), None, false, false, None);
        let edges: Array2<u32> = edges.mapv(|v| v as u32);

        // Hop distance from a corner, which on this grid is max(i, j) — a genuine wavefront.
        let d = geodesic_matrix_mesh(faces.view(), n_nodes, None, Some(&[0]), None, None, None);
        let labels: Array1<i64> = d.row(0).iter().map(|&x| x as i64).collect();

        let (ids, n_comp) = level_set_components(edges.view(), n_nodes, labels.view());

        // Oracle: per label, a DSU restricted to that label's induced subgraph.
        for lvl in 0..n as i64 {
            let members: Vec<usize> = (0..n_nodes).filter(|&i| labels[i] == lvl).collect();
            let mut parent: Vec<u32> = (0..n_nodes as u32).collect();
            for e in edges.rows() {
                if labels[e[0] as usize] == lvl && labels[e[1] as usize] == lvl {
                    union(&mut parent, e[0], e[1]);
                }
            }
            for (a, b) in members.iter().zip(members.iter().skip(1)) {
                let same_oracle = find(&mut parent, *a as u32) == find(&mut parent, *b as u32);
                assert_eq!(
                    same_oracle,
                    ids[*a] == ids[*b],
                    "level {lvl}: nodes {a} and {b} disagree"
                );
            }
        }

        // Every node is reachable here, so every node is assigned, and ids are dense.
        assert!(ids.iter().all(|&x| x >= 0));
        assert_eq!(ids.iter().max().unwrap(), &(n_comp as i32 - 1));
    }

    #[test]
    fn contract_vertices_drops_self_loops_and_dedups() {
        // Square 0-1-2-3 with a diagonal. Collapse {0,1} -> 0 and {2,3} -> 1.
        let edges = array![[0u32, 1], [1, 2], [2, 3], [3, 0], [0, 2]];
        let mapping = array![0u32, 0, 1, 1];
        let out = contract_vertices(edges.view(), mapping.view(), None);

        // 0-1 and 2-3 became self-loops and vanished; the remaining three all became 0-1 and
        // collapsed to one edge.
        assert_eq!(out, array![[0i64, 1]]);
    }

    #[test]
    fn contract_vertices_identity_is_unique_edges() {
        // Contracting through the identity map must reproduce exactly the deduplicated edge
        // list — same rows, same (max, min) order.
        let faces = array![[0u32, 1, 2], [1, 2, 3]];
        let (unique, _, _, _) = unique_edges(faces.view(), None, false, false, None);
        let edges: Array2<u32> = unique.mapv(|v| v as u32);
        let mapping: Array1<u32> = (0..4u32).collect();

        let out = contract_vertices(edges.view(), mapping.view(), None);
        assert_eq!(out, unique);
    }

    #[test]
    fn mst_picks_the_cheap_edges_and_maximize_inverts_it() {
        // Triangle 0-1-2 with weights 1, 2, 3 (edges 0-1, 1-2, 0-2).
        let edges = array![[0u32, 1], [1, 2], [0, 2]];
        let w = array![1.0f32, 2.0, 3.0];

        // Minimum: take 1 and 2, reject the 3 that would close the cycle.
        let mst = minimum_spanning_tree(edges.view(), 3, Some(&w.view()), false, None);
        assert_eq!(mst.to_vec(), vec![0i64, 1]);

        // Maximum: take 3 and 2. Returned ascending by weight *as sorted*, i.e. heaviest
        // first for maximize.
        let mst = minimum_spanning_tree(edges.view(), 3, Some(&w.view()), true, None);
        assert_eq!(mst.to_vec(), vec![2i64, 1]);
    }

    #[test]
    fn mst_of_a_disconnected_graph_is_a_forest() {
        // Two triangles, no edge between them: n_nodes - n_components = 6 - 2 = 4 edges.
        let edges = array![[0u32, 1], [1, 2], [0, 2], [3, 4], [4, 5], [3, 5]];
        let w = array![1.0f32, 2.0, 3.0, 1.0, 2.0, 3.0];
        let mst = minimum_spanning_tree(edges.view(), 6, Some(&w.view()), false, None);
        assert_eq!(mst.len(), 4);
        assert_eq!(mst.to_vec(), vec![0i64, 3, 1, 4]);

        // The result must be acyclic and span both components.
        let picked: Vec<[u32; 2]> = mst
            .iter()
            .map(|&i| [edges[[i as usize, 0]], edges[[i as usize, 1]]])
            .collect();
        let flat = Array2::from_shape_vec(
            (picked.len(), 2),
            picked.iter().flat_map(|e| e.to_vec()).collect(),
        )
        .unwrap();
        let comps = connected_components_graph(flat.view(), 6);
        assert_eq!(comps, vec![0, 0, 0, 3, 3, 3]);
    }

    #[test]
    fn mst_handles_negative_weights_and_ties() {
        // Negative weights are legal for Kruskal and must not be rejected (the heap-key bit
        // trick used elsewhere in this module cannot represent them; `total_cmp` can).
        let edges = array![[0u32, 1], [1, 2], [0, 2]];
        let w = array![-5.0f32, -1.0, -3.0];
        let mst = minimum_spanning_tree(edges.view(), 3, Some(&w.view()), false, None);
        assert_eq!(mst.to_vec(), vec![0i64, 2]);

        // All-equal weights: ties break on edge index, so the first two independent edges win.
        let w = array![7.0f32, 7.0, 7.0];
        let mst = minimum_spanning_tree(edges.view(), 3, Some(&w.view()), false, None);
        assert_eq!(mst.to_vec(), vec![0i64, 1]);

        // Unweighted behaves the same as all-equal weights.
        let mst = minimum_spanning_tree(edges.view(), 3, None, false, None);
        assert_eq!(mst.to_vec(), vec![0i64, 1]);
    }

    #[test]
    fn graph_primitives_handle_empty_input() {
        let edges = Array2::<u32>::zeros((0, 2));
        assert_eq!(connected_components_graph(edges.view(), 3), vec![0, 1, 2]);

        let labels = array![0i64, 0, -1];
        let (ids, n) = level_set_components(edges.view(), 3, labels.view());
        assert_eq!((ids, n), (vec![0, 1, -1], 2));

        let mapping = array![0u32, 0, 1];
        assert_eq!(
            contract_vertices(edges.view(), mapping.view(), None).shape(),
            &[0, 2]
        );

        assert_eq!(
            minimum_spanning_tree(edges.view(), 3, None, false, None).len(),
            0
        );
    }

    #[test]
    fn unique_edges_empty_input() {
        let faces = Array2::<u32>::zeros((0, 3));
        let coords = Array2::<f64>::zeros((0, 3));
        let (edges, index, inverse, lengths) =
            unique_edges(faces.view(), Some(coords.view()), true, true, None);
        assert_eq!(edges.shape(), &[0, 2]);
        assert_eq!(index.unwrap().len(), 0);
        assert_eq!(inverse.unwrap().len(), 0);
        assert_eq!(lengths.unwrap().len(), 0);
    }

    #[test]
    fn unique_edges_handles_strided_views() {
        // A reversed-row view is not contiguous — exercises the copy fallback.
        let faces = array![[0u32, 1, 2], [1, 2, 3]];
        let flipped = faces.slice(ndarray::s![..;-1, ..]);
        let (edges, _, _, _) = unique_edges(flipped, None, false, false, None);
        let (expect, _, _, _) = unique_edges(faces.view(), None, false, false, None);
        assert_eq!(edges, expect);
    }

    // -----------------------------------------------------------------------
    // Predecessors / paths
    // -----------------------------------------------------------------------

    /// Sum the weights along a node path, looking each edge up in the edge list.
    fn path_length(edges: &Array2<u32>, w: Option<&Array1<f32>>, path: &[u32]) -> f32 {
        path.windows(2)
            .map(|s| {
                let mut best = f32::INFINITY;
                for (i, e) in edges.rows().into_iter().enumerate() {
                    if (e[0] == s[0] && e[1] == s[1]) || (e[0] == s[1] && e[1] == s[0]) {
                        best = best.min(w.map_or(1.0, |w| w[i]));
                    }
                }
                assert!(best.is_finite(), "path uses non-edge {}-{}", s[0], s[1]);
                best
            })
            .sum()
    }

    #[test]
    fn predecessor_chains_reproduce_the_distances() {
        // The self-consistency property that matters: walking the chain must land on the
        // source, and the walked path must weigh exactly what the distance matrix claims.
        let n = 7;
        let (faces, coords) = grid(n, 1.5);
        let n_nodes = n * n;
        let (unique, _, _, lengths) =
            unique_edges(faces.view(), Some(coords.view()), false, true, None);
        let edges: Array2<u32> = unique.mapv(|v| v as u32);
        let w: Array1<f32> = lengths.unwrap().mapv(|v| v as f32);

        let sources = [0u32, 13, (n_nodes - 1) as u32];
        let (dist, pred) = geodesic_predecessors_graph(
            edges.view(),
            n_nodes,
            Some(&w.view()),
            false,
            Some(&sources),
            None,
            None,
        );

        for (r, &s) in sources.iter().enumerate() {
            assert_eq!(pred[[r, s as usize]], -1, "the source has no predecessor");
            for t in 0..n_nodes {
                let mut path = vec![t as u32];
                let mut cur = t as u32;
                while cur != s {
                    let p = pred[[r, cur as usize]];
                    assert!(p >= 0, "node {t} is reachable but the chain broke at {cur}");
                    path.push(p as u32);
                    cur = p as u32;
                }
                path.reverse();
                let walked = path_length(&edges, Some(&w), &path);
                assert!(
                    (walked - dist[[r, t]]).abs() < 1e-4,
                    "source {s} -> {t}: walked {walked}, matrix says {}",
                    dist[[r, t]]
                );
            }
        }
    }

    #[test]
    fn predecessors_agree_with_the_distance_matrix() {
        // Distances must not change just because predecessors were asked for.
        let n = 6;
        let (faces, coords) = grid(n, 1.0);
        let n_nodes = n * n;
        let (unique, _, _, lengths) =
            unique_edges(faces.view(), Some(coords.view()), false, true, None);
        let edges: Array2<u32> = unique.mapv(|v| v as u32);
        let w: Array1<f32> = lengths.unwrap().mapv(|v| v as f32);

        let expect = geodesic_matrix_graph(
            edges.view(),
            n_nodes,
            Some(&w.view()),
            false,
            None,
            None,
            None,
            None,
        );
        let (dist, _) = geodesic_predecessors_graph(
            edges.view(),
            n_nodes,
            Some(&w.view()),
            false,
            None,
            None,
            None,
        );
        assert_eq!(dist, expect);
    }

    #[test]
    fn unreachable_nodes_have_no_predecessor() {
        // Two disjoint triangles, plus an isolated node.
        let edges = array![[0u32, 1], [1, 2], [0, 2], [3, 4], [4, 5], [3, 5]];
        let (dist, pred) =
            geodesic_predecessors_graph(edges.view(), 7, None, false, Some(&[0]), None, None);

        for v in 3..7 {
            assert_eq!(dist[[0, v]], -1.0);
            assert_eq!(pred[[0, v]], -1);
        }
        assert_eq!(pred[[0, 0]], -1); // the source
        assert_eq!(pred[[0, 1]], 0);
        assert_eq!(pred[[0, 2]], 0);
    }

    #[test]
    fn zero_weight_edges_are_free_to_traverse() {
        // The TEASAR mechanism: zeroing an edge must make it cost nothing and must not leave
        // a predecessor cycle behind (which would hang the walk below).
        let edges = array![[0u32, 1], [1, 2], [2, 3], [0, 3]];
        let w = array![0.0f32, 0.0, 0.0, 10.0];
        let paths = geodesic_path_graph(edges.view(), 4, Some(&w.view()), false, 0, &[3]);

        // Around the zeroed chain (cost 0), not across the direct edge (cost 10).
        assert_eq!(paths[0], vec![0, 1, 2, 3]);
    }

    #[test]
    fn paths_are_shortest_and_source_first() {
        let n = 8;
        let (faces, coords) = grid(n, 2.0);
        let n_nodes = n * n;
        let (unique, _, _, lengths) =
            unique_edges(faces.view(), Some(coords.view()), false, true, None);
        let edges: Array2<u32> = unique.mapv(|v| v as u32);
        let w: Array1<f32> = lengths.unwrap().mapv(|v| v as f32);

        let targets: Vec<u32> = (0..n_nodes as u32).collect();
        let paths = geodesic_path_graph(edges.view(), n_nodes, Some(&w.view()), false, 0, &targets);
        let dist = geodesic_matrix_graph(
            edges.view(),
            n_nodes,
            Some(&w.view()),
            false,
            Some(&[0]),
            None,
            None,
            None,
        );

        for (t, path) in paths.iter().enumerate() {
            assert_eq!(path[0], 0, "paths start at the source");
            assert_eq!(*path.last().unwrap(), t as u32, "and end at the target");
            let walked = path_length(&edges, Some(&w), path);
            assert!((walked - dist[[0, t]]).abs() < 1e-4);
        }
        assert_eq!(paths[0], vec![0], "source-to-source is a single node");
    }

    #[test]
    fn paths_to_unreachable_targets_are_empty() {
        let edges = array![[0u32, 1], [2, 3]];
        let paths = geodesic_path_graph(edges.view(), 4, None, false, 0, &[1, 3, 0]);
        assert_eq!(paths, vec![vec![0, 1], vec![], vec![0]]);
    }

    #[test]
    fn predecessors_are_stable_across_thread_counts() {
        let n = 10;
        let (faces, coords) = grid(n, 1.0);
        let n_nodes = n * n;
        let (unique, _, _, lengths) =
            unique_edges(faces.view(), Some(coords.view()), false, true, None);
        let edges: Array2<u32> = unique.mapv(|v| v as u32);
        let w: Array1<f32> = lengths.unwrap().mapv(|v| v as f32);

        let run = |t: usize| {
            geodesic_predecessors_graph(
                edges.view(),
                n_nodes,
                Some(&w.view()),
                false,
                None,
                None,
                Some(t),
            )
        };
        assert_eq!(run(1), run(4));
    }

    // -----------------------------------------------------------------------
    // Geodesic clustering
    // -----------------------------------------------------------------------

    #[test]
    fn clusters_are_balls_of_the_given_radius() {
        // A path graph 0-1-...-9 with unit weights and max_dist = 2 hops: seeding at 0 claims
        // 0..2, then 3 claims 3..5, and so on.
        let edges: Array2<u32> = Array2::from_shape_vec(
            (9, 2),
            (0..9u32).flat_map(|i| [i, i + 1]).collect::<Vec<u32>>(),
        )
        .unwrap();
        let (labels, n) = geodesic_clusters(edges.view(), 10, 2.0, None, None);
        assert_eq!(labels, vec![0, 0, 0, 1, 1, 1, 2, 2, 2, 3]);
        assert_eq!(n, 4);
    }

    #[test]
    fn clusters_use_the_true_geodesic_not_the_traversal_path() {
        // The point of the design decision. A 5-cycle, seeded at 0, radius 2 hops: node 3 is
        // 2 hops away *the short way round* (0-4-3) even though the long way (0-1-2-3) is 3.
        // A depth-first walk that took the long branch first would reject it; a bounded
        // Dijkstra cannot.
        let edges = array![[0u32, 1], [1, 2], [2, 3], [3, 4], [4, 0]];
        let (labels, n) = geodesic_clusters(edges.view(), 5, 2.0, None, None);
        assert_eq!(
            labels,
            vec![0, 0, 0, 0, 0],
            "the whole cycle is within 2 hops"
        );
        assert_eq!(n, 1);
    }

    #[test]
    fn preferred_seeds_go_first_and_used_up_seeds_are_skipped() {
        let edges: Array2<u32> = Array2::from_shape_vec(
            (5, 2),
            (0..5u32).flat_map(|i| [i, i + 1]).collect::<Vec<u32>>(),
        )
        .unwrap();
        // Seed at 3 first: it claims 2..4. Then 1 claims 0..1 (2 is taken), then 5 is left.
        // The repeated seed 3 must be skipped, not restarted as a new cluster.
        let (labels, n) = geodesic_clusters(edges.view(), 6, 1.0, None, Some(&[3, 3, 1]));
        assert_eq!(labels, vec![1, 1, 0, 0, 0, 2]);
        assert_eq!(n, 3);
    }

    #[test]
    fn clusters_respect_edge_weights_and_cover_every_node() {
        let n = 9;
        let (faces, coords) = grid(n, 1.0);
        let n_nodes = n * n;
        let (unique, _, _, lengths) =
            unique_edges(faces.view(), Some(coords.view()), false, true, None);
        let edges: Array2<u32> = unique.mapv(|v| v as u32);
        let w: Array1<f32> = lengths.unwrap().mapv(|v| v as f32);

        let max_dist = 2.5f32;
        let (labels, n_clusters) =
            geodesic_clusters(edges.view(), n_nodes, max_dist, Some(&w.view()), None);

        assert!(labels.iter().all(|&l| l >= 0), "every node is labelled");
        assert_eq!(*labels.iter().max().unwrap(), n_clusters as i32 - 1);

        // Seeds are the first node of each cluster in index order, because that is the order
        // the fallback seeding walks. Every member must be within `max_dist` of its seed.
        let mut seeds: Vec<u32> = vec![u32::MAX; n_clusters];
        for (v, &l) in labels.iter().enumerate() {
            let s = &mut seeds[l as usize];
            *s = (*s).min(v as u32);
        }
        let d = geodesic_matrix_graph(
            edges.view(),
            n_nodes,
            Some(&w.view()),
            false,
            Some(&seeds),
            None,
            None,
            None,
        );
        for (v, &l) in labels.iter().enumerate() {
            let dv = d[[l as usize, v]];
            assert!(
                dv >= 0.0 && dv <= max_dist + 1e-5,
                "node {v} is {dv} from seed {}",
                seeds[l as usize]
            );
        }
    }

    #[test]
    fn zero_radius_isolates_every_node() {
        let edges = array![[0u32, 1], [1, 2]];
        let (labels, n) = geodesic_clusters(edges.view(), 3, 0.0, None, None);
        assert_eq!((labels, n), (vec![0, 1, 2], 3));
    }

    #[test]
    fn isolated_nodes_become_their_own_cluster() {
        let edges = array![[0u32, 1]];
        let (labels, n) = geodesic_clusters(edges.view(), 4, 10.0, None, None);
        assert_eq!((labels, n), (vec![0, 0, 1, 2], 3));
    }

    // -----------------------------------------------------------------------
    // GeodesicGraph::grow
    // -----------------------------------------------------------------------

    /// A path graph `0-1-...-(n-1)`.
    fn path_graph(n: u32) -> Array2<u32> {
        Array2::from_shape_vec(
            (n as usize - 1, 2),
            (0..n - 1).flat_map(|i| [i, i + 1]).collect::<Vec<u32>>(),
        )
        .unwrap()
    }

    /// The `grid` mesh as an edge list plus euclidean edge lengths — the general-graph form of
    /// the same oracle the mesh tests use.
    fn grid_graph(n: usize, s: f64) -> (Array2<u32>, Array1<f32>) {
        let (faces, coords) = grid(n, s);
        let (unique, _, _, lengths) =
            unique_edges(faces.view(), Some(coords.view()), false, true, None);
        (
            unique.mapv(|v| v as u32),
            lengths.unwrap().mapv(|v| v as f32),
        )
    }

    /// Is `nodes` a single connected piece of `edges`? Deliberately a naive flood-fill written
    /// against the raw edge list, so it shares no machinery with the code under test.
    fn is_connected(edges: ArrayView2<u32>, nodes: &[u32]) -> bool {
        let inside: std::collections::HashSet<u32> = nodes.iter().copied().collect();
        if inside.is_empty() {
            return true;
        }
        let mut seen: std::collections::HashSet<u32> = std::collections::HashSet::new();
        let mut stack = vec![nodes[0]];
        seen.insert(nodes[0]);
        while let Some(u) = stack.pop() {
            for e in edges.rows() {
                for (a, b) in [(e[0], e[1]), (e[1], e[0])] {
                    if a == u && inside.contains(&b) && seen.insert(b) {
                        stack.push(b);
                    }
                }
            }
        }
        seen.len() == inside.len()
    }

    /// Partition every item into disjoint connected fragments, the way a tiling driver does:
    /// seed at the first unclaimed item, grow, mark, repeat. Returns the fragments.
    fn partition(g: &mut GeodesicGraph, size: usize) -> Vec<Vec<u32>> {
        let mut claimed = vec![false; g.n_items()];
        let mut out = Vec::new();
        while let Some(seed) = claimed.iter().position(|&c| !c) {
            let frag = g.grow(seed as u32, size, Some(&claimed)).0;
            assert!(
                !frag.is_empty(),
                "growth from an unclaimed seed cannot be empty"
            );
            for &i in &frag {
                claimed[i as usize] = true;
            }
            out.push(frag);
        }
        out
    }

    #[test]
    fn grow_returns_a_ball_in_increasing_distance_order() {
        // Path 0-1-...-9, seeded in the middle: growth alternates outwards from 5.
        let edges = path_graph(10);
        let mut g = GeodesicGraph::new(edges.view(), 10, None, false, None);
        assert_eq!(g.grow(5, 1, None).0, vec![5]);
        assert_eq!(g.grow(5, 3, None).0, vec![5, 4, 6]);
        assert_eq!(g.grow(5, 5, None).0, vec![5, 4, 6, 3, 7]);
        // Asking for more than exists returns what there is, never pads or repeats.
        let all = g.grow(5, 100, None).0;
        assert_eq!(all.len(), 10);
        let mut sorted = all.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..10u32).collect::<Vec<_>>());
    }

    #[test]
    fn grow_reports_each_items_distance_to_the_seed() {
        // Path 0-1-...-9 seeded in the middle: distances follow the alternating walk out.
        let edges = path_graph(10);
        let mut g = GeodesicGraph::new(edges.view(), 10, None, false, None);
        let (idx, d) = g.grow(5, 5, None);
        assert_eq!(idx, vec![5, 4, 6, 3, 7]);
        assert_eq!(d, vec![0.0, 1.0, 1.0, 2.0, 2.0]);

        // Weighted, against the distance matrix as an independent oracle.
        let n = 121;
        let (edges, w) = grid_graph(11, 1.0);
        let wv = w.view();
        let mut g = GeodesicGraph::new(edges.view(), n, Some(&wv), false, None);
        let (idx, d) = g.grow(60, 40, None);
        let dm = geodesic_matrix_graph(
            edges.view(),
            n,
            Some(&wv),
            false,
            Some(&[60]),
            None,
            None,
            Some(1),
        );
        for (&i, &di) in idx.iter().zip(&d) {
            assert_eq!(di, dm[[0, i as usize]], "item {i}");
        }
        assert!(d.windows(2).all(|p| p[0] <= p[1]), "distances ascend");
    }

    #[test]
    fn grow_gives_every_item_on_a_node_the_same_distance() {
        // An item's position *is* its node's, so a node's items must share a distance exactly
        // — which is what lets a caller thin a patch by radius without the ties drifting.
        let edges = path_graph(4);
        let item_nodes = [0u32, 1, 1, 1, 3];
        let mut g = GeodesicGraph::new(edges.view(), 4, None, false, Some(&item_nodes));
        let (idx, d) = g.grow(0, 5, None);
        assert_eq!(idx, vec![0, 1, 2, 3, 4]);
        assert_eq!(d, vec![0.0, 1.0, 1.0, 1.0, 3.0]);

        // Seeding *on* the crowded node: its three items are all at zero, including the seed.
        let (idx, d) = g.grow(2, 4, None);
        assert_eq!(idx, vec![1, 2, 3, 0]);
        assert_eq!(d, vec![0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn grow_distances_survive_walls_and_a_short_harvest() {
        // A budget that fills mid-node must not leave `dists` out of step with the indices.
        let edges = path_graph(4);
        let item_nodes = [0u32, 1, 1, 1];
        let mut g = GeodesicGraph::new(edges.view(), 4, None, false, Some(&item_nodes));
        let (idx, d) = g.grow(0, 3, None);
        assert_eq!((idx.len(), d.len()), (3, 3));
        assert_eq!(d, vec![0.0, 1.0, 1.0]);

        // And when growth stops early against a wall.
        let claimed = [false, true, true, true];
        let (idx, d) = g.grow(0, 4, Some(&claimed));
        assert_eq!(idx, vec![0]);
        assert_eq!(d, vec![0.0]);

        // A budget of zero yields two empty arrays, not one.
        let (idx, d) = g.grow(0, 0, None);
        assert!(idx.is_empty() && d.is_empty());
    }

    #[test]
    fn grow_follows_edge_weights_not_hop_counts() {
        // A star: 0 in the middle, spokes at wildly different lengths. Hop-wise every leaf is
        // one step away, so only the weights can order them.
        let edges = array![[0u32, 1], [0, 2], [0, 3], [0, 4]];
        let w: Array1<f32> = array![9.0, 1.0, 5.0, 3.0];
        let mut g = GeodesicGraph::new(edges.view(), 5, Some(&w.view()), false, None);
        assert_eq!(g.grow(0, 3, None).0, vec![0, 2, 4]);

        // Unweighted, the same graph settles the spokes in index order instead.
        let mut hops = GeodesicGraph::new(edges.view(), 5, None, false, None);
        assert_eq!(hops.grow(0, 3, None).0, vec![0, 1, 2]);
    }

    #[test]
    fn grow_uses_the_true_geodesic_not_the_traversal_path() {
        // The `geodesic_clusters` argument, restated for a count bound. On a 5-cycle seeded at
        // 0, node 3 is 2 hops away the short way (0-4-3). A budget of 4 must therefore take
        // 0, then {1, 4}, then one of {2, 3} — never wandering the long way round first.
        let edges = array![[0u32, 1], [1, 2], [2, 3], [3, 4], [4, 0]];
        let mut g = GeodesicGraph::new(edges.view(), 5, None, false, None);
        let got = g.grow(0, 4, None).0;
        assert_eq!(got[0], 0);
        let mut near = got[1..3].to_vec();
        near.sort_unstable();
        assert_eq!(
            near,
            vec![1, 4],
            "both 1-hop neighbours settle before any 2-hop one"
        );
    }

    #[test]
    fn forbidden_items_are_walls_so_fragments_stay_disjoint_and_connected() {
        // Path of 12, partitioned into fragments of 4: growth from 0 can only run one way, so
        // the walls make the fragments contiguous blocks.
        let edges = path_graph(12);
        let mut g = GeodesicGraph::new(edges.view(), 12, None, false, None);
        let frags = partition(&mut g, 4);
        assert_eq!(frags.len(), 3);
        for (k, frag) in frags.iter().enumerate() {
            let mut sorted = frag.clone();
            sorted.sort_unstable();
            let base = (k * 4) as u32;
            assert_eq!(sorted, (base..base + 4).collect::<Vec<_>>());
        }
    }

    #[test]
    fn a_partition_covers_every_item_exactly_once() {
        // Same driver on a grid mesh, where growth has real freedom to leak if the walls did
        // not hold. The invariant that matters is exact cover.
        let n = 144;
        let (edges, w) = grid_graph(12, 1.0);
        let mut g = GeodesicGraph::new(edges.view(), n, None, false, None);

        for size in [1usize, 5, 17, 144, 500] {
            let frags = partition(&mut g, size);
            let mut seen = vec![0u32; n];
            for frag in &frags {
                assert!(frag.len() <= size, "a fragment never exceeds its budget");
                for &i in frag {
                    seen[i as usize] += 1;
                }
            }
            assert!(seen.iter().all(|&c| c == 1), "size={size}: exact cover");
            // Carving *connected* balls out of a 2D mesh inevitably strands pockets smaller
            // than `size`, so full-size fragments are not something to assert. What must hold
            // is that every fragment is a single connected piece — the property the walls
            // exist to preserve, and the one that would break first if they leaked.
            for frag in &frags {
                assert!(
                    is_connected(edges.view(), frag),
                    "size={size}: fragment {frag:?} is not connected"
                );
            }
        }
        // The weighted graph is a different search; the invariant is the same.
        let mut gw = GeodesicGraph::new(edges.view(), n, Some(&w.view()), false, None);
        let frags = partition(&mut gw, 10);
        let mut seen = vec![0u32; n];
        for frag in &frags {
            for &i in frag {
                seen[i as usize] += 1;
            }
        }
        assert!(seen.iter().all(|&c| c == 1));
    }

    #[test]
    fn growth_cannot_leave_its_connected_component() {
        // Two disjoint triangles. A budget larger than either component still stops at its edge.
        let edges = array![[0u32, 1], [1, 2], [2, 0], [3, 4], [4, 5], [5, 3]];
        let mut g = GeodesicGraph::new(edges.view(), 6, None, false, None);
        let mut got = g.grow(0, 100, None).0;
        got.sort_unstable();
        assert_eq!(got, vec![0, 1, 2]);
        let mut got = g.grow(4, 100, None).0;
        got.sort_unstable();
        assert_eq!(got, vec![3, 4, 5]);
    }

    #[test]
    fn empty_nodes_conduct_so_a_sparse_cloud_stays_connected() {
        // Path of 7 nodes carrying items only at the two ends: 0 and 1 sit on node 0, items 2
        // and 3 on node 6. Nodes 1..5 carry nothing and must conduct, or the two ends would
        // look like separate components to the growth.
        let edges = path_graph(7);
        let item_nodes = [0u32, 0, 6, 6];
        let mut g = GeodesicGraph::new(edges.view(), 7, None, false, Some(&item_nodes));
        assert_eq!(g.n_items(), 4);
        assert_eq!(g.n_nodes(), 7);
        assert_eq!(g.grow(0, 4, None).0, vec![0, 1, 2, 3]);
        // Both items on a node arrive together, in ascending item order.
        assert_eq!(g.grow(2, 2, None).0, vec![2, 3]);
    }

    #[test]
    fn a_fully_claimed_node_walls_but_an_empty_one_still_conducts() {
        // Path 0-1-2-3-4. Items: one on node 0, one on node 2, one on node 4; nodes 1 and 3
        // are empty conduits. Claim the middle item and grow from item 0: the claimed node 2
        // is now a wall, so item 2 (on node 4) is unreachable even though the graph is a
        // single component.
        let edges = path_graph(5);
        let item_nodes = [0u32, 2, 4];
        let mut g = GeodesicGraph::new(edges.view(), 5, None, false, Some(&item_nodes));
        assert_eq!(
            g.grow(0, 3, None).0,
            vec![0, 1, 2],
            "no walls without `forbidden`"
        );

        let claimed = [false, true, false];
        assert_eq!(
            g.grow(0, 3, Some(&claimed)).0,
            vec![0],
            "the claimed node walls growth off from everything beyond it"
        );

        // Nothing claimed on the middle node => it conducts again, claimed or not elsewhere.
        let claimed = [false, false, true];
        assert_eq!(g.grow(0, 3, Some(&claimed)).0, vec![0, 1]);
    }

    #[test]
    fn the_seed_node_is_never_walled_by_its_own_claimed_neighbours() {
        // A star with items on every node. Claim everything but the seed's own item: the seed
        // still grows (it is the source), but every neighbour is a wall, so it gathers itself
        // and stops.
        let edges = array![[0u32, 1], [0, 2], [0, 3]];
        let mut g = GeodesicGraph::new(edges.view(), 4, None, false, None);
        let claimed = [false, true, true, true];
        assert_eq!(g.grow(0, 4, Some(&claimed)).0, vec![0]);

        // And a seed whose own item is claimed still expands rather than walling itself in —
        // it just contributes nothing of its own.
        let claimed = [true, false, false, false];
        let got = g.grow(0, 4, Some(&claimed)).0;
        let mut sorted = got.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![1, 2, 3]);
    }

    #[test]
    fn repeated_growth_from_one_handle_is_reproducible() {
        // The whole point of the handle is that `Scratch` outlives the query. If `reset` left
        // anything behind, the second call would differ from the first.
        let (edges, w) = grid_graph(10, 1.0);
        let mut g = GeodesicGraph::new(edges.view(), 100, Some(&w.view()), false, None);
        let first = g.grow(44, 25, None).0;
        for _ in 0..5 {
            // Interleave unrelated queries: they must not perturb the repeat either.
            g.grow(0, 60, None);
            g.grow(99, 3, None);
            assert_eq!(g.grow(44, 25, None).0, first);
        }
    }

    #[test]
    fn grow_handles_degenerate_requests() {
        let edges = path_graph(4);
        let mut g = GeodesicGraph::new(edges.view(), 4, None, false, None);
        assert!(
            g.grow(2, 0, None).0.is_empty(),
            "a budget of zero gathers nothing"
        );

        // An isolated node with no edges at all.
        let empty: Array2<u32> = Array2::zeros((0, 2));
        let mut g = GeodesicGraph::new(empty.view(), 3, None, false, None);
        assert_eq!(g.grow(1, 10, None).0, vec![1]);

        // A node carrying no items is a legal graph, just an unreachable one to seed from.
        let item_nodes = [1u32];
        let mut g = GeodesicGraph::new(path_graph(3).view(), 3, None, false, Some(&item_nodes));
        assert_eq!(g.n_items(), 1);
        assert_eq!(g.grow(0, 5, None).0, vec![0]);
    }

    #[test]
    fn identity_items_and_explicit_items_agree() {
        // `item_nodes = [0, 1, ..., n-1]` must reproduce the default exactly — the default is
        // only an optimisation of that case, not different semantics.
        let (edges, w) = grid_graph(9, 1.0);
        let ident: Vec<u32> = (0..81u32).collect();
        let mut a = GeodesicGraph::new(edges.view(), 81, Some(&w.view()), false, None);
        let mut b = GeodesicGraph::new(edges.view(), 81, Some(&w.view()), false, Some(&ident));
        for seed in [0u32, 40, 80] {
            assert_eq!(a.grow(seed, 20, None).0, b.grow(seed, 20, None).0);
        }
        assert_eq!(partition(&mut a, 7), partition(&mut b, 7));
    }

    #[test]
    fn grow_agrees_with_the_distance_matrix() {
        // Independent oracle: the region of size k must be a set of k items whose distances are
        // the k smallest, and its order must be non-decreasing in distance.
        let (edges, w) = grid_graph(11, 1.0);
        let n = 121;
        let seed = 60u32;
        let dm = geodesic_matrix_graph(
            edges.view(),
            n,
            Some(&w.view()),
            false,
            Some(&[seed]),
            None,
            None,
            Some(1),
        );
        let d = |v: u32| dm[[0, v as usize]];

        let mut g = GeodesicGraph::new(edges.view(), n, Some(&w.view()), false, None);
        let region = g.grow(seed, 30, None).0;
        assert_eq!(region.len(), 30);
        for pair in region.windows(2) {
            assert!(d(pair[0]) <= d(pair[1]), "settle order is distance order");
        }
        // Nothing outside the region is closer than the farthest item inside it.
        let radius = d(*region.last().unwrap());
        let inside: std::collections::HashSet<u32> = region.iter().copied().collect();
        for v in 0..n as u32 {
            if !inside.contains(&v) {
                assert!(
                    d(v) >= radius,
                    "node {v} at {} is nearer than the ball edge {radius}",
                    d(v)
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // GeodesicGraph::farthest_seed
    // -----------------------------------------------------------------------

    /// Brute-force farthest-point seed: for each undone item, the distance to its nearest
    /// done item; take the largest. Deliberately recomputed from the full distance matrix
    /// every call, so it shares nothing with the incremental machinery under test.
    fn brute_seed(
        edges: &Array2<u32>,
        n_nodes: usize,
        w: Option<&Array1<f32>>,
        item_node: &[u32],
        done: &[bool],
    ) -> Option<u32> {
        let dm = geodesic_matrix_graph(
            edges.view(),
            n_nodes,
            w.map(|w| w.view()).as_ref(),
            false,
            None,
            None,
            None,
            Some(1),
        );
        let sources: Vec<u32> = done
            .iter()
            .enumerate()
            .filter(|(_, &d)| d)
            .map(|(i, _)| item_node[i])
            .collect();

        let mut best: Option<(u32, f32)> = None;
        if !sources.is_empty() {
            for (i, &d) in done.iter().enumerate() {
                if d {
                    continue;
                }
                // -1.0 is the unreachable sentinel; such a pair contributes nothing.
                let near = sources
                    .iter()
                    .map(|&s| dm[[s as usize, item_node[i] as usize]])
                    .filter(|&x| x >= 0.0)
                    .fold(f32::INFINITY, f32::min);
                if near.is_finite() && best.is_none_or(|(_, b)| near > b) {
                    best = Some((i as u32, near));
                }
            }
        }
        if let Some((i, _)) = best {
            return Some(i);
        }
        // Fallback: lowest-index undone item of the component with the most undone items.
        let comp = connected_components_graph(edges.view(), n_nodes);
        let mut counts = vec![0u32; n_nodes];
        for (i, &d) in done.iter().enumerate() {
            if !d {
                counts[comp[item_node[i] as usize] as usize] += 1;
            }
        }
        let best_label = (0..n_nodes).filter(|&l| counts[l] > 0).max_by_key(|&l| {
            // ties -> lowest label, so invert the index in the key
            (counts[l], std::cmp::Reverse(l))
        })?;
        done.iter().enumerate().find_map(|(i, &d)| {
            (!d && comp[item_node[i] as usize] as usize == best_label).then_some(i as u32)
        })
    }

    #[test]
    fn farthest_seed_spreads_over_a_path() {
        // Path 0-1-...-8, seeded at 0. FPS should take the far end, then bisect repeatedly.
        let edges = path_graph(9);
        let mut g = GeodesicGraph::new(edges.view(), 9, None, false, None);
        let mut done = vec![false; 9];
        done[0] = true;
        let mut picks = Vec::new();
        for _ in 0..4 {
            let s = g.farthest_seed(&done).unwrap();
            picks.push(s);
            done[s as usize] = true;
        }
        assert_eq!(picks, vec![8, 4, 2, 6]);
    }

    #[test]
    fn farthest_seed_matches_brute_force_throughout_a_run() {
        // The property that matters: every seed of a long incremental run agrees with a
        // reference that recomputes from the full distance matrix each time.
        let n = 121;
        let (edges, w) = grid_graph(11, 1.0);
        for weights in [None, Some(&w)] {
            let mut g = GeodesicGraph::new(
                edges.view(),
                n,
                weights.map(|w| w.view()).as_ref(),
                false,
                None,
            );
            let mut done = vec![false; n];
            for step in 0..40 {
                let mine = g.farthest_seed(&done);
                let theirs = brute_seed(&edges, n, weights, &g.item_node.clone(), &done);
                assert_eq!(mine, theirs, "step {step}, weighted={}", weights.is_some());
                done[mine.unwrap() as usize] = true;
            }
        }
    }

    #[test]
    fn farthest_seed_matches_brute_force_with_items() {
        // Many items per node, and nodes with none — the cloud case.
        let n = 81;
        let (edges, w) = grid_graph(9, 1.0);
        let item_nodes: Vec<u32> = (0..120u32).map(|i| (i * 7) % 81).collect();
        let mut g = GeodesicGraph::new(edges.view(), n, Some(&w.view()), false, Some(&item_nodes));
        let mut done = vec![false; 120];
        for step in 0..30 {
            let mine = g.farthest_seed(&done);
            let theirs = brute_seed(&edges, n, Some(&w), &item_nodes, &done);
            assert_eq!(mine, theirs, "step {step}");
            done[mine.unwrap() as usize] = true;
        }
    }

    #[test]
    fn farthest_seed_folds_batches_the_same_as_one_at_a_time() {
        // `_cover`-style usage marks a whole region done per call rather than a single item.
        // Folding a batch must land on the same field as folding its members separately.
        let n = 121;
        let (edges, w) = grid_graph(11, 1.0);
        let mut batched = GeodesicGraph::new(edges.view(), n, Some(&w.view()), false, None);
        let mut done = vec![false; n];

        for _ in 0..8 {
            let s = batched.farthest_seed(&done).unwrap();
            // Mark a whole grown region, not just the seed.
            for &i in &batched.grow(s, 9, None).0 {
                done[i as usize] = true;
            }
            // A graph that has never seen an intermediate state must agree.
            let mut fresh = GeodesicGraph::new(edges.view(), n, Some(&w.view()), false, None);
            assert_eq!(
                batched.farthest_seed(&done),
                fresh.farthest_seed(&done),
                "incremental folding must match a cold start"
            );
        }
    }

    #[test]
    fn farthest_seed_prefers_the_reachable_frontier_then_the_largest_component() {
        // A 6-node path, plus a 3-node island and a lone node. Seeded inside the path, FPS
        // must exhaust the path before touching either disconnected piece, then take the
        // *larger* piece first — a mesh full of specks is the reason this rule exists.
        let mut rows: Vec<u32> = (0..5u32).flat_map(|i| [i, i + 1]).collect();
        rows.extend_from_slice(&[6, 7, 7, 8]); // island 6-7-8; node 9 is alone
        let edges = Array2::from_shape_vec((rows.len() / 2, 2), rows).unwrap();
        let mut g = GeodesicGraph::new(edges.view(), 10, None, false, None);

        let mut done = vec![false; 10];
        done[0] = true;
        let mut order = Vec::new();
        for _ in 0..9 {
            let s = g.farthest_seed(&done).unwrap();
            order.push(s);
            done[s as usize] = true;
        }
        assert_eq!(
            &order[..5],
            &[5, 2, 1, 3, 4],
            "the reachable path goes first"
        );
        assert_eq!(&order[5..8], &[6, 8, 7], "then the larger island");
        assert_eq!(order[8], 9, "the lone node last");
        assert_eq!(g.farthest_seed(&done), None, "nothing left to seed");
    }

    #[test]
    fn farthest_seed_is_a_pure_function_of_done() {
        // The selection heap must not consume its winner: asking twice without marking
        // anything has to give the same answer twice. A pop-based selection would quietly
        // return the *second*-farthest on the repeat.
        let n = 121;
        let (edges, w) = grid_graph(11, 1.0);
        let mut g = GeodesicGraph::new(edges.view(), n, Some(&w.view()), false, None);
        let mut done = vec![false; n];
        done[0] = true;
        for _ in 0..15 {
            let s = g.farthest_seed(&done).unwrap();
            for _ in 0..3 {
                assert_eq!(g.farthest_seed(&done), Some(s), "repeat must not drift");
            }
            done[s as usize] = true;
        }
    }

    #[test]
    fn farthest_seed_enrols_every_item_of_a_node_it_reaches() {
        // A node carrying several items becomes reachable in one step; all of its items must
        // enter the candidate pool, not just the first. Path 0-1-2 with three items piled on
        // node 2 and one on node 0.
        let edges = path_graph(3);
        let item_nodes = [0u32, 2, 2, 2];
        let mut g = GeodesicGraph::new(edges.view(), 3, None, false, Some(&item_nodes));
        let mut done = vec![false, false, false, false];
        done[0] = true;
        // All three items on node 2 are equally far; they come out in index order and every
        // one of them must be offered before the pool runs dry.
        let mut picks = Vec::new();
        for _ in 0..3 {
            let s = g.farthest_seed(&done).unwrap();
            picks.push(s);
            done[s as usize] = true;
        }
        assert_eq!(picks, vec![1, 2, 3]);
        assert_eq!(g.farthest_seed(&done), None);
    }

    #[test]
    fn farthest_seed_with_nothing_done_starts_on_the_largest_component() {
        // Small component first in index order, larger one after: the larger must win.
        let edges = array![[0u32, 1], [2, 3], [3, 4], [4, 5]];
        let mut g = GeodesicGraph::new(edges.view(), 6, None, false, None);
        assert_eq!(g.farthest_seed(&[false; 6]), Some(2));
    }

    #[test]
    fn farthest_seed_rebuilds_when_done_shrinks() {
        // The incremental field assumes `done` only grows. If it shrinks the field is stale
        // and cannot be un-folded, so the answer must come from a rebuild — not from
        // whatever the stale field happened to hold.
        let n = 81;
        let (edges, w) = grid_graph(9, 1.0);
        let mut g = GeodesicGraph::new(edges.view(), n, Some(&w.view()), false, None);

        let mut done = vec![false; n];
        for &i in &[0usize, 8, 40, 72, 80] {
            done[i] = true;
            g.farthest_seed(&done);
        }
        // Now clear almost everything and ask again.
        let mut shrunk = vec![false; n];
        shrunk[40] = true;
        let mut fresh = GeodesicGraph::new(edges.view(), n, Some(&w.view()), false, None);
        assert_eq!(g.farthest_seed(&shrunk), fresh.farthest_seed(&shrunk));
        assert_eq!(
            g.farthest_seed(&shrunk),
            brute_seed(
                &edges,
                n,
                Some(&w),
                &(0..n as u32).collect::<Vec<_>>(),
                &shrunk
            )
        );
    }

    #[test]
    fn farthest_seed_handles_saturated_and_degenerate_input() {
        let edges = path_graph(4);
        let mut g = GeodesicGraph::new(edges.view(), 4, None, false, None);
        assert_eq!(g.farthest_seed(&[true; 4]), None);

        // Several done items on one node: the node is a single source, not several.
        let item_nodes = [0u32, 0, 3];
        let mut g = GeodesicGraph::new(edges.view(), 4, None, false, Some(&item_nodes));
        assert_eq!(g.farthest_seed(&[true, true, false]), Some(2));
        // An item sharing a node with a done one is at distance zero — the worst candidate.
        assert_eq!(g.farthest_seed(&[true, false, false]), Some(2));
    }

    // -----------------------------------------------------------------------
    // GeodesicGraph: the mirrored free functions, and `subset`
    // -----------------------------------------------------------------------

    #[test]
    fn methods_agree_with_the_free_functions_they_mirror() {
        // The contract of the whole handle: keeping the adjacency changes the cost, never the
        // answer. Anything else and callers could not migrate off the free functions.
        let n = 121;
        let (edges, w) = grid_graph(11, 1.0);
        let wv = w.view();
        let mut g = GeodesicGraph::new(edges.view(), n, Some(&wv), false, None);

        let srcs = [0u32, 60, 120];
        let tgts = [7u32, 55, 99];
        assert_eq!(
            g.distances(Some(&srcs), Some(&tgts), None, Some(1)),
            geodesic_matrix_graph(
                edges.view(),
                n,
                Some(&wv),
                false,
                Some(&srcs),
                Some(&tgts),
                None,
                Some(1)
            )
        );
        assert_eq!(
            g.nearest(Some(&srcs), Some(&tgts), None, Some(1)),
            geodesic_nearest_mesh_like(&edges, n, &wv, &srcs, &tgts, false)
        );
        assert_eq!(
            g.farthest(Some(&srcs), Some(&tgts), None, Some(1)),
            geodesic_nearest_mesh_like(&edges, n, &wv, &srcs, &tgts, true)
        );
        assert_eq!(
            g.predecessors(Some(&srcs), None, Some(1)),
            geodesic_predecessors_graph(
                edges.view(),
                n,
                Some(&wv),
                false,
                Some(&srcs),
                None,
                Some(1)
            )
        );
        assert_eq!(
            g.path(0, &tgts),
            geodesic_path_graph(edges.view(), n, Some(&wv), false, 0, &tgts)
        );
        assert_eq!(
            g.clusters(2.5, Some(&srcs)),
            geodesic_clusters(edges.view(), n, 2.5, Some(&wv), Some(&srcs))
        );
        assert_eq!(g.components(), connected_components_graph(edges.view(), n));

        // Unweighted too — that is the BFS kernel rather than Dijkstra.
        let g = GeodesicGraph::new(edges.view(), n, None, false, None);
        assert_eq!(
            g.distances(Some(&srcs), None, Some(3.0), Some(1)),
            geodesic_matrix_graph(
                edges.view(),
                n,
                None,
                false,
                Some(&srcs),
                None,
                Some(3.0),
                Some(1)
            )
        );
        assert_eq!(
            g.clusters(2.0, None),
            geodesic_clusters(edges.view(), n, 2.0, None, None)
        );
    }

    /// `geodesic_nearest_mesh`/`geodesic_farthest_mesh` take faces; this is the same call over
    /// an edge list, so the graph handle has something to be compared against.
    fn geodesic_nearest_mesh_like(
        edges: &Array2<u32>,
        n: usize,
        w: &ArrayView1<f32>,
        sources: &[u32],
        targets: &[u32],
        farthest: bool,
    ) -> (Array1<f32>, Array1<i32>) {
        let adj = Adjacency::from_edges(edges.view(), n, Some(w), false);
        geodesic_extreme_impl(&adj, Some(sources), Some(targets), None, Some(1), farthest)
    }

    #[test]
    fn a_directed_handle_follows_arc_direction() {
        // A one-way chain 0->1->2->3. Downstream is reachable, upstream is not.
        let edges = array![[0u32, 1], [1, 2], [2, 3]];
        let mut g = GeodesicGraph::new(edges.view(), 4, None, true, None);
        assert_eq!(g.grow(0, 10, None).0, vec![0, 1, 2, 3]);
        assert_eq!(
            g.grow(3, 10, None).0,
            vec![3],
            "nothing is downstream of the end"
        );
        let d = g.distances(Some(&[0]), None, None, Some(1));
        assert_eq!(d.row(0).to_vec(), vec![0.0, 1.0, 2.0, 3.0]);
        let d = g.distances(Some(&[3]), None, None, Some(1));
        assert_eq!(d.row(0).to_vec(), vec![-1.0, -1.0, -1.0, 0.0]);
        // Components stay *weakly* connected: the chain is one piece either way.
        assert_eq!(g.components(), vec![0, 0, 0, 0]);

        // And the undirected handle over the same edges sees it all both ways.
        let mut u = GeodesicGraph::new(edges.view(), 4, None, false, None);
        assert_eq!(u.grow(3, 10, None).0, vec![3, 2, 1, 0]);
    }

    #[test]
    fn subset_is_indistinguishable_from_a_graph_built_from_the_surviving_edges() {
        // The property that makes `subset` safe: not merely "same distances", but the same
        // object down to neighbour order, so every tie-break and therefore every result agrees.
        let n = 81;
        let (edges, w) = grid_graph(9, 1.0);
        let wv = w.view();
        let g = GeodesicGraph::new(edges.view(), n, Some(&wv), false, None);

        // Keep the middle block of the grid, in a deliberately jumbled order.
        let mut keep: Vec<u32> = (0..81u32)
            .filter(|v| (2..7).contains(&(v / 9)) && (2..7).contains(&(v % 9)))
            .collect();
        keep.swap(0, 7);
        keep.swap(3, 19);
        let (mut sub, kept_items) = g.subset(&keep);
        assert_eq!(sub.n_nodes(), keep.len());
        assert_eq!(
            kept_items, keep,
            "with no items attached, item i must still be node i"
        );
        assert_eq!(sub.item_nodes(), (0..keep.len() as u32).collect::<Vec<_>>());

        // Rebuild the same subgraph the long way round, from a filtered edge list.
        let mut new_id = vec![u32::MAX; n];
        for (i, &v) in keep.iter().enumerate() {
            new_id[v as usize] = i as u32;
        }
        let mut rows: Vec<u32> = Vec::new();
        let mut rw: Vec<f32> = Vec::new();
        for (e, &wt) in edges.rows().into_iter().zip(w.iter()) {
            let (a, b) = (new_id[e[0] as usize], new_id[e[1] as usize]);
            if a != u32::MAX && b != u32::MAX {
                rows.extend_from_slice(&[a, b]);
                rw.push(wt);
            }
        }
        let re = Array2::from_shape_vec((rows.len() / 2, 2), rows).unwrap();
        let rwa = Array1::from(rw);
        let mut fresh = GeodesicGraph::new(re.view(), keep.len(), Some(&rwa.view()), false, None);

        assert_eq!(
            sub.distances(None, None, None, Some(1)),
            fresh.distances(None, None, None, Some(1))
        );
        assert_eq!(sub.components(), fresh.components());
        assert_eq!(sub.clusters(2.0, None), fresh.clusters(2.0, None));
        for seed in [0u32, 5, 12] {
            assert_eq!(sub.grow(seed, 9, None).0, fresh.grow(seed, 9, None).0);
            assert_eq!(sub.path(seed, &[1, 20]), fresh.path(seed, &[1, 20]));
        }
    }

    #[test]
    fn subset_distances_match_the_parent_where_the_subgraph_is_the_whole_route() {
        // A subset only agrees with its parent where no shortest path left the subset. Taking a
        // whole connected component guarantees that, which is the common use.
        let edges = array![[0u32, 1], [1, 2], [2, 0], [3, 4], [4, 5]];
        let wv = array![1.0f32, 2.0, 4.0, 1.0, 1.0];
        let g = GeodesicGraph::new(edges.view(), 6, Some(&wv.view()), false, None);
        let (sub, kept) = g.subset(&[3, 4, 5]);
        assert_eq!(kept, vec![3, 4, 5]);
        let parent = g.distances(Some(&[3, 4, 5]), Some(&[3, 4, 5]), None, Some(1));
        assert_eq!(sub.distances(None, None, None, Some(1)), parent);
    }

    #[test]
    fn subset_carries_items_and_drops_those_whose_node_is_gone() {
        // Path 0-1-2-3 with items on nodes 0, 0, 2, 3. Keeping nodes {2, 0} must keep items
        // 0, 1 and 2, renumber them 0..2, and report their original indices.
        let edges = path_graph(4);
        let item_nodes = [0u32, 0, 2, 3];
        let g = GeodesicGraph::new(edges.view(), 4, None, false, Some(&item_nodes));
        let (mut sub, kept) = g.subset(&[2, 0]);
        // Items come out grouped by new node: node 0 (old 2) carries item 2, then node 1
        // (old 0) carries items 0 and 1. Item 3 rode on the dropped node 3 and is gone.
        assert_eq!(kept, vec![2, 0, 1]);
        assert_eq!(sub.n_items(), 3);
        assert_eq!(sub.n_nodes(), 2);
        assert_eq!(sub.item_nodes(), &[0, 1, 1]);
        // Old nodes 2 and 0 are not adjacent, so the induced subgraph has no edges at all and
        // growth cannot leave the node it starts on.
        assert_eq!(sub.components(), vec![0, 1]);
        assert_eq!(sub.grow(0, 5, None).0, vec![0]);
        assert_eq!(
            sub.grow(1, 5, None).0,
            vec![1, 2],
            "both items of new node 1"
        );
    }

    #[test]
    fn subset_rejects_repeats_and_out_of_range_nodes() {
        let g = GeodesicGraph::new(path_graph(4).view(), 4, None, false, None);
        assert!(std::panic::catch_unwind(|| g.subset(&[0, 1, 1])).is_err());
        assert!(std::panic::catch_unwind(|| g.subset(&[0, 9])).is_err());
        // The empty subset is legal, just empty.
        let (sub, kept) = g.subset(&[]);
        assert_eq!((sub.n_nodes(), sub.n_items(), kept.len()), (0, 0, 0));
    }

    #[test]
    fn item_components_label_by_lowest_node_index() {
        let edges = array![[1u32, 2], [4, 5]];
        let mut g = GeodesicGraph::new(edges.view(), 6, None, false, None);
        assert_eq!(g.item_components(), vec![0, 1, 1, 3, 4, 4]);
        // Matches the free function's labelling, which callers may already rely on.
        assert_eq!(
            g.item_components(),
            connected_components_graph(edges.view(), 6)
        );

        // With items attached, each item takes its node's label.
        let item_nodes = [5u32, 0, 2];
        let mut g = GeodesicGraph::new(edges.view(), 6, None, false, Some(&item_nodes));
        assert_eq!(g.item_components(), vec![4, 0, 1]);
    }
}
