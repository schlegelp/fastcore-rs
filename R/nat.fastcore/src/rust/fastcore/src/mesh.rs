use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rayon::prelude::*;
use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;

use crate::nblast::with_pool;

// ---------------------------------------------------------------------------
// Edge weight width
// ---------------------------------------------------------------------------

/// The width a graph's edge lengths — and therefore its distances — are carried at: `f32` or
/// `f64`.
///
/// A local trait for the same reason [`crate::linkage::Dissim`] and [`crate::matches::Score`]
/// are: it pins down exactly what the kernels need and nothing else. `num::Float` would drag in
/// a hundred methods, none of which is the one that actually matters here — [`Bits`](Self::Bits).
///
/// # Why `Bits`
///
/// Every search in this module keys its heap on the raw IEEE bit pattern rather than on the
/// float. For *non-negative* floats — which distances always are, since weights are lengths and
/// we start at zero — that pattern is monotone read as an unsigned integer, so `Ord` on the bits
/// *is* `Ord` on the floats, exactly, `+inf` included. That buys a derived `Ord` on
/// [`HeapEntry`] (hence no `partial_cmp().unwrap()` and no NaN panic path) and an integer
/// compare in the sift loop. Making the width generic therefore means carrying the integer of
/// matching width alongside it, which is what this associated type is.
///
/// # Why `Packed`
///
/// The same monotonicity is what lets [`Adjacency::compact`] order a row by neighbour and then
/// by weight in a single sort, so that the shortest of a set of parallel edges comes first.
/// Which needs a `(neighbour, weight-bits)` value that is `Ord` — and *how* it is spelt is worth
/// a type rather than a tuple, because the obvious tuple is measurably slower.
///
/// A `(u32, u32)` compares lexicographically in five instructions: two `cmp`/`cset` pairs and a
/// `csel`. The equivalent `u64` compares in two. LLVM cannot bridge the gap, and not for want of
/// trying: on a little-endian target the tuple's first field lands in the *low* half of a 64-bit
/// load, so one wide compare would order by weight before neighbour — the wrong answer. The only
/// way to get the single compare is to place the fields deliberately, which is what `f32`'s impl
/// does; the sort is the inner loop of the adjacency build, so it is worth the associated type.
///
/// `f64` has no such option — 32 + 64 bits do not fit in a register pair to begin with — so it
/// uses the tuple, and pays the five instructions it was always going to pay.
///
/// # Choosing a width
///
/// `f32` is the default throughout and is the right one for mesh and skeleton work: a 24-bit
/// mantissa resolves a 100 mm neuron to ~6 nm, and the distance arrays are the largest thing
/// these functions allocate — a `(V, V)` matrix at `V = 164k` is 107 GB in `f32` and 215 GB in
/// `f64`.
///
/// `f64` earns its keep where the *accumulation* is long rather than the graph large. Dijkstra
/// sums one weight per hop, so a path of `k` hops carries up to `k` roundings; at `f32` and
/// `k` in the tens of thousands — a densely sampled arbor, a fine mesh — the drift becomes
/// visible against an exact answer, and comparisons against `scipy.sparse.csgraph`, which works
/// in `f64` unconditionally, stop agreeing to the last bits. It also matters when weights span
/// a wide dynamic range, since `fl(du + w)` loses `w` entirely once `du` exceeds it by 2^24.
pub trait Weight:
    Copy + PartialOrd + std::ops::Add<Output = Self> + std::fmt::Display + Send + Sync + 'static
{
    /// The unsigned integer of the same width, used as the order-preserving heap key.
    type Bits: Copy + Ord + Send + Sync;
    /// A `(neighbour, weight-bits)` pair that is `Ord` by neighbour first and weight second.
    /// Opaque on purpose — see the note above; only [`pack`](Self::pack) and
    /// [`unpack`](Self::unpack) may assume a layout.
    type Packed: Copy + Ord + Send + Sync;

    const ZERO: Self;
    const ONE: Self;
    /// The unreachable sentinel the drivers prefill and return. `-1`, not `NaN`: it survives
    /// the trip through numpy and R unchanged, and it is what the rest of the crate uses.
    const NEG_ONE: Self;
    const INFINITY: Self;

    fn to_bits(self) -> Self::Bits;
    fn from_bits(bits: Self::Bits) -> Self;
    fn pack(node: u32, bits: Self::Bits) -> Self::Packed;
    fn unpack(packed: Self::Packed) -> (u32, Self::Bits);
    fn is_finite(self) -> bool;
    fn is_infinite(self) -> bool;
    /// A total order over *all* floats, negatives included — what
    /// [`minimum_spanning_tree`] sorts on, since unlike the geodesic searches it accepts
    /// negative weights and so cannot use the bit-pattern trick.
    fn total_cmp(&self, other: &Self) -> Ordering;
    /// Narrow an `f64` to this width. Mesh edge lengths are computed from `f64` coordinates and
    /// land here, which is the only place a value enters at a width that is not already `Self`.
    fn from_f64(x: f64) -> Self;
}

/// Everything but [`Weight::Packed`], which is the one member the two impls genuinely differ on.
macro_rules! impl_weight {
    ($t:ty, $bits:ty) => {
        const ZERO: Self = 0.0;
        const ONE: Self = 1.0;
        const NEG_ONE: Self = -1.0;
        const INFINITY: Self = <$t>::INFINITY;

        #[inline(always)]
        fn to_bits(self) -> Self::Bits {
            <$t>::to_bits(self)
        }
        #[inline(always)]
        fn from_bits(bits: Self::Bits) -> Self {
            <$t>::from_bits(bits)
        }
        #[inline(always)]
        fn is_finite(self) -> bool {
            <$t>::is_finite(self)
        }
        #[inline(always)]
        fn is_infinite(self) -> bool {
            <$t>::is_infinite(self)
        }
        #[inline(always)]
        fn total_cmp(&self, other: &Self) -> Ordering {
            <$t>::total_cmp(self, other)
        }
        #[inline(always)]
        fn from_f64(x: f64) -> Self {
            x as $t
        }
    };
}

impl Weight for f32 {
    type Bits = u32;
    /// Both halves in one integer, neighbour in the high bits. This is the placement the
    /// generic `(u32, u32)` cannot express and the reason `Packed` is a type — see above.
    type Packed = u64;

    impl_weight!(f32, u32);

    #[inline(always)]
    fn pack(node: u32, bits: u32) -> u64 {
        ((node as u64) << 32) | bits as u64
    }
    #[inline(always)]
    fn unpack(packed: u64) -> (u32, u32) {
        ((packed >> 32) as u32, packed as u32)
    }
}

impl Weight for f64 {
    type Bits = u64;
    /// 96 bits do not pack, so the derived lexicographic `Ord` on a tuple is the best available.
    type Packed = (u32, u64);

    impl_weight!(f64, u64);

    #[inline(always)]
    fn pack(node: u32, bits: u64) -> (u32, u64) {
        (node, bits)
    }
    #[inline(always)]
    fn unpack(packed: (u32, u64)) -> (u32, u64) {
        packed
    }
}

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
/// `(edges, index, inverse, lengths)` where `edges` is a `(n_unique, 2)` u32
/// array with rows `[min, max]` sorted ascending by `(max, min)` — byte-for-byte
/// the order and first-occurrence semantics of trimesh / `np.unique`. `index` and
/// `inverse` are i64: they are positions in the `3F` edge list, not node ids.
#[allow(clippy::type_complexity)]
pub fn unique_edges(
    faces: ArrayView2<u32>,
    coords: Option<ArrayView2<f64>>,
    return_index: bool,
    return_inverse: bool,
    threads: Option<usize>,
) -> (
    Array2<u32>,
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
            let mut edges: Vec<u32> = Vec::with_capacity(n_unique * 2);
            let mut prev = None;
            for &k in &keys {
                if prev != Some(k) {
                    edges.push(k as u32);
                    edges.push((k >> 32) as u32);
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

            let mut edges: Vec<u32> = Vec::new();
            let mut index: Vec<i64> = Vec::new();
            let mut inverse: Vec<i64> = if return_inverse { vec![0; n_edges] } else { Vec::new() };
            let mut prev: Option<u64> = None;
            let mut slot: i64 = -1;
            for &p in &packed {
                let key = (p >> 64) as u64;
                let orig = p as u64;
                if prev != Some(key) {
                    slot += 1;
                    edges.push(key as u32);
                    edges.push((key >> 32) as u32);
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
fn edge_lengths(edges: &[u32], coords: ArrayView2<f64>) -> Array1<f64> {
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

/// Invert a list of distinct node indices: `out[nodes[i]] == i`, `u32::MAX` where unlisted.
///
/// The map back from a node to *which of the caller's nodes* it is — needed by anything that
/// renumbers a subset, and the place the "must be distinct" contract is actually enforced,
/// since a repeat is a collision in this array and costs nothing extra to spot.
fn inverse_index(nodes: &[u32], n_nodes: usize) -> Vec<u32> {
    let mut out: Vec<u32> = vec![u32::MAX; n_nodes];
    for (i, &v) in nodes.iter().enumerate() {
        assert!(
            (v as usize) < n_nodes,
            "`nodes` contains node {v}, but n_nodes = {n_nodes}"
        );
        assert!(
            out[v as usize] == u32::MAX,
            "`nodes` contains node {v} more than once"
        );
        out[v as usize] = i as u32;
    }
    out
}

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
/// An `(n_unique, 2)` u32 array of the surviving edges as `[min, max]` rows, sorted ascending
/// by `(max, min)` — the same ordering [`unique_edges`] produces.
pub fn contract_vertices(
    edges: ArrayView2<u32>,
    mapping: ArrayView1<u32>,
    threads: Option<usize>,
) -> Array2<u32> {
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

        let mut out: Vec<u32> = Vec::new();
        let mut prev = None;
        for &k in &keys {
            if prev != Some(k) {
                out.push(k as u32);
                out.push((k >> 32) as u32);
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
///
/// Generic over the weight width; see [`Weight`]. Which one is chosen does not change *which*
/// edges are kept unless two weights are so close that they compare equal at `f32` and not at
/// `f64` — and then the tie-break on edge index still makes the answer deterministic.
pub fn minimum_spanning_tree<W: Weight>(
    edges: ArrayView2<u32>,
    n_nodes: usize,
    weights: Option<&ArrayView1<W>>,
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
            assert!(W::is_finite(x), "edge weights must be finite, got {x}");
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
                W::total_cmp(&w[a as usize], &w[b as usize]).then_with(|| a.cmp(&b))
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

/// Which edges are *bridges* — the ones whose removal would disconnect their component.
///
/// Tarjan's algorithm: one depth-first sweep tracking, per node, the earliest discovery time
/// reachable from its subtree by a single back edge. A tree edge `(u, v)` is a bridge exactly
/// when nothing in `v`'s subtree can climb above `v`, i.e. there is no second route around it.
///
/// The counterpart to [`minimum_spanning_tree`] rather than a variant of it: the MST asks which
/// edges to *keep* to stay connected, this asks which ones may not be *dropped*. That is the
/// question behind "prune this graph but do not shatter it", where a caller has a set of edges
/// it would like gone and needs to know which of them are load-bearing.
///
/// Parallel edges are honoured, which is why this does not go through [`Adjacency`]: two nodes
/// joined twice are joined by a cycle, so neither of those edges is a bridge, and a
/// deduplicated adjacency — which is what every search in this module wants — would fuse them
/// into one and report a bridge that is not there. Self-loops are never bridges.
///
/// Arguments
/// ---------
/// - `edges`:   (E, 2) array of edges given as node indices. Direction is ignored.
/// - `n_nodes`: Total number of nodes.
///
/// Returns
/// -------
/// A 1-D bool array with one flag per input edge, `true` for a bridge. A mask rather than a
/// list of indices because the caller's next move is nearly always to filter a parallel array
/// (`edges[!mask]`), and `flatnonzero` recovers the indices when it is not.
pub fn bridges(edges: ArrayView2<u32>, n_nodes: usize) -> Array1<bool> {
    let n_edges = check_edges(edges, n_nodes);
    let mut out = vec![false; n_edges];
    // No `n_nodes == 0` arm: `check_edges` has already rejected every edge in that case, so an
    // empty graph arrives here as an empty edge list.
    if n_edges == 0 {
        return Array1::from_vec(out);
    }

    // Before the counting pass, not after: the counters below are the offsets themselves, so
    // an edge list this large would wrap them rather than trip anything later.
    assert!(
        n_edges.saturating_mul(2) <= u32::MAX as usize,
        "too many edges: CSR offsets are u32"
    );

    // Arc CSR carrying, per arc, the *row* it came from — that edge id is what distinguishes
    // two parallel edges from one edge seen twice, and it is also how the walk below knows
    // which arc it entered a node by without a separate parent-node field (which would get
    // parallel edges wrong).
    let mut offsets: Vec<u32> = vec![0; n_nodes + 1];
    for e in edges.rows() {
        if e[0] != e[1] {
            offsets[e[0] as usize + 1] += 1;
            offsets[e[1] as usize + 1] += 1;
        }
    }
    for i in 0..n_nodes {
        offsets[i + 1] += offsets[i];
    }
    let n_arcs = offsets[n_nodes] as usize;

    let mut nbrs: Vec<u32> = vec![0; n_arcs];
    let mut eids: Vec<u32> = vec![0; n_arcs];
    {
        let mut cursor: Vec<u32> = offsets[..n_nodes].to_vec();
        for (i, e) in edges.rows().into_iter().enumerate() {
            let (u, v) = (e[0], e[1]);
            if u == v {
                continue;
            }
            for (a, b) in [(u, v), (v, u)] {
                let slot = &mut cursor[a as usize];
                nbrs[*slot as usize] = b;
                eids[*slot as usize] = i as u32;
                *slot += 1;
            }
        }
    }

    /// Not yet discovered. Discovery times are counted from 0, so `u32::MAX` cannot collide.
    const UNVISITED: u32 = u32::MAX;
    let mut disc: Vec<u32> = vec![UNVISITED; n_nodes];
    let mut low: Vec<u32> = vec![0; n_nodes];
    let mut timer: u32 = 0;

    // Explicit stack, not recursion: a mesh strip is a path tens of thousands of nodes long and
    // the natural recursive form would overflow on it. Each frame is the node, the edge we
    // entered it by, and how far through its arc row we have got.
    let mut stack: Vec<(u32, u32, u32)> = Vec::new();

    for s in 0..n_nodes as u32 {
        if disc[s as usize] != UNVISITED {
            continue;
        }
        disc[s as usize] = timer;
        low[s as usize] = timer;
        timer += 1;
        stack.push((s, u32::MAX, offsets[s as usize]));

        while !stack.is_empty() {
            let top = stack.len() - 1;
            let (v, pe, cur) = stack[top];
            if cur < offsets[v as usize + 1] {
                stack[top].2 = cur + 1;
                let (w, e) = (nbrs[cur as usize], eids[cur as usize]);
                // The arc we came in by — not a back edge. Testing the *edge id* rather than
                // the neighbour is what makes a second, parallel edge to the same node count:
                // it has its own id, so it is taken, and it correctly rules out a bridge.
                if e == pe {
                    continue;
                }
                if disc[w as usize] == UNVISITED {
                    disc[w as usize] = timer;
                    low[w as usize] = timer;
                    timer += 1;
                    stack.push((w, e, offsets[w as usize]));
                } else {
                    low[v as usize] = low[v as usize].min(disc[w as usize]);
                }
            } else {
                stack.pop();
                if let Some(&(u, _, _)) = stack.last() {
                    low[u as usize] = low[u as usize].min(low[v as usize]);
                    // Nothing under `v` reaches back past `v` itself, so the tree edge into it
                    // is the component's only route there.
                    if low[v as usize] > disc[u as usize] {
                        out[pe as usize] = true;
                    }
                }
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
///
/// `W` is the width lengths are stored and accumulated at; see [`Weight`].
pub struct Adjacency<W: Weight> {
    /// `offsets[v]..offsets[v + 1]` is the slice of `nbrs` holding v's neighbours.
    offsets: Vec<u32>,
    nbrs: Vec<u32>,
    /// Length of each arc, parallel to `nbrs`. `None` => unit weights.
    weights: Option<Vec<W>>,
    /// Whether an edge was stored as one arc or two. Recorded because the searches are not the
    /// only thing that cares: re-weighting an undirected edge has to move *both* of its arcs to
    /// keep the adjacency symmetric, and only the builder knows there are two.
    directed: bool,
}

impl<W: Weight> Adjacency<W> {
    /// TODO: the build is ~5-15% slower than the pre-generic version and nobody knows why.
    ///
    /// Making the width generic cost that much on `from_faces` / `from_edges` / `induced`,
    /// measured against the commit before it on the `8 src, limit=0.05` line of
    /// `examples/profile_mesh.rs` — the case where the build, not the search, is the whole
    /// cost. The searches themselves are at parity, so this is confined to the three
    /// constructors here.
    ///
    /// The one mechanism that *was* found is already fixed: a `(u32, u32)` pair compares in
    /// five instructions where the packed `u64` compares in two, which is why [`Weight::Packed`]
    /// is an associated type rather than a tuple. That accounted for some of it and not all.
    ///
    /// Ruled out, so as not to be re-investigated: the output gather in
    /// [`geodesic_matrix_impl`] (isolated, identical), the generic-vs-concrete gather codegen
    /// (the generic one is *smaller*), `vec![W::pack(0, 0); n]` losing `Vec`'s zeroed-allocation
    /// path (tried a `const ZERO_PACKED`; no effect), and both search kernels. `compact` and
    /// the constructors differ from their pre-generic form only in ways that must fold at
    /// `f32`, so the remaining suspicion is instantiation-dependent inlining — which wants a
    /// look at the IR rather than more guessing.
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
    fn set_edge(&mut self, u: u32, v: u32, w: W) -> bool {
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
    fn set_arc(&mut self, u: u32, v: u32, w: W) -> bool {
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
    /// Rows hold [`Weight::Packed`] values, ordered by neighbour first and payload second, so
    /// one sort per row does both; see `from_edges` and the note on `Packed` for why that is a
    /// type rather than a tuple.
    fn compact(offsets: &mut [u32], packed: &mut Vec<W::Packed>, n_nodes: usize) {
        let old: Vec<u32> = offsets.to_vec();
        let mut w: usize = 0;
        for u in 0..n_nodes {
            let lo = old[u] as usize;
            let hi = old[u + 1] as usize;
            debug_assert!(w <= lo);
            packed[lo..hi].sort_unstable();

            offsets[u] = w as u32;
            // No neighbour can be `u32::MAX`: the CSR offsets are `u32`, so `n_nodes` is
            // strictly below it and every id is below that.
            let mut prev = u32::MAX;
            for k in lo..hi {
                let p = packed[k];
                let v = W::unpack(p).0;
                // Keep the first entry per neighbour. Because the row is sorted and the
                // payload is the second element, "first" is the *smallest* payload — which is
                // what we want for parallel edges: the shortest one is the only one that can
                // ever be on a shortest path.
                if prev != v && v as usize != u {
                    packed[w] = p;
                    w += 1;
                    prev = v;
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
        let zero = W::ZERO.to_bits();
        let mut packed: Vec<W::Packed> = vec![W::pack(0, zero); offsets[n_nodes] as usize];
        let mut cursor: Vec<u32> = offsets[..n_nodes].to_vec();
        let mut put = |u: u32, v: u32| {
            let slot = &mut cursor[u as usize];
            packed[*slot as usize] = W::pack(v, zero);
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
        let nbrs: Vec<u32> = packed.iter().map(|&p| W::unpack(p).0).collect();

        // Weights last, so we only pay for arcs that survived dedup.
        //
        // The length itself is always computed in f64 — the coordinates arrive at that width,
        // and rounding once at the end is strictly better than rounding the deltas first — and
        // narrowed to `W` on the way into the array.
        //
        // d(u,v) and d(v,u) are computed independently but come out bit-identical: the
        // expression squares each delta, and (a-b)^2 == (b-a)^2 exactly in IEEE. The
        // adjacency is therefore *exactly* symmetric — an asymmetric weight would silently
        // break d(s,t) == d(t,s).
        let weights = coords.map(|c| {
            let mut out: Vec<W> = Vec::with_capacity(nbrs.len());
            for u in 0..n_nodes {
                let (ux, uy, uz) = (c[[u, 0]], c[[u, 1]], c[[u, 2]]);
                for &v in &nbrs[offsets[u] as usize..offsets[u + 1] as usize] {
                    let v = v as usize;
                    let (dx, dy, dz) = (ux - c[[v, 0]], uy - c[[v, 1]], uz - c[[v, 2]]);
                    out.push(W::from_f64((dx * dx + dy * dy + dz * dz).sqrt()));
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
        weights: Option<&ArrayView1<W>>,
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

        // Pack (neighbour, weight-bits) so a single sort orders by neighbour and then by
        // weight. That works because a non-negative float's IEEE bit pattern is monotone read
        // as an unsigned integer — the same fact the heap key relies on; see [`Weight::Bits`].
        // Sorting ascending therefore puts the *shortest* parallel edge first, and `compact`
        // keeps the first.
        let zero = W::ZERO.to_bits();
        let mut packed: Vec<W::Packed> = vec![W::pack(0, zero); offsets[n_nodes] as usize];
        let mut cursor: Vec<u32> = offsets[..n_nodes].to_vec();
        {
            let mut put = |u: u32, v: u32, wbits: W::Bits| {
                let slot = &mut cursor[u as usize];
                packed[*slot as usize] = W::pack(v, wbits);
                *slot += 1;
            };
            for (i, e) in edges.rows().into_iter().enumerate() {
                let wbits = match weights {
                    Some(w) => {
                        let x = w[i];
                        assert!(
                            x >= W::ZERO && W::is_finite(x),
                            "edge weights must be finite and non-negative, got {x}"
                        );
                        x.to_bits()
                    }
                    None => zero,
                };
                put(e[0], e[1], wbits);
                if !directed {
                    put(e[1], e[0], wbits);
                }
            }
        }
        drop(cursor);

        Self::compact(&mut offsets, &mut packed, n_nodes);

        let nbrs: Vec<u32> = packed.iter().map(|&p| W::unpack(p).0).collect();
        let weights = weights.map(|_| {
            packed
                .iter()
                .map(|&p| W::from_bits(W::unpack(p).1))
                .collect()
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
    fn induced(&self, keep: &[u32]) -> Adjacency<W> {
        let n_old = self.n_nodes();
        let new_id = inverse_index(keep, n_old);

        // Packed (neighbour, weight-bits) rows, as in `from_edges`, so one sort per row orders
        // by the *new* index while keeping each arc's weight welded to it.
        let mut offsets: Vec<u32> = vec![0; keep.len() + 1];
        let mut packed: Vec<W::Packed> = Vec::new();
        for (i, &v) in keep.iter().enumerate() {
            let r = self.row(v);
            let start = packed.len();
            for (k, &n) in self.nbrs[r.clone()].iter().enumerate() {
                let m = new_id[n as usize];
                if m != u32::MAX {
                    let bits = self
                        .weights
                        .as_ref()
                        .map_or_else(|| W::ZERO.to_bits(), |w| w[r.start + k].to_bits());
                    packed.push(W::pack(m, bits));
                }
            }
            packed[start..].sort_unstable();
            offsets[i + 1] = packed.len() as u32;
        }
        // No dedup or self-loop pass: the source rows have neither, and `new_id` is injective,
        // so neither can appear here.
        let nbrs: Vec<u32> = packed.iter().map(|&p| W::unpack(p).0).collect();
        let weights = self.weights.as_ref().map(|_| {
            packed
                .iter()
                .map(|&p| W::from_bits(W::unpack(p).1))
                .collect()
        });

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

/// Min-heap entry: a distance key and the node it belongs to. Eight bytes at `f32`, sixteen at
/// `f64`.
///
/// The distance is stored as its raw IEEE bit pattern — see [`Weight::Bits`] for why that is an
/// exact ordering rather than an approximation to be tolerated. It buys a derived `Ord` (hence
/// no `partial_cmp().unwrap()` and no NaN panic path), an integer compare in the sift loop, and
/// at `f32` a POD entry that packs four to a cache line.
///
/// `dist_bits` must stay the first field — the derived `Ord` is lexicographic in declaration
/// order. Tie-breaking on `node` makes the order total, so results are reproducible across
/// runs and thread counts.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct HeapEntry<B: Ord> {
    dist_bits: B,
    node: u32,
}

/// "No predecessor" — the source itself, and any node the search never reached.
///
/// The two are distinguishable via the distance (0.0 vs unreachable), so one sentinel is
/// enough; it surfaces to callers as `-1`, consistent with the rest of the module rather than
/// with scipy's `-9999`.
const NO_PRED: u32 = u32::MAX;

/// Per-worker scratch. Allocated once per rayon chunk and reused across every source in it.
struct Scratch<W: Weight> {
    /// Tentative distance per node. `INFINITY` = not reached.
    /// Invariant: all-`INFINITY` on entry to and exit from every search.
    dist: Vec<W>,
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
    heap: BinaryHeap<Reverse<HeapEntry<W::Bits>>>,
    /// BFS ping-pong frontiers. `Vec::new()` does not allocate, so the Dijkstra path pays
    /// nothing for these and the BFS path pays nothing for `heap`.
    cur: Vec<u32>,
    next: Vec<u32>,
}

impl<W: Weight> Scratch<W> {
    fn new(n_nodes: usize) -> Self {
        Scratch {
            dist: vec![W::INFINITY; n_nodes],
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

    /// Label each of `nodes` with the source its shortest path came from, in `src`.
    ///
    /// `nodes` must be in settle order, as a [`Collect`] visitor records it. That is what makes
    /// this one forward pass rather than a chain-walk per node: a node's predecessor always
    /// settles before the node, so `src[pred[v]]` is already correct by the time we reach `v`.
    /// A node with no predecessor is a source and is its own.
    ///
    /// `src` needs no reset for the same reason — nothing is read before it is written.
    fn resolve_sources_into(&mut self, nodes: &[u32]) {
        for &v in nodes {
            let p = self.pred[v as usize];
            self.src[v as usize] = if p == NO_PRED {
                v
            } else {
                self.src[p as usize]
            };
        }
    }

    /// [`resolve_sources_into`](Self::resolve_sources_into), gathered into a fresh array aligned
    /// with `nodes` — for callers handing the answer back rather than indexing `src` themselves.
    fn resolve_sources(&mut self, nodes: &[u32]) -> Vec<u32> {
        self.resolve_sources_into(nodes);
        nodes.iter().map(|&v| self.src[v as usize]).collect()
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
            self.dist.fill(W::INFINITY);
            if track_pred {
                self.pred.fill(NO_PRED);
            }
        } else {
            for &v in &self.touched {
                self.dist[v as usize] = W::INFINITY;
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
///
/// Generic over the distance width so a visitor may be written for one width only — [`Grow`]
/// and [`Collect`] are, since the type that drives them is `f32`-only — or for both, as
/// [`Targets`] and [`NoVisitor`] are.
trait Visitor<W: Weight> {
    /// Note that `node` has settled at distance `d`, and say how the search should proceed.
    fn settle(&mut self, node: u32, d: W) -> Visit;
}

/// Which targets a search is waiting on, and what it learned when it settled them.
///
/// Shared by the matrix, nearest and farthest drivers, because all three want the same thing
/// (stop as early as the question allows) and differ only in what they keep.
struct Targets<'a, W: Weight> {
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
    first: Option<(u32, W)>,
    /// Last target settled, i.e. the farthest — free, for the same reason.
    last: Option<(u32, W)>,
}

/// Targets never wall: a target is something to *find*, not something to route around, so
/// every settled node is expanded until the search is done.
impl<W: Weight> Visitor<W> for Targets<'_, W> {
    #[inline]
    fn settle(&mut self, node: u32, d: W) -> Visit {
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
fn dijkstra_drain<W: Weight, const PRED: bool, V: Visitor<W>>(
    adj: &Adjacency<W>,
    dist: &mut [W],
    pred: &mut [u32],
    reached: &mut Vec<u32>,
    heap: &mut BinaryHeap<Reverse<HeapEntry<W::Bits>>>,
    limit: W,
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
        let du = W::from_bits(dist_bits);

        match vis.settle(u, du) {
            Visit::Stop => return,
            // Settled, but not conducting: leave its neighbours alone and pop the next node.
            // The node keeps its distance and stays in `touched`, so the reset still finds it.
            Visit::Wall => continue,
            Visit::Expand => {}
        }

        let r = adj.row(u);
        for (&v, &w) in adj.nbrs[r.clone()].iter().zip(&weights[r]) {
            // Accumulating at the graph's own width keeps Dijkstra's invariant, whichever it
            // is: w >= 0 and round-to-nearest gives fl(du + w) >= du, so the key never moves
            // backwards.
            let nd = du + w;
            if nd > limit {
                continue; // prune here, not at pop — this is where the memory win lives
            }
            let slot = &mut dist[v as usize];
            if nd < *slot {
                if W::is_infinite(*slot) {
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
/// for no reason. Hop counts are integers and exact in f32 up to 2^24 (and in f64 far beyond
/// it); no mesh has a 16M-hop path, so the unweighted answer does not depend on the width.
///
/// `PRED` as there. A node is claimed by whichever frontier member reaches it first, so ties
/// within a level resolve in frontier order — deterministic, and acyclic for free because `dist`
/// strictly increases along the chain.
///
/// As there, `dist` may be warm: the guard is `level < *slot` rather than "unvisited", which on
/// a cold array (everything `INFINITY`) is the same test and on a warm one keeps only genuine
/// improvements. `reached` collects the nodes that go finite for the first time.
#[allow(clippy::too_many_arguments)]
fn bfs_drain<W: Weight, const PRED: bool, V: Visitor<W>>(
    adj: &Adjacency<W>,
    dist: &mut [W],
    pred: &mut [u32],
    reached: &mut Vec<u32>,
    cur: &mut Vec<u32>,
    next: &mut Vec<u32>,
    limit: W,
    vis: &mut V,
) {
    // `level` is the depth we are about to emit, so guarding *before* the increment keeps a
    // node at distance exactly `limit` and drops one at `limit + 1` — the same inclusive
    // boundary `dijkstra_drain` has, and the same one scipy has.
    let mut level: W = W::ZERO;
    while !cur.is_empty() && level < limit {
        level = level + W::ONE;
        for &u in cur.iter() {
            let r = adj.row(u);
            for &v in &adj.nbrs[r] {
                let slot = &mut dist[v as usize];
                if level < *slot {
                    if W::is_infinite(*slot) {
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

impl<W: Weight> Visitor<W> for NoVisitor {
    #[inline]
    fn settle(&mut self, _node: u32, _d: W) -> Visit {
        Visit::Expand
    }
}

/// Records every node the search settles, in settle order — i.e. by increasing distance.
///
/// The order is not incidental. It is what lets [`GeodesicGraph::ball`] resolve each node's
/// *nearest source* in one forward pass over the output: a node's predecessor always settles
/// before the node itself, so by the time we reach a node its predecessor's source is already
/// known. Scanning `pred` chains per node instead would be O(depth) apiece.
///
/// `f32`-only, because [`GeodesicGraph`] — its only caller — is.
struct Collect<'a> {
    nodes: &'a mut Vec<u32>,
    dists: &'a mut Vec<f32>,
}

impl Visitor<f32> for Collect<'_> {
    #[inline]
    fn settle(&mut self, node: u32, d: f32) -> Visit {
        self.nodes.push(node);
        self.dists.push(d);
        Visit::Expand
    }
}

/// [`Collect`]'s leaner sibling: settle order, and nothing else.
///
/// For callers that want the order a search discovered nodes in but have no use for the
/// distances — which the search has already left in `Scratch::dist` anyway, so `Collect` would
/// only make them allocate a second buffer holding a copy. Implemented on the vector itself
/// rather than a wrapper struct so callers pass `&mut order` directly, without a block whose
/// only job is to end the wrapper's borrow before the vector is read back.
impl<W: Weight> Visitor<W> for Vec<u32> {
    #[inline]
    fn settle(&mut self, node: u32, _d: W) -> Visit {
        self.push(node);
        Visit::Expand
    }
}

/// Search outwards from one source — the single-source case of [`search_from_many`].
#[inline]
fn search_from<W: Weight, const PRED: bool, V: Visitor<W>>(
    adj: &Adjacency<W>,
    source: u32,
    limit: W,
    vis: &mut V,
    scratch: &mut Scratch<W>,
) {
    search_from_many::<W, PRED, V>(adj, std::slice::from_ref(&source), limit, vis, scratch);
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
fn search_from_many<W: Weight, const PRED: bool, V: Visitor<W>>(
    adj: &Adjacency<W>,
    sources: &[u32],
    limit: W,
    vis: &mut V,
    scratch: &mut Scratch<W>,
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
        if *slot == W::ZERO {
            continue; // a repeated source; it is already on the frontier
        }
        debug_assert!(W::is_infinite(*slot), "scratch was not clean");
        *slot = W::ZERO;
        touched.push(s);
        if weighted {
            heap.push(Reverse(HeapEntry {
                dist_bits: W::ZERO.to_bits(),
                node: s,
            }));
        } else {
            // The two kernels differ in who settles the level-0 frontier: Dijkstra settles on
            // pop, so the sources go through `vis` inside the drain, whereas `bfs_drain` only
            // ever settles nodes it *discovers*. So settle them here — and note the frontier is
            // seeded *after* the visit, which is what gives `Wall` its meaning: a walled source
            // never enters `cur`, so it does not conduct.
            match vis.settle(s, W::ZERO) {
                Visit::Stop => return,
                Visit::Wall => {}
                Visit::Expand => cur.push(s),
            }
        }
    }

    if weighted {
        dijkstra_drain::<W, PRED, V>(adj, dist, pred, touched, heap, limit, vis);
    } else {
        bfs_drain::<W, PRED, V>(adj, dist, pred, touched, cur, next, limit, vis);
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
fn geodesic_matrix_impl<W: Weight>(
    adj: &Adjacency<W>,
    sources: Option<&[u32]>,
    targets: Option<&[u32]>,
    limit: Option<W>,
    threads: Option<usize>,
) -> Array2<W> {
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
        // par_chunks_mut(0) would panic. One dimension is zero, so the buffer is empty.
        return Array2::from_shape_vec((n_rows, n_cols), Vec::new())
            .expect("an empty buffer fits any shape with a zero dimension");
    }

    let (mask, n_targets) = target_mask(targets, n_nodes);
    let limit = limit.unwrap_or(W::INFINITY);

    // -1 is the crate's unreachable sentinel (navis maps it to np.inf on receipt). The gather
    // below writes every cell, so the prefill is defence-in-depth — but a memset is noise next
    // to S searches, and a missed cell surfacing as a plausible 0.0 instead of an obvious -1 is
    // not a trade worth making.
    let mut flat: Vec<W> = vec![W::NEG_ONE; n_rows * n_cols];

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
                    search_from::<W, false, _>(adj, s, limit, &mut tgt, &mut scratch);

                    // Gather at the end rather than writing cells as targets settle: this
                    // preserves the caller's `targets` order exactly and handles duplicate
                    // target ids for free. It is O(n_cols), which we pay regardless — every
                    // output cell has to be written.
                    match mask {
                        None => {
                            for (cell, &d) in row.iter_mut().zip(scratch.dist.iter()) {
                                *cell = if W::is_finite(d) { d } else { W::NEG_ONE };
                            }
                        }
                        Some(_) => {
                            for (cell, &t) in row.iter_mut().zip(targets) {
                                let d = scratch.dist[t as usize];
                                *cell = if W::is_finite(d) { d } else { W::NEG_ONE };
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
fn geodesic_extreme_impl<W: Weight>(
    adj: &Adjacency<W>,
    sources: Option<&[u32]>,
    targets: Option<&[u32]>,
    limit: Option<W>,
    threads: Option<usize>,
    farthest: bool,
) -> (Array1<W>, Array1<i32>) {
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
        return (Array1::from_vec(Vec::new()), Array1::from_vec(Vec::new()));
    }

    let (mask, n_targets) = target_mask(targets, n_nodes);
    let limit = limit.unwrap_or(W::INFINITY);

    let mut dists: Vec<W> = vec![W::NEG_ONE; n_rows];
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
                    search_from::<W, false, _>(adj, s, limit, &mut tgt, &mut scratch);

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
fn geodesic_predecessors_impl<W: Weight>(
    adj: &Adjacency<W>,
    sources: Option<&[u32]>,
    limit: Option<W>,
    threads: Option<usize>,
) -> (Array2<W>, Array2<i32>) {
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
            Array2::from_shape_vec((n_rows, n_nodes), Vec::new())
                .expect("an empty buffer fits any shape with a zero dimension"),
            Array2::zeros((n_rows, n_nodes)),
        );
    }

    let limit = limit.unwrap_or(W::INFINITY);
    let mut dflat: Vec<W> = vec![W::NEG_ONE; n_rows * n_nodes];
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
                    search_from::<W, true, _>(adj, s, limit, &mut tgt, &mut scratch);

                    for (cell, &d) in drow.iter_mut().zip(scratch.dist.iter()) {
                        *cell = if W::is_finite(d) { d } else { W::NEG_ONE };
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
/// A `(sources.len(), targets.len())` matrix at the chosen width. Unreachable pairs —
/// disconnected, or beyond `limit` — are `-1.0`.
///
/// `W` picks the width lengths are accumulated and returned at; see [`Weight`]. Note that
/// `coords` is `f64` either way — that is the coordinates' own precision, and each edge length
/// is computed from them in `f64` and rounded once on the way into the adjacency.
pub fn geodesic_matrix_mesh<W: Weight>(
    faces: ArrayView2<u32>,
    n_vertices: usize,
    coords: Option<ArrayView2<f64>>,
    sources: Option<&[u32]>,
    targets: Option<&[u32]>,
    limit: Option<W>,
    threads: Option<usize>,
) -> Array2<W> {
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
///
/// The weights' own width is the distances' width — `f32` in, `f32` out; `f64` in, `f64` out.
/// See [`Weight`] for which to want.
#[allow(clippy::too_many_arguments)]
pub fn geodesic_matrix_graph<W: Weight>(
    edges: ArrayView2<u32>,
    n_nodes: usize,
    weights: Option<&ArrayView1<W>>,
    directed: bool,
    sources: Option<&[u32]>,
    targets: Option<&[u32]>,
    limit: Option<W>,
    threads: Option<usize>,
) -> Array2<W> {
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
pub fn geodesic_nearest_mesh<W: Weight>(
    faces: ArrayView2<u32>,
    n_vertices: usize,
    coords: Option<ArrayView2<f64>>,
    sources: Option<&[u32]>,
    targets: Option<&[u32]>,
    limit: Option<W>,
    threads: Option<usize>,
) -> (Array1<W>, Array1<i32>) {
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
pub fn geodesic_farthest_mesh<W: Weight>(
    faces: ArrayView2<u32>,
    n_vertices: usize,
    coords: Option<ArrayView2<f64>>,
    sources: Option<&[u32]>,
    targets: Option<&[u32]>,
    limit: Option<W>,
    threads: Option<usize>,
) -> (Array1<W>, Array1<i32>) {
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
pub fn geodesic_predecessors_graph<W: Weight>(
    edges: ArrayView2<u32>,
    n_nodes: usize,
    weights: Option<&ArrayView1<W>>,
    directed: bool,
    sources: Option<&[u32]>,
    limit: Option<W>,
    threads: Option<usize>,
) -> (Array2<W>, Array2<i32>) {
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
pub fn geodesic_path_graph<W: Weight>(
    edges: ArrayView2<u32>,
    n_nodes: usize,
    weights: Option<&ArrayView1<W>>,
    directed: bool,
    source: u32,
    targets: &[u32],
) -> Vec<Vec<u32>> {
    let adj = Adjacency::from_edges(edges, n_nodes, weights, directed);
    geodesic_path_impl(&adj, source, targets)
}

/// `geodesic_path_graph` over a prebuilt adjacency.
fn geodesic_path_impl<W: Weight>(
    adj: &Adjacency<W>,
    source: u32,
    targets: &[u32],
) -> Vec<Vec<u32>> {
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
    search_from::<W, true, _>(adj, source, W::INFINITY, &mut tgt, &mut scratch);

    targets
        .iter()
        .map(|&t| {
            if !W::is_finite(scratch.dist[t as usize]) {
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
pub fn geodesic_clusters<W: Weight>(
    edges: ArrayView2<u32>,
    n_nodes: usize,
    max_dist: W,
    weights: Option<&ArrayView1<W>>,
    seeds: Option<&[u32]>,
) -> (Vec<i32>, usize) {
    let adj = Adjacency::from_edges(edges, n_nodes, weights, false);
    geodesic_clusters_impl(&adj, max_dist, seeds)
}

/// `geodesic_clusters` over a prebuilt adjacency.
///
/// Unlike the free function this inherits the adjacency's direction: given a directed one it
/// grows out-balls, which is a different (if equally well-defined) partition.
fn geodesic_clusters_impl<W: Weight>(
    adj: &Adjacency<W>,
    max_dist: W,
    seeds: Option<&[u32]>,
) -> (Vec<i32>, usize) {
    let n_nodes = adj.n_nodes();
    assert!(
        max_dist >= W::ZERO && W::is_finite(max_dist),
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
        search_from::<W, false, _>(adj, seed, max_dist, &mut tgt, &mut scratch);

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
// Spanning forest
// ---------------------------------------------------------------------------

/// Settle one round of [`parents_from_edges`] and fold what it reached into the running answer.
///
/// Split out because the two rounds — the caller's roots, then whatever they missed — differ
/// only in their seed set, and the book-keeping that turns a settled node into a parent entry
/// is the part that is easy to get subtly wrong.
fn spanning_sweep<W: Weight>(
    adj: &Adjacency<W>,
    seeds: &[u32],
    scratch: &mut Scratch<W>,
    parents: &mut [i32],
    order: &mut Vec<u32>,
) {
    let start = order.len();
    search_from_many::<W, true, _>(adj, seeds, W::INFINITY, order, scratch);
    for &v in &order[start..] {
        let p = scratch.pred[v as usize];
        parents[v as usize] = if p == NO_PRED { -1 } else { p as i32 };
    }
}

/// Orient a graph into a rooted spanning forest — one parent per node, `-1` at the roots.
///
/// The missing half of "I have an edge list and I want a tree". [`minimum_spanning_tree`] picks
/// *which* edges survive; this picks which way they point, which is what turns a bag of
/// undirected edges into something you can walk, root, or write out as SWC. Cycles in the input
/// are fine — each component contributes a spanning tree of itself, so this doubles as the
/// cycle-breaker `networkx.bfs_tree` is usually pressed into.
///
/// One search covers the whole graph. The obvious construction — a shortest-path tree per
/// component — is what a per-source predecessor call gives you, and it costs `O(components x
/// n_nodes)` in *output alone*: on a mesh that shatters into four thousand specks that is a
/// two-gigabyte array to answer a question whose answer is one `n_nodes`-long column. Here the
/// components are swept one after another into that single column, so the cost is `O(V + E)`
/// however finely the graph is fragmented.
///
/// Arguments
/// ---------
/// - `edges`:   (E, 2) array of edges given as node indices. Direction is ignored.
/// - `n_nodes`: Total number of nodes. Nodes named by no edge are isolated roots.
/// - `weights`: Length of each edge, or `None` for hop counts. `None` gives the breadth-first
///   tree; weights give the shortest-path tree, which is a different (and generally deeper)
///   spanning tree. Neither is the minimum spanning tree — for that, run
///   [`minimum_spanning_tree`] first and orient the edges it keeps.
/// - `roots`:   Nodes to root at, or `None` for "the lowest node index in each component" —
///   the same representative [`connected_components_graph`] labels components by. Components
///   holding none of `roots` fall back to that, so the result is always a complete forest.
///   Two roots in the *same* component split it into two trees, which is well defined (each
///   node goes to whichever root is nearer) and occasionally what you want.
///
/// Returns
/// -------
/// - `parents`: `(n_nodes, )` i32, the parent of each node, `-1` for a root.
/// - `order`:   `(n_nodes, )` u32, every node in the order it settled. A node always settles
///   after its parent, so this is a topological order — relabel by it and parents are
///   guaranteed to have lower ids than their children, which is exactly the SWC requirement.
///   It comes free: the search already visits nodes in this order, and computing it afterwards
///   from `parents` would cost another traversal.
///
/// Among equal-length routes the parent is whichever settled first, which is deterministic but
/// otherwise arbitrary — as it is for any spanning tree of a graph with more than one.
pub fn parents_from_edges<W: Weight>(
    edges: ArrayView2<u32>,
    n_nodes: usize,
    weights: Option<&ArrayView1<W>>,
    roots: Option<&[u32]>,
) -> (Array1<i32>, Array1<u32>) {
    let adj = Adjacency::from_edges(edges, n_nodes, weights, false);
    parents_from_edges_impl(&adj, roots)
}

/// `parents_from_edges` over a prebuilt adjacency.
fn parents_from_edges_impl<W: Weight>(
    adj: &Adjacency<W>,
    roots: Option<&[u32]>,
) -> (Array1<i32>, Array1<u32>) {
    let n_nodes = adj.n_nodes();
    // Defaulting to the empty slice rather than to every node: an unrooted component is not an
    // error here, it just falls to the loop below.
    let seeds = resolve(roots, &[], n_nodes, "roots");

    let mut parents: Vec<i32> = vec![-1; n_nodes];
    let mut order: Vec<u32> = Vec::with_capacity(n_nodes);
    if n_nodes == 0 {
        return (Array1::from_vec(parents), Array1::from_vec(order));
    }

    let mut scratch = Scratch::with_pred(n_nodes);

    // Preferred roots first, all in one search, then every node in index order as a fallback —
    // the same two-tier seeding `geodesic_clusters` uses, and for the same reason: it keeps
    // "skip what is already claimed" in one place.
    //
    // The scratch is deliberately *not* reset between rounds, and that is what keeps the whole
    // loop O(V + E) rather than O(components x n_nodes): a reset walks everything the previous
    // sweep touched, so paying it per component is the very cost this function exists to avoid.
    // Skipping it is sound because components are disjoint and every sweep here is unbounded,
    // so a later sweep can only ever reach nodes an earlier one left at `INFINITY`. The same
    // fact makes `dist` the visited flag, so there is no second array to keep in step with it.
    // Both rest on `limit` being `INFINITY` — a *bounded* sweep could stop short of a node a
    // later one would have to revisit, and then neither would hold.
    if !seeds.is_empty() {
        spanning_sweep(adj, seeds, &mut scratch, &mut parents, &mut order);
    }
    for v in 0..n_nodes as u32 {
        if !W::is_finite(scratch.dist[v as usize]) {
            spanning_sweep(adj, &[v], &mut scratch, &mut parents, &mut order);
        }
    }

    (Array1::from_vec(parents), Array1::from_vec(order))
}

// ---------------------------------------------------------------------------
// Minimum spanning tree under the geodesic metric
// ---------------------------------------------------------------------------

/// Minimum spanning tree over a *subset* of nodes, weighted by geodesic distance between them.
///
/// The tree that reconnects a scatter of surviving nodes through the graph they were carved
/// out of — the last step of a skeletonisation, where the mesh has been thinned to a few
/// thousand vertices that must be rejoined along the surface rather than through space.
///
/// The obvious route is to ask for the `k x k` geodesic matrix and hand it to a matrix MST.
/// That materialises `k^2` distances to use `k - 1` of them: 400 MB at k = 10k, before the
/// `O(k^2)` MST itself, and it needs `k` separate searches to fill. This never forms the
/// matrix. Instead — Mehlhorn's construction for the distance network — one multi-source
/// search partitions *every* node by which of `nodes` is nearest, and then each graph edge
/// whose endpoints fall in different cells offers one candidate: joining their two owners at
/// `d(u) + w(u, v) + d(v)`. An MST over those candidates is an MST of the full distance
/// network, so one sweep and one Kruskal replace `k` searches and a dense matrix.
///
/// The MST's edge weights come back exactly equal to the geodesic distances between the pairs
/// they join, so they are usable as lengths and not merely as an ordering.
///
/// Arguments
/// ---------
/// - `adj`:     The graph the distances are measured in.
/// - `nodes`:   The nodes to span, as indices into the graph. Must be distinct.
/// - `limit`:   Do not join nodes farther apart than this. The result is then the MST of the
///   graph on `nodes` keeping only pairs within `limit`, which is a *forest* when that graph
///   is disconnected — the same trade `scipy.sparse.csgraph.dijkstra(limit=...)` offers, except
///   that here it also prunes the sweep, so it buys time rather than merely discarding results.
/// - `threads`: Size of the thread pool, or `None` for all cores.
///
/// Returns
/// -------
/// - `edges`:   `(M, 2)` i64 rows of *positions in `nodes`*, not node indices — so `nodes[edges]`
///   maps back, and any per-node data the caller holds indexes the same way. Ascending by
///   weight, as [`minimum_spanning_tree`].
/// - `weights`: `(M, )` f32 geodesic distance across each of those edges.
///
/// `M` is `nodes.len() - 1` when every node can reach every other within `limit`, and less when
/// they cannot: nodes in different components of the graph are never joined.
fn geodesic_mst_impl<W: Weight>(
    adj: &Adjacency<W>,
    nodes: &[u32],
    limit: Option<W>,
    threads: Option<usize>,
) -> (Array2<i64>, Array1<W>) {
    let n_nodes = adj.n_nodes();

    // Which of `nodes` each graph node *is*, if any — what turns the "nearest source" a search
    // reports (a node index) back into a position in the caller's array.
    let term_of = inverse_index(nodes, n_nodes);
    let limit = limit.unwrap_or(W::INFINITY);
    // A NaN fails `>= ZERO` too — every IEEE comparison against NaN is false — so this one
    // test covers both.
    assert!(
        limit >= W::ZERO,
        "`limit` must be non-negative, got {limit}"
    );
    if nodes.len() < 2 {
        // Nothing to join.
        return (Array2::zeros((0, 2)), Array1::from_vec(Vec::new()));
    }

    // --- One sweep: every node's distance to the nearest of `nodes`, and which one that is.
    //
    // Both answers stay in the scratch and are read from it below. Copying them into arrays of
    // our own would cost two `n_nodes`-sized buffers and a scattered pass to fill them, to hold
    // what the search has already written — the same reason `geodesic_matrix_impl` reads
    // `scratch.dist` directly. Nothing is reset afterwards for the same reason: the scratch is
    // local and dies here, so restoring its invariant is pure writes.
    let mut scratch = Scratch::new(n_nodes);
    scratch.enable_sources(n_nodes);
    let mut settled: Vec<u32> = Vec::new();
    search_from_many::<W, true, _>(adj, nodes, limit, &mut settled, &mut scratch);
    // Fills `scratch.src`, which is what the candidate loop reads; the returned copy is the
    // form `ball` wants and is of no use here.
    scratch.resolve_sources_into(&settled);
    let (dist, owner) = (&scratch.dist, &scratch.src);

    // --- Candidates: one per graph edge that straddles two cells.
    //
    // Interior edges — both ends owned by the same node — are the overwhelming majority on any
    // real mesh and are rejected on a single integer compare against the *owner node*, so
    // `term_of` is only consulted for the rare boundary edge and the candidate list is sized by
    // the cell boundaries rather than by the edge count.
    let weights = adj.weights.as_deref();
    let mut cand: Vec<u32> = Vec::new();
    let mut cand_w: Vec<W> = Vec::new();
    for u in 0..n_nodes as u32 {
        let du = dist[u as usize];
        if !W::is_finite(du) {
            continue; // beyond `limit` of every node, or in a component holding none
        }
        let ou = owner[u as usize];
        let r = adj.row(u);
        for (k, &v) in adj.nbrs[r.clone()].iter().enumerate() {
            // Each undirected edge once. The adjacency stores both arcs, and the two would
            // yield the same candidate.
            if v <= u {
                continue;
            }
            let dv = dist[v as usize];
            if !W::is_finite(dv) || owner[v as usize] == ou {
                continue;
            }
            let w = weights.map_or(W::ONE, |w| w[r.start + k]);
            let cw = du + w + dv;
            // A candidate over `limit` cannot be on the answer: every pair within `limit` still
            // has a candidate at exactly its distance, which is what makes the prune sound.
            if cw > limit {
                continue;
            }
            cand.push(term_of[ou as usize]);
            cand.push(term_of[owner[v as usize] as usize]);
            cand_w.push(cw);
        }
    }

    // --- Kruskal, on the candidates rather than on a matrix. An empty candidate list needs no
    // special case: Kruskal keeps nothing and the gather below yields the empty tree.
    let cand = Array2::from_shape_vec((cand_w.len(), 2), cand)
        .expect("two entries pushed per candidate");
    let cand_w = Array1::from_vec(cand_w);
    let keep = minimum_spanning_tree(
        cand.view(),
        nodes.len(),
        Some(&cand_w.view()),
        false,
        threads,
    );

    let mut out_e: Vec<i64> = Vec::with_capacity(keep.len() * 2);
    let mut out_w: Vec<W> = Vec::with_capacity(keep.len());
    for &i in &keep {
        let e = cand.row(i as usize);
        out_e.push(e[0] as i64);
        out_e.push(e[1] as i64);
        out_w.push(cand_w[i as usize]);
    }
    (
        Array2::from_shape_vec((keep.len(), 2), out_e).expect("two entries pushed per kept edge"),
        Array1::from_vec(out_w),
    )
}

/// Minimum spanning tree over a subset of *mesh vertices*, weighted by geodesic distance.
///
/// See [`geodesic_mst_impl`] for what this computes and why it does not build the distance
/// matrix. Arguments are those of [`geodesic_matrix_mesh`], with `nodes` naming the vertices to
/// span.
pub fn geodesic_mst_mesh<W: Weight>(
    faces: ArrayView2<u32>,
    n_vertices: usize,
    coords: Option<ArrayView2<f64>>,
    nodes: &[u32],
    limit: Option<W>,
    threads: Option<usize>,
) -> (Array2<i64>, Array1<W>) {
    let adj = Adjacency::from_faces(faces, n_vertices, coords);
    geodesic_mst_impl(&adj, nodes, limit, threads)
}

/// Minimum spanning tree over a subset of nodes of an arbitrary graph, by geodesic distance.
///
/// The edge-list form of [`geodesic_mst_mesh`]. Always undirected — a minimum spanning tree of
/// a directed graph is a different problem (an arborescence) with a different algorithm.
pub fn geodesic_mst_graph<W: Weight>(
    edges: ArrayView2<u32>,
    n_nodes: usize,
    weights: Option<&ArrayView1<W>>,
    nodes: &[u32],
    limit: Option<W>,
    threads: Option<usize>,
) -> (Array2<i64>, Array1<W>) {
    let adj = Adjacency::from_edges(edges, n_nodes, weights, false);
    geodesic_mst_impl(&adj, nodes, limit, threads)
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

impl Visitor<f32> for Grow<'_> {
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
///
/// # Width
///
/// `f32` only, unlike the free functions, which are generic over [`Weight`]. The type exists
/// for the calling pattern where the *graph* is large and the queries small — meshes and
/// skeletons — which is exactly where `f32` is the right width and where doubling every
/// node-sized array it holds resident across a whole run would be felt. The incremental
/// farthest-point state alone is three of them. A caller who needs `f64` over a graph they hold
/// wants the free functions, which rebuild the adjacency per call and so are not the thing this
/// type is for.
pub struct GeodesicGraph {
    adj: Adjacency<f32>,
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
    scratch: Scratch<f32>,
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
    fn from_parts(adj: Adjacency<f32>, item_node: Vec<u32>) -> Self {
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
        search_from_many::<f32, true, _>(&self.adj, sources, max_dist, &mut vis, &mut self.scratch);

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
        search_from::<f32, false, _>(
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
            dijkstra_drain::<f32, false, _>(
                adj,
                fps_min,
                no_pred,
                fps_newly_finite,
                &mut scratch.heap,
                f32::INFINITY,
                &mut NoVisitor,
            );
        } else {
            bfs_drain::<f32, false, _>(
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

    /// "No weights", at the default width.
    ///
    /// Everything here is generic over [`Weight`] now, so a bare `None` in the weights slot of
    /// a call whose other arguments are all integers leaves `W` with nothing to infer from.
    /// Naming the width once is tidier than a turbofish on every unweighted test — and the
    /// tests that care about the width say so by using `f64` explicitly.
    const NO_W: Option<&'static ArrayView1<'static, f32>> = None;

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
        let adj = Adjacency::<f32>::from_faces(faces.view(), 4, None);

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
        let adj = Adjacency::<f32>::from_faces(faces.view(), 2, None);
        assert_eq!(&adj.nbrs[adj.row(0)], &[1]);
        assert_eq!(&adj.nbrs[adj.row(1)], &[0]);
    }

    #[test]
    fn arc_weights_are_exactly_symmetric() {
        // An asymmetric weight would silently break d(s,t) == d(t,s), so assert *bit*
        // equality, not approximate equality.
        let (faces, coords) = grid(6, 0.7);
        let adj = Adjacency::<f32>::from_faces(faces.view(), 36, Some(coords.view()));
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
        let d = geodesic_matrix_mesh::<f32>(
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
        let d =
            geodesic_matrix_mesh::<f32>(faces.view(), n * n, None, Some(&[0]), None, None, None);

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
        let d = geodesic_matrix_mesh::<f32>(
            faces.view(),
            6,
            Some(coords.view()),
            None,
            None,
            None,
            None,
        );

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
        let d = geodesic_matrix_mesh::<f32>(faces.view(), 4, None, None, None, None, None);
        assert_eq!(d[[3, 3]], 0.0);
        for j in 0..3 {
            assert_eq!(d[[3, j]], -1.0);
            assert_eq!(d[[j, 3]], -1.0);
        }
    }

    #[test]
    fn full_matrix_is_exactly_symmetric() {
        let (faces, coords) = grid(9, 1.3);
        let d = geodesic_matrix_mesh::<f32>(
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
        let full = geodesic_matrix_mesh::<f32>(
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
        let sub = geodesic_matrix_mesh::<f32>(
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
        let d = geodesic_matrix_mesh::<f32>(
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
        let full = geodesic_matrix_mesh::<f32>(
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

        let at = geodesic_matrix_mesh::<f32>(
            faces.view(),
            n,
            Some(coords.view()),
            Some(&[0]),
            None,
            Some(exact),
            None,
        );
        assert_eq!(at[[0, 3]], exact, "distance == limit must be kept");

        let just_under = geodesic_matrix_mesh::<f32>(
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
        let reference = geodesic_matrix_mesh::<f32>(
            faces.view(),
            121,
            Some(coords.view()),
            None,
            None,
            None,
            Some(1),
        );
        for n in [2usize, 3, 7, 16] {
            let got = geodesic_matrix_mesh::<f32>(
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

        let (dn, nn) = geodesic_nearest_mesh::<f32>(
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

        let (df, nf) = geodesic_farthest_mesh::<f32>(
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

        let full = geodesic_matrix_mesh::<f32>(
            faces.view(),
            n,
            Some(coords.view()),
            None,
            Some(&targets),
            None,
            None,
        );
        let (dn, _) = geodesic_nearest_mesh::<f32>(
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
        let (d, n) = geodesic_nearest_mesh::<f32>(
            faces.view(),
            6,
            None,
            Some(&[0, 1]),
            Some(&[4]),
            None,
            None,
        );
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
        assert_eq!(edges, array![[0u32, 1], [0, 2], [1, 2], [1, 3], [2, 3]]);
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
        assert_eq!(edges, array![[0u32, 0], [0, 1]]);
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
        let d =
            geodesic_matrix_mesh::<f32>(faces.view(), n_nodes, None, Some(&[0]), None, None, None);
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
        assert_eq!(out, array![[0u32, 1]]);
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
        let mst = minimum_spanning_tree(edges.view(), 3, NO_W, false, None);
        assert_eq!(mst.to_vec(), vec![0i64, 1]);
    }

    // -----------------------------------------------------------------------
    // Bridges
    // -----------------------------------------------------------------------

    /// The definition, done the slow way: an edge is a bridge iff dropping it raises the
    /// component count. O(E^2 alpha), so only for small graphs — but it is the property
    /// itself rather than a re-implementation of Tarjan, which is the point of an oracle.
    fn bridges_oracle(edges: ArrayView2<u32>, n_nodes: usize) -> Vec<bool> {
        let base = connected_components_graph(edges, n_nodes);
        let n_base = base.iter().collect::<std::collections::HashSet<_>>().len();
        (0..edges.nrows())
            .map(|drop| {
                let kept: Vec<u32> = (0..edges.nrows())
                    .filter(|&i| i != drop)
                    .flat_map(|i| [edges[[i, 0]], edges[[i, 1]]])
                    .collect();
                let kept = Array2::from_shape_vec((kept.len() / 2, 2), kept).unwrap();
                let c = connected_components_graph(kept.view(), n_nodes);
                c.iter().collect::<std::collections::HashSet<_>>().len() > n_base
            })
            .collect()
    }

    #[test]
    fn every_edge_of_a_tree_is_a_bridge_and_none_of_a_cycle_is() {
        // Path 0-1-2-3: removing any edge splits it.
        let path = array![[0u32, 1], [1, 2], [2, 3]];
        assert_eq!(bridges(path.view(), 4).to_vec(), vec![true, true, true]);

        // Close it into a ring and every edge has an alternative route.
        let ring = array![[0u32, 1], [1, 2], [2, 3], [3, 0]];
        assert_eq!(bridges(ring.view(), 4).to_vec(), vec![false; 4]);
    }

    #[test]
    fn bridges_finds_the_single_link_between_two_cycles() {
        // Two triangles joined by one edge — the classic case, and the only bridge.
        let edges = array![
            [0u32, 1],
            [1, 2],
            [2, 0], // triangle A
            [3, 4],
            [4, 5],
            [5, 3], // triangle B
            [2, 3], // the link
        ];
        let got = bridges(edges.view(), 6);
        assert_eq!(
            got.to_vec(),
            vec![false, false, false, false, false, false, true]
        );
        assert_eq!(got.to_vec(), bridges_oracle(edges.view(), 6));
    }

    #[test]
    fn parallel_edges_are_a_cycle_so_neither_is_a_bridge() {
        // The reason this cannot go through `Adjacency`, which would dedup the pair into one
        // arc and then quite correctly call that one arc a bridge.
        let edges = array![[0u32, 1], [0, 1]];
        assert_eq!(bridges(edges.view(), 2).to_vec(), vec![false, false]);
        assert_eq!(bridges(edges.view(), 2).to_vec(), bridges_oracle(edges.view(), 2));

        // One copy on its own *is* a bridge, so the doubling is what changed the answer.
        let single = array![[0u32, 1]];
        assert_eq!(bridges(single.view(), 2).to_vec(), vec![true]);

        // And a doubled edge hanging off a path leaves the path's own edges bridges.
        let mixed = array![[0u32, 1], [1, 2], [1, 2]];
        assert_eq!(bridges(mixed.view(), 3).to_vec(), vec![true, false, false]);
    }

    #[test]
    fn self_loops_and_isolated_nodes_are_never_bridges() {
        let edges = array![[0u32, 0], [0, 1], [2, 2]];
        assert_eq!(bridges(edges.view(), 4).to_vec(), vec![false, true, false]);
        assert_eq!(bridges(edges.view(), 4).to_vec(), bridges_oracle(edges.view(), 4));
    }

    #[test]
    fn bridges_match_the_brute_force_definition_on_random_graphs() {
        let mut state = 0x5DEECE66Du64;
        for case in 0..60 {
            let n_nodes = 4 + case % 9;
            let n_edges = 3 + case % 14;
            let edges = random_edges(&mut state, n_nodes, n_edges);
            assert_eq!(
                bridges(edges.view(), n_nodes).to_vec(),
                bridges_oracle(edges.view(), n_nodes),
                "case {case}: {edges:?}"
            );
        }
    }

    #[test]
    fn bridges_survive_a_path_deeper_than_the_call_stack() {
        // The recursive formulation of Tarjan blows up here; the explicit stack must not.
        let n = 200_000u32;
        let flat: Vec<u32> = (0..n - 1).flat_map(|i| [i, i + 1]).collect();
        let edges = Array2::from_shape_vec(((n - 1) as usize, 2), flat).unwrap();
        let got = bridges(edges.view(), n as usize);
        assert_eq!(got.len(), (n - 1) as usize);
        assert!(got.iter().all(|&b| b));
    }

    // -----------------------------------------------------------------------
    // Spanning forest
    // -----------------------------------------------------------------------

    /// Check the two invariants every spanning forest must have, whatever the tie-breaks:
    /// the parent links are real edges of the input, and `order` lists parents before children.
    fn check_forest(edges: ArrayView2<u32>, n_nodes: usize, parents: &[i32], order: &[u32]) {
        let present: std::collections::HashSet<(u32, u32)> = edges
            .rows()
            .into_iter()
            .flat_map(|e| [(e[0], e[1]), (e[1], e[0])])
            .collect();
        for (v, &p) in parents.iter().enumerate() {
            if p >= 0 {
                assert!(
                    present.contains(&(p as u32, v as u32)),
                    "parent link {p} -> {v} is not an edge of the graph"
                );
            }
        }

        assert_eq!(order.len(), n_nodes, "`order` must list every node exactly once");
        let mut seen = vec![false; n_nodes];
        for &v in order {
            let v = v as usize;
            assert!(!seen[v], "node {v} appears twice in `order`");
            let p = parents[v];
            assert!(
                p < 0 || seen[p as usize],
                "node {v} settles before its parent {p}"
            );
            seen[v] = true;
        }

        // A forest has exactly one non-root per non-root node, so the edge count pins the
        // component count — which must be the one the DSU agrees on.
        let n_roots = parents.iter().filter(|&&p| p < 0).count();
        let n_comp = connected_components_graph(edges, n_nodes)
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .len();
        assert_eq!(n_roots, n_comp, "one root per component");
    }

    #[test]
    fn parents_from_edges_orients_a_path_away_from_its_lowest_node() {
        // Edges given "backwards" on purpose: orientation must come from the search, not from
        // the order the endpoints happen to be written in.
        let edges = array![[1u32, 0], [2, 1], [3, 2]];
        let (parents, order) = parents_from_edges(edges.view(), 4, NO_W, None);
        assert_eq!(parents.to_vec(), vec![-1, 0, 1, 2]);
        assert_eq!(order.to_vec(), vec![0u32, 1, 2, 3]);
        check_forest(edges.view(), 4, parents.as_slice().unwrap(), order.as_slice().unwrap());
    }

    #[test]
    fn parents_from_edges_breaks_cycles() {
        // A 4-ring. Any spanning tree drops exactly one edge; BFS from 0 drops the far one.
        let edges = array![[0u32, 1], [1, 2], [2, 3], [3, 0]];
        let (parents, order) = parents_from_edges(edges.view(), 4, NO_W, None);
        check_forest(edges.view(), 4, parents.as_slice().unwrap(), order.as_slice().unwrap());
        assert_eq!(parents[0], -1);
        // 1 and 3 are one hop from the root, 2 is two hops via either.
        assert_eq!((parents[1], parents[3]), (0, 0));
        assert!(parents[2] == 1 || parents[2] == 3);
    }

    #[test]
    fn parents_from_edges_roots_each_component_at_its_lowest_node() {
        // Two paths and an isolated node — the same representatives
        // `connected_components_graph` labels components by.
        let edges = array![[2u32, 1], [1, 0], [5, 4]];
        let (parents, order) = parents_from_edges(edges.view(), 7, NO_W, None);
        assert_eq!(parents.to_vec(), vec![-1, 0, 1, -1, -1, 4, -1]);
        // Components are swept in ascending order of their lowest node.
        assert_eq!(order.to_vec(), vec![0u32, 1, 2, 3, 4, 5, 6]);
        check_forest(edges.view(), 7, parents.as_slice().unwrap(), order.as_slice().unwrap());
    }

    #[test]
    fn given_roots_win_and_the_rest_fall_back() {
        // Path 0-1-2-3 rooted at 3: every link reverses.
        let edges = array![[0u32, 1], [1, 2], [2, 3]];
        let (parents, order) = parents_from_edges(edges.view(), 4, NO_W, Some(&[3]));
        assert_eq!(parents.to_vec(), vec![1, 2, 3, -1]);
        assert_eq!(order.to_vec(), vec![3u32, 2, 1, 0]);

        // A root in one component leaves the other to the fallback rule.
        let edges = array![[0u32, 1], [1, 2], [5, 6]];
        let (parents, order) = parents_from_edges(edges.view(), 7, NO_W, Some(&[2]));
        assert_eq!(parents.to_vec(), vec![1, 2, -1, -1, -1, -1, 5]);
        check_forest(edges.view(), 7, parents.as_slice().unwrap(), order.as_slice().unwrap());

        // Two roots inside one component split it — each node goes to the nearer.
        let edges = array![[0u32, 1], [1, 2], [2, 3]];
        let (parents, _) = parents_from_edges(edges.view(), 4, NO_W, Some(&[0, 3]));
        assert_eq!(parents.to_vec(), vec![-1, 0, 3, -1]);
    }

    #[test]
    fn weights_give_the_shortest_path_tree_not_the_hop_tree() {
        // 0-2 direct but expensive; 0-1-2 is two cheap hops. Unweighted picks the direct edge,
        // weighted routes through 1.
        let edges = array![[0u32, 1], [1, 2], [0, 2]];
        let (hops, _) = parents_from_edges(edges.view(), 3, NO_W, None);
        assert_eq!(hops.to_vec(), vec![-1, 0, 0]);

        let w = array![1.0f32, 1.0, 5.0];
        let (weighted, order) = parents_from_edges(edges.view(), 3, Some(&w.view()), None);
        assert_eq!(weighted.to_vec(), vec![-1, 0, 1]);
        assert_eq!(order.to_vec(), vec![0u32, 1, 2]);
    }

    #[test]
    fn parents_from_edges_of_a_shattered_graph_is_one_sweep() {
        // The case a per-source predecessor call cannot serve: thousands of components. Nothing
        // here is timed — the point is that the output is a single n_nodes-long column and the
        // invariants hold across every one of them.
        let n_small = 3000usize;
        let mut flat: Vec<u32> = Vec::new();
        for k in 0..n_small as u32 {
            let base = k * 4;
            flat.extend_from_slice(&[base, base + 1, base + 1, base + 2, base + 2, base + 3]);
        }
        let n_nodes = n_small * 4;
        let edges = Array2::from_shape_vec((flat.len() / 2, 2), flat).unwrap();
        let (parents, order) = parents_from_edges(edges.view(), n_nodes, NO_W, None);
        check_forest(
            edges.view(),
            n_nodes,
            parents.as_slice().unwrap(),
            order.as_slice().unwrap(),
        );
        assert_eq!(parents.iter().filter(|&&p| p < 0).count(), n_small);
    }

    #[test]
    fn parents_from_edges_matches_its_invariants_on_random_graphs() {
        let mut state = 0x1234_5678u64;
        for case in 0..60 {
            let n_nodes = 5 + case % 20;
            let n_edges = 4 + case % 30;
            let edges = random_edges(&mut state, n_nodes, n_edges);
            let w: Array1<f32> = (0..n_edges).map(|_| rng(&mut state) as f32).collect();

            for weights in [None, Some(&w.view())] {
                let (parents, order) = parents_from_edges(edges.view(), n_nodes, weights, None);
                check_forest(
                    edges.view(),
                    n_nodes,
                    parents.as_slice().unwrap(),
                    order.as_slice().unwrap(),
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // Geodesic MST
    // -----------------------------------------------------------------------

    /// What the caller would otherwise do: materialise the k x k geodesic matrix and Kruskal it.
    /// Returns the total weight, which is the invariant an MST pins down — the edge *set* need
    /// not match, since ties have more than one right answer.
    fn dense_mst_weight(adj: &Adjacency<f32>, nodes: &[u32], limit: Option<f32>) -> f64 {
        let d = geodesic_matrix_impl(adj, Some(nodes), Some(nodes), limit, None);
        let k = nodes.len();
        let mut flat: Vec<u32> = Vec::new();
        let mut w: Vec<f32> = Vec::new();
        for i in 0..k {
            for j in i + 1..k {
                let x = d[[i, j]];
                if x >= 0.0 && limit.is_none_or(|l| x <= l) {
                    flat.extend_from_slice(&[i as u32, j as u32]);
                    w.push(x);
                }
            }
        }
        let e = Array2::from_shape_vec((w.len(), 2), flat).unwrap();
        let w = Array1::from_vec(w);
        minimum_spanning_tree(e.view(), k, Some(&w.view()), false, None)
            .iter()
            .map(|&i| w[i as usize] as f64)
            .sum()
    }

    #[test]
    fn geodesic_mst_matches_the_dense_matrix_it_avoids_building() {
        // A 9x9 grid mesh, sampling a scatter of vertices to span. Both the total weight and
        // the individual reported distances must match what the k x k route gives.
        let n = 9usize;
        let (faces, coords) = grid(n, 1.0);
        let adj = Adjacency::<f32>::from_faces(faces.view(), n * n, Some(coords.view()));
        let nodes: Vec<u32> = vec![0, 8, 40, 72, 80, 13, 55];

        let (edges, weights) = geodesic_mst_impl(&adj, &nodes, None, None);
        assert_eq!(edges.nrows(), nodes.len() - 1);

        let total: f64 = weights.iter().map(|&x| x as f64).sum();
        let want = dense_mst_weight(&adj, &nodes, None);
        assert!((total - want).abs() < 1e-4, "{total} vs {want}");

        // Each reported weight is the true geodesic distance between the pair it joins — the
        // construction's candidates are upper bounds in general, so this is not automatic.
        let d = geodesic_matrix_impl(&adj, Some(&nodes), Some(&nodes), None, None);
        for (e, &w) in edges.rows().into_iter().zip(weights.iter()) {
            let truth = d[[e[0] as usize, e[1] as usize]];
            assert!((w - truth).abs() < 1e-4, "{w} vs {truth} for {e:?}");
        }

        // Spanning: the returned edges connect all k nodes.
        let flat: Vec<u32> = edges.iter().map(|&x| x as u32).collect();
        let flat = Array2::from_shape_vec((edges.nrows(), 2), flat).unwrap();
        let comps = connected_components_graph(flat.view(), nodes.len());
        assert!(comps.iter().all(|&c| c == 0));
    }

    #[test]
    fn geodesic_mst_matches_the_dense_route_on_random_graphs() {
        let mut state = 0xC0FFEEu64;
        for case in 0..40 {
            let n_nodes = 12 + case % 18;
            let n_edges = n_nodes * 2;
            let edges = random_edges(&mut state, n_nodes, n_edges);
            let w: Array1<f32> = (0..n_edges)
                .map(|_| (rng(&mut state) * 10.0) as f32)
                .collect();
            let adj = Adjacency::from_edges(edges.view(), n_nodes, Some(&w.view()), false);

            // Every third node, so the subset is scattered rather than contiguous.
            let nodes: Vec<u32> = (0..n_nodes as u32).step_by(3).collect();
            for limit in [None, Some(6.0f32)] {
                let (mst, weights) = geodesic_mst_impl(&adj, &nodes, limit, None);
                let total: f64 = weights.iter().map(|&x| x as f64).sum();
                let want = dense_mst_weight(&adj, &nodes, limit);
                assert!(
                    (total - want).abs() < 1e-3,
                    "case {case} limit {limit:?}: {total} vs {want}"
                );

                // The distances are real, and so is the forest.
                let d = geodesic_matrix_impl(&adj, Some(&nodes), Some(&nodes), None, None);
                for (e, &x) in mst.rows().into_iter().zip(weights.iter()) {
                    let truth = d[[e[0] as usize, e[1] as usize]];
                    assert!((x - truth).abs() < 1e-3, "{x} vs {truth}");
                }
                assert!(mst.nrows() < nodes.len());
            }
        }
    }

    #[test]
    fn geodesic_mst_of_disconnected_nodes_is_a_forest() {
        // Two separate paths; nothing can join them, so we get k - 2 edges.
        let edges = array![[0u32, 1], [1, 2], [5, 6], [6, 7]];
        let w = array![1.0f32, 1.0, 1.0, 1.0];
        let adj = Adjacency::from_edges(edges.view(), 8, Some(&w.view()), false);
        let nodes = [0u32, 2, 5, 7];
        let (mst, weights) = geodesic_mst_impl(&adj, &nodes, None, None);
        assert_eq!(mst.nrows(), 2);
        for &x in weights.iter() {
            assert!((x - 2.0).abs() < 1e-6);
        }

        // `limit` shorter than either path leaves nothing to join at all.
        let (mst, _) = geodesic_mst_impl(&adj, &nodes, Some(1.0), None);
        assert_eq!(mst.nrows(), 0);
    }

    #[test]
    fn geodesic_mst_degenerate_inputs() {
        let edges = array![[0u32, 1], [1, 2]];
        let adj = Adjacency::from_edges(edges.view(), 3, NO_W, false);

        // Fewer than two nodes: nothing to span.
        for nodes in [vec![], vec![1u32]] {
            let (mst, w) = geodesic_mst_impl(&adj, &nodes, None, None);
            assert_eq!((mst.nrows(), w.len()), (0, 0));
        }

        // Unweighted graphs measure in hops.
        let (mst, w) = geodesic_mst_impl(&adj, &[0, 2], None, None);
        assert_eq!(mst.nrows(), 1);
        assert_eq!(w[0], 2.0);
    }

    #[test]
    #[should_panic(expected = "more than once")]
    fn geodesic_mst_rejects_duplicate_nodes() {
        let edges = array![[0u32, 1], [1, 2]];
        let adj = Adjacency::from_edges(edges.view(), 3, NO_W, false);
        geodesic_mst_impl(&adj, &[0, 2, 0], None, None);
    }

    #[test]
    fn geodesic_mst_mesh_and_graph_agree() {
        // The mesh entry point is the graph one over the mesh's unique edges, so given the same
        // metric they must return the same tree.
        let n = 7usize;
        let (faces, coords) = grid(n, 1.0);
        let n_verts = n * n;
        let nodes: Vec<u32> = vec![0, 6, 24, 42, 48];

        let (mesh_e, mesh_w) = geodesic_mst_mesh::<f32>(
            faces.view(),
            n_verts,
            Some(coords.view()),
            &nodes,
            None,
            None,
        );

        let (edges, _, _, lengths) =
            unique_edges(faces.view(), Some(coords.view()), false, false, None);
        let edges: Array2<u32> = edges.mapv(|v| v as u32);
        let w: Array1<f32> = lengths.unwrap().mapv(|x| x as f32);
        let (graph_e, graph_w) =
            geodesic_mst_graph(edges.view(), n_verts, Some(&w.view()), &nodes, None, None);

        assert_eq!(mesh_e, graph_e);
        for (a, b) in mesh_w.iter().zip(graph_w.iter()) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
    }

    /// A deterministic xorshift, so the fuzzing above needs no rand dependency.
    fn rng(state: &mut u64) -> f64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        (*state >> 11) as f64 / (1u64 << 53) as f64
    }

    /// A random edge list over `n_nodes`. Self-loops and parallel edges are left in on
    /// purpose — they are the cases the primitives above have to get right.
    fn random_edges(state: &mut u64, n_nodes: usize, n_edges: usize) -> Array2<u32> {
        let flat: Vec<u32> = (0..n_edges * 2)
            .map(|_| (rng(state) * n_nodes as f64) as u32)
            .collect();
        Array2::from_shape_vec((n_edges, 2), flat).unwrap()
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
            minimum_spanning_tree(edges.view(), 3, NO_W, false, None).len(),
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
            geodesic_predecessors_graph(edges.view(), 7, NO_W, false, Some(&[0]), None, None);

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
        let paths = geodesic_path_graph(edges.view(), 4, NO_W, false, 0, &[1, 3, 0]);
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
                NO_W,
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

    // -----------------------------------------------------------------------
    // Weight width
    // -----------------------------------------------------------------------

    /// `f64` weights and `f32` weights must give the *same* answer wherever `f32` is exact,
    /// or the two instantiations are not the same algorithm.
    ///
    /// Small integer weights are the case where that holds: every weight, and every sum of
    /// them along any path in these graphs, lands exactly on both widths, so the two matrices
    /// have to agree to the bit — not merely to a tolerance. Random graphs, so the agreement
    /// is tested across cycles, parallel edges, self-loops and disconnected components rather
    /// than on one hand-picked shape.
    #[test]
    fn the_two_widths_agree_wherever_f32_is_exact() {
        let mut state = 0x5eed_1234_u64;
        for &(n_nodes, n_edges) in &[(1usize, 0usize), (6, 8), (40, 90), (120, 400)] {
            let edges = random_edges(&mut state, n_nodes, n_edges);
            // 1..=16: integers, so exact at both widths, and small enough that no path sum
            // can leave f32's exactly-representable range.
            let w32: Array1<f32> = (0..n_edges)
                .map(|_| (rng(&mut state) * 16.0) as u32 as f32 + 1.0)
                .collect();
            let w64: Array1<f64> = w32.iter().map(|&x| x as f64).collect();

            let d32 = geodesic_matrix_graph(
                edges.view(),
                n_nodes,
                Some(&w32.view()),
                false,
                None,
                None,
                None,
                None,
            );
            let d64 = geodesic_matrix_graph(
                edges.view(),
                n_nodes,
                Some(&w64.view()),
                false,
                None,
                None,
                None,
                None,
            );
            assert_eq!(d32.shape(), d64.shape());
            for (a, b) in d32.iter().zip(d64.iter()) {
                assert_eq!(*a as f64, *b, "{n_nodes} nodes / {n_edges} edges");
            }

            // The predecessor trees have to match too — same relaxation order, so the same
            // route is chosen among equal-length ones.
            let (_, p32) = geodesic_predecessors_graph(
                edges.view(),
                n_nodes,
                Some(&w32.view()),
                false,
                None,
                None,
                None,
            );
            let (_, p64) = geodesic_predecessors_graph(
                edges.view(),
                n_nodes,
                Some(&w64.view()),
                false,
                None,
                None,
                None,
            );
            assert_eq!(p32, p64);

            // As do the derived structures: spanning forest, clusters, geodesic MST.
            assert_eq!(
                parents_from_edges(edges.view(), n_nodes, Some(&w32.view()), None),
                parents_from_edges(edges.view(), n_nodes, Some(&w64.view()), None)
            );
            assert_eq!(
                geodesic_clusters(edges.view(), n_nodes, 5.0f32, Some(&w32.view()), None),
                geodesic_clusters(edges.view(), n_nodes, 5.0f64, Some(&w64.view()), None)
            );

            let nodes: Vec<u32> = (0..n_nodes as u32).step_by(3).collect();
            let (e32, mw32) =
                geodesic_mst_graph(edges.view(), n_nodes, Some(&w32.view()), &nodes, None, None);
            let (e64, mw64) =
                geodesic_mst_graph(edges.view(), n_nodes, Some(&w64.view()), &nodes, None, None);
            assert_eq!(e32, e64);
            for (a, b) in mw32.iter().zip(mw64.iter()) {
                assert_eq!(*a as f64, *b);
            }
        }
    }

    /// Hop counts do not depend on the width at all: BFS emits integers, which both widths
    /// hold exactly at any depth a graph can reach.
    #[test]
    fn unweighted_searches_are_width_independent() {
        let (faces, _) = grid(9, 1.0);
        let d32 = geodesic_matrix_mesh::<f32>(faces.view(), 81, None, None, None, None, None);
        let d64 = geodesic_matrix_mesh::<f64>(faces.view(), 81, None, None, None, None, None);
        for (a, b) in d32.iter().zip(d64.iter()) {
            assert_eq!(*a as f64, *b);
        }
    }

    /// The point of the wider width: a long accumulation drifts at `f32` and does not at `f64`.
    ///
    /// The grid has a closed-form metric (see [`grid`]), so this is measured against an exact
    /// answer rather than against the other width — otherwise "closer" would only mean "they
    /// differ". Two roundings feed the `f32` drift: the edge lengths themselves, computed in
    /// `f64` from the coordinates and narrowed on the way in, and one addition per hop.
    #[test]
    fn f64_tracks_the_closed_form_where_f32_drifts() {
        let n = 48;
        let s = 0.3f64;
        let (faces, coords) = grid(n, s);

        let exact = |i: usize, j: usize| {
            s * (2f64.sqrt() * i.min(j) as f64 + (i as isize - j as isize).abs() as f64)
        };
        let worst = |d: &dyn Fn(usize, usize) -> f64| {
            let mut worst = 0f64;
            for i in 0..n {
                for j in 0..n {
                    worst = worst.max((d(i, j) - exact(i, j)).abs());
                }
            }
            worst
        };

        let d32 = geodesic_matrix_mesh::<f32>(
            faces.view(),
            n * n,
            Some(coords.view()),
            Some(&[0]),
            None,
            None,
            None,
        );
        let d64 = geodesic_matrix_mesh::<f64>(
            faces.view(),
            n * n,
            Some(coords.view()),
            Some(&[0]),
            None,
            None,
            None,
        );
        let e32 = worst(&|i, j| d32[[0, i * n + j]] as f64);
        let e64 = worst(&|i, j| d64[[0, i * n + j]]);

        // The far corner is ~20 units out over ~70 hops, where an f32 ulp is ~2e-6.
        assert!(e64 < 1e-12, "f64 should track the closed form: {e64}");
        assert!(
            e32 > 1e-7,
            "fixture assumption: f32 should visibly drift, got {e32}"
        );
        assert!(e64 < e32, "{e64} vs {e32}");
    }

    /// A shortest path `f32` cannot see at all.
    ///
    /// Two routes from 0 to 2: the direct edge at `1 + 1e-10`, and the two-hop route at
    /// `1 + 1e-8`. An `f32` ulp at 1.0 is ~1.2e-7, so at that width every one of those numbers
    /// *is* 1.0 and the two routes tie; at `f64` the direct edge is genuinely shorter. This is
    /// the case where the width changes the answer rather than merely the container it comes
    /// back in.
    #[test]
    fn f64_resolves_a_difference_f32_rounds_away() {
        let edges = array![[0u32, 1], [1, 2], [0, 2]];
        let w64 = array![1.0f64, 1e-8, 1.0000000001];
        let w32: Array1<f32> = w64.iter().map(|&x| x as f32).collect();

        let d32 = geodesic_matrix_graph(
            edges.view(),
            3,
            Some(&w32.view()),
            false,
            None,
            None,
            None,
            None,
        );
        let d64 = geodesic_matrix_graph(
            edges.view(),
            3,
            Some(&w64.view()),
            false,
            None,
            None,
            None,
            None,
        );

        // Everything collapses to 1.0 at f32 — including the weights, before the search runs.
        assert_eq!(d32[[0, 2]], 1.0f32);
        // At f64 the direct edge wins, and by exactly what it should.
        assert_eq!(d64[[0, 2]], 1.0000000001f64);
        assert!(d64[[0, 2]] < 1.0 + 1e-8, "the two-hop route should lose");
    }

    /// The conventions the drivers are built on hold at `f64` too: the `-1` unreachable
    /// sentinel, the inclusive `limit` boundary, and degenerate shapes.
    #[test]
    fn f64_keeps_the_module_conventions() {
        // Two components, so there are genuinely unreachable pairs.
        let edges = array![[0u32, 1], [2, 3]];
        let w = array![1.5f64, 2.5];
        let d = geodesic_matrix_graph(
            edges.view(),
            4,
            Some(&w.view()),
            false,
            None,
            None,
            None,
            None,
        );
        assert_eq!(d[[0, 1]], 1.5);
        assert_eq!(d[[0, 2]], -1.0);
        assert_eq!(d[[2, 3]], 2.5);

        // `limit` keeps a node at exactly the bound and drops anything past it.
        let at = geodesic_matrix_graph(
            edges.view(),
            4,
            Some(&w.view()),
            false,
            None,
            None,
            Some(1.5f64),
            None,
        );
        assert_eq!(at[[0, 1]], 1.5);
        assert_eq!(at[[2, 3]], -1.0);

        // Empty source or target sets give an empty matrix, not a panic.
        let empty = geodesic_matrix_graph(
            edges.view(),
            4,
            Some(&w.view()),
            false,
            Some(&[]),
            None,
            None,
            None,
        );
        assert_eq!(empty.shape(), &[0, 4]);

        // As does an empty graph through the predecessor driver.
        let none = Array2::<u32>::zeros((0, 2));
        let (dist, pred) = geodesic_predecessors_graph(
            none.view(),
            0,
            None::<&ArrayView1<f64>>,
            false,
            None,
            None,
            None,
        );
        assert_eq!((dist.shape(), pred.shape()), (&[0, 0][..], &[0, 0][..]));
    }
}
