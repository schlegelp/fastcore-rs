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

    // Walk every face and union the three vertices.
    for face in faces.rows() {
        let a = face[0];
        let b = face[1];
        let c = face[2];

        // Union a–b
        let ra = find(&mut parent, a);
        let rb = find(&mut parent, b);
        if ra != rb {
            // Attach larger root to smaller root so root IDs are consistent.
            if ra < rb {
                parent[rb as usize] = ra;
            } else {
                parent[ra as usize] = rb;
            }
        }

        // Union a–c  (re-find ra since it may have changed)
        let ra2 = find(&mut parent, a);
        let rc = find(&mut parent, c);
        if ra2 != rc {
            if ra2 < rc {
                parent[rc as usize] = ra2;
            } else {
                parent[ra2 as usize] = rc;
            }
        }
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

/// Per-worker scratch. Allocated once per rayon chunk and reused across every source in it.
struct Scratch {
    /// Tentative distance per node. `INFINITY` = not reached.
    /// Invariant: all-`INFINITY` on entry to and exit from every search.
    dist: Vec<f32>,
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
            touched: Vec::new(),
            heap: BinaryHeap::new(),
            cur: Vec::new(),
            next: Vec::new(),
        }
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
        if self.touched.len() * 4 >= self.dist.len() {
            self.dist.fill(f32::INFINITY);
        } else {
            for &v in &self.touched {
                self.dist[v as usize] = f32::INFINITY;
            }
        }
        self.touched.clear();
        self.heap.clear(); // retains capacity — no realloc churn on the next source
        self.cur.clear();
        self.next.clear();
    }
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

impl<'a> Targets<'a> {
    /// Note a settled node. Returns `true` if the search can stop.
    #[inline]
    fn settle(&mut self, node: u32, d: f32) -> bool {
        let is_target = match self.mask {
            Some(m) => m[node as usize],
            None => true,
        };
        if !is_target || node == self.exclude {
            return false;
        }
        if self.first.is_none() {
            self.first = Some((node, d));
            if self.stop_at_first {
                return true;
            }
        }
        self.last = Some((node, d));
        self.remaining = self.remaining.saturating_sub(1);
        self.remaining == 0
    }
}

/// Single-source Dijkstra over `adj`, leaving distances in `scratch.dist`.
///
/// Two things `scipy.sparse.csgraph.dijkstra` structurally cannot do, both here:
/// stop once every target has settled (scipy has no notion of targets — it materialises all N
/// columns and lets you slice afterwards), and prune at *relaxation* on `limit` so the heap
/// never grows past the ball of radius `limit`.
fn dijkstra_from(
    adj: &Adjacency,
    source: u32,
    limit: f32,
    tgt: &mut Targets,
    scratch: &mut Scratch,
) {
    let Scratch {
        dist,
        touched,
        heap,
        ..
    } = scratch;
    let weights = adj
        .weights
        .as_deref()
        .expect("dijkstra_from requires a weighted adjacency");

    dist[source as usize] = 0.0;
    touched.push(source);
    heap.push(Reverse(HeapEntry {
        dist_bits: 0,
        node: source,
    }));

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

        if tgt.settle(u, du) {
            return;
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
                    touched.push(v);
                }
                *slot = nd;
                heap.push(Reverse(HeapEntry {
                    dist_bits: nd.to_bits(),
                    node: v,
                }));
            }
        }
    }
}

/// Single-source BFS — the unweighted (hop-count) path.
///
/// Unit weights make the frontier monotone by construction, so this needs no priority queue at
/// all: two ping-pong frontiers give O(V + E) with no sift, no stale entries and no float
/// compares. Routing the unweighted case through `dijkstra_from` would be several times slower
/// for no reason.
///
/// Hop counts are integers and exact in f32 up to 2^24; no mesh has a 16M-hop path.
fn bfs_from(adj: &Adjacency, source: u32, limit: f32, tgt: &mut Targets, scratch: &mut Scratch) {
    let Scratch {
        dist,
        touched,
        cur,
        next,
        ..
    } = scratch;

    dist[source as usize] = 0.0;
    touched.push(source);
    cur.push(source);
    if tgt.settle(source, 0.0) {
        return;
    }

    // `level` is the depth we are about to emit, so guarding *before* the increment keeps a
    // node at distance exactly `limit` and drops one at `limit + 1` — the same inclusive
    // boundary `dijkstra_from` has, and the same one scipy has.
    let mut level: f32 = 0.0;
    while !cur.is_empty() && level < limit {
        level += 1.0;
        for &u in cur.iter() {
            let r = adj.row(u);
            for &v in &adj.nbrs[r] {
                let slot = &mut dist[v as usize];
                if slot.is_infinite() {
                    *slot = level;
                    touched.push(v);
                    next.push(v);
                    if tgt.settle(v, level) {
                        return;
                    }
                }
            }
        }
        std::mem::swap(cur, next);
        next.clear();
    }
}

#[inline]
fn search_from(adj: &Adjacency, source: u32, limit: f32, tgt: &mut Targets, scratch: &mut Scratch) {
    if adj.weights.is_some() {
        dijkstra_from(adj, source, limit, tgt, scratch);
    } else {
        bfs_from(adj, source, limit, tgt, scratch);
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
                    search_from(adj, s, limit, &mut tgt, &mut scratch);

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
                    search_from(adj, s, limit, &mut tgt, &mut scratch);

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
}
