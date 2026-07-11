//! Topology repair primitives for skeletons ("healing").
//!
//! A skeleton is stored the same way as everywhere else in `fastcore`: as a
//! `parents` array of node *indices* where roots are negative. A "fragmented"
//! skeleton is one whose parent/child edges form more than one connected
//! component. Healing reconnects those fragments into a single rooted tree by
//! inserting a minimal-length set of bridging edges between the spatially
//! closest fragments.
//!
//! This module exposes two small, neuron-agnostic primitives that together
//! cover the expensive part of `navis.heal_skeleton`:
//!
//! * [`stitch_fragments`] — given node coordinates and a per-node component
//!   label, return the inter-fragment bridge edges (node-index pairs) that
//!   connect the fragments with minimal total added length: a Boruvka MST over
//!   the fragments driven by a single R-tree, where each node walks its nearest
//!   neighbours until it reaches a different fragment. The result is a true
//!   minimum spanning tree of the "closest pair between fragments" graph, so the
//!   total added cable matches navis' healing exactly.
//! * [`reroot_rewire`] — given the original topology plus a set of new
//!   undirected edges and a preferred root, regenerate a valid `parents` array
//!   via BFS. This replaces navis' networkx `dfs_tree`/`minimum_spanning_tree`
//!   reroot.
//!
//! The neuron-specific policy (which nodes may be bridged, `max_dist`,
//! `min_size`, dropping disconnected fragments, …) is left to the caller
//! (navis), matching the general-interface convention of the rest of the crate.

use ndarray::{Array, Array1, ArrayView1, ArrayView2};
use rayon::prelude::*;
use rstar::{PointDistance, RTree, RTreeObject, AABB};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

/// Atomically lower `cell` to `val` if `val` is smaller (f64 stored as bits).
#[inline]
fn atomic_min_f64(cell: &AtomicU64, val: f64) {
    let mut cur = cell.load(Ordering::Relaxed);
    loop {
        if val >= f64::from_bits(cur) {
            return;
        }
        match cell.compare_exchange_weak(
            cur,
            val.to_bits(),
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => return,
            Err(actual) => cur = actual,
        }
    }
}

/// A candidate node in the spatial index: its coordinates plus its node index.
///
/// Generic over the dimensionality `D` so the same search can run in plain 3D
/// space or in an augmented space with extra feature columns (see
/// [`stitch_fragments`]).
struct Pt<const D: usize> {
    xyz: [f64; D],
    idx: u32,
}

impl<const D: usize> RTreeObject for Pt<D> {
    type Envelope = AABB<[f64; D]>;
    fn envelope(&self) -> Self::Envelope {
        AABB::from_point(self.xyz)
    }
}

impl<const D: usize> PointDistance for Pt<D> {
    fn distance_2(&self, p: &[f64; D]) -> f64 {
        (0..D).map(|k| (self.xyz[k] - p[k]).powi(2)).sum()
    }
}

/// Minimal Union-Find (disjoint-set) with path halving and union by size.
struct UnionFind {
    parent: Vec<u32>,
    size: Vec<u32>,
    n_sets: usize,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        UnionFind {
            parent: (0..n as u32).collect(),
            size: vec![1; n],
            n_sets: n,
        }
    }

    fn find(&mut self, mut x: u32) -> u32 {
        while self.parent[x as usize] != x {
            // Path halving: point `x` at its grandparent.
            let gp = self.parent[self.parent[x as usize] as usize];
            self.parent[x as usize] = gp;
            x = gp;
        }
        x
    }

    /// Union the sets containing `a` and `b`. Returns `true` if they were
    /// previously in different sets (i.e. a merge actually happened).
    fn union(&mut self, a: u32, b: u32) -> bool {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return false;
        }
        // Union by size; smaller root attaches to larger.
        let (small, large) = if self.size[ra as usize] < self.size[rb as usize] {
            (ra, rb)
        } else {
            (rb, ra)
        };
        self.parent[small as usize] = large;
        self.size[large as usize] += self.size[small as usize];
        self.n_sets -= 1;
        true
    }
}

/// A candidate bridge found during a Boruvka round:
/// `(my_super_root, squared_distance, node_a, node_b)`.
type Candidate = (u32, f64, u32, u32);

/// Reduce a round's candidate edges to the single cheapest outgoing edge per
/// super-component, then add them via union-find (skipping edges whose endpoints
/// are already joined). Returns the number of merges actually performed.
fn reduce_and_union(
    found: Vec<Candidate>,
    node_comp: &[i32],
    uf: &mut UnionFind,
    bridges: &mut Vec<(i32, i32, f32)>,
) -> usize {
    if found.is_empty() {
        return 0;
    }
    let mut best: HashMap<u32, (f64, u32, u32)> = HashMap::new();
    for (root, d2, a, b) in found {
        best.entry(root)
            .and_modify(|e| {
                if d2 < e.0 {
                    *e = (d2, a, b);
                }
            })
            .or_insert((d2, a, b));
    }
    // Add cheapest-first for deterministic, reproducible bridges.
    let mut edges: Vec<(f64, u32, u32)> = best.into_values().collect();
    edges.sort_by(|x, y| {
        x.0.partial_cmp(&y.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(x.1.cmp(&y.1))
            .then(x.2.cmp(&y.2))
    });

    let mut merges = 0;
    for (d2, a, b) in edges {
        let ca = node_comp[a as usize] as u32;
        let cb = node_comp[b as usize] as u32;
        if uf.union(ca, cb) {
            bridges.push((a as i32, b as i32, (d2.sqrt()) as f32));
            merges += 1;
        }
    }
    merges
}

/// Compute the minimal-length set of bridge edges that connect the fragments of
/// a skeleton into a single tree.
///
/// Arguments:
///
/// - `coords`: `(N, D)` array of node coordinates, with `D` either 3 or 4. The
///   search is a plain nearest-neighbour search in `D`-dimensional Euclidean
///   space, so a 4th column acts as an extra feature that pulls nodes with
///   similar values together — pass a (scaled) node radius there to prefer
///   bridging fragments of similar calibre. The scaling is the caller's job: a
///   larger 4th column makes that feature count for more. Note that `max_dist`
///   is then also measured in that augmented space.
/// - `components`: length-`N` array giving each node's connected-component
///   label (e.g. the output of [`crate::dag::connected_components`]). Labels may
///   be any integers; only their equality matters.
/// - `mask`: optional length-`N` boolean array marking the nodes that are
///   eligible to act as bridge endpoints (e.g. only leaves/roots, or a
///   user-supplied subset). `None` means every node is eligible. Nodes that are
///   not eligible are ignored entirely — a fragment with no eligible node cannot
///   be bridged.
/// - `max_dist`: upper bound on the length of any single bridge. Use
///   `f64::INFINITY` for no bound. Fragment pairs whose closest eligible nodes
///   are farther apart than this are left unconnected.
///
/// Returns:
///
/// A vector of `(node_a, node_b, distance)` bridges, where `node_a`/`node_b` are
/// node indices and `distance` is the Euclidean bridge length (in the same `D`
/// dimensions as `coords`). At most `(#components - 1)` bridges are returned;
/// fewer if `max_dist` prevents some fragments from being connected.
///
/// Panics if `coords` has anything other than 3 or 4 columns.
pub fn stitch_fragments(
    coords: &ArrayView2<f64>,
    components: &ArrayView1<i32>,
    mask: &Option<Array1<bool>>,
    max_dist: f64,
) -> Vec<(i32, i32, f32)> {
    match coords.ncols() {
        3 => stitch_impl::<3>(coords, components, mask, max_dist),
        4 => stitch_impl::<4>(coords, components, mask, max_dist),
        d => panic!("`coords` must have 3 or 4 columns, got {d}"),
    }
}

/// The actual stitching, monomorphised for a given dimensionality `D`.
fn stitch_impl<const D: usize>(
    coords: &ArrayView2<f64>,
    components: &ArrayView1<i32>,
    mask: &Option<Array1<bool>>,
    max_dist: f64,
) -> Vec<(i32, i32, f32)> {
    let n = coords.nrows();

    // Pull row `i` out as a fixed-size point.
    let point = |i: usize| -> [f64; D] { std::array::from_fn(|k| coords[[i, k]]) };

    // 1. Determine the eligible candidate nodes.
    let candidates: Vec<usize> = match mask {
        Some(m) => (0..n).filter(|&i| m[i]).collect(),
        None => (0..n).collect(),
    };
    if candidates.len() < 2 {
        return Vec::new();
    }

    // 2. Compress the component labels of the candidate nodes into 0..C and
    //    record, per node index, which compressed component it belongs to
    //    (`-1` for nodes that are not candidates).
    let mut label_to_comp: HashMap<i32, u32> = HashMap::new();
    let mut node_comp: Vec<i32> = vec![-1; n];
    for &i in &candidates {
        let next = label_to_comp.len() as u32;
        let c = *label_to_comp.entry(components[i]).or_insert(next);
        node_comp[i] = c as i32;
    }
    let n_comps = label_to_comp.len();
    if n_comps < 2 {
        // All eligible nodes are already in one component -> nothing to stitch.
        return Vec::new();
    }

    // 3. Build a single R-tree over the candidate nodes (items carry node index).
    let tree: RTree<Pt<D>> = RTree::bulk_load(
        candidates
            .iter()
            .map(|&i| Pt {
                xyz: point(i),
                idx: i as u32,
            })
            .collect(),
    );

    let max_dist_sq = if max_dist.is_finite() {
        max_dist * max_dist
    } else {
        f64::INFINITY
    };

    // 4. Boruvka's algorithm over the components: each round every
    //    super-component contributes its single *minimum* outgoing edge; add
    //    those, merge, repeat until one super-component remains. Because every
    //    minimum spanning tree of the same fragment graph has the same total
    //    weight, using true per-component minima makes the result identical (in
    //    length) to navis' MST — not merely a valid healing.
    //
    //    Each node finds its nearest cross-component neighbour by walking the
    //    R-tree's lazy nearest-neighbour iterator in ascending distance, stopping
    //    at the first node in a different super-component.
    //
    //    The walk is pruned by a per-super-component bound: the best cross-edge
    //    any of its nodes has found so far this round. Once a node's iterator
    //    passes that distance it cannot beat the bound, so it cannot supply the
    //    component's minimum and is abandoned. This keeps the result exact (the
    //    node holding the true minimum is never pruned before reaching it) while
    //    confining nodes deep inside a fragment — which would otherwise walk
    //    across the whole fragment — to a small ball.
    let mut uf = UnionFind::new(n_comps);
    let mut bridges: Vec<(i32, i32, f32)> = Vec::with_capacity(n_comps - 1);

    // Reused across rounds to avoid reallocating.
    const SEED_STEPS: usize = 8;

    while uf.n_sets > 1 {
        // Snapshot the current super-root of every candidate node so the
        // parallel sweep below is purely read-only.
        let comp_root: Vec<u32> = (0..n_comps as u32).map(|c| uf.find(c)).collect();
        let mut node_super_root: Vec<u32> = vec![u32::MAX; n];
        for &i in &candidates {
            node_super_root[i] = comp_root[node_comp[i] as usize];
        }

        // Per-super-component pruning bound (squared distance).
        let bound: Vec<AtomicU64> = (0..n_comps)
            .map(|_| AtomicU64::new(max_dist_sq.to_bits()))
            .collect();

        // Seed pass: a cheap, bounded peek at each node's few nearest neighbours.
        // Any cross-component hit lowers its component's bound, so the exact pass
        // below starts with a tight radius instead of searching from infinity.
        candidates.par_iter().for_each(|&i| {
            let my_root = node_super_root[i];
            let q = point(i);
            for pt in tree.nearest_neighbor_iter(&q).take(SEED_STEPS) {
                if node_super_root[pt.idx as usize] != my_root {
                    atomic_min_f64(&bound[my_root as usize], pt.distance_2(&q));
                    return;
                }
            }
        });

        // Exact pass: walk until the first cross-component neighbour, abandoning
        // the walk once it can no longer beat the component's bound.
        let found: Vec<Candidate> = candidates
            .par_iter()
            .filter_map(|&i| {
                let my_root = node_super_root[i];
                let cell = &bound[my_root as usize];
                let q = point(i);
                for pt in tree.nearest_neighbor_iter(&q) {
                    let d2 = pt.distance_2(&q);
                    // Cannot beat the component's best (nor exceed `max_dist`).
                    if d2 > f64::from_bits(cell.load(Ordering::Relaxed)) {
                        return None;
                    }
                    if node_super_root[pt.idx as usize] != my_root {
                        atomic_min_f64(cell, d2);
                        return Some((my_root, d2, i as u32, pt.idx));
                    }
                }
                None
            })
            .collect();

        if reduce_and_union(found, &node_comp, &mut uf, &mut bridges) == 0 {
            // No fragment can reach another within `max_dist`.
            break;
        }
    }

    bridges
}

/// Regenerate a `parents` array after adding a set of undirected edges.
///
/// This turns an edited edge set back into a valid rooted forest: it builds the
/// undirected adjacency from the original child->parent edges plus `new_edges`,
/// then BFS-orients everything away from a root.
///
/// Arguments:
///
/// - `parents`: the original topology as a length-`N` array of parent indices
///   (roots are negative).
/// - `new_edges`: `(M, 2)` array of undirected edges (node index pairs) to add,
///   e.g. the bridges from [`stitch_fragments`].
/// - `root`: preferred root node index. Its whole component is rooted here; pass
///   a negative value to auto-pick (lowest index) for every component.
///
/// Returns:
///
/// A length-`N` array of new parent indices (roots negative). Any component not
/// reachable from `root` is rooted at its lowest-index node, so the result is
/// always a valid forest even if `new_edges` did not fully connect the skeleton.
///
/// A plain BFS is sufficient (no MST needed): back-edges are simply ignored, so
/// even redundant `new_edges` that would introduce a cycle yield a valid tree.
pub fn reroot_rewire(
    parents: &ArrayView1<i32>,
    new_edges: &ArrayView2<i32>,
    root: i32,
) -> Array1<i32> {
    let n = parents.len();

    // Build undirected adjacency from the original edges + the new edges.
    let mut adj: Vec<Vec<i32>> = vec![Vec::new(); n];
    for (i, &p) in parents.iter().enumerate() {
        if p >= 0 {
            adj[i].push(p);
            adj[p as usize].push(i as i32);
        }
    }
    for e in new_edges.rows() {
        let a = e[0];
        let b = e[1];
        if a >= 0 && b >= 0 {
            adj[a as usize].push(b);
            adj[b as usize].push(a);
        }
    }

    let mut new_parents: Array1<i32> = Array::from_elem(n, -1);
    let mut visited: Vec<bool> = vec![false; n];
    let mut queue: Vec<i32> = Vec::new();

    // BFS that roots the component of `start` at `start`.
    let bfs = |start: usize,
                   adj: &Vec<Vec<i32>>,
                   visited: &mut Vec<bool>,
                   new_parents: &mut Array1<i32>,
                   queue: &mut Vec<i32>| {
        visited[start] = true;
        new_parents[start] = -1;
        queue.clear();
        queue.push(start as i32);
        let mut head = 0;
        while head < queue.len() {
            let node = queue[head] as usize;
            head += 1;
            for &nb in &adj[node] {
                if !visited[nb as usize] {
                    visited[nb as usize] = true;
                    new_parents[nb as usize] = node as i32;
                    queue.push(nb);
                }
            }
        }
    };

    // Seed the preferred root's component first (if given and in range).
    if root >= 0 && (root as usize) < n {
        bfs(
            root as usize,
            &adj,
            &mut visited,
            &mut new_parents,
            &mut queue,
        );
    }

    // Root every remaining component at its lowest-index node.
    for i in 0..n {
        if !visited[i] {
            bfs(i, &adj, &mut visited, &mut new_parents, &mut queue);
        }
    }

    new_parents
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr2, Array2};

    /// Helper: run `stitch_fragments` from owned arrays.
    fn stitch(
        coords: &Array2<f64>,
        comps: &[i32],
        mask: Option<Vec<bool>>,
        max_dist: f64,
    ) -> Vec<(i32, i32, f32)> {
        let comps = Array1::from(comps.to_vec());
        let mask = mask.map(Array1::from);
        stitch_fragments(&coords.view(), &comps.view(), &mask, max_dist)
    }

    #[test]
    fn two_fragments_pick_closest_pair() {
        // Fragment 0: nodes 0,1 near origin. Fragment 1: nodes 2,3 shifted +10 in x.
        // Closest cross pair is node 1 (x=1) and node 2 (x=10) -> distance 9.
        let coords = arr2(&[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [11.0, 0.0, 0.0],
        ]);
        let bridges = stitch(&coords, &[0, 0, 1, 1], None, f64::INFINITY);
        assert_eq!(bridges.len(), 1);
        let (a, b, d) = bridges[0];
        let pair = (a.min(b), a.max(b));
        assert_eq!(pair, (1, 2));
        assert!((d - 9.0).abs() < 1e-5, "distance was {d}");
    }

    #[test]
    fn three_fragments_return_two_bridges() {
        let coords = arr2(&[
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
        ]);
        let bridges = stitch(&coords, &[0, 1, 2], None, f64::INFINITY);
        assert_eq!(bridges.len(), 2);
        // Total added length should be the MST of the 3 collinear points: 10 + 10.
        let total: f32 = bridges.iter().map(|&(_, _, d)| d).sum();
        assert!((total - 20.0).abs() < 1e-4, "total was {total}");
    }

    #[test]
    fn max_dist_leaves_far_fragment_unconnected() {
        // Two close fragments (gap 9) and one far away (gap 100).
        let coords = arr2(&[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [111.0, 0.0, 0.0],
        ]);
        let bridges = stitch(&coords, &[0, 0, 1, 2], None, 20.0);
        // Only the 0<->1 fragments are within 20; fragment 2 stays isolated.
        assert_eq!(bridges.len(), 1);
        let (a, b, _) = bridges[0];
        assert_eq!((a.min(b), a.max(b)), (1, 2));
    }

    #[test]
    fn mask_restricts_bridge_endpoints() {
        // Fragment 0 nodes 0,1; fragment 1 nodes 2,3. Node 1 (the geometrically
        // closest) is masked out, so the bridge must use node 0 instead.
        let coords = arr2(&[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [11.0, 0.0, 0.0],
        ]);
        let mask = vec![true, false, true, false];
        let bridges = stitch(&coords, &[0, 0, 1, 1], Some(mask), f64::INFINITY);
        assert_eq!(bridges.len(), 1);
        let (a, b, d) = bridges[0];
        assert_eq!((a.min(b), a.max(b)), (0, 2));
        assert!((d - 10.0).abs() < 1e-5, "distance was {d}");
    }

    #[test]
    fn large_well_separated_fragments_are_connected() {
        // Two dense fragments of 200 nodes each, far apart. Every node's many
        // nearest neighbours are its own fragment-mates; only past ~200 do the
        // other fragment's nodes appear. A fixed small-k search would silently
        // fail to bridge these — this guards against that regression.
        let m = 200usize;
        let mut rows: Vec<[f64; 3]> = Vec::with_capacity(2 * m);
        let mut comps: Vec<i32> = Vec::with_capacity(2 * m);
        // Fragment 0: a tight cluster near the origin.
        for j in 0..m {
            let t = j as f64 * 0.01;
            rows.push([t, (j % 7) as f64 * 0.01, (j % 5) as f64 * 0.01]);
            comps.push(0);
        }
        // Fragment 1: a tight cluster far away in x.
        for j in 0..m {
            let t = j as f64 * 0.01;
            rows.push([1000.0 + t, (j % 7) as f64 * 0.01, (j % 5) as f64 * 0.01]);
            comps.push(1);
        }
        let coords = Array2::from(rows);
        let bridges = stitch(&coords, &comps, None, f64::INFINITY);
        assert_eq!(bridges.len(), 1, "the two fragments must be bridged");
        // The bridge connects a node from each fragment.
        let (a, b, _) = bridges[0];
        assert!(
            (a as usize) < m && (b as usize) >= m || (b as usize) < m && (a as usize) >= m,
            "bridge must cross fragments, got ({a}, {b})"
        );
    }

    #[test]
    fn single_component_returns_no_bridges() {
        let coords = arr2(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]);
        let bridges = stitch(&coords, &[0, 0], None, f64::INFINITY);
        assert!(bridges.is_empty());
    }

    #[test]
    fn fourth_column_steers_the_choice_of_bridge() {
        // Fragment 0 is a single node at the origin. Fragments 1 and 2 sit at the
        // same distance from it (10 units, in opposite directions), so in plain 3D
        // the tie is broken arbitrarily and both get bridged anyway.
        //
        // Add a 4th column ("radius"): fragment 0 matches fragment 2's value and is
        // far from fragment 1's. That must make 0 <-> 2 the cheaper edge, so it is
        // the one Boruvka picks for fragment 0.
        let coords = arr2(&[
            [0.0, 0.0, 0.0, 5.0],  // 0: fragment 0
            [-10.0, 0.0, 0.0, 0.0], // 1: fragment 1, radius far away
            [10.0, 0.0, 0.0, 5.0],  // 2: fragment 2, radius matches node 0
        ]);
        let bridges = stitch(&coords, &[0, 1, 2], None, f64::INFINITY);
        assert_eq!(bridges.len(), 2, "three fragments -> two bridges");

        // The 0 <-> 2 bridge is a pure 10-unit move in x (radius delta 0), whereas
        // 0 <-> 1 also pays 5 units of radius -> sqrt(10^2 + 5^2).
        let d02 = bridges
            .iter()
            .find(|(a, b, _)| (*a, *b) == (0, 2) || (*a, *b) == (2, 0))
            .expect("0 <-> 2 must be bridged")
            .2;
        assert!((d02 - 10.0).abs() < 1e-6, "got {d02}");

        let d01 = bridges
            .iter()
            .find(|(a, b, _)| (*a, *b) == (0, 1) || (*a, *b) == (1, 0))
            .map(|(_, _, d)| *d);
        if let Some(d) = d01 {
            let expected = (10.0f32 * 10.0 + 5.0 * 5.0).sqrt();
            assert!((d - expected).abs() < 1e-4, "got {d}");
        }
    }

    #[test]
    fn fourth_column_is_ignored_when_constant() {
        // A constant 4th column adds nothing to any pairwise distance, so the
        // result must be identical to the plain 3D case.
        let coords_3d = arr2(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [10.0, 0.0, 0.0]]);
        let coords_4d = arr2(&[
            [0.0, 0.0, 0.0, 7.0],
            [1.0, 0.0, 0.0, 7.0],
            [10.0, 0.0, 0.0, 7.0],
        ]);
        let a = stitch(&coords_3d, &[0, 0, 1], None, f64::INFINITY);
        let b = stitch(&coords_4d, &[0, 0, 1], None, f64::INFINITY);
        assert_eq!(a, b);
    }

    #[test]
    fn reroot_rewire_produces_single_rooted_tree() {
        // Two chains: 0->1->2 (root 0) and 3->4 (root 3), i.e. parents index-based.
        // parents: node1's parent is 0, node2's parent is 1, node4's parent is 3.
        let parents = Array1::from(vec![-1, 0, 1, -1, 3]);
        // Bridge node 2 <-> node 3.
        let new_edges = arr2(&[[2, 3]]);
        let new_parents = reroot_rewire(&parents.view(), &new_edges.view(), 0);

        // Exactly one root, at index 0.
        let roots: Vec<usize> = (0..new_parents.len())
            .filter(|&i| new_parents[i] < 0)
            .collect();
        assert_eq!(roots, vec![0]);

        // Every non-root node reaches root 0 by walking parents (no cycles).
        for start in 0..new_parents.len() {
            let mut node = start as i32;
            let mut steps = 0;
            while new_parents[node as usize] >= 0 {
                node = new_parents[node as usize];
                steps += 1;
                assert!(steps <= new_parents.len(), "cycle detected from {start}");
            }
            assert_eq!(node, 0, "node {start} does not reach root 0");
        }
    }

    #[test]
    fn reroot_rewire_roots_residual_components() {
        // Two disconnected chains, no bridge added: each must get its own root.
        let parents = Array1::from(vec![-1, 0, -1, 2]);
        let new_edges: Array2<i32> = Array2::zeros((0, 2));
        let new_parents = reroot_rewire(&parents.view(), &new_edges.view(), 0);
        let roots: Vec<usize> = (0..new_parents.len())
            .filter(|&i| new_parents[i] < 0)
            .collect();
        // Lowest-index node of each component: 0 and 2.
        assert_eq!(roots, vec![0, 2]);
    }
}
