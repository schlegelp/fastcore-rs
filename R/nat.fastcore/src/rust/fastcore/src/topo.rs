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
//!   the fragments, driven by a k-d tree that finds each node's nearest neighbour
//!   *in a different fragment* without enumerating its own (see [`KdTree`]). The
//!   result is a true minimum spanning tree of the "closest pair between
//!   fragments" graph, so the total added cable matches navis' healing exactly.
//! * [`reroot_rewire`] — given the original topology plus a set of new
//!   undirected edges and a preferred root, regenerate a valid `parents` array
//!   via BFS. This replaces navis' networkx `dfs_tree`/`minimum_spanning_tree`
//!   reroot.
//!
//! The neuron-specific policy (which nodes may be bridged, `max_dist`,
//! `min_size`, dropping disconnected fragments, …) is left to the caller
//! (navis), matching the general-interface convention of the rest of the crate.

use crate::kdtree::{box_dist2, dist2, KdTree, LEAF};
use ndarray::{Array, Array1, ArrayView1, ArrayView2};
use rayon::prelude::*;
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

/// Marker for "the points below this subtree are not all in one component".
const MIXED: u32 = u32::MAX;

/// *Nearest neighbour in a different component*, on top of the shared [`KdTree`].
///
/// This is the primitive a stock nearest-neighbour index cannot provide. Walking
/// neighbours in distance order and skipping own-component hits is quadratic when
/// fragments are spatially separated: a node deep inside a fragment has to
/// enumerate every one of its own fragment-mates before the search ever reaches a
/// foreign point, and no distance bound can prevent that (the bound is already
/// tight — the whole fragment simply lies inside it).
///
/// The fix is to prune at the *subtree* level rather than the point level. Each
/// round we label every subtree with the super-component shared by all points
/// below it, or [`MIXED`] if they disagree ([`KdTree::label`]). A subtree whose
/// label equals the querying node's own super-component cannot contain a foreign
/// point, so the search skips it in O(1) — turning "walk across my entire
/// fragment" into "step over it". Fragments are spatially coherent, so most
/// subtrees are single-component and the labels prune almost everything.
impl<const D: usize> KdTree<D> {
    /// Label every subtree with the super-component shared by all its points, or
    /// [`MIXED`]. `roots` gives each point's super-component in tree order.
    ///
    /// Super-components only ever merge, so labels get *coarser* — and the
    /// pruning therefore stronger — as Boruvka progresses.
    fn label(&self, roots: &[u32], out: &mut Vec<u32>) {
        out.clear();
        out.resize(self.nodes.len(), MIXED);
        // Children come after their parent, so reverse order is bottom-up.
        for t in (0..self.nodes.len()).rev() {
            let nd = &self.nodes[t];
            out[t] = if nd.left == LEAF {
                let slots = &roots[nd.start as usize..nd.end as usize];
                let first = slots[0];
                if slots.iter().all(|&r| r == first) {
                    first
                } else {
                    MIXED
                }
            } else {
                let (a, b) = (out[nd.left as usize], out[nd.right as usize]);
                // `MIXED == MIXED` correctly stays MIXED.
                if a == b {
                    a
                } else {
                    MIXED
                }
            };
        }
    }

    /// Find the nearest point to `q` whose super-component differs from
    /// `my_root`, considering only points closer than `best`.
    ///
    /// `best` is both the input bound and the output squared distance; `hit` is
    /// set to the winning point's *tree-order slot*. If nothing beats `best` the
    /// two are left untouched — so passing the component's current best-known
    /// bridge as `best` abandons nodes that cannot improve on it.
    #[allow(clippy::too_many_arguments)]
    fn nearest_foreign(
        &self,
        t: u32,
        q: &[f64; D],
        my_root: u32,
        roots: &[u32],
        labels: &[u32],
        best: &mut f64,
        hit: &mut u32,
    ) {
        let t = t as usize;
        // The entire subtree is in my super-component: nothing foreign below it.
        if labels[t] == my_root {
            return;
        }
        let nd = &self.nodes[t];
        if box_dist2(&nd.lo, &nd.hi, q) >= *best {
            return;
        }
        if nd.left == LEAF {
            let (start, end) = (nd.start as usize, nd.end as usize);
            for (k, (&root, p)) in roots[start..end]
                .iter()
                .zip(&self.pts[start..end])
                .enumerate()
            {
                if root == my_root {
                    continue;
                }
                let d2 = dist2(p, q);
                if d2 < *best {
                    *best = d2;
                    *hit = (start + k) as u32;
                }
            }
            return;
        }
        // Descend into the nearer child first: it tightens `best`, which then
        // prunes the sibling.
        let (l, r) = (nd.left, nd.right);
        let dl = {
            let c = &self.nodes[l as usize];
            box_dist2(&c.lo, &c.hi, q)
        };
        let dr = {
            let c = &self.nodes[r as usize];
            box_dist2(&c.lo, &c.hi, q)
        };
        let (near, far) = if dl <= dr { (l, r) } else { (r, l) };
        self.nearest_foreign(near, q, my_root, roots, labels, best, hit);
        self.nearest_foreign(far, q, my_root, roots, labels, best, hit);
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

/// What a Boruvka round learned about one candidate node.
enum Learned {
    /// Its exact nearest foreign neighbour: `(squared_distance, tree slot)`.
    Exact(f64, u32),
    /// Its query was abandoned; the nearest foreign neighbour is at least this
    /// far (squared) away.
    AtLeast(f64),
    /// Nothing new — a memo or floor already settled it.
    Nothing,
}

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

    // 3. Build the k-d tree over the candidate nodes. Points are held in tree
    //    order from here on; `tree.idx[slot]` maps back to a node index.
    let tree: KdTree<D> = KdTree::build(candidates.iter().map(|&i| (point(i), i as u32)).collect());
    let n_slots = candidates.len();

    // `max_dist` is inclusive, but the search below prunes anything not *strictly*
    // better than its bound, so start one ULP above the cap.
    let cap = if max_dist.is_finite() {
        let sq = max_dist * max_dist;
        f64::from_bits(sq.to_bits() + 1)
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
    //    A round is one `nearest_foreign` query per candidate node, pruned two
    //    ways. Subtree labels skip a node's own fragment wholesale (see [`KdTree`]).
    //    On top of that, a per-super-component bound — the shortest cross-edge any
    //    of its nodes has found so far this round — abandons any query that can no
    //    longer beat it: such a node cannot supply the component's minimum. The
    //    bound is a minimum over *real* cross-edges, so it never drops below the
    //    component's true minimum, and the node holding that minimum is therefore
    //    never pruned before reaching it. The result stays exact.
    //
    //    Two memos then carry work across rounds. Super-components only ever grow,
    //    so the foreign set only ever shrinks and a node's nearest-foreign distance
    //    only ever *increases*:
    //
    //    * `memo[s]` — node `s`'s exact nearest foreign neighbour. If that
    //      neighbour has not since been absorbed into `s`'s own super-component it
    //      is still the nearest one, so the round needs no query for `s` at all.
    //    * `floor[s]` — a lower bound on `s`'s nearest-foreign distance, recorded
    //      whenever a query is abandoned. A lower bound on a quantity that only
    //      grows stays valid for good, so once it reaches the component's bound `s`
    //      can be skipped without touching the tree.
    let mut uf = UnionFind::new(n_comps);
    let mut bridges: Vec<(i32, i32, f32)> = Vec::with_capacity(n_comps - 1);

    // All indexed by tree slot, so the hot loops stay in tree order.
    let mut memo: Vec<Option<(f64, u32)>> = vec![None; n_slots];
    let mut floor: Vec<f64> = vec![0.0; n_slots];
    let mut slot_root: Vec<u32> = vec![MIXED; n_slots];
    let mut labels: Vec<u32> = Vec::new();

    while uf.n_sets > 1 {
        // Snapshot each point's super-root, then label the subtrees, so the
        // parallel sweep below is purely read-only.
        let comp_root: Vec<u32> = (0..n_comps as u32).map(|c| uf.find(c)).collect();
        for s in 0..n_slots {
            slot_root[s] = comp_root[node_comp[tree.idx[s] as usize] as usize];
        }
        tree.label(&slot_root, &mut labels);

        // Per-super-component bound (squared distance).
        let bound: Vec<AtomicU64> = (0..n_comps).map(|_| AtomicU64::new(cap.to_bits())).collect();

        // Seed the bounds from the memo *before* the sweep: every node that cannot
        // improve on an already-known bridge is then skipped rather than searched.
        for s in 0..n_slots {
            if let Some((d2, t)) = memo[s] {
                if slot_root[t as usize] != slot_root[s] {
                    atomic_min_f64(&bound[slot_root[s] as usize], d2);
                }
            }
        }

        let learned: Vec<Learned> = (0..n_slots)
            .into_par_iter()
            .map(|s| {
                let my_root = slot_root[s];
                let cell = &bound[my_root as usize];
                let limit = f64::from_bits(cell.load(Ordering::Relaxed));

                // A memoised neighbour that is still foreign is still the nearest.
                if let Some((d2, t)) = memo[s] {
                    if slot_root[t as usize] != my_root {
                        return if d2 <= limit {
                            atomic_min_f64(cell, d2);
                            Learned::Exact(d2, t)
                        } else {
                            Learned::Nothing
                        };
                    }
                }
                // Already known to be no better than the component's bound.
                if floor[s] >= limit {
                    return Learned::Nothing;
                }

                let mut best = limit;
                let mut hit = MIXED;
                let q = tree.pts[s];
                tree.nearest_foreign(0, &q, my_root, &slot_root, &labels, &mut best, &mut hit);
                if hit == MIXED {
                    return Learned::AtLeast(limit);
                }
                atomic_min_f64(cell, best);
                Learned::Exact(best, hit)
            })
            .collect();

        let mut found: Vec<Candidate> = Vec::new();
        for (s, l) in learned.into_iter().enumerate() {
            match l {
                Learned::Exact(d2, t) => {
                    memo[s] = Some((d2, t));
                    found.push((slot_root[s], d2, tree.idx[s], tree.idx[t as usize]));
                }
                Learned::AtLeast(d2) => floor[s] = floor[s].max(d2),
                Learned::Nothing => {}
            }
        }

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

    /// The true MST weight over the fragments, by brute force: exact all-pairs
    /// distances between every pair of fragments, then Kruskal. Every MST of a
    /// graph has the same total weight, so this is the number `stitch_fragments`
    /// must reproduce -- independently of which particular edges it picks.
    fn brute_force_mst_weight(coords: &Array2<f64>, comps: &[i32]) -> f64 {
        let labels: Vec<i32> = {
            let mut l = comps.to_vec();
            l.sort_unstable();
            l.dedup();
            l
        };
        let c = labels.len();
        let pos = |lab: i32| labels.iter().position(|&x| x == lab).unwrap();

        // Closest pair between every pair of fragments.
        let mut edges: Vec<(f64, usize, usize)> = Vec::new();
        for a in 0..c {
            for b in (a + 1)..c {
                let mut best = f64::INFINITY;
                for i in 0..coords.nrows() {
                    if pos(comps[i]) != a {
                        continue;
                    }
                    for j in 0..coords.nrows() {
                        if pos(comps[j]) != b {
                            continue;
                        }
                        let d: f64 = (0..coords.ncols())
                            .map(|k| (coords[[i, k]] - coords[[j, k]]).powi(2))
                            .sum();
                        best = best.min(d);
                    }
                }
                edges.push((best.sqrt(), a, b));
            }
        }
        edges.sort_by(|x, y| x.0.total_cmp(&y.0));

        let mut uf = UnionFind::new(c);
        let mut total = 0.0;
        for (w, a, b) in edges {
            if uf.union(a as u32, b as u32) {
                total += w;
            }
        }
        total
    }

    /// A deterministic xorshift, so the fuzz below needs no rand dependency.
    fn rng(state: &mut u64) -> f64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        (*state >> 11) as f64 / (1u64 << 53) as f64
    }

    #[test]
    fn matches_brute_force_mst_weight() {
        // The Boruvka search is pruned three ways (subtree labels, the
        // per-component bound, and the cross-round memos). Any of them being
        // slightly too aggressive would still return a spanning set of bridges --
        // just not a *minimal* one. Only the total weight catches that, so check
        // it against an independent brute-force MST over many random layouts.
        let mut state = 0x1234_5678_9abc_def0u64;
        for trial in 0..60 {
            let n = 12 + (trial * 7) % 60;
            let n_comps = 2 + (trial * 3) % 6;

            let mut rows: Vec<[f64; 3]> = Vec::with_capacity(n);
            let mut comps: Vec<i32> = Vec::with_capacity(n);
            for i in 0..n {
                // Cluster each fragment around its own centre, with enough spread
                // that fragments both overlap (interleaved) and separate.
                let c = (i % n_comps) as f64;
                let spread = 1.0 + (trial % 5) as f64 * 4.0;
                rows.push([
                    c * 10.0 + rng(&mut state) * spread,
                    c * 3.0 + rng(&mut state) * spread,
                    rng(&mut state) * spread,
                ]);
                comps.push((i % n_comps) as i32);
            }
            let coords = Array2::from(rows);

            let bridges = stitch(&coords, &comps, None, f64::INFINITY);
            assert_eq!(
                bridges.len(),
                n_comps - 1,
                "trial {trial}: every fragment must be connected"
            );
            let got: f64 = bridges.iter().map(|&(_, _, d)| d as f64).sum();
            let want = brute_force_mst_weight(&coords, &comps);
            assert!(
                (got - want).abs() < 1e-3 * want.max(1.0),
                "trial {trial}: total bridge length {got} != true MST {want}"
            );
        }
    }

    #[test]
    fn well_separated_fragments_stitch_quickly() {
        // Guards the pathology this module was rewritten for. Two dense, spatially
        // disjoint fragments: every node's nearest ~8000 neighbours are its own
        // fragment-mates. A search that walks neighbours in distance order has to
        // enumerate all of them before it ever reaches the other fragment, which is
        // quadratic -- this took ~8s. Pruning whole subtrees by component makes it
        // instant, so a generous bound still separates the two by orders of
        // magnitude (and holds even in an unoptimised debug build).
        let m = 8_000usize;
        let mut rows: Vec<[f64; 3]> = Vec::with_capacity(2 * m);
        let mut comps: Vec<i32> = Vec::with_capacity(2 * m);
        let mut state = 0xdead_beef_cafe_f00du64;
        for i in 0..2 * m {
            let shift = if i < m { 0.0 } else { 100_000.0 };
            rows.push([
                shift + rng(&mut state) * 100.0,
                rng(&mut state) * 100.0,
                rng(&mut state) * 100.0,
            ]);
            comps.push(if i < m { 0 } else { 1 });
        }
        let coords = Array2::from(rows);

        let t = std::time::Instant::now();
        let bridges = stitch(&coords, &comps, None, f64::INFINITY);
        let elapsed = t.elapsed();

        assert_eq!(bridges.len(), 1);
        assert!(
            elapsed.as_secs_f64() < 5.0,
            "stitching two well-separated fragments took {elapsed:?} -- the \
             component-level subtree pruning has regressed to a per-point walk"
        );
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
