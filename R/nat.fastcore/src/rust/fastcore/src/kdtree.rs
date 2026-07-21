//! A static k-d tree, shared by the modules that need spatial queries.
//!
//! Deliberately minimal: built once from a point set, never mutated, and queried through
//! whichever traversal the caller needs. [`KdTree::knn`] — the `k` nearest neighbours of a
//! point, exact — lives here; [`crate::topo`] adds its own "nearest neighbour in a *different*
//! component" traversal as a second inherent impl, because the subtree pruning that makes that
//! one fast is specific to healing and has no business in the general tree.
//!
//! Generic over the dimension `D` so the same tree serves 3D point clouds and the 2D/nD cases
//! `topo` supports, with the coordinate loop unrolled at each instantiation.

/// Marker stored in [`KdNode::left`] to say "this node is a leaf".
pub(crate) const LEAF: u32 = u32::MAX;

/// Points per k-d tree leaf. Big enough that a leaf scan amortises the descent, small enough
/// that a leaf is rarely worth splitting further.
pub(crate) const LEAF_SIZE: usize = 16;

/// Squared Euclidean distance between two points.
#[inline]
pub(crate) fn dist2<const D: usize>(a: &[f64; D], b: &[f64; D]) -> f64 {
    let mut sum = 0.0;
    for k in 0..D {
        let d = a[k] - b[k];
        sum += d * d;
    }
    sum
}

/// Squared distance from `q` to the closest point of the box `lo..hi` (zero if `q` is inside
/// it).
#[inline]
pub(crate) fn box_dist2<const D: usize>(lo: &[f64; D], hi: &[f64; D], q: &[f64; D]) -> f64 {
    let mut sum = 0.0;
    for k in 0..D {
        let d = if q[k] < lo[k] {
            lo[k] - q[k]
        } else if q[k] > hi[k] {
            q[k] - hi[k]
        } else {
            0.0
        };
        sum += d * d;
    }
    sum
}

/// One node of the k-d tree. Leaves have `left == LEAF` and cover the points `start..end` of
/// the (tree-ordered) point array; internal nodes index their two children.
pub(crate) struct KdNode<const D: usize> {
    pub(crate) lo: [f64; D],
    pub(crate) hi: [f64; D],
    pub(crate) start: u32,
    pub(crate) end: u32,
    pub(crate) left: u32,
    pub(crate) right: u32,
}

/// A static k-d tree over `D`-dimensional points.
///
/// Points are permuted into tree order at build time, so a leaf scan walks contiguous memory
/// and `idx` maps each slot back to the caller's own indexing.
pub(crate) struct KdTree<const D: usize> {
    /// Point coordinates, permuted into tree order.
    pub(crate) pts: Vec<[f64; D]>,
    /// Original index of each point, in the same tree order.
    pub(crate) idx: Vec<u32>,
    pub(crate) nodes: Vec<KdNode<D>>,
}

impl<const D: usize> KdTree<D> {
    pub(crate) fn build(mut items: Vec<([f64; D], u32)>) -> Self {
        let mut nodes = Vec::with_capacity(2 * items.len() / LEAF_SIZE + 1);
        Self::split(&mut items, 0, &mut nodes);
        let (pts, idx) = items.into_iter().unzip();
        KdTree { pts, idx, nodes }
    }

    /// Recursively split `items` at the median of its widest axis, appending the resulting
    /// nodes in pre-order (so children always sit *after* their parent, which is what lets
    /// `topo`'s subtree labelling work bottom-up by iterating in reverse). Returns the index of
    /// the node covering `items`.
    fn split(items: &mut [([f64; D], u32)], offset: usize, nodes: &mut Vec<KdNode<D>>) -> u32 {
        let me = nodes.len() as u32;
        let mut lo = [f64::INFINITY; D];
        let mut hi = [f64::NEG_INFINITY; D];
        for (p, _) in items.iter() {
            for k in 0..D {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
        nodes.push(KdNode {
            lo,
            hi,
            start: offset as u32,
            end: (offset + items.len()) as u32,
            left: LEAF,
            right: LEAF,
        });
        if items.len() <= LEAF_SIZE {
            return me;
        }

        let axis = (0..D)
            .max_by(|&a, &b| (hi[a] - lo[a]).total_cmp(&(hi[b] - lo[b])))
            .unwrap();
        let mid = items.len() / 2;
        items.select_nth_unstable_by(mid, |a, b| a.0[axis].total_cmp(&b.0[axis]));
        let (left_items, right_items) = items.split_at_mut(mid);
        let left = Self::split(left_items, offset, nodes);
        let right = Self::split(right_items, offset + mid, nodes);
        nodes[me as usize].left = left;
        nodes[me as usize].right = right;
        me
    }

    /// The `k` nearest points to `q`, exactly, written into `out` as `(squared distance,
    /// original index)` pairs sorted nearest-first.
    ///
    /// `out` is a caller-owned buffer so a batch of queries allocates once rather than once per
    /// point; it is cleared on entry. Fewer than `k` results come back only when the tree holds
    /// fewer than `k` points.
    ///
    /// Ties are broken on the original index, which makes the answer a function of the point
    /// *set* alone — not of the order the tree happened to visit it in, and so not of the leaf
    /// size or the median split either. Coincident points are common in real point clouds
    /// (duplicated mesh vertices, resampled skeletons), so this is not a hypothetical.
    pub(crate) fn knn(&self, q: &[f64; D], k: usize, out: &mut Vec<(f64, u32)>) {
        out.clear();
        if k == 0 || self.pts.is_empty() {
            return;
        }
        out.reserve(k);
        self.knn_into(0, q, k, out);
    }

    fn knn_into(&self, t: u32, q: &[f64; D], k: usize, out: &mut Vec<(f64, u32)>) {
        let nd = &self.nodes[t as usize];
        // `>` rather than `>=`: a subtree exactly at the current k-th distance can still hold a
        // point that wins on the index tie-break above.
        if out.len() == k && box_dist2(&nd.lo, &nd.hi, q) > out[k - 1].0 {
            return;
        }
        if nd.left == LEAF {
            let (start, end) = (nd.start as usize, nd.end as usize);
            for (p, &i) in self.pts[start..end].iter().zip(&self.idx[start..end]) {
                Self::offer(out, k, dist2(p, q), i);
            }
            return;
        }
        // Descend into the nearer child first: it fills `out` with close hits, which tightens
        // the bound that then prunes the sibling.
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
        self.knn_into(near, q, k, out);
        self.knn_into(far, q, k, out);
    }

    /// Insert one candidate into the sorted top-`k` buffer, dropping the worst if it is full.
    ///
    /// A sorted insert rather than a binary heap: `k` here is tens, so the shift is a handful
    /// of cache-resident moves and beats the sift plus the heap's unordered final drain, which
    /// would need a sort anyway to report nearest-first.
    #[inline]
    fn offer(out: &mut Vec<(f64, u32)>, k: usize, d2: f64, i: u32) {
        if out.len() == k {
            let worst = out[k - 1];
            if (d2, i) >= (worst.0, worst.1) {
                return;
            }
            out.pop();
        }
        let pos = out.partition_point(|&e| e < (d2, i));
        out.insert(pos, (d2, i));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Brute-force top-k under the same `(distance, index)` order the tree uses.
    fn brute<const D: usize>(pts: &[[f64; D]], q: &[f64; D], k: usize) -> Vec<(f64, u32)> {
        let mut all: Vec<(f64, u32)> = pts
            .iter()
            .enumerate()
            .map(|(i, p)| (dist2(p, q), i as u32))
            .collect();
        all.sort_by(|a, b| a.partial_cmp(b).unwrap());
        all.truncate(k);
        all
    }

    /// Deterministic pseudo-random points — no `rand` dependency in a unit test.
    fn cloud(n: usize, seed: u64) -> Vec<[f64; 3]> {
        let mut s = seed;
        let mut next = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f64) / ((1u64 << 31) as f64)
        };
        (0..n).map(|_| [next(), next(), next()]).collect()
    }

    #[test]
    fn knn_is_exact_against_brute_force() {
        // Well past LEAF_SIZE, so the descent and the pruning both actually run.
        let pts = cloud(500, 7);
        let tree = KdTree::build(pts.iter().copied().zip(0..).collect());
        let mut out = Vec::new();
        for (i, q) in pts.iter().enumerate() {
            for &k in &[1usize, 5, 20] {
                tree.knn(q, k, &mut out);
                assert_eq!(out, brute(&pts, q, k), "point {i}, k = {k}");
            }
        }
    }

    #[test]
    fn the_query_point_is_its_own_nearest_neighbour() {
        let pts = cloud(100, 11);
        let tree = KdTree::build(pts.iter().copied().zip(0..).collect());
        let mut out = Vec::new();
        for (i, q) in pts.iter().enumerate() {
            tree.knn(q, 3, &mut out);
            assert_eq!(out[0], (0.0, i as u32));
        }
    }

    #[test]
    fn coincident_points_break_ties_on_index() {
        // Ten copies of the same point: which ones come back must not depend on tree order.
        let pts: Vec<[f64; 3]> = vec![[1.0, 2.0, 3.0]; 10];
        let tree = KdTree::build(pts.iter().copied().zip(0..).collect());
        let mut out = Vec::new();
        tree.knn(&[1.0, 2.0, 3.0], 4, &mut out);
        assert_eq!(out, vec![(0.0, 0), (0.0, 1), (0.0, 2), (0.0, 3)]);
    }

    #[test]
    fn asking_for_more_neighbours_than_exist_returns_all_of_them() {
        let pts = cloud(5, 3);
        let tree = KdTree::build(pts.iter().copied().zip(0..).collect());
        let mut out = Vec::new();
        tree.knn(&pts[0], 50, &mut out);
        assert_eq!(out.len(), 5);
        assert_eq!(out, brute(&pts, &pts[0], 50));
    }

    #[test]
    fn empty_tree_and_zero_k_yield_nothing() {
        let tree: KdTree<3> = KdTree::build(Vec::new());
        let mut out = vec![(1.0, 1)];
        tree.knn(&[0.0; 3], 5, &mut out);
        assert!(out.is_empty());

        let pts = cloud(20, 5);
        let tree = KdTree::build(pts.iter().copied().zip(0..).collect());
        tree.knn(&pts[0], 0, &mut out);
        assert!(out.is_empty());
    }
}
