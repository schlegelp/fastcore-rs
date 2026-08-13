//! A bounding volume hierarchy over triangles, built for one question.
//!
//! The question is [`Bvh::occluded`]: *does this segment hit anything at all?* Not what it
//! hits, not where, not which is nearest — just whether the far end is visible from the near
//! one. That is the whole of what [`crate::internals`] asks, sixteen times per face, and it
//! is a strictly cheaper query than the first-hit one a general collision library exposes:
//!
//! - **Any order will do.** A nearest-hit traversal has to descend the closer child first and
//!   keep shrinking the ray's far bound, so it cannot stop until every box still closer than
//!   the best hit is exhausted. This returns on the first triangle it touches, whichever it
//!   is, so a buried face — which is the case that matters, since those are the ones being
//!   looked for — usually costs a handful of node visits.
//! - **No hit record.** Nothing to fill in, so nothing to carry down the stack and nothing to
//!   write back. The traversal state is a stack of node indices and a `bool`.
//!
//! Against that, a miss is the expensive case: proving a ray escapes means visiting every box
//! it pierces. Those are the ~70% of faces that are plainly outside, so the build still has
//! to produce a tree that culls well, which is why this is binned-SAH rather than a median
//! split.
//!
//! # Precision
//!
//! Triangles are stored as `f32` **relative to the centre of the mesh's bounding box**. That
//! is what makes the halved memory traffic safe: neuron meshes arrive in nanometres with
//! coordinates in the millions, where a bare `f32` has ulps of ~0.1 nm against a ~30 nm edge,
//! but re-centred the exponent is spent on the mesh's own extent instead of on its distance
//! from an arbitrary origin. The ratio that results is scale-free — a mesh half a million
//! units across resolves to ~0.03 units whether those units are nanometres or microns — and
//! it leaves ~3 decimal digits between an ulp and the shortest edge in the mesh.
//!
//! `f64` throughout would be the conservative choice and costs about 40% in throughput, all
//! of it in cache: the triangle soup for a 3.5 M-face mesh is 126 MB at `f32` and 252 at
//! `f64`, against an L3 that holds neither.
//!
//! # Where the time goes
//!
//! Build is `O(F log F)` and parallel by subtree; query is the whole cost of the caller.
//! On a 3.5 M-face mesh the build is ~0.9 s and the 56 M queries of one `openness` pass are
//! ~11 s across 10 cores, so the tree is amortised over roughly sixty rays per triangle.

use rayon::prelude::*;

use crate::simplify::dot;

/// Triangles per leaf. Small enough that a leaf is one or two cache lines of the soup, large
/// enough that the intersection loop is not swamped by the node visit that got there.
const LEAF_SIZE: usize = 4;

/// Buckets per axis for the SAH sweep. The classic 12–16; past that the sweep costs more than
/// the tree it saves.
const N_BINS: usize = 12;

/// Subtrees smaller than this are built on the calling thread. Below it the `rayon::join` is
/// most of the work.
const PARALLEL_CUTOFF: usize = 8_192;

/// An axis-aligned box. Empty is `min > max` on every axis, which unions correctly.
#[derive(Clone, Copy, Debug)]
struct Aabb {
    min: [f32; 3],
    max: [f32; 3],
}

impl Aabb {
    const EMPTY: Self = Self {
        min: [f32::INFINITY; 3],
        max: [f32::NEG_INFINITY; 3],
    };

    #[inline]
    fn extend(&mut self, p: [f32; 3]) {
        for i in 0..3 {
            self.min[i] = self.min[i].min(p[i]);
            self.max[i] = self.max[i].max(p[i]);
        }
    }

    #[inline]
    fn merge(&mut self, o: &Aabb) {
        for i in 0..3 {
            self.min[i] = self.min[i].min(o.min[i]);
            self.max[i] = self.max[i].max(o.max[i]);
        }
    }

    /// Surface area, or zero when empty — the SAH's cost term.
    #[inline]
    fn area(&self) -> f32 {
        let d = [
            self.max[0] - self.min[0],
            self.max[1] - self.min[1],
            self.max[2] - self.min[2],
        ];
        if d[0] < 0.0 || d[1] < 0.0 || d[2] < 0.0 {
            return 0.0;
        }
        2.0 * (d[0] * d[1] + d[1] * d[2] + d[2] * d[0])
    }
}

/// A node, 32 bytes so two fit a cache line.
///
/// `count == 0` marks an interior node. Its left child is always the next node — the subtree
/// is laid out root-first — so only the right child needs an index, and `first` carries it.
/// For a leaf, `first` is instead where its triangles start in the soup.
#[derive(Clone, Copy)]
struct Node {
    bounds: Aabb,
    first: u32,
    count: u32,
}

/// Triangles, re-centred and packed for traversal.
pub(crate) struct Bvh {
    /// Nine floats per triangle: `v0`, then the two edge vectors `v1 - v0` and `v2 - v0`,
    /// which is what Möller–Trumbore actually reads. Storing edges rather than corners saves
    /// six subtractions per test and costs nothing to build.
    tris: Vec<f32>,
    nodes: Vec<Node>,
    /// What every stored coordinate was shifted by — see the module note on precision.
    center: [f64; 3],
    /// The mesh's bounding box, in world coordinates. Kept because the caller needs a segment
    /// length that clears the mesh, and this is the box that answers it — see [`Bvh::reach`].
    lo: [f64; 3],
    hi: [f64; 3],
}

/// The per-triangle boxes and centroids the build reads, in the *original* indexing.
///
/// Separate from the order being permuted because these two never change: they are the same
/// pair of slices at every level of the recursion, and only the slice of `order` narrows.
struct Prims<'a> {
    bounds: &'a [Aabb],
    centroids: &'a [[f32; 3]],
}

impl Bvh {
    /// Build over `faces` indexing `vertices`.
    ///
    /// Degenerate faces are kept: a zero-area triangle can never be hit — Möller–Trumbore
    /// rejects it on the determinant — but dropping it would renumber nothing and cost a
    /// compaction pass, so it simply sits in the tree as a flat box.
    ///
    /// Runs on the ambient rayon pool; callers wrap it in [`crate::threads::with_pool`].
    pub(crate) fn build(vertices: &[f64], faces: &[u32]) -> Self {
        let n = faces.len() / 3;

        // Re-centre on the bounding box rather than the centroid: the centroid of a neuron
        // mesh sits wherever the arbor happens to be dense, which is not where the extent is.
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        for p in vertices.chunks_exact(3) {
            for i in 0..3 {
                lo[i] = lo[i].min(p[i]);
                hi[i] = hi[i].max(p[i]);
            }
        }
        let center = if vertices.is_empty() {
            [0.0; 3]
        } else {
            [
                0.5 * (lo[0] + hi[0]),
                0.5 * (lo[1] + hi[1]),
                0.5 * (lo[2] + hi[2]),
            ]
        };

        if n == 0 {
            return Self {
                tris: Vec::new(),
                nodes: vec![Node {
                    bounds: Aabb::EMPTY,
                    first: 0,
                    count: 0,
                }],
                center,
                lo: [0.0; 3],
                hi: [0.0; 3],
            };
        }

        let corner = |v: u32| -> [f32; 3] {
            let b = 3 * v as usize;
            [
                (vertices[b] - center[0]) as f32,
                (vertices[b + 1] - center[1]) as f32,
                (vertices[b + 2] - center[2]) as f32,
            ]
        };

        let mut bounds = vec![Aabb::EMPTY; n];
        let mut centroids = vec![[0.0f32; 3]; n];
        bounds
            .par_iter_mut()
            .zip(centroids.par_iter_mut())
            .zip(faces.par_chunks_exact(3))
            .for_each(|((b, c), f)| {
                let (a, d, e) = (corner(f[0]), corner(f[1]), corner(f[2]));
                b.extend(a);
                b.extend(d);
                b.extend(e);
                // The box centre, not the vertex mean: the SAH bins boxes, and this is the
                // one that keeps a long thin triangle in the bin its extent argues for.
                *c = [
                    0.5 * (b.min[0] + b.max[0]),
                    0.5 * (b.min[1] + b.max[1]),
                    0.5 * (b.min[2] + b.max[2]),
                ];
            });

        let mut order: Vec<u32> = (0..n as u32).collect();
        let root = bounds.par_iter().copied().reduce(
            || Aabb::EMPTY,
            |mut a, b| {
                a.merge(&b);
                a
            },
        );

        // The recursion returns the whole tree with its root at index 0.
        let prims = Prims {
            bounds: &bounds,
            centroids: &centroids,
        };
        let nodes = Self::split(&prims, &mut order, 0, root);

        // Both are dead from here (`prims` borrowed them, and that borrow ended above), and
        // the soup allocated next is larger than either — so freeing them first is the
        // difference between holding two of the three F-sized arrays at once and all three.
        drop(bounds);
        drop(centroids);

        // Permute the soup into leaf order, which is the point of the permutation: a leaf's
        // triangles are then contiguous and its intersection loop is one linear read.
        let mut tris = vec![0.0f32; 9 * n];
        tris.par_chunks_exact_mut(9)
            .zip(order.par_iter())
            .for_each(|(t, &f)| {
                let f = &faces[3 * f as usize..3 * f as usize + 3];
                let (a, b, c) = (corner(f[0]), corner(f[1]), corner(f[2]));
                t[0..3].copy_from_slice(&a);
                for i in 0..3 {
                    t[3 + i] = b[i] - a[i];
                    t[6 + i] = c[i] - a[i];
                }
            });

        Self {
            tris,
            nodes,
            center,
            lo,
            hi,
        }
    }

    /// Build the subtree covering `order`, returning its nodes with the root at index 0.
    ///
    /// `first` is where `order` begins in the global permutation, so a leaf can record the
    /// position its triangles will end up at directly and the caller has only node *indices*
    /// left to patch.
    ///
    /// Returning a `Vec` per subtree rather than writing into one shared arena is what lets
    /// the two halves run on different threads without a lock or an index reservation; those
    /// node indices are shifted as the halves are concatenated, which is a linear pass over
    /// nodes that were just written and so are still in cache.
    fn split(p: &Prims<'_>, order: &mut [u32], first: usize, bounds: Aabb) -> Vec<Node> {
        let n = order.len();

        let leaf = || {
            vec![Node {
                bounds,
                first: first as u32,
                count: n as u32,
            }]
        };
        if n <= LEAF_SIZE {
            return leaf();
        }

        // Bin along the axis the centroids spread furthest on. A node whose centroids all
        // coincide has no split to find on any axis and becomes a leaf, however big it is —
        // a stack of coincident triangles, which real meshes do contain.
        let mut cb = Aabb::EMPTY;
        for &t in order.iter() {
            cb.extend(p.centroids[t as usize]);
        }
        let extent = [
            cb.max[0] - cb.min[0],
            cb.max[1] - cb.min[1],
            cb.max[2] - cb.min[2],
        ];
        let axis = (0..3)
            .max_by(|&i, &j| extent[i].total_cmp(&extent[j]))
            .expect("three axes");
        // `is_finite` first so a NaN coordinate — which a broken mesh does contain — lands
        // here as a leaf rather than sliding through the binning as a garbage split.
        if !extent[axis].is_finite() || extent[axis] <= 0.0 {
            return leaf();
        }

        let scale = N_BINS as f32 / extent[axis];
        let bin_of = |t: u32| -> usize {
            let x = (p.centroids[t as usize][axis] - cb.min[axis]) * scale;
            (x as usize).min(N_BINS - 1)
        };

        let mut bin_box = [Aabb::EMPTY; N_BINS];
        let mut bin_n = [0usize; N_BINS];
        for &t in order.iter() {
            let k = bin_of(t);
            bin_box[k].merge(&p.bounds[t as usize]);
            bin_n[k] += 1;
        }

        // Sweep the N_BINS - 1 planes from both ends, so each candidate's two half-costs are
        // available in O(1). Cost is the SAH proper, area-weighted counts; the constant
        // traversal term is dropped because it shifts every candidate equally.
        let mut left_area = [0.0f32; N_BINS - 1];
        let mut left_n = [0usize; N_BINS - 1];
        let mut acc = Aabb::EMPTY;
        let mut cnt = 0;
        for k in 0..N_BINS - 1 {
            acc.merge(&bin_box[k]);
            cnt += bin_n[k];
            left_area[k] = acc.area();
            left_n[k] = cnt;
        }
        let mut best = (f32::INFINITY, 0usize);
        acc = Aabb::EMPTY;
        cnt = 0;
        for k in (0..N_BINS - 1).rev() {
            acc.merge(&bin_box[k + 1]);
            cnt += bin_n[k + 1];
            let cost = left_area[k] * left_n[k] as f32 + acc.area() * cnt as f32;
            if cost < best.0 {
                best = (cost, k);
            }
        }

        // Not splitting costs the parent's own area times its triangles. Taking a leaf
        // whenever the best split loses to that is what stops the recursion from grinding a
        // cluster of overlapping triangles down one at a time.
        if best.0 >= bounds.area() * n as f32 {
            return leaf();
        }

        let mid = partition(order, |t| bin_of(t) <= best.1);
        // A bin sweep can land everything on one side when the centroids are degenerate in a
        // way the extent test missed; halve it rather than recurse on the same set forever.
        let mid = if mid == 0 || mid == n { n / 2 } else { mid };

        let (left, right) = order.split_at_mut(mid);
        let mut lb = Aabb::EMPTY;
        for &t in left.iter() {
            lb.merge(&p.bounds[t as usize]);
        }
        let mut hb = Aabb::EMPTY;
        for &t in right.iter() {
            hb.merge(&p.bounds[t as usize]);
        }

        let (mut ln, mut rn) = if n >= PARALLEL_CUTOFF {
            rayon::join(
                || Self::split(p, left, first, lb),
                || Self::split(p, right, first + mid, hb),
            )
        } else {
            (
                Self::split(p, left, first, lb),
                Self::split(p, right, first + mid, hb),
            )
        };

        // Layout: this node, then the whole left subtree, then the whole right. So the left
        // child is at index 1 — implicit, and why a node needs only one child index — and the
        // right is at `1 + ln.len()`. Both halves numbered their nodes relative to their own
        // root, so both need shifting by where that root landed. Leaf offsets need nothing:
        // they were given the global `first` on the way down.
        let shift_right = 1 + ln.len() as u32;
        shift_interiors(&mut ln, 1);
        shift_interiors(&mut rn, shift_right);

        let mut out = Vec::with_capacity(1 + ln.len() + rn.len());
        out.push(Node {
            bounds,
            first: shift_right,
            count: 0,
        });
        out.append(&mut ln);
        out.append(&mut rn);
        out
    }

    /// A segment length that clears the mesh from anywhere inside it: the bounding box
    /// diagonal, with a little to spare.
    ///
    /// As good as infinity for a ray that only has to get out, and it lives here because the
    /// box it comes from is one the build already walked every vertex for. Zero for an empty
    /// tree, which blocks nothing anyway.
    pub(crate) fn reach(&self) -> f64 {
        let d = [
            self.hi[0] - self.lo[0],
            self.hi[1] - self.lo[1],
            self.hi[2] - self.lo[2],
        ];
        1.01 * dot(d, d).sqrt()
    }

    /// Is the open segment from `origin` to `origin + dir * len` blocked?
    ///
    /// `dir` need not be normalised; `len` is in units of `dir`, so passing an un-normalised
    /// direction and `len = 1.0` tests the segment to `origin + dir` exactly. Returns on the
    /// first triangle hit.
    ///
    /// Hits closer than [`T_MIN`] along the ray are ignored, which is what keeps a ray
    /// launched from just above a face from being blocked by that same face.
    #[inline]
    pub fn occluded(&self, origin: [f64; 3], dir: [f64; 3], len: f64) -> bool {
        if self.tris.is_empty() {
            return false;
        }
        let o = [
            (origin[0] - self.center[0]) as f32,
            (origin[1] - self.center[1]) as f32,
            (origin[2] - self.center[2]) as f32,
        ];
        let d = [dir[0] as f32, dir[1] as f32, dir[2] as f32];
        let t_max = len as f32;

        // Reciprocals once for the whole traversal. An infinity here is deliberate and
        // correct: the slab test multiplies it by a difference that is zero exactly when the
        // ray lies in the slab's plane, and the NaN that produces is caught by the ordering
        // below, which uses `min`/`max` rather than comparisons that a NaN would fail.
        let inv = [1.0 / d[0], 1.0 / d[1], 1.0 / d[2]];

        let mut stack = [0u32; 64];
        let mut sp = 0usize;
        let mut node = 0u32;

        loop {
            let n = &self.nodes[node as usize];
            if n.count > 0 {
                let first = n.first as usize;
                for t in first..first + n.count as usize {
                    if self.hits(t, o, d, t_max) {
                        return true;
                    }
                }
            } else {
                let l = node + 1;
                let r = n.first;
                let hl = self.slab(l, o, inv, t_max);
                let hr = self.slab(r, o, inv, t_max);
                // No ordering: any hit ends the traversal, so there is no near-first
                // heuristic worth the compare — descend left and stack right.
                if hl {
                    if hr && sp < stack.len() {
                        stack[sp] = r;
                        sp += 1;
                    }
                    node = l;
                    continue;
                } else if hr {
                    node = r;
                    continue;
                }
            }
            if sp == 0 {
                return false;
            }
            sp -= 1;
            node = stack[sp];
        }
    }

    /// Slab test against node `i`'s box, over `[0, t_max]`.
    #[inline]
    fn slab(&self, i: u32, o: [f32; 3], inv: [f32; 3], t_max: f32) -> bool {
        let b = &self.nodes[i as usize].bounds;
        let mut lo = 0.0f32;
        let mut hi = t_max;
        for k in 0..3 {
            let t0 = (b.min[k] - o[k]) * inv[k];
            let t1 = (b.max[k] - o[k]) * inv[k];
            lo = lo.max(t0.min(t1));
            hi = hi.min(t0.max(t1));
        }
        lo <= hi
    }

    /// Möller–Trumbore against triangle `t`, over `(T_MIN, t_max)`.
    #[inline]
    fn hits(&self, t: usize, o: [f32; 3], d: [f32; 3], t_max: f32) -> bool {
        let tri = &self.tris[9 * t..9 * t + 9];
        let (v0, e1, e2) = (&tri[0..3], &tri[3..6], &tri[6..9]);

        let p = [
            d[1] * e2[2] - d[2] * e2[1],
            d[2] * e2[0] - d[0] * e2[2],
            d[0] * e2[1] - d[1] * e2[0],
        ];
        let det = e1[0] * p[0] + e1[1] * p[1] + e1[2] * p[2];
        // Two-sided: a neuron mesh's winding is not to be trusted, and "did the ray get out"
        // does not care which way the face it met was pointing. Zero-area triangles fail
        // here, which is how they stay unhittable.
        if det.abs() < 1e-12 {
            return false;
        }
        let inv_det = 1.0 / det;

        let s = [o[0] - v0[0], o[1] - v0[1], o[2] - v0[2]];
        let u = (s[0] * p[0] + s[1] * p[1] + s[2] * p[2]) * inv_det;
        if !(0.0..=1.0).contains(&u) {
            return false;
        }

        let q = [
            s[1] * e1[2] - s[2] * e1[1],
            s[2] * e1[0] - s[0] * e1[2],
            s[0] * e1[1] - s[1] * e1[0],
        ];
        let v = (d[0] * q[0] + d[1] * q[1] + d[2] * q[2]) * inv_det;
        if v < 0.0 || u + v > 1.0 {
            return false;
        }

        let dist = (e2[0] * q[0] + e2[1] * q[1] + e2[2] * q[2]) * inv_det;
        dist > T_MIN && dist < t_max
    }
}

/// How far along a ray a hit has to be to count, in units of `dir`.
///
/// Not a scale-relative epsilon and not the self-hit guard: that job belongs to the caller,
/// which launches from a point already lifted off the surface by a fraction of the local edge
/// length. This only excludes the exactly-degenerate `t = 0` case, so an absolute hair above
/// zero is all it needs to be.
const T_MIN: f32 = 1e-6;

/// Shift every interior node's child index by `by`, leaving leaf offsets alone.
fn shift_interiors(nodes: &mut [Node], by: u32) {
    for n in nodes.iter_mut() {
        if n.count == 0 {
            n.first += by;
        }
    }
}

/// Lomuto partition: everything satisfying `pred` first, returning where the two meet.
fn partition(xs: &mut [u32], pred: impl Fn(u32) -> bool) -> usize {
    let mut i = 0;
    for j in 0..xs.len() {
        if pred(xs[j]) {
            xs.swap(i, j);
            i += 1;
        }
    }
    i
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// A unit cube at the origin, 12 triangles, outward wound.
    fn cube() -> (Vec<f64>, Vec<u32>) {
        let v = vec![
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0,
            1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0,
        ];
        let f = vec![
            0, 2, 1, 0, 3, 2, // z = 0
            4, 5, 6, 4, 6, 7, // z = 1
            0, 1, 5, 0, 5, 4, // y = 0
            2, 3, 7, 2, 7, 6, // y = 1
            0, 4, 7, 0, 7, 3, // x = 0
            1, 2, 6, 1, 6, 5, // x = 1
        ];
        (v, f)
    }

    #[test]
    fn ray_through_the_middle_is_blocked() {
        let (v, f) = cube();
        let bvh = Bvh::build(&v, &f);
        // Straight at the cube from outside: two walls in the way.
        assert!(bvh.occluded([0.5, 0.5, -1.0], [0.0, 0.0, 1.0], 10.0));
    }

    #[test]
    fn ray_that_misses_is_not_blocked() {
        let (v, f) = cube();
        let bvh = Bvh::build(&v, &f);
        assert!(!bvh.occluded([5.0, 5.0, -1.0], [0.0, 0.0, 1.0], 10.0));
    }

    /// The segment is finite: a wall past its far end does not block it.
    #[test]
    fn segment_stops_at_its_length() {
        let (v, f) = cube();
        let bvh = Bvh::build(&v, &f);
        assert!(!bvh.occluded([0.5, 0.5, -1.0], [0.0, 0.0, 1.0], 0.5));
        assert!(bvh.occluded([0.5, 0.5, -1.0], [0.0, 0.0, 1.0], 1.5));
    }

    /// From inside, every direction is blocked; from outside and pointing away, none is.
    #[test]
    fn inside_is_enclosed() {
        let (v, f) = cube();
        let bvh = Bvh::build(&v, &f);
        for d in [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            [0.6, -0.3, 0.7],
        ] {
            assert!(bvh.occluded([0.5, 0.5, 0.5], d, 10.0), "{d:?}");
            assert!(!bvh.occluded([0.5, 0.5, 2.0], [d[0], d[1], d[2].abs()], 10.0));
        }
    }

    /// A ray launched just off a face must not be stopped by that face.
    #[test]
    fn launching_off_a_face_escapes() {
        let (v, f) = cube();
        let bvh = Bvh::build(&v, &f);
        assert!(!bvh.occluded([0.5, 0.5, 1.0 + 1e-4], [0.0, 0.0, 1.0], 10.0));
    }

    /// Axis-aligned rays put a zero in the slab test's reciprocals; the traversal has to
    /// survive the infinities that produces.
    #[test]
    fn axis_aligned_rays_are_handled() {
        let (v, f) = cube();
        let bvh = Bvh::build(&v, &f);
        // Grazes along the z = 0 plane, well outside the cube in y.
        assert!(!bvh.occluded([-1.0, 5.0, 0.0], [1.0, 0.0, 0.0], 10.0));
        assert!(bvh.occluded([-1.0, 0.5, 0.5], [1.0, 0.0, 0.0], 10.0));
    }

    #[test]
    fn empty_mesh_blocks_nothing() {
        let bvh = Bvh::build(&[], &[]);
        assert!(!bvh.occluded([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 1.0));
    }

    /// Enough triangles to exercise the recursion, the parallel cutoff and the leaf-offset
    /// patching: a grid of quads in the z = 0 plane, hit one cell at a time.
    #[test]
    fn many_triangles_keep_their_identity() {
        let n = 60;
        let mut v = Vec::new();
        for i in 0..=n {
            for j in 0..=n {
                v.extend_from_slice(&[i as f64, j as f64, 0.0]);
            }
        }
        let idx = |i: usize, j: usize| (i * (n + 1) + j) as u32;
        let mut f = Vec::new();
        for i in 0..n {
            for j in 0..n {
                f.extend_from_slice(&[idx(i, j), idx(i + 1, j), idx(i + 1, j + 1)]);
                f.extend_from_slice(&[idx(i, j), idx(i + 1, j + 1), idx(i, j + 1)]);
            }
        }
        let bvh = Bvh::build(&v, &f);

        // Every cell centre is covered, and only the sheet is.
        for i in 0..n {
            for j in 0..n {
                let p = [i as f64 + 0.5, j as f64 + 0.5, -1.0];
                assert!(bvh.occluded(p, [0.0, 0.0, 1.0], 10.0), "{i} {j}");
            }
        }
        assert!(!bvh.occluded([-1.0, -1.0, -1.0], [0.0, 0.0, 1.0], 10.0));
        assert!(!bvh.occluded(
            [n as f64 + 1.0, n as f64 + 1.0, -1.0],
            [0.0, 0.0, 1.0],
            10.0
        ));
    }

    /// Coordinates far from the origin are what the re-centring is for.
    #[test]
    fn far_from_the_origin_still_resolves() {
        let (v, f) = cube();
        let off = 1.0e6;
        let v: Vec<f64> = v.iter().map(|x| x + off).collect();
        let bvh = Bvh::build(&v, &f);
        assert!(bvh.occluded([0.5 + off, 0.5 + off, off - 1.0], [0.0, 0.0, 1.0], 10.0));
        assert!(!bvh.occluded([5.0 + off, 5.0 + off, off - 1.0], [0.0, 0.0, 1.0], 10.0));
    }
}
