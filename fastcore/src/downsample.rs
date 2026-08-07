//! Changing how densely a skeleton is sampled, without changing what it is.
//!
//! Six operations, in three families:
//!
//! - **Dropping nodes** — [`downsample_skeleton`] (keep every Nth), [`simplify_rdp`]
//!   (Ramer-Douglas-Peucker) and [`simplify_vw`] (Visvalingam-Whyatt). All three hand
//!   back `(kept, new_parents, new_weights, node_map)`, the same tuple as
//!   [`crate::dag::simplify_skeleton`], with each replacement edge carrying the summed
//!   length of the chain it stands in for — so total cable length survives exactly.
//! - **Adding nodes** — [`resample_skeleton`] places interpolated points at a fixed
//!   spacing, and reports for each one which original edge it sits on so the caller can
//!   interpolate radii and anything else it tracks per node.
//! - **Moving nodes** — [`smooth_skeleton`] (moving average) and
//!   [`smooth_skeleton_gaussian`] take the jitter out of a traced arbor without changing
//!   the node count at all.
//!
//! # Following the data
//!
//! A skeleton rarely travels alone: synapses, soma tags and manual annotations all hang off
//! particular nodes, and an operation that renumbers the nodes strands them. Everything here
//! that changes the node table therefore also reports where each *input* node's data should
//! go — `node_map` for the droppers and for [`Resampled`], indexed by input node and valued
//! in output nodes, the same direction as [`crate::simplify::simplify_mesh`]'s vertex map.
//! It is total in both cases: every input node names exactly one output node, the nearest
//! along the neurite, with ties going proximal.
//!
//! The two smoothers need no such map and do not have one. They move coordinates only —
//! every node keeps its ID and its parent — so anything attached to a node is still attached
//! to it afterwards. Only a *copy* of a node's position taken beforehand goes stale.
//!
//! # The segment model
//!
//! Every operation here works on the skeleton's *linear segments* — the runs between
//! roots, branch points and leafs, as [`crate::dag::break_segments`] returns them — and
//! never moves or drops the nodes at their ends. That is what makes all six of them
//! topology-preserving: the leaf count, the branch count and the shape of the tree come
//! out the other side untouched, and only the sampling *along* each neurite changes.
//!
//! It is also what makes them parallel. Segments meet only at their endpoints, and
//! endpoints are exactly the nodes nothing here is allowed to touch, so the interiors
//! partition the remaining nodes and every segment can be processed on its own thread.
//!
//! # Attribution
//!
//! The three linestring algorithms — RDP, Visvalingam-Whyatt, and resampling at a fixed
//! spacing — are ports of Chris L. Barnes' [`simples`](https://github.com/clbarnes/simples)
//! (MIT), which solves the same "simplify a polyline whose endpoints are pinned" problem
//! this module needs, one segment at a time. MIT is compatible with this crate's GPL-3.0.
//!
//! # Divergences from upstream
//!
//! Four, each because a skeleton is not a 2D map linestring:
//!
//! - **RDP is iterative, not recursive.** Upstream recurses once per kept point; a
//!   100k-node unbranched neurite — which is a shape this crate is explicitly tested
//!   against — would overflow the stack. Same output, explicit work stack.
//! - **Triangle areas come from the Gram determinant, not Heron's formula.** Heron loses
//!   most of its significant digits on a sliver triangle, and in Visvalingam-Whyatt
//!   *every* triangle that matters is a sliver: the whole algorithm is a search for the
//!   flattest one. `0.5 * sqrt(|u|²|v|² - (u·v)²)` is stable there and costs less.
//! - **Visvalingam-Whyatt is parameterised by an area threshold, not a point count.** A
//!   target point count has to be split across a skeleton's thousands of segments somehow,
//!   and every way of doing that is arbitrary; a threshold is a scale, applies to each
//!   segment on its own terms, and leaves the segments independent (so, parallel).
//! - **The smoothing kernel measures distance along the segment, not between the points.**
//!   Node spacing in a traced skeleton varies by an order of magnitude, and a kernel over
//!   straight-line distance quietly weighs a hairpin's far arm as if it were a neighbour.
//!   Upstream's Gaussian is also unusable as written: its cut-off test is inverted
//!   (`smooth.rs:127` returns `None` when the weight is *above* the cut-off), so it drops
//!   the near points and keeps the far ones.

use ndarray::{Array, Array1, Array2, ArrayView1, ArrayView2};
use num::Float;
use rayon::prelude::*;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::ops::AddAssign;

use crate::dag::{break_segments, compact, rewire_kept, topology_nodes};
use crate::threads::with_pool;

// ---------------------------------------------------------------------------- geometry
//
// Points are pulled out of the `(N, D)` coordinate array into a flat, per-segment buffer
// before any of the algorithms run. Two reasons: the inner loops then walk contiguous
// memory instead of striding through an ndarray by node index, and every routine below
// works in whatever dimensionality it is handed rather than hard-coding three.

/// Gather a segment's coordinates into a flat `len * d` buffer, in segment order.
fn gather(coords: &ArrayView2<f64>, seg: &[i32], d: usize) -> Vec<f64> {
    let mut buf = Vec::with_capacity(seg.len() * d);
    for &node in seg {
        for k in 0..d {
            buf.push(coords[[node as usize, k]]);
        }
    }
    buf
}

/// Point `i` of a flat buffer.
#[inline]
fn point(pts: &[f64], d: usize, i: usize) -> &[f64] {
    &pts[i * d..(i + 1) * d]
}

/// Squared distance between two points of a flat buffer.
#[inline]
fn dist2(pts: &[f64], d: usize, a: usize, b: usize) -> f64 {
    let (pa, pb) = (point(pts, d, a), point(pts, d, b));
    (0..d).map(|k| (pa[k] - pb[k]).powi(2)).sum()
}

/// Squared distance from point `p` to the *segment* `a`-`b` (not the infinite line): the
/// perpendicular distance where the projection falls inside, and the distance to the
/// nearer end where it does not.
///
/// `len2` is the squared length of `a`-`b`, hoisted out because RDP holds it fixed across
/// a whole scan. A degenerate `a == b` never divides: the projection test sends it down
/// the `<= 0` branch.
fn point_segment_dist2(pts: &[f64], d: usize, a: usize, b: usize, p: usize, len2: f64) -> f64 {
    let (pa, pb, pp) = (point(pts, d, a), point(pts, d, b), point(pts, d, p));

    let along: f64 = (0..d).map(|k| (pp[k] - pa[k]) * (pb[k] - pa[k])).sum();
    if along <= 0.0 {
        return dist2(pts, d, p, a);
    }
    if along >= len2 {
        return dist2(pts, d, p, b);
    }

    let t = along / len2;
    (0..d)
        .map(|k| (pp[k] - (pa[k] + t * (pb[k] - pa[k]))).powi(2))
        .sum()
}

/// Area of the triangle `a`-`b`-`c`, via the Gram determinant.
///
/// `0.5 * sqrt(|u|²|v|² - (u·v)²)` for `u = b - a`, `v = c - a`. Exact in any
/// dimensionality (unlike a cross product) and stable on the near-degenerate triangles
/// Visvalingam-Whyatt spends all its time on (unlike Heron's formula). Cancellation can
/// still push the radicand a hair below zero on a perfectly collinear triple, hence the
/// clamp.
fn tri_area(pts: &[f64], d: usize, a: usize, b: usize, c: usize) -> f64 {
    let (pa, pb, pc) = (point(pts, d, a), point(pts, d, b), point(pts, d, c));

    let (mut uu, mut vv, mut uv) = (0.0, 0.0, 0.0);
    for k in 0..d {
        let u = pb[k] - pa[k];
        let v = pc[k] - pa[k];
        uu += u * u;
        vv += v * v;
        uv += u * v;
    }

    0.5 * (uu * vv - uv * uv).max(0.0).sqrt()
}

/// Cumulative distance along a segment, `arc[0] == 0`.
fn arc_lengths(pts: &[f64], d: usize) -> Vec<f64> {
    let n = pts.len() / d;
    let mut arc = vec![0.0; n];
    for i in 1..n {
        arc[i] = arc[i - 1] + dist2(pts, d, i - 1, i).sqrt();
    }
    arc
}

// --------------------------------------------------------------------------- plumbing

/// Check that `coords` describes the same nodes as `parents`, and return its dimensionality.
fn check_coords(parents: &ArrayView1<i32>, coords: &ArrayView2<f64>) -> usize {
    assert_eq!(
        coords.nrows(),
        parents.len(),
        "`coords` must have one row per node"
    );
    coords.ncols()
}

/// Reject a scale parameter that is not a finite, non-negative number.
///
/// Checked here rather than in each binding, for the reason [`crate::simplify`] gives for
/// the same rule: this is where the parameter *means* something. NaN is the case worth
/// spelling out — every comparison against it is false, so a NaN `epsilon` would not error,
/// it would silently drop every node it was meant to protect.
fn check_scale(name: &str, value: f64) {
    assert!(
        value.is_finite() && value >= 0.0,
        "`{name}` must be a finite, non-negative number, got {value}"
    );
}

/// Run a per-segment "which interior nodes survive" rule and rewire what is left.
///
/// The shared body of all three node-dropping entry points: they differ only in the rule,
/// so everything either side of it -- deriving the segments, seeding the mask with the
/// nodes that carry topology, folding in `preserve`, and handing the result to
/// [`rewire_kept`] -- lives here once.
fn drop_with<T, F>(
    parents: &ArrayView1<i32>,
    weights: &Option<Array1<T>>,
    preserve: &Option<Array1<bool>>,
    threads: Option<usize>,
    rule: F,
) -> (Vec<i32>, Array1<i32>, Option<Vec<T>>, Array1<i32>)
where
    T: Float + AddAssign,
    F: Fn(&[i32]) -> Vec<i32> + Sync + Send,
{
    if let Some(p) = preserve {
        assert_eq!(
            p.len(),
            parents.len(),
            "`preserve` must have one entry per node"
        );
    }

    // The mask starts as the nodes that carry topology -- which are exactly the nodes that
    // appear as a segment *endpoint*, plus the isolated roots that appear in no segment at
    // all and so are left alone by construction.
    let segments = break_segments(parents);
    let mut keep = topology_nodes(parents);

    let extra: Vec<Vec<i32>> = with_pool(threads, || segments.par_iter().map(|s| rule(s)).collect());
    for node in extra.into_iter().flatten() {
        keep[node as usize] = true;
    }

    if let Some(p) = preserve {
        for (k, &flag) in keep.iter_mut().zip(p) {
            *k |= flag;
        }
    }

    rewire_kept(parents, &keep, weights)
}

// ---------------------------------------------------------------------- dropping nodes

/// Keep every `factor`-th node of every segment, and everything that carries topology.
///
/// The plain "make this skeleton smaller" operation, and `navis.downsample_neuron`: it
/// pays no attention to geometry, so it is the one to reach for when the skeleton is
/// already evenly sampled and you just want fewer nodes. Roots, branch points and leafs
/// always survive, which is why the result is still the same neuron -- only its
/// unbranched stretches are sampled `factor` times more coarsely.
///
/// Arguments:
///
/// - `parents`: array of parent indices (roots are negative)
/// - `factor`: keep one node in every `factor`, counting from each segment's distal end.
///   `1` keeps everything; the useful range starts at 2.
/// - `preserve`: optional length-`N` mask of extra nodes that must survive -- nodes
///   carrying synapses, say, or the ends of a region of interest
/// - `weights`: optional per-node length of the child->parent edge; `None` counts edges
///
/// Returns:
///
/// `(kept, new_parents, new_weights, node_map)` -- see [`crate::dag::simplify_skeleton`]
/// for the convention. Total cable length is preserved, and `node_map` says which surviving
/// node each input node's data belongs to now.
///
/// Panics if `factor` is 0.
pub fn downsample_skeleton<T>(
    parents: &ArrayView1<i32>,
    factor: usize,
    preserve: &Option<Array1<bool>>,
    weights: &Option<Array1<T>>,
) -> (Vec<i32>, Array1<i32>, Option<Vec<T>>, Array1<i32>)
where
    T: Float + AddAssign,
{
    assert!(factor >= 1, "`factor` must be >= 1");

    // Position within the segment, not node index: counting off the raw index would keep
    // an arbitrary scatter of nodes rather than an evenly spaced subset of each neurite.
    // `threads: None` because there is no arithmetic here to speak of -- this is a
    // memory-bound walk that gains nothing from its own thread pool.
    drop_with(parents, weights, preserve, None, |seg| {
        seg.iter()
            .step_by(factor)
            .copied()
            .collect()
    })
}

/// Drop the nodes that do not bend a neurite, by Ramer-Douglas-Peucker.
///
/// Where [`downsample_skeleton`] thins by counting, this thins by *shape*: a node
/// survives only if removing it would move the traced path by more than `epsilon`. Long
/// straight stretches collapse to their two ends while a tight curve keeps every node it
/// needs, so the same tolerance gives a much better skeleton per node than a fixed
/// factor does.
///
/// Arguments:
///
/// - `parents`: array of parent indices (roots are negative)
/// - `coords`: `(N, D)` array of node coordinates, one row per node
/// - `epsilon`: how far the simplified path may stray from the original, in the units of
///   `coords`. `0` still drops nodes that are *exactly* collinear, and nothing else.
/// - `preserve`: optional length-`N` mask of extra nodes that must survive
/// - `weights`: optional per-node length of the child->parent edge; `None` counts edges
/// - `threads`: cap on the rayon worker count for this call; `None` uses the global pool.
///   See [`crate::threads`] for which of the two levers to reach for.
///
/// Returns:
///
/// `(kept, new_parents, new_weights, node_map)` -- see [`crate::dag::simplify_skeleton`]
/// for the convention. Total cable length is preserved: the replacement edges carry the
/// length of the chains they stand in for, *not* the shorter straight line the simplified
/// path takes. Distances therefore stay right even where the geometry has been cut across.
///
/// # Complexity
///
/// `O(n log n)` per segment on the smooth curves a traced neurite actually follows, but
/// RDP is quadratic in the worst case, and its worst case is a segment on which it keeps
/// almost everything: each split then peels off one node and re-scans a span one shorter.
/// An `epsilon` well below the tracing jitter on a very long unbranched neurite is the way
/// to hit it. If that is the regime you are in, you want [`downsample_skeleton`] or a
/// larger `epsilon` -- an RDP that keeps every node is not buying anything anyway.
pub fn simplify_rdp<T>(
    parents: &ArrayView1<i32>,
    coords: &ArrayView2<f64>,
    epsilon: f64,
    preserve: &Option<Array1<bool>>,
    weights: &Option<Array1<T>>,
    threads: Option<usize>,
) -> (Vec<i32>, Array1<i32>, Option<Vec<T>>, Array1<i32>)
where
    T: Float + AddAssign,
{
    let d = check_coords(parents, coords);
    check_scale("epsilon", epsilon);
    let epsilon_sq = epsilon * epsilon;

    drop_with(parents, weights, preserve, threads, |seg| {
        rdp_segment(seg, coords, d, epsilon_sq)
    })
}

/// The interior nodes of one segment that RDP keeps.
fn rdp_segment(seg: &[i32], coords: &ArrayView2<f64>, d: usize, epsilon_sq: f64) -> Vec<i32> {
    let n = seg.len();
    if n <= 2 {
        return Vec::new(); // endpoints only; nothing in between to decide about
    }
    let pts = gather(coords, seg, d);

    // Iterative, with an explicit stack of the spans still to split -- see the module
    // docs. Each span is bounded by two points already kept, and contributes at most one
    // more, so the stack cannot outgrow the segment.
    let mut kept = vec![false; n];
    let mut stack = vec![(0usize, n - 1)];

    while let Some((start, end)) = stack.pop() {
        if end <= start + 1 {
            continue; // adjacent: no interior points to weigh
        }

        let len2 = dist2(&pts, d, start, end);
        let mut worst = (start, f64::NEG_INFINITY);
        for i in (start + 1)..end {
            let dist = point_segment_dist2(&pts, d, start, end, i, len2);
            if dist > worst.1 {
                worst = (i, dist);
            }
        }

        // Strictly greater, so `epsilon = 0` still drops exactly-collinear points and
        // keeps everything else -- the standard reading of the tolerance.
        if worst.1 > epsilon_sq {
            kept[worst.0] = true;
            stack.push((start, worst.0));
            stack.push((worst.0, end));
        }
    }

    let mut out = Vec::with_capacity(n - 2);
    out.extend((1..n - 1).filter(|&i| kept[i]).map(|i| seg[i]));
    out
}

/// Drop the nodes that contribute least area, by Visvalingam-Whyatt.
///
/// The other geometric thinning. Where [`simplify_rdp`] asks how far the path *moves*,
/// this asks how much area each node adds to it, and repeatedly removes whichever node
/// adds least. The difference shows under aggressive simplification: RDP will happily
/// keep one spike and flatten everything around it, while Visvalingam-Whyatt sheds detail
/// evenly and so keeps a neurite looking like itself.
///
/// Arguments:
///
/// - `parents`: array of parent indices (roots are negative)
/// - `coords`: `(N, D)` array of node coordinates, one row per node
/// - `min_area`: remove a node while the triangle it forms with its two surviving
///   neighbours is smaller than this, in the *squared* units of `coords`. `0` or less is
///   a no-op.
/// - `preserve`: optional length-`N` mask of extra nodes that must survive
/// - `weights`: optional per-node length of the child->parent edge; `None` counts edges
/// - `threads`: cap on the rayon worker count for this call; `None` uses the global pool
///
/// Returns:
///
/// `(kept, new_parents, new_weights, node_map)` -- see [`crate::dag::simplify_skeleton`]
/// for the convention. Total cable length is preserved, as in [`simplify_rdp`].
pub fn simplify_vw<T>(
    parents: &ArrayView1<i32>,
    coords: &ArrayView2<f64>,
    min_area: f64,
    preserve: &Option<Array1<bool>>,
    weights: &Option<Array1<T>>,
    threads: Option<usize>,
) -> (Vec<i32>, Array1<i32>, Option<Vec<T>>, Array1<i32>)
where
    T: Float + AddAssign,
{
    let d = check_coords(parents, coords);
    check_scale("min_area", min_area);

    drop_with(parents, weights, preserve, threads, |seg| {
        vw_segment(seg, coords, d, min_area)
    })
}

/// One node's standing in the removal queue.
///
/// The area is stored as its raw IEEE bit pattern, as [`crate::mesh`]'s heap entries store
/// their distances: for the non-negative, non-NaN values [`tri_area`] returns, the integer
/// order *is* the float order, which buys a derived `Ord` -- no `partial_cmp().unwrap()`,
/// no NaN path -- and an integer compare in the sift loop.
///
/// Field order is the ordering: area first, then position, so that two equally flat nodes
/// are settled on the lower position rather than on the heap's internal order. Without that
/// tie-break the same skeleton could simplify two ways. The whole entry goes into the heap
/// under [`Reverse`], since [`BinaryHeap`] is a max-heap and we want the flattest node.
///
/// `stamp` is the lazy-deletion counter: removing a node invalidates its neighbours'
/// recorded areas, and re-stamping them is cheaper than finding and mutating their entries
/// in place. It sits last so it never influences the order.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct Candidate {
    area_bits: u64,
    pos: usize,
    stamp: u32,
}

impl Candidate {
    fn new(area: f64, pos: usize, stamp: u32) -> Self {
        Self {
            area_bits: area.to_bits(),
            pos,
            stamp,
        }
    }

    fn area(&self) -> f64 {
        f64::from_bits(self.area_bits)
    }
}

/// The interior nodes of one segment that Visvalingam-Whyatt keeps.
fn vw_segment(seg: &[i32], coords: &ArrayView2<f64>, d: usize, min_area: f64) -> Vec<i32> {
    let n = seg.len();
    if n <= 2 {
        return Vec::new();
    }
    if min_area <= 0.0 {
        return seg[1..n - 1].to_vec(); // no triangle is smaller than zero area
    }
    let pts = gather(coords, seg, d);

    // The survivors as a doubly-linked list over positions, so removing a node is O(1)
    // and its neighbours are always the *current* ones rather than the original ones --
    // which is the whole point of the algorithm: each removal flattens the triangles
    // either side of it, and they have to be re-weighed against the new neighbourhood.
    let mut prev: Vec<usize> = (0..n).map(|i| i.saturating_sub(1)).collect();
    let mut next: Vec<usize> = (0..n).map(|i| (i + 1).min(n - 1)).collect();
    let mut alive = vec![true; n];
    let mut stamp = vec![0u32; n];

    let mut heap: BinaryHeap<Reverse<Candidate>> = (1..n - 1)
        .map(|pos| Reverse(Candidate::new(tri_area(&pts, d, pos - 1, pos, pos + 1), pos, 0)))
        .collect();

    while let Some(Reverse(cand)) = heap.pop() {
        if !alive[cand.pos] || cand.stamp != stamp[cand.pos] {
            continue; // superseded by a re-stamp, or already removed
        }
        // The heap's minimum is a lower bound over every live candidate, and re-stamped
        // entries are pushed before we ever get here, so this is the whole segment done.
        if cand.area() >= min_area {
            break;
        }

        alive[cand.pos] = false;
        let (before, after) = (prev[cand.pos], next[cand.pos]);
        next[before] = after;
        prev[after] = before;

        for m in [before, after] {
            if m > 0 && m < n - 1 && alive[m] {
                stamp[m] += 1;
                heap.push(Reverse(Candidate::new(
                    tri_area(&pts, d, prev[m], m, next[m]),
                    m,
                    stamp[m],
                )));
            }
        }
    }

    let mut out = Vec::with_capacity(n - 2);
    out.extend((1..n - 1).filter(|&i| alive[i]).map(|i| seg[i]));
    out
}

// ------------------------------------------------------------------------ adding nodes

/// A resampled skeleton: what [`resample_skeleton`] hands back.
///
/// Everything is indexed by *output* node, which is a new index space -- the skeleton has
/// a different number of nodes than it went in with. [`Resampled::source`] and
/// [`Resampled::alpha`] are the bridge back to the input, and exist so that this crate
/// does not have to know what else a caller keeps per node: radius, label, confidence and
/// anything else interpolate the same way, from the same two indices and the same
/// fraction.
pub struct Resampled {
    /// `(M, )` parent index per output node, negative for roots.
    pub parents: Array1<i32>,
    /// `(M, D)` coordinates.
    pub coords: Array2<f64>,
    /// `(M, 2)` the *input* node indices of the edge each output node sits on: column 0
    /// the child (distal) end, column 1 the parent (proximal) end. A node carried over
    /// unchanged has its own index in both columns.
    pub source: Array2<i32>,
    /// `(M, )` how far along that edge, from the child end. Zero for a node carried over
    /// unchanged, so `attr[source[:, 0]] * (1 - alpha) + attr[source[:, 1]] * alpha`
    /// interpolates any per-node quantity over the whole output, carried-over nodes
    /// included.
    pub alpha: Array1<f64>,
    /// `(N, )` the reverse direction: for each *input* node, the output node nearest to it
    /// along the neurite, with ties going proximal. Total -- every input node gets one.
    ///
    /// [`Resampled::source`] and [`Resampled::alpha`] carry per-node *columns* forward, but
    /// they cannot answer "where did node 12345 go": an input node between two output nodes
    /// has no output row of its own, so the mapping is not invertible. That is the question
    /// anything *attached* to a node asks -- a synapse, a soma tag, a manual annotation --
    /// and this answers it. Carried-over nodes map to themselves.
    pub node_map: Array1<i32>,
}

/// Place nodes at a fixed spacing along every neurite.
///
/// The inverse problem to [`downsample_skeleton`]: rather than thinning what is there,
/// this re-samples each segment from scratch, so a skeleton whose node density varies
/// tenfold between neurites comes out evenly sampled throughout. It is the step most
/// morphometrics want in front of them -- anything that averages a quantity *per node*
/// is otherwise weighted by how finely each neurite happened to be traced.
///
/// Each segment is divided into `round(length / spacing)` equal parts (at least one), so
/// both of its endpoints land exactly and no runt edge is left over at the end. Spacing
/// is therefore uniform *within* a segment and within half a node of `spacing` between
/// segments -- a segment shorter than `spacing / 2` collapses to a single straight edge.
///
/// Arguments:
///
/// - `parents`: array of parent indices (roots are negative)
/// - `coords`: `(N, D)` array of node coordinates, one row per node
/// - `spacing`: target distance between adjacent nodes, in the units of `coords`
/// - `threads`: cap on the rayon worker count for this call; `None` uses the global pool
///
/// Returns:
///
/// A [`Resampled`]. Output nodes are ordered: every carried-over node first, in input
/// order, then each segment's new interior nodes, distal to proximal, segments ordered by
/// their distal node -- so the result is reproducible and the first `K` rows still line up
/// with the input's roots, branch points and leafs.
///
/// Panics if `spacing` is not positive.
pub fn resample_skeleton(
    parents: &ArrayView1<i32>,
    coords: &ArrayView2<f64>,
    spacing: f64,
    threads: Option<usize>,
) -> Resampled {
    check_scale("spacing", spacing);
    assert!(spacing > 0.0, "`spacing` must be positive");
    let d = check_coords(parents, coords);
    let (segments, keep) = (break_segments(parents), topology_nodes(parents));

    // Carried-over nodes come first, in input order, and `position` maps an input node to
    // its output row. Placing them up front (rather than interleaving them with the new
    // nodes segment by segment) is what lets a caller line the two node tables up.
    let (kept, position) = compact(parents.len(), |idx| keep[idx]);

    let sampled: Vec<(Vec<Sample>, Vec<u32>)> = with_pool(threads, || {
        segments
            .par_iter()
            .map(|seg| resample_segment(seg, coords, d, spacing))
            .collect()
    });

    let n_new: usize = sampled.iter().map(|(s, _)| s.len()).sum();
    let m = kept.len() + n_new;

    let mut out_parents: Array1<i32> = Array::from_elem(m, -1);
    let mut out_coords: Array2<f64> = Array2::zeros((m, d));
    let mut out_source: Array2<i32> = Array2::zeros((m, 2));
    let mut out_alpha: Array1<f64> = Array1::zeros(m);
    let mut node_map: Array1<i32> = Array::from_elem(parents.len(), -1);

    // The carried-over nodes keep their coordinates exactly -- resampling must not nudge
    // a branch point, or the neurites meeting there would come apart.
    for (slot, &node) in kept.iter().enumerate() {
        for k in 0..d {
            out_coords[[slot, k]] = coords[[node as usize, k]];
        }
        out_source[[slot, 0]] = node;
        out_source[[slot, 1]] = node;
        node_map[node as usize] = slot as i32;
    }

    let mut offset = kept.len();
    for (seg, (samples, nearest)) in segments.iter().zip(sampled.iter()) {
        for (i, s) in samples.iter().enumerate() {
            let slot = offset + i;
            // The coordinates are re-derived here rather than carried on the `Sample`,
            // from the same three numbers a caller interpolates a radius with. Storing
            // them would be a heap allocation per output node for a value that is a
            // subtraction away.
            let (child, parent) = (s.child as usize, s.parent as usize);
            for k in 0..d {
                let (a, b) = (coords[[child, k]], coords[[parent, k]]);
                out_coords[[slot, k]] = a + s.alpha * (b - a);
            }
            out_source[[slot, 0]] = s.child;
            out_source[[slot, 1]] = s.parent;
            out_alpha[slot] = s.alpha;
        }

        // Re-thread the segment: distal endpoint -> new nodes, distal to proximal ->
        // proximal endpoint. Every non-root endpoint starts exactly one segment, so no
        // node's parent is written twice, and the roots keep the -1 they were built with.
        let mut chain: Vec<i32> = Vec::with_capacity(samples.len() + 2);
        chain.push(position[seg[0] as usize]);
        chain.extend((0..samples.len()).map(|i| (offset + i) as i32));
        chain.push(position[seg[seg.len() - 1] as usize]);
        for pair in chain.windows(2) {
            out_parents[pair[0] as usize] = pair[1];
        }

        // `chain` is also the segment's output nodes in order, which is what `nearest`
        // indexes into: the dropped interior nodes hand their data to whichever of them
        // ended up closest. The two endpoints already mapped to themselves above.
        for (i, &k) in nearest.iter().enumerate() {
            node_map[seg[i + 1] as usize] = chain[k as usize];
        }

        offset += samples.len();
    }

    Resampled {
        parents: out_parents,
        coords: out_coords,
        source: out_source,
        alpha: out_alpha,
        node_map,
    }
}

/// One interpolated node, before it knows its index.
///
/// Deliberately POD and 16 bytes: the coordinates are *derivable* from these three numbers
/// and the input, by exactly the interpolation [`Resampled::alpha`] documents, so carrying
/// them would be a heap allocation per output node for nothing.
#[derive(Copy, Clone)]
struct Sample {
    child: i32,
    parent: i32,
    alpha: f64,
}

/// How many equal parts a segment of length `total` is divided into.
///
/// A zero-length segment (every node coincident, which EM tracing does produce) has no
/// direction to sample along; the `max(1)` collapses it to a single edge instead of
/// dividing by zero.
fn divisions(total: f64, spacing: f64) -> usize {
    if total > 0.0 {
        ((total / spacing).round() as usize).max(1)
    } else {
        1
    }
}

/// One resampled segment: its new interior nodes distal to proximal, and where each
/// *original* interior node's data should go.
///
/// The second vector has one entry per node in `seg[1..n-1]`, indexing the segment's output
/// nodes in order -- `0` the distal endpoint, `1..=samples.len()` the new nodes, and
/// `samples.len() + 1` the proximal endpoint. Resolving that to a global index needs the
/// output offset, which only the caller knows.
fn resample_segment(
    seg: &[i32],
    coords: &ArrayView2<f64>,
    d: usize,
    spacing: f64,
) -> (Vec<Sample>, Vec<u32>) {
    let n = seg.len();
    // A two-node segment cannot be subdivided into more than the edge it already is unless
    // it is longer than `spacing`, and on a real arbor most segments are exactly two nodes
    // (twigs) -- 522k of 728k on the performance fixture. Answering those from one distance
    // keeps the common case out of `gather` and `arc_lengths` entirely.
    if n < 2 {
        return (Vec::new(), Vec::new());
    }
    if n == 2 {
        let total = (0..d)
            .map(|k| (coords[[seg[1] as usize, k]] - coords[[seg[0] as usize, k]]).powi(2))
            .sum::<f64>()
            .sqrt();
        // Deciding with the *same* expression the general path uses, rather than a
        // shortcut like `total <= 1.5 * spacing`, so the two cannot disagree on the
        // rounding boundary. Falling through is always safe; returning early when the
        // general path would have added a node would not be.
        if divisions(total, spacing) == 1 {
            // Both endpoints survive and there is nothing in between, so no node needs
            // rehousing either way.
            return (Vec::new(), Vec::new());
        }
    }

    let pts = gather(coords, seg, d);
    let arc = arc_lengths(&pts, d);
    let total = arc[n - 1];
    let parts = divisions(total, spacing);

    let mut out = Vec::with_capacity(parts.saturating_sub(1));
    let mut edge = 0usize;
    for i in 1..parts {
        let target = total * (i as f64) / (parts as f64);

        // `target` ascends, so the edge pointer only ever moves forward: the whole
        // segment is walked once across all samples, not once per sample.
        while edge + 2 < n && arc[edge + 1] < target {
            edge += 1;
        }

        let span = arc[edge + 1] - arc[edge];
        let alpha = if span > 0.0 {
            ((target - arc[edge]) / span).clamp(0.0, 1.0)
        } else {
            0.0
        };

        out.push(Sample {
            child: seg[edge],
            parent: seg[edge + 1],
            alpha,
        });
    }

    // Output node `k` sits at arc length `total * k / parts` by construction, so the nearest
    // one to an original node is a division away -- no second walk over the segment. `round`
    // breaks a tie upwards, which is towards the root, matching `rewire_kept`.
    let nearest = (1..n - 1)
        .map(|i| {
            let k = if total > 0.0 {
                (arc[i] / total * parts as f64).round() as usize
            } else {
                0 // a segment of coincident nodes: every one of them is "at" the distal end
            };
            k.min(parts) as u32
        })
        .collect();

    (out, nearest)
}

// ------------------------------------------------------------------------ moving nodes

/// Smooth a skeleton with a moving average along each neurite.
///
/// Takes the tracing jitter out of a skeleton without touching its topology or its node
/// count: every node keeps its identity and its parent, and only its coordinates move.
/// Roots, branch points and leafs are pinned -- a branch point that drifted would drag
/// three neurites apart -- so this is safe to run before measuring angles, tortuosity or
/// tangent vectors, all of which a raw traced skeleton overstates.
///
/// The window shrinks symmetrically as it approaches a segment's ends, which keeps the
/// smoothed path centred on the original rather than letting it shrink towards the middle.
///
/// Arguments:
///
/// - `parents`: array of parent indices (roots are negative)
/// - `coords`: `(N, D)` array of node coordinates, one row per node
/// - `window`: nodes in the window, counting the node itself. Even values round down to
///   the odd value below, since the window is symmetric. `0` and `1` are no-ops.
/// - `threads`: cap on the rayon worker count for this call; `None` uses the global pool
///
/// Returns:
///
/// An `(N, D)` array of new coordinates, in the input's node order. Nodes that belong to
/// no segment -- an isolated root -- are copied through.
pub fn smooth_skeleton(
    parents: &ArrayView1<i32>,
    coords: &ArrayView2<f64>,
    window: usize,
    threads: Option<usize>,
) -> Array2<f64> {
    let d = check_coords(parents, coords);
    let half = window / 2;

    smooth_with(parents, coords, d, threads, |pts, out| {
        let n = pts.len() / d;
        if half == 0 {
            out.extend_from_slice(&pts[d..(n - 1) * d]);
            return;
        }

        // Prefix sums, so a window costs one subtraction however wide it is. The
        // alternative -- re-summing per node -- is O(n * window), which is the whole cost
        // of the function on a densely traced neuron.
        let mut prefix = vec![0.0; (n + 1) * d];
        for i in 0..n {
            for k in 0..d {
                prefix[(i + 1) * d + k] = prefix[i * d + k] + pts[i * d + k];
            }
        }

        // The window shrinks to whatever fits symmetrically, so a node one step in from
        // an end is averaged over three rather than being pulled inwards by a lopsided
        // window.
        for pos in 1..n - 1 {
            let w = half.min(pos).min(n - 1 - pos);
            let (lo, hi) = (pos - w, pos + w);
            let count = (hi - lo + 1) as f64;
            out.extend((0..d).map(|k| (prefix[(hi + 1) * d + k] - prefix[lo * d + k]) / count));
        }
    })
}

/// Smooth a skeleton with a Gaussian kernel along each neurite.
///
/// The same operation as [`smooth_skeleton`] with a softer, scale-based kernel: `sigma` is
/// a distance in the units of `coords` rather than a count of nodes, so the amount of
/// smoothing does not change when the skeleton is resampled. That is usually what you
/// want -- and it is why the kernel measures distance *along* the neurite rather than
/// between the points, which would let the far arm of a hairpin pull on the near one.
///
/// Segment ends are pinned by reflecting the neurite about them, so a node one step in
/// from a leaf is smoothed against a symmetric neighbourhood rather than being dragged
/// inwards by a one-sided one.
///
/// Arguments:
///
/// - `parents`: array of parent indices (roots are negative)
/// - `coords`: `(N, D)` array of node coordinates, one row per node
/// - `sigma`: kernel width, as a distance along the neurite
/// - `truncate`: how many `sigma` out to keep summing; 4 covers all but 1e-4 of the mass
/// - `threads`: cap on the rayon worker count for this call; `None` uses the global pool
///
/// Returns:
///
/// An `(N, D)` array of new coordinates, in the input's node order.
///
/// Panics if `sigma` is not positive or `truncate` is negative.
pub fn smooth_skeleton_gaussian(
    parents: &ArrayView1<i32>,
    coords: &ArrayView2<f64>,
    sigma: f64,
    truncate: f64,
    threads: Option<usize>,
) -> Array2<f64> {
    check_scale("sigma", sigma);
    assert!(sigma > 0.0, "`sigma` must be positive");
    check_scale("truncate", truncate);
    let d = check_coords(parents, coords);

    let cutoff = truncate * sigma;
    let denom = 2.0 * sigma * sigma;
    let weigh = |dist: f64| (-dist * dist / denom).exp();

    smooth_with(parents, coords, d, threads, |pts, out| {
        let n = pts.len() / d;
        let arc = arc_lengths(pts, d);

        // One scratch accumulator for the whole segment rather than one per node: the
        // inner loop is a handful of multiply-adds, so a heap allocation per node would
        // cost several times the arithmetic it carries.
        let mut acc = vec![0.0; d];

        for pos in 1..n - 1 {
            acc.copy_from_slice(point(pts, d, pos));
            let mut total = 1.0; // the node itself, at distance 0

            // Proximal, then -- if we ran out of neurite before running out of kernel --
            // its mirror image beyond the endpoint. The mirror is folded into the
            // accumulator rather than materialised: `2 * about - p` is the same three
            // subtractions whether or not it passes through a `Vec` on the way.
            let mut q = pos + 1;
            while q < n && arc[q] - arc[pos] <= cutoff {
                let (w, p) = (weigh(arc[q] - arc[pos]), point(pts, d, q));
                for k in 0..d {
                    acc[k] += p[k] * w;
                }
                total += w;
                q += 1;
            }
            if q == n {
                for j in 1..n {
                    let dist = 2.0 * arc[n - 1] - arc[n - 1 - j] - arc[pos];
                    if dist > cutoff {
                        break;
                    }
                    let w = weigh(dist);
                    let (about, p) = (point(pts, d, n - 1), point(pts, d, n - 1 - j));
                    for k in 0..d {
                        acc[k] += (2.0 * about[k] - p[k]) * w;
                    }
                    total += w;
                }
            }

            // ...and the same distally.
            let mut q = pos;
            while q > 0 && arc[pos] - arc[q - 1] <= cutoff {
                q -= 1;
                let (w, p) = (weigh(arc[pos] - arc[q]), point(pts, d, q));
                for k in 0..d {
                    acc[k] += p[k] * w;
                }
                total += w;
            }
            if q == 0 {
                for j in 1..n {
                    let dist = arc[pos] + arc[j] - 2.0 * arc[0];
                    if dist > cutoff {
                        break;
                    }
                    let w = weigh(dist);
                    let (about, p) = (point(pts, d, 0), point(pts, d, j));
                    for k in 0..d {
                        acc[k] += (2.0 * about[k] - p[k]) * w;
                    }
                    total += w;
                }
            }

            out.extend(acc.iter().map(|v| v / total));
        }
    })
}

/// Run a per-segment smoothing kernel and assemble the result.
///
/// The shared frame of both smoothers. Each kernel is handed one segment's coordinates as
/// a flat `n * d` buffer and appends the new positions of its *interior* nodes -- also
/// flat, also in segment order, so `(n - 2) * d` values. Endpoints are not the kernel's to
/// move, so they are not its to report.
///
/// Flat buffers rather than a point per node because that is the difference between two
/// allocations per segment and one per node; at a million nodes the latter costs several
/// times the arithmetic it carries, and leaves throughput at the mercy of the platform
/// allocator. The `n <= 2` guard and the gather live here so neither kernel repeats them.
///
/// Segment interiors are disjoint -- endpoints are the only shared nodes, and no smoother
/// touches them -- so the write-back cannot collide however the segments were scheduled.
fn smooth_with<F>(
    parents: &ArrayView1<i32>,
    coords: &ArrayView2<f64>,
    d: usize,
    threads: Option<usize>,
    kernel: F,
) -> Array2<f64>
where
    F: Fn(&[f64], &mut Vec<f64>) + Sync + Send,
{
    let segments = break_segments(parents);

    let moved: Vec<Vec<f64>> = with_pool(threads, || {
        segments
            .par_iter()
            .map(|seg| {
                if seg.len() <= 2 {
                    return Vec::new(); // endpoints only; nothing to move
                }
                let pts = gather(coords, seg, d);
                let mut out = Vec::with_capacity((seg.len() - 2) * d);
                kernel(&pts, &mut out);
                out
            })
            .collect()
    });

    let mut out = coords.to_owned();
    for (seg, positions) in segments.iter().zip(moved.iter()) {
        for (i, &node) in seg.iter().enumerate().take(seg.len() - 1).skip(1) {
            let row = &positions[(i - 1) * d..i * d];
            for k in 0..d {
                out[[node as usize, k]] = row[k];
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, Array2};

    /// A straight chain of `n` nodes at unit spacing along x, rooted at 0.
    fn chain(n: usize) -> (Array1<i32>, Array2<f64>) {
        let parents: Array1<i32> = (0..n as i32).map(|i| i - 1).collect();
        let coords = Array2::from_shape_fn((n, 3), |(i, k)| if k == 0 { i as f64 } else { 0.0 });
        (parents, coords)
    }

    /// The 7-node tree from the `dag.rs` docstrings: a root, a branch at 1, two arms.
    fn tree() -> Array1<i32> {
        arr1(&[-1, 0, 1, 2, 1, 4, 5])
    }

    fn no_weights() -> Option<Array1<f32>> {
        None
    }

    // ------------------------------------------------------------------ downsampling

    #[test]
    fn downsample_factor_one_keeps_everything() {
        let (parents, _) = chain(10);
        let (kept, new_parents, _, _) =
            downsample_skeleton(&parents.view(), 1, &None, &no_weights());
        assert_eq!(kept.len(), 10);
        assert_eq!(new_parents, parents);
    }

    #[test]
    fn downsample_keeps_every_nth_and_both_ends() {
        let (parents, _) = chain(11);
        // The chain runs 10 (leaf) -> 0 (root), so positions count down from the leaf.
        let (kept, _, _, _) = downsample_skeleton(&parents.view(), 2, &None, &no_weights());
        assert_eq!(kept, vec![0, 2, 4, 6, 8, 10]);

        let (kept, _, _, _) = downsample_skeleton(&parents.view(), 5, &None, &no_weights());
        assert_eq!(kept, vec![0, 5, 10]);
    }

    #[test]
    fn downsample_preserves_topology_nodes_and_cable() {
        let parents = tree();
        let weights: Array1<f32> = arr1(&[0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]);
        let (kept, _, new_weights, _) =
            downsample_skeleton(&parents.view(), 100, &None, &Some(weights.clone()));

        // Root 0, branch 1, leafs 3 and 6 survive a factor nothing else could.
        assert_eq!(kept, vec![0, 1, 3, 6]);
        // ...and the cable they stand in for comes with them.
        let total: f32 = new_weights.unwrap().iter().sum();
        assert_eq!(total, weights.iter().sum::<f32>());
    }

    #[test]
    fn downsample_honours_preserve() {
        let (parents, _) = chain(11);
        let mut preserve = Array1::from_elem(11, false);
        preserve[7] = true;
        let (kept, _, _, _) =
            downsample_skeleton(&parents.view(), 5, &Some(preserve), &no_weights());
        assert_eq!(kept, vec![0, 5, 7, 10]);
    }

    #[test]
    #[should_panic(expected = "`factor` must be >= 1")]
    fn downsample_rejects_factor_zero() {
        let (parents, _) = chain(5);
        downsample_skeleton(&parents.view(), 0, &None, &no_weights());
    }

    /// `preserve` is a property of the shared "assemble a keep-mask" step, so it has to
    /// work the same way whichever rule decided the rest of the mask.
    #[test]
    fn preserve_works_on_every_dropper() {
        let (parents, coords) = chain(11);
        let mut preserve = Array1::from_elem(11, false);
        preserve[7] = true;
        let (p, c, keep) = (parents.view(), coords.view(), Some(preserve));

        for kept in [
            downsample_skeleton(&p, 100, &keep, &no_weights()).0,
            simplify_rdp(&p, &c, 1e9, &keep, &no_weights(), None).0,
            simplify_vw(&p, &c, 1e9, &keep, &no_weights(), None).0,
        ] {
            // Only the root, the leaf and the preserved node survive thresholds that
            // aggressive -- and node 7 is there only because it was named.
            assert_eq!(kept, vec![0, 7, 10]);
        }
    }

    /// `node_map` is likewise part of the shared step, so all three droppers agree on it.
    #[test]
    fn node_map_is_the_same_for_every_dropper() {
        let (parents, coords) = chain(11);
        let (p, c) = (parents.view(), coords.view());

        for map in [
            downsample_skeleton(&p, 100, &None, &no_weights()).3,
            simplify_rdp(&p, &c, 1e9, &None, &no_weights(), None).3,
            simplify_vw(&p, &c, 1e9, &None, &no_weights(), None).3,
        ] {
            // Root 0 and leaf 10 survive as slots 0 and 1. Nodes 1-5 are nearer the root
            // (node 5 by the proximal tie-break), 6-9 nearer the leaf.
            assert_eq!(map.to_vec(), vec![0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]);
        }
    }

    /// Every input node lands somewhere, and never on a node that was itself dropped.
    #[test]
    fn node_map_is_total_and_points_at_survivors() {
        let parents = tree();
        let coords = Array2::from_shape_fn((7, 3), |(i, k)| if k == 0 { i as f64 } else { 0.0 });
        let (kept, _, _, map) =
            simplify_rdp(&parents.view(), &coords.view(), 0.0, &None, &no_weights(), None);

        assert_eq!(map.len(), 7);
        for (node, &slot) in map.iter().enumerate() {
            assert!(slot >= 0, "node {node} mapped nowhere");
            assert!((slot as usize) < kept.len(), "node {node} mapped out of range");
        }
        // A survivor maps to its own slot, so `kept[map[node]] == node`.
        for (slot, &node) in kept.iter().enumerate() {
            assert_eq!(map[node as usize], slot as i32);
        }
    }

    /// NaN is the case worth a test of its own: every comparison against it is false, so
    /// an unchecked NaN tolerance would not error, it would silently drop every node.
    #[test]
    fn scale_parameters_reject_nan() {
        let (parents, coords) = chain(5);
        let (p, c) = (parents.view(), coords.view());

        // These are expected panics, so the default hook's backtrace spam is noise.
        let hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));

        let outcomes = [
            std::panic::catch_unwind(|| {
                simplify_rdp(&p, &c, f64::NAN, &None, &no_weights(), None);
            }),
            std::panic::catch_unwind(|| {
                simplify_vw(&p, &c, f64::NAN, &None, &no_weights(), None);
            }),
            std::panic::catch_unwind(|| {
                resample_skeleton(&p, &c, f64::NAN, None);
            }),
            std::panic::catch_unwind(|| {
                smooth_skeleton_gaussian(&p, &c, f64::NAN, 4.0, None);
            }),
        ];

        std::panic::set_hook(hook);
        assert!(outcomes.iter().all(|o| o.is_err()));
    }

    // --------------------------------------------------------------------------- RDP

    #[test]
    fn rdp_collapses_a_straight_line() {
        let (parents, coords) = chain(50);
        let (kept, new_parents, _, _) =
            simplify_rdp(&parents.view(), &coords.view(), 0.5, &None, &no_weights(), None);
        assert_eq!(kept, vec![0, 49]);
        assert_eq!(new_parents, arr1(&[-1, 0]));
    }

    #[test]
    fn rdp_keeps_the_corner() {
        // An L: 5 nodes out along x, then 4 up along y. Only the corner bends the path.
        let (parents, mut coords) = chain(9);
        for i in 5..9 {
            coords[[i, 0]] = 4.0;
            coords[[i, 1]] = (i - 4) as f64;
        }
        let (kept, _, _, _) =
            simplify_rdp(&parents.view(), &coords.view(), 0.5, &None, &no_weights(), None);
        assert_eq!(kept, vec![0, 4, 8]);
    }

    #[test]
    fn rdp_epsilon_zero_keeps_everything_off_the_line() {
        let (parents, mut coords) = chain(20);
        for i in 0..20 {
            coords[[i, 1]] = if i % 2 == 0 { 0.0 } else { 0.001 };
        }
        let (kept, _, _, _) =
            simplify_rdp(&parents.view(), &coords.view(), 0.0, &None, &no_weights(), None);
        assert_eq!(kept.len(), 20);
    }

    #[test]
    fn rdp_preserves_cable_length() {
        let (parents, coords) = chain(50);
        let weights: Option<Array1<f64>> = Some(Array1::from_elem(50, 1.0));
        let (_, _, new_weights, _) =
            simplify_rdp(&parents.view(), &coords.view(), 10.0, &None, &weights, None);
        // 49 edges of length 1, all folded into the single surviving edge.
        assert_eq!(new_weights.unwrap().iter().sum::<f64>(), 49.0);
    }

    /// The shape that makes RDP recurse once per node: on a zig-zag the farthest point
    /// from every chord is the one right next to its start, so each split peels off a
    /// single node and the next span is only one shorter. A recursive implementation
    /// would be `n` frames deep here. This is also RDP's quadratic worst case (see
    /// [`simplify_rdp`]), which is why the chain is thousands of nodes long rather than
    /// the hundreds of thousands the other stress fixtures use.
    #[test]
    fn rdp_survives_a_deeply_recursive_chain() {
        let n = 10_000;
        let parents: Array1<i32> = (0..n as i32).map(|i| i - 1).collect();
        let coords = Array2::from_shape_fn((n, 3), |(i, k)| match k {
            0 => i as f64,
            1 => ((i % 2) as f64) * 10.0,
            _ => 0.0,
        });
        let (kept, _, _, _) =
            simplify_rdp(&parents.view(), &coords.view(), 1.0, &None, &no_weights(), None);
        assert_eq!(kept.len(), n);
    }

    /// The other extreme, and the one real skeletons look like: a smooth curve, where
    /// each split lands mid-span and the work is `O(n log n)`. Large `n` is cheap here,
    /// so this is where the scale goes.
    #[test]
    fn rdp_handles_a_long_smooth_curve() {
        let n = 200_000;
        let parents: Array1<i32> = (0..n as i32).map(|i| i - 1).collect();
        let coords = Array2::from_shape_fn((n, 3), |(i, k)| {
            let t = i as f64 / 100.0;
            match k {
                0 => t,
                1 => t.sin() * 50.0,
                _ => 0.0,
            }
        });
        let (kept, _, _, _) =
            simplify_rdp(&parents.view(), &coords.view(), 1.0, &None, &no_weights(), None);
        assert!(kept.len() > 2 && kept.len() < n, "kept {}", kept.len());
    }

    // ---------------------------------------------------------------------------- VW

    #[test]
    fn vw_collapses_a_straight_line() {
        let (parents, coords) = chain(50);
        let (kept, _, _, _) =
            simplify_vw(&parents.view(), &coords.view(), 1e-9, &None, &no_weights(), None);
        assert_eq!(kept, vec![0, 49]);
    }

    #[test]
    fn vw_drops_the_smaller_spike_first() {
        // Two spikes off a straight line, one ten times taller than the other. Node 1
        // spans a triangle of area 0.1, node 3 one of area 1.0.
        let (parents, mut coords) = chain(5);
        coords[[1, 1]] = 0.1;
        coords[[3, 1]] = 1.0;
        let (kept, _, _, _) =
            simplify_vw(&parents.view(), &coords.view(), 0.5, &None, &no_weights(), None);
        // The small spike goes. Node 2 survives despite starting under the threshold:
        // losing node 1 widens its triangle to 1.0, which is the point of re-weighing a
        // node against its *surviving* neighbours rather than its original ones.
        assert_eq!(kept, vec![0, 2, 3, 4]);
    }

    #[test]
    fn vw_zero_threshold_is_a_no_op() {
        let (parents, coords) = chain(20);
        let (kept, _, _, _) =
            simplify_vw(&parents.view(), &coords.view(), 0.0, &None, &no_weights(), None);
        assert_eq!(kept.len(), 20);
    }

    #[test]
    fn vw_is_reproducible_on_ties() {
        // Every triangle has the same area, so the tie-break is the only thing deciding
        // which nodes go. Two runs must agree.
        let (parents, mut coords) = chain(30);
        for i in 0..30 {
            coords[[i, 1]] = ((i % 2) as f64) * 0.5;
        }
        let first = simplify_vw(&parents.view(), &coords.view(), 0.3, &None, &no_weights(), None).0;
        let second = simplify_vw(&parents.view(), &coords.view(), 0.3, &None, &no_weights(), None).0;
        assert_eq!(first, second);
    }

    // -------------------------------------------------------------------- resampling

    #[test]
    fn resample_hits_the_spacing() {
        let (parents, coords) = chain(11); // 10 units long
        let out = resample_skeleton(&parents.view(), &coords.view(), 2.0, None);
        // 5 edges of 2 units => 6 nodes.
        assert_eq!(out.parents.len(), 6);
        let mut xs: Vec<f64> = out.coords.column(0).to_vec();
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(xs, vec![0.0, 2.0, 4.0, 6.0, 8.0, 10.0]);
    }

    #[test]
    fn resample_keeps_the_endpoints_first_and_exact() {
        let parents = tree();
        let coords = Array2::from_shape_fn((7, 3), |(i, k)| if k == 0 { i as f64 } else { 0.0 });
        let out = resample_skeleton(&parents.view(), &coords.view(), 0.5, None);

        // Root 0, branch 1, leafs 3 and 6 come first, in input order, unmoved.
        for (slot, node) in [0usize, 1, 3, 6].iter().enumerate() {
            assert_eq!(out.coords[[slot, 0]], *node as f64);
            assert_eq!(out.source[[slot, 0]], *node as i32);
            assert_eq!(out.source[[slot, 1]], *node as i32);
            assert_eq!(out.alpha[slot], 0.0);
        }
    }

    #[test]
    fn resample_interpolates_along_the_right_edge() {
        let (parents, coords) = chain(3); // 0 -- 1 -- 2 along x
        let out = resample_skeleton(&parents.view(), &coords.view(), 0.5, None);
        // 2 units at 0.5 => 4 edges => 3 interior nodes at x = 0.5, 1.0, 1.5.
        assert_eq!(out.parents.len(), 5);
        for slot in 2..5 {
            let (c, p, a) = (
                out.source[[slot, 0]] as usize,
                out.source[[slot, 1]] as usize,
                out.alpha[slot],
            );
            // Interpolating the coordinates via source/alpha must reproduce the point.
            let want = coords[[c, 0]] * (1.0 - a) + coords[[p, 0]] * a;
            assert!((out.coords[[slot, 0]] - want).abs() < 1e-12);
        }
    }

    #[test]
    fn resample_collapses_a_short_segment() {
        let (parents, coords) = chain(5); // 4 units long
        let out = resample_skeleton(&parents.view(), &coords.view(), 100.0, None);
        assert_eq!(out.parents.len(), 2);
        assert_eq!(out.parents, arr1(&[-1, 0]));
    }

    #[test]
    fn resample_survives_coincident_nodes() {
        let parents: Array1<i32> = arr1(&[-1, 0, 1, 2]);
        let coords = Array2::zeros((4, 3));
        let out = resample_skeleton(&parents.view(), &coords.view(), 1.0, None);
        assert_eq!(out.parents.len(), 2);
        // A segment with no length has no nearer end, so its interior falls to the distal
        // endpoint -- the leaf, node 3, which is output slot 1.
        assert_eq!(out.node_map.to_vec(), vec![0, 1, 1, 1]);
    }

    /// Each input node hands its data to the output node nearest it along the neurite.
    #[test]
    fn resample_maps_input_nodes_to_the_nearest_output_node() {
        let (parents, coords) = chain(5); // 4 units long, x = 0..4
        let out = resample_skeleton(&parents.view(), &coords.view(), 2.0, None);

        // 4 units at a spacing of 2 => 2 edges. Root 0 and leaf 4 are carried over as
        // slots 0 and 1; the one new node, at x = 2, is slot 2.
        assert_eq!(out.parents.len(), 3);
        assert_eq!(out.coords[[2, 0]], 2.0);
        // Nodes 1 and 3 are both exactly halfway between two output nodes, and both go
        // the proximal way: node 1 to the root (slot 0), node 3 to the new node (slot 2).
        // Node 2 lands on the new node exactly.
        assert_eq!(out.node_map.to_vec(), vec![0, 0, 2, 2, 1]);
    }

    /// Nothing may map onto a node that is not in the output, and nothing may go unmapped.
    #[test]
    fn resample_node_map_is_total_and_in_range() {
        let parents = tree();
        let coords = Array2::from_shape_fn((7, 3), |(i, k)| if k == 0 { i as f64 } else { 0.0 });

        for spacing in [0.25, 1.0, 100.0] {
            let out = resample_skeleton(&parents.view(), &coords.view(), spacing, None);
            assert_eq!(out.node_map.len(), 7);
            for (node, &slot) in out.node_map.iter().enumerate() {
                assert!(
                    slot >= 0 && (slot as usize) < out.parents.len(),
                    "spacing {spacing}: node {node} mapped to {slot}"
                );
            }
            // Roots, branch points and leafs are carried over, so they map to themselves --
            // and those are the first rows of the output, in input order.
            for (slot, node) in [0usize, 1, 3, 6].iter().enumerate() {
                assert_eq!(out.node_map[*node], slot as i32);
            }
        }
    }

    #[test]
    #[should_panic(expected = "`spacing` must be positive")]
    fn resample_rejects_zero_spacing() {
        let (parents, coords) = chain(5);
        resample_skeleton(&parents.view(), &coords.view(), 0.0, None);
    }

    // --------------------------------------------------------------------- smoothing

    /// Both smoothers, on the shape they exist for. Endpoint pinning is not re-asserted
    /// here -- `smoothing_pins_branch_points` owns that claim for both of them.
    #[test]
    fn smoothing_flattens_a_zig_zag() {
        let n = 31;
        let (parents, mut coords) = chain(n);
        for i in 0..n {
            coords[[i, 1]] = if i % 2 == 0 { 1.0 } else { -1.0 };
        }

        for out in [
            smooth_skeleton(&parents.view(), &coords.view(), 5, None),
            smooth_skeleton_gaussian(&parents.view(), &coords.view(), 3.0, 4.0, None),
        ] {
            for i in 2..n - 2 {
                assert!(out[[i, 1]].abs() < 0.5, "node {i} at {}", out[[i, 1]]);
            }
        }
    }

    #[test]
    fn moving_average_window_one_is_a_no_op() {
        let (parents, mut coords) = chain(10);
        coords[[4, 1]] = 5.0;
        let out = smooth_skeleton(&parents.view(), &coords.view(), 1, None);
        assert_eq!(out, coords);
    }

    #[test]
    fn smoothing_pins_branch_points() {
        let parents = tree();
        let coords = Array2::from_shape_fn((7, 3), |(i, k)| ((i * 3 + k) % 5) as f64);
        for out in [
            smooth_skeleton(&parents.view(), &coords.view(), 5, None),
            smooth_skeleton_gaussian(&parents.view(), &coords.view(), 2.0, 4.0, None),
        ] {
            // Root 0, branch 1, leafs 3 and 6.
            for node in [0usize, 1, 3, 6] {
                for k in 0..3 {
                    assert_eq!(out[[node, k]], coords[[node, k]]);
                }
            }
        }
    }

    /// The reflection is there so the ends do not pull the neurite inwards. On a straight
    /// line nothing should move at all -- with a one-sided kernel, everything near the
    /// ends would.
    #[test]
    fn gaussian_leaves_a_straight_line_alone() {
        let (parents, coords) = chain(30);
        let out = smooth_skeleton_gaussian(&parents.view(), &coords.view(), 4.0, 4.0, None);
        for i in 0..30 {
            assert!(
                (out[[i, 0]] - coords[[i, 0]]).abs() < 1e-9,
                "node {i} moved to {}",
                out[[i, 0]]
            );
        }
    }

    #[test]
    #[should_panic(expected = "`sigma` must be positive")]
    fn gaussian_rejects_zero_sigma() {
        let (parents, coords) = chain(5);
        smooth_skeleton_gaussian(&parents.view(), &coords.view(), 0.0, 4.0, None);
    }

    // ------------------------------------------------------------------- degeneracies

    /// Every entry point, on the shapes that break tree code: one node, two nodes, a
    /// forest of isolated nodes. None of them may panic, and none may lose a node that
    /// carries topology.
    #[test]
    fn degenerate_shapes() {
        let cases: Vec<Array1<i32>> = vec![
            arr1(&[-1]),
            arr1(&[-1, -1, -1]),
            arr1(&[-1, 0]),
            arr1(&[-1, 0, 0]), // root is a branch point
        ];

        for parents in cases {
            let n = parents.len();
            let coords = Array2::from_shape_fn((n, 3), |(i, k)| (i + k) as f64);
            let p = parents.view();
            let c = coords.view();

            for kept in [
                downsample_skeleton(&p, 3, &None, &no_weights()).0,
                simplify_rdp(&p, &c, 1.0, &None, &no_weights(), None).0,
                simplify_vw(&p, &c, 1.0, &None, &no_weights(), None).0,
            ] {
                assert_eq!(kept.len(), n, "lost a node of {parents:?}");
            }

            let out = resample_skeleton(&p, &c, 1.0, None);
            assert!(out.parents.len() >= n);

            assert_eq!(smooth_skeleton(&p, &c, 5, None), coords);
            assert_eq!(
                smooth_skeleton_gaussian(&p, &c, 1.0, 4.0, None),
                coords
            );
        }
    }
}
