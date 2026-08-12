//! Closing the holes cut into a triangle mesh.
//!
//! Subsetting a mesh drops every face that loses a corner, which leaves the cut
//! cross-sections standing open. This module finds those openings and triangulates them
//! shut, in four steps that are separate because the two callers enter at different
//! points:
//!
//! 1. Find the boundary — [`boundary_halfedges`] over a whole mesh, or
//!    [`exposed_halfedges`] when you know which vertices are about to go.
//! 2. [`trace_loops`] walks those half-edges into closed rings, each of them *simple* — no
//!    vertex twice, and so a polygon.
//! 3. [`triangulate_rings`] ear-clips each ring into a cap, assuming that.
//!
//! No vertices are ever added, only faces. That keeps every vertex index — in the face
//! array, in whatever per-vertex data the caller carries alongside — pointing at what it
//! pointed at before, which is the property that lets a cap be applied after the subset
//! rather than during it.
//!
//! # Half-edges, and why the direction is kept
//!
//! A boundary edge has exactly one face left on it, and that face winds it one particular
//! way. Keeping that direction is what makes step 2 a walk rather than a search — the ring
//! is already oriented — and it is what tells step 3 which way round to wind the cap, since
//! a cap that agrees with its ring would have the two disagreeing about which side is out.
//!
//! # Where the time goes
//!
//! In [`boundary_halfedges`], entirely in grouping `3F` edges. The numpy formulation of
//! this is `np.unique(keys, return_inverse=True, return_counts=True)`, which is a *stable
//! argsort* — 75 ms of an 84 ms call on a 578k-face mesh, and not something numpy can be
//! talked out of: the bare `np.sort` of the same keys is already 51 ms.
//!
//! So the shape here is: sort the bare `u64` keys in parallel, scan for the ones that
//! occur once, and take a second pass over the faces to recover the directed half-edge
//! carrying each. That second pass is what avoids sorting `(key, position)` pairs — twice
//! the memory traffic — for a payload that is a rounding error of the input. 84 ms becomes
//! 8 ms.
//!
//! [`exposed_halfedges`] never looks at the mesh as a whole to begin with. A face can only
//! expose an edge if it loses *exactly one* corner, which on a real prune is a percent or
//! so of the faces that go; everything after that first pass is confined to the collar
//! around the cut.

use ndarray::{Array1, Array2, ArrayView2};
use rayon::prelude::*;

use crate::mesh::{edge_key, sorted_edge_keys};
use crate::points::eigh3;
use crate::simplify::{cross, dot, normalize, sub};
use crate::threads::with_pool;

// ---------------------------------------------------------------------------
// Finding the boundary
// ---------------------------------------------------------------------------

/// Run `f` over the faces in blocks of `block` faces, concatenating what each returns.
///
/// `f` is handed the index of the block's first face and the block's flat slice, and the
/// results come back in face order. The blocking is the point: both passes in
/// [`exposed_halfedges`] reject almost every face on a couple of array lookups, and at
/// per-face granularity rayon's own bookkeeping costs several times that. The `map` is
/// indexed, so the concatenation is a straight copy rather than the linked-list merge an
/// unindexed `collect` would do.
fn blocked<T: Clone + Send>(
    faces: &[u32],
    block: usize,
    f: impl Fn(usize, &[u32]) -> Vec<T> + Sync,
) -> Vec<T> {
    faces
        .par_chunks(3 * block)
        .enumerate()
        .map(|(bi, chunk)| f(bi * block, chunk))
        .collect::<Vec<_>>()
        .concat()
}

/// The edge keys exactly one face carries, sorted ascending.
///
/// Sorting bare keys rather than `(key, position)` pairs halves the bytes moved; the
/// positions are recovered by [`halfedges_with_keys`] afterwards, which is cheaper than
/// carrying them through the sort as long as the boundary is small relative to the mesh —
/// and a boundary that isn't is a mesh with more hole than surface.
///
/// A run of one is an edge with one face on it, which is what a boundary is.
fn solo_keys(faces: &[u32]) -> Vec<u64> {
    let keys = sorted_edge_keys(faces);
    keys.chunk_by(|a, b| a == b)
        .filter(|run| run.len() == 1)
        .map(|run| run[0])
        .collect()
}

/// The directed half-edges whose undirected key is in `keys`, in half-edge order.
///
/// `keys` must be sorted. Order matters downstream: [`trace_loops`] is greedy, so which
/// ring a half-edge lands in at a non-manifold vertex depends on the order it arrives in,
/// and half-edge order is the one order that does not depend on how the work was split.
fn halfedges_with_keys(faces: &[u32], keys: &[u64]) -> Array2<u32> {
    let out: Vec<u32> = faces
        .par_chunks_exact(3)
        .flat_map_iter(|f| {
            [[f[0], f[1]], [f[1], f[2]], [f[2], f[0]]]
                .into_iter()
                .filter(|e| keys.binary_search(&edge_key(e[0], e[1])).is_ok())
                .flatten()
        })
        .collect();
    Array2::from_shape_vec((out.len() / 2, 2), out).expect("collected pairs")
}

/// Find every edge of a mesh that has only one face on it.
///
/// This has to group the edges of the whole mesh, which is the expensive way round — use
/// [`exposed_halfedges`] where you already know which vertices are going away.
///
/// Arguments
/// ---------
/// - `faces`:   (F, 3) array of triangular faces given as vertex indices.
/// - `threads`: Size of the thread pool, or `None` for all cores.
///
/// Returns
/// -------
/// A `(K, 2)` array of directed half-edges, wound the way their one remaining face winds
/// them, in the order they appear in the conceptual `3F` edge list.
pub fn boundary_halfedges(faces: ArrayView2<u32>, threads: Option<usize>) -> Array2<u32> {
    // The Python wrapper always hands us C-order (borrowed as-is); a strided view from a
    // Rust caller gets copied into standard layout.
    let storage = faces.as_standard_layout();
    let s: &[u32] = storage.as_slice().expect("standard layout is contiguous");

    with_pool(threads, || {
        let solo = solo_keys(s);
        if solo.is_empty() {
            return Array2::zeros((0, 2));
        }
        halfedges_with_keys(s, &solo)
    })
}

/// Find the edges a subset is about to expose.
///
/// Call this with the *original* faces, before subsetting.
///
/// A face survives only if all three of its corners do, so an edge ends up on a new
/// boundary exactly when it loses a face to the cut but keeps one. Both halves of that test
/// are local to the cut, so this never has to group the edges of the whole mesh.
///
/// Edges that were already boundary are left out: they belong to openings the mesh came
/// with — a neurite truncated at the edge of the dataset, say — and sealing those is not
/// this function's business. [`boundary_halfedges`] is the one that finds those.
///
/// Arguments
/// ---------
/// - `faces`:   (F, 3) array of faces *before* subsetting.
/// - `dropped`: One flag per vertex: whether the subset drops it. Must be long enough to
///   index by every vertex `faces` names — the caller checks this, and a short mask panics.
/// - `threads`: Size of the thread pool, or `None` for all cores.
///
/// Returns
/// -------
/// A `(K, 2)` array of directed half-edges, wound the way the one face they have left winds
/// them, with indices into the *original* vertices.
pub fn exposed_halfedges(
    faces: ArrayView2<u32>,
    dropped: &[bool],
    threads: Option<usize>,
) -> Array2<u32> {
    let storage = faces.as_standard_layout();
    let s: &[u32] = storage.as_slice().expect("standard layout is contiguous");

    with_pool(threads, || {
        // Both passes below are blocked rather than per-face: the body is a handful of array
        // lookups for all but a fraction of a percent of the faces, so at face granularity
        // rayon's own bookkeeping — and the linked-list merge an unindexed `collect` does —
        // cost several times the work itself. On a 578k-face mesh that is 2.1 ms against 0.6.
        const BLOCK: usize = 4096;

        // Only a face that loses *exactly one* corner can leave an edge behind: lose two
        // and there is no edge left with both ends still standing. The edge it exposes is
        // the one opposite the corner it lost.
        let candidates: Vec<[u32; 2]> = blocked(s, BLOCK, |_, block| {
            let mut out = Vec::new();
            for f in block.chunks_exact(3) {
                let lost = [
                    dropped[f[0] as usize],
                    dropped[f[1] as usize],
                    dropped[f[2] as usize],
                ];
                if lost[0] as u8 + lost[1] as u8 + lost[2] as u8 != 1 {
                    continue;
                }
                let gone = lost.iter().position(|&x| x).expect("exactly one is set");
                out.push([f[(gone + 1) % 3], f[(gone + 2) % 3]]);
            }
            out
        });
        if candidates.is_empty() {
            return Array2::zeros((0, 2));
        }

        let mut cand: Vec<u64> = candidates.iter().map(|e| edge_key(e[0], e[1])).collect();
        cand.sort_unstable();
        cand.dedup();

        // Both ends of a candidate survive, so every surviving face carrying one has a
        // corner in this collar — which is what keeps the pass below off the bulk of a mesh
        // that a small prune barely touches.
        let mut collar = vec![false; dropped.len()];
        for e in &candidates {
            collar[e[0] as usize] = true;
            collar[e[1] as usize] = true;
        }

        // Every (candidate edge, half-edge on it) a surviving face contributes. There are
        // only ever a handful per candidate, so this stays proportional to the cut rather
        // than to the mesh, and counting it is a sort of something tiny.
        let mut hits: Vec<(u32, u32)> = blocked(s, BLOCK, |base, block| {
            let mut out = Vec::new();
            for (j, f) in block.chunks_exact(3).enumerate() {
                let (a, b, c) = (f[0] as usize, f[1] as usize, f[2] as usize);
                if dropped[a] || dropped[b] || dropped[c] {
                    continue;
                }
                if !(collar[a] || collar[b] || collar[c]) {
                    continue;
                }
                for (e, (u, v)) in [(f[0], f[1]), (f[1], f[2]), (f[2], f[0])]
                    .into_iter()
                    .enumerate()
                {
                    if let Ok(p) = cand.binary_search(&edge_key(u, v)) {
                        out.push((p as u32, ((base + j) * 3 + e) as u32));
                    }
                }
            }
            out
        });
        hits.sort_unstable();

        // Counting over the collar gives the same answer as counting over the whole mesh,
        // for a small fraction of the work: exactly one surviving face left on a candidate
        // edge means it is now a boundary.
        let mut solo: Vec<u32> = hits
            .chunk_by(|a, b| a.0 == b.0)
            .filter(|run| run.len() == 1)
            .map(|run| run[0].1)
            .collect();
        // Back into half-edge order — see [`halfedges_with_keys`] on why that order.
        solo.sort_unstable();

        let mut out = Vec::with_capacity(solo.len() * 2);
        for h in solo {
            let f = &s[(h as usize / 3) * 3..(h as usize / 3) * 3 + 3];
            let (a, b) = match h % 3 {
                0 => (f[0], f[1]),
                1 => (f[1], f[2]),
                _ => (f[2], f[0]),
            };
            out.push(a);
            out.push(b);
        }
        Array2::from_shape_vec((out.len() / 2, 2), out).expect("collected pairs")
    })
}

// ---------------------------------------------------------------------------
// Tracing rings
// ---------------------------------------------------------------------------

/// Walk directed half-edges into closed rings.
///
/// Greedy: at a non-manifold boundary vertex several half-edges leave at once, so we take
/// whichever is still free. Every half-edge lands in exactly one ring, which is what makes
/// this cover the whole boundary — a cycle basis (what `networkx` would give, and what
/// `trimesh.repair.fill_holes` uses) quietly drops the edges that are not part of a simple
/// cycle.
///
/// Every ring comes back *simple* — no vertex twice — which is what makes it a polygon and
/// so the thing [`triangulate_rings`] is defined on. A greedy walk does not give that on
/// its own: at a pinch, where several boundary edges meet at one point, it can leave and
/// re-enter the same vertex, and what it traced is then a figure of eight. So a walk is cut
/// where it crosses itself and the pieces handed on separately — the same half-edges,
/// grouped the way the caller can use.
///
/// A walk that runs into a dead end abandons what is still in hand, and so is a ring of
/// fewer than three vertices; in both cases the half-edges it consumed stay consumed, so
/// the traversal always terminates. Cycles already split off from an abandoned walk are
/// closed on their own account and are kept.
///
/// Single-threaded on purpose: the walk is inherently sequential — which target is still
/// free depends on what earlier walks took — but it is proportional to the *boundary*
/// rather than to the mesh, so it costs a fraction of finding that boundary in parallel:
/// 0.7 ms for the 80k half-edges of a mesh with 23k holes in it, against 9 ms to find them.
///
/// Returns
/// -------
/// `(rings, offsets)` in CSR form: ring `i` is `rings[offsets[i]..offsets[i + 1]]`, so
/// `offsets` has one more entry than there are rings. Flat rather than a `Vec<Vec<u32>>`
/// because that is the layout [`triangulate_rings`] wants and the one that crosses into
/// numpy or R without a per-ring object.
pub fn trace_loops(halfedges: ArrayView2<u32>) -> (Array1<u32>, Array1<i64>) {
    let storage = halfedges.as_standard_layout();
    let he: &[u32] = storage.as_slice().expect("standard layout is contiguous");
    if he.is_empty() {
        return (Array1::from_vec(Vec::new()), Array1::from_vec(vec![0]));
    }

    // Outgoing targets per tail vertex, as CSR keyed by vertex id directly — the same shape
    // [`crate::mesh::bridges`] builds its adjacency in, and for the same reason. A
    // `HashMap<u32, Vec<u32>>` would hash every half-edge twice (once building, once
    // walking) and put every vertex's targets in its own heap block, which is a pointer
    // chase per step of a walk that is already the one sequential stage here.
    let n = he.iter().copied().max().expect("non-empty") as usize + 1;

    // Counts first, then an exclusive prefix sum in place, so `start[v]..start[v + 1]` is
    // `v`'s run. A count still at zero when a half-edge arrives means this is the vertex's
    // first mention, which is how `tails` ends up in first-appearance order — the order the
    // greedy choice below depends on, and the one thing a hash map could not have given.
    let mut start = vec![0u32; n + 1];
    let mut tails: Vec<u32> = Vec::new();
    for e in he.chunks_exact(2) {
        if start[e[0] as usize] == 0 {
            tails.push(e[0]);
        }
        start[e[0] as usize] += 1;
    }
    let mut acc = 0;
    for slot in start.iter_mut() {
        let count = *slot;
        *slot = acc;
        acc += count;
    }

    // Fill each run in half-edge order. `cursor` finishes at the end of each vertex's run,
    // which is exactly where the walk wants to start popping from: taking the *last* target
    // appended is what a `Vec::pop` did.
    let mut cursor = start.clone();
    let mut heads = vec![0u32; he.len() / 2];
    for e in he.chunks_exact(2) {
        let v = e[0] as usize;
        heads[cursor[v] as usize] = e[1];
        cursor[v] += 1;
    }

    let mut rings: Vec<u32> = Vec::new();
    let mut offsets: Vec<i64> = vec![0];

    // The walk is built here rather than straight into `rings`, because a walk that
    // touches a vertex twice has to be cut in two before it is handed on, and a ring
    // already written into the CSR cannot be cut. `where_on_path[v]` is `v`'s index in
    // `path`, or `NOT_ON_PATH`; the array spans the vertex space like `start` does, so
    // the test is one lookup rather than a scan back along the walk.
    const NOT_ON_PATH: u32 = u32::MAX;
    let mut path: Vec<u32> = Vec::new();
    let mut where_on_path = vec![NOT_ON_PATH; n];

    for &root in &tails {
        let r = root as usize;
        while cursor[r] > start[r] {
            // `path` is left empty by the drain that ends each walk.
            where_on_path[r] = 0;
            path.push(root);
            cursor[r] -= 1;
            let mut v = heads[cursor[r] as usize];
            while v != root {
                // Take whichever half-edge out of `v` is still free. An exhausted run covers
                // both "never a tail" (its run is empty to begin with) and "already all
                // spent" — either way a dead end.
                let s = v as usize;
                if cursor[s] <= start[s] {
                    break;
                }
                // Arriving where the walk has already been closes a loop of its own:
                // everything since that visit is a cycle touching the rest of the walk
                // only at this vertex. Emit it and rewind to where it started, which
                // leaves `path` simple and lets the walk carry on through the vertex a
                // second time. This is what a pinch — several boundary edges meeting at
                // one point — looks like from inside the walk, and splitting here is the
                // difference between handing on a polygon and handing on a figure of
                // eight, which no triangulator downstream is defined on.
                if where_on_path[s] != NOT_ON_PATH {
                    let from = where_on_path[s] as usize;
                    emit(&mut rings, &mut offsets, &path[from..]);
                    for u in path.drain(from..) {
                        where_on_path[u as usize] = NOT_ON_PATH;
                    }
                }
                cursor[s] -= 1;
                where_on_path[s] = path.len() as u32;
                path.push(v);
                v = heads[cursor[s] as usize];
            }
            // `v == root` means the walk closed. A dead end abandons only what is left in
            // `path`: cycles split off above closed on themselves and stand on their own.
            if v == root {
                emit(&mut rings, &mut offsets, &path);
            }
            for u in path.drain(..) {
                where_on_path[u as usize] = NOT_ON_PATH;
            }
        }
    }

    (Array1::from_vec(rings), Array1::from_vec(offsets))
}

/// Append a ring to the CSR, dropping anything too short to bound a face.
fn emit(rings: &mut Vec<u32>, offsets: &mut Vec<i64>, ring: &[u32]) {
    if ring.len() >= 3 {
        rings.extend_from_slice(ring);
        offsets.push(rings.len() as i64);
    }
}

// ---------------------------------------------------------------------------
// Triangulating rings
// ---------------------------------------------------------------------------

/// Twice the signed area of a 2-D ring; positive means counter-clockwise.
fn signed_area(flat: &[[f64; 2]]) -> f64 {
    let n = flat.len();
    (0..n)
        .map(|i| {
            let (a, b) = (flat[i], flat[(i + 1) % n]);
            a[0] * b[1] - a[1] * b[0]
        })
        .sum()
}

/// An orthonormal pair spanning the plane perpendicular to `normal`, or `None` if the
/// normal is degenerate and names no plane at all.
///
/// [`normalize`]'s `None` covers zero, NaN and infinite lengths alike, which here all mean
/// the same thing — there is no plane to be had — and all send the ring to the best-fit
/// fallback. Only the zero case can arise from a ring whose signed areas cancel; the other
/// two need coordinates that already overflowed, and normalising by either would put NaN
/// into the projection and from there into the ear-clipper.
///
/// `pub(crate)` for [`crate::internals`], which needs the same frame to spray rays over a
/// face's hemisphere — and for the same reason wants the degenerate case handed back rather
/// than divided through.
pub(crate) fn basis(normal: [f64; 3]) -> Option<([f64; 3], [f64; 3])> {
    let n = normalize(normal)?;
    // Any vector not parallel to the normal will do to get started.
    let other = if n[0].abs() < 0.9 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let u = normalize(cross(n, other))?;
    Some((u, cross(n, u)))
}

/// The best-fit plane of a centred ring: the two directions it varies most in, largest
/// first.
///
/// An eigendecomposition of the 3x3 scatter matrix beats an SVD of the `(n, 3)` points, and
/// it is the fallback for a ring the area-weighted normal could not flatten. [`eigh3`]
/// already returns its eigenvectors as columns in descending eigenvalue order, which is
/// exactly the order wanted here.
fn scatter_basis(centred: &[[f64; 3]]) -> ([f64; 3], [f64; 3]) {
    let mut m = [[0.0f64; 3]; 3];
    for p in centred {
        for i in 0..3 {
            for j in 0..3 {
                m[i][j] += p[i] * p[j];
            }
        }
    }
    let (_, vecs) = eigh3(m);
    ([vecs[0][0], vecs[1][0], vecs[2][0]], [vecs[0][1], vecs[1][1], vecs[2][1]])
}

/// Project a centred ring onto an orthonormal pair.
fn project(centred: &[[f64; 3]], u: [f64; 3], w: [f64; 3]) -> Vec<[f64; 2]> {
    centred.iter().map(|&p| [dot(p, u), dot(p, w)]).collect()
}

/// Ear-clip a flattened ring, appending `[a, b, c]` triples to `out`.
///
/// `false` — and nothing appended — if the flattening self-intersects: earcut running out
/// of ears part way through is exactly what that looks like, and it shows up as fewer than
/// the `n - 2` triangles a simple polygon always yields.
fn earcut_ring(
    ring: &[u32],
    flat: &[[f64; 2]],
    ec: &mut earcut::Earcut<f64>,
    scratch: &mut Vec<u32>,
    out: &mut Vec<u32>,
) -> bool {
    let n = ring.len();
    ec.earcut(flat.iter().copied(), &[] as &[u32], scratch);
    if scratch.len() != 3 * (n - 2) {
        return false;
    }
    // The ring runs the way its remaining faces wind it, so the cap has to run the other
    // way or the two will disagree about which side is out. earcut emits counter-clockwise
    // triangles whichever way the ring itself goes, so it is only a counter-clockwise ring
    // that needs flipping.
    let flip = signed_area(flat) > 0.0;
    for t in scratch.chunks_exact(3) {
        let (a, b, c) = (ring[t[0] as usize], ring[t[1] as usize], ring[t[2] as usize]);
        if flip {
            out.extend_from_slice(&[c, b, a]);
        } else {
            out.extend_from_slice(&[a, b, c]);
        }
    }
    true
}

/// Ear-clip a ring in three dimensions, wound against the ring.
///
/// The last resort, for a ring no flattening can handle. Both attempts above are
/// projections, so they fail on the same rings — not the ones that are least planar, but
/// the ones whose *shadow* crosses itself, which a gently curved ring can manage as easily
/// as a folded one. This never projects, so that failure cannot arise: it clips whichever
/// ear is cheapest by `area + 0.05 * perimeter^2`, taken over the ring as it stands.
///
/// Both terms are areas, so the choice does not move if the mesh is measured in nm rather
/// than um; the perimeter term is what stops it from paying for a small area with a long
/// thin sliver, which minimising area alone does. This is a heuristic about the *shape* of
/// the cap and, unlike the planar clip, cannot promise it does not fold over itself
/// somewhere — which is why it sits after the two attempts that can. It is worth having
/// because the alternative is a fan out of one vertex: on a real invaginated soma, a
/// 1223-vertex ring through the pinches, it takes the median cap edge from 2.7 um to 65 nm.
///
/// The cost is `n` clips over a shrinking ring. Only two ears change weight per clip — the
/// ones that gain a new neighbour — so the weights are carried rather than rebuilt, which
/// on that same ring is ~2.4k evaluations rather than ~750k. What is left is the scan for
/// the cheapest, over a flat `f64` array. Liepa's minimum-weight triangulation would give
/// better-shaped caps, but it is `O(n^3)` and so wants a size cap, and the rings small
/// enough to afford it are the ones a fan already handled acceptably.
fn earclip3d(ring: &[u32], vertices: &[f64], out: &mut Vec<u32>) {
    let at = |v: u32| {
        let i = 3 * v as usize;
        [vertices[i], vertices[i + 1], vertices[i + 2]]
    };
    let len = |a: [f64; 3]| dot(a, a).sqrt();
    // The cost of clipping the ear at `k`, i.e. of the triangle its two neighbours make
    // with it. Reads `idx` at the moment it is called, so it has to be recomputed for the
    // two ears either side of a clip and for nothing else.
    let weight = |idx: &[u32], k: usize| {
        let n = idx.len();
        let (a, b, c) = (at(idx[(k + n - 1) % n]), at(idx[k]), at(idx[(k + 1) % n]));
        let (ab, ac) = (sub(b, a), sub(c, a));
        len(cross(ab, ac)) * 0.5 + 0.05 * (len(ab) + len(sub(c, b)) + len(ac)).powi(2)
    };

    let mut idx: Vec<u32> = ring.to_vec();
    let mut weights: Vec<f64> = (0..idx.len()).map(|k| weight(&idx, k)).collect();
    while idx.len() > 3 {
        // `>` keeps the earliest of equal candidates, so a ring with several identical ears
        // clips the same one whichever order the scan happens to reach them in.
        let mut best = 0usize;
        for k in 1..weights.len() {
            if weights[best] > weights[k] {
                best = k;
            }
        }
        let n = idx.len();
        out.extend_from_slice(&[idx[(best + 1) % n], idx[best], idx[(best + n - 1) % n]]);
        idx.remove(best);
        weights.remove(best);
        // `best` now holds what followed the clipped ear and `best - 1` what preceded it;
        // both have gained a neighbour, and no other ear has changed.
        let n = idx.len();
        weights[best % n] = weight(&idx, best % n);
        weights[(best + n - 1) % n] = weight(&idx, (best + n - 1) % n);
    }
    out.extend_from_slice(&[idx[2], idx[1], idx[0]]);
}

/// Triangulate one ring, appending its cap to `out`.
fn cap_ring(
    ring: &[u32],
    vertices: &[f64],
    ec: &mut earcut::Earcut<f64>,
    scratch: &mut Vec<u32>,
    out: &mut Vec<u32>,
) {
    let n = ring.len();
    if n < 3 {
        return;
    }
    if n == 3 {
        // Already a triangle; reversed, so that it opposes the ring the way a cap must.
        out.extend_from_slice(&[ring[2], ring[1], ring[0]]);
        return;
    }

    let mut centroid = [0.0f64; 3];
    for &v in ring {
        for k in 0..3 {
            centroid[k] += vertices[3 * v as usize + k];
        }
    }
    for c in &mut centroid {
        *c /= n as f64;
    }
    let centred: Vec<[f64; 3]> = ring
        .iter()
        .map(|&v| {
            let p = &vertices[3 * v as usize..3 * v as usize + 3];
            [p[0] - centroid[0], p[1] - centroid[1], p[2] - centroid[2]]
        })
        .collect();

    // Newell's area-weighted normal. Cheaper than a best-fit plane and, on the rings a cut
    // actually produces, it fails slightly less often too.
    let mut normal = [0.0f64; 3];
    for i in 0..n {
        let c = cross(centred[i], centred[(i + 1) % n]);
        for k in 0..3 {
            normal[k] += c[k];
        }
    }
    if let Some((u, w)) = basis(normal) {
        if earcut_ring(ring, &project(&centred, u, w), ec, scratch, out) {
            return;
        }
    }

    // Second attempt through the best-fit plane, then out of the plane altogether.
    let (u, w) = scatter_basis(&centred);
    if !earcut_ring(ring, &project(&centred, u, w), ec, scratch, out) {
        earclip3d(ring, vertices, out);
    }
}

/// Check that a `(rings, offsets)` pair is the CSR [`trace_loops`] produces, describing what
/// is wrong if it isn't.
///
/// Here rather than in each binding for the same reason [`crate::smoothing::Filter::check`]
/// is: the bindings differ in how they *report* a bad argument — a `ValueError` in Python, a
/// panic in R — but not in what counts as one, and three copies of these bounds is three
/// chances for the wording and the predicates to drift apart.
///
/// Only the pair's internal consistency; that `rings` names real vertices is the caller's,
/// since only it knows how many there are.
pub fn check_rings(rings: &[u32], offsets: &[i64]) -> Result<(), String> {
    let (Some(&first), Some(&last)) = (offsets.first(), offsets.last()) else {
        return Err("`offsets` must have at least one entry (a leading 0)".into());
    };
    if first != 0 || last != rings.len() as i64 {
        return Err(format!(
            "`offsets` must run from 0 to len(rings) = {}, got {first}..{last}",
            rings.len()
        ));
    }
    if offsets.windows(2).any(|w| w[1] < w[0]) {
        return Err("`offsets` must be non-decreasing".into());
    }
    Ok(())
}

/// Triangulate boundary rings, wound against the direction they run in.
///
/// `rings` and `offsets` must satisfy [`check_rings`]; a pair that does not will panic on an
/// out-of-range slice rather than return something wrong.
///
/// Rings are independent, so this runs one per worker; each keeps its own reusable earcut
/// arena, which is what makes a mesh with tens of thousands of small holes cost about what
/// its total boundary length says it should.
///
/// Every ring is closed one way or another: ear-clipping through the area-weighted normal,
/// failing that through the best-fit plane, and failing both of those — which happens when
/// the ring's shadow crosses itself, whatever plane it is cast on — ear-clipping in three
/// dimensions instead. All three are closed and correctly wound, which is what callers
/// depend on; only the last is a heuristic about the *shape* of the cap.
///
/// Over the 933 openings of an invaginated neuron mesh, the normal took 94.0%, the best-fit
/// plane a further 1.1%, and the three-dimensional clip the remaining 4.9%. So the middle
/// rung does earn its place, though narrowly — it rescues about one in five of the rings the
/// normal cannot flatten. Its other entry, a ring whose signed areas cancel so exactly that
/// there is no area-weighted normal to project through at all, did not come up once.
///
/// Arguments
/// ---------
/// - `rings`, `offsets`: boundary rings in the CSR form [`trace_loops`] returns. A ring is
///   assumed *simple*, which is what that function guarantees. One built by hand that names
///   a vertex twice is not a polygon, and while its cap still comes back closed and
///   correctly wound, its shape is not specified. [`check_rings`] deliberately does not test
///   for this: it is `O(n)` per ring, and the caller usually knows already.
/// - `vertices`:         (V, 3) vertex positions.
/// - `threads`:          Size of the thread pool, or `None` for all cores.
///
/// Returns
/// -------
/// An `(M, 3)` array of new faces, indices into `vertices`, ring by ring.
pub fn triangulate_rings(
    rings: &[u32],
    offsets: &[i64],
    vertices: ArrayView2<f64>,
    threads: Option<usize>,
) -> Array2<u32> {
    let storage = vertices.as_standard_layout();
    let v: &[f64] = storage.as_slice().expect("standard layout is contiguous");

    with_pool(threads, || {
        let caps: Vec<Vec<u32>> = offsets
            .par_windows(2)
            .map_init(
                || (earcut::Earcut::new(), Vec::new()),
                |(ec, scratch), w| {
                    let mut out = Vec::new();
                    cap_ring(
                        &rings[w[0] as usize..w[1] as usize],
                        v,
                        ec,
                        scratch,
                        &mut out,
                    );
                    out
                },
            )
            .collect();

        let flat = caps.concat();
        Array2::from_shape_vec((flat.len() / 3, 3), flat).expect("collected triples")
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::tests_support::grid;
    use ndarray::{array, Array2};

    /// Fan from the ring's first vertex, wound against the ring.
    ///
    /// What [`cap_ring`] used to close a ring with when both flattenings failed. Kept as
    /// the baseline the three-dimensional clip is measured against — a fan reaches from
    /// one vertex to every other, so its longest edge is about the width of the opening.
    fn fan(ring: &[u32], out: &mut Vec<u32>) {
        for i in 2..ring.len() {
            out.extend_from_slice(&[ring[0], ring[i], ring[i - 1]]);
        }
    }

    /// A closed tetrahedron: no boundary at all.
    fn tetrahedron() -> (Array2<u32>, Array2<f64>) {
        (
            array![[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]],
            array![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0]
            ],
        )
    }

    #[test]
    fn closed_mesh_has_no_boundary() {
        let (faces, _) = tetrahedron();
        assert_eq!(boundary_halfedges(faces.view(), None).nrows(), 0);
    }

    #[test]
    fn grid_boundary_is_its_outer_edge() {
        let n = 6;
        let (faces, _) = grid(n, 1.0);
        let b = boundary_halfedges(faces.view(), None);
        assert_eq!(b.nrows(), 4 * (n - 1));
        // Every boundary vertex is on the outer edge of the grid.
        for e in b.rows() {
            for &v in e {
                let (i, j) = (v as usize / n, v as usize % n);
                assert!(i == 0 || j == 0 || i == n - 1 || j == n - 1, "interior {v}");
            }
        }
    }

    #[test]
    fn boundary_traces_into_one_ring() {
        let n = 5;
        let (faces, _) = grid(n, 1.0);
        let (rings, offsets) = trace_loops(boundary_halfedges(faces.view(), None).view());
        assert_eq!(offsets.len(), 2, "one ring");
        assert_eq!(rings.len(), 4 * (n - 1));
    }

    #[test]
    fn dropping_one_corner_exposes_the_opposite_edge() {
        // Two triangles sharing edge (1, 2). Dropping vertex 0 removes the first face and
        // leaves (1, 2) with only the second on it.
        let faces = array![[0u32, 1, 2], [1, 3, 2]];
        let dropped = [true, false, false, false];
        let e = exposed_halfedges(faces.view(), &dropped, None);
        assert_eq!(e.nrows(), 1);
        // Wound the way the surviving face winds it: face (1, 3, 2) carries (2, 1).
        assert_eq!(e.row(0).to_vec(), vec![2, 1]);
    }

    #[test]
    fn pre_existing_boundary_is_not_exposed() {
        // A lone triangle: all three edges are already boundary. Dropping a corner leaves
        // no face at all, so nothing is newly exposed.
        let faces = array![[0u32, 1, 2]];
        let dropped = [true, false, false];
        assert_eq!(exposed_halfedges(faces.view(), &dropped, None).nrows(), 0);
    }

    #[test]
    fn exposed_agrees_with_boundary_of_the_subset() {
        // Punch a hole in a grid by dropping one interior vertex. What `exposed_halfedges`
        // reports on the original faces must be what is left standing after the cut, minus
        // the grid's own outer edge.
        let n = 7;
        let (faces, _) = grid(n, 1.0);
        let mut dropped = vec![false; n * n];
        dropped[3 * n + 3] = true;

        let exposed = exposed_halfedges(faces.view(), &dropped, None);
        assert!(exposed.nrows() > 0);

        let kept: Vec<u32> = faces
            .rows()
            .into_iter()
            .filter(|f| !f.iter().any(|&v| dropped[v as usize]))
            .flatten()
            .copied()
            .collect();
        let kept = Array2::from_shape_vec((kept.len() / 3, 3), kept).unwrap();

        let all: std::collections::HashSet<(u32, u32)> = boundary_halfedges(kept.view(), None)
            .rows()
            .into_iter()
            .map(|e| (e[0], e[1]))
            .collect();
        let outer = 4 * (n - 1);
        assert_eq!(exposed.nrows(), all.len() - outer);
        for e in exposed.rows() {
            assert!(all.contains(&(e[0], e[1])), "{:?} not on the new boundary", e);
        }
    }

    #[test]
    fn a_capped_grid_is_closed() {
        // Cap the grid's outer ring and the mesh has no boundary left anywhere.
        let n = 6;
        let (faces, vertices) = grid(n, 1.0);
        let b = boundary_halfedges(faces.view(), None);
        let (rings, offsets) = trace_loops(b.view());
        let caps = triangulate_rings(
            rings.as_slice().unwrap(),
            offsets.as_slice().unwrap(),
            vertices.view(),
            None,
        );
        assert_eq!(caps.nrows(), 4 * (n - 1) - 2, "a ring of k caps to k - 2");

        let mut all: Vec<u32> = faces.iter().copied().collect();
        all.extend(caps.iter().copied());
        let all = Array2::from_shape_vec((all.len() / 3, 3), all).unwrap();
        assert_eq!(boundary_halfedges(all.view(), None).nrows(), 0);
    }

    #[test]
    fn caps_wind_against_their_ring() {
        // A square hole in the z = 0 plane, wound counter-clockwise seen from +z. The cap
        // must therefore wind clockwise, i.e. have a normal pointing at -z.
        let vertices = array![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0]
        ];
        let rings = [0u32, 1, 2, 3];
        let caps = triangulate_rings(&rings, &[0, 4], vertices.view(), None);
        assert_eq!(caps.nrows(), 2);
        for t in caps.rows() {
            let p: Vec<[f64; 3]> = t
                .iter()
                .map(|&v| {
                    let r = vertices.row(v as usize);
                    [r[0], r[1], r[2]]
                })
                .collect();
            let e1 = [p[1][0] - p[0][0], p[1][1] - p[0][1], p[1][2] - p[0][2]];
            let e2 = [p[2][0] - p[0][0], p[2][1] - p[0][1], p[2][2] - p[0][2]];
            assert!(cross(e1, e2)[2] < 0.0, "cap agrees with its ring: {:?}", t);
        }
    }

    #[test]
    fn non_planar_ring_still_closes() {
        // A ring the area-weighted normal cannot flatten without self-intersection — this
        // is the path that falls through to the best-fit plane and, failing that, out of
        // the plane. Whichever rung takes it, the cap has to have n - 2 triangles and use
        // every ring vertex.
        let vertices = array![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 3.0],
            [2.0, 0.0, 0.0],
            [2.0, 2.0, 3.0],
            [1.0, 1.0, 0.0],
            [0.0, 2.0, 3.0]
        ];
        let rings = [0u32, 1, 2, 3, 4, 5];
        let caps = triangulate_rings(&rings, &[0, 6], vertices.view(), None);
        assert_eq!(caps.nrows(), 4);
        let used: std::collections::HashSet<u32> = caps.iter().copied().collect();
        assert_eq!(used.len(), 6);
    }

    #[test]
    fn ring_through_the_same_vertex_twice_still_closes() {
        // A ring naming the same vertex twice: it touches itself, so it is not a polygon
        // and neither flattening can find `n - 2` ears. `trace_loops` no longer hands one
        // of these on — it cuts the walk where it crosses itself — but a caller can still
        // build one, and when it does the cap has to come back regardless.
        //
        // These are the real coordinates of one such ring off a punched neuron mesh,
        // kept because they are also the input that sends `mapbox_earcut` — what navis
        // reached for before this module existed — into an infinite loop on its
        // best-fit-plane retry.
        let vertices = array![
            [5571.95996094, 22467.96875, 16704.00390625],
            [5618.43554688, 22463.57617188, 16698.08398438],
            [5519.92675781, 22390.08789062, 16695.66992188],
            [5576.05859375, 22375.62695312, 16725.99609375],
            [5618.43554688, 22463.57617188, 16698.08398438], // == row 1
            [5611.96044922, 22447.96679688, 16756.00585938]
        ];
        let caps = triangulate_rings(&[0, 1, 2, 3, 4, 5], &[0, 6], vertices.view(), None);
        assert_eq!(caps.nrows(), 4, "n - 2 triangles, whichever route got there");
        let used: std::collections::HashSet<u32> = caps.iter().copied().collect();
        assert_eq!(used.len(), 6, "every ring vertex used");
    }

    #[test]
    fn degenerate_ring_still_closes() {
        // Every vertex collinear: no plane at all, so both projections are hopeless and
        // this lands in the three-dimensional clip. The triangles are degenerate whatever
        // closes it — the point is that something does, with the right count.
        let vertices = array![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0]
        ];
        let caps = triangulate_rings(&[0, 1, 2, 3], &[0, 4], vertices.view(), None);
        assert_eq!(caps.nrows(), 2);
    }

    #[test]
    fn a_walk_that_crosses_itself_is_cut_where_it_crosses() {
        // Two triangles sharing vertex 0, but entered from vertex 1 rather than from the
        // shared vertex — so the greedy walk leaves 0, goes right round the second
        // triangle and arrives back at 0 with edges still to spend. Traced in one piece
        // that is `1 0 3 4 0 2`, which names 0 twice and is no polygon.
        let he = array![[1u32, 0], [0, 2], [2, 1], [0, 3], [3, 4], [4, 0]];
        let (rings, offsets) = trace_loops(he.view());

        assert_eq!(offsets.len(), 3, "cut into two rings");
        assert_eq!(rings.len(), 6, "every half-edge still used exactly once");
        for w in offsets.windows(2) {
            let ring = &rings.as_slice().unwrap()[w[0] as usize..w[1] as usize];
            let unique: std::collections::HashSet<u32> = ring.iter().copied().collect();
            assert_eq!(
                unique.len(),
                ring.len(),
                "ring {ring:?} names a vertex twice"
            );
        }
    }

    #[test]
    fn a_ring_no_plane_can_hold_is_clipped_locally() {
        // Real coordinates, centred, of a 21-vertex opening off an invaginated soma —
        // one of the rings this module used to fan. Both flattenings self-intersect, and
        // it is worth being clear that this is not the same thing as being un-planar: a
        // gently curved ring can cast a crossed shadow, and a folded one need not.
        //
        // The property under test is that the cap stays *local*. A fan reaches from one
        // vertex to every other, so its longest edge is about the width of the opening;
        // on this ring that is 115 nm against the clip's 63 nm, and on the 1223-vertex
        // soma ring these come from, 4.2 um against 1.4 um.
        let v = array![
            [-26.794643, 22.679315, -40.629464],
            [-16.107143, 44.007440, -25.691964],
            [-26.794643, 29.085565, -15.035714],
            [5.205357, 22.679315, 16.964286],
            [-26.794643, 16.273065, 16.964286],
            [-18.794643, 14.679315, 48.964286],
            [-34.794643, -9.320685, 24.964286],
            [-26.794643, -34.914435, 16.964286],
            [5.205357, -41.320685, 16.964286],
            [-26.794643, -33.320685, -7.035714],
            [5.205357, -33.320685, -7.035714],
            [29.205357, -33.320685, 16.964286],
            [37.205357, -9.320685, 48.964286],
            [29.205357, 14.679315, 48.964286],
            [29.205357, 14.679315, 16.964286],
            [37.205357, 22.679315, -8.629464],
            [37.205357, -9.320685, -15.035714],
            [5.205357, -17.320685, -23.035714],
            [5.205357, -1.320685, -39.035714],
            [-26.794643, -1.320685, -39.035714],
            [5.205357, 22.679315, -53.441964],
        ];
        let n = v.nrows();
        let ring: Vec<u32> = (0..n as u32).collect();
        let caps = triangulate_rings(&ring, &[0, n as i64], v.view(), None);

        assert_eq!(caps.nrows(), n - 2, "n - 2 triangles");
        let used: std::collections::HashSet<u32> = caps.iter().copied().collect();
        assert_eq!(used.len(), n, "every ring vertex used");

        // Winding: the ring runs the way its remaining faces do, so for every ring edge
        // the cap has to carry the same edge the other way round.
        let directed: std::collections::HashSet<(u32, u32)> = caps
            .rows()
            .into_iter()
            .flat_map(|t| [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])])
            .collect();
        for i in 0..n {
            let (a, b) = (ring[i], ring[(i + 1) % n]);
            assert!(
                directed.contains(&(b, a)),
                "cap does not oppose ring edge {a}->{b}"
            );
        }

        let longest = |tris: &[u32]| {
            tris.chunks_exact(3)
                .map(|t| {
                    (0..3)
                        .map(|k| {
                            let p = v.row(t[k] as usize);
                            let q = v.row(t[(k + 1) % 3] as usize);
                            let d = [q[0] - p[0], q[1] - p[1], q[2] - p[2]];
                            dot(d, d).sqrt()
                        })
                        .fold(0.0f64, f64::max)
                })
                .fold(0.0f64, f64::max)
        };
        let mut fanned = Vec::new();
        fan(&ring, &mut fanned);
        assert!(
            longest(caps.as_slice().unwrap()) < longest(&fanned),
            "the clip should not span the opening the way a fan does"
        );
    }

    #[test]
    fn empty_inputs_come_back_empty() {
        let faces = Array2::<u32>::from_shape_vec((0, 3), vec![]).unwrap();
        assert_eq!(boundary_halfedges(faces.view(), None).nrows(), 0);
        assert_eq!(exposed_halfedges(faces.view(), &[], None).nrows(), 0);

        let (rings, offsets) = trace_loops(Array2::zeros((0, 2)).view());
        assert_eq!(rings.len(), 0);
        assert_eq!(offsets.to_vec(), vec![0]);

        let v = Array2::<f64>::from_shape_vec((0, 3), vec![]).unwrap();
        assert_eq!(triangulate_rings(&[], &[0], v.view(), None).nrows(), 0);
    }

    #[test]
    fn dead_ends_do_not_hang() {
        // A path that never closes: greedy tracing has to abandon it and terminate.
        let he = array![[0u32, 1], [1, 2], [2, 3]];
        let (rings, offsets) = trace_loops(he.view());
        assert_eq!(rings.len(), 0);
        assert_eq!(offsets.to_vec(), vec![0]);
    }

    #[test]
    fn figure_of_eight_splits_into_two_rings() {
        // Two triangles meeting at vertex 0 — a non-manifold boundary vertex. Greedy
        // tracing has to cover every half-edge, which a cycle basis would not.
        let he = array![[0u32, 1], [1, 2], [2, 0], [0, 3], [3, 4], [4, 0]];
        let (rings, offsets) = trace_loops(he.view());
        assert_eq!(offsets.len(), 3, "two rings");
        assert_eq!(rings.len(), 6, "every half-edge used exactly once");
    }

    #[test]
    fn scatter_basis_spans_the_plane_a_ring_varies_in() {
        // A ring spread most along x, then y, and barely at all along z. The retry plane
        // has to be the xy one, whichever way round the eigenvectors' signs came out.
        // (`eigh3` itself is tested in `points`; what is checked here is that this picks
        // its two *largest* columns, which is what "best-fit plane" means.)
        let centred = [
            [3.0, 0.0, 0.0],
            [-3.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, -2.0, 0.0],
        ];
        let (u, w) = scatter_basis(&centred);
        assert!((u[0].abs() - 1.0).abs() < 1e-12, "widest spread is x: {u:?}");
        assert!((w[1].abs() - 1.0).abs() < 1e-12, "then y: {w:?}");
        assert!(dot(u, w).abs() < 1e-12, "orthonormal");
    }
}
