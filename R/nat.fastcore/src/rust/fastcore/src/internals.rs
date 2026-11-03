//! Stripping the surface a neuron mesh keeps on its inside.
//!
//! An **invagination** is a piece of membrane that bulges *into* the cell — typically the
//! boundary of a mitochondrion or a vesicle that touches the membrane from within. Segmented
//! meshes are full of them, and they are ruinous for anything that walks the surface: each one
//! is a tunnel or a pocket the wave front of a skeletonisation can take a shortcut through, or
//! split on. On the mesh this was developed against they account for 3,076 of its handles and
//! most of the 3,227 leafs its skeleton comes out with, against ~500 once they are gone.
//!
//! The property that defines an invagination — the cell encloses it — is also the property
//! that makes it invisible from outside, which is what [`drop_internals`] exploits:
//!
//! 1. **[`openness`]** — for every face, fire a spray of rays into the hemisphere above it and
//!    record the fraction that escape. Outer membrane lands at 0.5–1.0, pocket wall at 0. The
//!    distribution is sharply bimodal, so the threshold that separates the two is not really a
//!    tuning parameter.
//! 2. **Smooth** that field over the faces, so the iso-contour the cut follows is short and
//!    smooth rather than ragged. On a 3.5 M-face mesh this takes the result from 7,339 holes
//!    with 54,112 boundary vertices to 513 with 12,094 — a far better cut. Smoothing decides
//!    where the contour sits, but not how far the cut reaches: diffusion averages a pocket
//!    against the wide-open membrane beside it, so on a pocket narrower than `smooth` reaches
//!    the thresholded region shrinks back towards the blind floor. The raw field settles the
//!    extent instead — it is bimodal, and a face that saw nothing reads exactly zero — so the
//!    cut grows from the contour through faces that saw nothing, and stops where the pocket
//!    does.
//! 3. **Cut**: drop the buried faces, then drop the components that enclose no volume — the
//!    inside-out shreds of pocket wall left behind. The cut has just opened every component
//!    it touched, so each one is closed before it is measured; see [`enclosing_mask`] for
//!    what measuring an open surface instead costs.
//! 4. **Cap** the resulting holes — the pocket mouths, and anything the mesh arrived open at
//!    — with [`crate::caps`]. Closed, but not manifold: capping may not add vertices here, so
//!    a chord occasionally lands on an edge the mesh already had.
//!
//! Then repeat: capping a mouth turns a partially-open neighbour into a fully buried one, so
//! later passes find pockets the first threshold could not see. It converges fast — 18.7% of
//! faces buried, then 0.5%, then 0.2%.
//!
//! The extent rule in step 2 is what makes that true rather than nearly true. Without it a cut
//! that stops inside a pocket leaves a cap that is itself buried — a cap at the bottom of a
//! tube can see no more than the floor it replaced — so the next pass cuts that cap and lays
//! an identical one back. On a real neuron that accounted for 40-48% of every pass after the
//! first; on a mesh whose every pocket is in that state the passes made no progress at all.
//!
//! # Winding
//!
//! Faces must be wound **outward**. Step 1 fires into the hemisphere the face normal points
//! into, and step 3 reads a signed volume, so both of them are questions about which side is
//! out. A mesh wound consistently *inward* reads as entirely buried and comes back empty,
//! which at least fails loudly; one wound *inconsistently* does not, because the faces that
//! disagree read as buried and get cut out of otherwise healthy membrane. This is also why
//! [`crate::caps::triangulate_rings`] winding its caps against their rings is load-bearing
//! here rather than cosmetic: a cap that agreed with its ring would be a hole punched in the
//! next pass's field.
//!
//! # Why the ray casting is ours
//!
//! [`openness`] is ~90% of the runtime, and what it asks of a ray is unusually little: not
//! what it hit, nor where, nor which hit came first, only *whether it got out*. A general
//! collision library answers a strictly harder question and pays for it — see [`crate::bvh`],
//! which is a hundred lines of tree and forty of traversal because that is all the question
//! needs.
//!
//! # Two of `mesh`'s answers are load-bearing here
//!
//! Step 3 needs [`crate::mesh::mesh_face_components`] with `manifold_only`, and it is worth
//! saying why, because the coarser readings fail in a way that is not loud. "Encloses positive
//! volume" is a question about a *surface* with a well-defined inside, so the components have
//! to break at exactly the junctions that have no inside. Vertex connectivity welds an
//! organelle to the membrane it kisses through their shared vertex ring; face connectivity
//! welds it through the pinch edge itself. Either way the inside-out shreds sail through the
//! filter attached to something real — and on one run at four rays per face, a component whose
//! net signed volume had flipped negative took the entire neuron with it, leaving 22 faces.
//!
//! Step 4 needs rings that are simple and a triangulator that does not have to flatten them,
//! which is what [`crate::caps::trace_loops`] and [`crate::caps::triangulate_rings`] give. A
//! pocket mouth is the hardest ring this pipeline can produce — large, and folded enough that
//! its shadow crosses itself on every plane.

use ndarray::{Array1, Array2, ArrayView2};
use rayon::prelude::*;

use crate::bvh::Bvh;
use crate::caps::{basis, boundary_halfedges, trace_loops, triangulate_rings};
use crate::mesh::{face_of, mesh_face_components, sorted_edge_keys_indexed};
use crate::simplify::{cross, dot, normalize, sub};
use crate::threads::with_pool;

/// Faces a component must have before it is allowed to count as enclosing anything.
///
/// A closed surface needs four, so anything under it is a scrap by arithmetic rather than by
/// judgement — which is why this is a constant and the pipeline's real knobs are arguments.
const MIN_COMPONENT_FACES: u32 = 4;

// ---------------------------------------------------------------------------
// Face geometry
// ---------------------------------------------------------------------------

/// The `i`th vertex of a flat `(V, 3)` coordinate buffer.
#[inline]
fn point(v: &[f64], i: u32) -> [f64; 3] {
    let b = 3 * i as usize;
    [v[b], v[b + 1], v[b + 2]]
}

/// Centroid of one face.
#[inline]
fn face_center(v: &[f64], f: &[u32]) -> [f64; 3] {
    let (a, b, c) = (point(v, f[0]), point(v, f[1]), point(v, f[2]));
    [
        (a[0] + b[0] + c[0]) / 3.0,
        (a[1] + b[1] + c[1]) / 3.0,
        (a[2] + b[2] + c[2]) / 3.0,
    ]
}

/// Mean of the face centres: the anchor a signed volume is taken on.
fn faces_centroid(v: &[f64], faces: &[u32]) -> [f64; 3] {
    let mut o = [0.0f64; 3];
    for f in faces.chunks_exact(3) {
        for (slot, x) in o.iter_mut().zip(face_center(v, f)) {
            *slot += x;
        }
    }
    for x in o.iter_mut() {
        *x /= (faces.len() / 3) as f64;
    }
    o
}

/// `sum over faces of (a - o).((b - o) x (c - o)) / 6`.
///
/// On a closed surface this is its volume and `o` cancels. On an open one it is the volume
/// of that surface closed by coning its holes to `o`, which is a quite different thing — see
/// [`enclosing_mask`], whose whole job is to not be caught by the difference.
fn tet_sum(v: &[f64], faces: &[u32], o: [f64; 3]) -> f64 {
    faces
        .chunks_exact(3)
        .map(|f| {
            let (a, b, c) = (
                sub(point(v, f[0]), o),
                sub(point(v, f[1]), o),
                sub(point(v, f[2]), o),
            );
            dot(a, cross(b, c)) / 6.0
        })
        .sum()
}

/// Unit normal and centroid of one face; the normal is `None` for a degenerate triangle.
#[inline]
fn face_frame(v: &[f64], f: &[u32]) -> (Option<[f64; 3]>, [f64; 3]) {
    let (a, b, c) = (point(v, f[0]), point(v, f[1]), point(v, f[2]));
    (normalize(cross(sub(b, a), sub(c, a))), face_center(v, f))
}

/// How far off the surface to launch, when the caller does not say: 5% of the median edge
/// length, which keeps the whole thing scale-free — these meshes turn up in nanometres and in
/// microns, and nothing else here has a length in it.
///
/// Sampled rather than measured: a stride that takes ~10k faces is enough to place a median to
/// well inside the factor of two that matters, and saves sorting `3F` lengths.
fn default_offset(v: &[f64], faces: &[u32]) -> f64 {
    let n = faces.len() / 3;
    if n == 0 {
        return 0.0;
    }
    let stride = (n / 10_000).max(1);
    let mut lens: Vec<f64> = (0..n)
        .step_by(stride)
        .map(|i| {
            let e = sub(point(v, faces[3 * i + 1]), point(v, faces[3 * i]));
            dot(e, e).sqrt()
        })
        .collect();
    lens.sort_unstable_by(f64::total_cmp);
    0.05 * lens[lens.len() / 2]
}

// ---------------------------------------------------------------------------
// Openness
// ---------------------------------------------------------------------------

/// One round of SplitMix64.
#[inline]
fn mix(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Two uniforms in `[0, 1)` for ray `r` of face `f`.
///
/// Hashed from the indices rather than drawn from a stream, which is what makes the answer
/// independent of how the faces were split across threads — and of whether this is a full
/// sweep or a re-cast of a handful of faces. A streaming generator would make the second of
/// those quietly disagree with the first.
#[inline]
fn uniforms(seed: u64, f: u32, r: u32) -> (f64, f64) {
    let h = mix(seed ^ mix(((f as u64) << 20) | r as u64));
    let g = mix(h);
    const SCALE: f64 = 1.0 / (1u64 << 53) as f64;
    (((h >> 11) as f64) * SCALE, ((g >> 11) as f64) * SCALE)
}

/// Fraction of rays leaving each face that escape the mesh.
///
/// The one thing that separates an invagination from real membrane is that the cell encloses
/// it, so this asks that directly: fire a cosine-weighted spray of rays out of each face and
/// count how many get away.
///
/// Arguments
/// ---------
/// - `vertices`: (V, 3) vertex positions.
/// - `faces`:    (F, 3) triangular faces as vertex indices.
/// - `mask`:   Cast only from the faces this marks. The whole mesh still blocks rays — only
///   the *sources* are restricted — so each value is exactly the one a full sweep would have
///   produced. `None` casts from every face.
/// - `n_rays`:   Rays per face. 16 is plenty; the signal is bimodal, so this only has to
///   resolve "none got out" from "some did". 8 halves the cost for no measured difference; 4
///   is visibly past the edge.
/// - `offset`:   How far off the surface to start each ray. `None` is [`default_offset`].
/// - `seed`:     Fixes the spray. See [`uniforms`] for what it does and does not depend on.
/// - `threads`:  Size of the thread pool, or `None` for all cores.
///
/// Returns
/// -------
/// A `Vec<f64>` of fractions in `[0, 1]` — one per face, or one per *selected* face in face
/// order when `mask` was given. A degenerate face has no hemisphere to sample and comes back
/// as 1.0.
pub fn openness(
    vertices: ArrayView2<f64>,
    faces: ArrayView2<u32>,
    mask: Option<&[bool]>,
    n_rays: u32,
    offset: Option<f64>,
    seed: u64,
    threads: Option<usize>,
) -> Vec<f64> {
    assert!(n_rays >= 1, "`n_rays` must be at least 1");
    let vs = vertices.as_standard_layout();
    let v: &[f64] = vs.as_slice().expect("standard layout is contiguous");
    let fs = faces.as_standard_layout();
    let f: &[u32] = fs.as_slice().expect("standard layout is contiguous");

    with_pool(threads, || {
        let bvh = Bvh::build(v, f);
        openness_with(&bvh, v, f, mask, n_rays, offset, seed)
    })
}

/// [`openness`] against a tree the caller already has. Runs on the ambient pool.
fn openness_with(
    bvh: &Bvh,
    v: &[f64],
    faces: &[u32],
    mask: Option<&[bool]>,
    n_rays: u32,
    offset: Option<f64>,
    seed: u64,
) -> Vec<f64> {
    let n = faces.len() / 3;
    let offset = offset.unwrap_or_else(|| default_offset(v, faces));
    // Long enough to leave the mesh from anywhere inside it, which is as good as infinity for
    // a ray that only has to get out. The tree already walked every vertex for the box this
    // comes from, so asking it is cheaper than measuring again.
    let far = bvh.reach();

    let selected: Vec<u32> = match mask {
        Some(w) => {
            assert_eq!(w.len(), n, "`mask` must have one flag per face");
            (0..n as u32).filter(|&i| w[i as usize]).collect()
        }
        None => (0..n as u32).collect(),
    };

    selected
        .par_iter()
        .map(|&i| {
            let (normal, center) = face_frame(v, &faces[3 * i as usize..3 * i as usize + 3]);
            // A zero-area face has no hemisphere over it and no rays to fire. Calling it fully
            // open is the safe way round: this function only ever removes what it is sure is
            // buried.
            let Some(nrm) = normal else {
                return 1.0;
            };

            // An arbitrary tangent frame. The spray is rotationally symmetric about the
            // normal, so which one it is does not matter — only that it is orthonormal, which
            // is exactly what `caps` needs of a ring's normal and why the two share this.
            let Some((u, w)) = basis(nrm) else {
                return 1.0;
            };

            let src = [
                center[0] + offset * nrm[0],
                center[1] + offset * nrm[1],
                center[2] + offset * nrm[2],
            ];

            let mut escaped = 0u32;
            for r in 0..n_rays {
                // Cosine-weighted: the density a diffuse surface sees, so faces are not
                // over-sampled towards the grazing angles where a ray tells you least.
                let (z, t) = uniforms(seed, i, r);
                let phi = t * std::f64::consts::TAU;
                let st = (1.0 - z).sqrt();
                let (sp, cp) = phi.sin_cos();
                let cz = z.sqrt();
                let d = [
                    st * cp * u[0] + st * sp * w[0] + cz * nrm[0],
                    st * cp * u[1] + st * sp * w[1] + cz * nrm[1],
                    st * cp * u[2] + st * sp * w[2] + cz * nrm[2],
                ];
                if !bvh.occluded(src, d, far) {
                    escaped += 1;
                }
            }
            escaped as f64 / n_rays as f64
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Face adjacency and the field over it
// ---------------------------------------------------------------------------

/// Faces that share an edge carrying exactly two of them, as a CSR neighbour list.
///
/// The adjacency behind [`crate::mesh::mesh_face_components`]'s `manifold_only` reading, in
/// the form the diffusion and the dilation want rather than as labels. Same reason for taking
/// only the two-face edges: this is a walk over a surface, and at a pinch ring there is no
/// single surface to walk.
/// A triangle has three edges and each can join it to at most one other face, so a row is
/// three wide and no offset array is needed — only how much of each row is filled.
struct FaceGraph {
    degree: Vec<u8>,
    neighbours: Vec<u32>,
}

impl FaceGraph {
    /// Runs on the ambient rayon pool.
    fn build(faces: &[u32], n_faces: usize) -> Self {
        // The same buffer `mesh_face_components` unions over, grouped the same way: a run of
        // equal keys is one undirected edge, and a position over three is the face that named
        // it. Taken from `mesh` rather than rebuilt so the `3F` convention and the key packing
        // stay defined in one place — which is what that function's docs ask of consumers.
        let packed = sorted_edge_keys_indexed(faces);

        let mut degree = vec![0u8; n_faces];
        let mut neighbours = vec![u32::MAX; 3 * n_faces];
        let mut join = |a: u32, b: u32| {
            let d = &mut degree[a as usize];
            neighbours[3 * a as usize + *d as usize] = b;
            *d += 1;
        };
        for run in packed.chunk_by(|a, b| a >> 64 == b >> 64) {
            // Exactly two is a manifold edge — the reading this needs, since a pinch ring is
            // where an organelle parts company with the membrane and there is no single
            // surface to walk across. A degenerate face names an edge twice and would pair
            // with itself; skip that rather than give it a self-loop.
            if run.len() == 2 {
                let (a, b) = (face_of(run[0]), face_of(run[1]));
                if a != b {
                    join(a, b);
                    join(b, a);
                }
            }
        }

        Self { degree, neighbours }
    }

    #[inline]
    fn neighbours_of(&self, i: usize) -> &[u32] {
        &self.neighbours[3 * i..3 * i + self.degree[i] as usize]
    }

    fn len(&self) -> usize {
        self.degree.len()
    }

    /// Diffuse a per-face scalar: each round replaces a value with the mean of itself and the
    /// mean of its neighbours.
    ///
    /// Thresholding the raw field cuts along a ragged contour. A few rounds of this shorten it
    /// several fold without moving where the cut sits — the faces removed barely change, the
    /// boundary they leave changes a great deal.
    fn smooth(&self, x: &[f64], iterations: u32) -> Vec<f64> {
        let mut cur = x.to_vec();
        let mut next = vec![0.0; x.len()];
        for _ in 0..iterations {
            next.par_iter_mut().enumerate().for_each(|(i, out)| {
                let nb = self.neighbours_of(i);
                let mean = if nb.is_empty() {
                    cur[i]
                } else {
                    nb.iter().map(|&j| cur[j as usize]).sum::<f64>() / nb.len() as f64
                };
                *out = 0.5 * cur[i] + 0.5 * mean;
            });
            std::mem::swap(&mut cur, &mut next);
        }
        cur
    }

    /// Grow a face mask across shared edges for as long as `allow` says, without a hop limit.
    ///
    /// [`Self::dilate`] grows a fixed distance regardless of what it crosses; this grows an
    /// arbitrary distance but only through faces that pass a test, so the region it ends up
    /// with is bounded by the mesh rather than by a number.
    fn flood(&self, mask: &mut [bool], allow: impl Fn(usize) -> bool) {
        let mut frontier: Vec<u32> = (0..self.len() as u32)
            .filter(|&i| mask[i as usize])
            .collect();
        while let Some(i) = frontier.pop() {
            for &j in self.neighbours_of(i as usize) {
                if !mask[j as usize] && allow(j as usize) {
                    mask[j as usize] = true;
                    frontier.push(j);
                }
            }
        }
    }

    /// Grow a face mask by `hops` steps across shared edges.
    fn dilate(&self, mask: &mut [bool], hops: u32) {
        let mut frontier: Vec<u32> = (0..self.len() as u32)
            .filter(|&i| mask[i as usize])
            .collect();
        for _ in 0..hops {
            if frontier.is_empty() {
                break;
            }
            let mut next = Vec::new();
            for &i in &frontier {
                for &j in self.neighbours_of(i as usize) {
                    if !mask[j as usize] {
                        mask[j as usize] = true;
                        next.push(j);
                    }
                }
            }
            frontier = next;
        }
    }
}

// ---------------------------------------------------------------------------
// Dropping what encloses nothing
// ---------------------------------------------------------------------------

/// What [`enclosing_mask`] made of the faces it was handed.
struct Enclosure {
    /// Whether to keep each face. Its component is kept if it bounds positive volume — or if
    /// the capping could not close it, in which case it was never judged at all.
    keep: Vec<bool>,
    /// Whether each face's component has the faces to bound anything: four, for a closed
    /// surface. Arithmetic rather than measurement, so the guard in [`drop_internals`] has no
    /// say in it.
    big_enough: Vec<bool>,
    /// Faces whose component the capping left open, and which were therefore kept unjudged.
    unclosed: usize,
}

/// Which faces sit in a component that encloses positive volume.
///
/// Deliberately *not* "keep the largest component": neuron meshes routinely arrive in many
/// legitimate pieces — one hemibrain example mesh has 299 — and keeping only the biggest
/// silently deletes real neurite. Only inside-out or degenerate scraps are dropped.
///
/// # Every component is closed before it is measured
///
/// Summing a surface's tetrahedra on a point `p` gives its volume only when it is closed.
/// What it gives otherwise is the volume of that surface *closed by coning every hole to
/// `p`*, and the components handed here are precisely the ones the cut has just opened — the
/// mouths are not capped until the next step. Coning to a point is a ruinous way to close a
/// pocket mouth: the cones reach right across the cell and lie on top of one another, so what
/// they add and subtract has nothing to do with the cell.
///
/// One number puts the scale on it. Every face the cut removes takes `(x - p).n dA / 3` out
/// of the sum, which is large wherever the surface is far from `p` — while all a membrane
/// encloses is its own area times its own thickness. So the mouths need only be a sliver of
/// the component to outweigh it. On one neuron 160 µm across, 3,414 faces' worth of
/// mouths — 0.6% of the component — read its +6.8e10 as -7.5e10 and dropped all 558,144 of
/// its faces, which was most of the neuron.
///
/// So close each component first, with the same [`cap_holes`] the pipeline uses a step later
/// — applied to the component alone, which is the surface the question is about. The two
/// closings are not the same caps: boundary is decided by the parity of the faces on an edge,
/// so an edge where two components meet reads differently in isolation than it does in the
/// whole mesh. That is a reason to keep them separate, not to share them. It does mean this
/// measurement now leans on [`cap_holes`]: change what that closes and you change which
/// components survive. On a closed surface the anchor cancels, so the reading becomes a property of the
/// component and of nothing else.
///
/// The anchor is still each component's own centroid, because the cancelling is exact in
/// reals and not in floating point: these meshes sit hundreds of thousands of dataset
/// nanometres from the origin, and subtracting the centroid *before* the cross product is
/// what keeps every term small. The tempting one-pass form is a trap — the identity
/// `V(p) = V(0) - p.(sum of face cross products)/6` removes the cone and is exact in reals,
/// but it reaches the answer as the difference of two numbers the size of the coordinates,
/// which at 1e6 agrees with the origin reading bit for bit. Make every *term* small and
/// nothing cancels.
///
/// # And where it cannot close one, it does not answer
///
/// Capping adds no vertices, so there are rings it cannot close — a boundary that pinches at
/// a vertex can split into pieces too short to bound a face, and then nothing is capped at
/// all. A component still carrying boundary is one whose reading is the cone again, so that
/// is asked rather than assumed: close the component, then check that it closed. One that
/// did not is kept, unjudged, and counted in [`Pass::unclosed`].
///
/// Refusing to answer is the only honest thing to do with a question this cannot put
/// properly, and it is cheap — on eleven segmented neurons of 0.3 M to 2.4 M faces, not one
/// component failed to close. The branch is there for the mesh that does.
fn enclosing_mask(v: &[f64], faces: &[u32]) -> Enclosure {
    let n = faces.len() / 3;
    let n_faces_arr = ArrayView2::from_shape((n, 3), faces).expect("(F, 3)");
    let labels = mesh_face_components(n_faces_arr, true, None);

    // Gather each component's faces into a run of its own, so it can be lifted out and closed
    // on its own. Labels are the smallest face index in each component, so they index `0..n`
    // directly and a counting pass does this without a sort: tally into `offset`, sum it in
    // place, then scatter. A component's rows are then `grouped[offset[l]..offset[l + 1]]`,
    // which is also where its face count comes from.
    let mut offset = vec![0u32; n + 1];
    for &l in labels.iter() {
        offset[l as usize + 1] += 1;
    }
    for i in 0..n {
        offset[i + 1] += offset[i];
    }
    let mut cursor = offset.clone();
    let mut grouped = vec![0u32; n];
    for (i, &l) in labels.iter().enumerate() {
        grouped[cursor[l as usize] as usize] = i as u32;
        cursor[l as usize] += 1;
    }

    // Stated once, and read three times below: to skip the measuring, and in each of the two
    // masks. The empty slots, which are most of `0..n`, fail it too.
    let enough = |l: usize| offset[l + 1] - offset[l] >= MIN_COMPONENT_FACES;

    let view = ArrayView2::from_shape((v.len() / 3, 3), v).expect("(V, 3)");
    // Per component: the volume its closed form encloses, and whether it closed at all.
    let read: Vec<(f64, bool)> = (0..n)
        .into_par_iter()
        .map(|l| {
            // Under the floor it bounds nothing whatever it measures, so do not measure it.
            if !enough(l) {
                return (0.0, true);
            }
            let rows = &grouped[offset[l] as usize..offset[l + 1] as usize];

            let piece: Vec<u32> = rows
                .iter()
                .flat_map(|&i| faces[3 * i as usize..3 * i as usize + 3].iter().copied())
                .collect();
            let mut shut = piece.clone();
            shut.extend_from_slice(&cap_holes(view, &piece));

            // Ask whether that worked, rather than assume it. Capping adds no vertices, so a
            // ring it cannot close leaves the component open — and the reading on an open
            // surface is the cone to the anchor, which is the one thing this function exists
            // not to be fooled by. A component still carrying boundary is one this cannot
            // answer for, and saying so is better than answering wrongly.
            let edges = ArrayView2::from_shape((shut.len() / 3, 3), &shut[..]).expect("(F, 3)");
            let closed = boundary_halfedges(edges, None).is_empty();

            // The anchor comes off the component as the cut left it; the caps are derived
            // from it and should not be allowed to move it.
            let o = faces_centroid(v, &piece);
            (tet_sum(v, &shut, o), closed)
        })
        .collect();

    let unclosed = labels
        .iter()
        .filter(|&&l| enough(l as usize) && !read[l as usize].1)
        .count();
    Enclosure {
        keep: labels
            .iter()
            .map(|&l| {
                let (volume, closed) = read[l as usize];
                enough(l as usize) && (volume > 0.0 || !closed)
            })
            .collect(),
        big_enough: labels.iter().map(|&l| enough(l as usize)).collect(),
        unclosed,
    }
}

// ---------------------------------------------------------------------------
// Capping
// ---------------------------------------------------------------------------

/// Triangulate every opening in `faces`, adding no vertices.
///
/// Note this also closes openings the mesh arrived with — a neurite truncated at the edge of
/// the dataset, say. For skeletonisation that is what we want: a stump left open splits the
/// wave front just as a pocket mouth does.
fn cap_holes(v: ArrayView2<f64>, faces: &[u32]) -> Vec<u32> {
    let view = ArrayView2::from_shape((faces.len() / 3, 3), faces).expect("(F, 3)");
    let halfedges = boundary_halfedges(view, None);
    if halfedges.is_empty() {
        return Vec::new();
    }
    let (rings, offsets) = trace_loops(halfedges.view());
    if rings.is_empty() {
        return Vec::new();
    }
    let caps = triangulate_rings(
        rings.as_slice().expect("owned"),
        offsets.as_slice().expect("owned"),
        v,
        None,
    );
    caps.into_raw_vec_and_offset().0
}

// ---------------------------------------------------------------------------
// The pipeline
// ---------------------------------------------------------------------------

/// What one pass of [`drop_internals`] did, for callers that want to watch it converge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Pass {
    /// Faces the mesh had going in.
    pub faces_before: usize,
    /// Faces the cut took: those whose smoothed openness fell below the threshold, plus
    /// whatever the extent flood reached from them.
    pub buried: usize,
    /// Faces re-cast this pass — the whole mesh on the first, the collar around the last
    /// pass's caps afterwards.
    pub recast: usize,
    /// Faces dropped for enclosing no volume: the inside-out shreds of pocket wall the cut
    /// leaves behind.
    pub dropped: usize,
    /// Faces that step condemned and did not get, because what it was about to remove came
    /// to more of the mesh than it was about to keep, and the backstop in [`drop_internals`]
    /// held it back. Zero on a normal pass.
    pub spared: usize,
    /// Faces whose component the capping could not close, and which that step therefore kept
    /// without judging: an open surface has no volume to read. Zero on a clean mesh; a large
    /// number means the capping is struggling, not that the mesh is full of cavities.
    pub unclosed: usize,
    /// Faces added to close the openings the cut left.
    pub capped: usize,
    /// Faces the mesh had coming out.
    pub faces_after: usize,
}

/// Strip invaginated surface from a mesh and close it back up.
///
/// See the module docs for what this is for and why each step is the way it is.
///
/// # Iterating, and why later passes are cheap
///
/// A pass only changes the mesh where the previous one cut, so the openness field is only
/// stale near the new caps: 97% of what pass 1 finds sits within 20 faces of a cap, and of the
/// 2.0 M faces no walk from a cap reaches at all, 164 are buried. So each pass after the first
/// carries the field forward and re-casts only within `hops` of the caps, which is ~12% of the
/// mesh in pass 2. The approximation is in *which faces get a fresh value*, not in the values:
/// the whole mesh still blocks rays.
///
/// # The guard on dropping components
///
/// Dropping the components that enclose no volume is the step with the most to go wrong in
/// it, because a volume is only as good as the closing it was read on. Most of that is
/// handled where the question is asked: [`enclosing_mask`] closes each component and then
/// checks that it closed, and keeps the ones it could not, unjudged — see [`Pass::unclosed`].
///
/// The backstop here is for the rest. If the reading still condemns more than half of the
/// surface it judged, it is held back for that pass. A cell is not mostly cavity, so a
/// reading saying so has gone wrong in some way this cannot name, and a mesh that keeps a few
/// inside-out scraps is a far better failure than a neuron thrown away. [`Pass::spared`]
/// counts what it was not allowed to take.
///
/// What is held back is the *reading*, and only the reading. A component with too few faces
/// to bound anything still goes, because that is arithmetic and no doubt about a volume makes
/// three faces a cell — and for the same reason those components are left out of the count on
/// both sides of the test, so that a drift of loose scraps cannot decide whether the reading
/// is believed for the surface around them.
///
/// Arguments
/// ---------
/// - `vertices`:  (V, 3) vertex positions.
/// - `faces`:     (F, 3) triangular faces as vertex indices.
/// - `threshold`: Faces whose smoothed openness falls below this are cut. Do not raise it far:
///   the buried mode sits at exactly zero, so any small value works, while above ~0.1 the cut
///   starts eating real membrane. The rule that says so is the boundary-edge count, which is
///   flat from 0.02 to 0.10 and then explodes — that is the cut outrunning the capping.
/// - `n_rays`:    Rays per face; see [`openness`].
/// - `smooth`:    Diffusion rounds applied to the field before thresholding. This moves the
///   contour the cut starts from, not how far the cut reaches — the extent comes off the raw
///   field, so a pocket narrower than this reaches is still cut whole. Raise it far enough
///   and it lifts the pocket clear of the threshold altogether, at which point nothing is cut
///   rather than too much.
/// - `iterations`: Passes of the whole cycle. Three is plenty.
/// - `hops`:      How far past the previous pass's caps to re-cast. Worth keeping above
///   `smooth`, which is how far a stale value at the rim can diffuse inward. `None` re-casts
///   everything, which costs about twice as much and moves no mesh metric more than a percent.
/// - `seed`:      Fixes the ray spray.
/// - `threads`:   Size of the thread pool, or `None` for all cores.
///
/// Returns
/// -------
/// `(vertices, faces, keep, passes)`. `keep` gives each surviving vertex's index in the input:
/// the repair only ever *removes* vertices — caps re-use the ones already on the boundary — so
/// that is all a caller needs to carry a vertex map, connectors or any per-vertex annotation
/// across. `passes` is one [`Pass`] per pass actually run.
#[allow(clippy::too_many_arguments)]
pub fn drop_internals(
    vertices: ArrayView2<f64>,
    faces: ArrayView2<u32>,
    threshold: f64,
    n_rays: u32,
    smooth: u32,
    iterations: u32,
    hops: Option<u32>,
    seed: u64,
    threads: Option<usize>,
) -> (Array2<f64>, Array2<u32>, Array1<u32>, Vec<Pass>) {
    assert_eq!(vertices.ncols(), 3, "`vertices` must have shape (V, 3)");
    assert_eq!(faces.ncols(), 3, "`faces` must have shape (F, 3)");
    // Checked here rather than in each binding: this is where `n_rays` means something, so a
    // direct Rust or R caller gets the same answer the Python wrapper already gives.
    assert!(n_rays >= 1, "`n_rays` must be at least 1");

    let vs = vertices.as_standard_layout();
    let mut v: Vec<f64> = vs.as_slice().expect("standard layout").to_vec();
    let fs = faces.as_standard_layout();
    let mut f: Vec<u32> = fs.as_slice().expect("standard layout").to_vec();

    let mut keep: Vec<u32> = (0..(v.len() / 3) as u32).collect();
    let mut field: Vec<f64> = Vec::new();
    // Where the last pass's caps start. They are appended at the tail and nothing reorders
    // faces between passes, so one index says everything a per-face mask would — and it is the
    // seed the next pass grows its re-cast region from.
    let mut cap_start: Option<usize> = None;
    let mut log = Vec::new();

    with_pool(threads, || {
        for _ in 0..iterations {
            let n = f.len() / 3;
            if n == 0 {
                break;
            }
            let graph = FaceGraph::build(&f, n);
            let bvh = Bvh::build(&v, &f);

            // Which faces this pass asked afresh. `None` is "all of them" — the first pass,
            // and any pass when `hops` is off — and is what the flood below reads to keep
            // itself inside values this pass actually measured.
            let (recast, region) = match (cap_start, hops) {
                (Some(start), Some(hops)) => {
                    let mut region = vec![false; n];
                    region[start..].fill(true);
                    graph.dilate(&mut region, hops);
                    let fresh = openness_with(&bvh, &v, &f, Some(&region), n_rays, None, seed);
                    // One value per selected face, in face order — so walking the field and
                    // the region together lands each where it belongs.
                    let recast = fresh.len();
                    let mut fresh = fresh.into_iter();
                    for (slot, &inside) in field.iter_mut().zip(&region) {
                        if inside {
                            *slot = fresh.next().expect("one value per selected face");
                        }
                    }
                    (recast, Some(region))
                }
                _ => {
                    field = openness_with(&bvh, &v, &f, None, n_rays, None, seed);
                    (n, None)
                }
            };

            let smoothed = graph.smooth(&field, smooth);
            let mut buried: Vec<bool> = smoothed.iter().map(|&x| x < threshold).collect();

            // Smoothing decides where the cut starts; the raw field decides how far it goes.
            // Diffusion averages a pocket against the wide-open membrane beside it, so on a
            // pocket narrower than `smooth` reaches, the thresholded region shrinks back
            // towards the blind floor and the cut stops inside the pocket — whereupon the cap
            // it leaves is itself buried, and all the next pass can do is lay the same cap
            // again. The raw field cannot be washed out that way: it is bimodal, and a face
            // that saw nothing at all reads exactly zero. So grow the cut through the faces
            // that saw nothing, which carries it to where the pocket stops being one.
            //
            // Only through faces this pass re-cast, though. Everywhere else `field` is as
            // stale as `hops` allows, and a stale zero is a face that saw nothing *before*
            // the last cut opened the mesh up — which is exactly the face most likely to see
            // something now. The threshold can read those: a wrong one costs the one face. A
            // flood cannot, because it has no hop limit of its own, so a wrong one is a
            // doorway it walks through and keeps going. Confining it to the collar is what
            // puts it back under the bound `hops` is documented to give. It is a small
            // correction — four faces in 331,708 on the neuron this was built against — but
            // it is the difference between an approximation with a stated reach and one
            // without.
            graph.flood(&mut buried, |i| {
                field[i] == 0.0 && region.as_ref().is_none_or(|r| r[i])
            });

            let n_buried = buried.iter().filter(|&&b| b).count();
            if n_buried == 0 {
                log.push(Pass {
                    faces_before: n,
                    buried: 0,
                    recast,
                    dropped: 0,
                    spared: 0,
                    unclosed: 0,
                    capped: 0,
                    faces_after: n,
                });
                break;
            }

            // Cut, then drop what the cut left enclosing nothing. The second filter has to see
            // the cut mesh — its components are not the parent's — so the intermediate face
            // array is unavoidable, but tracking which original face each row came from means
            // the field is rebuilt once rather than filtered twice.
            let survivors: Vec<usize> = (0..n).filter(|&i| !buried[i]).collect();
            let cut: Vec<u32> = survivors
                .iter()
                .flat_map(|&i| f[3 * i..3 * i + 3].iter().copied())
                .collect();
            // A volume is only as good as the closing it was read on, and `enclosing_mask`
            // now says which components it could not close rather than leaving that to be
            // hoped for: those are kept, unjudged. So most of the way this step can go wrong
            // is handled where the question is asked, per component.
            //
            // The backstop is for the rest. If what survives that still excludes more of the
            // mesh than it keeps, believe the mesh rather than the reading: a cell is not
            // mostly cavity, and the inside-out shreds are a rim of pocket wall, not the bulk
            // of a neuron. Being wrong this way round leaves some scraps in the mesh; being
            // wrong the other way throws a neuron away, which is what this went looking for.
            // What the backstop holds back is the *reading*, so `big_enough` still stands —
            // three faces bound nothing by arithmetic, and no doubt about a volume makes them
            // a cell.
            //
            // Both sides of that test are about the measurement, then: how much of what it
            // judged did it condemn. Counting the components below the floor on either side
            // would let a drift of loose triangles decide whether a reading is believed.
            let e = enclosing_mask(&v, &cut);
            let judged = e.big_enough.iter().filter(|&&b| b).count();
            let condemned = e
                .big_enough
                .iter()
                .zip(&e.keep)
                .filter(|(&big, &keep)| big && !keep)
                .count();
            let overruled = 2 * condemned > judged;
            let verdict = if overruled { &e.big_enough } else { &e.keep };
            let dropped = verdict.iter().filter(|&&k| !k).count();
            let spared = if overruled { condemned } else { 0 };
            let unclosed = e.unclosed;

            let mut kept_faces = Vec::with_capacity(cut.len());
            let mut kept_field = Vec::with_capacity(survivors.len());
            for (row, &i) in survivors.iter().enumerate() {
                if verdict[row] {
                    kept_faces.extend_from_slice(&f[3 * i..3 * i + 3]);
                    kept_field.push(field[i]);
                }
            }
            f = kept_faces;
            field = kept_field;

            // Compact the vertices. Positions are not unique in these meshes, so a
            // nearest-neighbour lookup could not recover this mapping afterwards — it has to
            // be carried.
            let mut used = vec![false; v.len() / 3];
            for &i in &f {
                used[i as usize] = true;
            }
            let mut remap = vec![u32::MAX; used.len()];
            let mut new_v = Vec::with_capacity(v.len());
            let mut new_keep = Vec::with_capacity(keep.len());
            for (i, &u) in used.iter().enumerate() {
                if u {
                    remap[i] = (new_v.len() / 3) as u32;
                    new_v.extend_from_slice(&v[3 * i..3 * i + 3]);
                    new_keep.push(keep[i]);
                }
            }
            for i in f.iter_mut() {
                *i = remap[*i as usize];
            }
            v = new_v;
            keep = new_keep;

            let n_before = f.len() / 3;
            let view = ArrayView2::from_shape((v.len() / 3, 3), &v[..]).expect("(V, 3)");
            let new_faces = cap_holes(view, &f);
            f.extend_from_slice(&new_faces);
            let n_after = f.len() / 3;

            // The caps' own openness is never read: they seed the next pass's region, so they
            // are always among the faces it re-casts.
            cap_start = Some(n_before);
            field.resize(n_after, 0.0);

            log.push(Pass {
                faces_before: n,
                buried: n_buried,
                recast,
                dropped,
                spared,
                unclosed,
                capped: n_after - n_before,
                faces_after: n_after,
            });
        }
    });

    let n_v = v.len() / 3;
    let n_f = f.len() / 3;
    (
        Array2::from_shape_vec((n_v, 3), v).expect("(V, 3)"),
        Array2::from_shape_vec((n_f, 3), f).expect("(F, 3)"),
        Array1::from_vec(keep),
        log,
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// An axis-aligned box from `lo` to `hi`, wound outward.
    fn box_mesh(lo: [f64; 3], hi: [f64; 3]) -> (Vec<f64>, Vec<u32>) {
        let v = vec![
            lo[0], lo[1], lo[2], hi[0], lo[1], lo[2], hi[0], hi[1], lo[2], lo[0], hi[1], lo[2],
            lo[0], lo[1], hi[2], hi[0], lo[1], hi[2], hi[0], hi[1], hi[2], lo[0], hi[1], hi[2],
        ];
        let f = vec![
            0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4, 2, 3, 7, 2, 7, 6, 0, 4, 7, 0, 7,
            3, 1, 2, 6, 1, 6, 5,
        ];
        (v, f)
    }

    /// Reverse each triangle's winding, turning a surface inside out.
    fn flip(faces: &[u32]) -> Vec<u32> {
        faces
            .chunks_exact(3)
            .flat_map(|f| [f[0], f[2], f[1]])
            .collect()
    }

    /// A small box floating inside a large one: the classic free-floating organelle. The outer
    /// surface is open, the inner is not.
    fn nested() -> (Vec<f64>, Vec<u32>) {
        let (mut v, mut f) = box_mesh([0.0, 0.0, 0.0], [10.0, 10.0, 10.0]);
        let (iv, iff) = box_mesh([4.0, 4.0, 4.0], [6.0, 6.0, 6.0]);
        let base = (v.len() / 3) as u32;
        v.extend_from_slice(&iv);
        f.extend(iff.iter().map(|i| i + base));
        (v, f)
    }

    /// A box with a blind square tube pushed into its lid — the invagination this is all
    /// about, in the smallest form that still behaves like one. The tube is narrow and deep,
    /// so its wall sees nothing, and cutting the wall away opens the mouth: the one small
    /// case where `enclosing_mask` is handed a component the pipeline itself has opened.
    /// Repaired, it should come back as the plain box the pocket was pushed into.
    fn pocket() -> (Vec<f64>, Vec<u32>) {
        let ring = |z: f64, h: f64| {
            let (lo, hi) = (5.0 - h, 5.0 + h);
            vec![lo, lo, z, hi, lo, z, hi, hi, z, lo, hi, z]
        };
        let mut v = ring(0.0, 5.0); // 0..4  outer floor
        v.extend(ring(10.0, 5.0)); //  4..8  outer lid
        v.extend(ring(10.0, 0.5)); //  8..12 the mouth, in the lid
        v.extend(ring(2.0, 0.5)); //  12..16 the tube's own floor
        let (b, t, m, p) = (0u32, 4u32, 8u32, 12u32);

        // Outward is away from the cytoplasm, which inside the tube means into the lumen.
        let mut f = vec![b, b + 2, b + 1, b, b + 3, b + 2];
        for i in 0..4u32 {
            let j = (i + 1) % 4;
            f.extend_from_slice(&[t + i, b + i, b + j, t + i, b + j, t + j]); // outer wall
            f.extend_from_slice(&[t + i, t + j, m + i, t + j, m + j, m + i]); // lid, holed
            f.extend_from_slice(&[m + i, p + j, p + i, m + i, m + j, p + j]); // tube wall
        }
        f.extend_from_slice(&[p, p + 1, p + 2, p, p + 2, p + 3]);
        (v, f)
    }

    /// A closed loop of square-section tube: thin-walled, extended, and bent back on itself,
    /// which is the shape a neurite arbor has and a box has not.
    ///
    /// Stations run round a circle of radius `major`; each carries a square cross-section of
    /// half-width `minor`, cornered outer-top, outer-bottom, inner-bottom, inner-top. The
    /// eight faces of a station start with the two of its *outer* wall, so a test can take a
    /// bite out of the part furthest from the centre by index alone.
    fn tube_loop(stations: usize, major: f64, minor: f64) -> (Vec<f64>, Vec<u32>) {
        let mut v = Vec::with_capacity(12 * stations);
        for k in 0..stations {
            let (sin, cos) = (std::f64::consts::TAU * k as f64 / stations as f64).sin_cos();
            for (du, dw) in [(1.0, 1.0), (1.0, -1.0), (-1.0, -1.0), (-1.0, 1.0)] {
                let radius = major + du * minor;
                v.extend_from_slice(&[radius * cos, radius * sin, dw * minor]);
            }
        }
        let mut f = Vec::with_capacity(24 * stations);
        for k in 0..stations {
            let (a, b) = (4 * k as u32, 4 * ((k + 1) % stations) as u32);
            for i in 0..4u32 {
                let j = (i + 1) % 4;
                f.extend_from_slice(&[a + i, a + j, b + j, a + i, b + j, b + i]);
            }
        }
        (v, f)
    }

    /// A shallow dish of four triangles, wound inward: a scrap that encloses nothing, but
    /// shallow enough that its rays still get away, so the cut leaves it for the enclosing
    /// filter to deal with.
    fn inside_out_dish(x: f64) -> (Vec<f64>, Vec<u32>) {
        let mut v = Vec::new();
        for (dx, dy) in [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)] {
            v.extend_from_slice(&[x + dx, dy, 0.0]);
        }
        v.extend_from_slice(&[x + 0.5, 0.5, 0.2]);
        let f = (0..4u32)
            .flat_map(|i| [(i + 1) % 4, i, 4])
            .collect::<Vec<u32>>();
        (v, f)
    }

    /// What `enclosing_mask` would read if it did not close the component first.
    fn cone_volume(v: &[f64], faces: &[u32]) -> f64 {
        tet_sum(v, faces, faces_centroid(v, faces))
    }

    /// `drop_internals` with the arguments these tests mostly do not vary.
    fn repair(
        v: &[f64],
        f: &[u32],
        smooth: u32,
        iterations: u32,
    ) -> (Array2<f64>, Array2<u32>, Array1<u32>, Vec<Pass>) {
        drop_internals(
            view2(v, 3),
            view2u(f),
            0.05,
            16,
            smooth,
            iterations,
            Some(20),
            1985,
            None,
        )
    }

    fn view2<'a>(v: &'a [f64], cols: usize) -> ArrayView2<'a, f64> {
        ArrayView2::from_shape((v.len() / cols, cols), v).unwrap()
    }
    fn view2u<'a>(f: &'a [u32]) -> ArrayView2<'a, u32> {
        ArrayView2::from_shape((f.len() / 3, 3), f).unwrap()
    }

    #[test]
    fn a_lone_box_is_wide_open() {
        let (v, f) = box_mesh([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let o = openness(view2(&v, 3), view2u(&f), None, 16, None, 1985, None);
        assert_eq!(o.len(), 12);
        // Every face is on the outside; a cosine-weighted spray off a flat convex surface
        // escapes essentially always, and can only be blocked by a neighbouring wall at a
        // grazing angle.
        assert!(o.iter().all(|&x| x > 0.5), "{o:?}");
    }

    #[test]
    fn a_buried_surface_sees_nothing() {
        let (v, f) = nested();
        let o = openness(view2(&v, 3), view2u(&f), None, 16, None, 1985, None);
        let (outer, inner) = o.split_at(12);
        assert!(outer.iter().all(|&x| x > 0.5), "outer {outer:?}");
        assert!(inner.iter().all(|&x| x == 0.0), "inner {inner:?}");
    }

    /// Winding is load-bearing, and this pins the documented failure mode: rays follow the
    /// face normals, so a mesh wound inward reads as entirely buried and the repair empties
    /// it. Loud, at least — the inconsistent case is the one that is not.
    #[test]
    fn an_inward_wound_mesh_reads_as_buried() {
        let (v, f) = box_mesh([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let inward = flip(&f);
        let o = openness(view2(&v, 3), view2u(&inward), None, 16, None, 1985, None);
        assert!(o.iter().all(|&x| x == 0.0), "{o:?}");

        let (_, nf, _, _) = drop_internals(
            view2(&v, 3),
            view2u(&inward),
            0.05,
            16,
            2,
            1,
            None,
            1985,
            None,
        );
        assert_eq!(nf.nrows(), 0);
    }

    /// The `mask` argument restricts the *sources*, not the blockers, so a restricted sweep
    /// has to agree with the full one face for face.
    #[test]
    fn restricting_the_sources_changes_no_value() {
        let (v, f) = nested();
        let full = openness(view2(&v, 3), view2u(&f), None, 16, None, 1985, None);
        let mut mask = vec![false; f.len() / 3];
        for i in [0, 5, 13, 19, 23] {
            mask[i] = true;
        }
        let some = openness(view2(&v, 3), view2u(&f), Some(&mask), 16, None, 1985, None);
        let expect: Vec<f64> = [0, 5, 13, 19, 23].iter().map(|&i| full[i]).collect();
        assert_eq!(some, expect);
    }

    #[test]
    fn inside_out_components_are_dropped() {
        let (v, f) = box_mesh([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let mut both = f.clone();
        let (iv, iff) = box_mesh([5.0, 5.0, 5.0], [6.0, 6.0, 6.0]);
        let base = (v.len() / 3) as u32;
        let mut v2 = v.clone();
        v2.extend_from_slice(&iv);
        both.extend(flip(&iff).iter().map(|i| i + base));

        let mask = enclosing_mask(&v2, &both).keep;
        assert!(mask[..12].iter().all(|&x| x), "the real box should survive");
        assert!(
            mask[12..].iter().all(|&x| !x),
            "the inside-out one should not"
        );
    }

    #[test]
    fn small_scraps_are_dropped() {
        let (v, f) = box_mesh([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        // One triangle on its own encloses nothing and is below `min_faces` besides.
        let mut f2 = f.clone();
        f2.extend_from_slice(&[0, 1, 2]);
        let mut v2 = v.clone();
        v2.extend_from_slice(&[0.0, 0.0, 0.0]);
        let mask = enclosing_mask(&v2, &f2).keep;
        assert!(!mask[12]);
    }

    /// The whole pipeline on the nested pair: the inner box is invisible from outside, so it
    /// should go, and the outer one should come back untouched.
    #[test]
    fn the_floating_organelle_goes_and_the_cell_stays() {
        let (v, f) = nested();
        let (nv, nf, keep, passes) = repair(&v, &f, 2, 3);
        assert_eq!(nf.nrows(), 12, "only the outer box should be left");
        assert_eq!(
            passes[0].dropped, 0,
            "the organelle is buried and cut, not dropped for enclosing nothing"
        );
        assert_eq!(nv.nrows(), 8);
        // The outer box's vertices came first and are untouched.
        assert_eq!(keep.to_vec(), (0..8).collect::<Vec<u32>>());
        assert_eq!(passes[0].buried, 12);
        // Nothing was opened, so nothing needed capping.
        assert_eq!(passes[0].capped, 0);
    }

    /// A clean mesh must come back as it went in — the property that would let this be turned
    /// on by default.
    #[test]
    fn a_clean_mesh_is_left_alone() {
        let (v, f) = box_mesh([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let (nv, nf, keep, _) = repair(&v, &f, 2, 3);
        assert_eq!(nf.nrows(), 12);
        assert_eq!(nv.nrows(), 8);
        assert_eq!(keep.len(), 8);
    }

    /// `keep` has to index the *input* vertices and reproduce the output exactly, since that
    /// is what callers carry per-vertex data across on.
    #[test]
    fn keep_indexes_the_input() {
        let (v, f) = nested();
        let (nv, _, keep, _) = repair(&v, &f, 2, 3);
        for (row, &k) in keep.iter().enumerate() {
            for c in 0..3 {
                assert_eq!(nv[[row, c]], v[3 * k as usize + c]);
            }
        }
    }

    #[test]
    fn an_empty_mesh_survives() {
        let v: Array2<f64> = Array2::zeros((0, 3));
        let f: Array2<u32> = Array2::zeros((0, 3));
        let (nv, nf, keep, passes) =
            drop_internals(v.view(), f.view(), 0.05, 16, 2, 3, Some(20), 1985, None);
        assert_eq!(nv.nrows(), 0);
        assert_eq!(nf.nrows(), 0);
        assert_eq!(keep.len(), 0);
        assert!(passes.is_empty());
    }

    /// Cutting a hole and capping it: the pipeline has to close what it opens.
    #[test]
    fn openings_are_capped() {
        // A box with a pocket pushed into one face would be the real case; a cheap stand-in is
        // a box whose lid has been removed, which `cap_holes` should close.
        let (v, f) = box_mesh([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let open: Vec<u32> = f[6..].to_vec(); // drop the two z = 0 triangles
        let caps = cap_holes(view2(&v, 3), &open);
        assert_eq!(caps.len() / 3, 2, "a square opening takes two triangles");
        let closed: Vec<u32> = open.iter().chain(caps.iter()).copied().collect();
        assert!(boundary_halfedges(view2u(&closed), None).is_empty());
    }

    /// The diffusion has to preserve a constant field and pull a spike towards its
    /// neighbourhood without overshooting it.
    #[test]
    fn smoothing_is_an_average() {
        let (_, f) = box_mesh([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let graph = FaceGraph::build(&f, 12);
        let flat = vec![0.7; 12];
        for (a, b) in graph.smooth(&flat, 5).iter().zip(&flat) {
            assert!((a - b).abs() < 1e-12);
        }
        let mut spike = vec![0.0; 12];
        spike[0] = 1.0;
        let out = graph.smooth(&spike, 1);
        assert!(out[0] < 1.0 && out[0] > 0.4);
        assert!(out.iter().skip(1).any(|&x| x > 0.0));
        assert!(out.iter().all(|&x| (0.0..=1.0).contains(&x)));
    }

    #[test]
    fn dilation_grows_by_one_ring_per_hop() {
        let (_, f) = box_mesh([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let graph = FaceGraph::build(&f, 12);
        let mut mask = vec![false; 12];
        mask[0] = true;
        graph.dilate(&mut mask, 1);
        let after_one = mask.iter().filter(|&&x| x).count();
        assert!(after_one > 1 && after_one < 12);
        graph.dilate(&mut mask, 10);
        assert!(mask.iter().all(|&x| x), "the box is one surface");
    }

    /// Two faces meeting only at a corner are not adjacent; two sharing an edge are.
    #[test]
    fn adjacency_is_across_edges() {
        let f = vec![0, 1, 2, 0, 2, 3, 4, 5, 6];
        let graph = FaceGraph::build(&f, 3);
        assert_eq!(graph.neighbours_of(0), &[1]);
        assert_eq!(graph.neighbours_of(1), &[0]);
        assert!(graph.neighbours_of(2).is_empty());
    }

    /// A ray spray must not depend on how the work was split, which is what makes a re-cast
    /// agree with a full sweep — and is why the sampler is hashed rather than streamed.
    ///
    /// Note this cannot be checked by varying the seed and watching the *result*: on a mesh
    /// this clean every face is saturated at 0 or 1, so a different spray gives the same
    /// fractions. That the seed reaches the rays at all is the test below.
    #[test]
    fn the_spray_does_not_depend_on_the_split() {
        let (v, f) = nested();
        let a = openness(view2(&v, 3), view2u(&f), None, 8, None, 7, None);
        let b = openness(view2(&v, 3), view2u(&f), None, 8, None, 7, Some(1));
        assert_eq!(a, b);
    }

    #[test]
    fn the_sampler_separates_face_ray_and_seed() {
        assert_ne!(uniforms(7, 0, 0), uniforms(8, 0, 0));
        assert_ne!(uniforms(7, 0, 0), uniforms(7, 1, 0));
        assert_ne!(uniforms(7, 0, 0), uniforms(7, 0, 1));
        // The pair has to be two independent draws, not one value twice.
        let (a, b) = uniforms(7, 3, 2);
        assert_ne!(a, b);
        for f in 0..64u32 {
            for r in 0..16u32 {
                let (x, y) = uniforms(1985, f, r);
                assert!((0.0..1.0).contains(&x) && (0.0..1.0).contains(&y));
            }
        }
    }

    #[test]
    fn a_closed_component_reads_the_same_wherever_it_sits() {
        // On a closed surface the anchor cancels, which is what the divergence theorem
        // promises — so a box that does not contain the origin reads positive either way.
        // The open case, which the closing takes care of, is the test below.
        let (v, f) = box_mesh([100.0, 100.0, 100.0], [101.0, 101.0, 101.0]);
        assert!(enclosing_mask(&v, &f).keep.iter().all(|&x| x));
        let flipped = flip(&f);
        assert!(enclosing_mask(&v, &flipped).keep.iter().all(|&x| !x));
    }

    /// The components `enclosing_mask` is handed are the ones the cut has just opened, and an
    /// open surface's tetrahedra sum to the volume of the cone to whatever point they were
    /// taken on. Read on the origin that cone is as big as the mesh is far away — and signed
    /// by which way the opening happens to face, so it is as likely to swallow the real volume
    /// as to inflate it. Closing the component first takes that term out altogether, so
    /// neither where the mesh sits nor which way its holes point may change the answer.
    #[test]
    fn an_open_component_survives_being_far_from_the_origin() {
        for corner in [[0.0, 0.0, 0.0], [1e6, 1e6, 1e6]] {
            let hi = [corner[0] + 1.0, corner[1] + 1.0, corner[2] + 1.0];
            let (v, f) = box_mesh(corner, hi);
            // Take a lid off, which is what a cut leaves behind: still obviously a cell, and
            // open until the capping step closes it. Both lids, because the cone term changes
            // sign with the opening and only one of the two ever failed loudly.
            for lid in [0, 2] {
                let mut open = f.clone();
                open.drain(3 * lid..3 * lid + 6); // the two triangles of one face
                assert!(
                    enclosing_mask(&v, &open).keep.iter().all(|&x| x),
                    "box at {corner:?} with lid {lid} off should be kept"
                );
                assert!(
                    enclosing_mask(&v, &flip(&open)).keep.iter().all(|&x| !x),
                    "and inside out it should not"
                );
            }
        }
    }

    /// A component the capping cannot close is kept rather than judged.
    ///
    /// [`cap_holes`] may add no vertices, so there are rings it cannot close: here the
    /// boundary pinches at a vertex and the pieces it splits into are too short to bound a
    /// face, so nothing is capped at all. What is left on that still-open surface is the
    /// cone to its anchor — the reading this whole function exists to distrust — and it is
    /// negative, so a measurement taken anyway would throw the component away. Keeping it is
    /// the only honest answer; [`Pass::unclosed`] is how the caller hears that it happened.
    #[test]
    fn a_component_that_will_not_close_is_kept_rather_than_judged() {
        #[rustfmt::skip]
        let v = vec![
            0.0, 0.0, 0.0,
            1.0, 0.0, 0.0,
            0.5, 1.0, 0.5,
            1.5, 1.0, 0.0,
            0.0, 1.0, 0.0,
        ];
        let f = vec![0, 2, 1, 0, 4, 2, 1, 3, 2, 2, 4, 3];

        assert!(
            cap_holes(view2(&v, 3), &f).is_empty(),
            "the premise: this is a boundary the capping cannot do anything with"
        );
        assert!(
            !boundary_halfedges(view2u(&f), None).is_empty(),
            "and it is a boundary"
        );
        assert!(
            cone_volume(&v, &f) < 0.0,
            "the premise: measured anyway, it reads as enclosing nothing"
        );

        let e = enclosing_mask(&v, &f);
        assert!(
            e.keep.iter().all(|&k| k),
            "so it must be kept, not measured"
        );
        assert_eq!(e.unclosed, 4, "and reported as unjudged");
    }

    /// A mouth the cut opens must not get a vote on whether the cell is a cell.
    ///
    /// A box cannot show this. Cone a convex shell's holes to a point inside it and the cones
    /// stay inside, so all they can do is eat the volume down towards zero. A membrane is the
    /// opposite shape — thin, spread out, and bent back on itself — so its cones reach across
    /// the whole cell, lie on top of one another, and carry the sum straight past zero. A
    /// loop of tube is the smallest thing with that shape. On a real neuron it took mouths
    /// worth 0.6% of the faces to do it.
    #[test]
    fn a_mouth_cannot_outvote_the_cell_it_opens() {
        let stations = 48;
        let (v, f) = tube_loop(stations, 40.0, 1.0);
        assert!(
            enclosing_mask(&v, &f).keep.iter().all(|&x| x),
            "closed, and the right way out to begin with"
        );

        // Bite the outer wall off a quarter of the stations — the faces furthest from the
        // centre, which are the ones carrying the sum. A station is eight faces, its outer
        // wall the first two.
        let mut open = f.clone();
        for k in (0..stations / 4).rev() {
            open.drain(24 * k..24 * k + 6);
        }
        assert!(
            cone_volume(&v, &open) < 0.0,
            "the open sum should read this cell as inside out, which is the whole problem"
        );
        assert!(
            enclosing_mask(&v, &open).keep.iter().all(|&x| x),
            "and closing it first should settle that it is not"
        );
        // Inside out it still has to read inside out, or the test proves nothing.
        assert!(enclosing_mask(&v, &flip(&open)).keep.iter().all(|&x| !x));
    }

    /// Loose scraps must not vote on whether the volume reading is believed.
    ///
    /// The backstop asks how much of what the reading *judged* it condemned. Count the
    /// components below the floor on either side of that and a drift of stray triangles
    /// decides the question for surface they have nothing to do with: here the reading
    /// condemns two dishes out of twenty judged faces — nowhere near half — and the twenty
    /// loose triangles must not turn that into a reason to keep the dishes.
    #[test]
    fn scraps_do_not_trip_the_backstop() {
        let (mut v, mut f) = nested();
        for k in 0..2 {
            let (dv, df) = inside_out_dish(10.0 * (k + 1) as f64);
            let base = (v.len() / 3) as u32;
            v.extend_from_slice(&dv);
            f.extend(df.iter().map(|i| i + base));
        }
        for k in 0..20u32 {
            let base = (v.len() / 3) as u32;
            let x = 100.0 * (k + 1) as f64;
            v.extend_from_slice(&[x, 0.0, 0.0, x + 1.0, 0.0, 0.0, x, 1.0, 0.0]);
            f.extend_from_slice(&[base, base + 1, base + 2]);
        }

        let (_, nf, _, passes) = repair(&v, &f, 2, 3);
        assert_eq!(
            passes[0].spared, 0,
            "the backstop should stay out of it: {passes:?}"
        );
        assert_eq!(
            passes[0].dropped, 28,
            "two dishes and twenty triangles, all of them: {passes:?}"
        );
        assert_eq!(nf.nrows(), 12, "leaving the cell: {passes:?}");
    }

    /// The enclosing filter may not eat the mesh. Asked to, it is held back for the pass and
    /// says so, because a cell is not mostly cavity and the ray cast gets another go.
    #[test]
    fn the_enclosing_filter_is_overruled_when_it_asks_for_the_mesh() {
        // A cell with an organelle in it, so that something is actually buried: a pass with
        // nothing to cut stops before the enclosing filter is ever reached.
        let (mut v, mut f) = nested();
        // Four dishes of four faces each, against the twelve the cut leaves, so the filter
        // wants more than half of what it is looking at.
        for k in 0..4 {
            let (dv, df) = inside_out_dish(10.0 * (k + 1) as f64);
            let base = (v.len() / 3) as u32;
            v.extend_from_slice(&dv);
            f.extend(df.iter().map(|i| i + base));
        }
        // And some loose triangles, which the filter wants for a different reason: one face
        // bounds nothing by arithmetic, not by measurement, so the guard has no say in it.
        for k in 0..8u32 {
            let base = (v.len() / 3) as u32;
            let x = 100.0 * (k + 1) as f64;
            v.extend_from_slice(&[x, 0.0, 0.0, x + 1.0, 0.0, 0.0, x, 1.0, 0.0]);
            f.extend_from_slice(&[base, base + 1, base + 2]);
        }

        let (_, nf, _, passes) = repair(&v, &f, 2, 3);
        assert_eq!(
            passes[0].spared, 16,
            "the filter asked for most of the mesh, so the four dishes stay: {passes:?}"
        );
        assert_eq!(
            passes[0].dropped, 8,
            "but the loose triangles go regardless: {passes:?}"
        );
        // And the cell survives, which is the point of holding the filter back. The scraps go
        // in the end anyway: capping closes them, and a closed inside-out shell is exactly
        // what the ray cast is for.
        assert_eq!(nf.nrows(), 12, "{passes:?}");
        assert!(
            passes[1..].iter().all(|p| p.spared == 0),
            "and once they are gone the filter should be trusted again: {passes:?}"
        );
    }

    /// The same, end to end — and here the pipeline opens the component itself, which is the
    /// real shape of the bug and the one that cost a whole neuron. See [`enclosing_mask`] for
    /// what that cost.
    #[test]
    fn the_pipeline_does_not_care_where_the_mesh_sits() {
        let (v, f) = pocket();
        let run = |dz: f64| {
            let moved: Vec<f64> = v
                .iter()
                .enumerate()
                .map(|(i, x)| if i % 3 == 2 { x + dz } else { *x })
                .collect();
            let (_, nf, keep, passes) = repair(&moved, &f, 2, 3);
            (nf, keep, passes)
        };

        // Properties, not counts: the pocket is gone, what is left is closed, and the passes
        // ran out of things to bury rather than out of iterations.
        let (here, keep, passes) = run(0.0);
        assert!(
            here.nrows() < f.len() / 3,
            "the pocket should have been cut away"
        );
        assert!(
            boundary_halfedges(here.view(), None).is_empty(),
            "and the mouth it left closed"
        );
        assert_eq!(
            passes.last().expect("at least one pass").buried,
            0,
            "and the repair should have converged"
        );

        for dz in [1e6, -1e6] {
            let (nf, k, p) = run(dz);
            assert_eq!(nf, here, "moved {dz} along z");
            assert_eq!(k, keep, "moved {dz} along z");
            assert_eq!(p, passes, "moved {dz} along z");
        }
    }

    /// Smoothing may move where the cut starts; it must not decide how far the cut gets.
    ///
    /// Diffusion averages a narrow pocket against the wide-open lid beside it, so with the
    /// contour alone the threshold reaches only the tube's blind floor. Cutting that leaves a
    /// cap at the bottom of the tube, as buried as the floor it replaced, for the next pass to
    /// cut and lay again — no progress, for every pass of the budget, and the tube still
    /// there at the end. The extent flood carries the cut out through the faces that saw
    /// nothing, so the pocket goes whatever `smooth` is set to.
    ///
    /// Up to a point: `smooth = 4` on a mesh this small lifts every face clear of the
    /// threshold, and a flood with no seed reaches nothing. That is the contour failing, not
    /// the extent, and it fails the safe way round — nothing is cut at all.
    #[test]
    fn smoothing_does_not_decide_how_far_the_cut_reaches() {
        let (v, f) = pocket();
        for smooth in 0..=3 {
            let (_, nf, _, passes) = repair(&v, &f, smooth, 6);
            // The tube is gone, not merely re-floored: capping the mouth costs fewer faces
            // than the eight of tube wall the cut takes with it.
            assert!(
                nf.nrows() < f.len() / 3,
                "smooth = {smooth} left the pocket in place: {passes:?}"
            );
            assert!(
                boundary_halfedges(nf.view(), None).is_empty(),
                "smooth = {smooth} left the mouth open"
            );
            // Converged, rather than stopped by the budget — which is what a stall looks like.
            assert_eq!(
                passes.last().expect("at least one pass").buried,
                0,
                "smooth = {smooth} did not converge: {passes:?}"
            );
            assert!(passes.len() < 6, "smooth = {smooth} spent the whole budget");
        }
    }

    #[test]
    fn faces_must_be_triangles() {
        let v = array![[0.0, 0.0, 0.0]];
        let f: Array2<u32> = Array2::zeros((1, 4));
        let r = std::panic::catch_unwind(|| {
            drop_internals(v.view(), f.view(), 0.05, 16, 2, 1, None, 1985, None)
        });
        assert!(r.is_err());
    }
}
