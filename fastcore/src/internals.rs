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
//!    with 54,112 boundary vertices to 513 with 12,094 — the same faces removed, a far better
//!    cut.
//! 3. **Cut**: drop the buried faces, then drop the components that enclose no volume — the
//!    inside-out shreds of pocket wall left behind.
//! 4. **Cap** the resulting holes, which are the pocket mouths, with [`crate::caps`].
//!
//! Then repeat: capping a mouth turns a partially-open neighbour into a fully buried one, so
//! later passes find pockets the first threshold could not see. It converges fast — 18.7% of
//! faces buried, then 0.5%, then 0.2%.
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

/// Unit normal and centroid of one face; the normal is `None` for a degenerate triangle.
#[inline]
fn face_frame(v: &[f64], f: &[u32]) -> (Option<[f64; 3]>, [f64; 3]) {
    let (a, b, c) = (point(v, f[0]), point(v, f[1]), point(v, f[2]));
    (
        normalize(cross(sub(b, a), sub(c, a))),
        [
            (a[0] + b[0] + c[0]) / 3.0,
            (a[1] + b[1] + c[1]) / 3.0,
            (a[2] + b[2] + c[2]) / 3.0,
        ],
    )
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

/// Which faces sit in a component that encloses positive volume.
///
/// Deliberately *not* "keep the largest component": neuron meshes routinely arrive in many
/// legitimate pieces — one hemibrain example mesh has 299 — and keeping only the biggest
/// silently deletes real neurite. Only inside-out or degenerate scraps are dropped.
///
/// This is also the one place in the pipeline where signed volume can be trusted: it is read
/// per component *before* capping, when each sheet is still closed. Once the mesh has been cut
/// and capped it is open and partly non-manifold, and its divergence-theorem volume means
/// nothing.
fn enclosing_mask(v: &[f64], faces: &[u32]) -> Vec<bool> {
    let n = faces.len() / 3;
    let n_faces_arr = ArrayView2::from_shape((n, 3), faces).expect("(F, 3)");
    let labels = mesh_face_components(n_faces_arr, true, None);

    // Labels are the smallest face index in each component, so they index arrays of length F
    // directly — no compaction needed.
    let mut volume = vec![0.0f64; n];
    let mut count = vec![0u32; n];
    for (i, l) in labels.iter().enumerate() {
        let f = &faces[3 * i..3 * i + 3];
        let (a, b, c) = (point(v, f[0]), point(v, f[1]), point(v, f[2]));
        // The signed volume of the tetrahedron on the origin. Summed over a closed surface
        // the origin cancels, which is what makes this a property of the component and not of
        // where it happens to sit.
        volume[*l as usize] += dot(a, cross(b, c)) / 6.0;
        count[*l as usize] += 1;
    }

    labels
        .iter()
        .map(|&l| volume[l as usize] > 0.0 && count[l as usize] >= MIN_COMPONENT_FACES)
        .collect()
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
    /// Faces whose smoothed openness fell below the threshold.
    pub buried: usize,
    /// Faces re-cast this pass — the whole mesh on the first, the collar around the last
    /// pass's caps afterwards.
    pub recast: usize,
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
/// Arguments
/// ---------
/// - `vertices`:  (V, 3) vertex positions.
/// - `faces`:     (F, 3) triangular faces as vertex indices.
/// - `threshold`: Faces whose smoothed openness falls below this are cut. Do not raise it far:
///   the buried mode sits at exactly zero, so any small value works, while above ~0.1 the cut
///   starts eating real membrane. The rule that says so is the boundary-edge count, which is
///   flat from 0.02 to 0.10 and then explodes — that is the cut outrunning the capping.
/// - `n_rays`:    Rays per face; see [`openness`].
/// - `smooth`:    Diffusion rounds applied to the field before thresholding.
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

            let recast = match (cap_start, hops) {
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
                    recast
                }
                _ => {
                    field = openness_with(&bvh, &v, &f, None, n_rays, None, seed);
                    n
                }
            };

            let smoothed = graph.smooth(&field, smooth);
            let buried: Vec<bool> = smoothed.iter().map(|&x| x < threshold).collect();
            let n_buried = buried.iter().filter(|&&b| b).count();
            if n_buried == 0 {
                log.push(Pass {
                    faces_before: n,
                    buried: 0,
                    recast,
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
            let enclosing = enclosing_mask(&v, &cut);

            let mut kept_faces = Vec::with_capacity(cut.len());
            let mut kept_field = Vec::with_capacity(survivors.len());
            for (row, &i) in survivors.iter().enumerate() {
                if enclosing[row] {
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

        let mask = enclosing_mask(&v2, &both);
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
        let mask = enclosing_mask(&v2, &f2);
        assert!(!mask[12]);
    }

    /// The whole pipeline on the nested pair: the inner box is invisible from outside, so it
    /// should go, and the outer one should come back untouched.
    #[test]
    fn the_floating_organelle_goes_and_the_cell_stays() {
        let (v, f) = nested();
        let (nv, nf, keep, passes) = drop_internals(
            view2(&v, 3),
            view2u(&f),
            0.05,
            16,
            2,
            3,
            Some(20),
            1985,
            None,
        );
        assert_eq!(nf.nrows(), 12, "only the outer box should be left");
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
        let (nv, nf, keep, _) = drop_internals(
            view2(&v, 3),
            view2u(&f),
            0.05,
            16,
            2,
            3,
            Some(20),
            1985,
            None,
        );
        assert_eq!(nf.nrows(), 12);
        assert_eq!(nv.nrows(), 8);
        assert_eq!(keep.len(), 8);
    }

    /// `keep` has to index the *input* vertices and reproduce the output exactly, since that
    /// is what callers carry per-vertex data across on.
    #[test]
    fn keep_indexes_the_input() {
        let (v, f) = nested();
        let (nv, _, keep, _) = drop_internals(
            view2(&v, 3),
            view2u(&f),
            0.05,
            16,
            2,
            3,
            Some(20),
            1985,
            None,
        );
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
    fn signed_volume_sees_through_the_origin() {
        // A box that does not contain the origin still has positive signed volume, which is
        // what the divergence theorem promises and what `enclosing_mask` leans on.
        let (v, f) = box_mesh([100.0, 100.0, 100.0], [101.0, 101.0, 101.0]);
        assert!(enclosing_mask(&v, &f).iter().all(|&x| x));
        let flipped = flip(&f);
        assert!(enclosing_mask(&v, &flipped).iter().all(|&x| !x));
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
