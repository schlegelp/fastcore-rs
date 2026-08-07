//! Smoothing a triangle mesh: moving its vertices to take the noise out of a surface
//! without changing how many there are or which faces they form.
//!
//! Three filters, all built on the same one-ring average:
//!
//! - [`Filter::Laplacian`] — the plain diffusion step. Simple, effective, and it
//!   *shrinks*: on a neuron mesh at the settings navis ships (`lambda = 0.5`, five
//!   iterations) it costs 88% of the enclosed volume.
//! - [`Filter::Taubin`] — alternating shrink and inflate passes, tuned so the two
//!   cancel below a cut-off frequency. The standard answer to the shrinkage, and the
//!   default here.
//! - [`Filter::Humphrey`] — the HC filter of Vollmer et al., which pulls each vertex
//!   back towards where it started rather than towards a lower frequency.
//!
//! and three weightings of that average — [`Weights::Uniform`],
//! [`Weights::InverseDistance`] and [`Weights::Cotangent`].
//!
//! # What this is a replacement for
//!
//! `trimesh.smoothing.filter_laplacian`, which is what navis' `smooth_mesh` calls, and
//! whose volume constraint is worth describing because this module deliberately does not
//! reproduce it. Upstream rescales the mesh after each iteration by
//! `(vol_before / vol_after).cbrt()` — *about the origin*. That is not a shape operation:
//!
//! - It translates the mesh. On the 722817260 test neuron, at navis' own defaults, the
//!   constraint displaces the result by 41 um. The mesh is 19-26 um across.
//! - It is not translation invariant, so the same mesh smoothed at two different offsets
//!   comes out two different shapes — and far enough from the origin the volume ratio goes
//!   negative and the cube root returns NaN.
//! - It divides by the smoothed volume, so a mesh with a hole big enough to make that zero
//!   is a division by zero rather than a diagnostic.
//!
//! [`smooth_mesh`] scales about the mesh's own centroid instead, which is the same
//! operation without any of the above, and reports what it did in [`Smoothed::volume`]
//! rather than silently producing a NaN. It also does it *once, at the end*, which is not
//! an approximation of doing it every iteration but exactly equal to it — see
//! [`Smoothed::volume`] for why.
//!
//! # Where the time goes
//!
//! Not in the arithmetic. Ten iterations over a 421k-vertex mesh is ~7.6M multiply-adds
//! per pass, which is nothing; upstream spends 57% of its five seconds building the
//! operator (`vertex_neighbors` is a list of 421k Python lists, 636 MB of heap for 10 MB
//! of vertices) and another 40% in the volume constraint's `vertices[faces]` gather, which
//! materialises a 63 MB temporary per iteration.
//!
//! So the design here is about touching memory once: one CSR adjacency built from the
//! faces, and weights that are *never materialised at all*. [`Umbrella::average`] derives
//! each vertex's weights from the geometry as it walks its one-ring, which costs the same
//! handful of flops as reading them back and means the geometry-dependent weightings are
//! recomputed every iteration — the correct flow rather than a first-iteration snapshot —
//! for free.

use ndarray::{Array2, ArrayView2};
use rayon::prelude::*;

use crate::threads::with_pool;

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------

/// How a vertex's one-ring is weighted when it is averaged.
///
/// All three are normalised to sum to one per vertex, so every filter below is an
/// interpolation between a vertex and a point in the convex hull of its neighbours, and
/// none of them can change the scale of the mesh on their own.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Weights {
    /// Every neighbour counts the same. What `trimesh` does by default, and what to use
    /// when the mesh is evenly tessellated or when you want the smoothing to also
    /// regularise the *sampling* — the uniform umbrella pulls vertices towards even
    /// spacing as well as towards the local plane.
    Uniform,
    /// Neighbours weighted by `1 / distance`. Cheap partial compensation for uneven
    /// tessellation: a vertex with one very close neighbour and one distant one is pulled
    /// mostly towards the close one.
    InverseDistance,
    /// `cot(alpha) + cot(beta)`, the two angles opposite the edge in the faces sharing it.
    ///
    /// The discretisation of the Laplace-Beltrami operator, and the only one of the three
    /// that approximates mean-curvature flow: it depends on the shape of the surface
    /// rather than on how it happens to be triangulated, so it moves vertices along the
    /// normal and leaves them where they are within the surface. That is what you want on
    /// meshes out of EM segmentation, whose triangles vary wildly in size and aspect.
    ///
    /// Cotangents go negative on obtuse triangles, which makes the explicit step unstable
    /// — a negative weight pushes a vertex *away* from its neighbour and the iteration
    /// diverges. Negative contributions are therefore clamped to zero, the usual remedy.
    /// A vertex whose weights all vanish that way falls back to [`Weights::Uniform`].
    Cotangent,
}

/// Which smoothing filter to run.
///
/// The parameters are checked by [`smooth_mesh`], which panics on values outside the
/// ranges given here rather than quietly diverging several iterations later.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Filter {
    /// `x <- x + lambda * (avg(x) - x)`, once per iteration.
    ///
    /// `lambda` in `[0, 1]`; above 1 the explicit step overshoots and the mesh oscillates
    /// apart. Removes high frequencies fast and low ones slowly, and since a closed
    /// surface's volume *is* a low frequency, it removes that too: this is the filter that
    /// shrinks. Reach for it when the mesh is a means to an end (a visualisation, a
    /// distance field) rather than when its enclosed volume means something, or pair it
    /// with `volume_correction`.
    Laplacian {
        /// Diffusion speed. `0` is a no-op.
        lambda: f64,
    },
    /// Taubin's `lambda | mu` band-pass: a `lambda` step followed by a `mu` step, where
    /// `mu` is negative, so each iteration shrinks and then re-inflates.
    ///
    /// The two cancel exactly at the pass-band frequency `1/lambda + 1/mu` and reinforce
    /// above it, which is how this removes noise without removing the shape. Requires
    /// `0 < lambda <= 1` and `-1 <= mu < -lambda`; the classic pair is `0.5 | -0.53`.
    ///
    /// **One iteration is one full `lambda`-then-`mu` pair**, i.e. two passes over the
    /// mesh — not one, as `trimesh.smoothing.filter_taubin` counts them. Counting
    /// half-steps lets an odd `iterations` end on a `lambda` pass, which is a shrink that
    /// nothing undoes; the whole point of the filter is that the passes come in pairs.
    Taubin {
        /// Shrinking pass. Same role as [`Filter::Laplacian`]'s.
        lambda: f64,
        /// Inflating pass, negative and slightly larger in magnitude than `lambda`.
        mu: f64,
    },
    /// The HC (Humphrey's Classes) filter of Vollmer, Mencl and Muller: take a full
    /// Laplacian step, then push back along the displacement from the *original* position.
    ///
    /// Where Taubin fights shrinkage in the frequency domain, this fights it in the
    /// spatial one, which makes it the gentler of the two on a mesh with fine detail worth
    /// keeping. Both `alpha` and `beta` are in `[0, 1]`.
    Humphrey {
        /// How hard vertices are pulled back towards their original positions. `0`
        /// ignores them entirely, leaving plain Laplacian smoothing.
        alpha: f64,
        /// How much of the push-back is applied at the vertex rather than spread over its
        /// one-ring. `1` is the most aggressive.
        beta: f64,
    },
}

impl Weights {
    /// The names [`Weights::from_name`] accepts, in declaration order.
    pub const NAMES: [&'static str; 3] = ["uniform", "inverse_distance", "cotangent"];

    /// Look a weighting up by the name a binding's caller spelled.
    ///
    /// Here rather than in each binding for the same reason
    /// [`crate::linkage::Method::from_name`] is: the set of names is a property of the
    /// enum, and two hand-maintained copies of it — one per binding surface, in two
    /// languages and two build systems — drift silently.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "uniform" => Some(Self::Uniform),
            "inverse_distance" => Some(Self::InverseDistance),
            "cotangent" => Some(Self::Cotangent),
            _ => None,
        }
    }
}

impl Filter {
    /// The names [`Filter::from_parts`] accepts, in the order they are documented.
    pub const METHODS: [&'static str; 3] = ["taubin", "laplacian", "humphrey"];

    /// Which of `lambda`, `mu`, `alpha`, `beta` a given method reads.
    ///
    /// `None` for a method that does not exist. Exposed because a binding wants to reject
    /// a parameter belonging to *another* method — passing `alpha` to Taubin has asked for
    /// something, and quietly dropping it is the one outcome that looks like success — and
    /// because that table drifting between surfaces is exactly what this being one table
    /// prevents.
    pub fn params_of(method: &str) -> Option<&'static [&'static str]> {
        match method {
            "laplacian" => Some(&["lambda"]),
            "taubin" => Some(&["lambda", "mu"]),
            "humphrey" => Some(&["alpha", "beta"]),
            _ => None,
        }
    }

    /// Build a filter from a method name and whichever parameters the caller supplied,
    /// filling in that method's defaults for the rest.
    ///
    /// `None` for an unknown method; the caller is expected to have checked
    /// [`Filter::params_of`] first if it cares about strays. The *defaults* live here
    /// rather than in each binding because two copies of `-0.53` in two languages can
    /// disagree without any test noticing.
    pub fn from_parts(
        method: &str,
        lambda: Option<f64>,
        mu: Option<f64>,
        alpha: Option<f64>,
        beta: Option<f64>,
    ) -> Option<Self> {
        match method {
            "laplacian" => Some(Self::Laplacian {
                lambda: lambda.unwrap_or(0.5),
            }),
            "taubin" => Some(Self::Taubin {
                lambda: lambda.unwrap_or(0.5),
                mu: mu.unwrap_or(-0.53),
            }),
            "humphrey" => Some(Self::Humphrey {
                alpha: alpha.unwrap_or(0.1),
                beta: beta.unwrap_or(0.5),
            }),
            _ => None,
        }
    }

    /// Check the parameters, describing the first bad one rather than panicking on it.
    ///
    /// A `Result` because a binding needs the failure as an ordinary, catchable error of
    /// its host language — a Rust panic reaches Python as a `PanicException` a caller can
    /// neither catch by type nor learn from, and reaches R as "User function panicked"
    /// with the message dropped entirely. [`Filter::validate`] is the panicking wrapper
    /// for callers who want the assert.
    ///
    /// `lambda_name` is what the calling surface spells λ: `"lambda"` in Rust and R,
    /// `"lamb"` in Python, where `lambda` is a keyword. The parameter exists so that the
    /// ranges themselves are written once — the alternative was a second copy of the
    /// arithmetic in the Python binding purely to rename one word in the message.
    pub fn check(&self, lambda_name: &str) -> Result<(), String> {
        let finite = |what: &str, v: f64| {
            if v.is_finite() {
                Ok(())
            } else {
                Err(format!("`{what}` must be finite, got {v}"))
            }
        };
        match *self {
            Self::Laplacian { lambda } => {
                finite(lambda_name, lambda)?;
                if !(0.0..=1.0).contains(&lambda) {
                    return Err(format!(
                        "`{lambda_name}` must be in [0, 1], got {lambda}; above 1 the \
                         explicit step overshoots and the mesh oscillates apart"
                    ));
                }
            }
            Self::Taubin { lambda, mu } => {
                finite(lambda_name, lambda)?;
                finite("mu", mu)?;
                if !(0.0 < lambda && lambda <= 1.0) {
                    return Err(format!("`{lambda_name}` must be in (0, 1], got {lambda}"));
                }
                if mu >= -lambda {
                    return Err(format!(
                        "`mu` must be negative and larger in magnitude than \
                         `{lambda_name}` ({lambda}), got {mu}; otherwise the pass-band \
                         frequency 1/lambda + 1/mu is not positive and the filter is a \
                         plain shrink-then-shrink"
                    ));
                }
                if mu < -1.0 {
                    return Err(format!(
                        "`mu` must be in [-1, 0), got {mu}; below -1 the inflating step \
                         overshoots"
                    ));
                }
            }
            Self::Humphrey { alpha, beta } => {
                for (name, v) in [("alpha", alpha), ("beta", beta)] {
                    finite(name, v)?;
                    if !(0.0..=1.0).contains(&v) {
                        return Err(format!("`{name}` must be in [0, 1], got {v}"));
                    }
                }
            }
        }
        Ok(())
    }

    /// Panic with a description of the offending parameter, or return.
    ///
    /// What [`smooth_mesh`] calls, matching the rest of the crate: shape and range errors
    /// from a Rust caller are asserts here, as they are in [`crate::mesh`],
    /// [`crate::simplify`] and [`crate::downsample`]. Bindings call [`Filter::check`].
    fn validate(&self) {
        if let Err(msg) = self.check("lambda") {
            panic!("{msg}");
        }
    }
}

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------

/// What happened to the mesh's enclosed volume.
///
/// Reported rather than assumed, because whether the correction meant anything depends on
/// the mesh. On a closed one it is exactly what it says: the ratio is the ratio of enclosed
/// volumes, and restoring it restores the volume.
///
/// A mesh that is *not* closed still usually gets a correction, and deliberately. Both
/// measurements cone every face back to the same anchor (see [`signed_volume`]), so their
/// ratio remains a consistent measure of how much the surface shrank even where neither
/// number is an enclosed volume on its own. That matters because meshes worth smoothing are
/// almost never watertight — the 722817260 test neuron is not — and refusing on that basis
/// would refuse on nearly every mesh this exists for.
///
/// [`Volume::Undefined`] is the case where even that fails: the ratio is zero, infinite,
/// NaN or negative, so it has no cube root worth taking. A flat sheet is the clean example
/// — both volumes are exactly zero. Those come back with the two measurements that made it
/// undecidable, so a caller can say so instead of shipping a mesh scaled by a meaningless
/// factor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Volume {
    /// No correction was asked for.
    Off,
    /// The mesh was scaled about its centroid by this factor to restore the input volume.
    /// Values below 1 mean the filter had inflated the mesh, above 1 that it had shrunk it.
    Scaled(f64),
    /// A correction was asked for and could not be made: the ratio of the two signed
    /// volumes is not finite and positive. The vertices are returned smoothed but unscaled.
    Undefined {
        /// Signed volume of the input mesh.
        before: f64,
        /// Signed volume of the smoothed mesh, before any scaling.
        after: f64,
    },
}

/// A smoothed mesh.
///
/// The faces are not here because they are not touched: smoothing moves vertices and
/// nothing else, so the caller's face array, and anything else indexed by vertex — synapse
/// attachments, labels, radii — is still valid. Only a *copy* of a vertex position taken
/// beforehand goes stale.
#[derive(Debug, Clone, PartialEq)]
pub struct Smoothed {
    /// `(V, 3)` new vertex positions, in the input's vertex order.
    pub vertices: Array2<f64>,
    /// What the volume correction did; [`Volume::Off`] when none was requested.
    ///
    /// # Why the correction runs once, at the end
    ///
    /// It is not an approximation of correcting every iteration — it is equal to it.
    ///
    /// Every filter here is built out of `avg`, whose weights are normalised to sum to one
    /// per vertex. A row-stochastic average commutes with any affine map: `avg(s*x + t)`
    /// weights the neighbours the same way (the two geometry-dependent weightings are
    /// scale- and translation-invariant — inverse distances are renormalised per row, and
    /// cotangents are functions of angles) and so equals `s*avg(x) + t`. The filters are
    /// affine combinations of `x` and `avg(x)`, hence affine-equivariant too, and so is
    /// any composition of them. Scaling first and smoothing, or smoothing and scaling
    /// afterwards, therefore land on the same vertices.
    ///
    /// Which makes doing it once strictly better: upstream's per-iteration constraint pays
    /// a full pass over the faces and a `(F, 3, 3)` gather *per iteration* — 40% of its
    /// runtime — for a result it could have had at the end for one pass.
    pub volume: Volume,
}

// ---------------------------------------------------------------------------
// The one-ring
// ---------------------------------------------------------------------------

/// Vertex adjacency in CSR layout, plus what the weightings need on top of it.
///
/// Same layout and the same reasons as [`crate::mesh::Adjacency`] — one heap allocation
/// per vertex would burn megabytes of allocator overhead on a large mesh and scatter the
/// one-rings across the heap, and the one-ring scan is the memory-bound inner loop of
/// every filter here. What differs is that there are no weights: see [`Umbrella::average`].
struct Umbrella {
    /// `offsets[v]..offsets[v + 1]` is the slice of `nbrs` holding v's neighbours, sorted,
    /// deduplicated and free of self-loops.
    offsets: Vec<u32>,
    nbrs: Vec<u32>,
    /// Incident faces per vertex, in the same layout. Only built for
    /// [`Weights::Cotangent`], which is the only weighting that needs to know which
    /// triangle a neighbour arrived through — the other two are functions of the one-ring
    /// alone.
    face_offsets: Vec<u32>,
    faces_of: Vec<u32>,
    /// Whether each vertex lies on a boundary, i.e. is an endpoint of an edge used by
    /// exactly one face. Free to compute here: see [`Umbrella::new`].
    border: Vec<bool>,
}

impl Umbrella {
    /// Build the one-ring of every vertex from a flat `3F` face array.
    ///
    /// `with_faces` additionally builds the vertex-to-face lists that
    /// [`Weights::Cotangent`] needs; the other weightings never look at them, so they are
    /// not paid for.
    ///
    /// Non-manifold input is merely data, as in [`crate::simplify`]: an edge shared by
    /// three faces contributes three arcs and dedups to one neighbour, and a degenerate
    /// face `(a, a, b)` contributes a self-loop that is dropped. Neither is an error, and
    /// neither has to be, because nothing below reads more topology than "which vertices
    /// are adjacent to which".
    fn new(faces: &[u32], n_vertices: usize, with_faces: bool) -> Self {
        let n_faces = faces.len() / 3;
        let n_arcs = n_faces.saturating_mul(6);
        assert!(
            n_arcs <= u32::MAX as usize,
            "too many faces: CSR offsets are u32"
        );

        // Count. Each face gives each of its vertices two arcs, and each vertex one
        // incidence.
        let mut offsets = vec![0u32; n_vertices + 1];
        let mut face_offsets = vec![0u32; if with_faces { n_vertices + 1 } else { 0 }];
        for tri in faces.chunks_exact(3) {
            for &v in tri {
                assert!(
                    (v as usize) < n_vertices,
                    "face references vertex {v}, but the mesh has {n_vertices} vertices"
                );
                offsets[v as usize + 1] += 2;
                if with_faces {
                    face_offsets[v as usize + 1] += 1;
                }
            }
        }
        for i in 0..n_vertices {
            offsets[i + 1] += offsets[i];
            if with_faces {
                face_offsets[i + 1] += face_offsets[i];
            }
        }

        // Scatter.
        let mut nbrs = vec![0u32; offsets[n_vertices] as usize];
        let mut cursor: Vec<u32> = offsets[..n_vertices].to_vec();
        let mut faces_of = vec![0u32; if with_faces { n_faces * 3 } else { 0 }];
        let mut fcursor: Vec<u32> = if with_faces {
            face_offsets[..n_vertices].to_vec()
        } else {
            Vec::new()
        };
        for (fi, tri) in faces.chunks_exact(3).enumerate() {
            for k in 0..3 {
                let (v, a, b) = (tri[k], tri[(k + 1) % 3], tri[(k + 2) % 3]);
                let slot = &mut cursor[v as usize];
                nbrs[*slot as usize] = a;
                nbrs[*slot as usize + 1] = b;
                *slot += 2;
                if with_faces {
                    let fslot = &mut fcursor[v as usize];
                    faces_of[*fslot as usize] = fi as u32;
                    *fslot += 1;
                }
            }
        }

        // Compact each row, and pick up the boundary flags on the way through.
        //
        // A neighbour `u` appears in `v`'s row exactly once per face containing the edge
        // `{v, u}` — the scatter above emits `v -> u` from each such face and nowhere else
        // — so the run length of `u` within the sorted row *is* that edge's face count. A
        // run of one is therefore a boundary edge, and knowing that costs a comparison
        // inside a loop that was already going to walk every entry to deduplicate it. The
        // alternative, sorting `3F` packed edge keys, is a second O(F log F) pass over
        // data this one has in registers.
        //
        // [`crate::simplify`] derives the same flag a second way, from a one-ring tally
        // over its own `refs` table (`Simplifier::init_borders`), because it has that
        // table and no CSR and this has the CSR and no table. The two agree on every mesh
        // with no degenerate faces, and can disagree on one that has them: given a face
        // `(a, a, b)`, that tally counts `b` once and calls it a border, while the run
        // length here is two and calls it interior. Neither reading is wrong — a face with
        // two corners at the same point has no well-defined boundary — but if you are
        // comparing `preserve_border` across the two functions on degenerate input, that
        // is where the difference comes from.
        //
        // This loop is sequential, and on a 421k-vertex mesh it is ~70% of the build and
        // ~60% of a default 10-iteration call. The rows are independent, so it can be
        // blocked and run in parallel (measured: 16 ms -> 2.3 ms, ~1.6x end-to-end); that
        // is worth doing as its own change, alongside `mesh::Adjacency::compact`, which
        // has the identical shape and the identical opportunity.
        let old = offsets.clone();
        let mut border = vec![false; n_vertices];
        let mut w = 0usize;
        for v in 0..n_vertices {
            let (lo, hi) = (old[v] as usize, old[v + 1] as usize);
            debug_assert!(w <= lo);
            nbrs[lo..hi].sort_unstable();

            offsets[v] = w as u32;
            let mut k = lo;
            while k < hi {
                let u = nbrs[k];
                let mut run = 1;
                while k + run < hi && nbrs[k + run] == u {
                    run += 1;
                }
                // A self-loop is a degenerate face, not a neighbour, and says nothing
                // about boundaries either way.
                if u as usize != v {
                    if run == 1 {
                        border[v] = true;
                    }
                    nbrs[w] = u;
                    w += 1;
                }
                k += run;
            }
        }
        offsets[n_vertices] = w as u32;
        nbrs.truncate(w);
        nbrs.shrink_to_fit();

        Umbrella {
            offsets,
            nbrs,
            face_offsets,
            faces_of,
            border,
        }
    }

    #[inline]
    fn row(&self, v: usize) -> &[u32] {
        &self.nbrs[self.offsets[v] as usize..self.offsets[v + 1] as usize]
    }

    #[inline]
    fn incident_faces(&self, v: usize) -> &[u32] {
        &self.faces_of[self.face_offsets[v] as usize..self.face_offsets[v + 1] as usize]
    }

    /// Weighted average of `field` over every vertex's one-ring, into `out`.
    ///
    /// `geom` and `field` are separate because the HC filter averages a *displacement*
    /// field over an unrelated geometry; for the other two they are the same buffer.
    /// Vertices with an empty one-ring — an unreferenced vertex — pass their own value
    /// through, so they simply never move.
    ///
    /// # Why the weights are not stored
    ///
    /// A `(1 / distance)` is a subtract, a dot product and a reciprocal square root: about
    /// as expensive as the load that reading it back from a `Vec<f64>` parallel to `nbrs`
    /// would cost, on a kernel that is already memory-bound. Not storing them buys three
    /// things — no `2.5M x f64` array resident alongside the mesh, no second pass to fill
    /// it, and no lifetime problem writing into disjoint CSR rows from rayon — and the
    /// geometry-dependent weightings come out recomputed from the *current* positions
    /// every iteration, which is the flow they are supposed to discretise rather than a
    /// snapshot taken before the first step (which is what a stored operator gives you,
    /// and what `trimesh` does).
    fn average(&self, faces: &[u32], geom: &[f64], field: &[f64], w: Weights, out: &mut [f64]) {
        out.par_chunks_exact_mut(3)
            .enumerate()
            .for_each(|(v, out)| {
                let row = self.row(v);
                if row.is_empty() {
                    out.copy_from_slice(&field[3 * v..3 * v + 3]);
                    return;
                }

                let (acc, total) = match w {
                    Weights::Uniform => uniform_sum(row, field),
                    Weights::InverseDistance => {
                        let mut acc = [0.0f64; 3];
                        let mut total = 0.0f64;
                        let c = &geom[3 * v..3 * v + 3];
                        for &u in row {
                            let g = &geom[3 * u as usize..3 * u as usize + 3];
                            let d2 = (0..3).map(|k| (c[k] - g[k]).powi(2)).sum::<f64>();
                            // A coincident neighbour would otherwise be an infinite
                            // weight, which is to say the only one — and coincident
                            // vertices are common in EM meshes. Cap rather than skip: a
                            // vertex whose neighbours are all coincident with it still
                            // needs somewhere to go, and that somewhere is where it is.
                            let wgt = 1.0 / d2.sqrt().max(1e-12);
                            let p = &field[3 * u as usize..3 * u as usize + 3];
                            for k in 0..3 {
                                acc[k] += wgt * p[k];
                            }
                            total += wgt;
                        }
                        (acc, total)
                    }
                    Weights::Cotangent => {
                        let mut acc = [0.0f64; 3];
                        let mut total = 0.0f64;
                        for &fi in self.incident_faces(v) {
                            let tri = &faces[3 * fi as usize..3 * fi as usize + 3];
                            // Rotate the triangle so `v` comes first. `v` is in it by
                            // construction, so the third arm is unreachable-but-cheap
                            // rather than a search.
                            let (p, q) = if tri[0] as usize == v {
                                (tri[1], tri[2])
                            } else if tri[1] as usize == v {
                                (tri[2], tri[0])
                            } else {
                                (tri[0], tri[1])
                            };
                            // The edge `v-p` is weighted by the cotangent at `q`, and vice
                            // versa. A face that names `v` twice has no such angle.
                            if p as usize != v {
                                let c = cotangent(geom, q, v as u32, p);
                                accumulate(&mut acc, &mut total, field, p, c);
                            }
                            if q as usize != v {
                                let c = cotangent(geom, p, v as u32, q);
                                accumulate(&mut acc, &mut total, field, q, c);
                            }
                        }
                        // Every angle obtuse, or every incident face degenerate. Falling
                        // back to the uniform umbrella keeps such a vertex smoothing at
                        // all, where renormalising nothing would freeze it.
                        if total.is_finite() && total > 0.0 {
                            (acc, total)
                        } else {
                            uniform_sum(row, field)
                        }
                    }
                };

                for k in 0..3 {
                    out[k] = acc[k] / total;
                }
            });
    }
}

/// Unweighted sum of `field` over a one-ring, and the count to divide it by.
///
/// [`Weights::Uniform`]'s whole kernel, and also what [`Weights::Cotangent`] falls back to
/// when the clamp has left it nothing to renormalise — which is why it is a function rather
/// than the body of a match arm.
#[inline]
fn uniform_sum(row: &[u32], field: &[f64]) -> ([f64; 3], f64) {
    let mut acc = [0.0f64; 3];
    for &u in row {
        let p = &field[3 * u as usize..3 * u as usize + 3];
        for k in 0..3 {
            acc[k] += p[k];
        }
    }
    (acc, row.len() as f64)
}

/// Cotangent of the angle at `apex` in the triangle `apex`, `a`, `b`, clamped at zero.
///
/// `cos / sin` as `dot / |cross|`, which needs no trigonometry and no normalisation: the
/// two vector lengths cancel between numerator and denominator.
///
/// Returns `0` for a degenerate triangle (`|cross| == 0`, i.e. zero area) and for an
/// obtuse angle. The clamp is the standard fix for the cotangent operator's one real flaw
/// — a negative weight pushes a vertex away from its neighbour, so an obtuse fan makes the
/// explicit iteration diverge instead of converge. The cost is that smoothing on a mesh of
/// mostly-obtuse triangles degrades towards the uniform umbrella, which is the right way
/// to fail.
#[inline]
fn cotangent(pos: &[f64], apex: u32, a: u32, b: u32) -> f64 {
    let o = &pos[3 * apex as usize..3 * apex as usize + 3];
    let p = &pos[3 * a as usize..3 * a as usize + 3];
    let q = &pos[3 * b as usize..3 * b as usize + 3];
    let u = [p[0] - o[0], p[1] - o[1], p[2] - o[2]];
    let v = [q[0] - o[0], q[1] - o[1], q[2] - o[2]];
    let dot = u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
    let cross = [
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    ];
    let sin = (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt();
    if sin > 0.0 && dot > 0.0 {
        dot / sin
    } else {
        0.0
    }
}

/// Add `w * field[u]` into a running one-ring accumulator.
#[inline]
fn accumulate(acc: &mut [f64; 3], total: &mut f64, field: &[f64], u: u32, w: f64) {
    if w > 0.0 {
        let p = &field[3 * u as usize..3 * u as usize + 3];
        for k in 0..3 {
            acc[k] += w * p[k];
        }
        *total += w;
    }
}

// ---------------------------------------------------------------------------
// Volume
// ---------------------------------------------------------------------------

/// Chunk size for the deterministic parallel reductions below.
///
/// Floating-point addition is not associative, so a reduction whose tree depends on how
/// rayon happened to split the work gives a different last bit run to run. Folding
/// fixed-size chunks in parallel and summing the *ordered* partials serially fixes the
/// tree to the input length alone, which is what makes two runs on the same mesh return
/// bit-identical volumes — and therefore a bit-identical scale factor.
const REDUCE_CHUNK: usize = 4096;

/// Signed volume enclosed by a triangle mesh, by the divergence theorem, with each face
/// capped off to `about` rather than to the origin.
///
/// `sum over faces of (a - o) . ((b - o) x (c - o)) / 6`: each face and `o` form a
/// tetrahedron whose signed volumes cancel everywhere except inside the surface, so on a
/// closed mesh the answer does not depend on `o` at all.
///
/// In floating point it depends on `o` enormously, which is why this takes one. Anchored at
/// the origin, a mesh 100 um across sitting at EM coordinates — 10^5 nm out, routinely —
/// sums terms of order 10^15 to reach an answer of order 10^9: six digits of cancellation,
/// on a quantity whose cube root is then used to rescale the mesh. `trimesh` anchors at the
/// origin and this is the second of the two reasons its volume constraint returns NaN on a
/// mesh that has been translated (the first being that it *scales* about the origin too).
/// Anchored at the mesh's own centroid the terms are the size of the answer and nothing
/// cancels.
///
/// For an *open* mesh the result is `o`-dependent even in exact arithmetic — which is why
/// [`Volume::Undefined`] exists, and why the caller passes the same `o` for the before and
/// after measurements so that at least their ratio compares like with like.
fn signed_volume(faces: &[u32], pos: &[f64], about: [f64; 3]) -> f64 {
    let partials: Vec<f64> = faces
        .par_chunks(REDUCE_CHUNK * 3)
        .map(|chunk| {
            chunk
                .chunks_exact(3)
                .map(|tri| {
                    let at = |i: u32| {
                        let p = &pos[3 * i as usize..3 * i as usize + 3];
                        [p[0] - about[0], p[1] - about[1], p[2] - about[2]]
                    };
                    let (a, b, c) = (at(tri[0]), at(tri[1]), at(tri[2]));
                    let cross = [
                        a[1] * b[2] - a[2] * b[1],
                        a[2] * b[0] - a[0] * b[2],
                        a[0] * b[1] - a[1] * b[0],
                    ];
                    cross[0] * c[0] + cross[1] * c[1] + cross[2] * c[2]
                })
                .sum::<f64>()
        })
        .collect();
    partials.iter().sum::<f64>() / 6.0
}

/// Mean of the vertex positions.
fn centroid(pos: &[f64]) -> [f64; 3] {
    let n = (pos.len() / 3) as f64;
    if n == 0.0 {
        return [0.0; 3];
    }
    let partials: Vec<[f64; 3]> = pos
        .par_chunks(REDUCE_CHUNK * 3)
        .map(|chunk| {
            let mut acc = [0.0f64; 3];
            for p in chunk.chunks_exact(3) {
                for k in 0..3 {
                    acc[k] += p[k];
                }
            }
            acc
        })
        .collect();
    let mut acc = [0.0f64; 3];
    for p in &partials {
        for k in 0..3 {
            acc[k] += p[k];
        }
    }
    [acc[0] / n, acc[1] / n, acc[2] / n]
}

/// Restore the input's enclosed volume by scaling the smoothed mesh about its own centroid.
///
/// About the centroid, not the origin: a scaling has a fixed point, and the only one that
/// is a property of the *mesh* rather than of the coordinate system it happens to be
/// expressed in is one of the mesh's own points. Anchoring anywhere else makes the
/// operation depend on where the mesh sits, which is how `trimesh` ends up translating a
/// neuron by twice its own diameter — see the module docs. Anchoring at the *smoothed*
/// centroid also makes the guarantee exact in the useful direction: the correction changes
/// the mesh's size and provably not its position.
///
/// `before` is measured about `about`, and so is the after; passing the same anchor for
/// both is what makes their ratio meaningful on a mesh that is not quite closed.
///
/// A ratio that is not finite and positive means at least one of the two volumes is zero,
/// non-finite, or of the opposite sign — an open mesh, a degenerate one, or one whose
/// winding flipped. None of those has a cube root worth taking, so the vertices are left as
/// the filter produced them and the two measurements are handed back for the caller to
/// report. Consistently inverted winding is *not* in that set: both volumes come out
/// negative, the ratio is positive, and the correction is as valid as it is on a mesh wound
/// the other way.
fn correct_volume(faces: &[u32], before: f64, about: [f64; 3], pos: &mut [f64]) -> Volume {
    let after = signed_volume(faces, pos, about);
    let ratio = before / after;
    if !ratio.is_finite() || ratio <= 0.0 {
        return Volume::Undefined { before, after };
    }
    let scale = ratio.cbrt();
    let c = centroid(pos);
    pos.par_chunks_exact_mut(3).for_each(|p| {
        for k in 0..3 {
            p[k] = c[k] + scale * (p[k] - c[k]);
        }
    });
    Volume::Scaled(scale)
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Smooth a triangle mesh.
///
/// Moves vertices; touches nothing else. The face array, the vertex count and the vertex
/// order all come back unchanged, so anything the caller has indexed by vertex stays
/// attached to the vertex it was attached to.
///
/// See [`Filter`] for what the three filters do and when to reach for each, [`Weights`]
/// for the three ways of weighting a one-ring, and the module documentation for what this
/// deliberately does differently from `trimesh.smoothing.filter_laplacian`.
///
/// Arguments:
///
/// - `faces`: `(F, 3)` triangular faces, as vertex indices
/// - `vertices`: `(V, 3)` vertex positions
/// - `filter`: which filter to run, and its parameters
/// - `weights`: how to weight each vertex's one-ring
/// - `iterations`: passes to run. For [`Filter::Taubin`] one iteration is a full
///   `lambda`-then-`mu` pair; see that variant
/// - `preserve_border`: pin every vertex on a mesh boundary — an endpoint of an edge used
///   by exactly one face. Without this, an open mesh's rim rolls inwards under any of
///   these filters, because a boundary vertex's one-ring lies entirely to one side of it
/// - `lock`: optional `(V, )` flags marking vertices that must not move. Unioned with
///   `preserve_border`, not an alternative to it. Locked vertices still pull on their
///   neighbours, which is what makes them a boundary condition rather than a hole. Named
///   to match [`crate::simplify::simplify_mesh`]'s `lock`, which is the same concept with
///   the same `preserve_border` companion
/// - `volume_correction`: rescale about the centroid afterwards so the enclosed volume
///   matches the input's. See [`Smoothed::volume`]
/// - `threads`: cap on the rayon worker count for this call; `None` uses the global pool
///
/// Returns:
///
/// A [`Smoothed`] carrying the new `(V, 3)` positions and what the volume correction did.
///
/// # Panics
///
/// If `faces` or `vertices` is not 3 columns wide, if a face names a vertex the mesh does
/// not have, if `lock` is not `V` long, or if `filter`'s parameters are outside the ranges
/// [`Filter`] documents.
// The filter's own parameters are already folded into one argument, which is what `Filter`
// is for; what is left is the mesh, and five independent choices about how to treat it.
// Bundling those into a config struct would only move the same list somewhere the caller
// has to name a type to fill in.
#[allow(clippy::too_many_arguments)]
pub fn smooth_mesh(
    faces: ArrayView2<u32>,
    vertices: ArrayView2<f64>,
    filter: Filter,
    weights: Weights,
    iterations: usize,
    preserve_border: bool,
    lock: Option<&[bool]>,
    volume_correction: bool,
    threads: Option<usize>,
) -> Smoothed {
    assert_eq!(faces.ncols(), 3, "`faces` must have shape (F, 3)");
    assert_eq!(vertices.ncols(), 3, "`vertices` must have shape (V, 3)");
    filter.validate();
    let n = vertices.nrows();
    if let Some(l) = lock {
        assert_eq!(
            l.len(),
            n,
            "`lock` must have one flag per vertex: {} vs {n} vertices",
            l.len()
        );
    }

    // Both arrays are borrowed as-is from the Python side, which always hands us C-order;
    // a strided view from a Rust caller is copied into standard layout here so the kernels
    // below can index flat.
    let fstore = faces.as_standard_layout();
    let f: &[u32] = fstore.as_slice().expect("standard layout is contiguous");
    let vstore = vertices.as_standard_layout();
    let v0: &[f64] = vstore.as_slice().expect("standard layout is contiguous");

    with_pool(threads, || {
        let mut umbrella = Umbrella::new(f, n, weights == Weights::Cotangent);

        // One flag array covering both ways of pinning a vertex, or none at all — which
        // is the common case and lets the update loop skip the test entirely.
        let frozen: Option<Vec<bool>> = match (preserve_border, lock) {
            (false, None) => None,
            // `take` rather than `clone`: this is the only read of the field, and the
            // copy would be a second `Vec<bool>` the size of the mesh.
            (true, None) => Some(std::mem::take(&mut umbrella.border)),
            (false, Some(l)) => Some(l.to_vec()),
            (true, Some(l)) => Some(
                umbrella
                    .border
                    .iter()
                    .zip(l)
                    .map(|(&b, &l)| b || l)
                    .collect(),
            ),
        };

        let mut cur = v0.to_vec();
        let mut scratch = vec![0.0f64; 3 * n];

        match filter {
            Filter::Laplacian { lambda } => {
                for _ in 0..iterations {
                    umbrella.average(f, &cur, &cur, weights, &mut scratch);
                    step(&mut cur, &scratch, lambda, frozen.as_deref());
                }
            }
            // Shrink, then inflate. The two passes are the same code with opposite signs;
            // what makes them a band-pass rather than a wash is that `|mu| > lambda`, so
            // the inflate over-corrects by exactly enough to leave frequencies below the
            // pass-band where they started.
            Filter::Taubin { lambda, mu } => {
                for _ in 0..iterations {
                    umbrella.average(f, &cur, &cur, weights, &mut scratch);
                    step(&mut cur, &scratch, lambda, frozen.as_deref());
                    umbrella.average(f, &cur, &cur, weights, &mut scratch);
                    step(&mut cur, &scratch, mu, frozen.as_deref());
                }
            }
            Filter::Humphrey { alpha, beta } => {
                let original = v0;
                let mut push = vec![0.0f64; 3 * n];
                for _ in 0..iterations {
                    // A *full* Laplacian step, not a damped one: HC's damping is the
                    // push-back below, so taking a partial step here would damp twice.
                    //
                    // Swapped rather than copied back, which is also what saves a third
                    // `3n` buffer: after the swap `cur` holds the stepped positions and
                    // `scratch` holds the pre-step ones, which is exactly the `prev` the
                    // displacement below needs. `scratch` is then free again as soon as
                    // `push` has been computed from it.
                    umbrella.average(f, &cur, &cur, weights, &mut scratch);
                    std::mem::swap(&mut cur, &mut scratch);
                    // Here rather than only at the end of the iteration, and the
                    // difference is not bookkeeping. Everything below reads `cur` — as
                    // the geometry the weights come from, and through `push` — so a
                    // frozen vertex left at the position this step gave it would act on
                    // its neighbours from somewhere it never actually was, and put them
                    // somewhere they would not otherwise be. Restoring only at the end
                    // would still satisfy "a locked vertex does not move"; it would not
                    // satisfy "a locked vertex is a boundary condition".
                    if let Some(frozen) = frozen.as_deref() {
                        restore(&mut cur, original, frozen);
                    }

                    // How far the step carried each vertex from the blend of where it
                    // started and where it was — the displacement HC undoes part of.
                    push.par_iter_mut()
                        .zip(cur.par_iter())
                        .zip(original.par_iter())
                        .zip(scratch.par_iter())
                        .for_each(|(((b, &c), &o), &q)| *b = c - (alpha * o + (1.0 - alpha) * q));

                    // Undo it, splitting the correction between the vertex itself and the
                    // average over its one-ring; `beta` is the split.
                    umbrella.average(f, &cur, &push, weights, &mut scratch);
                    cur.par_iter_mut()
                        .zip(push.par_iter())
                        .zip(scratch.par_iter())
                        .for_each(|((c, &b), &a)| *c -= beta * b + (1.0 - beta) * a);

                    // A frozen vertex's own `push` is zero after the restore above, but
                    // the one-ring average of its neighbours' is not, so this subtraction
                    // would still move it.
                    if let Some(frozen) = frozen.as_deref() {
                        restore(&mut cur, original, frozen);
                    }
                }
            }
        }

        // Both volumes are measured about the *input* centroid so that a mesh which is not
        // quite closed — where the measurement is anchor-dependent even in exact arithmetic
        // — at least has its before and after compared on the same terms.
        let volume = if volume_correction {
            let anchor = centroid(v0);
            correct_volume(f, signed_volume(f, v0, anchor), anchor, &mut cur)
        } else {
            Volume::Off
        };

        Smoothed {
            vertices: Array2::from_shape_vec((n, 3), cur).expect("3n values, n rows of 3"),
            volume,
        }
    })
}

/// Move every unfrozen vertex a `factor` of the way from where it is to `target`.
///
/// The whole update, for [`Filter::Laplacian`] and both halves of [`Filter::Taubin`]: a
/// negative `factor` is the inflating pass and needs no separate code path.
fn step(pos: &mut [f64], target: &[f64], factor: f64, frozen: Option<&[bool]>) {
    match frozen {
        None => pos
            .par_iter_mut()
            .zip(target.par_iter())
            .for_each(|(p, &t)| *p += factor * (t - *p)),
        Some(frozen) => pos
            .par_chunks_exact_mut(3)
            .zip(target.par_chunks_exact(3))
            .zip(frozen.par_iter())
            .for_each(|((p, t), &f)| {
                if !f {
                    for k in 0..3 {
                        p[k] += factor * (t[k] - p[k]);
                    }
                }
            }),
    }
}

/// Put every frozen vertex back where it started.
///
/// The HC filter's inner passes are written against whole buffers rather than per-vertex —
/// each is a fused elementwise expression over four of them, which a per-vertex skip would
/// break up for no gain — so pinning is applied by rewinding after each pass rather than
/// by never taking it. [`Filter::Humphrey`] calls this twice per iteration for that reason:
/// once is not enough, because between the two passes a stale frozen vertex is read by its
/// neighbours.
fn restore(pos: &mut [f64], original: &[f64], frozen: &[bool]) {
    pos.par_chunks_exact_mut(3)
        .zip(original.par_chunks_exact(3))
        .zip(frozen.par_iter())
        .for_each(|((p, o), &f)| {
            if f {
                p.copy_from_slice(o);
            }
        });
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::tests_support::{grid, uv_sphere};
    use ndarray::{array, Array2};

    const ALL_WEIGHTS: [Weights; 3] = [
        Weights::Uniform,
        Weights::InverseDistance,
        Weights::Cotangent,
    ];

    /// The three filters at settings that actually smooth, for the tests that must hold
    /// whichever one is running.
    const ALL_FILTERS: [Filter; 3] = [
        Filter::Laplacian { lambda: 0.5 },
        Filter::Taubin {
            lambda: 0.5,
            mu: -0.53,
        },
        Filter::Humphrey {
            alpha: 0.1,
            beta: 0.5,
        },
    ];

    /// `smooth_mesh` with the arguments these tests mostly do not vary.
    fn smooth(
        faces: &Array2<u32>,
        verts: &Array2<f64>,
        filter: Filter,
        weights: Weights,
        iterations: usize,
    ) -> Array2<f64> {
        smooth_mesh(
            faces.view(),
            verts.view(),
            filter,
            weights,
            iterations,
            false,
            None,
            false,
            None,
        )
        .vertices
    }

    fn max_abs_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f64, f64::max)
    }

    /// Volume of a closed mesh, measured the way a caller would rather than the way the
    /// module does internally — so the tests are not checking `signed_volume` against
    /// itself.
    fn volume_of(faces: &Array2<u32>, verts: &Array2<f64>) -> f64 {
        let mut v = 0.0;
        for tri in faces.rows() {
            let p = |i: usize| {
                let r = verts.row(tri[i] as usize);
                [r[0], r[1], r[2]]
            };
            let (a, b, c) = (p(0), p(1), p(2));
            v += a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
                + a[2] * (b[0] * c[1] - b[1] * c[0]);
        }
        v / 6.0
    }

    // --------------------------------------------------------------- the one-ring

    /// The grid's rim is a boundary and its interior is not; a closed sphere has no
    /// boundary at all. Both are read off the run lengths in the dedup pass, so getting
    /// this wrong would silently disarm `preserve_border`.
    #[test]
    fn border_flags_follow_the_topology() {
        let (faces, _) = grid(5, 1.0);
        let u = Umbrella::new(faces.as_slice().unwrap(), 25, false);
        for i in 0..5 {
            for j in 0..5 {
                let rim = i == 0 || j == 0 || i == 4 || j == 4;
                assert_eq!(u.border[i * 5 + j], rim, "vertex ({i}, {j})");
            }
        }

        let (faces, verts) = uv_sphere(12, 12);
        let u = Umbrella::new(faces.as_slice().unwrap(), verts.nrows(), false);
        // The generator leaves the two polar rings open, so only those are boundary.
        let interior = (12..verts.nrows() - 12).filter(|&v| !u.border[v]).count();
        assert_eq!(
            interior,
            verts.nrows() - 24,
            "no interior vertex is a border"
        );
    }

    /// Valence 6 in the interior of the diagonal-split grid, and no duplicates or
    /// self-loops anywhere.
    #[test]
    fn one_ring_is_deduplicated() {
        let (faces, _) = grid(5, 1.0);
        let u = Umbrella::new(faces.as_slice().unwrap(), 25, false);
        assert_eq!(u.row(12).len(), 6, "interior valence");
        for v in 0..25 {
            let row = u.row(v);
            assert!(row.windows(2).all(|w| w[0] < w[1]), "sorted and unique");
            assert!(!row.contains(&(v as u32)), "no self-loop");
        }
    }

    // --------------------------------------------------------------- fixed points

    /// A flat, regularly triangulated grid is an exact fixed point of the uniform
    /// umbrella: an interior vertex's six neighbours are `(i±1, j)`, `(i, j±1)` and
    /// `(i±1, j±1)`, which average back to `(i, j)`. With the rim pinned, nothing at all
    /// moves — and it must be exact, not approximate, because there is no rounding in a
    /// sum of six exactly-representable values divided by six.
    ///
    /// Run for all three filters, because this is also what pins down what *pinning*
    /// means: not merely that a frozen vertex ends where it started, but that it never
    /// acts on its neighbours from anywhere else. A filter that let the rim wander
    /// mid-iteration and put it back afterwards would still pass a
    /// "pinned vertices did not move" test, and would fail this one — the interior next
    /// to the rim would be dragged by an excursion that officially never happened.
    #[test]
    fn a_flat_grid_is_a_fixed_point() {
        let (faces, verts) = grid(7, 1.0);
        for f in [
            Filter::Laplacian { lambda: 1.0 },
            Filter::Taubin {
                lambda: 0.5,
                mu: -0.53,
            },
            Filter::Humphrey {
                alpha: 0.1,
                beta: 0.5,
            },
        ] {
            let out = smooth_mesh(
                faces.view(),
                verts.view(),
                f,
                Weights::Uniform,
                25,
                true,
                None,
                false,
                None,
            );
            assert_eq!(out.vertices, verts, "{f:?} moved a vertex");
            assert_eq!(out.volume, Volume::Off);
        }
    }

    /// Zero iterations returns the input untouched, whatever else was asked for.
    #[test]
    fn zero_iterations_is_the_identity() {
        let (faces, verts) = uv_sphere(10, 10);
        for f in ALL_FILTERS {
            for w in ALL_WEIGHTS {
                assert_eq!(smooth(&faces, &verts, f, w, 0), verts, "{f:?} / {w:?}");
            }
        }
    }

    /// One full-strength Laplacian pass puts a lone spike exactly back in the plane its
    /// six neighbours span. The arithmetic is small enough to be exact, so this pins the
    /// *value* of the update and not just its direction.
    #[test]
    fn a_spike_collapses_to_its_ring() {
        let (faces, mut verts) = grid(5, 1.0);
        verts[[12, 2]] = 1.0; // the middle vertex, lifted out of the plane
        let out = smooth(
            &faces,
            &verts,
            Filter::Laplacian { lambda: 1.0 },
            Weights::Uniform,
            1,
        );
        assert_eq!(out[[12, 2]], 0.0);
    }

    // --------------------------------------------------------------- what smoothing does

    /// The point of the exercise: noise on a flat surface goes away, and does so under
    /// every filter and every weighting.
    #[test]
    fn noise_on_a_plane_is_removed() {
        let (faces, flat) = grid(12, 1.0);
        let mut noisy = flat.clone();
        // Deterministic sign-alternating displacement rather than a PRNG: it is the
        // highest frequency the grid can carry, which is exactly what these filters are
        // supposed to be best at.
        for v in 0..noisy.nrows() {
            noisy[[v, 2]] = if v % 2 == 0 { 0.4 } else { -0.4 };
        }

        // Measured in `z` alone, and this fixture is chosen so that that is the whole
        // story: the clean surface is the plane `z = 0`, so displacement out of it is
        // exactly the noise, and any sliding *within* the plane — which the uniform
        // umbrella does plenty of, see `cotangent_drifts_less_than_uniform` — does not
        // contaminate the measurement. Nothing is pinned, so all of it is the filter's to
        // remove; `preserve_border` has its own test.
        let energy = |z: &Array2<f64>| -> f64 { z.column(2).iter().map(|z| z * z).sum() };
        let rough = energy(&noisy);

        for f in ALL_FILTERS {
            for w in ALL_WEIGHTS {
                let out = smooth(&faces, &noisy, f, w, 20);
                let left = energy(&out);
                assert!(out.iter().all(|x| x.is_finite()), "{f:?} / {w:?}");

                // Every combination has to remove most of it. The binding case is HC on
                // cotangent weights, which converges to a fixed point still carrying ~31%
                // and stays there however long it runs — a maximally corrugated grid makes
                // nearly every triangle obtuse, the clamp then zeroes most of the
                // cotangents, and what survives is a lopsided operator that HC's pull-back
                // towards the (very rough) input holds in place. Both halves of that are
                // the documented behaviour of their respective parts rather than a defect;
                // on a surface whose triangles are not folded over, cotangent weights are
                // the *best* behaved of the three.
                assert!(left < 0.35 * rough, "{f:?} / {w:?} left {left} of {rough}");

                // Laplacian and Taubin have no term holding on to the input, so on this
                // fixture they are held to the tighter bar.
                if !matches!(f, Filter::Humphrey { .. }) {
                    assert!(left < 0.06 * rough, "{f:?} / {w:?} left {left} of {rough}");
                }
            }
        }
    }

    /// Taubin exists because Laplacian shrinks; this is the claim in the module docs, on
    /// the shape it is claimed about.
    #[test]
    fn taubin_holds_the_volume_laplacian_loses() {
        let (faces, verts) = uv_sphere(24, 24);
        let v0 = volume_of(&faces, &verts);

        let lap = volume_of(
            &faces,
            &smooth(
                &faces,
                &verts,
                Filter::Laplacian { lambda: 0.5 },
                Weights::Uniform,
                20,
            ),
        );
        let tau = volume_of(
            &faces,
            &smooth(
                &faces,
                &verts,
                Filter::Taubin {
                    lambda: 0.5,
                    mu: -0.53,
                },
                Weights::Uniform,
                20,
            ),
        );

        assert!(lap < 0.75 * v0, "laplacian should shrink: {lap} vs {v0}");
        assert!(
            tau > 0.95 * v0,
            "taubin should hold its volume: {tau} vs {v0}"
        );
    }

    /// HC pulls back towards the input, so it moves vertices less far than the plain
    /// Laplacian step it is built on.
    #[test]
    fn humphrey_stays_nearer_the_original() {
        let (faces, verts) = uv_sphere(16, 16);
        let lap = smooth(
            &faces,
            &verts,
            Filter::Laplacian { lambda: 1.0 },
            Weights::Uniform,
            10,
        );
        let hc = smooth(
            &faces,
            &verts,
            Filter::Humphrey {
                alpha: 0.5,
                beta: 0.5,
            },
            Weights::Uniform,
            10,
        );
        assert!(
            max_abs_diff(&hc, &verts) < max_abs_diff(&lap, &verts),
            "HC moved further than plain Laplacian"
        );
    }

    /// Cotangent weights earn their keep on a mesh whose triangles are not all the same
    /// size, which is every mesh out of EM segmentation and — conveniently — a UV sphere,
    /// whose rings crowd together at the poles.
    ///
    /// The uniform umbrella cannot tell "this vertex is off the surface" from "this vertex
    /// has closer neighbours on one side than the other", so on uneven sampling it slides
    /// vertices *along* the surface towards even spacing. The cotangent operator is a
    /// function of the angles rather than of the tessellation, so it moves them along the
    /// normal and leaves them alone within the surface. Both are run with the volume
    /// correction on so that shrinkage is not what is being measured.
    #[test]
    fn cotangent_drifts_less_than_uniform() {
        let (faces, verts) = uv_sphere(24, 24);
        let drift = |w| {
            let out = smooth_mesh(
                faces.view(),
                verts.view(),
                Filter::Laplacian { lambda: 0.5 },
                w,
                10,
                false,
                None,
                true,
                None,
            )
            .vertices;
            out.rows()
                .into_iter()
                .zip(verts.rows())
                .map(|(p, q)| (0..3).map(|k| (p[k] - q[k]).powi(2)).sum::<f64>())
                .sum::<f64>()
        };
        let (uniform, cotangent) = (drift(Weights::Uniform), drift(Weights::Cotangent));
        assert!(
            cotangent < 0.5 * uniform,
            "cotangent drifted {cotangent}, uniform {uniform}"
        );
    }

    // --------------------------------------------------------------- pinning

    /// Pinned vertices come back at bitwise the same coordinates — not nearly, exactly —
    /// under every filter, because "pinned" is worth nothing if it means "moved a little".
    #[test]
    fn pinned_vertices_do_not_move() {
        let (faces, verts) = uv_sphere(12, 12);
        let mut lock = vec![false; verts.nrows()];
        for (i, p) in lock.iter_mut().enumerate() {
            *p = i % 5 == 0;
        }

        for f in ALL_FILTERS {
            for w in ALL_WEIGHTS {
                let out = smooth_mesh(
                    faces.view(),
                    verts.view(),
                    f,
                    w,
                    10,
                    false,
                    Some(&lock),
                    false,
                    None,
                )
                .vertices;
                for (v, &pinned) in lock.iter().enumerate() {
                    if pinned {
                        assert_eq!(out.row(v), verts.row(v), "vertex {v}, {f:?} / {w:?}");
                    }
                }
                assert!(
                    max_abs_diff(&out, &verts) > 0.0,
                    "{f:?} / {w:?} moved nothing at all"
                );
            }
        }
    }

    /// `preserve_border` pins exactly the rim, and nothing else — an unpinned interior
    /// still has to smooth.
    #[test]
    fn preserve_border_pins_the_rim() {
        let (faces, flat) = grid(9, 1.0);
        let mut noisy = flat.clone();
        for v in 0..noisy.nrows() {
            noisy[[v, 2]] = if v % 2 == 0 { 0.3 } else { -0.3 };
        }
        let out = smooth_mesh(
            faces.view(),
            noisy.view(),
            Filter::Laplacian { lambda: 0.5 },
            Weights::Uniform,
            10,
            true,
            None,
            false,
            None,
        )
        .vertices;

        for i in 0..9 {
            for j in 0..9 {
                let v = i * 9 + j;
                if i == 0 || j == 0 || i == 8 || j == 8 {
                    assert_eq!(out.row(v), noisy.row(v), "rim vertex ({i}, {j}) moved");
                }
            }
        }
        assert!(out[[40, 2]].abs() < 0.05, "interior did not smooth");
    }

    /// The two ways of pinning are a union, not alternatives.
    #[test]
    fn preserve_border_and_pin_compose() {
        let (faces, verts) = grid(5, 1.0);
        let mut noisy = verts.clone();
        for v in 0..noisy.nrows() {
            noisy[[v, 2]] = if v % 2 == 0 { 0.3 } else { -0.3 };
        }
        let mut lock = vec![false; 25];
        lock[12] = true; // an interior vertex, so not covered by the border flags

        let out = smooth_mesh(
            faces.view(),
            noisy.view(),
            Filter::Laplacian { lambda: 0.5 },
            Weights::Uniform,
            5,
            true,
            Some(&lock),
            false,
            None,
        )
        .vertices;
        assert_eq!(out.row(12), noisy.row(12), "explicitly locked vertex moved");
        assert_eq!(out.row(0), noisy.row(0), "border vertex moved");
        assert_ne!(out.row(7), noisy.row(7), "an unpinned vertex should move");
    }

    // --------------------------------------------------------------- volume correction

    /// The correction does what it says: the volume comes back.
    #[test]
    fn volume_correction_restores_the_volume() {
        let (faces, verts) = uv_sphere(24, 24);
        let v0 = volume_of(&faces, &verts);
        let out = smooth_mesh(
            faces.view(),
            verts.view(),
            Filter::Laplacian { lambda: 0.5 },
            Weights::Uniform,
            10,
            false,
            None,
            true,
            None,
        );
        let v1 = volume_of(&faces, &out.vertices);
        assert!((v1 / v0 - 1.0).abs() < 1e-9, "volume {v1} vs {v0}");
        match out.volume {
            Volume::Scaled(s) => assert!(s > 1.0, "should have been scaled up, got {s}"),
            other => panic!("expected Scaled, got {other:?}"),
        }
    }

    /// The correction moves the size and provably not the position — which is the whole
    /// difference from `trimesh`, whose scaling about the origin displaces the 722817260
    /// neuron by 41 um at navis' own defaults.
    #[test]
    fn volume_correction_does_not_move_the_mesh() {
        let (faces, verts) = uv_sphere(20, 20);
        let out = smooth_mesh(
            faces.view(),
            verts.view(),
            Filter::Laplacian { lambda: 0.5 },
            Weights::Uniform,
            10,
            false,
            None,
            true,
            None,
        );
        let before = centroid(out.vertices.as_slice().unwrap());
        // Re-derive the smoothed-but-unscaled centroid the long way and compare: the
        // correction is anchored there, so it must be a fixed point.
        let plain = smooth(
            &faces,
            &verts,
            Filter::Laplacian { lambda: 0.5 },
            Weights::Uniform,
            10,
        );
        let after = centroid(plain.as_slice().unwrap());
        for k in 0..3 {
            assert!(
                (before[k] - after[k]).abs() < 1e-12,
                "centroid moved on axis {k}: {} vs {}",
                before[k],
                after[k]
            );
        }
    }

    /// Smoothing a mesh at the origin and smoothing the same mesh 100 um away must give
    /// the same shape. `trimesh` fails this outright — its volume constraint returns NaN
    /// at this offset — and it is the reason both the volume measurement and the scaling
    /// here are anchored to the mesh rather than to the coordinate system.
    #[test]
    fn everything_is_translation_equivariant() {
        let (faces, verts) = uv_sphere(20, 20);
        let offset = 1e5;
        let moved = verts.map(|x| x + offset);

        for f in ALL_FILTERS {
            for w in ALL_WEIGHTS {
                for correct in [false, true] {
                    let here = smooth_mesh(
                        faces.view(),
                        verts.view(),
                        f,
                        w,
                        10,
                        false,
                        None,
                        correct,
                        None,
                    );
                    let there = smooth_mesh(
                        faces.view(),
                        moved.view(),
                        f,
                        w,
                        10,
                        false,
                        None,
                        correct,
                        None,
                    );
                    let back = there.vertices.map(|x| x - offset);
                    // The mesh is unit-radius and sits at 1e5, so f64 resolves it to
                    // ~1e-11 there; anything at that level is the translation itself.
                    assert!(
                        max_abs_diff(&here.vertices, &back) < 1e-8,
                        "{f:?} / {w:?} / correct={correct} is not translation equivariant"
                    );
                }
            }
        }
    }

    /// The claim that justifies correcting the volume once at the end rather than every
    /// iteration: the filters commute with a uniform scaling, so the two are equal.
    #[test]
    fn smoothing_is_scale_equivariant() {
        let (faces, verts) = uv_sphere(16, 16);
        let s = 7.5;

        for f in ALL_FILTERS {
            for w in ALL_WEIGHTS {
                let scale_then_smooth = smooth(&faces, &verts.map(|x| x * s), f, w, 8);
                let smooth_then_scale = smooth(&faces, &verts, f, w, 8).map(|x| x * s);
                assert!(
                    max_abs_diff(&scale_then_smooth, &smooth_then_scale) < 1e-10,
                    "{f:?} / {w:?} does not commute with scaling"
                );
            }
        }
    }

    /// An open mesh has no volume to restore, and says so instead of scaling by a
    /// meaningless factor. A flat grid's signed volume is exactly zero.
    #[test]
    fn an_open_mesh_reports_undefined_volume() {
        let (faces, flat) = grid(6, 1.0);
        let mut noisy = flat.clone();
        for v in 0..noisy.nrows() {
            noisy[[v, 2]] = if v % 2 == 0 { 0.2 } else { -0.2 };
        }
        let out = smooth_mesh(
            faces.view(),
            noisy.view(),
            Filter::Laplacian { lambda: 0.5 },
            Weights::Uniform,
            5,
            false,
            None,
            true,
            None,
        );
        match out.volume {
            Volume::Undefined { .. } => {}
            other => panic!("expected Undefined, got {other:?}"),
        }
        // Undefined means "left alone", not "left unsmoothed".
        assert!(out.vertices.iter().all(|x| x.is_finite()));
        assert!(out.vertices[[21, 2]].abs() < 0.2);
    }

    /// Consistently inverted winding is not a degenerate case: both volumes come out
    /// negative and the ratio is as valid as it ever was.
    #[test]
    fn inverted_winding_still_corrects() {
        let (faces, verts) = uv_sphere(16, 16);
        let flipped = {
            let mut f = faces.clone();
            for mut row in f.rows_mut() {
                row.swap(1, 2);
            }
            f
        };
        assert!(volume_of(&flipped, &verts) < 0.0, "fixture is inside-out");

        let out = smooth_mesh(
            flipped.view(),
            verts.view(),
            Filter::Laplacian { lambda: 0.5 },
            Weights::Uniform,
            10,
            false,
            None,
            true,
            None,
        );
        assert!(matches!(out.volume, Volume::Scaled(_)), "{:?}", out.volume);
        let ratio = volume_of(&flipped, &out.vertices) / volume_of(&flipped, &verts);
        assert!((ratio - 1.0).abs() < 1e-9, "volume ratio {ratio}");
    }

    // --------------------------------------------------------------- degenerate input

    /// The shapes EM meshes are actually made of: zero-area faces, faces naming the same
    /// vertex twice, duplicated faces, and vertices no face mentions. None is an error,
    /// nothing comes back NaN, and an unreferenced vertex stays exactly where it was.
    #[test]
    fn degenerate_geometry_is_merely_data() {
        let faces = array![
            [0u32, 1, 2],
            [0, 1, 2], // duplicate
            [1, 2, 3],
            [4, 4, 5], // names a vertex twice
            [5, 6, 7], // zero-area: three collinear points
        ];
        let verts = array![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.5],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [9.0, 9.0, 9.0], // referenced by nothing
        ];

        for f in ALL_FILTERS {
            for w in ALL_WEIGHTS {
                let out = smooth(&faces, &verts, f, w, 5);
                assert!(
                    out.iter().all(|x| x.is_finite()),
                    "{f:?} / {w:?} produced a non-finite coordinate"
                );
                assert_eq!(out.row(8), verts.row(8), "{f:?} / {w:?} moved an orphan");
                assert_eq!(out.dim(), verts.dim());
            }
        }
    }

    /// A mesh with no faces at all is a no-op, not a panic.
    #[test]
    fn an_empty_mesh_is_a_no_op() {
        let faces = Array2::<u32>::zeros((0, 3));
        let verts = array![[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]];
        let out = smooth_mesh(
            faces.view(),
            verts.view(),
            Filter::Taubin {
                lambda: 0.5,
                mu: -0.53,
            },
            Weights::Cotangent,
            10,
            true,
            None,
            true,
            None,
        );
        assert_eq!(out.vertices, verts);
        // Zero volume before and after, so the ratio is NaN rather than 1.
        assert!(matches!(out.volume, Volume::Undefined { .. }));
    }

    /// Cotangent weights degrade to the uniform umbrella where the geometry gives them
    /// nothing to work with — every incident triangle degenerate — rather than freezing
    /// the vertex or dividing by zero.
    #[test]
    fn cotangent_falls_back_to_uniform_on_degenerate_faces() {
        // Four collinear points: every face has zero area, so every cotangent is dropped.
        let faces = array![[0u32, 1, 2], [1, 2, 3]];
        let verts = array![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0]
        ];
        let cot = smooth(
            &faces,
            &verts,
            Filter::Laplacian { lambda: 0.5 },
            Weights::Cotangent,
            3,
        );
        let uni = smooth(
            &faces,
            &verts,
            Filter::Laplacian { lambda: 0.5 },
            Weights::Uniform,
            3,
        );
        assert_eq!(cot, uni);
    }

    /// Coincident vertices are a capped weight rather than an infinite one.
    #[test]
    fn coincident_vertices_do_not_blow_up() {
        let faces = array![[0u32, 1, 2], [1, 2, 3]];
        let verts = array![
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0], // exactly on top of vertex 0
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0]
        ];
        for w in ALL_WEIGHTS {
            let out = smooth(&faces, &verts, Filter::Laplacian { lambda: 0.5 }, w, 5);
            assert!(out.iter().all(|x| x.is_finite()), "{w:?}");
        }
    }

    // --------------------------------------------------------------- rejected input

    #[test]
    #[should_panic(expected = "`lambda` must be in [0, 1]")]
    fn laplacian_lambda_above_one_is_rejected() {
        Filter::Laplacian { lambda: 1.5 }.validate();
    }

    #[test]
    #[should_panic(expected = "larger in magnitude than `lambda`")]
    fn taubin_mu_smaller_than_lambda_is_rejected() {
        Filter::Taubin {
            lambda: 0.5,
            mu: -0.4,
        }
        .validate();
    }

    #[test]
    #[should_panic(expected = "`mu` must be negative")]
    fn taubin_positive_mu_is_rejected() {
        Filter::Taubin {
            lambda: 0.5,
            mu: 0.53,
        }
        .validate();
    }

    #[test]
    #[should_panic(expected = "`beta` must be in [0, 1]")]
    fn humphrey_beta_out_of_range_is_rejected() {
        Filter::Humphrey {
            alpha: 0.5,
            beta: 2.0,
        }
        .validate();
    }

    #[test]
    #[should_panic(expected = "one flag per vertex")]
    fn a_short_lock_array_is_rejected() {
        let (faces, verts) = grid(4, 1.0);
        let lock = vec![false; 3];
        smooth_mesh(
            faces.view(),
            verts.view(),
            Filter::Laplacian { lambda: 0.5 },
            Weights::Uniform,
            1,
            false,
            Some(&lock),
            false,
            None,
        );
    }

    #[test]
    #[should_panic(expected = "face references vertex")]
    fn a_face_naming_a_missing_vertex_is_rejected() {
        let faces = array![[0u32, 1, 9]];
        let verts = array![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        smooth(
            &faces,
            &verts,
            Filter::Laplacian { lambda: 0.5 },
            Weights::Uniform,
            1,
        );
    }

    // --------------------------------------------------------------- threading

    /// The `threads` cap changes how the work is split and must not change the answer;
    /// the reductions are chunked deterministically for exactly this reason.
    #[test]
    fn the_thread_count_does_not_change_the_result() {
        let (faces, verts) = uv_sphere(16, 16);
        let run = |t| {
            smooth_mesh(
                faces.view(),
                verts.view(),
                Filter::Taubin {
                    lambda: 0.5,
                    mu: -0.53,
                },
                Weights::Cotangent,
                8,
                false,
                None,
                true,
                t,
            )
        };
        let one = run(Some(1));
        let many = run(Some(4));
        assert_eq!(one.vertices, many.vertices);
        assert_eq!(one.volume, many.volume);
    }
}
