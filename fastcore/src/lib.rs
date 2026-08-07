//! Fast algorithms for neuron skeletons and neuron similarity.
//!
//! This is the core of [`fastcore-rs`](https://github.com/schlegelp/fastcore-rs);
//! the `navis-fastcore` (Python) and `nat.fastcore` (R) packages are bindings on
//! top of it.
//!
//! # Modules
//!
//! - [`dag`] — traversal and geometry on rooted trees (the skeletons): geodesic
//!   distances, linear segments, Strahler index, twig pruning, node
//!   classification, connected components, synapse flow centrality.
//! - [`topo`] — repairing fragmented skeletons: [`topo::stitch_fragments`] finds
//!   the minimal-length edges that reconnect the pieces,
//!   [`topo::reroot_rewire`] re-derives the parent vector afterwards.
//! - [`mesh`] — triangle meshes as vertex graphs: connected components, unique edges,
//!   parallel geodesic search — for distances ([`mesh::geodesic_matrix_graph`]) or for
//!   the route itself ([`mesh::geodesic_path_graph`]) — and the traversal primitives
//!   mesh algorithms need ([`mesh::level_set_components`], [`mesh::contract_vertices`],
//!   [`mesh::minimum_spanning_tree`], [`mesh::geodesic_clusters`]) without building a
//!   graph object first.
//! - [`downsample`] — changing how densely a skeleton is sampled without changing what it
//!   is: dropping nodes ([`downsample::downsample_skeleton`], and the geometry-aware
//!   [`downsample::simplify_rdp`] / [`downsample::simplify_vw`]), adding interpolated ones
//!   at a fixed spacing ([`downsample::resample_skeleton`]), or moving them to take out
//!   the tracing jitter ([`downsample::smooth_skeleton`]). All of it works on the linear
//!   segments between roots, branch points and leafs and leaves those fixed, so the
//!   topology comes out untouched.
//! - [`caps`] — closing the openings a cut leaves in a mesh: finding the boundary
//!   ([`caps::boundary_halfedges`] over a whole mesh, [`caps::exposed_halfedges`] when the
//!   vertices about to go are known), walking it into rings ([`caps::trace_loops`]) and
//!   ear-clipping those shut ([`caps::triangulate_rings`]). Only faces are added, never
//!   vertices, so every index a caller already holds still means what it meant.
//! - [`simplify`] — decimating a triangle mesh by quadric-error edge collapse, and —
//!   the reason it is here rather than in a dependency — reporting which vertex of the
//!   simplified mesh every vertex of the original ended up in, so per-vertex data
//!   (synapses, radii, labels) survives the simplification.
//! - [`smoothing`] — the other half of mesh cleanup: moving vertices to take the noise
//!   out of a surface, by Laplacian, Taubin or HC filtering over a uniform,
//!   inverse-distance or cotangent one-ring. Face array, vertex count and vertex order
//!   all come out untouched, and the volume correction scales about the mesh's own
//!   centroid rather than about the origin — which is what `trimesh` does, and why it
//!   translates a neuron by twice its own diameter.
//! - [`points`] — raw 3D point clouds: [`points::dotprops`] derives the unit tangent
//!   vector and `alpha` of every point's local neighbourhood, which is what [`nblast`]
//!   consumes and what callers previously had to produce with scipy.
//! - [`nblast`] / [`synblast`] — NBLAST neuron similarity, on skeleton points and
//!   on synapses respectively.
//! - [`matches`] — pulling the top matches back out of a score matrix (top-N, an
//!   absolute threshold, or a percentage band around each group's best), without
//!   copying or transposing a matrix that may be tens of GB.
//! - [`linkage`] — hierarchical clustering of a score matrix, fusing symmetrisation,
//!   the similarity→distance transform and condensing into one pass so no `n x n`
//!   temporary is ever materialised, then clustering that buffer in place.
//! - [`cmtk`] — CMTK spatial transforms: read a `*.list` registration (affine +
//!   cubic B-spline warp) and apply it to points, forward or inverse, without
//!   shelling out to CMTK's `streamxform`.
//! - [`elastix`] — Elastix spatial transforms: read a `TransformParameters` file
//!   (and the initial-transform chain hanging off it) and apply it to points,
//!   without shelling out to `transformix` — which also buys an inverse, something
//!   Elastix itself cannot do.
//! - [`tps`] / [`mls`] — landmark-based warps (thin-plate spline, moving least
//!   squares), the fallback when no image registration exists. Both fuse the
//!   distance computation into the accumulation, so the `n_points x n_landmarks`
//!   matrix the reference implementations materialise is never built.
//! - [`threads`] — how wide everything above runs. Most entry points take a
//!   per-call `threads` cap; [`threads::set_num_threads`] sets the process-wide
//!   default, which is what a caller running this library across several
//!   processes wants (see that module for why the default is wrong there).
//!
//! # Representing a tree
//!
//! Trees are index-based: an `ArrayView1<i32>` of parent indices in which **roots
//! are negative**. Mapping arbitrary node IDs onto these indices is the bindings'
//! job, not the core's.

pub mod nblast;

pub mod nblast_knn;

pub mod synblast;

pub mod matches;

pub mod linkage;

pub mod cmtk;

pub mod elastix;

pub mod tps;

pub mod mls;

pub mod dag;

pub mod downsample;

pub mod mesh;

pub mod caps;

pub mod simplify;

pub mod smoothing;

pub mod points;

pub mod topo;

pub mod threads;

mod kdtree;
