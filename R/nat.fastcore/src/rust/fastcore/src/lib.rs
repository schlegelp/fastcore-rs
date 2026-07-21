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
//!   parallel geodesic search, and the traversal primitives mesh algorithms need
//!   ([`mesh::level_set_components`], [`mesh::contract_vertices`],
//!   [`mesh::minimum_spanning_tree`]) without building a graph object first.
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

pub mod mesh;

pub mod topo;
