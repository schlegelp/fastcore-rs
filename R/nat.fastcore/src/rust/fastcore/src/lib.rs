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
//! - [`mesh`] — connected components of a triangle mesh.
//! - [`nblast`] / [`synblast`] — NBLAST neuron similarity, on skeleton points and
//!   on synapses respectively.
//!
//! # Representing a tree
//!
//! Trees are index-based: an `ArrayView1<i32>` of parent indices in which **roots
//! are negative**. Mapping arbitrary node IDs onto these indices is the bindings'
//! job, not the core's.

pub mod nblast;

pub mod synblast;

pub mod dag;

pub mod mesh;

pub mod topo;
