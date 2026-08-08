//! Work small enough to run on the calling thread must not touch the global rayon pool.
//!
//! [`fastcore::threads::set_num_threads`] can size that pool at most once per process, and
//! whatever builds it first wins — including a bare `rayon::current_num_threads()`. A
//! caller running this library under a process pool sizes the pool at worker start-up; if
//! some earlier small call has already built it at "every core the machine has", that
//! worker is stuck with 224 threads it did not ask for, which is the situation
//! `fastcore::threads` exists to prevent.
//!
//! This lives in `tests/` rather than beside the code because each integration test file is
//! its own binary — and a process that has already built the pool cannot un-build it, so
//! this can only be asserted once.

use fastcore::raster::{rasterize_segments, RasterOptions};
use fastcore::threads::set_num_threads;
use ndarray::array;

#[test]
fn a_short_walk_does_not_build_the_global_pool() {
    let coords = array![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]];
    let edges = array![[0u32, 1u32], [1, 2]];
    let opts = RasterOptions {
        scale: 4.0,
        ..Default::default()
    };

    let out = rasterize_segments(coords.view(), edges.view(), &opts);
    assert!(out.count_ones() > 0, "the shape should have been drawn");

    // Fails if the call above reached for rayon, in which case the pool is already running
    // at the machine's core count and this cannot resize it.
    set_num_threads(4).expect("rasterising a 2-edge shape must leave the pool unbuilt");
}
