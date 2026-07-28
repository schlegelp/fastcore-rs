//! Profile `topo::stitch_fragments` on synthetic fragment layouts.
//!
//! Run single-threaded (the honest algorithmic comparison) with:
//!
//! ```sh
//! RAYON_NUM_THREADS=1 cargo run --release --example profile_topo
//! ```
//!
//! The three scenarios stress different regimes:
//!
//! * `interleaved` — fragments share the same volume (a neuron cut into pieces).
//!   Every node has foreign neighbours nearby.
//! * `tiled` — fragments occupy disjoint cells of a grid. Nodes deep inside a
//!   fragment have hundreds of own-fragment neighbours before any foreign one.
//! * `blobs` — two well-separated clouds; the worst case, and the one that used
//!   to go quadratic.

use fastcore::topo::stitch_fragments;
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Instant;

/// `n_frags` clouds sharing one volume: every fragment is spread over the whole
/// cube, so foreign neighbours are always close by.
fn interleaved(n: usize, n_frags: usize) -> (Array2<f64>, Array1<i32>) {
    let mut rng = StdRng::seed_from_u64(0);
    let mut coords = Array2::<f64>::zeros((n, 3));
    let mut comps = Array1::<i32>::zeros(n);
    for i in 0..n {
        for k in 0..3 {
            coords[[i, k]] = rng.gen_range(0.0..1000.0);
        }
        comps[i] = (i % n_frags) as i32;
    }
    (coords, comps)
}

/// `n_frags` compact clouds, each in its own cell of a 3D grid: spatially
/// disjoint, so a node in the middle of a fragment sees only fragment-mates for
/// a long way.
fn tiled(n: usize, n_frags: usize) -> (Array2<f64>, Array1<i32>) {
    let mut rng = StdRng::seed_from_u64(0);
    let side = (n_frags as f64).cbrt().ceil() as usize;
    let mut coords = Array2::<f64>::zeros((n, 3));
    let mut comps = Array1::<i32>::zeros(n);
    for i in 0..n {
        let c = i % n_frags;
        // Cell origin, with a gap between cells (cell extent 100, pitch 300).
        let (cx, cy, cz) = (c % side, (c / side) % side, c / (side * side));
        let base = [cx as f64 * 300.0, cy as f64 * 300.0, cz as f64 * 300.0];
        for k in 0..3 {
            coords[[i, k]] = base[k] + rng.gen_range(0.0..100.0);
        }
        comps[i] = c as i32;
    }
    (coords, comps)
}

/// Two clouds of `m` points each, extent 100, centres `sep` apart.
fn blobs(m: usize, sep: f64) -> (Array2<f64>, Array1<i32>) {
    let mut rng = StdRng::seed_from_u64(0);
    let mut coords = Array2::<f64>::zeros((2 * m, 3));
    let mut comps = Array1::<i32>::zeros(2 * m);
    for i in 0..2 * m {
        let shift = if i < m { 0.0 } else { sep };
        coords[[i, 0]] = shift + rng.gen_range(0.0..100.0);
        coords[[i, 1]] = rng.gen_range(0.0..100.0);
        coords[[i, 2]] = rng.gen_range(0.0..100.0);
        comps[i] = if i < m { 0 } else { 1 };
    }
    (coords, comps)
}

fn time(label: String, coords: &Array2<f64>, comps: &Array1<i32>) {
    let t = Instant::now();
    let bridges = stitch_fragments(&coords.view(), &comps.view(), &None, f64::INFINITY);
    let elapsed = t.elapsed().as_secs_f64();
    let total: f64 = bridges.iter().map(|&(_, _, d)| d as f64).sum();
    println!(
        "{label:<40} {elapsed:>9.3}s   {:>6} bridges   total {total:.1}",
        bridges.len()
    );
}

fn main() {
    println!(
        "threads: {}\n",
        std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "default".into())
    );

    println!("-- blobs: two clouds, 4k nodes each, varying separation --");
    for sep in [50.0, 100.0, 200.0, 1_000.0, 100_000.0] {
        let (c, k) = blobs(4_000, sep);
        time(format!("  sep = {sep}"), &c, &k);
    }

    println!("\n-- blobs: separation 1e5, varying size --");
    for m in [500usize, 1_000, 2_000, 4_000, 8_000, 16_000] {
        let (c, k) = blobs(m, 100_000.0);
        time(format!("  {m} nodes/fragment"), &c, &k);
    }

    println!("\n-- tiled: disjoint fragments on a grid --");
    for (n, f) in [
        (50_000usize, 250usize),
        (223_000, 1_050),
        (1_000_000, 2_000),
    ] {
        let (c, k) = tiled(n, f);
        time(format!("  {n} nodes, {f} fragments"), &c, &k);
    }

    println!("\n-- interleaved: fragments share one volume --");
    for (n, f) in [
        (100_000usize, 1_000usize),
        (1_330_000, 2_001),
        (5_330_000, 2_001),
    ] {
        let (c, k) = interleaved(n, f);
        time(format!("  {n} nodes, {f} fragments"), &c, &k);
    }
}
