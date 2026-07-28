//! Profile `mesh::geodesic_matrix_mesh` on synthetic meshes.
//!
//! ```sh
//! cargo run --release --example profile_mesh
//! ```
//!
//! Run single-threaded (the honest algorithmic comparison against scipy, which holds the
//! GIL and so can never be parallelised from Python) with:
//!
//! ```sh
//! RAYON_NUM_THREADS=1 cargo run --release --example profile_mesh
//! ```
//!
//! The reference this replaces is `scipy.sparse.csgraph.dijkstra`, measured on the same
//! sphere meshes at roughly 1.3 ms/source at 10k vertices, 5.7 ms/source at 41k and
//! 26 ms/source at 164k — single-threaded and float64, in every case.

use fastcore::mesh::geodesic_matrix_mesh;
use ndarray::{Array2, ArrayView2};
use std::time::Instant;

/// A UV sphere with `n_lat * n_lon` vertices, triangulated into quads-cut-to-triangles.
///
/// Stands in for a decimated connectomics mesh: closed, manifold, valence ~6, and with
/// edge lengths in a narrow band.
fn uv_sphere(n_lat: usize, n_lon: usize) -> (Array2<u32>, Array2<f64>) {
    let mut coords = Vec::with_capacity(n_lat * n_lon * 3);
    for i in 0..n_lat {
        // Avoid the exact poles so no ring degenerates to a point.
        let theta = std::f64::consts::PI * (i as f64 + 0.5) / n_lat as f64;
        for j in 0..n_lon {
            let phi = 2.0 * std::f64::consts::PI * j as f64 / n_lon as f64;
            coords.push(theta.sin() * phi.cos());
            coords.push(theta.sin() * phi.sin());
            coords.push(theta.cos());
        }
    }

    let id = |i: usize, j: usize| (i * n_lon + j % n_lon) as u32;
    let mut faces = Vec::new();
    for i in 0..n_lat - 1 {
        for j in 0..n_lon {
            faces.extend_from_slice(&[id(i, j), id(i + 1, j), id(i + 1, j + 1)]);
            faces.extend_from_slice(&[id(i, j), id(i + 1, j + 1), id(i, j + 1)]);
        }
    }

    let n_faces = faces.len() / 3;
    (
        Array2::from_shape_vec((n_faces, 3), faces).unwrap(),
        Array2::from_shape_vec((n_lat * n_lon, 3), coords).unwrap(),
    )
}

fn time_it<F: FnMut() -> R, R>(mut f: F, reps: usize) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let t = Instant::now();
        std::hint::black_box(f());
        best = best.min(t.elapsed().as_secs_f64());
    }
    best
}

fn run(label: &str, faces: ArrayView2<u32>, coords: ArrayView2<f64>, n: usize) {
    let sources: Vec<u32> = (0..64).map(|k| (k * n / 64) as u32).collect();
    let few: Vec<u32> = (0..8).map(|k| (k * n / 8) as u32).collect();

    // Weighted, 64 sources, all targets — the workload scipy is used for today.
    let t = time_it(
        || geodesic_matrix_mesh(faces, n, Some(coords), Some(&sources), None, None, None),
        3,
    );
    println!(
        "{label:>10}  V={n:>8}  weighted, 64 src, all tgt : {:8.1} ms  ({:6.3} ms/src)",
        t * 1e3,
        t * 1e3 / 64.0
    );

    // Unweighted takes the BFS path — no heap at all.
    let t_bfs = time_it(
        || geodesic_matrix_mesh(faces, n, None, Some(&sources), None, None, None),
        3,
    );
    println!(
        "{:>10}  {:>10}  unweighted (BFS)          : {:8.1} ms  ({:.1}x vs Dijkstra)",
        "",
        "",
        t_bfs * 1e3,
        t / t_bfs
    );

    // A local query: few sources, tight limit. This is where pruning at relaxation and the
    // touched-list reset earn their keep — scipy cannot early-exit on targets at all.
    let t_lim = time_it(
        || geodesic_matrix_mesh(faces, n, Some(coords), Some(&few), None, Some(0.05), None),
        3,
    );
    println!(
        "{:>10}  {:>10}  8 src, limit=0.05         : {:8.3} ms",
        "",
        "",
        t_lim * 1e3
    );
}

fn main() {
    println!("threads: {}\n", rayon::current_num_threads());
    for (n_lat, n_lon) in [(80, 128), (160, 256), (320, 512)] {
        let (faces, coords) = uv_sphere(n_lat, n_lon);
        let n = n_lat * n_lon;
        run("uv_sphere", faces.view(), coords.view(), n);
        println!();
    }
}
