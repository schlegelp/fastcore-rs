//! Phase-level profiler for the NBLAST all-by-all pipeline on real dotprops.
//!
//! Reads the binary dump produced by `_arena/dump_dotprops.py` and times the
//! Delaunay build, the nearest-neighbour descent, and the scoring separately.
//!
//!   cargo run --release --example profile_nblast -p fastcore -- [dump_dir]

use std::path::PathBuf;
use std::time::Instant;

use aann::{graph_from_simplices, PreparedF32, PreparedF64};
use fastcore::nblast::{build_index, load_smat, score_pair};
use ndarray_017::{Array1, Array2};
use rayon::prelude::*;
use shull::delaunay4d;

/// f32 twin of `build_index` (coordinates packed at half the footprint).
fn build_index_f32(points: &[[f64; 3]]) -> PreparedF32 {
    let n = points.len();
    let flat: Vec<f32> = points.iter().flatten().map(|&v| v as f32).collect();
    let arr = Array2::from_shape_vec((n, 3), flat).unwrap();
    let (indptr, indices): (Array1<usize>, Array1<usize>) = match delaunay4d(arr.view()) {
        Ok((tets, _, _)) => {
            let sf: Vec<u64> = tets.iter().flatten().map(|&v| v as u64).collect();
            let simp = Array2::from_shape_vec((tets.len(), 4), sf).unwrap();
            graph_from_simplices(simp.view(), n)
        }
        Err(_) => {
            let mut ip = vec![0usize];
            let mut ix = Vec::new();
            for k in 0..n {
                for j in 0..n {
                    if j != k {
                        ix.push(j);
                    }
                }
                ip.push(ix.len());
            }
            (Array1::from(ip), Array1::from(ix))
        }
    };
    PreparedF32::new(arr.view(), indptr.view(), indices.view())
}

/// Spread the low 21 bits of `x` so each occupies every 3rd bit (Morton/Z-order).
fn part1by2(mut x: u64) -> u64 {
    x &= 0x1f_ffff;
    x = (x | x << 32) & 0x1f00_0000_00ff_ffff;
    x = (x | x << 16) & 0x1f00_00ff_0000_00ff;
    x = (x | x << 8) & 0x100f_00f0_0f00_f00f;
    x = (x | x << 4) & 0x10c3_0c30_c30c_30c3;
    x = (x | x << 2) & 0x1249_2492_4924_9249;
    x
}

/// Permutation that sorts `points` along a 3D Morton curve (cache-locality reorder).
fn morton_perm(points: &[[f64; 3]]) -> Vec<usize> {
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for p in points {
        for d in 0..3 {
            lo[d] = lo[d].min(p[d]);
            hi[d] = hi[d].max(p[d]);
        }
    }
    let mut span = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
    for d in 0..3 {
        if span[d] <= 0.0 {
            span[d] = 1.0;
        }
    }
    let scale = ((1u64 << 21) - 1) as f64;
    let code = |p: &[f64; 3]| -> u64 {
        (0..3)
            .map(|d| {
                let q = (((p[d] - lo[d]) / span[d]) * scale) as u64;
                part1by2(q) << d
            })
            .fold(0u64, |a, b| a | b)
    };
    let mut idx: Vec<usize> = (0..points.len()).collect();
    idx.sort_by_key(|&i| code(&points[i]));
    idx
}

fn read_f64(path: &PathBuf) -> Vec<f64> {
    let bytes = std::fs::read(path).unwrap_or_else(|_| panic!("cannot read {path:?}"));
    bytes
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn read_u64(path: &PathBuf) -> Vec<u64> {
    let bytes = std::fs::read(path).unwrap();
    bytes
        .chunks_exact(8)
        .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn split(flat: Vec<f64>, lens: &[usize]) -> Vec<Vec<[f64; 3]>> {
    let mut out = Vec::with_capacity(lens.len());
    let mut off = 0usize;
    for &n in lens {
        let mut cloud = Vec::with_capacity(n);
        for p in 0..n {
            let b = (off + p) * 3;
            cloud.push([flat[b], flat[b + 1], flat[b + 2]]);
        }
        out.push(cloud);
        off += n;
    }
    out
}

fn secs(f: impl FnOnce()) -> f64 {
    let t = Instant::now();
    f();
    t.elapsed().as_secs_f64()
}

fn main() {
    let dir: PathBuf = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/Users/philipps/Github/fastcore-rs/_arena/dump".into())
        .into();

    let header = read_u64(&dir.join("lengths.bin"));
    let n = header[0] as usize;
    let lens: Vec<usize> = header[1..=n].iter().map(|&x| x as usize).collect();
    let points = split(read_f64(&dir.join("points.bin")), &lens);
    let vects = split(read_f64(&dir.join("vects.bin")), &lens);
    let total_pts: usize = lens.iter().sum();
    let pair_pts: usize = (n - 1) * total_pts; // point-scorings across all ordered pairs
    println!(
        "{n} neurons, {total_pts} points (mean {}), {} threads",
        total_pts / n,
        rayon::current_num_threads()
    );
    println!("{} ordered pairs, {} point-scorings\n", n * (n - 1), pair_pts);

    let smat = load_smat();

    // --- Phase 1: build all indices (parallel wall-clock) ---
    let mut indices: Vec<PreparedF64> = Vec::new();
    let t_build = secs(|| {
        indices = points.par_iter().map(|p| build_index(p)).collect();
    });

    // Serial sub-attribution within build (ratios matter, not absolute wall time).
    let (mut t_del, mut t_graph, mut t_prep) = (0.0, 0.0, 0.0);
    for p in &points {
        let np = p.len();
        let flat: Vec<f64> = p.iter().flatten().copied().collect();
        let arr = Array2::from_shape_vec((np, 3), flat).unwrap();
        let mut tets = None;
        t_del += secs(|| tets = Some(delaunay4d(arr.view())));
        if let Some(Ok((tets, _, _))) = tets {
            let sf: Vec<u64> = tets.iter().flatten().map(|&v| v as u64).collect();
            let simp = Array2::from_shape_vec((tets.len(), 4), sf).unwrap();
            let mut csr = None;
            t_graph += secs(|| csr = Some(graph_from_simplices(simp.view(), np)));
            let (indptr, idx) = csr.unwrap();
            t_prep += secs(|| {
                let _ = PreparedF64::new(arr.view(), indptr.view(), idx.view());
            });
        }
    }

    // --- Phase 2a: NN descent only (parallel) ---
    let mut nn_sum = 0usize;
    let t_nn = secs(|| {
        nn_sum = (0..n * n)
            .into_par_iter()
            .map(|k| {
                let (i, j) = (k / n, k % n);
                if i == j {
                    return 0;
                }
                let (_d, ix) = indices[j].query_prepared(&indices[i], None);
                ix.len()
            })
            .sum();
    });

    // --- Phase 2b: NN + scoring (parallel) = the real inner loop ---
    let self_hits: Vec<f64> = points.iter().map(|p| smat.self_hit(p.len())).collect();
    let mut scores = vec![0.0f32; n * n];
    let t_full = secs(|| {
        scores.par_iter_mut().enumerate().for_each(|(k, out)| {
            let (i, j) = (k / n, k % n);
            if i == j {
                *out = 1.0;
                return;
            }
            let (d, ix) = indices[j].query_prepared(&indices[i], None);
            let raw = score_pair(
                d.as_slice().unwrap(),
                ix.as_slice().unwrap(),
                &vects[i],
                &vects[j],
                None,
                None,
                None,
                &smat,
            );
            *out = (raw / self_hits[i]) as f32;
        });
    });
    std::hint::black_box((nn_sum, &scores));

    // --- f32 NN descent (half-footprint coordinates) for comparison ---
    let mut f32_indices: Vec<PreparedF32> = Vec::new();
    let t_build32 = secs(|| {
        f32_indices = points.par_iter().map(|p| build_index_f32(p)).collect();
    });
    let mut nn32_sum = 0usize;
    let t_nn32 = secs(|| {
        nn32_sum = (0..n * n)
            .into_par_iter()
            .map(|k| {
                let (i, j) = (k / n, k % n);
                if i == j {
                    return 0;
                }
                let (_d, ix) = f32_indices[j].query_prepared(&f32_indices[i], None);
                ix.len()
            })
            .sum();
    });
    std::hint::black_box(nn32_sum);

    // --- Morton-reordered f64 NN (cache locality) ---
    let reordered: Vec<Vec<[f64; 3]>> = points
        .par_iter()
        .map(|p| {
            let perm = morton_perm(p);
            perm.iter().map(|&i| p[i]).collect()
        })
        .collect();
    let mut mz_indices: Vec<PreparedF64> = Vec::new();
    let t_build_mz = secs(|| {
        mz_indices = reordered.par_iter().map(|p| build_index(p)).collect();
    });
    let mut nn_mz_sum = 0usize;
    let t_nn_mz = secs(|| {
        nn_mz_sum = (0..n * n)
            .into_par_iter()
            .map(|k| {
                let (i, j) = (k / n, k % n);
                if i == j {
                    return 0;
                }
                let (_d, ix) = mz_indices[j].query_prepared(&mz_indices[i], None);
                ix.len()
            })
            .sum();
    });
    std::hint::black_box(nn_mz_sum);

    let build_sub = t_del + t_graph + t_prep;
    println!("PHASE                wall[s]    share");
    println!("-------------------------------------");
    println!("build (Delaunay+idx) {t_build:>8.3}   {:>5.1}%", pct(t_build, t_build + t_full));
    println!("  ├─ delaunay4d      {:>8.3}   {:>5.1}%", t_del, pct(t_del, build_sub));
    println!("  ├─ graph_from_simp {:>8.3}   {:>5.1}%", t_graph, pct(t_graph, build_sub));
    println!("  └─ Prepared::new   {:>8.3}   {:>5.1}%", t_prep, pct(t_prep, build_sub));
    println!("  (sub-times are serial sums; ratios indicate relative cost)");
    println!("score loop (NN+score){t_full:>8.3}   {:>5.1}%", pct(t_full, t_build + t_full));
    println!("  ├─ NN descent only {:>8.3}", t_nn);
    println!("  └─ scoring (≈diff)  {:>8.3}", (t_full - t_nn).max(0.0));
    println!("-------------------------------------");
    println!("TOTAL build+loop     {:>8.3}", t_build + t_full);
    println!("\nvariants vs baseline f64 NN descent ({:.3}s):", t_nn);
    println!(
        "  f32 coords         build {:>6.3}   NN {:>7.3}   ({:.2}x)",
        t_build32, t_nn32, t_nn / t_nn32
    );
    println!(
        "  Morton-reordered   build {:>6.3}   NN {:>7.3}   ({:.2}x)",
        t_build_mz, t_nn_mz, t_nn / t_nn_mz
    );
    println!(
        "\nscore throughput: {:.1} M point-scorings/s",
        pair_pts as f64 / t_full / 1e6
    );
}

fn pct(x: f64, total: f64) -> f64 {
    if total > 0.0 {
        100.0 * x / total
    } else {
        0.0
    }
}
