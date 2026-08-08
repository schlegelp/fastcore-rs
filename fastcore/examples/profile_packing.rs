//! Profile the collage primitives: `raster::rasterize_segments` and `packing::pack_masks`.
//!
//! ```sh
//! cargo run --release --example profile_packing
//! ```
//!
//! Set `FASTCORE_PROFILE_THREADS=1,2,4,8,14` to print a scaling table instead of a single
//! run.

use fastcore::packing::{pack_masks, pack_rectangles};
use fastcore::raster::{rasterize_segments, rasterize_segments_batch, Bitmap, RasterOptions};
use ndarray::{Array2, ArrayView2};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Instant;

/// A random branching tree in 2-D, standing in for a neuron's arbor.
fn arbor(n: usize, rng: &mut StdRng) -> (Array2<f64>, Array2<u32>) {
    let mut coords: Vec<f64> = vec![0.0, 0.0];
    let mut edges: Vec<u32> = Vec::new();
    // Growth fronts: a walk that occasionally splits, which is what gives an arbor its
    // long thin branches rather than a blob.
    let mut fronts: Vec<(usize, f64)> = vec![(0, rng.gen_range(0.0..std::f64::consts::TAU))];
    while coords.len() / 2 < n {
        let k = rng.gen_range(0..fronts.len());
        let (parent, dir) = fronts[k];
        let dir = dir + rng.gen_range(-0.35..0.35);
        let step = rng.gen_range(0.004..0.02);
        let (px, py) = (coords[parent * 2], coords[parent * 2 + 1]);
        let child = coords.len() / 2;
        coords.push(px + step * dir.cos());
        coords.push(py + step * dir.sin());
        edges.push(child as u32);
        edges.push(parent as u32);
        fronts[k] = (child, dir);
        if rng.gen_bool(0.04) {
            fronts.push((child, dir + rng.gen_range(0.4..1.2)));
        }
        if fronts.len() > 1 && rng.gen_bool(0.02) {
            fronts.swap_remove(k);
        }
    }
    let n = coords.len() / 2;
    let e = edges.len() / 2;
    (
        Array2::from_shape_vec((n, 2), coords).unwrap(),
        Array2::from_shape_vec((e, 2), edges).unwrap(),
    )
}

/// A closed wobbly outline with `n` vertices — the "solid shape" case, which is what a
/// projected mesh looks like to the rasteriser once its faces are down to unique edges.
fn blob(n: usize, rng: &mut StdRng) -> (Array2<f64>, Array2<u32>) {
    let mut coords = Vec::with_capacity(n * 2);
    let mut r = 1.0f64;
    for i in 0..n {
        let a = std::f64::consts::TAU * i as f64 / n as f64;
        r = (r + rng.gen_range(-0.02..0.02)).clamp(0.6, 1.4);
        coords.push(r * a.cos());
        coords.push(r * a.sin());
    }
    let edges: Vec<u32> = (0..n)
        .flat_map(|i| [i as u32, ((i + 1) % n) as u32])
        .collect();
    (
        Array2::from_shape_vec((n, 2), coords).unwrap(),
        Array2::from_shape_vec((n, 2), edges).unwrap(),
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

fn ms(t: f64) -> String {
    format!("{:8.2}", t * 1e3)
}

fn main() {
    let threads: Vec<usize> = std::env::var("FASTCORE_PROFILE_THREADS")
        .ok()
        .map(|s| s.split(',').map(|x| x.trim().parse().unwrap()).collect())
        .unwrap_or_else(|| vec![0]);
    let t = |n: usize| if n == 0 { None } else { Some(n) };

    let mut rng = StdRng::seed_from_u64(0);
    // Enough shapes to leave the page around 70% ink, i.e. crowded but still packable -
    // which is where the scan actually has to work for its answers.
    let n_shapes: usize = 120;
    let arbors: Vec<(Array2<f64>, Array2<u32>)> =
        (0..n_shapes).map(|_| arbor(2_000, &mut rng)).collect();
    let blobs: Vec<(Array2<f64>, Array2<u32>)> =
        (0..n_shapes).map(|_| blob(4_000, &mut rng)).collect();
    let one_big = blob(300_000, &mut rng);

    fn views(v: &[(Array2<f64>, Array2<u32>)]) -> Vec<(ArrayView2<'_, f64>, ArrayView2<'_, u32>)> {
        v.iter().map(|(c, e)| (c.view(), e.view())).collect()
    }
    let arbor_v = views(&arbors);
    let blob_v = views(&blobs);

    // 1300 x 980 px, i.e. a 13 x 9.8 unit page at 100 px/unit - the docstring's workload.
    let page = (980usize, 1300usize);
    let scale = 100.0;
    let pad = 2;

    let base = RasterOptions {
        scale,
        pad,
        fill: false,
        turn: 0,
    };
    let plain = RasterOptions { pad: 0, ..base };
    let filled = RasterOptions { fill: true, ..base };

    println!("== rasterise, one shape at a time (not batched) ==");
    println!("                                            total    per shape");
    for (label, shapes, opts) in [
        ("arbors (2k edges, no fill, no pad)", &arbor_v, &plain),
        ("arbors (2k edges, pad 2)", &arbor_v, &base),
        ("blobs  (4k edges, pad 2)", &blob_v, &base),
        ("blobs  (4k edges, pad 2, fill)", &blob_v, &filled),
    ] {
        let dt = time_it(
            || {
                shapes
                    .iter()
                    .map(|(c, e)| rasterize_segments(c.view(), e.view(), opts))
                    .collect::<Vec<_>>()
            },
            3,
        );
        let n = shapes.len();
        println!("{n:>5} {label:35} {} {}", ms(dt), ms(dt / n as f64));
    }
    println!();
    println!("== one big shape, decomposed ==");
    // Each option adds one stage, so the differences say what each stage costs.
    for (label, opts) in [
        ("walk only          ", plain),
        ("walk + fill        ", RasterOptions { pad: 0, fill: true, ..base }),
        ("walk + dilate      ", base),
        ("walk + fill + dilate", filled),
    ] {
        let dt = time_it(
            || rasterize_segments(one_big.0.view(), one_big.1.view(), &opts),
            5,
        );
        println!("  300k edges, {label} {}", ms(dt));
    }
    let m = rasterize_segments(one_big.0.view(), one_big.1.view(), &plain);
    println!(
        "  its bitmap is {}x{} px",
        m.height(),
        m.width()
    );

    println!();
    println!("== rasterise, batched ==");
    for n in &threads {
        let a = time_it(
            || rasterize_segments_batch(&arbor_v, &base, None, t(*n)),
            3,
        );
        let b = time_it(
            || rasterize_segments_batch(&blob_v, &filled, None, t(*n)),
            3,
        );
        let c = time_it(
            || {
                rasterize_segments_batch(
                    std::slice::from_ref(&(one_big.0.view(), one_big.1.view())),
                    &filled,
                    None,
                    t(*n),
                )
            },
            3,
        );
        println!(
            "threads {:>3}: {n_shapes} arbors {}   {n_shapes} blobs {}   1 big blob {}",
            n,
            ms(a),
            ms(b),
            ms(c)
        );
    }

    // What a real neuron list looks like: a few big meshes among many small ones. The batch
    // splits by shape, so without a shape also splitting its own walk the biggest one is a
    // floor on the wall clock however many cores there are.
    println!();
    println!("== rasterise, batched, uneven sizes ==");
    let mixed: Vec<(Array2<f64>, Array2<u32>)> = (0..24)
        .map(|i| blob(if i < 4 { 300_000 } else { 4_000 }, &mut rng))
        .collect();
    let mixed_v = views(&mixed);
    for n in &threads {
        let dt = time_it(|| rasterize_segments_batch(&mixed_v, &filled, None, t(*n)), 3);
        println!("threads {:>3}: 4 big + 20 small {}", n, ms(dt));
    }

    // Masks for the packing runs, upright plus a quarter turn.
    let arbor_masks: Vec<Vec<Bitmap>> = {
        let up = rasterize_segments_batch(&arbor_v, &base, None, None);
        let turned =
            rasterize_segments_batch(&arbor_v, &RasterOptions { turn: 1, ..base }, None, None);
        up.into_iter().zip(turned).map(|(a, b)| vec![a, b]).collect()
    };
    let one_variant: Vec<Vec<Bitmap>> = arbor_masks
        .iter()
        .map(|v| vec![v[0].clone()])
        .collect();

    let ink: usize = arbor_masks.iter().map(|v| v[0].count_ones()).sum();
    println!();
    println!(
        "page {}x{} px, {n_shapes} shapes, {:.1}% of the page is ink",
        page.0,
        page.1,
        100.0 * ink as f64 / (page.0 * page.1) as f64
    );

    // A radial cost surface, as a masked collage uses.
    let cost: Vec<f64> = (0..page.0 * page.1)
        .map(|c| {
            let (y, x) = ((c / page.1) as f64, (c % page.1) as f64);
            (y - 490.0).hypot(x - 650.0)
        })
        .collect();

    println!();
    println!("== pack_masks ==");
    for n in &threads {
        let one = time_it(
            || pack_masks(&one_variant, page, None, None, true, t(*n)),
            3,
        );
        let two = time_it(
            || pack_masks(&arbor_masks, page, None, None, true, t(*n)),
            3,
        );
        let cost_one = time_it(
            || pack_masks(&one_variant, page, None, Some(&cost), true, t(*n)),
            3,
        );
        let by_cost = time_it(
            || pack_masks(&arbor_masks, page, None, Some(&cost), true, t(*n)),
            3,
        );
        println!(
            "threads {:>3}: bottom-up {} / {}   cost surface {} / {}   (1 variant / 2)",
            n,
            ms(one),
            ms(two),
            ms(cost_one),
            ms(by_cost)
        );
    }

    println!();
    println!("== pack_rectangles ==");
    let sizes: Vec<(f64, f64)> = arbor_masks
        .iter()
        .map(|v| (v[0].width() as f64, v[0].height() as f64))
        .collect();
    let dt = time_it(
        || pack_rectangles(&sizes, (1300.0, 980.0), true, true, None),
        3,
    );
    println!("{n_shapes} rectangles, rotation allowed: {}", ms(dt));

    println!();
    println!("== Bitmap ops ==");
    let big = {
        let mut b = Bitmap::new(page.0, page.1);
        let mut r = StdRng::seed_from_u64(1);
        for _ in 0..40_000 {
            b.set(r.gen_range(0..page.0), r.gen_range(0..page.1));
        }
        b
    };
    let flat = big.to_bools();
    println!(
        "page-sized: to_bools {}  from_bools {}  dilate_disk(2) {}  fill_holes {}",
        ms(time_it(|| big.to_bools(), 5)),
        ms(time_it(|| Bitmap::from_bools(&flat, page.0, page.1), 5)),
        ms(time_it(|| big.dilate_disk(2), 5)),
        ms(time_it(
            || {
                let mut c = big.clone();
                c.fill_holes();
                c
            },
            5
        )),
    );
}
