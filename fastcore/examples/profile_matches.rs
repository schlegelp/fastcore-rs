//! Bandwidth profiler for match extraction on a synthetic score matrix.
//!
//! Every criterion here is a fixed number of streaming passes over the matrix, so the
//! number that matters is **effective GB/s**, not wall time: if it drops well below what
//! the machine can stream, the traversal has stopped being cache-friendly. That is the
//! regression this harness exists to catch — in particular for `axis=1`, where the naive
//! implementation reads a 64-byte cache line to consume 4 bytes.
//!
//!   cargo run --release --example profile_matches -p fastcore -- [n] [n_cols]

use std::time::Instant;

use fastcore::matches::{count_matches, matches_above, top_matches, Criterion, MatchAxis, MatchOpts};
use ndarray::Array2;

/// Passes over the matrix each criterion makes, so GB/s reflects real traffic.
fn passes(crit: Option<Criterion>, axis: MatchAxis) -> f64 {
    match (crit, axis) {
        (None, _) => 1.0,                                       // top-N: count + fill in one
        (Some(Criterion::Threshold(_)), _) => 2.0,              // count, fill
        (Some(Criterion::Percentage(_)), MatchAxis::Rows) => 2.0, // extremum+count fuse in L2
        (Some(Criterion::Percentage(_)), MatchAxis::Cols) => 3.0, // extremum, count, fill
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n_rows: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(20_000);
    let n_cols: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(n_rows);

    let bytes = (n_rows * n_cols * std::mem::size_of::<f32>()) as f64;
    println!(
        "{n_rows} x {n_cols} f32 = {:.2} GB, {} threads\n",
        bytes / 1e9,
        rayon::current_num_threads()
    );

    // A deterministic spread over [-1, 1); the actual distribution only shifts how often
    // the top-N buffer accepts, which is noise next to the compare stream.
    let mut s = 0x9E37_79B9_7F4A_7C15u64;
    let m = Array2::from_shape_fn((n_rows, n_cols), |_| {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        // `s >> 40` keeps 24 bits; 2^24 normalises to [0, 1), then rescale to [-1, 1).
        ((s >> 40) as f32 / 16_777_216.0) * 2.0 - 1.0
    });
    let v = m.view();

    // Warm up: the matrix was just written by the generator, so the first *parallel* read
    // otherwise pays for page/TLB warming and reports a time that has nothing to do with
    // the kernel. Without this the first row in the table reads as a 3-4x regression.
    let _ = top_matches(v, 1, MatchOpts::default()).unwrap();

    let run = |label: &str, axis: MatchAxis, crit: Option<Criterion>| {
        let o = MatchOpts {
            axis,
            ..Default::default()
        };
        let t = Instant::now();
        let found = match crit {
            None => {
                let r = top_matches(v, 10, o).unwrap();
                r.indices.len()
            }
            Some(c) => {
                let r = matches_above(v, c, o).unwrap();
                r.indices.len()
            }
        };
        let el = t.elapsed().as_secs_f64();
        let gbs = bytes * passes(crit, axis) / el / 1e9;
        println!("{label:<28} {el:>7.3} s   {gbs:>6.1} GB/s   {found:>12} matches");
    };

    for &axis in &[MatchAxis::Rows, MatchAxis::Cols] {
        let a = if axis == MatchAxis::Rows { "rows" } else { "cols" };
        run(&format!("top-N (n=10), {a}"), axis, None);
        run(&format!("threshold=0.9, {a}"), axis, Some(Criterion::Threshold(0.9)));
        run(&format!("percentage=0.05, {a}"), axis, Some(Criterion::Percentage(0.05)));
    }

    // The counting half alone: what you'd run to size a result before allocating it.
    let t = Instant::now();
    let c = count_matches(v, Criterion::Threshold(0.9), MatchOpts::default()).unwrap();
    let el = t.elapsed().as_secs_f64();
    println!(
        "{:<28} {el:>7.3} s   {:>6.1} GB/s   {:>12} matches",
        "count only, rows",
        bytes / el / 1e9,
        c.iter().sum::<i64>()
    );
}
