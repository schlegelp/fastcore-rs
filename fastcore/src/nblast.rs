use ndarray::{Array1, Array2};
use ndarray::parallel::prelude::*;
use kiddo::SquaredEuclidean;
use kiddo::immutable::float::kdtree::ImmutableKdTree;

// Get the nearest neighbor for each query point
pub fn top_nn(
    pos_array: &Vec<[f64; 3]>,
    query_array: &Vec<[f64; 3]>,
    parallel: bool,
) -> Vec<(f64, usize)> {

    // Run the query
    let results: Vec<(f64, usize)> = if parallel {
        query_array
            .par_iter()
            .map(|query_row| bosque::tree::nearest_one(&pos_array, &query_row))
            .collect()
    } else {
        query_array
            .iter()
            .map(|query_row| bosque::tree::nearest_one(&pos_array, &query_row))
            .collect()
    };

    results
}

// Get top nearest neighbors but split results into distances and indices
pub fn top_nn_split(
    pos_array: &Vec<[f64; 3]>,
    query_array: &Vec<[f64; 3]>,
    parallel: bool,
) -> (Vec<f64>, Vec<usize>) {
    let results = top_nn(pos_array, query_array, parallel);

    // Unzip the results into two vectors, one for distances and one for indices
    // I have checked whether unzipping afterwards instead of right away makes
    // much of a difference and it doesn't seem do so.
    let (distances, indices): (Vec<f64>, Vec<usize>) = results.into_iter().unzip();

    (distances, indices)
}

// Load and parse the NBLAST scoring matrix
pub fn load_smat() -> (Array2<f64>, Vec<[f64; 2]>, Vec<[f64; 2]>) {
    // Get the current filepath should be src/nblast.rs
    // The mat should be in the same directory as the module
    // let filepath = PathBuf::from("../fastcore/fastcore.data/smat_fcwb.csv");
    // println!("smat file path: {:?}", filepath);

    // This statically includes the smat file as a byte array
    let data = include_bytes!("../fastcore.data/smat_fcwb.csv");

    // let mut rdr = csv::Reader::from_path(filepath).unwrap();
    let mut rdr = csv::Reader::from_reader(data.as_ref());

    let mut smat: Vec<Vec<f64>> = vec![];
    let mut bins_vec: Vec<[f64; 2]> = vec![];
    let mut bins_dist: Vec<[f64; 2]> = vec![];

    // Read the header row
    if let Ok(row) = rdr.headers() {
        bins_vec = row
            .iter()
            .skip(1)
            .map(|x| {
                let bounds: Vec<&str> = x.split(',').collect();
                let left_bound: f64 = bounds[0].trim_start_matches('(').parse().unwrap();
                let right_bound: f64 = bounds[1].trim_end_matches(']').parse().unwrap();
                [left_bound, right_bound]
            })
            .collect();
    }

    // Read the remaining rows
    for result in rdr.records() {
        let record = result.unwrap();
        let row: Vec<f64> = record
            .iter()
            .skip(1)
            .map(|x| x.parse::<f64>().unwrap())
            .collect();
        let index_value = record.get(0).unwrap();
        let bounds: Vec<&str> = index_value.split(',').collect();
        let left_bound: f64 = bounds[0].trim_start_matches('(').parse().unwrap();
        let right_bound: f64 = bounds[1].trim_end_matches(']').parse().unwrap();
        bins_dist.push([left_bound, right_bound]);
        smat.push(row);
    }

    // We're converting the smat to an ndarray here because the vector of vectors
    // is not guaranteed to be contiguous in memory which could make it slower
    // to access.
    let smat_array: Array2<f64> = Array2::from_shape_vec((smat.len(), smat[0].len()), smat.into_iter().flatten().collect()).unwrap();

    (smat_array, bins_vec, bins_dist)
}

// Calculate NBLAST score from distances and vector dotproducts
fn calc_nblast_score(
    dists: &Vec<f64>,
    dotprods: &Vec<f64>,
    smat: &Array2<f64>,
    bins_vec: &Vec<[f64; 2]>,
    bins_dist: &Vec<[f64; 2]>,
) -> f64 {
    let mut dist_binned: Vec<usize> = vec![0; dists.len()];
    let mut dp_binned: Vec<usize> = vec![0; dotprods.len()];
    let mut score: f64 = 0.0;

    // Bin distances
    for (i, dist) in dists.iter().enumerate() {
        for (j, bounds) in bins_dist.iter().rev().enumerate() {
            if dist >= &bounds[0] {
                dist_binned[i] = bins_dist.len() - j - 1;
                break;
            }
        }
    }

    // Bin dotproducts
    for (i, dp) in dotprods.iter().enumerate() {
        for (j, bounds) in bins_vec.iter().rev().enumerate() {
            if dp >= &bounds[0] {
                dp_binned[i] = j;
                break;
            }
        }
    }

    // Sum up the scores
    for (dist, dotprod) in dist_binned.iter().zip(dp_binned) {
        score += smat[(*dist, dotprod)];
    }
    score
}

// Calculate NBLAST score for a self hit
fn calc_self_hit(
    n_nodes: usize,
    smat: &Array2<f64>
) -> f64 {
    let mut score: f64 = 0.0;

    // Self-hit means 0 distance and perfectly aligned vectors
    // I.e. the top right corner of the scoring matrix
    let k = smat.shape()[1] - 1;
    let max_score = smat[(0, k)];
    score += max_score * n_nodes as f64;

    score
}

// Calculate dotproducts for nearest neighbour vectors
fn calc_dotproducts(
    query_vec: &Vec<[f64; 3]>,
    target_vec: &Vec<[f64; 3]>,
    nn_indices: &Vec<usize>,
) -> Vec<f64> {
    let mut dotprods: Vec<f64> = vec![0.0; query_vec.len()];
    for (i, nn) in nn_indices.iter().enumerate() {
        dotprods[i] = query_vec[i][0] * target_vec[*nn][0]
            + query_vec[i][1] * target_vec[*nn][1]
            + query_vec[i][2] * target_vec[*nn][2];
        dotprods[i] = dotprods[i].abs()
    }
    dotprods
}

// Run NBLAST for a single query - target pair
pub fn nblast_single(
    query_array: Vec<[f64; 3]>,
    query_vec: &Vec<[f64; 3]>,
    mut target_array: Vec<[f64; 3]>,
    target_vec: &Vec<[f64; 3]>,
    normalize: bool,
    parallel: bool,
    make_tree: bool
) -> f64 {
    if make_tree {
        bosque::tree::build_tree(&mut target_array);
    }

    // Get the nearest neighbor for each query point
    let (distances, indices) = top_nn_split(&target_array, &query_array, parallel);

    // Calculate dotproducts for nearest neighbor vectors
    let dotprods = calc_dotproducts(&query_vec, &target_vec, &indices);

    // Calculate the nblast score
    let (smat, bins_vec, bins_dist) = load_smat();
    let score = calc_nblast_score(
        &distances,
        &dotprods,
        &smat,
        &bins_vec,
        &bins_dist,
    );

    if normalize {
        // Calculate the self hit score
        let self_hit = calc_self_hit(query_vec.len(), &smat);
        // Normalize the score
        score / self_hit
    } else {
        score
    }
}

// Run all-by-all NBLAST query using bosque as the backend
pub fn nblast_allbyall_bosque(
    points: Vec<Vec<[f64; 3]>>,
    vecs: Vec<Vec<[f64; 3]>>,
) -> Array2<f32> {
    // Prepare the output array
    // Note we're using a 1d array here because otherwise we end up running into
    // issues with parallelization. We will reshape it to 2d at the very end.
    // Also note that we fill the array with values from 0 to n^2 - 1 so that
    // we can use the initial value to calculate the row and column indices
    // as we iterate over each cell of the array.
    let mut dists: Array1<f32> = Array1::from_iter((0..(points.len()*points.len())).map(|x| x as f32));

    // Load the scoring matrix
    let (smat, bins_vec, bins_dist) = load_smat();

    rayon::ThreadPoolBuilder::new().num_threads(10).build_global().unwrap();

    // Go over each cell of the matrix and run a single query-target NBLAST
    dists.par_map_inplace(|x| {
        let i = *x as usize;
        let row_ix = i / points.len();  // this is already floor division
        let col_ix = i - (row_ix * points.len());

        // Get the nearest neighbor for each query point
        // Ideas for a potential speed ups:
        // 1. Avoid splitting results in the top_nn function and instead return a single vector of tuples
        // 2. Have only one loop over the results that calculates both dotprods as well as the final score
        // Scratch that: tried and didn't seem to be making much of a difference
        // Notes:
        // - the vast majority of time (99.9%) is spend in this function -> we need to see if we can optimize this
        let (distances, indices) = top_nn_split(&points[row_ix], &points[col_ix], false);

        // Calculate dotproducts for nearest neighbor vectors
        let dotprods = calc_dotproducts(&vecs[col_ix], &vecs[row_ix], &indices);

        let score = calc_nblast_score(
            &distances,
            &dotprods,
            &smat,
            &bins_vec,
            &bins_dist,
        ) as f32;

        *x = score;
    });

    dists.into_shape((points.len(), points.len())).unwrap()
}

// Run all-by-all NBLAST query using kiddo as NN backend
pub fn nblast_allbyall_kiddo(
    points: Vec<Vec<[f64; 3]>>,
    vecs: Vec<Vec<[f64; 3]>>,
) -> Array2<f32> {
    // Prepare the output array
    // Note we're using a 1d array here because otherwise we end up running into
    // issues with parallelization. We will reshape it to 2d at the very end.
    // Also note that we fill the array with values from 0 to n^2 - 1 so that
    // we can use the initial value to calculate the row and column indices
    // as we iterate over each cell of the array.
    let mut dists: Array1<f32> = Array1::from_iter((0..(points.len()*points.len())).map(|x| x as f32));

    // Load the scoring matrix
    let (smat, bins_vec, bins_dist) = load_smat();

    // For some reason we have to convert points like this
    // let points2: Vec[] = points.iter().map(|p| vec![p[0], p[1], p[2]]).collect();

    // println!("Trying to make one tree!");
    // let entries = vec![
    // [0f64, 0f64, 0f64],
    // [1f64, 1f64, 1f64],
    // [2f64, 2f64, 2f64],
    // [3f64, 3f64, 3f64]
    // ];
    // let p = vec![points[0][0], points[0][1], points[0][2]];
    // println!("p: {:?}", p);
    // let p2 = points[0].as_slice().to_vec();
    // println!("p2: {:?}", p2);
    // let tree: ImmutableKdTree<f64, usize, 3, 32> = ImmutableKdTree::new_from_slice(&p);
    // println!("Made one tree!");

    // Prepare the trees
    let trees: Vec<ImmutableKdTree<f64, usize, 3, 32>> = points
        .iter()
        .map(|point_set| ImmutableKdTree::new_from_slice(point_set.as_slice()))
        .collect();

    // println!("Trees: {:?}", trees.len());

    // Go over each cell of the matrix and run a single query-target NBLAST
    dists.par_map_inplace(|x| {
        let i = *x as usize;
        let row_ix = i / points.len();  // this is already floor division
        let col_ix = i - (row_ix * points.len());

        // Get the target tree
        //let tgt = trees.get(row_ix).unwrap();
        let tgt = &trees[row_ix];

        // Get the nearest neighbor for each query point
        let results: Vec<(usize, f64)> = points
            .get(col_ix)
            .unwrap()
            .iter()
            .map(|p| {
                let nn = tgt.nearest_one::<SquaredEuclidean>(p);
                (nn.item as usize, nn.distance.sqrt())
            })
            .collect();

        // Extract the distances and indices from `results`
        let (indices, distances): (Vec<usize>, Vec<f64>) = results.into_iter().unzip();

        // Calculate dotproducts for nearest neighbor vectors
        let dotprods = calc_dotproducts(&vecs[col_ix], &vecs[row_ix], &indices);

        let score = calc_nblast_score(
            &distances,
            &dotprods,
            &smat,
            &bins_vec,
            &bins_dist,
        ) as f32;

        *x = score;
    });

    dists.into_shape((points.len(), points.len())).unwrap()
}
