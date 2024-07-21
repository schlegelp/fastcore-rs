use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2};
use ndarray::Array2;
use pyo3::prelude::*;
use pyo3::prelude::{PyResult, Python};

use fastcore::nblast::{top_nn_split, nblast_allbyall_bosque, nblast_allbyall_kiddo, nblast_single};

// Use bosque to run a nearest neighbor search
// Note: in my tests this was not faster than pykdtree. That said, there might
// still be mileage if we can use it threaded instead of multi-process.
// See https://pyo3.rs/v0.20.2/parallelism on how to release the GIL.
#[pyfunction]
#[pyo3(signature = (pos, query, parallel=true), name = "top_nn")]
pub fn top_nn_py<'py>(
    py: Python<'py>,
    pos: PyReadonlyArray2<f64>,
    query: PyReadonlyArray2<f64>,
    parallel: bool,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<usize>>)> {
    // Turn the input arrays into vectors of arrays (this is what bosque expects)
    let mut pos_array: Vec<[f64; 3]> = pos
        .as_array()
        .rows()
        .into_iter()
        .map(|row| [row[0], row[1], row[2]])
        .collect();

    // Turn the query array into a vector of arrays (this is what bosque expects)
    let query_array: Vec<[f64; 3]> = query
        .as_array()
        .rows()
        .into_iter()
        .map(|row| [row[0], row[1], row[2]])
        .collect();

    bosque::tree::build_tree(&mut pos_array);

    // Unzip the results into two vectors, one for distances and one for indices
    let (distances, indices): (Vec<f64>, Vec<usize>) =
        top_nn_split(&pos_array, &query_array, parallel);

    // Turn the vectors into numpy arrays
    let distances_py: Py<PyArray1<f64>> = distances.into_pyarray(py).to_owned();
    let indices_py: Py<PyArray1<usize>> = indices.into_pyarray(py).to_owned();

    Ok((distances_py, indices_py))
}

// Run a single NBLAST query
#[pyfunction]
#[pyo3(name = "nblast_single")]
pub fn nblast_single_py<'py>(
    query_array_py: PyReadonlyArray2<f64>,
    query_vec_py: PyReadonlyArray2<f64>,
    target_array_py: PyReadonlyArray2<f64>,
    target_vec_py: PyReadonlyArray2<f64>,
    parallel: bool,
) -> f64 {
    // Turn the input arrays into vectors of arrays (this is what bosque expects)
    let query_array: Vec<[f64; 3]> = query_array_py
        .as_array()
        .rows()
        .into_iter()
        .map(|row| [row[0], row[1], row[2]])
        .collect();
    let query_vec: Vec<[f64; 3]> = query_vec_py
        .as_array()
        .rows()
        .into_iter()
        .map(|row| [row[0], row[1], row[2]])
        .collect();
    let target_array: Vec<[f64; 3]> = target_array_py
        .as_array()
        .rows()
        .into_iter()
        .map(|row| [row[0], row[1], row[2]])
        .collect();
    let target_vec: Vec<[f64; 3]> = target_vec_py
        .as_array()
        .rows()
        .into_iter()
        .map(|row| [row[0], row[1], row[2]])
        .collect();

    let score = nblast_single(
        query_array,
        &query_vec,
        target_array,
        &target_vec,
        true,
        parallel,
        true
    );

    score
}

// Run an all-by-all NBLAST query
#[pyfunction]
#[pyo3(name = "nblast_allbyall")]
pub fn nblast_allbyall_py<'py>(
    py: Python<'py>,
    points_py: Vec<PyReadonlyArray2<f64>>,
    vecs_py: Vec<PyReadonlyArray2<f64>>,
    backend: &str,
) -> &'py PyArray2<f32> {
    // Convert points_py to a vector of arrays and build the trees right away
    let mut points: Vec<Vec<[f64; 3]>> = vec![];
    for point_set in points_py.iter() {
        let mut tree: Vec<[f64; 3]> = point_set.as_array().rows().into_iter().map(|row| [row[0], row[1], row[2]]).collect();
        bosque::tree::build_tree(&mut tree);
        points.push(tree);
    }
    // Convert vecs_py to a vector of arrays
    let vecs: Vec<Vec<[f64; 3]>> = vecs_py
        .iter()
        .map(|x| {
            x.as_array()
                .rows()
                .into_iter()
                .map(|row| [row[0], row[1], row[2]])
                .collect()
        })
        .collect();
    // Run the NBLAST
    //let dists = nblast_allbyall(points, vecs);
    let dists: Array2<f32>;
    if backend == "bosque" {
        dists = nblast_allbyall_bosque(points, vecs);
    } else if backend == "kiddo" {
        dists = nblast_allbyall_kiddo(points, vecs);
    } else {
        panic!("Invalid backend: {}", backend);
    }

    // Turn `scores` into Python array and return it
    dists.into_pyarray(py)

}