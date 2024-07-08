use pyo3::prelude::*;
use pyo3::prelude::{pymodule, PyModule, PyResult, Python};
use ndarray::Array2;


mod nblast;
use nblast::*;

mod dag;
use dag::*;

#[pyfunction]
fn run_test() -> PyResult<()> {
    let data: Array2<f64>;
    let indices: Vec<[f64; 2]>;
    let columns: Vec<[f64; 2]>;
    (data, indices, columns) = nblast::load_smat();
    println!("Indices: {:?}", indices);
    println!("Columns: {:?}", columns);
    println!("Data: {:?}", data);
    Ok(())
}

#[pymodule]
#[pyo3(name = "_fastcore")]
fn fastcore(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_test, m)?)?;
    m.add_function(wrap_pyfunction!(node_indices, m)?)?;
    m.add_function(wrap_pyfunction!(generate_segments_py, m)?)?;
    m.add_function(wrap_pyfunction!(break_segments_py, m)?)?;
    m.add_function(wrap_pyfunction!(all_dists_to_root_py, m)?)?;
    m.add_function(wrap_pyfunction!(dist_to_root_py, m)?)?;
    m.add_function(wrap_pyfunction!(top_nn_py, m)?)?;
    m.add_function(wrap_pyfunction!(geodesic_distances_py, m)?)?;
    m.add_function(wrap_pyfunction!(nblast_single_py, m)?)?;
    m.add_function(wrap_pyfunction!(nblast_allbyall_py, m)?)?;
    m.add_function(wrap_pyfunction!(synapse_flow_centrality_py, m)?)?;
    m.add_function(wrap_pyfunction!(connected_components_py, m)?)?;
    m.add_function(wrap_pyfunction!(prune_twigs_py, m)?)?;
    m.add_function(wrap_pyfunction!(strahler_index_py, m)?)?;
    m.add_function(wrap_pyfunction!(classify_nodes_py, m)?)?;

    Ok(())
}