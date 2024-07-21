use pyo3::prelude::*;
use pyo3::prelude::{pymodule, PyModule, PyResult, Python};

mod nblast;
use nblast::*;

mod dag;
use dag::*;

#[pymodule]
#[pyo3(name = "_fastcore")]
fn fastcore(_py: Python, m: &PyModule) -> PyResult<()> {
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