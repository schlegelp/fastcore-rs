use pyo3::prelude::*;
use pyo3::prelude::{pymodule, PyModule, PyResult, Python};

mod nblast;
use nblast::*;

mod dag;
use dag::*;

mod mesh;
use mesh::*;

#[pymodule(gil_used = false)]
#[pyo3(name = "_fastcore")]
mod fastcore {
    #[pymodule_export]
    use super::has_cycles_py;

    #[pymodule_export]
    use super::node_indices_16;

    #[pymodule_export]
    use super::node_indices_32;

    #[pymodule_export]
    use super::node_indices_64;

    #[pymodule_export]
    use super::generate_segments_py;

    #[pymodule_export]
    use super::break_segments_py;

    #[pymodule_export]
    use super::all_dists_to_root_py;

    #[pymodule_export]
    use super::dist_to_root_py;

    #[pymodule_export]
    use super::geodesic_distances_py;

    #[pymodule_export]
    use super::geodesic_nearest_py;

    #[pymodule_export]
    use super::geodesic_pairs_py;

    #[pymodule_export]
    use super::nblast_allbyall_py;

    #[pymodule_export]
    use super::nblast_py;

    #[pymodule_export]
    use super::smat_auto_limit_py;

    #[pymodule_export]
    use super::synapse_flow_centrality_py;

    #[pymodule_export]
    use super::connected_components_py;

    #[pymodule_export]
    use super::prune_twigs_py;

    #[pymodule_export]
    use super::strahler_index_py;

    #[pymodule_export]
    use super::classify_nodes_py;

    #[pymodule_export]
    use super::mesh_connected_components_py;

}
