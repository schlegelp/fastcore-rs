use pyo3::prelude::*;
use pyo3::prelude::{pymodule, PyModule, PyResult, Python};

mod nblast;
use nblast::*;

mod matches;
use matches::*;

mod cmtk;
use cmtk::*;

mod dag;
use dag::*;

mod mesh;
use mesh::*;

mod topo;
use topo::*;

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
    use super::geodesic_farthest_py;

    #[pymodule_export]
    use super::geodesic_pairs_py;

    #[pymodule_export]
    use super::nblast_allbyall_py;

    #[pymodule_export]
    use super::nblast_py;

    #[pymodule_export]
    use super::nblast_pairs_py;

    #[pymodule_export]
    use super::top_matches_py;

    #[pymodule_export]
    use super::matches_above_py;

    #[pymodule_export]
    use super::count_matches_py;

    #[pymodule_export]
    use super::synblast_allbyall_py;

    #[pymodule_export]
    use super::synblast_py;

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
    use super::subtree_height_py;

    #[pymodule_export]
    use super::classify_nodes_py;

    #[pymodule_export]
    use super::mesh_connected_components_py;

    #[pymodule_export]
    use super::geodesic_matrix_mesh_py;

    #[pymodule_export]
    use super::geodesic_matrix_graph_py;

    #[pymodule_export]
    use super::geodesic_nearest_mesh_py;

    #[pymodule_export]
    use super::geodesic_farthest_mesh_py;

    #[pymodule_export]
    use super::stitch_fragments_py;

    #[pymodule_export]
    use super::reroot_rewire_py;

    #[pymodule_export]
    use super::PyCmtkRegistration;
}
