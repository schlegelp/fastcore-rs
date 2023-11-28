use pyo3::prelude::*;

use numpy::ndarray::{Array, Zip};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2, PyReadonlyArray1};

// For each node ID in A find its index in B.
//
// Typically `A` will be parent IDs and `B` will be node IDs.
// Negative IDs (= parents of root nodes) will be passed through.
//
// Note that there is no check whether all IDs in A actually exist in B. If
// an ID in A does not exist in B it gets a negative index (i.e. like roots).
#[pyfunction]
fn _node_indices<'py>(py: Python<'py>,
                      nodes: PyReadonlyArray1<i32>,
                      parents: PyReadonlyArray1<i32>
                      ) -> &'py PyArray<i, 1<[usize; 1]>> {
        let n: usize = nodes.len();
        // let mut xs = vec![n; 5].into_pyarray(py);
}

#[pyfunction]
fn _generate_segments(){

}



#[pymodule]
#[pyo3(name = "_fastcore")]
fn fastcore(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_node_indices, m)?)?;
    m.add_function(wrap_pyfunction!(_generate_segments, m)?)?;

    Ok(())
}