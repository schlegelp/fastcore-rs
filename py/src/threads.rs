use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use fastcore::threads::{num_threads, set_num_threads, ThreadPoolError};

/// Set the number of threads this process uses for parallel work.
///
/// Call before any other `navis_fastcore` function: the pool is built once per
/// process, and whatever runs first fixes its size.
///
/// Arguments:
///
/// - `n`: number of threads; must be >= 1.
///
/// Raises `ValueError` for `n < 1` and `RuntimeError` if the pool already exists
/// at a different size.
#[pyfunction]
#[pyo3(name = "set_num_threads", signature = (n))]
pub fn set_num_threads_py(n: usize) -> PyResult<()> {
    match set_num_threads(n) {
        Ok(()) => Ok(()),
        Err(e @ ThreadPoolError::ZeroThreads) => Err(PyValueError::new_err(e.to_string())),
        Err(e @ ThreadPoolError::AlreadyInitialised { .. }) => {
            Err(PyRuntimeError::new_err(e.to_string()))
        }
    }
}

/// Number of threads available to parallel work in this process.
///
/// Note that calling this *builds* the thread pool if it does not exist yet,
/// which is what a later `set_num_threads` would then fail on. Ask after
/// setting, not before.
#[pyfunction]
#[pyo3(name = "get_num_threads", signature = ())]
pub fn get_num_threads_py() -> usize {
    num_threads()
}
