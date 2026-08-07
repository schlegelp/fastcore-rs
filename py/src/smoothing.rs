use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use fastcore::smoothing::{smooth_mesh, Filter, Volume, Weights};

use crate::mesh::as_opt_flags;

/// What `smooth_mesh` hands back: the new positions, and — only when a requested volume
/// correction turned out to be undecidable — the two signed volumes that made it so.
///
/// `None` in the second slot therefore means "nothing to say", covering both "no correction
/// was asked for" and "it was applied". The scale factor itself is not returned: it is the
/// *result* of the correction rather than a decision the caller has to make, and the R
/// binding already ships without it.
type SmoothOut<'py> = (Bound<'py, PyArray2<f64>>, Option<(f64, f64)>);

/// Turn a name the caller spelled into the core's enum.
///
/// The names, the defaults and the "which parameter belongs to which method" table all
/// live in `fastcore::smoothing`, as `linkage::Method::from_name` and friends do — see
/// `Filter::params_of`. What is left here is only the mapping onto `ValueError`, which is
/// the one thing that cannot live in the core.
fn filter_of(
    method: &str,
    lamb: Option<f64>,
    mu: Option<f64>,
    alpha: Option<f64>,
    beta: Option<f64>,
) -> PyResult<Filter> {
    let params = Filter::params_of(method).ok_or_else(|| {
        PyValueError::new_err(format!(
            "`method` must be one of {}, got \"{method}\"",
            quoted(&Filter::METHODS)
        ))
    })?;
    // A parameter belonging to another method is an error rather than something quietly
    // dropped: a call that passes `alpha` to Taubin has asked for something, and ignoring
    // it is the one outcome that looks like success.
    for (name, value) in [("lamb", lamb), ("mu", mu), ("alpha", alpha), ("beta", beta)] {
        // The core spells lambda in full; Python cannot, so it is the one name that has to
        // be translated on the way in as well as out.
        let core_name = if name == "lamb" { "lambda" } else { name };
        if value.is_some() && !params.contains(&core_name) {
            return Err(PyValueError::new_err(format!(
                "`{name}` does not apply to method=\"{method}\""
            )));
        }
    }

    let filter = Filter::from_parts(method, lamb, mu, alpha, beta)
        .expect("params_of already accepted this method");
    // Ranges are checked once, in the core; `"lamb"` is how this surface spells lambda, so
    // the message comes back naming the argument the caller actually passed.
    filter.check("lamb").map_err(PyValueError::new_err)?;
    Ok(filter)
}

fn weights_of(weights: &str) -> PyResult<Weights> {
    Weights::from_name(weights).ok_or_else(|| {
        PyValueError::new_err(format!(
            "`weights` must be one of {}, got \"{weights}\"",
            quoted(&Weights::NAMES)
        ))
    })
}

/// `["a", "b", "c"]` as `"a", "b" or "c"`, for the two messages above.
fn quoted(names: &[&str]) -> String {
    let mut out = String::new();
    for (i, n) in names.iter().enumerate() {
        if i > 0 {
            out.push_str(if i == names.len() - 1 { " or " } else { ", " });
        }
        out.push('"');
        out.push_str(n);
        out.push('"');
    }
    out
}

/// Smooth a triangle mesh.
///
/// Arguments
/// ---------
/// - `faces`:             (F, 3) uint32 array of triangular faces (vertex indices).
/// - `vertices`:          (V, 3) float64 vertex positions.
/// - `method`:            `"laplacian"`, `"taubin"` or `"humphrey"`.
/// - `iterations`:        Passes to run. For Taubin, one pass is a full lambda/mu pair.
/// - `lamb`, `mu`:        Laplacian and Taubin parameters; defaults 0.5 and -0.53.
/// - `alpha`, `beta`:     Humphrey parameters; defaults 0.1 and 0.5.
/// - `weights`:           `"uniform"`, `"inverse_distance"` or `"cotangent"`.
/// - `preserve_border`:   Pin every vertex on a mesh boundary.
/// - `lock`:              Optional (V, ) bool array of vertices that must not move.
/// - `volume_correction`: Rescale about the centroid to restore the input volume.
/// - `threads`:           Size of the thread pool, or `None` for all cores.
///
/// Returns
/// -------
/// A 2-tuple `(vertices, volumes)`; see `SmoothOut`.
#[pyfunction]
#[pyo3(
    name = "smooth_mesh",
    signature = (
        faces,
        vertices,
        method="taubin",
        iterations=10,
        lamb=None,
        mu=None,
        alpha=None,
        beta=None,
        weights="uniform",
        preserve_border=false,
        lock=None,
        volume_correction=false,
        threads=None,
    )
)]
// One entry point rather than one per filter, so that the obvious call is the one that
// does the right thing; what that costs is this argument list, of which at most two are
// ever live at once.
#[allow(clippy::too_many_arguments)]
pub fn smooth_mesh_py<'py>(
    py: Python<'py>,
    faces: PyReadonlyArray2<u32>,
    vertices: PyReadonlyArray2<f64>,
    method: &str,
    iterations: usize,
    lamb: Option<f64>,
    mu: Option<f64>,
    alpha: Option<f64>,
    beta: Option<f64>,
    weights: &str,
    preserve_border: bool,
    lock: Option<PyReadonlyArray1<bool>>,
    volume_correction: bool,
    threads: Option<usize>,
) -> PyResult<SmoothOut<'py>> {
    let filter = filter_of(method, lamb, mu, alpha, beta)?;
    let weights = weights_of(weights)?;

    // Shapes and the face index range are the wrapper's job, as they are for
    // `simplify_mesh_py`: `_prep_mesh` has already forced (F, 3) uint32 / (V, 3) float64
    // and checked `faces.max()`, vectorised, before this is reached.
    let (faces, vertices) = (faces.as_array(), vertices.as_array());
    let lock = as_opt_flags(&lock, "lock", vertices.nrows())?;

    // Off the GIL: several passes over a mesh that can be millions of vertices, and the
    // core takes the rayon pool for the duration. The views are taken first because the
    // `PyReadonlyArray` guards carry a Python token and cannot cross the boundary.
    let out = py.detach(|| {
        smooth_mesh(
            faces,
            vertices,
            filter,
            weights,
            iterations,
            preserve_border,
            lock,
            volume_correction,
            threads,
        )
    });

    let volumes = match out.volume {
        Volume::Undefined { before, after } => Some((before, after)),
        Volume::Off | Volume::Scaled(_) => None,
    };
    Ok((out.vertices.into_pyarray(py), volumes))
}
