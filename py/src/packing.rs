use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use fastcore::packing::{pack_masks, pack_rectangles, Rect};
use fastcore::raster::{rasterize_segments_batch, Bitmap, RasterOptions};

/// Borrow a `(h, w)` bool array as a [`Bitmap`].
fn to_bitmap(a: &PyReadonlyArray2<bool>) -> PyResult<Bitmap> {
    let v = a.as_array();
    let (h, w) = (v.nrows(), v.ncols());
    let store = v.as_standard_layout();
    let flat = store
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("mask must be a 2-D array of bool"))?;
    Ok(Bitmap::from_bools(flat, h, w))
}

/// Hand a [`Bitmap`] back as a `(h, w)` bool array.
fn from_bitmap<'py>(py: Python<'py>, b: &Bitmap) -> Bound<'py, PyArray2<bool>> {
    let arr = b.to_bools().into_pyarray(py);
    arr.reshape([b.height(), b.width()])
        .expect("the unpacked length is height * width by construction")
}

/// `(N, 2)` lower-left corners, `NaN` where an item was skipped.
///
/// Shared by both packers: the "NaN means skipped" convention is one thing, so it is
/// written down once.
fn positions_array<'py, T: Copy + Into<f64>>(
    py: Python<'py>,
    positions: &[Option<(T, T)>],
) -> Bound<'py, PyArray2<f64>> {
    let flat: Vec<f64> = positions
        .iter()
        .flat_map(|p| p.map_or([f64::NAN; 2], |(x, y)| [x.into(), y.into()]))
        .collect();
    flat.into_pyarray(py)
        .reshape([positions.len(), 2])
        .expect("two values per item by construction")
}

/// Rasterise line segments into binary masks, one shape per core.
///
/// Arguments
/// ---------
/// - `coords`: one `(N, 2)` float64 array per shape — the two in-plane coordinates.
/// - `edges`:  one `(E, 2)` uint32 array per shape, indexing into the matching `coords`.
/// - `scale`:  pixels per coordinate unit.
/// - `pad`:    margin left around the shape, in pixels, and the radius it is grown by.
/// - `fill`:   one flag per shape — fill its interior, as a solid shape wants.
/// - `turn`:   quarter turns counter-clockwise to apply before rasterising, mod 4.
/// - `threads`: size of the thread pool, or `None` for all cores.
///
/// Returns
/// -------
/// One `(h, w)` bool array per shape, each just big enough for the shape plus `pad`
/// pixels of margin on every side.
#[pyfunction]
#[pyo3(
    name = "rasterize_segments",
    signature = (coords, edges, scale, pad=0, fill=None, turn=0, threads=None)
)]
#[allow(clippy::too_many_arguments)]
pub fn rasterize_segments_py<'py>(
    py: Python<'py>,
    coords: Vec<PyReadonlyArray2<'py, f64>>,
    edges: Vec<PyReadonlyArray2<'py, u32>>,
    scale: f64,
    pad: usize,
    fill: Option<Vec<bool>>,
    turn: u8,
    threads: Option<usize>,
) -> PyResult<Vec<Bound<'py, PyArray2<bool>>>> {
    if coords.len() != edges.len() {
        return Err(PyValueError::new_err(format!(
            "got {} coordinate arrays but {} edge arrays",
            coords.len(),
            edges.len()
        )));
    }
    if let Some(f) = &fill {
        if f.len() != coords.len() {
            return Err(PyValueError::new_err(format!(
                "got {} fill flags for {} shapes",
                f.len(),
                coords.len()
            )));
        }
    }
    if !scale.is_finite() || scale <= 0.0 {
        return Err(PyValueError::new_err(format!(
            "`scale` must be finite and positive, got {scale}"
        )));
    }

    // Shapes are checked here rather than left to the core's asserts, which would surface
    // as a PanicException with a traceback pointing at nothing the caller wrote. The edge
    // *range* is not checked here — see the note in the Python wrapper, which does it
    // vectorised while it already holds the array.
    let shapes: Vec<_> = coords
        .iter()
        .zip(edges.iter())
        .enumerate()
        .map(|(i, (c, e))| {
            let (c, e) = (c.as_array(), e.as_array());
            if c.ncols() != 2 {
                return Err(PyValueError::new_err(format!(
                    "`coords[{i}]` must be (N, 2), got {:?}",
                    c.shape()
                )));
            }
            if e.ncols() != 2 {
                return Err(PyValueError::new_err(format!(
                    "`edges[{i}]` must be (E, 2), got {:?}",
                    e.shape()
                )));
            }
            Ok((c, e))
        })
        .collect::<PyResult<_>>()?;

    let opts = RasterOptions {
        scale,
        pad,
        fill: false,
        turn,
    };
    // Off the GIL: this is the parallel part, and for a mesh each shape is a walk over
    // hundreds of thousands of edges.
    let out = py.detach(|| rasterize_segments_batch(&shapes, &opts, fill.as_deref(), threads));

    Ok(out.iter().map(|b| from_bitmap(py, b)).collect())
}

/// What `pack_masks` hands back: positions, variants, and the finished page.
type MaskPackOut<'py> = (
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray2<bool>>,
);

/// Place binary masks onto a page so that no two of them share a pixel.
///
/// Arguments
/// ---------
/// - `masks`:      one list per item, holding the variants to try (upright, and turned if
///                 rotation is allowed). Each variant is a `(h, w)` bool array.
/// - `page_shape`: `(height, width)` in pixels.
/// - `grid`:       a `(height, width)` bool page that is already partly occupied, or
///                 `None` for an empty one.
/// - `cost`:       a `(height, width)` float64 array — each mask goes where this is lowest
///                 under its centre. `None` means as far down, then as far left, as it
///                 will go.
/// - `optional`:   skip masks that fit nowhere instead of failing the packing.
/// - `threads`:    size of the thread pool, or `None` for all cores.
///
/// Returns
/// -------
/// `(positions, variant, grid)`. `positions` is `(N, 2)` float64 of lower-left `(x, y)`
/// corners, `NaN` for a mask that was skipped; `variant` says which variant was used;
/// `grid` is the page with everything drawn onto it. Returns `None` if a mask fit nowhere
/// and `optional` was not set.
#[pyfunction]
#[pyo3(
    name = "pack_masks",
    signature = (masks, page_shape, grid=None, cost=None, optional=false, threads=None)
)]
pub fn pack_masks_py<'py>(
    py: Python<'py>,
    masks: Vec<Vec<PyReadonlyArray2<'py, bool>>>,
    page_shape: (usize, usize),
    grid: Option<PyReadonlyArray2<'py, bool>>,
    cost: Option<PyReadonlyArray2<'py, f64>>,
    optional: bool,
    threads: Option<usize>,
) -> PyResult<Option<MaskPackOut<'py>>> {
    let (height, width) = page_shape;

    let prepared: Vec<Vec<Bitmap>> = masks
        .iter()
        .map(|v| v.iter().map(to_bitmap).collect::<PyResult<_>>())
        .collect::<PyResult<_>>()?;

    let grid = grid
        .map(|g| {
            let b = to_bitmap(&g)?;
            if (b.height(), b.width()) != (height, width) {
                return Err(PyValueError::new_err(format!(
                    "`grid` is {}x{} but the page is {height}x{width}",
                    b.height(),
                    b.width()
                )));
            }
            Ok(b)
        })
        .transpose()?;

    // Borrowed rather than copied: the wrapper has already made it C-contiguous float64,
    // and this is 8 MB for a page-sized surface.
    let cost = cost
        .as_ref()
        .map(|c| {
            let v = c.as_array();
            if (v.nrows(), v.ncols()) != (height, width) {
                return Err(PyValueError::new_err(format!(
                    "`cost` is {}x{} but the page is {height}x{width}",
                    v.nrows(),
                    v.ncols()
                )));
            }
            c.as_slice()
                .map_err(|_| PyValueError::new_err("`cost` must be a C-contiguous array"))
        })
        .transpose()?;

    let packed =
        py.detach(|| pack_masks(&prepared, (height, width), grid, cost, optional, threads));

    let Some(packed) = packed else {
        return Ok(None);
    };

    let variant: Vec<i64> = packed.variant.iter().map(|&v| v as i64).collect();
    let positions: Vec<Option<(u32, u32)>> = packed
        .positions
        .iter()
        .map(|p| p.map(|(x, y)| (x as u32, y as u32)))
        .collect();

    Ok(Some((
        positions_array(py, &positions),
        variant.into_pyarray(py),
        from_bitmap(py, &packed.grid),
    )))
}

/// What `pack_rectangles` hands back: positions, rotation flags, and the free space left.
type RectPackOut<'py> = (
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<bool>>,
    Bound<'py, PyArray2<f64>>,
);

/// Pack rectangles into a page by the MaxRects heuristic.
///
/// Arguments
/// ---------
/// - `sizes`:          `(N, 2)` float64 widths and heights.
/// - `page_size`:      `(width, height)`.
/// - `allow_rotation`: whether a rectangle may go in turned a quarter turn.
/// - `optional`:       skip rectangles that fit nowhere instead of failing the packing.
/// - `free`:           `(M, 4)` float64 free space `(x, y, width, height)` to pack into,
///                     e.g. what an earlier call left over. `None` means the whole page.
///
/// Returns
/// -------
/// `(positions, rotated, free)`, with `positions` `(N, 2)` float64 lower-left corners and
/// `NaN` for a rectangle that was skipped. Returns `None` if a rectangle fit nowhere and
/// `optional` was not set.
#[pyfunction]
#[pyo3(
    name = "pack_rectangles",
    signature = (sizes, page_size, allow_rotation=false, optional=false, free=None)
)]
pub fn pack_rectangles_py<'py>(
    py: Python<'py>,
    sizes: PyReadonlyArray2<'py, f64>,
    page_size: (f64, f64),
    allow_rotation: bool,
    optional: bool,
    free: Option<PyReadonlyArray2<'py, f64>>,
) -> PyResult<Option<RectPackOut<'py>>> {
    let s = sizes.as_array();
    if s.ncols() != 2 {
        return Err(PyValueError::new_err(format!(
            "`sizes` must be (N, 2), got {:?}",
            s.shape()
        )));
    }
    let sizes: Vec<(f64, f64)> = s.rows().into_iter().map(|r| (r[0], r[1])).collect();

    let free = free
        .map(|f| {
            let v = f.as_array();
            if v.ncols() != 4 {
                return Err(PyValueError::new_err(format!(
                    "`free` must be (M, 4), got {:?}",
                    v.shape()
                )));
            }
            Ok(v.rows()
                .into_iter()
                .map(|r| Rect {
                    x: r[0],
                    y: r[1],
                    w: r[2],
                    h: r[3],
                })
                .collect::<Vec<_>>())
        })
        .transpose()?;

    let packed = py.detach(|| pack_rectangles(&sizes, page_size, allow_rotation, optional, free));

    let Some(packed) = packed else {
        return Ok(None);
    };

    let pos = positions_array(py, &packed.positions);

    let m = packed.free.len();
    let mut rest = Vec::with_capacity(m * 4);
    for r in &packed.free {
        rest.extend_from_slice(&[r.x, r.y, r.w, r.h]);
    }
    let rest = rest
        .into_pyarray(py)
        .reshape([m, 4])
        .expect("four values per rectangle by construction");

    Ok(Some((pos, packed.rotated.into_pyarray(py), rest)))
}
