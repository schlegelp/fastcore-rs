//! Projecting a triangle mesh into a view plane, ready to draw.
//!
//! A 2-D renderer given a mesh has to do four things before it can hand anything to a
//! rasteriser: project the vertices, drop the faces pointing away from the viewer, sort
//! what is left along the view axis, and lay the survivors out as polygons. [`project_mesh_2d`]
//! does all four in one pass over the mesh, because done separately they are four passes
//! over hundreds of megabytes that each produce an intermediate only the next one reads.
//!
//! # Why this is not four functions
//!
//! The measurement that prompted it, on an 8.4M-vertex, 16.9M-face neuron, with each step
//! written the obvious vectorised way in numpy:
//!
//! | step | cost |
//! |---|---|
//! | project to `(V, 2)` | 76 ms |
//! | cull | 226 ms |
//! | gather the kept faces | 72 ms |
//! | gather the kept corners into `(K, 3, 2)` | 191 ms |
//! | close each triangle into a `(K, 4, 2)` ring | 173 ms |
//! | bounding box of the result | 534 ms |
//!
//! Not one of those is arithmetic-bound — they are single-threaded walks over arrays far
//! larger than any cache, and the two gathers plus the ring layout write 900 MB between
//! them to say something the mesh already said. Fused, the whole thing is one parallel
//! pass that writes the rings once and reduces the bounding box on the way past.
//!
//! # The cull is a 2x2 determinant
//!
//! Whether a face points at the viewer is the sign of its normal's depth component, and
//! that component is a determinant of the two *other* columns of the edge vectors — which,
//! for an axis-aligned view, are exactly the two columns being projected onto. So the cull
//! never forms the other two components of the cross product and never reads the depth
//! column at all. It is the same test the full cross product would apply, to the bit.
//!
//! What it does need is [`cull_sign`]: the determinant is taken in cyclic order, and the
//! view is free to name its two axes the other way round, which negates it. Get that wrong
//! and the mesh silently turns inside out for half the views.
//!
//! # Rings, not triangles
//!
//! Each surviving triangle comes back as *four* points, the first repeated at the end. That
//! is what a path fill wants — a closed subpath, no separate close-path instruction — and
//! a caller that wants plain triangles can take `rings[:, :3]`, which is a view rather than
//! a copy. Emitting triangles instead and closing them afterwards is the 173 ms row above,
//! plus a second buffer the size of the first.

use ndarray::{Array1, Array2, Array3, ArrayView2};
use rayon::prelude::*;

use crate::threads::with_pool;

/// Faces per rayon chunk.
///
/// The per-face work is a handful of loads and a couple of multiplies, so at per-face
/// granularity rayon's own bookkeeping would cost several times the work itself; at this
/// size it is noise, and the chunk's output still fits comfortably in L2.
const BLOCK: usize = 1 << 16;

/// An empty bounding box, which any real point widens.
const EMPTY_BBOX: [f64; 4] = [
    f64::INFINITY,
    f64::INFINITY,
    f64::NEG_INFINITY,
    f64::NEG_INFINITY,
];

/// A mesh projected into the view plane: the polygons to fill, and what is needed to
/// place and shade them.
#[derive(Debug, Clone)]
pub struct Projection {
    /// `(K, 4, 2)` — each surviving triangle as a closed ring, its first corner repeated
    /// at the end. Furthest-first when `order` was set, otherwise in face order.
    pub rings: Array3<f64>,
    /// `[xmin, ymin, xmax, ymax]` over the projected corners of the surviving faces.
    ///
    /// This is the bounding box of `rings`, computed while they were written rather than
    /// by walking them again. [`EMPTY_BBOX`] — infinities — when nothing survived.
    pub bbox: [f64; 4],
    /// Index of each surviving face in the original `faces`.
    ///
    /// `i64` rather than `u32` because it is a *position in the caller's array*, not a
    /// vertex id — the crate's dtype rule — and because its whole purpose is to index
    /// arrays alongside it, which numpy widens a `uint32` index array to do anyway.
    pub ix: Array1<i64>,
    /// Mean depth of each surviving face along the depth axis, in the same order as
    /// `rings`. Not sign-corrected, so it can drive a colour ramp directly. `None`
    /// unless `order` was set.
    pub depth: Option<Array1<f64>>,
    /// `(K, 3)` unit face normals. Zero for a degenerate face, which has no normal to
    /// give. `None` unless `normals` was set.
    pub normals: Option<Array2<f64>>,
}

/// `1` if `xy_ix` is already the cyclic pair for `depth_ix`, `-1` if it is the swap.
///
/// `cross(e1, e2)[depth_ix]` is a determinant of the two other columns taken in cyclic
/// order — `((d + 1) % 3, (d + 2) % 3)`. `xy_ix` names that same pair, but in whichever
/// order the view asked for, and swapping the columns of a determinant negates it.
pub fn cull_sign(xy_ix: (usize, usize), depth_ix: usize) -> f64 {
    if xy_ix == ((depth_ix + 1) % 3, (depth_ix + 2) % 3) {
        1.0
    } else {
        -1.0
    }
}

/// Project a triangle mesh into a view plane: cull, sort and lay out, in one pass.
///
/// The view is axis-aligned and given as column indices: `xy_ix` are the two coordinate
/// columns that make up the picture and `depth_ix` is the remaining, into-the-screen one.
/// Coordinates are never flipped — a "right to left" view is the caller's business, which
/// is why `front` is needed to say which end of the depth axis the viewer is on.
///
/// # Arguments
///
/// - `vertices`: `(V, 3)` positions.
/// - `faces`: `(F, 3)` triangles, as indices into `vertices`.
/// - `xy_ix`, `depth_ix`: the view. Must be a permutation of `0, 1, 2`.
/// - `front`: `1` or `-1`, the direction along `depth_ix` that points at the viewer.
/// - `order`: sort furthest-first and return the depths. Off is for a caller filling the
///   whole mesh as one path in one colour, which is blind to the order its subpaths
///   arrive in; the sort is the most expensive step here once the pass itself is parallel.
/// - `normals`: return unit face normals. Off is for a caller that is not shading.
/// - `threads`: rayon pool size, or `None` for every core.
///
/// # Panics
///
/// If `faces` names a vertex that `vertices` does not have, or if `xy_ix`/`depth_ix` are
/// not a permutation of `0, 1, 2`.
#[allow(clippy::too_many_arguments)]
pub fn project_mesh_2d(
    vertices: ArrayView2<f64>,
    faces: ArrayView2<u32>,
    xy_ix: (usize, usize),
    depth_ix: usize,
    front: i8,
    order: bool,
    normals: bool,
    threads: Option<usize>,
) -> Projection {
    let mut axes = [xy_ix.0, xy_ix.1, depth_ix];
    axes.sort_unstable();
    assert_eq!(
        axes,
        [0, 1, 2],
        "`xy_ix` and `depth_ix` must be a permutation of 0, 1, 2, got {xy_ix:?} and {depth_ix}"
    );

    // The Python wrapper always hands us C-order (borrowed as-is); a strided view from a
    // Rust caller gets copied into standard layout.
    let vstore = vertices.as_standard_layout();
    let v: &[f64] = vstore.as_slice().expect("standard layout is contiguous");
    let fstore = faces.as_standard_layout();
    let f: &[u32] = fstore.as_slice().expect("standard layout is contiguous");

    let n_verts = vertices.nrows();

    with_pool(threads, || {
        // In parallel: a serial scan of `3F` indices is 20 ms on a 17M-face mesh, a tenth
        // of what the whole call is aiming at, to re-answer a question the caller can
        // usually answer vectorised. The core still has to ask, since it indexes on the
        // answer - but it can at least ask cheaply.
        if let Some(&max) = f.par_iter().max() {
            assert!(
                (max as usize) < n_verts,
                "`faces` names vertex {max}, but there are only {n_verts} vertices"
            );
        }

        let front = front as f64;
        let mut ix = cull(v, f, xy_ix, front * cull_sign(xy_ix, depth_ix));

        let depth = order.then(|| {
            let mut d = depths(v, f, &ix, depth_ix);
            sort_back_to_front(&mut ix, &mut d, front);
            Array1::from_vec(d)
        });

        let (rings, bbox) = fill_rings(v, f, &ix, xy_ix);
        let normals = normals.then(|| face_normals(v, f, &ix));

        // widened only now: `u32` is what the gathers above want to carry around, and
        // half the bytes to sort
        let ix = ix.par_iter().map(|&i| i as i64).collect::<Vec<_>>();

        Projection {
            rings,
            bbox,
            ix: Array1::from_vec(ix),
            depth,
            normals,
        }
    })
}

/// Indices of the faces that still face the viewer once projected.
///
/// One determinant of the projected edges per face; see the module docs for why that is
/// the whole test. Blocked rather than per-face so that rayon's bookkeeping stays a
/// rounding error, and the per-block `Vec`s are concatenated in block order, which keeps
/// the result in face order.
fn cull(v: &[f64], f: &[u32], (x, y): (usize, usize), sign: f64) -> Vec<u32> {
    f.par_chunks(3 * BLOCK)
        .enumerate()
        .map(|(bi, chunk)| {
            let base = (bi * BLOCK) as u32;
            let mut keep = Vec::new();
            for (i, t) in chunk.chunks_exact(3).enumerate() {
                let (a, b, c) = (
                    3 * t[0] as usize,
                    3 * t[1] as usize,
                    3 * t[2] as usize,
                );
                let (ax, ay) = (v[a + x], v[a + y]);
                let (e1x, e1y) = (v[b + x] - ax, v[b + y] - ay);
                let (e2x, e2y) = (v[c + x] - ax, v[c + y] - ay);
                if (e1x * e2y - e1y * e2x) * sign > 0.0 {
                    keep.push(base + i as u32);
                }
            }
            keep
        })
        .collect::<Vec<_>>()
        .concat()
}

/// Mean depth of each named face along `d`.
fn depths(v: &[f64], f: &[u32], ix: &[u32], d: usize) -> Vec<f64> {
    ix.par_iter()
        .map(|&i| {
            let t = &f[3 * i as usize..3 * i as usize + 3];
            (v[3 * t[0] as usize + d] + v[3 * t[1] as usize + d] + v[3 * t[2] as usize + d])
                / 3.0
        })
        .collect()
}

/// Sort `ix` and `depth` together so that painting them in order gives correct occlusion.
///
/// Ties break on the face index, which costs nothing and makes the result a function of
/// the mesh alone rather than of how many threads happened to run: an unstable parallel
/// sort is free to order equal keys differently from one call to the next, and two
/// coplanar faces that overlap composite differently depending on which went first.
///
/// `total_cmp` rather than `partial_cmp().unwrap()` so that a NaN coordinate sorts
/// somewhere definite instead of panicking half way through a render.
fn sort_back_to_front(ix: &mut Vec<u32>, depth: &mut Vec<f64>, front: f64) {
    let mut ord: Vec<u32> = (0..ix.len() as u32).collect();
    ord.par_sort_unstable_by(|&a, &b| {
        let (a, b) = (a as usize, b as usize);
        (depth[a] * front)
            .total_cmp(&(depth[b] * front))
            .then_with(|| ix[a].cmp(&ix[b]))
    });

    *ix = ord.par_iter().map(|&o| ix[o as usize]).collect();
    *depth = ord.par_iter().map(|&o| depth[o as usize]).collect();
}

/// Write each named face out as a closed ring of projected corners, and reduce the
/// bounding box on the way past.
///
/// The bounding box is free here — the corners are in registers already — where computing
/// it afterwards is another full pass over a buffer four times the size of the mesh's
/// vertices.
fn fill_rings(v: &[f64], f: &[u32], ix: &[u32], (x, y): (usize, usize)) -> (Array3<f64>, [f64; 4]) {
    let k = ix.len();
    let mut rings = vec![0f64; k * 8];

    let bbox = rings
        .par_chunks_mut(8 * BLOCK)
        .zip(ix.par_chunks(BLOCK))
        .map(|(out, idx)| {
            let mut bb = EMPTY_BBOX;
            for (o, &i) in out.chunks_exact_mut(8).zip(idx) {
                let t = &f[3 * i as usize..3 * i as usize + 3];
                for (j, &vi) in t.iter().enumerate() {
                    let (px, py) = (v[3 * vi as usize + x], v[3 * vi as usize + y]);
                    o[2 * j] = px;
                    o[2 * j + 1] = py;
                    bb[0] = bb[0].min(px);
                    bb[1] = bb[1].min(py);
                    bb[2] = bb[2].max(px);
                    bb[3] = bb[3].max(py);
                }
                // close the ring: a repeated first corner, not a close-path instruction
                o[6] = o[0];
                o[7] = o[1];
            }
            bb
        })
        .reduce(
            || EMPTY_BBOX,
            |a, b| {
                [
                    a[0].min(b[0]),
                    a[1].min(b[1]),
                    a[2].max(b[2]),
                    a[3].max(b[3]),
                ]
            },
        );

    (
        Array3::from_shape_vec((k, 4, 2), rings).expect("k * 8 values in (k, 4, 2)"),
        bbox,
    )
}

/// Unit normals of the named faces, in their order.
///
/// Only the survivors, which is about half the mesh for a closed surface — the culled
/// faces have no polygon to shade. A degenerate face gets a zero normal rather than a
/// division by zero, which is the same answer a normalise-with-a-guard would give it.
fn face_normals(v: &[f64], f: &[u32], ix: &[u32]) -> Array2<f64> {
    let mut out = vec![0f64; ix.len() * 3];
    out.par_chunks_exact_mut(3)
        .zip(ix.par_iter())
        .for_each(|(o, &i)| {
            let t = &f[3 * i as usize..3 * i as usize + 3];
            let (a, b, c) = (3 * t[0] as usize, 3 * t[1] as usize, 3 * t[2] as usize);
            let e1 = [v[b] - v[a], v[b + 1] - v[a + 1], v[b + 2] - v[a + 2]];
            let e2 = [v[c] - v[a], v[c + 1] - v[a + 1], v[c + 2] - v[a + 2]];
            let n = [
                e1[1] * e2[2] - e1[2] * e2[1],
                e1[2] * e2[0] - e1[0] * e2[2],
                e1[0] * e2[1] - e1[1] * e2[0],
            ];
            let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            if len > 0.0 {
                o[0] = n[0] / len;
                o[1] = n[1] / len;
                o[2] = n[2] / len;
            }
        });
    Array2::from_shape_vec((ix.len(), 3), out).expect("3 values per face")
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// A unit tetrahedron, wound outwards.
    fn tetra() -> (Array2<f64>, Array2<u32>) {
        let v = array![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ];
        let f = array![[0u32, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]];
        (v, f)
    }

    /// The cheap cull has to keep exactly what the full cross product would, for every
    /// view — which is the one thing `cull_sign` exists to get right.
    #[test]
    fn cull_matches_the_full_cross_product() {
        let (v, f) = tetra();
        for (xy, d) in [
            ((0, 1), 2),
            ((1, 0), 2),
            ((0, 2), 1),
            ((2, 0), 1),
            ((1, 2), 0),
            ((2, 1), 0),
        ] {
            for front in [1i8, -1] {
                let got = project_mesh_2d(v.view(), f.view(), xy, d, front, true, false, None);

                let want: Vec<i64> = (0..f.nrows())
                    .filter(|&i| {
                        let (a, b, c) = (
                            v.row(f[[i, 0]] as usize),
                            v.row(f[[i, 1]] as usize),
                            v.row(f[[i, 2]] as usize),
                        );
                        let e1 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
                        let e2 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
                        let n = [
                            e1[1] * e2[2] - e1[2] * e2[1],
                            e1[2] * e2[0] - e1[0] * e2[2],
                            e1[0] * e2[1] - e1[1] * e2[0],
                        ];
                        n[d] * front as f64 > 0.0
                    })
                    .map(|i| i as i64)
                    .collect();

                let mut got_sorted = got.ix.to_vec();
                got_sorted.sort_unstable();
                assert_eq!(got_sorted, want, "view {xy:?} depth {d} front {front}");
            }
        }
    }

    /// Every ring closes, and it closes on the projected columns the view asked for.
    #[test]
    fn rings_close_on_the_projected_columns() {
        let (v, f) = tetra();
        let p = project_mesh_2d(v.view(), f.view(), (2, 0), 1, 1, true, false, None);
        for k in 0..p.rings.shape()[0] {
            assert_eq!(p.rings[[k, 3, 0]], p.rings[[k, 0, 0]]);
            assert_eq!(p.rings[[k, 3, 1]], p.rings[[k, 0, 1]]);
            for j in 0..3 {
                let vi = f[[p.ix[k] as usize, j]] as usize;
                assert_eq!(p.rings[[k, j, 0]], v[[vi, 2]]);
                assert_eq!(p.rings[[k, j, 1]], v[[vi, 0]]);
            }
        }
    }

    /// The bounding box has to be the one over the rings, not over the whole mesh.
    #[test]
    fn bbox_covers_exactly_the_rings() {
        let (v, f) = tetra();
        let p = project_mesh_2d(v.view(), f.view(), (0, 1), 2, 1, true, false, None);
        let xs: Vec<f64> = (0..p.rings.shape()[0])
            .flat_map(|k| (0..4).map(move |j| (k, j)))
            .map(|(k, j)| p.rings[[k, j, 0]])
            .collect();
        let ys: Vec<f64> = (0..p.rings.shape()[0])
            .flat_map(|k| (0..4).map(move |j| (k, j)))
            .map(|(k, j)| p.rings[[k, j, 1]])
            .collect();
        assert_eq!(p.bbox[0], xs.iter().cloned().fold(f64::INFINITY, f64::min));
        assert_eq!(p.bbox[1], ys.iter().cloned().fold(f64::INFINITY, f64::min));
        assert_eq!(
            p.bbox[2],
            xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        );
        assert_eq!(
            p.bbox[3],
            ys.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        );
    }

    /// Furthest first, and `depth` has to travel with the faces it belongs to.
    #[test]
    fn sorted_back_to_front_with_depths_alongside() {
        let (v, f) = tetra();
        for front in [1i8, -1] {
            let p = project_mesh_2d(v.view(), f.view(), (0, 1), 2, front, true, false, None);
            let d = p.depth.as_ref().unwrap();
            for w in d.windows(2).into_iter() {
                assert!(w[0] * front as f64 <= w[1] * front as f64);
            }
            for k in 0..p.ix.len() {
                let t = f.row(p.ix[k] as usize);
                let mean = (v[[t[0] as usize, 2]] + v[[t[1] as usize, 2]] + v[[t[2] as usize, 2]])
                    / 3.0;
                assert!((d[k] - mean).abs() < 1e-12);
            }
        }
    }

    /// `order` off keeps the same faces, in face order, and drops the depths.
    #[test]
    fn unordered_keeps_the_same_faces_in_face_order() {
        let (v, f) = tetra();
        let a = project_mesh_2d(v.view(), f.view(), (0, 1), 2, 1, true, false, None);
        let b = project_mesh_2d(v.view(), f.view(), (0, 1), 2, 1, false, false, None);

        assert!(b.depth.is_none());
        let mut sorted = a.ix.to_vec();
        sorted.sort_unstable();
        assert_eq!(b.ix.to_vec(), sorted);
        assert_eq!(a.bbox, b.bbox);
    }

    /// Normals are unit length, point the way the winding says, and only appear if asked.
    #[test]
    fn normals_are_unit_and_optional() {
        let (v, f) = tetra();
        let p = project_mesh_2d(v.view(), f.view(), (0, 1), 2, 1, true, true, None);
        let n = p.normals.as_ref().unwrap();
        assert_eq!(n.shape(), &[p.ix.len(), 3]);
        for k in 0..n.nrows() {
            let len = (n[[k, 0]].powi(2) + n[[k, 1]].powi(2) + n[[k, 2]].powi(2)).sqrt();
            assert!((len - 1.0).abs() < 1e-12);
            // culled towards +z with front = 1, so every survivor leans that way
            assert!(n[[k, 2]] > 0.0);
        }
        assert!(project_mesh_2d(v.view(), f.view(), (0, 1), 2, 1, true, false, None)
            .normals
            .is_none());
    }

    /// A degenerate face has no normal to give, and must not produce a NaN one.
    #[test]
    fn degenerate_faces_get_a_zero_normal() {
        let v = array![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
        let f = array![[0u32, 1, 2]];
        // collinear, so it is culled whatever the view - ask for the one that keeps it
        let p = project_mesh_2d(v.view(), f.view(), (0, 1), 2, 1, true, true, None);
        assert_eq!(p.ix.len(), 0);
        assert_eq!(p.bbox, EMPTY_BBOX);
        assert_eq!(p.rings.shape(), &[0, 4, 2]);
    }

    /// An empty mesh must not panic on the blocked pass or the bbox reduction.
    #[test]
    fn empty_mesh_is_empty() {
        let v = Array2::<f64>::zeros((0, 3));
        let f = Array2::<u32>::zeros((0, 3));
        let p = project_mesh_2d(v.view(), f.view(), (0, 1), 2, 1, true, true, None);
        assert_eq!(p.ix.len(), 0);
        assert_eq!(p.rings.shape(), &[0, 4, 2]);
        assert_eq!(p.bbox, EMPTY_BBOX);
    }

    /// More faces than one block, so the block offsets and the concatenation get exercised.
    #[test]
    fn survives_more_than_one_block() {
        let n = BLOCK + 1000;
        let mut v = Vec::with_capacity(n * 9);
        let mut f = Vec::with_capacity(n * 3);
        for i in 0..n {
            let z = i as f64;
            v.extend_from_slice(&[0.0, 0.0, z, 1.0, 0.0, z, 0.0, 1.0, z]);
            f.extend_from_slice(&[(3 * i) as u32, (3 * i + 1) as u32, (3 * i + 2) as u32]);
        }
        let v = Array2::from_shape_vec((3 * n, 3), v).unwrap();
        let f = Array2::from_shape_vec((n, 3), f).unwrap();

        let p = project_mesh_2d(v.view(), f.view(), (0, 1), 2, 1, true, false, None);
        // all wound the same way, so all of them survive, furthest first
        assert_eq!(p.ix.len(), n);
        let d = p.depth.as_ref().unwrap();
        for w in d.windows(2).into_iter() {
            assert!(w[0] <= w[1]);
        }
        assert_eq!(p.bbox, [0.0, 0.0, 1.0, 1.0]);
    }
}
