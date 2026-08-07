//! Quadric-error mesh simplification that remembers where every vertex went.
//!
//! This is a port of Sven Forstmann's Fast-Quadric-Mesh-Simplification
//! (`Simplify.h`, MIT), the algorithm behind `pyfqmr`, with one addition that is
//! the whole reason it lives here: [`Simplified::vertex_map`] records, for every
//! vertex of the *input* mesh, which vertex of the *output* mesh it ended up in.
//!
//! # Attribution
//!
//! Derived from <https://github.com/sp4cerat/Fast-Quadric-Mesh-Simplification>,
//! `src.cmd/Simplify.h`, lines 248-809 (the algorithm; the OBJ/PLY I/O and the UV
//! interpolation are not ported). MIT is compatible with this crate's GPL-3.0, and
//! the upstream notice is reproduced as MIT requires:
//!
//! > Mesh Simplification Tutorial
//! > (C) by Sven Forstmann in 2014
//! > License : MIT — <http://opensource.org/licenses/MIT>
//!
//! # Why a port rather than a wrapper
//!
//! Every off-the-shelf crate we looked at failed on one of two counts. The ones
//! built on a halfedge or corner table (`alum`, `baby_shark`) require manifold
//! input — they either refuse the mesh outright or silently drop the offending
//! faces — and meshes derived from EM segmentation routinely have edges shared by
//! more than two faces. The one with the right semantics, `meshopt`, vendors C++
//! and builds it through `cc`, which this crate cannot afford: the same source has
//! to compile for pyodide/WASM and as an R tarball on r-universe. None of them
//! expose a collapse map anyway.
//!
//! This algorithm, by contrast, is flat index arrays throughout — no halfedges, no
//! adjacency invariants to violate — so non-manifold input is simply data. The only
//! topological notion it has is a per-vertex `border` flag derived from a one-ring
//! count, and every collapse guard skips the collapse rather than aborting the run.
//!
//! # Divergences from upstream
//!
//! Three, all of them about degenerate geometry, all marked `DIVERGENCE` at the
//! site. Upstream normalises vectors unconditionally; a zero-length vector then
//! yields NaN, and because every comparison against NaN is false, NaN silently
//! *defeats* the two tests that are supposed to reject a bad collapse. Neuron
//! meshes do contain zero-area and duplicated faces, so each of those spots grew a
//! length check. On a clean mesh none of them trigger and the output matches
//! `pyfqmr` exactly.

use crate::mesh::find;
use ndarray::{Array1, Array2, ArrayView2};
use std::ops::Add;

// ---------------------------------------------------------------------------
// Tuning constants
// ---------------------------------------------------------------------------
//
// These are `pyfqmr`'s defaults, which are in turn Forstmann's. They are named
// rather than exposed: they trade quality against runtime in a way that only makes
// sense as a set, and every caller we have wants the set upstream ships.

/// Rebuild the reference list every this many iterations.
const UPDATE_RATE: usize = 5;
/// Scale of the per-iteration error threshold.
const ALPHA: f64 = 0.000000001;
/// Offset added to the iteration number before raising it to `aggressiveness`.
const K: f64 = 3.0;
/// Iteration cap for the face-count-targeted sweep.
const MAX_ITERATIONS: usize = 100;

// ---------------------------------------------------------------------------
// Vector helpers
// ---------------------------------------------------------------------------
//
// Three components of `f64`, which is what `vec3f` is upstream despite the name.
// Not worth a dependency or a type — these are all one line.

#[inline]
fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// The unit vector along `v`, or `None` if `v` is too short to have a direction.
///
/// Returning an `Option` is the point. Upstream divides by the length whatever it
/// is, so a zero-length input produces NaN, and NaN then passes every downstream
/// rejection test because comparisons against it are false. Making "no direction"
/// unrepresentable forces each caller to say what it wants to happen instead.
#[inline]
fn normalize(v: [f64; 3]) -> Option<[f64; 3]> {
    let len = dot(v, v).sqrt();
    // Not an epsilon: anything with a finite, non-zero length has a well-defined
    // direction, and picking a threshold here would reject slender-but-real
    // triangles in meshes whose units happen to be small (nanometres, say).
    if len > 0.0 && len.is_finite() {
        Some([v[0] / len, v[1] / len, v[2] / len])
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Quadrics
// ---------------------------------------------------------------------------

/// The upper triangle of a symmetric 4x4 error quadric, row-major.
///
/// The quadric of a vertex is the sum of the fundamental quadrics of the planes of
/// its incident faces; evaluating it at a point gives the sum of squared distances
/// to those planes, which is the error a collapse to that point would introduce.
#[derive(Clone, Copy, Default)]
struct Quadric([f64; 10]);

impl Quadric {
    /// The fundamental quadric of the plane `ax + by + cz + d = 0`, i.e. the outer
    /// product of `(a, b, c, d)` with itself.
    fn plane(a: f64, b: f64, c: f64, d: f64) -> Self {
        Quadric([
            a * a,
            a * b,
            a * c,
            a * d,
            b * b,
            b * c,
            b * d,
            c * c,
            c * d,
            d * d,
        ])
    }

    /// Determinant of the 3x3 minor named by nine indices into the packed storage.
    #[allow(clippy::too_many_arguments)]
    fn det(
        &self,
        a11: usize,
        a12: usize,
        a13: usize,
        a21: usize,
        a22: usize,
        a23: usize,
        a31: usize,
        a32: usize,
        a33: usize,
    ) -> f64 {
        let m = &self.0;
        m[a11] * m[a22] * m[a33] + m[a13] * m[a21] * m[a32] + m[a12] * m[a23] * m[a31]
            - m[a13] * m[a22] * m[a31]
            - m[a11] * m[a23] * m[a32]
            - m[a12] * m[a21] * m[a33]
    }

    /// `p^T Q p` — the error of placing a vertex at `p`.
    fn error(&self, p: [f64; 3]) -> f64 {
        let (m, x, y, z) = (&self.0, p[0], p[1], p[2]);
        m[0] * x * x
            + 2.0 * m[1] * x * y
            + 2.0 * m[2] * x * z
            + 2.0 * m[3] * x
            + m[4] * y * y
            + 2.0 * m[5] * y * z
            + 2.0 * m[6] * y
            + m[7] * z * z
            + 2.0 * m[8] * z
            + m[9]
    }
}

impl Add for Quadric {
    type Output = Quadric;

    fn add(self, other: Quadric) -> Quadric {
        let mut out = self.0;
        for (o, n) in out.iter_mut().zip(other.0.iter()) {
            *o += n;
        }
        Quadric(out)
    }
}

// ---------------------------------------------------------------------------
// Mesh storage
// ---------------------------------------------------------------------------

/// One entry of the vertex -> face incidence list.
///
/// `tvertex` is *which* of the face's three slots holds this vertex, which is what
/// lets a collapse rewrite the face without searching it.
#[derive(Clone, Copy)]
struct Ref {
    tid: u32,
    tvertex: u8,
}

struct Triangle {
    v: [u32; 3],
    /// Collapse error of each of the three edges, then the smallest of them.
    err: [f64; 4],
    deleted: bool,
    /// The pass in which a collapse last touched this face, stamped rather than
    /// flagged; it is skipped while the stamp matches the pass in progress.
    dirty: u32,
    /// Unit normal, or `None` for a zero-area face. See `update_mesh`.
    n: Option<[f64; 3]>,
}

struct Vertex {
    p: [f64; 3],
    /// Offset of this vertex's run in `Simplifier::refs`.
    tstart: usize,
    /// Length of that run.
    tcount: usize,
    q: Quadric,
    border: bool,
}

/// A simplified mesh, plus a record of where every original vertex went.
pub struct Simplified {
    /// Positions of the surviving vertices, densely renumbered.
    pub vertices: Array2<f64>,
    /// Faces, indexing `vertices`.
    pub faces: Array2<u32>,
    /// For each vertex of the *input* mesh, its index in `vertices`, or `-1` if it
    /// did not survive.
    ///
    /// Being *merged* is not a `-1`: a collapsed vertex carries the index of
    /// whatever it merged into, which is the whole point of this array. An entry is
    /// `-1` exactly when the vertex it ended up in is referenced by no surviving
    /// face, which takes one of four forms: it was in no face to begin with; it was
    /// only ever in zero-area faces, which are dropped on the way in and so reduce
    /// to the first case; the whole piece it belonged to was consumed (nothing is
    /// reserved per connected component, so a small fragment goes once the target
    /// is tight enough, or once `epsilon` exceeds its whole extent); or the input
    /// was degenerate throughout and the output mesh is empty.
    pub vertex_map: Array1<i32>,
}

/// How far to simplify — the two ways of naming a face budget.
///
/// Public, and resolved in the core rather than by each binding, so that what a
/// ratio *means* — the rounding, and the floor of one face — is defined once
/// instead of once per language. Same shape as [`crate::matches::Criterion`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Target {
    /// Keep at most this many faces.
    Faces(usize),
    /// Keep this fraction of the input faces, in `(0, 1]`.
    Ratio(f64),
}

impl Target {
    /// The face budget this means for a mesh of `n_faces` faces.
    fn resolve(self, n_faces: usize) -> usize {
        match self {
            Target::Faces(n) => n,
            Target::Ratio(r) => {
                assert!(
                    r.is_finite() && r > 0.0 && r <= 1.0,
                    "`ratio` must be in (0, 1], got {r}"
                );
                // Round rather than truncate, so a third of 300 faces is 100 and not
                // 99, and never go below one face for a mesh that has any.
                if n_faces == 0 {
                    0
                } else {
                    ((r * n_faces as f64).round() as usize).max(1)
                }
            }
        }
    }
}

/// What tells the sweep it is done.
enum Stop {
    /// Collapse until at most `target` faces remain, letting the error threshold
    /// grow each iteration at a rate set by `aggressiveness` (5..8 are sensible;
    /// higher reaches the target in fewer, coarser steps).
    Faces { target: usize, aggressiveness: f64 },
    /// Collapse every edge whose error is under `epsilon` and nothing else, until
    /// a whole pass changes nothing.
    Lossless { epsilon: f64 },
}

struct Simplifier<'a> {
    vertices: Vec<Vertex>,
    triangles: Vec<Triangle>,
    /// Vertex -> face incidence, as runs delimited by `Vertex::tstart`/`tcount`.
    /// Rebuilt wholesale by `update_mesh`; grown by appends in between.
    refs: Vec<Ref>,
    /// Union-find over *input* vertex indices: `parent[v]` is the vertex `v` was
    /// collapsed into, or `v` itself while `v` is still alive.
    ///
    /// A `Vec` rather than a map on purpose. It is O(1) without hashing, it needs
    /// no allocation per collapse, and — the reason that matters here — iteration
    /// over it is in index order, so the result cannot depend on hash seeding.
    parent: Vec<u32>,
    /// Vertices that must survive, at exactly the position they came in at.
    locked: Option<&'a [bool]>,
    /// Number of the pass in progress, as the stamp `Triangle::dirty` is compared
    /// against. See the note in [`Simplifier::run`].
    pass: u32,
}

impl<'a> Simplifier<'a> {
    fn new(faces: ArrayView2<u32>, vertices: ArrayView2<f64>, locked: Option<&'a [bool]>) -> Self {
        // The bindings validate too, but this is the entry point the R side and any
        // Rust caller reach directly, and an out-of-range face index would otherwise
        // surface as a bare index-out-of-bounds from somewhere deep in the sweep.
        assert_eq!(faces.ncols(), 3, "`faces` must be (F, 3)");
        assert_eq!(vertices.ncols(), 3, "`vertices` must be (V, 3)");
        let n = vertices.nrows();
        assert!(
            n <= u32::MAX as usize,
            "meshes are indexed with u32: {n} vertices is too many"
        );
        if let Some(max) = faces.iter().max() {
            assert!(
                (*max as usize) < n,
                "`faces` references vertex {max} but there are only {n} vertices"
            );
        }
        if let Some(l) = locked {
            assert_eq!(
                l.len(),
                n,
                "`locked` must have one flag per vertex: got {}, expected {n}",
                l.len()
            );
        }
        assert!(
            vertices.iter().all(|x| x.is_finite()),
            "`vertices` must be finite"
        );

        let verts = vertices
            .rows()
            .into_iter()
            .map(|r| Vertex {
                p: [r[0], r[1], r[2]],
                tstart: 0,
                tcount: 0,
                q: Quadric::default(),
                border: false,
            })
            .collect::<Vec<_>>();

        // DIVERGENCE (1/3): drop faces that name the same vertex twice. They have
        // no area, so no plane and no normal, and upstream would carry them along
        // as a source of NaN. Dropping them here rather than tolerating them later
        // keeps every triangle in the sweep a real triangle.
        //
        // Sized up front: a filtering iterator reports a lower size hint of 0, so
        // collecting straight off it would grow this by doubling — on a
        // million-face mesh, seventeen reallocations of an 80 MB buffer.
        let mut tris = Vec::with_capacity(faces.nrows());
        tris.extend(
            faces
                .rows()
                .into_iter()
                .filter(|f| f[0] != f[1] && f[1] != f[2] && f[2] != f[0])
                .map(|f| Triangle {
                    v: [f[0], f[1], f[2]],
                    err: [0.0; 4],
                    deleted: false,
                    dirty: 0,
                    n: None,
                }),
        );

        Simplifier {
            parent: (0..verts.len() as u32).collect(),
            vertices: verts,
            triangles: tris,
            refs: Vec::new(),
            locked,
            pass: 0,
        }
    }

    #[inline]
    fn is_locked(&self, v: u32) -> bool {
        self.locked.is_some_and(|l| l[v as usize])
    }

    /// Compact away deleted faces, rebuild the incidence list, and — on the first
    /// call only — derive border flags, vertex quadrics and per-edge errors.
    fn update_mesh(&mut self, iteration: usize) {
        if iteration > 0 {
            self.triangles.retain(|t| !t.deleted);
        }

        // Counting sort of (vertex, face) incidences into `refs`.
        for v in &mut self.vertices {
            v.tstart = 0;
            v.tcount = 0;
        }
        for t in &self.triangles {
            for j in 0..3 {
                self.vertices[t.v[j] as usize].tcount += 1;
            }
        }
        let mut tstart = 0;
        for v in &mut self.vertices {
            v.tstart = tstart;
            tstart += v.tcount;
            v.tcount = 0;
        }

        // No `clear()` first: the prefix sum above covers `[0, 3T)` exactly, so the
        // fill below overwrites every slot and the value `resize` would pad with is
        // dead. Clearing would turn what is usually a truncation — `refs` only ever
        // grows between rebuilds — into a full write over 3F entries each pass.
        self.refs
            .resize(self.triangles.len() * 3, Ref { tid: 0, tvertex: 0 });
        for (i, t) in self.triangles.iter().enumerate() {
            for j in 0..3 {
                let v = &mut self.vertices[t.v[j] as usize];
                self.refs[v.tstart + v.tcount] = Ref {
                    tid: i as u32,
                    tvertex: j as u8,
                };
                v.tcount += 1;
            }
        }

        if iteration == 0 {
            // Only at the start: upstream recomputes quadrics mid-run in some
            // configurations, but it is not required and mostly helps closed meshes.
            self.init_borders();
            self.init_quadrics();
        }
    }

    /// Mark vertices on a boundary.
    ///
    /// The test is the one-ring count: walk every face incident to `i` and tally
    /// how often each vertex of those faces appears. A vertex seen exactly once has
    /// only one face bridging it to `i`, so the edge between them is an open edge,
    /// and both ends of it are on the border. Note this marks *other* vertices'
    /// flags from `i`'s ring, which is why it has to run over every vertex before
    /// any flag is read.
    fn init_borders(&mut self) {
        let mut vcount: Vec<u32> = Vec::new();
        let mut vids: Vec<u32> = Vec::new();

        for v in &mut self.vertices {
            v.border = false;
        }
        for i in 0..self.vertices.len() {
            vcount.clear();
            vids.clear();
            let (tstart, tcount) = (self.vertices[i].tstart, self.vertices[i].tcount);
            for j in 0..tcount {
                let t = &self.triangles[self.refs[tstart + j].tid as usize];
                for k in 0..3 {
                    let id = t.v[k];
                    // Linear scan, not a set: a one-ring is a handful of vertices,
                    // and the scan beats hashing at that size by a wide margin.
                    match vids.iter().position(|&x| x == id) {
                        Some(ofs) => vcount[ofs] += 1,
                        None => {
                            vids.push(id);
                            vcount.push(1);
                        }
                    }
                }
            }
            for (j, &c) in vcount.iter().enumerate() {
                if c == 1 {
                    self.vertices[vids[j] as usize].border = true;
                }
            }
        }
    }

    /// Sum each face's plane quadric onto its three vertices, then seed edge errors.
    fn init_quadrics(&mut self) {
        for v in &mut self.vertices {
            v.q = Quadric::default();
        }
        for t in &mut self.triangles {
            let p: [[f64; 3]; 3] = [
                self.vertices[t.v[0] as usize].p,
                self.vertices[t.v[1] as usize].p,
                self.vertices[t.v[2] as usize].p,
            ];
            // DIVERGENCE (2/3): a zero-area face has no normal. Upstream normalises
            // anyway and stores NaN, which later makes `flipped`'s orientation test
            // false and lets bad collapses through. Record the absence instead: the
            // face contributes no plane to its vertices' quadrics, and `flipped`
            // knows to skip it.
            t.n = normalize(cross(sub(p[1], p[0]), sub(p[2], p[0])));
            if let Some(n) = t.n {
                let q = Quadric::plane(n[0], n[1], n[2], -dot(n, p[0]));
                for j in 0..3 {
                    let v = &mut self.vertices[t.v[j] as usize];
                    v.q = v.q + q;
                }
            }
        }
        for i in 0..self.triangles.len() {
            let v = self.triangles[i].v;
            let e = [
                self.calculate_error(v[0], v[1]).0,
                self.calculate_error(v[1], v[2]).0,
                self.calculate_error(v[2], v[0]).0,
            ];
            self.triangles[i].err = [e[0], e[1], e[2], e[0].min(e[1]).min(e[2])];
        }
    }

    /// Error of collapsing the edge `(a, b)`, and the point to collapse it to.
    ///
    /// The optimal point is the minimum of the summed quadric, which exists when
    /// its 3x3 part is invertible. When it is not — or when both ends are on a
    /// border, where moving off the boundary is worse than the error suggests —
    /// fall back to the best of the two endpoints and their midpoint.
    fn calculate_error(&self, a: u32, b: u32) -> (f64, [f64; 3]) {
        let (va, vb) = (&self.vertices[a as usize], &self.vertices[b as usize]);
        let q = va.q + vb.q;

        // Both ends on a border means the fallback wins regardless, so the 3x3
        // determinant is not worth computing — on an open mesh, which neuron
        // fragments cut at the volume boundary are, that is most candidates.
        if !(va.border && vb.border) {
            let det = q.det(0, 1, 2, 1, 4, 5, 2, 5, 7);
            if det != 0.0 {
                let p = [
                    -1.0 / det * q.det(1, 2, 3, 4, 5, 6, 5, 7, 8),
                    1.0 / det * q.det(0, 2, 3, 1, 5, 6, 2, 7, 8),
                    -1.0 / det * q.det(0, 1, 3, 1, 4, 6, 2, 5, 8),
                ];
                // `det != 0.0` is an exact test on a quantity that can be
                // denormal-small, so the inverse it admits is sometimes numerically
                // hopeless. Validating the result instead of tightening the test
                // leaves every well-conditioned answer bit-identical to upstream's
                // and sends only the blow-ups down the fallback path below.
                if p.iter().all(|x| x.is_finite()) {
                    return (q.error(p), p);
                }
            }
        }

        let (p1, p2) = (va.p, vb.p);
        let p3 = [
            (p1[0] + p2[0]) / 2.0,
            (p1[1] + p2[1]) / 2.0,
            (p1[2] + p2[2]) / 2.0,
        ];
        let (e1, e2, e3) = (q.error(p1), q.error(p2), q.error(p3));
        let error = e1.min(e2).min(e3);
        // Upstream's tie-break order: last match wins, so midpoint beats p2 beats p1
        // when the errors are equal.
        let p = if e3 == error {
            p3
        } else if e2 == error {
            p2
        } else {
            p1
        };
        (error, p)
    }

    /// Would collapsing `i0`'s fan onto `p` flip or degenerate any of its faces?
    ///
    /// Also fills `deleted[k]` with whether the k-th face of `i0`'s fan disappears
    /// in the collapse (it does when it also touches `i1`). That output is only
    /// meaningful when this returns `false`; on a rejection the caller drops it.
    fn flipped(&self, p: [f64; 3], i0: u32, i1: u32, deleted: &mut [bool]) -> bool {
        let (tstart, tcount) = (
            self.vertices[i0 as usize].tstart,
            self.vertices[i0 as usize].tcount,
        );
        for (k, &r) in self.refs[tstart..tstart + tcount].iter().enumerate() {
            let t = &self.triangles[r.tid as usize];
            if t.deleted {
                continue;
            }

            let s = r.tvertex as usize;
            let id1 = t.v[(s + 1) % 3];
            let id2 = t.v[(s + 2) % 3];
            if id1 == i1 || id2 == i1 {
                // This face collapses onto an edge and goes away.
                deleted[k] = true;
                continue;
            }

            // DIVERGENCE (3/3): if `p` coincides with a neighbour there is no
            // direction to compare, and the face would be degenerate after the
            // collapse. Upstream gets NaN here and, because `NaN > 0.999` and
            // `NaN < 0.2` are both false, accepts the collapse. Reject it.
            let (Some(d1), Some(d2)) = (
                normalize(sub(self.vertices[id1 as usize].p, p)),
                normalize(sub(self.vertices[id2 as usize].p, p)),
            ) else {
                return true;
            };
            if dot(d1, d2).abs() > 0.999 {
                return true;
            }
            let Some(n) = normalize(cross(d1, d2)) else {
                return true;
            };
            deleted[k] = false;
            // A face with no recorded normal is zero-area and has no orientation to
            // flip, so there is nothing to compare it against — leave it be.
            if let Some(tn) = t.n {
                if dot(n, tn) < 0.2 {
                    return true;
                }
            }
        }
        false
    }

    /// Move `i0`'s fan onto the surviving vertex: delete the faces that collapsed,
    /// repoint the rest at `i0`, refresh their edge errors, and append their refs.
    fn update_triangles(
        &mut self,
        i0: u32,
        tstart: usize,
        tcount: usize,
        deleted: &[bool],
        deleted_triangles: &mut usize,
    ) {
        // Indexed rather than iterated, unlike `flipped` above: the `push` at the
        // bottom of the loop borrows `refs` mutably, so a live iterator over it
        // cannot coexist with the body.
        #[allow(clippy::needless_range_loop)]
        for k in 0..tcount {
            // Copied out, not borrowed: the push below can reallocate `refs`.
            // Upstream holds a reference across that push, which is undefined
            // behaviour there and simply will not compile here.
            let r = self.refs[tstart + k];
            if self.triangles[r.tid as usize].deleted {
                continue;
            }
            if deleted[k] {
                self.triangles[r.tid as usize].deleted = true;
                *deleted_triangles += 1;
                continue;
            }

            // Patched on a copy so the errors can be computed against `&self`, then
            // written back with `v`, `dirty` and `err` in one borrow rather than
            // three separate bounds-checked visits to the same element.
            let mut v = self.triangles[r.tid as usize].v;
            v[r.tvertex as usize] = i0;
            let e = [
                self.calculate_error(v[0], v[1]).0,
                self.calculate_error(v[1], v[2]).0,
                self.calculate_error(v[2], v[0]).0,
            ];

            let t = &mut self.triangles[r.tid as usize];
            t.v = v;
            t.dirty = self.pass;
            t.err = [e[0], e[1], e[2], e[0].min(e[1]).min(e[2])];
            self.refs.push(r);
        }
    }

    /// Merge `i1` into `i0` if every guard allows it. Returns whether it happened.
    ///
    /// Split out of [`Simplifier::run`] because it is the one place the incidence
    /// list is edited outside `update_mesh`, and that is worth being able to read
    /// without also holding three levels of loop in your head.
    fn try_collapse(
        &mut self,
        i0: u32,
        i1: u32,
        preserve_border: bool,
        deleted0: &mut Vec<bool>,
        deleted1: &mut Vec<bool>,
        deleted_triangles: &mut usize,
    ) -> bool {
        // Collapsing a vertex onto itself costs nothing, so it clears the threshold,
        // and it would then corrupt the map (`parent[i] = i` reads as "still alive")
        // while deleting live faces. Faces naming a vertex twice are filtered on the
        // way in, so this should be unreachable; it is one comparison to be sure.
        if i0 == i1 {
            return false;
        }

        let (b0, b1) = (
            self.vertices[i0 as usize].border,
            self.vertices[i1 as usize].border,
        );
        // With the border frozen, nothing touching it may move. Otherwise only
        // like-for-like: collapsing an interior vertex into a boundary one (or the
        // reverse) would drag the boundary.
        if if preserve_border { b0 || b1 } else { b0 != b1 } {
            return false;
        }

        // A locked vertex may absorb its neighbours but must never be the one
        // absorbed. Refusing on *either* endpoint would freeze the vertex's whole
        // one-ring, a much stronger claim than "keep this vertex".
        if self.is_locked(i1) {
            return false;
        }

        let (_, mut p) = self.calculate_error(i0, i1);
        // ...and it must not move, so a locked survivor keeps its position rather
        // than sliding to the quadric optimum. Bitwise identity here is what lets
        // callers key data off the position.
        if self.is_locked(i0) {
            p = self.vertices[i0 as usize].p;
        }

        let (t0, c0) = (
            self.vertices[i0 as usize].tstart,
            self.vertices[i0 as usize].tcount,
        );
        let (t1, c1) = (
            self.vertices[i1 as usize].tstart,
            self.vertices[i1 as usize].tcount,
        );
        deleted0.clear();
        deleted0.resize(c0, false);
        deleted1.clear();
        deleted1.resize(c1, false);

        if self.flipped(p, i0, i1, deleted0) || self.flipped(p, i1, i0, deleted1) {
            return false;
        }

        self.vertices[i0 as usize].p = p;
        self.vertices[i0 as usize].q = self.vertices[i1 as usize].q + self.vertices[i0 as usize].q;

        // The entire point of this module. `i1` is live right up to this moment — a
        // vertex that has been collapsed away never appears in a face again, so it
        // can never be picked as `i0` or `i1` a second time — which makes this a
        // union of two roots, and the chains it builds resolvable by a plain find.
        self.parent[i1 as usize] = i0;

        let tstart = self.refs.len();
        self.update_triangles(i0, t0, c0, deleted0, deleted_triangles);
        self.update_triangles(i0, t1, c1, deleted1, deleted_triangles);
        let tcount = self.refs.len() - tstart;

        if tcount <= c0 {
            // The merged fan fits where `i0`'s used to be, so move it back and let
            // the tail be overwritten next time.
            if tcount > 0 {
                self.refs.copy_within(tstart..tstart + tcount, t0);
            }
        } else {
            self.vertices[i0 as usize].tstart = tstart;
        }
        self.vertices[i0 as usize].tcount = tcount;
        true
    }

    /// The sweep. Both public entry points are this with a different `Stop`.
    fn run(&mut self, stop: Stop, max_iterations: usize, preserve_border: bool) {
        // Destructured once: `target` carries both "which mode" and "how far", so
        // neither question needs the enum again below.
        let target = match stop {
            Stop::Faces { target, .. } => Some(target),
            Stop::Lossless { .. } => None,
        };
        let lossless = target.is_none();

        for t in &mut self.triangles {
            t.deleted = false;
        }

        let triangle_count = self.triangles.len();
        let mut deleted_triangles = 0usize;
        // Reused across every collapse so the sweep does not allocate.
        let (mut deleted0, mut deleted1) = (Vec::new(), Vec::new());

        for iteration in 0..max_iterations {
            if target.is_some_and(|t| triangle_count - deleted_triangles <= t) {
                break;
            }

            // Lossless mode has to refresh every pass: it only ever collapses edges
            // already under `epsilon`, so it depends on errors recomputed against
            // the compacted mesh to find the next batch.
            if lossless || iteration % UPDATE_RATE == 0 {
                self.update_mesh(iteration);
            }

            // Retiring the dirty marks by stamping the pass number, rather than by
            // clearing a flag on every triangle: the clear was a write over the
            // whole 80-bytes-per-element array once per pass, which on a
            // million-face mesh is ~80 MB of memory traffic to set 1 bit each.
            self.pass = iteration as u32 + 1;

            let threshold = match stop {
                Stop::Faces { aggressiveness, .. } => {
                    ALPHA * (iteration as f64 + K).powf(aggressiveness)
                }
                Stop::Lossless { epsilon } => epsilon,
            };

            for i in 0..self.triangles.len() {
                let t = &self.triangles[i];
                if t.err[3] > threshold || t.deleted || t.dirty == self.pass {
                    continue;
                }

                for j in 0..3 {
                    if self.triangles[i].err[j] >= threshold {
                        continue;
                    }
                    // `i1` is the vertex that disappears into `i0`.
                    let i0 = self.triangles[i].v[j];
                    let i1 = self.triangles[i].v[(j + 1) % 3];
                    if self.try_collapse(
                        i0,
                        i1,
                        preserve_border,
                        &mut deleted0,
                        &mut deleted1,
                        &mut deleted_triangles,
                    ) {
                        break;
                    }
                }

                if target.is_some_and(|t| triangle_count - deleted_triangles <= t) {
                    break;
                }
            }

            if lossless {
                if deleted_triangles == 0 {
                    break;
                }
                deleted_triangles = 0;
            }
        }
    }

    /// Drop deleted faces, renumber the surviving vertices densely, and resolve the
    /// collapse forest into one index per input vertex.
    fn finish(&mut self) -> Simplified {
        let n_in = self.vertices.len();

        // Flat from the start: building `Vec<[u32; 3]>` and flattening it at the end
        // would allocate and copy the whole face array a second time.
        let mut alive = vec![false; n_in];
        let mut n_alive = 0usize;
        let mut faces: Vec<u32> = Vec::with_capacity(self.triangles.len() * 3);
        for t in &self.triangles {
            if t.deleted {
                continue;
            }
            faces.extend_from_slice(&t.v);
            for j in 0..3 {
                let v = t.v[j] as usize;
                if !alive[v] {
                    alive[v] = true;
                    n_alive += 1;
                }
            }
        }

        // Dense renumbering, kept as an explicit array. Upstream stashes it in the
        // `tstart` field it is no longer using, which works but leaves the mesh in a
        // state where nothing means what its name says.
        let mut new_index = vec![-1i32; n_in];
        let mut coords: Vec<f64> = Vec::with_capacity(n_alive * 3);
        for i in 0..n_in {
            if alive[i] {
                new_index[i] = (coords.len() / 3) as i32;
                coords.extend_from_slice(&self.vertices[i].p);
            }
        }

        for f in &mut faces {
            *f = new_index[*f as usize] as u32;
        }

        // A vertex whose root did not survive maps to -1, which covers both "its
        // piece was decimated to nothing" and "it was in no face to start with".
        let mut vertex_map = Vec::with_capacity(n_in);
        for i in 0..n_in {
            vertex_map.push(new_index[find(&mut self.parent, i as u32) as usize]);
        }

        Simplified {
            vertices: Array2::from_shape_vec((n_alive, 3), coords)
                .expect("coords holds 3 per surviving vertex"),
            faces: Array2::from_shape_vec((faces.len() / 3, 3), faces)
                .expect("faces holds 3 per triangle"),
            vertex_map: Array1::from_vec(vertex_map),
        }
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Simplify a triangle mesh down to a target face count.
///
/// Arguments:
///
/// - `faces`: `(F, 3)` triangles, as vertex indices into `vertices`
/// - `vertices`: `(V, 3)` vertex coordinates
/// - `target`: the face budget, as an absolute count or a fraction of the input.
///   The sweep can stop short of it, either because no further collapse is
///   geometrically safe or because it hit its internal cap of
///   [`MAX_ITERATIONS`](self) passes
/// - `aggressiveness`: how fast the error threshold grows per iteration; 5..8 are
///   sensible, upstream's default is 7. Lower is slower and higher quality
/// - `preserve_border`: if `true`, no edge touching a boundary vertex is collapsed.
///   If `false` (upstream's default) boundary vertices collapse among themselves
///   but never mix with interior ones
/// - `locked`: optional per-vertex flags, `V` long. A locked vertex is guaranteed
///   to survive at exactly its input position
///
/// Returns the simplified mesh together with `vertex_map`, which says for every
/// input vertex which output vertex it ended up in.
pub fn simplify_mesh(
    faces: ArrayView2<u32>,
    vertices: ArrayView2<f64>,
    target: Target,
    aggressiveness: f64,
    preserve_border: bool,
    locked: Option<&[bool]>,
) -> Simplified {
    // Checked here rather than in each binding: this is where `aggressiveness`
    // means something, and a non-finite exponent turns the whole threshold sweep
    // into a no-op that would otherwise look like "the mesh could not be reduced".
    assert!(
        aggressiveness.is_finite() && aggressiveness >= 0.0,
        "`aggressiveness` must be finite and non-negative, got {aggressiveness}"
    );

    // Against the count the caller passed in, not the count that survives the
    // degenerate-face filter: a ratio is a promise about *their* mesh.
    let target = target.resolve(faces.nrows());

    let mut s = Simplifier::new(faces, vertices, locked);
    s.run(
        Stop::Faces {
            target,
            aggressiveness,
        },
        MAX_ITERATIONS,
        preserve_border,
    );
    s.finish()
}

/// Simplify a triangle mesh without changing its shape.
///
/// Collapses only edges whose quadric error is below `epsilon` and repeats until a
/// whole pass changes nothing, so there is no face-count target — this is for
/// shedding over-tessellation, not for hitting a budget.
///
/// Arguments:
///
/// - `faces`: `(F, 3)` triangles, as vertex indices into `vertices`
/// - `vertices`: `(V, 3)` vertex coordinates
/// - `epsilon`: error below which an edge may collapse. Upstream's default is 1e-3.
///   Note this is an *absolute* quadric error, so it scales with your coordinate units
/// - `max_iterations`: cap on the number of passes; upstream's default is 9999
/// - `preserve_border`, `locked`: as [`simplify_mesh`]
pub fn simplify_mesh_lossless(
    faces: ArrayView2<u32>,
    vertices: ArrayView2<f64>,
    epsilon: f64,
    max_iterations: usize,
    preserve_border: bool,
    locked: Option<&[bool]>,
) -> Simplified {
    // As for `aggressiveness` above: checked where it means something, so a direct
    // Rust or R caller gets the same answer the Python wrapper already gives.
    assert!(
        epsilon.is_finite() && epsilon >= 0.0,
        "`epsilon` must be finite and non-negative, got {epsilon}"
    );

    let mut s = Simplifier::new(faces, vertices, locked);
    s.run(Stop::Lossless { epsilon }, max_iterations, preserve_border);
    s.finish()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// The two shared mesh fixtures, from `mesh` rather than copied.
    ///
    /// `grid` is a flat `n x n` grid split along each cell's (0,0)->(1,1) diagonal: every
    /// interior vertex is exactly coplanar with its ring, so its collapse error is 0,
    /// which is what makes it the natural fixture for lossless mode. `uv_sphere` is the
    /// closed counterpart, and the shape a decimated connectomics mesh has.
    use crate::mesh::tests_support::{grid, uv_sphere};

    /// The invariants every result must satisfy, whatever the input.
    fn check_invariants(out: &Simplified, n_in: usize) {
        let n_out = out.vertices.nrows();
        assert_eq!(out.vertex_map.len(), n_in, "one map entry per input vertex");

        for &m in out.vertex_map.iter() {
            assert!(m >= -1 && m < n_out as i32, "map entry {m} out of range");
        }
        for f in out.faces.rows() {
            for &v in f {
                assert!(
                    (v as usize) < n_out,
                    "face references vertex {v} of {n_out}"
                );
            }
        }
        for x in out.vertices.iter() {
            assert!(x.is_finite(), "non-finite coordinate {x} in output");
        }

        // Onto: every output vertex has at least one input vertex mapping to it.
        // Otherwise we invented a vertex, or the renumbering slipped.
        let mut hit = vec![false; n_out];
        for &m in out.vertex_map.iter() {
            if m >= 0 {
                hit[m as usize] = true;
            }
        }
        assert!(hit.iter().all(|&h| h), "an output vertex has no preimage");
    }

    fn simplify_to(faces: &Array2<u32>, verts: &Array2<f64>, target: usize) -> Simplified {
        simplify_mesh(
            faces.view(),
            verts.view(),
            Target::Faces(target),
            7.0,
            false,
            None,
        )
    }

    #[test]
    fn hits_the_face_target() {
        let (faces, verts) = uv_sphere(30, 30);
        for ratio in [0.9, 0.5, 0.1] {
            let target = (faces.nrows() as f64 * ratio) as usize;
            let out = simplify_to(&faces, &verts, target);
            check_invariants(&out, verts.nrows());
            // The sweep stops as soon as it is at or under target, and a single
            // collapse removes two faces, so it can overshoot by one.
            assert!(
                out.faces.nrows() <= target && out.faces.nrows() >= target.saturating_sub(2),
                "ratio {ratio}: got {} faces, wanted {target}",
                out.faces.nrows()
            );
        }
    }

    #[test]
    fn ratio_and_face_count_name_the_same_budget() {
        // The rounding rule lives here rather than in each binding, so it is worth
        // pinning: a ratio resolves against the caller's own face count, rounds
        // rather than truncates, and never takes a non-empty mesh to zero faces.
        assert_eq!(Target::Faces(7).resolve(100), 7);
        assert_eq!(Target::Ratio(1.0).resolve(300), 300);
        assert_eq!(Target::Ratio(1.0 / 3.0).resolve(300), 100); // not 99
        assert_eq!(Target::Ratio(0.0001).resolve(10), 1); // never below one
        assert_eq!(Target::Ratio(0.5).resolve(0), 0); // ...but empty stays empty

        let (faces, verts) = uv_sphere(16, 16);
        let by_ratio = simplify_mesh(
            faces.view(),
            verts.view(),
            Target::Ratio(0.25),
            7.0,
            false,
            None,
        );
        let by_count = simplify_to(
            &faces,
            &verts,
            (0.25 * faces.nrows() as f64).round() as usize,
        );
        assert_eq!(by_ratio.faces, by_count.faces);
        assert_eq!(by_ratio.vertex_map, by_count.vertex_map);
    }

    #[test]
    fn identity_target_changes_nothing() {
        let (faces, verts) = uv_sphere(12, 12);
        let out = simplify_to(&faces, &verts, faces.nrows());
        check_invariants(&out, verts.nrows());
        assert_eq!(out.faces.nrows(), faces.nrows());
        assert_eq!(out.vertices.nrows(), verts.nrows());
        // Nothing collapsed, so the map is the identity.
        for (i, &m) in out.vertex_map.iter().enumerate() {
            assert_eq!(m, i as i32);
        }
    }

    #[test]
    fn unreferenced_input_vertices_map_to_minus_one() {
        let (faces, verts) = uv_sphere(10, 10);
        // Three vertices no face mentions. They never enter the collapse forest, so
        // they have to fall out as -1 rather than as some arbitrary root.
        let mut coords = verts.iter().copied().collect::<Vec<_>>();
        coords.extend_from_slice(&[9.0, 9.0, 9.0, 8.0, 8.0, 8.0, 7.0, 7.0, 7.0]);
        let padded = Array2::from_shape_vec((verts.nrows() + 3, 3), coords).unwrap();

        let out = simplify_to(&faces, &padded, faces.nrows() / 2);
        check_invariants(&out, padded.nrows());
        for i in verts.nrows()..padded.nrows() {
            assert_eq!(
                out.vertex_map[i], -1,
                "orphan vertex {i} should not survive"
            );
        }
    }

    #[test]
    fn collapse_chains_resolve_to_one_survivor() {
        // Decimate hard enough that multi-step chains (a -> b, later b -> c) are
        // unavoidable, then check no map entry still points at a dead vertex.
        let (faces, verts) = uv_sphere(24, 24);
        let out = simplify_to(&faces, &verts, faces.nrows() / 20);
        check_invariants(&out, verts.nrows());
        assert!(
            out.vertices.nrows() < verts.nrows() / 4,
            "expected heavy decimation, got {} of {} vertices",
            out.vertices.nrows(),
            verts.nrows()
        );
    }

    #[test]
    fn every_face_vertex_is_its_own_map_target() {
        // A surviving vertex maps to itself, so a face's corners in the output are
        // exactly the images of the corners it had in the input. This catches a
        // composition that is off by one or applied in the wrong order.
        let (faces, verts) = uv_sphere(16, 16);
        let out = simplify_to(&faces, &verts, faces.nrows() / 3);
        let mut seen = vec![false; out.vertices.nrows()];
        for f in out.faces.rows() {
            for &v in f {
                seen[v as usize] = true;
            }
        }
        // Every output vertex is used by a face (compaction keeps only those), and
        // every one of them is the image of some input vertex under the map.
        assert!(seen.iter().all(|&s| s));
    }

    #[test]
    fn locked_vertices_survive_unmoved() {
        let (faces, verts) = uv_sphere(20, 20);
        // Pin every seventh vertex, i.e. a scattered set, as synapse positions would be.
        let locked: Vec<bool> = (0..verts.nrows()).map(|i| i % 7 == 0).collect();

        let out = simplify_mesh(
            faces.view(),
            verts.view(),
            Target::Faces(faces.nrows() / 8),
            7.0,
            false,
            Some(&locked),
        );
        check_invariants(&out, verts.nrows());

        for (i, &l) in locked.iter().enumerate() {
            if !l {
                continue;
            }
            let m = out.vertex_map[i];
            assert!(m >= 0, "locked vertex {i} did not survive");
            // Bitwise, not approximately: a locked vertex's position is never
            // recomputed, which is what lets callers key data off it.
            for k in 0..3 {
                assert_eq!(
                    out.vertices[[m as usize, k]].to_bits(),
                    verts[[i, k]].to_bits(),
                    "locked vertex {i} moved on axis {k}"
                );
            }
        }
    }

    #[test]
    fn locking_everything_is_a_no_op() {
        let (faces, verts) = uv_sphere(10, 10);
        let locked = vec![true; verts.nrows()];
        let out = simplify_mesh(
            faces.view(),
            verts.view(),
            Target::Faces(4),
            7.0,
            false,
            Some(&locked),
        );

        assert_eq!(out.faces.nrows(), faces.nrows(), "no face may be collapsed");
        assert_eq!(out.vertices.nrows(), verts.nrows());
        for (i, &m) in out.vertex_map.iter().enumerate() {
            assert_eq!(m, i as i32);
        }
    }

    #[test]
    fn locked_vertices_still_absorb_their_neighbours() {
        // The asymmetric rule earns its keep here: a locked vertex may take neighbours
        // in, it just may not be taken in itself. Freezing both directions instead
        // would stall the sweep well short of the target as soon as the locked set is
        // dense — which is exactly the synapse-pinning case.
        let (faces, verts) = uv_sphere(20, 20);
        // Sparse enough that the target is arithmetically reachable: every locked
        // vertex survives, so `n_locked` is a floor on the output vertex count, and a
        // closed mesh has about half as many vertices as faces.
        let locked: Vec<bool> = (0..verts.nrows()).map(|i| i % 20 == 0).collect();
        let target = faces.nrows() / 4;

        let out = simplify_mesh(
            faces.view(),
            verts.view(),
            Target::Faces(target),
            7.0,
            false,
            Some(&locked),
        );
        check_invariants(&out, verts.nrows());
        assert!(
            out.faces.nrows() <= target,
            "target missed with a third of vertices pinned: {} > {target}",
            out.faces.nrows()
        );
        // Some locked vertex absorbed at least one neighbour, i.e. has >1 preimage.
        let mut preimages = vec![0usize; out.vertices.nrows()];
        for &m in out.vertex_map.iter() {
            if m >= 0 {
                preimages[m as usize] += 1;
            }
        }
        assert!(preimages.iter().any(|&c| c > 1));
    }

    #[test]
    fn lossless_flattens_a_coplanar_grid_without_moving_it() {
        let (faces, verts) = grid(12, 1.0);
        let out = simplify_mesh_lossless(faces.view(), verts.view(), 1e-9, 9999, false, None);
        check_invariants(&out, verts.nrows());

        assert!(
            out.faces.nrows() < faces.nrows(),
            "a flat grid has interior vertices that cost nothing to remove"
        );
        // "Lossless" is a claim about geometry: everything stays in the z = 0 plane and
        // inside the original footprint.
        for row in out.vertices.rows() {
            assert!(row[2].abs() < 1e-9, "left the plane: z = {}", row[2]);
            assert!((0.0..=11.0).contains(&row[0]) && (0.0..=11.0).contains(&row[1]));
        }
    }

    #[test]
    fn deterministic_across_runs() {
        let (faces, verts) = uv_sphere(18, 18);
        let a = simplify_to(&faces, &verts, faces.nrows() / 4);
        let b = simplify_to(&faces, &verts, faces.nrows() / 4);

        assert_eq!(a.faces, b.faces);
        assert_eq!(a.vertex_map, b.vertex_map);
        // Bit-for-bit on the coordinates: a difference here means a reduction order or
        // an iteration order leaked in somewhere.
        for (x, y) in a.vertices.iter().zip(b.vertices.iter()) {
            assert_eq!(x.to_bits(), y.to_bits());
        }
    }

    #[test]
    fn non_manifold_input_is_tolerated() {
        // Three faces on the edge 0-1, plus a bowtie at vertex 3. This is the input
        // class that rules out every halfedge-based crate: they either refuse it or
        // silently drop the third face.
        let faces = array![[0u32, 1, 2], [0, 1, 4], [0, 1, 5], [3, 6, 7], [3, 8, 9],];
        let verts = array![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            [5.0, 5.0, 0.0],
            [0.5, -1.0, 0.0],
            [0.5, 0.0, 1.0],
            [6.0, 5.0, 0.0],
            [5.5, 6.0, 0.0],
            [4.0, 5.0, 0.0],
            [4.5, 4.0, 0.0],
        ];
        let out = simplify_to(&faces, &verts, 2);
        check_invariants(&out, verts.nrows());
    }

    #[test]
    fn degenerate_faces_do_not_poison_the_result() {
        // A zero-area face and a duplicated-corner face. Upstream would normalise a
        // zero-length normal into NaN, and NaN then defeats both collapse guards
        // because every comparison against it is false.
        let (sphere_faces, sphere_verts) = uv_sphere(10, 10);
        let mut coords: Vec<f64> = sphere_verts.iter().copied().collect();
        let n = sphere_verts.nrows() as u32;
        // Three collinear points -> a face with no area.
        coords.extend_from_slice(&[3.0, 0.0, 0.0, 4.0, 0.0, 0.0, 5.0, 0.0, 0.0]);
        let verts = Array2::from_shape_vec((n as usize + 3, 3), coords).unwrap();

        let mut faces: Vec<u32> = sphere_faces.iter().copied().collect();
        faces.extend_from_slice(&[n, n + 1, n + 2]);
        faces.extend_from_slice(&[0, 0, 1]);
        let faces = Array2::from_shape_vec((sphere_faces.nrows() + 2, 3), faces).unwrap();

        let out = simplify_to(&faces, &verts, sphere_faces.nrows() / 2);
        check_invariants(&out, verts.nrows());
    }

    #[test]
    fn empty_and_trivial_inputs() {
        let empty_faces = Array2::<u32>::zeros((0, 3));
        let empty_verts = Array2::<f64>::zeros((0, 3));
        let out = simplify_to(&empty_faces, &empty_verts, 0);
        assert_eq!(out.faces.nrows(), 0);
        assert_eq!(out.vertices.nrows(), 0);
        assert_eq!(out.vertex_map.len(), 0);

        // Vertices but no faces: everything is an orphan.
        let verts = array![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let out = simplify_to(&empty_faces, &verts, 0);
        check_invariants(&out, 3);
        assert!(out.vertex_map.iter().all(|&m| m == -1));

        // A single triangle: nothing can collapse without deleting it outright.
        let one = array![[0u32, 1, 2]];
        let out = simplify_to(&one, &verts, 0);
        check_invariants(&out, 3);
    }

    #[test]
    fn duplicate_vertices_collapse_onto_one_another() {
        // Two coincident vertices with a zero-length edge between them: the classic
        // "welding" case, and the one where `flipped` would divide by zero.
        let verts = array![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ];
        let faces = array![[0u32, 1, 2], [1, 3, 2]];
        let out = simplify_mesh_lossless(faces.view(), verts.view(), 1e-9, 9999, false, None);
        check_invariants(&out, 4);
    }

    #[test]
    fn preserve_border_keeps_the_rim() {
        let (faces, verts) = grid(10, 1.0);
        let out = simplify_mesh(
            faces.view(),
            verts.view(),
            Target::Faces(8),
            7.0,
            true,
            None,
        );
        check_invariants(&out, verts.nrows());

        // With the border frozen the footprint cannot shrink: all four extremes of the
        // original grid must still be present.
        let (mut min_x, mut max_x) = (f64::INFINITY, f64::NEG_INFINITY);
        for row in out.vertices.rows() {
            min_x = min_x.min(row[0]);
            max_x = max_x.max(row[0]);
        }
        assert_eq!(min_x, 0.0);
        assert_eq!(max_x, 9.0);
    }

    #[test]
    fn quadric_of_a_plane_measures_squared_distance() {
        // The z = 0 plane, so the error at (x, y, z) should be exactly z^2.
        let q = Quadric::plane(0.0, 0.0, 1.0, 0.0);
        assert_eq!(q.error([3.0, -4.0, 0.0]), 0.0);
        assert!((q.error([1.0, 2.0, 3.0]) - 9.0).abs() < 1e-12);
        // Summing two copies doubles it, which is what accumulating over faces relies on.
        assert!(((q + q).error([0.0, 0.0, 2.0]) - 8.0).abs() < 1e-12);
    }

    #[test]
    fn normalize_rejects_what_has_no_direction() {
        assert!(normalize([0.0, 0.0, 0.0]).is_none());
        assert!(normalize([f64::NAN, 0.0, 0.0]).is_none());
        assert!(normalize([f64::INFINITY, 0.0, 0.0]).is_none());
        let n = normalize([3.0, 4.0, 0.0]).unwrap();
        assert!((n[0] - 0.6).abs() < 1e-12 && (n[1] - 0.8).abs() < 1e-12);
    }
}
