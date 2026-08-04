"""Mesh fixtures and invariants shared by the mesh, simplification and property suites.

The same role `topologies.py` plays for skeletons. Fixtures live here rather than in
`conftest.py` because they are plain functions the tests call with their own
arguments, not pytest fixtures, and because the property suite needs them at
import time to build strategies.
"""

import numpy as np


def grid_mesh(n=12, spacing=1.0):
    """An `n x n` grid triangulated along the (0,0)->(1,1) diagonal of each cell.

    Has a closed-form metric (see `test_mesh.test_matches_grid_closed_form`), so it
    doubles as an oracle that does not depend on scipy. Every interior vertex is
    exactly coplanar with its ring, which also makes it the natural fixture for
    lossless simplification.
    """
    idx = lambda i, j: i * n + j  # noqa: E731

    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            faces.append([idx(i, j), idx(i + 1, j), idx(i + 1, j + 1)])
            faces.append([idx(i, j), idx(i + 1, j + 1), idx(i, j + 1)])

    verts = np.array(
        [[i * spacing, j * spacing, 0.0] for i in range(n) for j in range(n)],
        dtype=np.float64,
    )
    return np.array(faces, dtype=np.uint32), verts


def uv_sphere(n_lat=20, n_lon=20):
    """A closed UV sphere: valence ~6, no boundary, edge lengths in a narrow band.

    The shape a decimated connectomics mesh has, and the same generator the Rust
    suite and `fastcore/examples/profile_mesh.rs` use.
    """
    verts = []
    for i in range(n_lat):
        # Avoid the exact poles so no ring degenerates to a point.
        theta = np.pi * (i + 0.5) / n_lat
        for j in range(n_lon):
            phi = 2 * np.pi * j / n_lon
            verts.append(
                [
                    np.sin(theta) * np.cos(phi),
                    np.sin(theta) * np.sin(phi),
                    np.cos(theta),
                ]
            )

    idx = lambda i, j: i * n_lon + j % n_lon  # noqa: E731
    faces = []
    for i in range(n_lat - 1):
        for j in range(n_lon):
            faces.append([idx(i, j), idx(i + 1, j), idx(i + 1, j + 1)])
            faces.append([idx(i, j), idx(i + 1, j + 1), idx(i, j + 1)])

    return np.array(faces, dtype=np.uint32), np.array(verts, dtype=np.float64)


def referenced(faces):
    """Vertices a *real* face names.

    Faces that name a vertex twice have no area, so they carry no plane and no
    normal and are dropped on the way in. A vertex those are its only mention of is
    therefore no better off than one in no face at all: there is no surviving face
    for it to live in, so it cannot come back.
    """
    if not len(faces):
        return set()
    faces = np.asarray(faces)
    real = faces[
        (faces[:, 0] != faces[:, 1])
        & (faces[:, 1] != faces[:, 2])
        & (faces[:, 2] != faces[:, 0])
    ]
    return set(real.ravel().tolist())


def check_simplify_invariants(out, faces, vertices):
    """Everything that must hold of a simplification result, whatever went in.

    One copy, called from both the example-based and the property suite — an
    invariant added to only one of two near-identical checkers is exactly the
    failure a shared one prevents.
    """
    v, f, vmap = out
    n_in = len(vertices)

    assert v.dtype == np.float64 and f.dtype == np.uint32 and vmap.dtype == np.int32
    assert v.ndim == 2 and v.shape[1] == 3
    assert f.ndim == 2 and f.shape[1] == 3

    assert len(vmap) == n_in, "one map entry per input vertex"
    assert np.isfinite(v).all(), "a non-finite coordinate escaped"
    assert len(f) <= len(faces), "simplifying cannot add faces"

    if n_in:
        assert vmap.min() >= -1
        assert vmap.max() < len(v)
    if len(f):
        assert f.max() < len(v), "a face references a vertex that is not there"

    # Onto: no output vertex without a preimage, i.e. the renumbering and the
    # collapse forest agree about who survived.
    assert set(vmap[vmap >= 0].tolist()) == set(range(len(v)))

    # A vertex no surviving face could reference cannot survive.
    used = referenced(faces)
    for i in range(n_in):
        if i not in used:
            assert vmap[i] == -1
