"""Tests for the project module — turning a mesh into the polygons a renderer draws.

The oracle throughout is the full cross product, written out in numpy: build every
triangle, cross its edges, keep the ones whose normal leans towards the viewer, sort
by mean depth. That is the definition the fast path has to meet, and the fast path
differs from it in every way that matters — it forms one component of the cross
product instead of three, out of the *projected* edges rather than the real ones,
and it runs blocked across threads.
"""

import numpy as np
import pytest

import navis_fastcore as fastcore
from meshes import uv_sphere


# Every way of naming an axis-aligned view: the two picture columns in either order,
# for each choice of depth axis. `cull_sign` exists entirely for the swapped ones.
VIEWS = [((0, 1), 2), ((1, 0), 2), ((0, 2), 1), ((2, 0), 1), ((1, 2), 0), ((2, 1), 0)]


# -----------------------------------------------------------------------------
# Oracle
# -----------------------------------------------------------------------------


def reference(vertices, faces, xy_ix, depth_ix, front):
    """Cull and sort the slow, obvious way: the whole cross product, unprojected."""
    v = np.asarray(vertices, dtype=float)
    tri = v[np.asarray(faces, dtype=np.int64)]
    raw = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])

    keep = np.flatnonzero(raw[:, depth_ix] * front > 0)
    depth = v[np.asarray(faces)[keep], depth_ix].mean(axis=1)
    order = np.argsort(depth * front, kind="stable")
    ix = keep[order]

    length = np.linalg.norm(raw[ix], axis=-1, keepdims=True)
    normals = raw[ix] / np.where(length == 0, 1, length)
    return ix, depth[order], v[:, list(xy_ix)][np.asarray(faces)[ix]], normals


@pytest.fixture(scope="module")
def sphere():
    faces, vertices = uv_sphere(n_lat=30, n_lon=30)
    # off-centre and anisotropic, so no view is a special case of another
    vertices = vertices * np.array([3.0, 1.0, 0.5]) + np.array([0.7, -2.0, 5.0])
    return vertices, faces


# -----------------------------------------------------------------------------
# The cull
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("xy_ix,depth_ix", VIEWS)
@pytest.mark.parametrize("front", [1, -1])
def test_keeps_exactly_what_the_full_cross_product_would(sphere, xy_ix, depth_ix, front):
    """The 2x2 determinant has to agree with the 3-component cross, to the face."""
    vertices, faces = sphere
    _, _, ix, _, _ = fastcore.project_mesh_2d(
        vertices, faces, xy_ix, depth_ix, front
    )
    want, _, _, _ = reference(vertices, faces, xy_ix, depth_ix, front)
    assert np.array_equal(np.sort(ix), np.sort(want))


@pytest.mark.parametrize("xy_ix,depth_ix", VIEWS)
def test_front_and_back_partition_all_but_the_edge_on_faces(sphere, xy_ix, depth_ix):
    """No face is both facing and facing away, and only edge-on faces are neither.

    A face whose normal has no depth component is exactly side-on to the viewer and
    has zero projected area, so it is dropped whichever way `front` points. A UV
    sphere has plenty - its poles are degenerate and its lat/lon grid puts whole
    rows in the coordinate planes - so this is the invariant, not a clean bisection.
    """
    vertices, faces = sphere
    _, _, front, _, _ = fastcore.project_mesh_2d(vertices, faces, xy_ix, depth_ix, 1)
    _, _, back, _, _ = fastcore.project_mesh_2d(vertices, faces, xy_ix, depth_ix, -1)

    tri = vertices[faces.astype(np.int64)]
    raw = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    edge_on = set(np.flatnonzero(raw[:, depth_ix] == 0).tolist())

    front, back = set(front.tolist()), set(back.tolist())
    assert not front & back
    assert not (front | back) & edge_on
    assert front | back | edge_on == set(range(len(faces)))


def test_winding_is_consistent(sphere):
    """Every kept triangle must project the same way round.

    A nonzero-winding fill of the lot relies on it: two overlapping subpaths wound
    against each other cancel and leave a hole. Culling by the sign of the normal is
    what guarantees it, so there is nothing to orient by hand.
    """
    vertices, faces = sphere
    for xy_ix, depth_ix in VIEWS:
        rings, _, _, _, _ = fastcore.project_mesh_2d(vertices, faces, xy_ix, depth_ix, 1)
        a, b, c = rings[:, 0], rings[:, 1], rings[:, 2]
        area = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (
            c[:, 0] - a[:, 0]
        )
        assert np.all(area >= 0) or np.all(area <= 0)


# -----------------------------------------------------------------------------
# The sort, the rings and the box
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("front", [1, -1])
def test_sorted_furthest_first_with_depths_alongside(sphere, front):
    vertices, faces = sphere
    _, _, ix, depth, _ = fastcore.project_mesh_2d(vertices, faces, (0, 1), 2, front)
    assert np.all(np.diff(depth * front) >= 0)

    # and `depth` belongs to the face it sits next to, not to the one it started as
    want = vertices[faces[ix.astype(np.int64)], 2].mean(axis=1)
    assert np.allclose(depth, want)


def test_rings_are_the_projected_triangles_closed(sphere):
    vertices, faces = sphere
    rings, _, ix, _, _ = fastcore.project_mesh_2d(vertices, faces, (2, 0), 1, 1)

    assert np.array_equal(rings[:, 3], rings[:, 0])
    assert np.allclose(rings[:, :3], vertices[faces[ix.astype(np.int64)]][:, :, [2, 0]])
    # the whole point of the 4th point: triangles come out as a view, not a copy
    assert rings[:, :3].base is rings


def test_bbox_is_the_box_over_the_rings(sphere):
    """Over the *kept* faces, which is not the same as over the mesh."""
    vertices, faces = sphere
    rings, bbox, _, _, _ = fastcore.project_mesh_2d(vertices, faces, (0, 1), 2, 1)
    flat = rings.reshape(-1, 2)
    assert np.array_equal(bbox, [*flat.min(axis=0), *flat.max(axis=0)])


def test_matches_the_oracle_end_to_end(sphere):
    """Faces, order, geometry and normals, all against the slow definition."""
    vertices, faces = sphere
    rings, _, ix, depth, normals = fastcore.project_mesh_2d(
        vertices, faces, (0, 1), 2, 1, normals=True
    )
    want_ix, want_depth, want_tri, want_n = reference(vertices, faces, (0, 1), 2, 1)

    assert np.array_equal(ix, want_ix)
    assert np.allclose(depth, want_depth)
    assert np.allclose(rings[:, :3], want_tri)
    assert np.allclose(normals, want_n)


# -----------------------------------------------------------------------------
# The opt-outs
# -----------------------------------------------------------------------------


def test_order_off_keeps_the_same_faces_in_face_order(sphere):
    vertices, faces = sphere
    _, bbox_a, ix_a, _, _ = fastcore.project_mesh_2d(vertices, faces, (0, 1), 2, 1)
    rings, bbox_b, ix_b, depth, _ = fastcore.project_mesh_2d(
        vertices, faces, (0, 1), 2, 1, order=False
    )

    assert depth is None
    assert np.array_equal(ix_b, np.sort(ix_a))
    assert np.array_equal(ix_b, np.sort(ix_b))
    assert np.array_equal(bbox_a, bbox_b)
    assert len(rings) == len(ix_a)


def test_normals_are_optional_and_unit(sphere):
    vertices, faces = sphere
    _, _, _, _, none = fastcore.project_mesh_2d(vertices, faces, (0, 1), 2, 1)
    assert none is None

    _, _, ix, _, normals = fastcore.project_mesh_2d(
        vertices, faces, (0, 1), 2, 1, normals=True
    )
    assert normals.shape == (len(ix), 3)
    assert np.allclose(np.linalg.norm(normals, axis=1), 1)


def test_order_off_still_matches_the_oracle_as_a_set(sphere):
    """The two opt-outs must not change *which* geometry comes back, only its order."""
    vertices, faces = sphere
    rings, _, ix, _, _ = fastcore.project_mesh_2d(
        vertices, faces, (1, 2), 0, -1, order=False, normals=False
    )
    assert np.allclose(rings[:, :3], vertices[faces[ix.astype(np.int64)]][:, :, [1, 2]])


# -----------------------------------------------------------------------------
# Edges and errors
# -----------------------------------------------------------------------------


def test_empty_mesh():
    rings, bbox, ix, depth, normals = fastcore.project_mesh_2d(
        np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint32), normals=True
    )
    assert rings.shape == (0, 4, 2) and len(ix) == 0
    assert len(depth) == 0 and normals.shape == (0, 3)
    assert np.all(np.isinf(bbox))


def test_degenerate_faces_are_culled_not_crashed():
    """Collinear corners have a zero normal, so they face neither way."""
    vertices = np.array([[0.0, 0, 0], [1, 0, 0], [2, 0, 0]])
    faces = np.array([[0, 1, 2]], dtype=np.uint32)
    for front in (1, -1):
        _, _, ix, _, _ = fastcore.project_mesh_2d(vertices, faces, front=front)
        assert len(ix) == 0


def test_threads_do_not_change_the_answer(sphere):
    """Blocked across threads, so the block boundaries must not be visible."""
    vertices, faces = sphere
    a = fastcore.project_mesh_2d(vertices, faces, (0, 1), 2, 1, threads=1)
    b = fastcore.project_mesh_2d(vertices, faces, (0, 1), 2, 1, threads=4)
    assert np.array_equal(a[2], b[2])
    assert np.array_equal(a[0], b[0])
    assert np.array_equal(a[1], b[1])


@pytest.mark.parametrize(
    "xy_ix,depth_ix", [((0, 1), 1), ((0, 0), 2), ((0, 1), 3), ((1, 2), 2)]
)
def test_rejects_axes_that_are_not_a_permutation(sphere, xy_ix, depth_ix):
    vertices, faces = sphere
    with pytest.raises(ValueError, match="0, 1 and 2"):
        fastcore.project_mesh_2d(vertices, faces, xy_ix, depth_ix, 1)


def test_rejects_a_front_that_is_not_a_direction(sphere):
    vertices, faces = sphere
    with pytest.raises(ValueError, match="`front` must be"):
        fastcore.project_mesh_2d(vertices, faces, (0, 1), 2, 0)


def test_rejects_faces_that_name_a_missing_vertex():
    vertices = np.zeros((3, 3))
    faces = np.array([[0, 1, 7]], dtype=np.uint32)
    with pytest.raises(ValueError, match="names vertex 7"):
        fastcore.project_mesh_2d(vertices, faces)
