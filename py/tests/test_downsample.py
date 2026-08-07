"""Downsampling, resampling and smoothing.

The unifying invariant, and the reason all six functions live in one module: none of
them may change the *topology*. Whatever they do to the sampling along a neurite, the
roots, branch points and leafs come out the other side, and the tree they form is the
tree that went in. Most of what follows is that claim, asked in different ways.
"""

import numpy as np
import pytest

import navis_fastcore as fastcore
import topologies

#: The three methods that drop nodes, as `(name, callable(topo, coords) -> triple)`.
#: They share an output contract, so they share most of their tests.
DROPPERS = {
    "downsample": lambda topo, xyz, w: fastcore.downsample_skeleton(
        topo.node_ids, topo.parent_ids, 3, weights=w
    ),
    "rdp": lambda topo, xyz, w: fastcore.simplify_rdp(
        topo.node_ids, topo.parent_ids, xyz, 10.0, weights=w
    ),
    "vw": lambda topo, xyz, w: fastcore.simplify_vw(
        topo.node_ids, topo.parent_ids, xyz, 100.0, weights=w
    ),
}


# ------------------------------------------------------------- shared to all droppers


@pytest.mark.parametrize("method", list(DROPPERS))
def test_droppers_honour_the_shared_contract(topo, method):
    """Topology preserved node-for-node, still a forest, cable length intact.

    The checker is `topologies.check_dropping_invariants`, shared with the property
    suite - the three methods differ only in which nodes they drop, so the contract
    they owe is one contract, and it is stated once.
    """
    ids, parents, weights, node_map = DROPPERS[method](topo, topo.coords, topo.weights)

    topologies.check_dropping_invariants(
        (topo.node_ids, topo.parent_ids),
        (ids, parents),
        weights=topo.weights,
        new_weights=weights,
        node_map=node_map,
    )


@pytest.mark.parametrize("method", list(DROPPERS))
def test_droppers_return_no_weights_when_given_none(topo, method):
    _, _, weights, _ = DROPPERS[method](topo, topo.coords, None)
    assert weights is None


@pytest.mark.parametrize("method", list(DROPPERS))
def test_droppers_map_dropped_nodes_to_the_nearest_survivor(real_topo, method):
    """`node_map` names the nearest survivor, not merely the next one rootwards.

    The proximal end is where the rewiring walk arrives anyway; taking the *nearer* of
    a chain's two ends is the extra thing this promises, and on a real skeleton the two
    rules disagree for most dropped nodes. The check is against every survivor rather
    than against the two obvious candidates, since nothing else may be closer either.
    """
    topo = real_topo
    ids, _, _, node_map = DROPPERS[method](topo, topo.coords, topo.weights)

    was_dropped = ~np.isin(topo.node_ids, ids)
    assert was_dropped.sum() > 100, "the fixture has to lose nodes for this to mean much"
    dropped = topo.node_ids[was_dropped]

    # Geodesic distance from each dropped node to every survivor. Nothing may be closer
    # than the survivor it was actually handed to.
    dists = fastcore.geodesic_matrix(
        topo.node_ids,
        topo.parent_ids,
        weights=topo.weights,
        sources=dropped,
        targets=ids,
    )
    slot = {node: i for i, node in enumerate(ids.tolist())}
    chosen = np.array([slot[m] for m in node_map[was_dropped].tolist()])
    got = dists[np.arange(len(dropped)), chosen]
    assert (got <= dists.min(axis=1) + 1e-4).all()


# ------------------------------------------------------------------ downsample_skeleton


def test_downsample_factor_one_is_a_no_op(topo):
    ids, parents, _, _ = fastcore.downsample_skeleton(topo.node_ids, topo.parent_ids, 1)
    np.testing.assert_array_equal(ids, topo.node_ids)
    np.testing.assert_array_equal(parents, topo.parent_ids)


def test_downsample_huge_factor_matches_simplify_skeleton(topo):
    """With a factor no node can satisfy, only the topology nodes are left - which is
    exactly what `simplify_skeleton` returns."""
    got = fastcore.downsample_skeleton(
        topo.node_ids, topo.parent_ids, 10**6, weights=topo.weights
    )
    want = fastcore.simplify_skeleton(topo.node_ids, topo.parent_ids, weights=topo.weights)

    np.testing.assert_array_equal(got[0], want[0])
    np.testing.assert_array_equal(got[1], want[1])
    np.testing.assert_allclose(got[2], want[2])


def test_downsample_preserve_keeps_the_named_nodes(topo):
    # Every fifth node, whether the factor would have kept it or not.
    preserve = np.asarray(topo.node_ids)[::5]
    ids, _, _, _ = fastcore.downsample_skeleton(
        topo.node_ids, topo.parent_ids, 10**6, preserve=preserve
    )
    assert set(preserve.tolist()) <= set(np.asarray(ids).tolist())


def test_downsample_is_monotone_in_factor(topo):
    """A coarser factor can only ever drop more, never bring a node back."""
    coarse = set(
        np.asarray(
            fastcore.downsample_skeleton(topo.node_ids, topo.parent_ids, 4)[0]
        ).tolist()
    )
    fine = set(
        np.asarray(
            fastcore.downsample_skeleton(topo.node_ids, topo.parent_ids, 2)[0]
        ).tolist()
    )
    assert coarse <= fine


def test_downsample_rejects_a_bad_factor():
    ids, parents = np.arange(3), np.array([-1, 0, 1])
    with pytest.raises(ValueError, match="factor"):
        fastcore.downsample_skeleton(ids, parents, 0)


# --------------------------------------------------------------------------------- RDP


def dropped_node_deviations(topo, xyz, kept):
    """How far each dropped node sits from the chord that replaced it.

    RDP's contract, stated directly: a node it drops must lie within `epsilon` of the
    straight line between the two survivors that end up bracketing it in its segment.
    """
    row = {int(n): i for i, n in enumerate(np.asarray(topo.node_ids).tolist())}
    out = []

    for seg in fastcore.break_segments(topo.node_ids, topo.parent_ids):
        seg = [int(n) for n in np.asarray(seg).tolist()]
        # The survivors' positions within the segment. Both ends must be among them -
        # they are a leaf, a branch point or a root.
        anchors = [i for i, node in enumerate(seg) if node in kept]
        assert anchors[0] == 0 and anchors[-1] == len(seg) - 1

        for lo, hi in zip(anchors, anchors[1:]):
            a, b = xyz[row[seg[lo]]], xyz[row[seg[hi]]]
            d = b - a
            denom = float(d @ d)
            for i in range(lo + 1, hi):
                p = xyz[row[seg[i]]]
                t = 0.0 if denom == 0 else float(np.clip(((p - a) @ d) / denom, 0, 1))
                out.append(float(np.linalg.norm(p - (a + t * d))))

    return np.array(out)


def test_rdp_zero_epsilon_drops_only_collinear_nodes(topo):
    """`epsilon = 0` is not quite a no-op: a node that lies *exactly* on the line
    between its neighbours still goes, because dropping it moves nothing. Real
    skeletons have plenty of those - they come off a lattice - so this is the
    tolerance test at its limit rather than an identity check.
    """
    xyz = topo.coords
    ids, _, _, _ = fastcore.simplify_rdp(topo.node_ids, topo.parent_ids, xyz, 0.0)

    deviations = dropped_node_deviations(topo, xyz, set(np.asarray(ids).tolist()))
    if len(deviations):
        assert deviations.max() < 1e-9


def test_rdp_huge_epsilon_matches_simplify_skeleton(topo):
    got = fastcore.simplify_rdp(
        topo.node_ids, topo.parent_ids, topo.coords, 1e9
    )
    want = fastcore.simplify_skeleton(topo.node_ids, topo.parent_ids)
    np.testing.assert_array_equal(got[0], want[0])
    np.testing.assert_array_equal(got[1], want[1])


def test_rdp_is_monotone_in_epsilon(topo):
    xyz = topo.coords
    coarse = set(
        np.asarray(
            fastcore.simplify_rdp(topo.node_ids, topo.parent_ids, xyz, 50.0)[0]
        ).tolist()
    )
    fine = set(
        np.asarray(
            fastcore.simplify_rdp(topo.node_ids, topo.parent_ids, xyz, 5.0)[0]
        ).tolist()
    )
    assert coarse <= fine


def test_rdp_respects_its_tolerance(real_topo):
    """Every dropped node must lie within `epsilon` of the path that replaced it.

    Checked on a real skeleton, where the claim has something to bite on: a straight
    run collapses and a curve does not.
    """
    topo = real_topo
    xyz = topo.coords
    epsilon = 50.0
    ids, _, _, _ = fastcore.simplify_rdp(topo.node_ids, topo.parent_ids, xyz, epsilon)

    deviations = dropped_node_deviations(topo, xyz, set(np.asarray(ids).tolist()))
    # The test is only worth anything if nodes were actually dropped.
    assert len(deviations) > 100
    assert deviations.max() <= epsilon + 1e-6


# ---------------------------------------------------------------------------------- VW


def test_vw_zero_area_is_a_no_op(topo):
    ids, parents, _, _ = fastcore.simplify_vw(
        topo.node_ids, topo.parent_ids, topo.coords, 0.0
    )
    np.testing.assert_array_equal(ids, topo.node_ids)
    np.testing.assert_array_equal(parents, topo.parent_ids)


def test_vw_huge_area_matches_simplify_skeleton(topo):
    got = fastcore.simplify_vw(
        topo.node_ids, topo.parent_ids, topo.coords, 1e12
    )
    want = fastcore.simplify_skeleton(topo.node_ids, topo.parent_ids)
    np.testing.assert_array_equal(got[0], want[0])
    np.testing.assert_array_equal(got[1], want[1])


def test_vw_is_reproducible(topo):
    """Ties are routine - coordinates come off a lattice - and must not be settled by
    whichever thread got there first."""
    xyz = topo.coords
    runs = [
        fastcore.simplify_vw(topo.node_ids, topo.parent_ids, xyz, 200.0)[0]
        for _ in range(3)
    ]
    for other in runs[1:]:
        np.testing.assert_array_equal(runs[0], other)


# -------------------------------------------------------------------------- resampling


def test_resample_keeps_the_topology_nodes_first_and_unmoved(topo):
    xyz = topo.coords
    ids, parents, out_xyz, source, alpha, _ = fastcore.resample_skeleton(
        topo.node_ids, topo.parent_ids, xyz, 10.0
    )

    want = fastcore.simplify_skeleton(topo.node_ids, topo.parent_ids)[0]
    np.testing.assert_array_equal(ids[: len(want)], want)
    np.testing.assert_allclose(out_xyz[: len(want)], xyz[source[: len(want), 0]])
    np.testing.assert_array_equal(alpha[: len(want)], 0.0)


def test_resample_is_a_forest_with_the_same_shape(topo):
    ids, parents, _, _, _, node_map = fastcore.resample_skeleton(
        topo.node_ids, topo.parent_ids, topo.coords, 10.0
    )
    topologies.check_is_forest(ids, parents)
    topologies.check_node_map(topo.node_ids, ids, node_map)
    # Resampling mints new IDs, so only the counts per class can be compared.
    topologies.check_topology_preserved(
        (topo.node_ids, topo.parent_ids), (ids, parents), same_nodes=False
    )


def test_resample_source_and_alpha_reproduce_the_coordinates(topo):
    """The documented interpolation must give back the coordinates the function chose -
    otherwise a caller interpolating a radius the same way would get something else."""
    xyz = topo.coords
    _, _, out_xyz, source, alpha, _ = fastcore.resample_skeleton(
        topo.node_ids, topo.parent_ids, xyz, 10.0
    )
    want = xyz[source[:, 0]] * (1 - alpha)[:, None] + xyz[source[:, 1]] * alpha[:, None]
    np.testing.assert_allclose(out_xyz, want, atol=1e-9)


def test_resample_hits_the_requested_spacing(real_topo):
    """No edge may come out longer than `spacing` by more than the even-division rule
    allows, on a real skeleton.

    Nothing is asserted about the *median* edge: most of this arbor's segments are
    twigs shorter than `spacing`, and those collapse to one short edge each. Short
    edges are the expected outcome there, not a miss.
    """
    topo = real_topo
    xyz = topo.coords
    spacing = 500.0

    ids, parents, out_xyz, _, _, _ = fastcore.resample_skeleton(
        topo.node_ids, topo.parent_ids, xyz, spacing
    )
    lengths = fastcore.parent_dist(ids, parents, out_xyz, root_dist=0)
    lengths = lengths[np.asarray(parents) >= 0]

    # A segment of length L becomes round(L / spacing) equal edges, so the longest an
    # edge can be is L / (L/spacing - 0.5), maximised at 1.5x spacing when L/spacing
    # lands just under 1.5. Segments too short to divide give edges shorter still.
    assert lengths.max() <= spacing * 1.5 + 1e-6

    # ...and interpolated nodes were in fact added, rather than every segment coming
    # out too short to subdivide (which would satisfy the bound above vacuously).
    endpoints = fastcore.simplify_skeleton(topo.node_ids, topo.parent_ids)[0]
    assert len(ids) > len(endpoints)


def test_resample_finer_spacing_gives_more_nodes(real_topo):
    topo = real_topo
    xyz = topo.coords
    sizes = [
        len(fastcore.resample_skeleton(topo.node_ids, topo.parent_ids, xyz, s)[0])
        for s in (2000.0, 500.0, 100.0)
    ]
    assert sizes[0] < sizes[1] < sizes[2]


@pytest.mark.parametrize("spacing", [100.0, 500.0, 2000.0])
def test_resample_node_map_lands_within_three_quarters_of_a_spacing(real_topo, spacing):
    """An input node's data goes somewhere near enough that it still means something.

    Output nodes divide each segment into equal parts of at most ``1.5 * spacing`` -- a
    segment shorter than that becomes a single part -- so the nearest one is never more
    than ``0.75 * spacing`` along the neurite, and the straight line is shorter still.
    """
    topo = real_topo
    ids, _, xyz, _, _, node_map = fastcore.resample_skeleton(
        topo.node_ids, topo.parent_ids, topo.coords, spacing
    )
    topologies.check_node_map(topo.node_ids, ids, node_map)

    slot = {node: i for i, node in enumerate(ids.tolist())}
    landed = np.array([slot[m] for m in node_map.tolist()])
    offset = np.linalg.norm(topo.coords - xyz[landed], axis=1)
    assert offset.max() <= 0.75 * spacing + 1e-6


def test_resample_rejects_a_bad_spacing():
    ids, parents = np.arange(3), np.array([-1, 0, 1])
    xyz = np.zeros((3, 3))
    with pytest.raises(ValueError, match="spacing"):
        fastcore.resample_skeleton(ids, parents, xyz, 0)


# --------------------------------------------------------------------------- smoothing


SMOOTHERS = {
    "moving_average": lambda ids, parents, xyz: fastcore.smooth_skeleton(
        ids, parents, xyz, window=5
    ),
    "gaussian": lambda ids, parents, xyz: fastcore.smooth_skeleton_gaussian(
        ids, parents, xyz, sigma=20.0
    ),
}


@pytest.mark.parametrize("method", list(SMOOTHERS))
def test_smoothing_pins_the_topology_nodes(topo, method):
    xyz = topo.coords
    out = SMOOTHERS[method](topo.node_ids, topo.parent_ids, xyz)

    assert out.shape == xyz.shape
    pinned = fastcore.simplify_skeleton(topo.node_ids, topo.parent_ids)[0]
    index = {int(n): i for i, n in enumerate(np.asarray(topo.node_ids).tolist())}
    rows = [index[int(n)] for n in np.asarray(pinned).tolist()]
    np.testing.assert_array_equal(out[rows], xyz[rows])


@pytest.mark.parametrize("method", list(SMOOTHERS))
def test_smoothing_stays_inside_the_bounding_box(topo, method):
    """Smoothing is a weighted mean of nearby nodes, so nothing may leave the cloud's
    box - except that the endpoint reflection can push a node one span past it, which
    is why the tolerance is the box's own size rather than zero."""
    xyz = topo.coords
    out = SMOOTHERS[method](topo.node_ids, topo.parent_ids, xyz)
    span = xyz.max(axis=0) - xyz.min(axis=0)
    assert (out >= xyz.min(axis=0) - span - 1e-9).all()
    assert (out <= xyz.max(axis=0) + span + 1e-9).all()


def test_smoothing_window_one_is_a_no_op(topo):
    xyz = topo.coords
    out = fastcore.smooth_skeleton(topo.node_ids, topo.parent_ids, xyz, window=1)
    np.testing.assert_array_equal(out, xyz)


def test_smoothing_shortens_the_cable(real_topo):
    """The point of smoothing: a jittery traced neurite gets shorter, because the
    jitter was length that was not really there."""
    topo = real_topo
    xyz = topo.coords

    before = fastcore.parent_dist(topo.node_ids, topo.parent_ids, xyz, root_dist=0).sum()
    for out in (
        fastcore.smooth_skeleton(topo.node_ids, topo.parent_ids, xyz, window=5),
        fastcore.smooth_skeleton_gaussian(topo.node_ids, topo.parent_ids, xyz, 500.0),
    ):
        after = fastcore.parent_dist(topo.node_ids, topo.parent_ids, out, root_dist=0).sum()
        assert after < before


def test_gaussian_is_monotone_in_sigma(real_topo):
    """More smoothing, less cable."""
    topo = real_topo
    xyz = topo.coords

    lengths = []
    for sigma in (100.0, 500.0, 2000.0):
        out = fastcore.smooth_skeleton_gaussian(
            topo.node_ids, topo.parent_ids, xyz, sigma
        )
        lengths.append(
            fastcore.parent_dist(topo.node_ids, topo.parent_ids, out, root_dist=0).sum()
        )
    assert lengths[0] > lengths[1] > lengths[2]


def test_smoothing_rejects_bad_parameters():
    ids, parents = np.arange(3), np.array([-1, 0, 1])
    xyz = np.zeros((3, 3))
    with pytest.raises(ValueError, match="sigma"):
        fastcore.smooth_skeleton_gaussian(ids, parents, xyz, 0.0)
    with pytest.raises(ValueError, match="truncate"):
        fastcore.smooth_skeleton_gaussian(ids, parents, xyz, 1.0, truncate=-1)


# ------------------------------------------------------------------------- input checks


@pytest.mark.parametrize(
    "call",
    [
        lambda ids, parents, xyz: fastcore.simplify_rdp(ids, parents, xyz, 1.0),
        lambda ids, parents, xyz: fastcore.simplify_vw(ids, parents, xyz, 1.0),
        lambda ids, parents, xyz: fastcore.resample_skeleton(ids, parents, xyz, 1.0),
        lambda ids, parents, xyz: fastcore.smooth_skeleton(ids, parents, xyz),
        lambda ids, parents, xyz: fastcore.smooth_skeleton_gaussian(
            ids, parents, xyz, 1.0
        ),
    ],
)
def test_coords_are_validated(call):
    ids, parents = np.arange(3), np.array([-1, 0, 1])
    with pytest.raises(ValueError, match="coords"):
        call(ids, parents, np.zeros((3, 2)))  # not 3D
    with pytest.raises(ValueError, match="coords"):
        call(ids, parents, np.zeros((2, 3)))  # wrong length


def test_threads_do_not_change_the_answer(real_topo):
    """Every geometric method is parallel over segments; a different worker count must
    give a bit-identical result."""
    topo = real_topo
    xyz = topo.coords
    ids, parents = topo.node_ids, topo.parent_ids

    for one, many in [
        (
            fastcore.simplify_rdp(ids, parents, xyz, 50.0, threads=1)[0],
            fastcore.simplify_rdp(ids, parents, xyz, 50.0, threads=4)[0],
        ),
        (
            fastcore.simplify_vw(ids, parents, xyz, 200.0, threads=1)[0],
            fastcore.simplify_vw(ids, parents, xyz, 200.0, threads=4)[0],
        ),
        (
            fastcore.resample_skeleton(ids, parents, xyz, 500.0, threads=1)[2],
            fastcore.resample_skeleton(ids, parents, xyz, 500.0, threads=4)[2],
        ),
        (
            fastcore.smooth_skeleton(ids, parents, xyz, 5, threads=1),
            fastcore.smooth_skeleton(ids, parents, xyz, 5, threads=4),
        ),
        (
            fastcore.smooth_skeleton_gaussian(ids, parents, xyz, 500.0, threads=1),
            fastcore.smooth_skeleton_gaussian(ids, parents, xyz, 500.0, threads=4),
        ),
    ]:
        np.testing.assert_array_equal(one, many)
