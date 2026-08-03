"""Tests for the topo (skeleton healing) module.

These are self-contained and do not require navis. The end-to-end equivalence
with ``navis.heal_skeleton`` is checked separately (it needs navis installed).
"""

import navis_fastcore as fastcore
import numpy as np
import pandas as pd
import pytest

from pathlib import Path


def _load_swc(file="722817260.swc", dtype=np.int64):
    fp = Path(__file__).parent / file
    swc = pd.read_csv(fp, comment="#", header=None, sep=" ")
    node_ids = swc[0].values.astype(dtype)
    coords = swc[[2, 3, 4]].values.astype(np.float64)
    parent_ids = swc[6].values.astype(dtype)
    return node_ids, parent_ids, coords


def _load_radius(file="722817260.swc"):
    fp = Path(__file__).parent / file
    swc = pd.read_csv(fp, comment="#", header=None, sep=" ")
    return swc[5].values.astype(np.float64)


def _n_components(node_ids, parent_ids):
    return np.unique(fastcore.connected_components(node_ids, parent_ids)).size


def test_stitch_two_fragments():
    node_ids = np.arange(4)
    parent_ids = np.array([-1, 0, -1, 2])
    coords = np.array([[0, 0, 0], [1, 0, 0], [10, 0, 0], [11, 0, 0]], dtype=float)
    edges, dists = fastcore.stitch_fragments(node_ids, parent_ids, coords)
    assert edges.shape == (1, 2)
    assert set(edges[0]) == {1, 2}
    assert np.isclose(dists[0], 9.0)


def test_heal_connects_all_fragments():
    node_ids = np.arange(4)
    parent_ids = np.array([-1, 0, -1, 2])
    coords = np.array([[0, 0, 0], [1, 0, 0], [10, 0, 0], [11, 0, 0]], dtype=float)
    healed = fastcore.heal_skeleton(node_ids, parent_ids, coords)
    assert _n_components(node_ids, healed) == 1
    # Exactly one root remains.
    assert (healed < 0).sum() == 1


def test_heal_respects_max_dist():
    node_ids = np.arange(4)
    parent_ids = np.array([-1, 0, -1, 2])
    coords = np.array([[0, 0, 0], [1, 0, 0], [100, 0, 0], [101, 0, 0]], dtype=float)
    # Gap of 99 exceeds max_dist -> fragments stay separate, both keep a root.
    healed = fastcore.heal_skeleton(node_ids, parent_ids, coords, max_dist=10)
    assert _n_components(node_ids, healed) == 2
    assert (healed < 0).sum() == 2


def test_heal_min_size_excludes_small_fragment():
    # Fragment {0,1} (size 2) and singleton {2} (size 1).
    node_ids = np.arange(3)
    parent_ids = np.array([-1, 0, -1])
    coords = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float)
    # min_size=2 makes node 2 ineligible -> it cannot be attached.
    healed = fastcore.heal_skeleton(node_ids, parent_ids, coords, min_size=2)
    assert _n_components(node_ids, healed) == 2


def test_heal_is_idempotent_on_connected_skeleton():
    node_ids = np.arange(7)
    parent_ids = np.array([-1, 0, 1, 2, 1, 4, 5])
    coords = np.random.rand(7, 3)
    healed = fastcore.heal_skeleton(node_ids, parent_ids, coords)
    # Already one component: topology is unchanged.
    np.testing.assert_array_equal(healed, parent_ids)


def test_heal_real_skeleton_all_and_leafs():
    node_ids, parent_ids, coords = _load_swc()
    rng = np.random.default_rng(0)
    # Cut 100 random internal edges to fragment the neuron.
    internal = np.where(parent_ids >= 0)[0]
    cut = rng.choice(internal, size=100, replace=False)
    frag = parent_ids.copy()
    frag[cut] = -1
    assert _n_components(node_ids, frag) > 1

    for method in ("ALL", "LEAFS"):
        healed = fastcore.heal_skeleton(node_ids, frag, coords, method=method)
        assert _n_components(node_ids, healed) == 1
        assert (healed < 0).sum() == 1
        # No cycles: every node walks to the single root.
        id2ix = {int(n): i for i, n in enumerate(node_ids)}
        healed_ix = np.array(
            [id2ix[int(p)] if p >= 0 else -1 for p in healed], dtype=np.int64
        )
        assert not fastcore.dag._fastcore.has_cycles(healed_ix.astype(np.int32))


def _fragment(parent_ids, n_cuts=100, seed=0):
    rng = np.random.default_rng(seed)
    internal = np.where(parent_ids >= 0)[0]
    frag = parent_ids.copy()
    frag[rng.choice(internal, size=n_cuts, replace=False)] = -1
    return frag


@pytest.mark.parametrize("method", ["ALL", "LEAFS"])
def test_heal_is_deterministic_across_runs(method):
    """Consecutive runs on the same neuron must heal it exactly the same way.

    The bridge search is parallel, and the per-fragment bound it prunes with is
    shared across threads. Nodes that tie for their fragment's shortest bridge --
    routine, since coordinates come off a lattice -- used to have that race decide
    which of the equally short bridges was kept, so the healed skeleton differed
    from run to run while always having the same total added cable.
    """
    node_ids, parent_ids, coords = _load_swc()
    frag = _fragment(parent_ids, n_cuts=400)

    first = fastcore.heal_skeleton(node_ids, frag, coords, method=method)
    for run in range(1, 25):
        np.testing.assert_array_equal(
            fastcore.heal_skeleton(node_ids, frag, coords, method=method),
            first,
            err_msg=f"run {run} healed the same neuron differently",
        )


def test_use_radius_off_by_default():
    """`use_radius` falsy must be identical to not passing radius at all."""
    node_ids, parent_ids, coords = _load_swc()
    radius = _load_radius()
    frag = _fragment(parent_ids)

    base = fastcore.heal_skeleton(node_ids, frag, coords)
    for off in (False, 0, 0.0, None):
        got = fastcore.heal_skeleton(
            node_ids, frag, coords, radius=radius, use_radius=off
        )
        assert np.array_equal(base, got)


def test_use_radius_changes_healing_and_still_connects():
    node_ids, parent_ids, coords = _load_swc()
    radius = _load_radius()
    frag = _fragment(parent_ids)

    base = fastcore.heal_skeleton(node_ids, frag, coords)
    for weight in (True, 5.0):
        healed = fastcore.heal_skeleton(
            node_ids, frag, coords, radius=radius, use_radius=weight
        )
        # Still a single, valid tree ...
        assert _n_components(node_ids, healed) == 1
        assert (healed < 0).sum() == 1
    # ... but a heavily weighted radius must actually steer the result.
    heavy = fastcore.heal_skeleton(
        node_ids, frag, coords, radius=radius, use_radius=20.0
    )
    assert not np.array_equal(base, heavy)


def test_use_radius_requires_radius():
    node_ids, parent_ids, coords = _load_swc()
    frag = _fragment(parent_ids)
    with pytest.raises(ValueError):
        fastcore.heal_skeleton(node_ids, frag, coords, use_radius=True)


def test_constant_radius_is_a_no_op():
    """A constant radius shifts every node equally in the 4th dimension, so it
    cannot change any pairwise distance -- the healing must be unchanged."""
    node_ids, parent_ids, coords = _load_swc()
    frag = _fragment(parent_ids)

    base = fastcore.heal_skeleton(node_ids, frag, coords)
    const = fastcore.heal_skeleton(
        node_ids,
        frag,
        coords,
        radius=np.full(len(node_ids), 7.0),
        use_radius=3.0,
    )
    assert np.array_equal(base, const)


def test_stitch_fragments_accepts_use_radius():
    node_ids, parent_ids, coords = _load_swc()
    radius = _load_radius()
    frag = _fragment(parent_ids)

    n_frags = _n_components(node_ids, frag)
    edges, dists = fastcore.stitch_fragments(
        node_ids, frag, coords, radius=radius, use_radius=2.0
    )
    assert len(edges) == n_frags - 1
    # Distances are measured in the augmented space, so they are at least as
    # large as the plain 3D distance between the same node pairs.
    id2ix = {int(n): i for i, n in enumerate(node_ids)}
    for (a, b), d in zip(edges, dists):
        ia, ib = id2ix[int(a)], id2ix[int(b)]
        d3 = np.linalg.norm(coords[ia] - coords[ib])
        assert d + 1e-4 >= d3


# --- reroot_rewire ----------------------------------------------------------
#
# Step 2 of healing on its own: an edited edge set, oriented back into a forest.


def test_reroot_rewire_reproduces_healing():
    """Fed the bridges `heal_skeleton` picks, it must produce what healing produced."""
    node_ids, parent_ids, coords = _load_swc()
    frag = _fragment(parent_ids)

    edges, _ = fastcore.stitch_fragments(node_ids, frag, coords)
    # Healing prefers the existing first root, and so must we to compare.
    root = node_ids[frag < 0][0]

    np.testing.assert_array_equal(
        fastcore.reroot_rewire(node_ids, frag, edges, root=root),
        fastcore.heal_skeleton(node_ids, frag, coords),
    )


def test_reroot_rewire_with_no_edges_is_a_re_orientation():
    """No new edges: same undirected edges, same components, still a forest."""
    node_ids, parent_ids, _ = _load_swc()

    out = fastcore.reroot_rewire(node_ids, parent_ids, [])

    def edges(parents):
        return sorted(
            tuple(sorted((int(n), int(p))))
            for n, p in zip(node_ids, parents)
            if p >= 0
        )

    assert edges(out) == edges(parent_ids)
    assert _n_components(node_ids, out) == _n_components(node_ids, parent_ids)
    assert not fastcore.has_cycles(node_ids, out)


def test_reroot_rewire_roots_where_asked():
    node_ids = np.array([10, 20, 30, 40])
    parent_ids = np.array([-1, 10, -1, 30])

    out = fastcore.reroot_rewire(node_ids, parent_ids, [(20, 30)], root=40)

    assert out.tolist() == [20, 30, 40, -1]
    assert not fastcore.has_cycles(node_ids, out)


def test_reroot_rewire_auto_picks_a_root_per_component():
    """Unreached components are not left orphaned - each keeps a root of its own."""
    node_ids = np.array([10, 20, 30, 40])
    parent_ids = np.array([-1, 10, -1, 30])

    out = fastcore.reroot_rewire(node_ids, parent_ids, [])

    assert (out < 0).sum() == 2
    assert not fastcore.has_cycles(node_ids, out)


def test_reroot_rewire_ignores_an_edge_that_would_close_a_cycle():
    """A redundant edge is a back-edge to the orienting walk, not a cycle."""
    node_ids = np.array([1, 2, 3])
    parent_ids = np.array([-1, 1, 2])

    out = fastcore.reroot_rewire(node_ids, parent_ids, [(3, 1)])

    assert not fastcore.has_cycles(node_ids, out)
    assert (out < 0).sum() == 1


def test_reroot_rewire_rejects_unknown_ids():
    node_ids = np.array([1, 2, 3])
    parent_ids = np.array([-1, 1, 2])

    with pytest.raises(ValueError, match="not found"):
        fastcore.reroot_rewire(node_ids, parent_ids, [(1, 999)])
    with pytest.raises(ValueError, match="not found"):
        fastcore.reroot_rewire(node_ids, parent_ids, [], root=999)


def test_reroot_rewire_rejects_a_bad_edge_shape():
    node_ids = np.array([1, 2, 3])
    parent_ids = np.array([-1, 1, 2])

    with pytest.raises(ValueError, match=r"\(M, 2\)"):
        fastcore.reroot_rewire(node_ids, parent_ids, [1, 2, 3])
