import pytest
import time

import navis_fastcore as fastcore
import numpy as np
import pandas as pd

from numpy.testing import assert_array_equal
from scipy.spatial import cKDTree as KDTree
from pathlib import Path
from collections import namedtuple

Dotprop = namedtuple("Dotprop", ["points", "vect"])

# Set random state
np.random.seed(0)


def _load_swc(file="722817260.swc", dtype=np.int64, synapses=False):
    fp = Path(__file__).parent / file
    swc = pd.read_csv(fp, comment="#", header=None, sep=" ")
    nodes = swc[0].values.astype(dtype)
    coords = swc[[2, 3, 4]].values.astype(
        np.float64
    )  # the data type for this does not matter
    parents = swc[6].values.astype(dtype)

    if synapses:
        connectors = pd.read_csv(Path(__file__).parent / file.replace(".swc", ".csv"))
        presynapses = (
            connectors[connectors.type == "pre"]
            .node_id.value_counts()
            .reindex(nodes)
            .fillna(0)
            .values.astype(np.uint32, order="C", copy=False)
        )
        postsynapses = (
            connectors[connectors.type == "post"]
            .node_id.value_counts()
            .reindex(nodes)
            .fillna(0)
            .values.astype(np.uint32, order="C", copy=False)
        )
        return nodes, parents, coords, presynapses, postsynapses

    return nodes, parents, coords


# Number of nodes in the default test neuron
N_NODES = len(_load_swc()[0])


def swc64():
    return _load_swc(dtype=np.int32, synapses=False)


def swc32():
    return _load_swc(dtype=np.int64, synapses=False)


@pytest.fixture
def coords():
    return _load_swc()[2]


@pytest.fixture
def swc_with_synapses():
    return _load_swc(synapses=True)


def _node_indices_py(A, B):
    """Reference Python implementation of _node_indices."""
    ix_dict = dict(zip(B, np.arange(len(B))))

    return np.array([ix_dict.get(p, -1) for p in A])


@pytest.mark.parametrize("swc", [swc32(), swc64()])
def test_node_indices(swc):
    nodes, parents, _ = swc
    start = time.time()
    indices = fastcore.dag._ids_to_indices(nodes, parents)
    dur = time.time() - start

    start = time.time()
    indices_py = _node_indices_py(parents, nodes)
    dur_py = time.time() - start

    assert all(indices == indices_py)

    print("Indices:", indices)
    print(f"Timing: {dur:.4f}s (compared to {dur_py:.4f}s in pure numpy Python)")


@pytest.mark.parametrize("swc", [swc32(), swc64()])
@pytest.mark.parametrize("weights", [None, np.random.rand(N_NODES)])
def test_generate_segments(swc, weights):
    nodes, parents, _ = swc
    start = time.time()
    segments, lengths = fastcore.generate_segments(nodes, parents, weights=weights)
    dur = time.time() - start

    assert len(segments) == len(lengths)

    # `lengths[i]` must describe `segments[i]`. The weighted branch used to sort the
    # segments longest-first but hand back the lengths in their original order, so the
    # two did not line up -- and nothing here caught it.
    if weights is None:
        expected = [len(s) for s in segments]
    else:
        w = dict(zip(nodes, weights))
        expected = [sum(w[n] for n in s) for s in segments]
    np.testing.assert_allclose(lengths, expected, rtol=1e-5)

    # Segments come out longest-first.
    assert list(lengths) == sorted(lengths, reverse=True)

    # Every node is covered and none is invented. N.B. segments deliberately OVERLAP at
    # branch points (a walk includes the already-seen node it stops at), so this is set
    # coverage, not a partition.
    assert set(np.concatenate(segments)) == set(nodes)

    print(f"Timing: {dur:.4f}s")


def test_geodesic_pairs_disconnected_is_negative():
    """Nodes in different components have no path between them.

    That is reported as -1, matching `geodesic_matrix`. It used to be reported as 1.0,
    which is indistinguishable from a genuine one-edge distance.
    """
    # Two separate components: {1, 2} and {3, 4}
    nodes = np.array([1, 2, 3, 4], dtype=np.int64)
    parents = np.array([-1, 1, -1, 3], dtype=np.int64)

    dists = fastcore.geodesic_pairs(nodes, parents, np.array([[1, 3], [1, 2], [3, 4]]))

    assert dists[0] == -1  # crosses components
    assert dists[1] == 1  # a real one-edge distance
    assert dists[2] == 1

    # ... and it must agree with what the full matrix says about the same pair.
    matrix = fastcore.geodesic_matrix(nodes, parents)
    assert matrix[0, 2] == dists[0]


def test_dist_to_root_counts_edges():
    """`dist_to_root` counts edges, so a root is at distance 0, matching `geodesic_matrix`."""
    from navis_fastcore import _fastcore

    # chain 0 <- 1 <- 2
    parents = np.array([-1, 0, 1], dtype=np.int32)

    assert _fastcore.dist_to_root(parents, 0) == 0.0
    assert _fastcore.dist_to_root(parents, 1) == 1.0
    assert _fastcore.dist_to_root(parents, 2) == 2.0

    # Must agree with all_dists_to_root, which it used to be off-by-one against.
    assert_array_equal(
        [_fastcore.dist_to_root(parents, n) for n in range(3)],
        _fastcore.all_dists_to_root(parents, None, None),
    )


def test_has_cycles():
    """Cycle detection: no false positives on forests, no false negatives on cycles."""
    from navis_fastcore import _fastcore

    # Acyclic: chain, forest of two trees, branching tree
    assert not _fastcore.has_cycles(np.array([-1, 0, 1], dtype=np.int32))
    assert not _fastcore.has_cycles(np.array([-1, 0, 1, -1, 3], dtype=np.int32))
    assert not _fastcore.has_cycles(np.array([-1, 0, 0, 1, 1, 2], dtype=np.int32))

    # Cyclic: self-loop, 2-cycle, 3-cycle with no root
    assert _fastcore.has_cycles(np.array([0], dtype=np.int32))
    assert _fastcore.has_cycles(np.array([1, 0], dtype=np.int32))
    assert _fastcore.has_cycles(np.array([1, 2, 0], dtype=np.int32))

    # A clean tree visited first, with a cycle off to the side. The pruning must not
    # swallow the cycle just because nodes 0-2 were already cleared.
    assert _fastcore.has_cycles(np.array([-1, 0, 1, 4, 3], dtype=np.int32))


@pytest.mark.parametrize("swc", [swc32(), swc64()])
def test_break_segments(swc):
    nodes, parents, _ = swc
    start = time.time()
    segments = fastcore.break_segments(nodes, parents)
    dur = time.time() - start

    print("Broken segments:", segments)
    print(f"Timing: {dur:.4f}s")


@pytest.mark.parametrize("swc", [swc32(), swc64()])
@pytest.mark.parametrize("node_colors", [None, np.random.rand(N_NODES)])
def test_segment_coords(swc, node_colors):
    nodes, parents, coords = swc
    start = time.time()
    coords = fastcore.segment_coords(nodes, parents, coords, node_colors=node_colors)
    dur = time.time() - start

    print("Segment coordinates:", coords)
    print(f"Timing: {dur:.4f}s")


@pytest.mark.parametrize("swc", [swc32(), swc64()])
@pytest.mark.parametrize("directed", [True, False])
@pytest.mark.parametrize("sources", [None, [1, 2, 3]])
@pytest.mark.parametrize("targets", [None, [1, 2, 3]])
@pytest.mark.parametrize("weights", [None, np.random.rand(N_NODES)])
def test_geodesic_distance(swc, directed, sources, targets, weights):
    nodes, parents, _ = swc
    start = time.time()
    dists = fastcore.geodesic_matrix(
        nodes,
        parents,
        directed=directed,
        sources=sources,
        targets=targets,
        weights=weights,
    )
    dur = time.time() - start

    print("Distances:", dists)
    print(f"Timing: {dur:.4f}s")


@pytest.mark.parametrize("swc", [swc32(), swc64()])
@pytest.mark.parametrize("directed", [True, False])
@pytest.mark.parametrize(
    "sources,targets",
    [
        (None, None),
        ([1, 2, 3, 50, 51], [4, 5, 6, 50, 99]),  # overlapping sets (self-exclusion)
        ([10, 20, 30], None),
    ],
)
@pytest.mark.parametrize("weights", [None, np.random.rand(N_NODES)])
def test_geodesic_nearest(swc, directed, sources, targets, weights):
    nodes, parents, _ = swc

    dist, nearest = fastcore.geodesic_nearest(
        nodes,
        parents,
        sources=sources,
        targets=targets,
        directed=directed,
        weights=weights,
    )

    # Build an oracle from the full (masked) distance matrix
    mat = fastcore.geodesic_matrix(
        nodes,
        parents,
        directed=directed,
        sources=sources,
        targets=targets,
        weights=weights,
    ).astype(float)
    mat[mat < 0] = np.inf

    src_arr = np.asarray(sources) if sources is not None else nodes
    tgt_arr = np.asarray(targets) if targets is not None else nodes

    # Mask self-matches (a source that is also a target must not match itself)
    for i, s in enumerate(src_arr):
        mat[i, tgt_arr == s] = np.inf

    oracle_dist = mat.min(axis=1)
    has = np.isfinite(oracle_dist)

    # Distances must match the per-row minimum
    got = dist.astype(float).copy()
    got[got < 0] = np.inf
    assert np.allclose(got[has], oracle_dist[has], atol=1e-4)

    # Sources with no reachable (distinct) target must be -1 / -1
    assert np.all(dist[~has] == -1)
    assert np.all(nearest[~has] == -1)

    # The returned nearest target must actually sit at the minimum distance
    # (compare distances rather than IDs to be robust against ties).
    tgt_to_col = {t: j for j, t in enumerate(tgt_arr)}
    for i in np.where(has)[0]:
        j = tgt_to_col[nearest[i]]
        assert np.isclose(mat[i, j], oracle_dist[i], atol=1e-4)


@pytest.mark.parametrize("swc", [swc32(), swc64()])
@pytest.mark.parametrize("directed", [True, False])
@pytest.mark.parametrize(
    "sources,targets",
    [
        (None, None),
        ([1, 2, 3, 50, 51], [4, 5, 6, 50, 99]),  # overlapping sets (self-exclusion)
        ([10, 20, 30], None),
    ],
)
@pytest.mark.parametrize("weights", [None, np.random.rand(N_NODES)])
def test_geodesic_farthest(swc, directed, sources, targets, weights):
    nodes, parents, _ = swc

    dist, farthest = fastcore.geodesic_farthest(
        nodes,
        parents,
        sources=sources,
        targets=targets,
        directed=directed,
        weights=weights,
    )

    # Build an oracle from the full (masked) distance matrix. Note that unreachable pairs must
    # be masked with -inf here so they *lose* the maximum instead of winning it.
    mat = fastcore.geodesic_matrix(
        nodes,
        parents,
        directed=directed,
        sources=sources,
        targets=targets,
        weights=weights,
    ).astype(float)
    mat[mat < 0] = -np.inf

    src_arr = np.asarray(sources) if sources is not None else nodes
    tgt_arr = np.asarray(targets) if targets is not None else nodes

    # Mask self-matches (a source that is also a target must not match itself)
    for i, s in enumerate(src_arr):
        mat[i, tgt_arr == s] = -np.inf

    oracle_dist = mat.max(axis=1)
    has = np.isfinite(oracle_dist)

    # Distances must match the per-row maximum
    got = dist.astype(float).copy()
    got[got < 0] = -np.inf
    assert np.allclose(got[has], oracle_dist[has], atol=1e-4)

    # Sources with no reachable (distinct) target must be -1 / -1
    assert np.all(dist[~has] == -1)
    assert np.all(farthest[~has] == -1)

    # The returned farthest target must actually sit at the maximum distance
    # (compare distances rather than IDs to be robust against ties).
    tgt_to_col = {t: j for j, t in enumerate(tgt_arr)}
    for i in np.where(has)[0]:
        j = tgt_to_col[farthest[i]]
        assert np.isclose(mat[i, j], oracle_dist[i], atol=1e-4)


@pytest.mark.parametrize("swc", [swc32(), swc64()])
@pytest.mark.parametrize("directed", [True, False])
@pytest.mark.parametrize("weights", [None, np.random.rand(N_NODES)])
def test_geodesic_pairs(swc, directed, weights):
    nodes, parents, _ = swc
    pairs = np.vstack(
        (
            np.random.choice(nodes, 200),
            np.random.choice(nodes, 200),
        )
    ).T
    start = time.time()
    dists = fastcore.geodesic_pairs(
        nodes,
        parents,
        pairs=pairs,
        directed=directed,
        weights=weights,
    )
    dur = time.time() - start

    print("Distances:", dists)
    print(f"Timing: {dur:.4f}s")


@pytest.mark.parametrize("mode", ["centrifugal", "centripetal", "sum"])
def test_synapse_flow_centrality(swc_with_synapses, mode):
    nodes, parents, _, presynapses, postsynapses = swc_with_synapses
    start = time.time()
    cent = fastcore.synapse_flow_centrality(
        nodes, parents, presynapses, postsynapses, mode=mode
    )
    dur = time.time() - start

    print("Synapse flows:", cent)
    print(f"Timing: {dur:.4f}s")


@pytest.mark.parametrize("swc", [swc32(), swc64()])
@pytest.mark.parametrize("root_dist", [None, 0])
def test_parent_dist(swc, coords, root_dist):
    nodes, parents, _ = swc
    start = time.time()
    dists = fastcore.dag.parent_dist(nodes, parents, coords, root_dist)
    dur = time.time() - start

    print("Parent distances:", dists)
    print(f"Timing: {dur:.4f}s")


@pytest.mark.parametrize("swc", [swc32(), swc64()])
def test_connected_components(swc):
    nodes, parents, _ = swc
    start = time.time()
    cc = fastcore.connected_components(nodes, parents)
    dur = time.time() - start

    print("# of connected components:", len(np.unique(cc)))
    # print(type(segments), type(segments[0]))
    print(f"Timing: {dur:.4f}s")


@pytest.mark.parametrize("swc", [swc32(), swc64()])
@pytest.mark.parametrize("threshold", [5, 10])
@pytest.mark.parametrize("weights", [None, np.random.rand(N_NODES)])
@pytest.mark.parametrize("mask", [None, (np.random.rand(N_NODES) > 0.5).astype(bool)])
def test_prune_twigs(swc, threshold, weights, mask):
    nodes, parents, _ = swc
    start = time.time()
    pruned = fastcore.prune_twigs(
        nodes, parents, threshold=threshold, weights=weights, mask=mask
    )
    dur = time.time() - start

    print("Pruned nodes:", pruned)
    print(f"Timing: {dur:.4f}s")


@pytest.mark.parametrize("swc", [swc32(), swc64()])
@pytest.mark.parametrize("method", ["standard", "greedy"])
@pytest.mark.parametrize("min_twig_size", [None, 5])
def test_strahler_index(swc, method, min_twig_size):
    nodes, parents, _ = swc
    start = time.time()
    si = fastcore.strahler_index(
        nodes, parents, method=method, min_twig_size=min_twig_size
    )
    dur = time.time() - start

    print("Strahler indices:", si)
    print(f"Timing: {dur:.4f}s")


def test_subtree_height_example():
    #   0 - 1 - 2 - 3
    #        \
    #         4 - 5 - 6
    #              \
    #               7
    nodes = np.arange(8)
    parents = np.array([-1, 0, 1, 2, 1, 4, 5, 5])

    heights = fastcore.subtree_height(nodes, parents)

    # Node 1 must take its 3-hop branch (4, 5, 6), not its 2-hop one (2, 3)
    assert_array_equal(heights, [4, 3, 1, 0, 2, 1, 0, 0])


@pytest.mark.parametrize("swc", [swc32(), swc64()])
def test_subtree_height(swc):
    nodes, parents, coords = swc
    weights = fastcore.dag.parent_dist(nodes, parents, coords, root_dist=0)

    heights = fastcore.subtree_height(nodes, parents, weights=weights)

    assert heights.dtype == np.float32
    assert len(heights) == len(nodes)
    assert (heights >= 0).all()

    # Leafs (nodes that are nobody's parent) sit at the bottom -> height exactly 0
    is_leaf = ~np.isin(nodes, parents)
    assert (heights[is_leaf] == 0).all()

    # Cross-check against the reference formulation: a node's height is the depth of the deepest
    # leaf below it minus its own depth. We compute the reference in float64 because that
    # subtraction cancels two large depths against each other -- which is exactly why
    # `subtree_height` does *not* work that way, so the tolerance below is the reference's error
    # budget, not ours.
    depth = fastcore.dist_to_root(nodes, parents, weights=weights).astype(np.float64)
    ix = {n: i for i, n in enumerate(nodes.tolist())}
    parent_ix = np.array([ix.get(p, -1) for p in parents.tolist()], dtype=np.int64)
    deepest = np.full(len(nodes), -np.inf)
    leaf_ix = np.where(is_leaf)[0]
    for i in leaf_ix[np.argsort(-depth[leaf_ix])]:
        dl, v = depth[i], i
        while v != -1 and deepest[v] < dl:
            deepest[v] = dl
            v = parent_ix[v]
    expected = deepest - depth

    assert np.isfinite(expected).all()  # every node has at least one leaf below it
    np.testing.assert_allclose(heights, expected, rtol=1e-3, atol=1.0)


@pytest.mark.parametrize("swc", [swc32(), swc64()])
def test_dist_to_root(swc):
    nodes, parents, coords = swc
    weights = fastcore.dag.parent_dist(nodes, parents, coords, root_dist=0)

    dists = fastcore.dist_to_root(nodes, parents, weights=weights)

    assert dists.dtype == np.float32
    assert len(dists) == len(nodes)
    assert (dists[parents < 0] == 0).all()  # a root is 0 away from itself

    # Same as the geodesic distance from the root to every node
    roots = nodes[parents < 0]
    assert len(roots) == 1, "test SWC is expected to have a single root"
    ref = fastcore.geodesic_matrix(nodes, parents, sources=roots, weights=weights)
    np.testing.assert_allclose(dists, ref[0], rtol=1e-5, atol=1e-3)

    # `sources` subsets and preserves the order the caller passed them in
    some = nodes[[100, 5, 1]]
    np.testing.assert_array_equal(
        fastcore.dist_to_root(nodes, parents, sources=some, weights=weights),
        dists[[100, 5, 1]],
    )


@pytest.mark.parametrize("swc", [swc32(), swc64()])
def test_classify_nodes(swc):
    nodes, parents, _ = swc
    start = time.time()
    types = fastcore.classify_nodes(nodes, parents)
    dur = time.time() - start

    print("Node types:", types)
    print(f"Timing: {dur:.4f}s")

    # A node's type is determined by its number of children, except for roots which
    # outrank every other class (a root with no children is still a root, not a leaf).
    n_children = (
        pd.Series(parents[parents >= 0]).value_counts().reindex(nodes).fillna(0).values
    )
    is_root = parents < 0

    assert types.dtype == np.int32
    assert len(types) == len(nodes)
    assert set(np.unique(types)) <= {0, 1, 2, 3}  # never the -1 "unvisited" sentinel
    assert (types[is_root] == 0).all()
    assert (types[~is_root & (n_children == 0)] == 1).all()
    assert (types[~is_root & (n_children >= 2)] == 2).all()
    assert (types[~is_root & (n_children == 1)] == 3).all()


def test_classify_nodes_example():
    """The example from the docstring: 0=root, 1=leaf, 2=branch point, 3=slab."""
    node_ids = np.arange(8)
    parent_ids = np.array([-1, 0, 1, 2, 1, 4, 5, 5])

    types = fastcore.classify_nodes(node_ids, parent_ids)

    assert_array_equal(types, [0, 2, 3, 1, 3, 2, 1, 1])


@pytest.mark.parametrize("swc", [swc32(), swc64()])
def test_classify_nodes_is_order_invariant(swc):
    """Classification depends only on the topology, not on the order nodes are given in.

    `classify_nodes` counts children rather than walking leaf-to-root, so this should hold
    trivially -- but it is the property that lets us skip sorting the leaves, so pin it.
    """
    nodes, parents, _ = swc
    types = fastcore.classify_nodes(nodes, parents)

    perm = np.random.permutation(len(nodes))
    shuffled = fastcore.classify_nodes(nodes[perm], parents[perm])

    assert_array_equal(shuffled, types[perm])


# NBLAST is covered by tests/test_nblast.py.


if __name__ == "__main__":
    print("Done")
