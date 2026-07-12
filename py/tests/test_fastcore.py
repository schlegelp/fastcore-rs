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

    # print("Segments:", segments)
    print(type(segments), type(segments[0]))
    print(f"Timing: {dur:.4f}s")


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
