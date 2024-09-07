import pytest
import time

import navis_fastcore as fastcore
import numpy as np
import pandas as pd

from pathlib import Path
from collections import namedtuple

Dotprop = namedtuple("Dotprop", ["points", "vect"])

# Set random state
np.random.seed(0)


def _load_swc(file="722817260.swc", synapses=False):
    fp = Path(__file__).parent / file
    swc = pd.read_csv(fp, comment="#", header=None, sep=" ")
    nodes = swc[0].values.astype(np.int64)
    coords = swc[[2, 3, 4]].values.astype(np.float64)
    parents = swc[6].values.astype(np.int64)

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


@pytest.fixture
def swc():
    return _load_swc(synapses=False)


@pytest.fixture
def swc_with_synapses():
    return _load_swc(synapses=True)


@pytest.fixture
def nodes():
    return _load_swc()[0]


@pytest.fixture
def parents():
    return _load_swc()[1]


@pytest.fixture
def coords():
    return _load_swc()[2]


@pytest.fixture
def presynapses():
    return _load_swc(synapses=True)[3]


@pytest.fixture
def postsynapses():
    return _load_swc(synapses=True)[4]


def _node_indices_py(A, B):
    """Reference Python implementation of _node_indices."""
    ix_dict = dict(zip(B, np.arange(len(B))))

    return np.array([ix_dict.get(p, -1) for p in A])


def test_node_indices(nodes, parents):
    start = time.time()
    indices = fastcore.dag._ids_to_indices(nodes, parents)
    dur = time.time() - start

    start = time.time()
    indices_py = _node_indices_py(parents, nodes)
    dur_py = time.time() - start

    assert all(indices == indices_py)

    print("Indices:", indices)
    print(f"Timing: {dur:.4f}s (compared to {dur_py:.4f}s in pure numpy Python)")


@pytest.mark.parametrize("weights", [None, np.random.rand(N_NODES)])
def test_generate_segments(nodes, parents, weights):
    start = time.time()
    segments = fastcore.generate_segments(nodes, parents, weights=weights)
    dur = time.time() - start

    # print("Segments:", segments)
    print(type(segments), type(segments[0]))
    print(f"Timing: {dur:.4f}s")


def test_break_segments(nodes, parents):
    start = time.time()
    segments = fastcore.break_segments(nodes, parents)
    dur = time.time() - start

    print("Broken segments:", segments)
    print(f"Timing: {dur:.4f}s")


@pytest.mark.parametrize("node_colors", [None, np.random.rand(N_NODES)])
def test_segment_coords(nodes, parents, coords, node_colors):
    start = time.time()
    coords = fastcore.segment_coords(nodes, parents, coords, node_colors=node_colors)
    dur = time.time() - start

    print("Segment coordinates:", coords)
    print(f"Timing: {dur:.4f}s")


@pytest.mark.parametrize("directed", [True, False])
@pytest.mark.parametrize("sources", [None, [0, 1, 2]])
@pytest.mark.parametrize("targets", [None, [0, 1, 2]])
@pytest.mark.parametrize("weights", [None, np.random.rand(N_NODES)])
def test_geodesic_distance(nodes, parents, directed, sources, targets, weights):
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


@pytest.mark.parametrize("mode", ["centrifugal", "centripetal", "sum"])
def test_synapse_flow_centrality(nodes, parents, presynapses, postsynapses, mode):
    start = time.time()
    cent = fastcore.synapse_flow_centrality(
        nodes, parents, presynapses, postsynapses, mode=mode
    )
    dur = time.time() - start

    print("Synapse flows:", cent)
    print(f"Timing: {dur:.4f}s")


@pytest.mark.parametrize("root_dist", [None, 0])
def test_parent_dist(nodes, parents, coords, root_dist):
    start = time.time()
    dists = fastcore.dag.parent_dist(nodes, parents, coords, root_dist)
    dur = time.time() - start

    print("Parent distances:", dists)
    print(f"Timing: {dur:.4f}s")


def test_connected_components(nodes, parents):
    start = time.time()
    cc = fastcore.connected_components(nodes, parents)
    dur = time.time() - start

    print("# of connected components:", len(np.unique(cc)))
    # print(type(segments), type(segments[0]))
    print(f"Timing: {dur:.4f}s")


@pytest.mark.parametrize("threshold", [5, 10])
@pytest.mark.parametrize("weights", [None, np.random.rand(N_NODES)])
def test_prune_twigs(nodes, parents, threshold, weights):
    start = time.time()
    pruned = fastcore.prune_twigs(nodes, parents, threshold=threshold, weights=weights)
    dur = time.time() - start

    print("Pruned nodes:", pruned)
    print(f"Timing: {dur:.4f}s")


@pytest.mark.parametrize("method", ["standard", "greedy"])
@pytest.mark.parametrize("min_twig_size", [None, 5])
def test_strahler_index(nodes, parents, method, min_twig_size):
    start = time.time()
    si = fastcore.strahler_index(
        nodes, parents, method=method, min_twig_size=min_twig_size
    )
    dur = time.time() - start

    print("Strahler indices:", si)
    print(f"Timing: {dur:.4f}s")


def test_classify_nodes(nodes, parents):
    start = time.time()
    types = fastcore.classify_nodes(nodes, parents)
    dur = time.time() - start

    print("Node types:", types)
    print(f"Timing: {dur:.4f}s")


def test_nblast_single():
    fp = Path(__file__).parent / "722817260.swc"
    swc1 = pd.read_csv(fp, comment="#", header=None, sep=" ")
    xyz1 = swc1[[2, 3, 4]].values.astype(np.float64) / 1000
    vec1 = calculate_tangent_vectors(xyz1, k=5)

    fp = Path(__file__).parent / "754534424.swc"
    swc2 = pd.read_csv(fp, comment="#", header=None, sep=" ")
    xyz2 = swc2[[2, 3, 4]].values.astype(np.float64) / 1000
    vec2 = calculate_tangent_vectors(xyz2, k=5)

    start = time.time()
    score = fastcore._fastcore.nblast_single(xyz1, vec1, xyz2, vec2, parallel=True)
    dur = time.time() - start

    print(f"NBLAST score: {score} ({dur:.4f}s)")


def calculate_tangent_vectors(points, k):
    """Calculate tangent vectors.

    Parameters
    ----------
    k :         int
                Number of nearest neighbours to use for tangent vector
                calculation.

    Returns
    -------
    Dotprops
                Only if ``inplace=False``.

    """
    # Create the KDTree and get the k-nearest neighbors for each point
    from scipy.spatial import cKDTree as KDTree

    dist, ix = KDTree(points).query(points, k=k)

    # Get points: array of (N, k, 3)
    pt = points[ix]

    # Generate centers for each cloud of k nearest neighbors
    centers = np.mean(pt, axis=1)

    # Generate vector from center
    cpt = pt - centers.reshape((pt.shape[0], 1, 3))

    # Get inertia (N, 3, 3)
    inertia = cpt.transpose((0, 2, 1)) @ cpt

    # Extract vector and alpha
    u, s, vh = np.linalg.svd(inertia)
    vect = vh[:, 0, :]
    # alpha = (s[:, 0] - s[:, 1]) / np.sum(s, axis=1)

    return vect


if __name__ == "__main__":
    print("Done")