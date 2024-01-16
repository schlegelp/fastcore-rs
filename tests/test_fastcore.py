import fastcore
import numpy as np
import pandas as pd
import time
from pathlib import Path


def _node_indices_py(A, B):
    """Reference Python implementation of _node_indices."""
    ix_dict = dict(zip(B, np.arange(len(B))))

    return np.array([ix_dict.get(p, -1) for p in A])


def test_node_indices():
    nodes, parents = _load_swc()

    start = time.time()
    indices = fastcore._fastcore.node_indices(nodes, parents)
    dur = time.time() - start

    start = time.time()
    indices_py = _node_indices_py(parents, nodes)
    dur_py = time.time() - start

    assert all(indices == indices_py)

    print("Indices:", indices)
    print(f"Timing: {dur:.4f}s (compared to {dur_py:.4f}s in pure numpy Python)")


def test_generate_segments():
    nodes, parents = _load_swc()

    start = time.time()
    segments = fastcore._fastcore.generate_segments(
        fastcore._fastcore.node_indices(nodes, parents)
    )
    dur = time.time() - start

    # print("Segments:", segments)
    print(type(segments), type(segments[0]))
    print(f"Timing: {dur:.4f}s")


def test_geodesic_distance():
    nodes, parents = _load_swc()

    start = time.time()
    dists = fastcore._fastcore.geodesic_distances(
        fastcore._fastcore.node_indices(nodes, parents)
    )
    dur = time.time() - start

    print("Distances:", dists)
    print(f"Timing: {dur:.4f}s")


def test_synapse_flow_centrality():
    nodes, parents, presynapses, postsynapses = _load_swc(synapses=True)

    start = time.time()
    dists = fastcore._fastcore.synapse_flow_centrality(
        fastcore._fastcore.node_indices(nodes, parents),
        presynapses,
        postsynapses,
    )
    dur = time.time() - start

    print("Synapse flows:", dists)
    print(f"Timing: {dur:.4f}s")


def _load_swc(file="722817260.swc", synapses=False):
    fp = Path(__file__).parent / file
    swc = pd.read_csv(fp, comment="#", header=None, sep=" ")
    nodes = swc[0].values.astype(np.int64)
    parents = swc[6].values.astype(np.int64)

    if synapses:
        connectors = pd.read_csv(Path(__file__).parent / file.replace(".swc", ".csv"))
        presynapses = (
            connectors[connectors.type == "pre"]
            .node_id.value_counts()
            .reindex(nodes)
            .fillna(0)
            .values.astype(np.int32, order="C", copy=False)
        )
        postsynapses = (
            connectors[connectors.type == "post"]
            .node_id.value_counts()
            .reindex(nodes)
            .fillna(0)
            .values.astype(np.int32, order="C", copy=False)
        )
        return nodes, parents, presynapses, postsynapses

    return nodes, parents


def test_connected_components():
    nodes, parents = _load_swc()

    start = time.time()
    cc = fastcore._fastcore.connected_components(
        fastcore._fastcore.node_indices(nodes, parents)
    )
    dur = time.time() - start

    print("# of connected components:", len(np.unique(cc)))
    # print(type(segments), type(segments[0]))
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
    test_node_indices()
    test_generate_segments()
    test_geodesic_distance()
    print("Done")
