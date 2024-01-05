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
    fp = Path(__file__).parent / "722817260.swc"
    swc = pd.read_csv(fp, comment="#", header=None, sep=" ")
    nodes = swc[0].values.astype(np.int64)
    parents = swc[6].values.astype(np.int64)
    # nodes = np.array([10, 11, 12, 15, 16], dtype=np.int32)
    # parents = np.array([-1, 10, 11, 12, 12], dtype=np.int32)

    start = time.time()
    indices = fastcore._fastcore._node_indices(nodes, parents)
    dur = time.time() - start

    start = time.time()
    indices_py = _node_indices_py(parents, nodes)
    dur_py = time.time() - start

    assert all(indices == indices_py)

    print("Indices:", indices)
    print(f"Timing: {dur:.4f}s (compared to {dur_py:.4f}s in pure numpy Python)")


def test_generate_segments():
    fp = Path(__file__).parent / "722817260.swc"
    swc = pd.read_csv(fp, comment="#", header=None, sep=" ")
    nodes = swc[0].values.astype(np.int64)
    parents = swc[6].values.astype(np.int64)

    start = time.time()
    segments = fastcore._fastcore._generate_segments(
        fastcore._fastcore._node_indices(nodes, parents)
    )
    dur = time.time() - start

    #print("Segments:", segments)
    print(type(segments), type(segments[0]))
    print(f"Timing: {dur:.4f}s")


if __name__ == "__main__":
    test_node_indices()
    test_generate_segments()
    print("Done")
