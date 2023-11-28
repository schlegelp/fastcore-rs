import fastcore
import numpy as np


def test_something():
    nodes = np.array([10, 11, 12, 15, 10], dtype=np.int32)
    parents = np.array([-1, 10, 11, 12, 12], dtype=np.int32)
    indices = np.zeros(len(nodes), dtype=np.int32)
    print("Indices before", indices)
    fastcore._fastcore._node_indices(nodes, parents, indices)
    print("Indices after", indices)


if __name__ == "__main__":
    test_something()
    print("Done")
