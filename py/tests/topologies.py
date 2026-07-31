"""Skeleton topologies shared by the parity, property and performance suites.

Nearly every bug in tree code is a *degenerate shape*, not a wrong algorithm: a
component with a single node, a root that is also a branch point, an ID column
that is not in row order. This module enumerates those shapes once so every
suite tests against the same matrix.

A [`Topology`][] is the crate's tree representation as the bindings take it:

- `node_ids` — `(N,)` arbitrary IDs, **not** necessarily sorted or contiguous.
- `parent_ids` — `(N,)` the parent's *ID* for each node; roots are `-1`.
- `weights` — `(N,)` float32 length of each node's child->parent edge, indexed
  by the **child**. A root's entry is unused; it is set to 0.

`SMALL` is the default matrix: cheap enough that every parity test and property
test can run the whole thing.

Scale is covered elsewhere, deliberately - running the igraph oracle over 100k
nodes costs minutes for what the small matrix already establishes:

- `synthetic_neuron` and `fragmented_neuron` build a realistic arbor, and a
  realistic *shattered* arbor, at any size. The performance suite measures every
  operation against both at 100k and 1M nodes (`tests/test_perf.py`).
- `STRESS` holds the degenerate extremes - a 100k-deep chain and a 100k-wide star.
  Their job is to break recursive traversals and per-node allocation, which is
  checked where it belongs: in the Rust unit tests, against 200k-node versions.
"""

from collections import namedtuple
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["Topology", "SMALL", "STRESS", "load_swc"]

Topology = namedtuple("Topology", ["name", "node_ids", "parent_ids", "weights"])

DATA = Path(__file__).parent


def _weights(n, seed, zeros=(), roots=None):
    """Deterministic positive edge weights, with `zeros` forced to 0.

    Seeded per topology rather than off the global RNG so that importing this
    module cannot be perturbed by (or perturb) another test's `np.random.seed`.

    Roots are forced to 0. A root has no child->parent edge, so its slot is
    meaningless - but it is *read* by anything that sums a weight vector over a
    node set, so leaving noise there makes a fixture disagree with a real
    skeleton (where `load_swc` necessarily leaves roots at 0) for reasons that
    have nothing to do with the code under test.
    """
    w = np.random.default_rng(seed).uniform(0.5, 10.0, size=n).astype(np.float32)
    w[list(zeros)] = 0.0
    if roots is not None:
        w[roots] = 0.0
    return w


def _topo(name, parent_ids, node_ids=None, seed=0, zeros=(), dtype=np.int64):
    """Assemble a Topology from a parent-ID vector."""
    parent_ids = np.asarray(parent_ids, dtype=np.int64)
    n = len(parent_ids)
    if node_ids is None:
        node_ids = np.arange(n, dtype=dtype)
    else:
        node_ids = np.asarray(node_ids, dtype=dtype)
    roots = np.flatnonzero(parent_ids < 0)
    return Topology(name, node_ids, parent_ids, _weights(n, seed, zeros, roots))


# --------------------------------------------------------------------------- small

def single_node():
    """One node, no edges. Everything downstream must cope with an empty edge set."""
    return _topo("single_node", [-1])


def isolated_nodes():
    """Five components of one node each: every node is simultaneously root and leaf."""
    return _topo("isolated_nodes", [-1] * 5, seed=1)


def two_nodes():
    """The smallest non-trivial tree: one root, one leaf, one edge."""
    return _topo("two_nodes", [-1, 0], seed=2)


def linear_chain(n=20):
    """No branch points at all - degenerate for anything keyed on branching."""
    return _topo("linear_chain", [-1] + list(range(n - 1)), seed=3)


def doctest_tree():
    """The 7-node tree used in the `dag.py` docstrings.

    Keeping it here means a parity failure can be read straight against the
    documented example.
    """
    return _topo("doctest_tree", [-1, 0, 1, 2, 1, 4, 5], seed=4)


def binary_tree(depth=5):
    """Perfectly balanced: every interior node is a branch point."""
    n = 2**depth - 1
    return _topo("binary_tree", [-1] + [(i - 1) // 2 for i in range(1, n)], seed=5)


def trifurcation():
    """An *interior* node with three children.

    Branch detection that tests `in_degree == 2` rather than `> 1` passes on a
    binary tree and fails here. The trifurcation is deliberately not at the root
    - a root out-ranks "branch point" in the classification, so a trifurcating
    root would test the root rule instead of the branch rule.
    """
    #   0 (root)
    #   |
    #   1
    #  /|\
    # 2 3 4
    # |
    # 5
    return _topo("trifurcation", [-1, 0, 1, 1, 1, 2], seed=6)


def root_is_branch():
    """The root itself has two children.

    igraph's `vs.select(_indegree_gt=1, _outdegree=1)` deliberately *excludes*
    such a root (it has out-degree 0), so it is a branch point that is not in
    the "branch" set - exactly the kind of asymmetry that breaks a transcribed
    reference implementation.
    """
    return _topo("root_is_branch", [-1, 0, 0, 1, 2], seed=7)


def fragmented():
    """Three components of sizes 1, 5 and 12, interleaved in row order.

    Multi-root handling is the single biggest risk area: anything that assumes
    "the root" rather than "a root" fails here.
    """
    parents = np.full(18, -1, dtype=np.int64)
    # Component A: node 0 alone.
    # Component B: 1 -> 2 -> 3, plus 4, 5 hanging off 2.
    parents[[2, 3, 4, 5]] = [1, 2, 2, 2]
    # Component C: a chain 6..13 with 14..17 branching off it.
    parents[7:14] = np.arange(6, 13)
    parents[[14, 15, 16, 17]] = [8, 8, 11, 16]
    return _topo("fragmented", parents, seed=8)


def zero_weights():
    """Coincident nodes produce zero-length edges.

    These must read as *reachable at distance 0*, never as unreachable.
    """
    return _topo("zero_weights", [-1] + list(range(11)), seed=9, zeros=(3, 4, 7))


def unsorted_ids():
    """Row order is neither ID order nor topological order.

    Catches code that assumes `node_ids` is sorted, or that a node's row index
    can stand in for its ID.
    """
    #   IDs:  50 <- 20 <- 90, 20 <- 70 <- 10, 50 <- 30
    node_ids = np.array([90, 20, 50, 10, 70, 30], dtype=np.int64)
    parent_ids = np.array([20, 50, -1, 70, 20, 50], dtype=np.int64)
    roots = np.flatnonzero(parent_ids < 0)
    return Topology("unsorted_ids", node_ids, parent_ids, _weights(6, 10, roots=roots))


def non_contiguous_ids():
    """IDs with gaps, far from 0.

    Catches anything that uses an ID as an array offset.
    """
    base = binary_tree(4)
    node_ids = (1000 + np.arange(len(base.node_ids)) * 7).astype(np.int64)
    parent_ids = np.where(base.parent_ids < 0, -1, 1000 + base.parent_ids * 7)
    return Topology("non_contiguous_ids", node_ids, parent_ids, base.weights)


def uint64_ids():
    """uint64 node IDs with int64 parent IDs - navis' actual convention.

    Node IDs from segmentation backends are routinely uint64, but `-1` cannot be
    expressed there, so the parent column stays signed. `_ids_to_indices` casts
    both to a common signed type; uint64 *parents* are rejected outright.
    """
    base = doctest_tree()
    return Topology(
        "uint64_ids",
        base.node_ids.astype(np.uint64),
        base.parent_ids.astype(np.int64),
        base.weights,
    )


def load_swc(file="722817260.swc", name=None):
    """A real skeleton, with true Euclidean edge lengths as weights."""
    swc = pd.read_csv(DATA / file, comment="#", header=None, sep=" ")
    node_ids = swc[0].values.astype(np.int64)
    parent_ids = swc[6].values.astype(np.int64)
    coords = swc[[2, 3, 4]].values.astype(np.float64)

    # Edge length per child. Roots keep 0 - their entry is never read.
    ix = pd.Index(node_ids).get_indexer(parent_ids)
    weights = np.zeros(len(node_ids), dtype=np.float32)
    has_parent = ix >= 0
    weights[has_parent] = np.linalg.norm(
        coords[has_parent] - coords[ix[has_parent]], axis=1
    ).astype(np.float32)

    return Topology(name or file.split(".")[0], node_ids, parent_ids, weights)


def real_swc():
    """~4.5k nodes with realistic branch statistics."""
    return load_swc("722817260.swc", name="real_swc")


# -------------------------------------------------------------------------- stress

def deep_chain(n=100_000):
    """A 100k-node chain of depth 100k.

    The natural way to write a sub-tree walk is a recursive DFS, and it will
    blow the stack here. This fixture is the reason every traversal in the crate
    has to be iterative.
    """
    return _topo("deep_chain", [-1] + list(range(n - 1)), seed=11)


def wide_star(n=100_000):
    """100k leaves on one root.

    The mirror image of `deep_chain`: depth 1, fan-out 100k. Adjacency stored as
    a `Vec<Vec<i32>>` degenerates into one huge inner vector here.
    """
    return _topo("wide_star", [-1] + [0] * (n - 1), seed=12)


def real_swc_large():
    """The larger of the two committed skeletons."""
    return load_swc("754534424.swc", name="real_swc_large")


def synthetic_neuron(n, seed=42):
    """A tree of `n` nodes shaped roughly like a skeleton, at any size.

    A long backbone (a third of the nodes) with the rest hanging off random
    earlier nodes as twigs. Not a statistically faithful neuron, but it has the
    two properties the performance suite needs: a depth that grows with `n`, so
    traversal cost is not hidden by a shallow tree, and an exactly reproducible
    shape at every size, so timings at `n` and `10n` are comparable.
    """
    parents = np.full(n, -1, dtype=np.int64)
    backbone = max(1, n // 3)
    parents[1:backbone] = np.arange(backbone - 1)
    if n > backbone:
        # Each remaining node attaches to a uniformly random *earlier* node,
        # which cannot create a cycle.
        rng = np.random.default_rng(seed)
        parents[backbone:] = rng.integers(0, np.arange(backbone, n))
    return _topo(f"synthetic_{n}", parents, seed=seed)


def fragmented_neuron(n=100_000, seed=43):
    """One large arbor plus several thousand small fragments, at any size.

    The shape a segmentation-derived skeleton actually arrives in, and the one most
    likely to catch multi-root bugs: half the nodes form a single arbor, the rest are
    scattered across thousands of 1-20 node pieces. Anything that assumes "the root"
    rather than "a root", or that walks the root list once per component, degrades
    here in a way no single-component fixture can show.

    Deterministic at every size, so timings at `n` and `10n` stay comparable.
    """
    rng = np.random.default_rng(seed)
    parents = np.full(n, -1, dtype=np.int64)

    # The large component: a backbone with twigs, as `synthetic_neuron` builds it.
    big = max(1, n // 2)
    backbone = max(1, big // 3)
    parents[1:backbone] = np.arange(backbone - 1)
    if big > backbone:
        parents[backbone:big] = rng.integers(0, np.arange(backbone, big))

    # The rest: consecutive runs of 1-20 nodes, each its own component.
    rest = n - big
    if rest > 0:
        sizes = rng.integers(1, 21, size=rest)
        starts = big + np.concatenate([[0], np.cumsum(sizes)])
        starts = starts[starts < n]

        idx = np.arange(big, n)
        frag_start = starts[np.searchsorted(starts, idx, side="right") - 1]
        offset = idx - frag_start
        # Offset 0 is the fragment's root; everything else attaches to a random
        # *earlier* node of the same fragment, so no fragment can contain a cycle.
        parents[big:] = np.where(
            offset == 0,
            -1,
            frag_start + rng.integers(0, np.maximum(offset, 1)),
        )

    return _topo(f"fragmented_{n}", parents, seed=seed)


#: The default matrix. Cheap enough to run in full for every parity test.
SMALL = [
    single_node,
    isolated_nodes,
    two_nodes,
    linear_chain,
    doctest_tree,
    binary_tree,
    trifurcation,
    root_is_branch,
    fragmented,
    zero_weights,
    unsorted_ids,
    non_contiguous_ids,
    uint64_ids,
    real_swc,
]

#: Shapes that only fail at scale. Opt-in - these are too slow for every test.
STRESS = [deep_chain, wide_star, real_swc_large]


def parent_map(node_ids, parent_ids):
    """`{node_id: parent_id}`, with -1 for roots. Shared by both test suites."""
    return dict(zip(node_ids.tolist(), parent_ids.tolist()))


def ancestors(parents, node):
    """`node` and every node above it, root last.

    The "walk up the parent map" loop, written once - it was open-coded in six places
    across the two test modules.
    """
    out = []
    while node != -1:
        out.append(node)
        node = parents[node]
    return out


