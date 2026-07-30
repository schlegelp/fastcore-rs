"""Parity between fastcore and igraph on the tree algorithms navis shares with it.

navis currently runs each of these twice - once through fastcore, once through an
igraph fallback - and picks whichever backend is installed. Retiring the fallback
means fastcore has to be demonstrably the same function, not merely a faster one.
Proving that *here* rather than in navis means navis' migration consumes an
already-verified backend instead of porting and verifying at the same time.

igraph is a test-only dependency; this module skips wholesale without it. The
reference implementations live in `igraph_oracle.py`, transcribed from navis'
fallbacks and cited by line.

Where fastcore and igraph legitimately differ, the divergence is *pinned* by a
test that asserts the exact relationship rather than papered over with a loose
tolerance - see `test_generate_segments_length_definition`.
"""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

igraph = pytest.importorskip("igraph")  # noqa: F841  - test-only dependency

import navis_fastcore as fastcore  # noqa: E402
import igraph_oracle as oracle  # noqa: E402

#: fastcore searches in float32, igraph in float64. Over a ~54,000-unit path on
#: the real skeleton that is a measured worst case of 5.7e-07 *relative* error, so
#: the comparison has to be relative; a fixed atol would either fail on long paths
#: or be meaningless on short ones. `atol` only covers values at or near zero.
RTOL, ATOL = 1e-5, 1e-5


def _sample(values, k, seed):
    """A deterministic subset of `values`, or all of them if there are fewer."""
    values = np.asarray(values)
    if len(values) <= k:
        return values
    rng = np.random.default_rng(seed)
    return values[np.sort(rng.choice(len(values), size=k, replace=False))]


# --------------------------------------------------------------------------- meta


def test_oracle_graph_matches_topology(topo):
    """`as_igraph` must reproduce the topology, or every test below is vacuous."""
    g = oracle.as_igraph(topo)

    assert g.vcount() == len(topo.node_ids)
    assert g.ecount() == int((topo.parent_ids >= 0).sum())
    assert g.is_directed()
    assert_array_equal(np.asarray(g.vs["node_id"]), topo.node_ids)

    # Edges must run child -> parent, so roots (and only roots) have out-degree 0.
    assert_array_equal(
        np.asarray(g.outdegree()) == 0,
        topo.parent_ids < 0,
    )


# ----------------------------------------------------------------- classify_nodes


def test_classify_nodes(topo):
    """Node types are integer codes - compare exactly, never approximately."""
    ours = fastcore.classify_nodes(topo.node_ids, topo.parent_ids)
    theirs = oracle.classify_nodes(oracle.as_igraph(topo))

    assert_array_equal(ours, theirs)


def test_classify_nodes_root_outranks_branch():
    """A root with two children is a root, not a branch point.

    Pinned separately because it is a precedence rule rather than a computation,
    and because both implementations could drift to "branch" together.
    """
    node_ids = np.array([0, 1, 2], dtype=np.int64)
    parent_ids = np.array([-1, 0, 0], dtype=np.int64)

    types = fastcore.classify_nodes(node_ids, parent_ids)

    assert types[0] == oracle.ROOT
    assert list(types[1:]) == [oracle.LEAF, oracle.LEAF]


def test_classify_nodes_isolated_is_root():
    """An isolated node is a root, not a leaf - it has no parent."""
    node_ids = np.array([7, 8], dtype=np.int64)
    parent_ids = np.array([-1, -1], dtype=np.int64)

    assert list(fastcore.classify_nodes(node_ids, parent_ids)) == [oracle.ROOT] * 2


# ------------------------------------------------------------ connected components


def test_connected_components(topo):
    """Compare partitions, not labels.

    fastcore labels a component by its root's node ID and igraph by a running
    integer, so the raw arrays never match; canonicalising both still catches a
    node placed in the wrong component.
    """
    ours = fastcore.connected_components(topo.node_ids, topo.parent_ids)
    theirs = oracle.connected_components(oracle.as_igraph(topo))

    assert_array_equal(
        oracle.canonical_labels(ours),
        oracle.canonical_labels(theirs),
    )


def test_connected_components_labels_are_root_ids(topo):
    """fastcore's documented contract: the label *is* the component's root ID."""
    labels = fastcore.connected_components(topo.node_ids, topo.parent_ids)
    roots = set(topo.node_ids[topo.parent_ids < 0].tolist())

    assert set(labels.tolist()) <= roots
    # ...and every root labels itself.
    is_root = topo.parent_ids < 0
    assert_array_equal(labels[is_root], topo.node_ids[is_root])


# ------------------------------------------------------------------- dist_to_root


@pytest.mark.parametrize("weighted", [True, False])
def test_dist_to_root(topo, weighted):
    weights = topo.weights if weighted else None

    ours = fastcore.dist_to_root(topo.node_ids, topo.parent_ids, weights=weights)
    theirs = oracle.dist_to_root(oracle.as_igraph(topo, weighted), weighted)

    assert np.allclose(ours, theirs, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("weighted", [True, False])
def test_dist_to_root_sources_order(topo, weighted):
    """`sources` must come back in the order asked for, not in node-table order.

    navis indexes the result positionally, so a silently re-sorted return value
    would misattribute distances rather than raise.
    """
    weights = topo.weights if weighted else None
    full = fastcore.dist_to_root(topo.node_ids, topo.parent_ids, weights=weights)

    # Reverse order: distinguishes "ordered as requested" from "happens to be sorted".
    sources = topo.node_ids[::-1]
    subset = fastcore.dist_to_root(
        topo.node_ids, topo.parent_ids, sources=sources, weights=weights
    )

    assert np.allclose(subset, full[::-1], rtol=RTOL, atol=ATOL)


# ---------------------------------------------------------------- geodesic_matrix


@pytest.mark.parametrize("weighted", [True, False])
@pytest.mark.parametrize("directed", [True, False])
def test_geodesic_matrix_all_by_all(topo, weighted, directed):
    """Full matrix, on everything small enough for igraph to answer in Python.

    igraph hands back a list of lists, so an all-by-all on the real skeleton would
    materialise ~19M Python floats. That case is covered by the partial test below.
    """
    if len(topo.node_ids) > 200:
        pytest.skip("all-by-all via igraph is O(N^2) Python objects")

    weights = topo.weights if weighted else None

    ours = fastcore.geodesic_matrix(
        topo.node_ids, topo.parent_ids, directed=directed, weights=weights
    )
    theirs = oracle.geodesic_matrix(
        oracle.as_igraph(topo, weighted), directed=directed, weighted=weighted
    )

    assert np.allclose(ours, theirs, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("weighted", [True, False])
@pytest.mark.parametrize("directed", [True, False])
def test_geodesic_matrix_partial(topo, weighted, directed):
    """A sources x targets block, with the row/column order the caller asked for."""
    weights = topo.weights if weighted else None
    sources = _sample(topo.node_ids, 25, seed=1)
    targets = _sample(topo.node_ids, 20, seed=2)

    ours = fastcore.geodesic_matrix(
        topo.node_ids,
        topo.parent_ids,
        directed=directed,
        sources=sources,
        targets=targets,
        weights=weights,
    )

    g = oracle.as_igraph(topo, weighted)
    ix = {nid: i for i, nid in enumerate(topo.node_ids.tolist())}
    theirs = oracle.geodesic_matrix(
        g,
        sources=[ix[s] for s in sources.tolist()],
        targets=[ix[t] for t in targets.tolist()],
        directed=directed,
        weighted=weighted,
    )

    assert ours.shape == (len(sources), len(targets))
    assert np.allclose(ours, theirs, rtol=RTOL, atol=ATOL)


def test_geodesic_matrix_partial_zero_weight_ancestors():
    """REGRESSION - `directed=True` used to leak across zero-weight edges, but
    only in the partial (sources/targets) implementation.

    `geodesic_matrix` has two backends: an all-by-all one and a partial one used
    when `sources`/`targets` are given. Only the partial one was wrong.

    In `geodesic_distances_partial`, `active_sources` holds the sources on the
    current root-path - all *strict ancestors* of the node being visited, since
    the node itself is pushed only afterwards (`dag.rs:779`). Under `directed`
    every one of them is in the rejected direction, so the loop at `dag.rs:760`
    must write nothing; it achieves that by breaking out on the first source that
    is not below us.

    That guard uses depth as a proxy for ancestry, which holds only while every
    edge weight is strictly positive. Give the edge weight 0 and an ancestor has
    *equal* depth, so a strict `<` did not fire and the pair was written with
    distance 0 - i.e. a parent reported as reachable from its child's direction.
    Breaking on `<=` fixes it: the only legitimate equal-depth source is the node
    itself, and that pair is written separately at `dag.rs:775`.

    Zero-length edges are not exotic - coincident nodes are routine in traced and
    resampled skeletons.

    Was confined to this function: `geodesic_nearest`, `geodesic_farthest` and
    `geodesic_pairs` all returned -1 correctly for the same input.
    """
    #  chain 0 <- 1 <- 2 <- 3, with the 2->1 edge weighted 0
    node_ids = np.array([0, 1, 2, 3], dtype=np.int64)
    parent_ids = np.array([-1, 0, 1, 2], dtype=np.int64)
    weights = np.array([0.0, 3.0, 0.0, 7.0], dtype=np.float32)

    partial = fastcore.geodesic_matrix(
        node_ids, parent_ids, directed=True, weights=weights,
        sources=node_ids, targets=node_ids,
    )
    full = fastcore.geodesic_matrix(
        node_ids, parent_ids, directed=True, weights=weights
    )

    # Node 1 is node 2's parent, so root-ward-only makes it unreachable.
    assert full[1, 2] == -1  # the all-by-all backend gets this right
    assert partial[1, 2] == -1  # ...the partial one returns 0.0
    assert_array_equal(partial, full)


def test_geodesic_matrix_unreachable_is_negative_one():
    """Unreachable pairs are -1, not inf and not nan.

    navis tests `>= 0` on the result, so an inf here would read as "reachable".
    """
    node_ids = np.array([0, 1, 2, 3], dtype=np.int64)
    parent_ids = np.array([-1, 0, -1, 2], dtype=np.int64)  # two components

    m = fastcore.geodesic_matrix(node_ids, parent_ids)

    assert m[0, 2] == -1 and m[1, 3] == -1
    assert (m[:2, :2] >= 0).all() and (m[2:, 2:] >= 0).all()


# ----------------------------------------------------------------- break_segments


def test_break_segments(topo):
    """Exact match including segment order.

    Order is part of the contract, not an implementation detail: navis'
    `segment_analysis` and its NEURON interface enumerate these, so the ordering
    ends up in *their* output. navis' fallback sorts its seeds explicitly for
    exactly this reason (navis: `graph/graph_utils.py:395-400`).
    """
    ours = fastcore.break_segments(topo.node_ids, topo.parent_ids)
    theirs = oracle.break_segments(oracle.as_igraph(topo))

    assert len(ours) == len(theirs)
    for a, b in zip(ours, theirs):
        assert_array_equal(a, b)


def test_break_segments_partition_edges(topo):
    """Structural check, independent of igraph: the segments cover every edge once.

    Segments share their endpoints (a branch point ends one and starts another),
    so it is the *edges* that partition, not the nodes.
    """
    segs = fastcore.break_segments(topo.node_ids, topo.parent_ids)

    edges = [
        (int(s[i]), int(s[i + 1])) for s in segs for i in range(len(s) - 1)
    ]
    expected = {
        (int(n), int(p))
        for n, p in zip(topo.node_ids, topo.parent_ids)
        if p >= 0
    }

    assert len(edges) == len(expected), "an edge is covered twice or not at all"
    assert set(edges) == expected


# -------------------------------------------------------------- generate_segments


@pytest.mark.parametrize("weighted", [True, False])
def test_generate_segments_sequences(topo, weighted):
    """The node sequences themselves must match igraph exactly.

    Compared as a multiset rather than in order: navis' fallback breaks ties in
    segment length by falling through to comparing the node lists themselves,
    which is arbitrary. The sort *contract* is checked separately below.
    """
    weights = topo.weights if weighted else None

    ours, _ = fastcore.generate_segments(
        topo.node_ids, topo.parent_ids, weights=weights
    )
    theirs, _ = oracle.generate_segments(oracle.as_igraph(topo, weighted), weighted)

    assert sorted(tuple(s.tolist()) for s in ours) == sorted(
        tuple(s.tolist()) for s in theirs
    )


@pytest.mark.parametrize("weighted", [True, False])
def test_generate_segments_sorted_longest_first(topo, weighted):
    """The documented contract: "sorted from longest to shortest"."""
    weights = topo.weights if weighted else None

    _, lengths = fastcore.generate_segments(
        topo.node_ids, topo.parent_ids, weights=weights
    )

    assert np.all(np.diff(lengths) <= 0), f"not monotonically decreasing: {lengths}"


@pytest.mark.parametrize("weighted", [True, False])
def test_generate_segments_length_is_first_to_last(topo, weighted):
    """A segment's length is the distance between its endpoints.

    fastcore used to sum the weight vector over *every* node in a segment,
    including the terminal one - but a segment stops *at* a branch point, whose
    own child->parent edge continues into the parent segment. That made every
    segment ending at a branch point exactly one edge too long, while segments
    ending at a root agreed (a root's weight slot is 0), and it left navis'
    `_generate_segments(..., return_lengths=True)` returning different numbers
    depending on which backend was installed.

    Both now measure first node to last, so this is a plain equality - weighted
    and unweighted alike, since `weights=None` means every edge weighs 1.
    """
    weights = topo.weights if weighted else None

    ours, our_lengths = fastcore.generate_segments(
        topo.node_ids, topo.parent_ids, weights=weights
    )
    theirs, their_lengths = oracle.generate_segments(
        oracle.as_igraph(topo, weighted), weighted
    )
    by_seg = {tuple(s.tolist()): ln for s, ln in zip(theirs, their_lengths)}

    for seg, ours_len in zip(ours, our_lengths):
        geodesic = by_seg[tuple(seg.tolist())]
        assert np.isclose(ours_len, geodesic, rtol=RTOL, atol=ATOL), (
            f"segment {seg[:3]}...{seg[-1]}: fastcore {ours_len}, "
            f"geodesic first->last {geodesic}"
        )


def test_generate_segments_single_node_segment_is_zero_length():
    """A lone node spans no edges, so its length is 0 - not its own weight."""
    node_ids = np.array([0, 1], dtype=np.int64)
    parent_ids = np.array([-1, -1], dtype=np.int64)
    weights = np.array([7.0, 9.0], dtype=np.float32)

    _, unweighted = fastcore.generate_segments(node_ids, parent_ids)
    _, weighted = fastcore.generate_segments(node_ids, parent_ids, weights=weights)

    assert list(unweighted) == [0, 0]
    assert list(weighted) == [0.0, 0.0]
