"""Property tests over randomly generated forests.

The parity suite pins fastcore against igraph on a fixed matrix of shapes. These
tests come at it from the other side: generate arbitrary forests and assert the
*invariants* each function has to satisfy, whatever the input. They catch classes
of bug a fixed fixture set cannot, and - unlike the parity suite - they need no
igraph, so they run everywhere.

Where an invariant is not enough on its own, the reference is a brute-force
implementation written for obviousness rather than speed. Those are independent
of both fastcore and igraph, which is what makes them worth having even where an
oracle already exists.

`hypothesis` is a test-only dependency; this module skips wholesale without it.
"""

import numpy as np
import pytest

hypothesis = pytest.importorskip("hypothesis")  # noqa: F841 - test-only dependency

from hypothesis import given, strategies as st  # noqa: E402

import navis_fastcore as fastcore  # noqa: E402
from meshes import check_simplify_invariants  # noqa: E402
from topologies import (  # noqa: E402
    ancestors,
    check_dropping_invariants,
    check_is_forest,
    check_topology_preserved,
    parent_map,
)

# Settings profiles ("fastcore" by default, "thorough" for the nightly job) are
# registered in `conftest.py`: pytest resolves `--hypothesis-profile` before it
# imports any test module, so registering them here would be too late.


@st.composite
def forests(draw, min_size=1, max_size=60):
    """An arbitrary rooted forest, as `(node_ids, parent_ids, weights)`.

    Each node attaches to an *earlier* node or to nothing, which makes a cycle
    structurally impossible and lets the generator stay a one-liner per node. Node
    IDs are deliberately not row indices - shuffled, non-contiguous and offset - so
    that anything confusing an ID for a position fails here.
    """
    n = draw(st.integers(min_value=min_size, max_value=max_size))

    # `None` = root. Otherwise attach to any strictly earlier row.
    parent_rows = [None]
    for i in range(1, n):
        parent_rows.append(draw(st.one_of(st.none(), st.integers(0, i - 1))))

    # IDs that are neither sorted nor contiguous nor zero-based.
    ids = draw(st.permutations(list(range(1000, 1000 + 7 * n, 7))))
    node_ids = np.array(ids, dtype=np.int64)
    parent_ids = np.array(
        [-1 if p is None else node_ids[p] for p in parent_rows], dtype=np.int64
    )

    weights = np.array(
        draw(st.lists(st.floats(0.0, 100.0), min_size=n, max_size=n)),
        dtype=np.float32,
    )
    weights[parent_ids < 0] = 0.0  # a root has no edge; its slot is never read

    return node_ids, parent_ids, weights


@st.composite
def forest_and_node(draw):
    """A forest plus one of its node IDs - the shape most of these tests want."""
    forest = draw(forests())
    return forest, draw(st.sampled_from(forest[0].tolist()))


# ------------------------------------------------------------------- descendants


@given(forests())
def test_descendants_matches_brute_force(forest):
    """Against the definition: x is a descendant of v if v is on x's root-path."""
    node_ids, parent_ids, _ = forest
    parents = parent_map(node_ids, parent_ids)

    got = fastcore.descendants(node_ids, parent_ids, node_ids)
    for source, sub in zip(node_ids.tolist(), got):
        want = {x for x in node_ids.tolist() if source in ancestors(parents, x)}
        assert set(sub.tolist()) == want


@given(forests())
def test_descendants_of_roots_partition_the_forest(forest):
    """Every node lies below exactly one root."""
    node_ids, parent_ids, _ = forest
    roots = node_ids[parent_ids < 0]

    subs = fastcore.descendants(node_ids, parent_ids, roots)

    covered = [nid for sub in subs for nid in sub.tolist()]
    assert sorted(covered) == sorted(node_ids.tolist()), "not a partition"


@given(forests())
def test_descendants_of_siblings_are_disjoint(forest):
    """Two nodes with the same parent share no descendants."""
    node_ids, parent_ids, _ = forest
    by_parent = {}
    for nid, pid in zip(node_ids.tolist(), parent_ids.tolist()):
        by_parent.setdefault(pid, []).append(nid)

    for pid, siblings in by_parent.items():
        if pid == -1 or len(siblings) < 2:
            continue
        subs = fastcore.descendants(node_ids, parent_ids, siblings)
        seen = set()
        for sub in subs:
            assert not (seen & set(sub.tolist()))
            seen |= set(sub.tolist())


# ----------------------------------------------------------------- paths_to_root


@given(forests())
def test_paths_to_root_is_the_parent_chain(forest):
    node_ids, parent_ids, _ = forest
    parents = parent_map(node_ids, parent_ids)

    for source, path in zip(
        node_ids.tolist(), fastcore.paths_to_root(node_ids, parent_ids, node_ids)
    ):
        assert path.tolist() == ancestors(parents, source)


@given(forests())
def test_paths_to_root_agrees_with_dist_to_root(forest):
    """A path's edge count is the unweighted distance to the root."""
    node_ids, parent_ids, _ = forest

    paths = fastcore.paths_to_root(node_ids, parent_ids, node_ids)
    dists = fastcore.dist_to_root(node_ids, parent_ids)

    assert [len(p) - 1 for p in paths] == dists.astype(int).tolist()


# ------------------------------------------------------------------------ reroot


@given(forest_and_node())
def test_reroot_preserves_the_undirected_edge_set(forest_node):
    (node_ids, parent_ids, _), new_root = forest_node

    rerooted = fastcore.reroot(node_ids, parent_ids, [new_root])

    def edges(parents):
        return sorted(
            tuple(sorted((int(n), int(p))))
            for n, p in zip(node_ids, parents)
            if p >= 0
        )

    assert edges(rerooted) == edges(parent_ids)


@given(forest_and_node())
def test_reroot_yields_a_valid_forest(forest_node):
    """One root per component, no cycles, and the named node is a root."""
    (node_ids, parent_ids, _), new_root = forest_node

    rerooted = fastcore.reroot(node_ids, parent_ids, [new_root])

    n_components = len(np.unique(fastcore.connected_components(node_ids, parent_ids)))
    assert (rerooted < 0).sum() == n_components
    assert rerooted[node_ids == new_root][0] == -1
    # Same components as before, and still acyclic (a cycle would have merged two).
    assert (
        len(np.unique(fastcore.connected_components(node_ids, rerooted)))
        == n_components
    )


@given(forest_and_node())
def test_reroot_is_idempotent(forest_node):
    (node_ids, parent_ids, _), new_root = forest_node

    once = fastcore.reroot(node_ids, parent_ids, [new_root])
    twice = fastcore.reroot(node_ids, once, [new_root])

    assert once.tolist() == twice.tolist()


@given(forest_and_node())
def test_reroot_preserves_undirected_distances(forest_node):
    """Re-rooting changes orientation, not geometry."""
    (node_ids, parent_ids, weights), new_root = forest_node

    before = fastcore.geodesic_matrix(node_ids, parent_ids, weights=weights)
    rerooted = fastcore.reroot(node_ids, parent_ids, [new_root])
    # Weights are indexed by child, so reversing an edge moves its weight to the
    # other endpoint; rebuild the vector to match the new orientation.
    after_weights = np.zeros_like(weights)
    dist = dict(
        zip(
            zip(node_ids.tolist(), parent_ids.tolist()),
            weights.tolist(),
        )
    )
    for i, (nid, pid) in enumerate(zip(node_ids.tolist(), rerooted.tolist())):
        if pid >= 0:
            after_weights[i] = dist.get((nid, pid), dist.get((pid, nid), 0.0))
    after = fastcore.geodesic_matrix(node_ids, rerooted, weights=after_weights)

    assert np.allclose(before, after, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------- contract_nodes


@given(forests())
def test_contract_nodes_identity_is_a_no_op(forest):
    node_ids, parent_ids, _ = forest

    ids, parents = fastcore.contract_nodes(node_ids, parent_ids, node_ids)

    assert ids.tolist() == node_ids.tolist()
    assert parents.tolist() == parent_ids.tolist()


@given(forests(), st.data())
def test_contract_nodes_merging_into_a_parent_drops_exactly_one_node(forest, data):
    node_ids, parent_ids, _ = forest
    with_parent = node_ids[parent_ids >= 0].tolist()
    if not with_parent:
        return
    victim = data.draw(st.sampled_from(with_parent))
    parents = parent_map(node_ids, parent_ids)

    mapping = node_ids.copy()
    mapping[node_ids == victim] = parents[victim]

    ids, new_parents = fastcore.contract_nodes(node_ids, parent_ids, mapping)

    assert len(ids) == len(node_ids) - 1
    assert victim not in ids.tolist()
    # Still a forest, and no more roots than we started with.
    assert (new_parents < 0).sum() == (parent_ids < 0).sum()


@given(forests())
def test_contract_nodes_onto_roots_leaves_one_node_per_component(forest):
    node_ids, parent_ids, _ = forest
    roots = fastcore.connected_components(node_ids, parent_ids)

    ids, parents = fastcore.contract_nodes(node_ids, parent_ids, roots)

    assert sorted(ids.tolist()) == sorted(np.unique(roots).tolist())
    assert (parents < 0).all()


# -------------------------------------------------------------- simplify_skeleton


@given(forests())
def test_simplify_skeleton_keeps_the_non_slabs(forest):
    node_ids, parent_ids, _ = forest
    types = fastcore.classify_nodes(node_ids, parent_ids)

    ids, _, _ = fastcore.simplify_skeleton(node_ids, parent_ids)

    assert ids.tolist() == node_ids[types != 3].tolist()


@given(forests())
def test_simplify_skeleton_conserves_cable(forest):
    node_ids, parent_ids, weights = forest

    _, _, new_weights = fastcore.simplify_skeleton(
        node_ids, parent_ids, weights=weights
    )

    before = weights[parent_ids >= 0].sum()
    assert np.isclose(new_weights.sum(), before, rtol=1e-4, atol=1e-4)


@given(forests())
def test_simplify_skeleton_yields_a_valid_forest(forest):
    node_ids, parent_ids, weights = forest

    ids, parents, _ = fastcore.simplify_skeleton(
        node_ids, parent_ids, weights=weights
    )

    # Every parent survived, and the result has one root per original component.
    assert set(parents[parents >= 0].tolist()) <= set(ids.tolist())
    n_components = len(np.unique(fastcore.connected_components(node_ids, parent_ids)))
    assert (parents < 0).sum() == n_components


@given(forests())
def test_simplify_skeleton_preserves_root_to_leaf_distances(forest):
    """Dropping slabs must not move any surviving node relative to its root."""
    node_ids, parent_ids, weights = forest

    before = dict(
        zip(
            node_ids.tolist(),
            fastcore.dist_to_root(node_ids, parent_ids, weights=weights).tolist(),
        )
    )
    ids, parents, new_weights = fastcore.simplify_skeleton(
        node_ids, parent_ids, weights=weights
    )
    after = fastcore.dist_to_root(ids, parents, weights=new_weights)

    for nid, d in zip(ids.tolist(), after.tolist()):
        assert np.isclose(d, before[nid], rtol=1e-4, atol=1e-4)


# --------------------------------------------------------------------- adjacency


@given(forests(), st.booleans(), st.booleans())
def test_adjacency_round_trips_to_the_edge_set(forest, directed, transpose):
    node_ids, parent_ids, weights = forest
    n = len(node_ids)

    indptr, indices, data = fastcore.adjacency(
        node_ids, parent_ids, weights=weights,
        directed=directed, transpose=transpose,
    )

    got = set()
    for row in range(n):
        for slot in range(indptr[row], indptr[row + 1]):
            got.add((row, int(indices[slot])))

    want = set()
    for child, pid in enumerate(parent_ids.tolist()):
        if pid < 0:
            continue
        parent = int(np.flatnonzero(node_ids == pid)[0])
        edge = (parent, child) if transpose else (child, parent)
        want.add(edge)
        if not directed:
            want.add(edge[::-1])

    assert got == want


@given(forests())
def test_adjacency_is_symmetric_when_undirected(forest):
    node_ids, parent_ids, weights = forest
    n = len(node_ids)
    csr_matrix = pytest.importorskip("scipy.sparse").csr_matrix

    indptr, indices, data = fastcore.adjacency(
        node_ids, parent_ids, weights=weights, directed=False
    )
    dense = csr_matrix((data, indices, indptr), shape=(n, n)).toarray()

    assert np.array_equal(dense, dense.T)


@given(forests())
def test_adjacency_transpose_is_the_matrix_transpose(forest):
    node_ids, parent_ids, weights = forest
    n = len(node_ids)
    csr_matrix = pytest.importorskip("scipy.sparse").csr_matrix

    def dense(**kwargs):
        indptr, indices, data = fastcore.adjacency(
            node_ids, parent_ids, weights=weights, **kwargs
        )
        return csr_matrix((data, indices, indptr), shape=(n, n)).toarray()

    assert np.array_equal(dense(transpose=True), dense(transpose=False).T)


# ------------------------------------------------------------------ longest_path


@given(forests())
def test_longest_path_is_the_farthest_node_to_its_root(forest):
    node_ids, parent_ids, weights = forest

    path = fastcore.longest_path(node_ids, parent_ids, weights=weights)
    dists = dict(
        zip(
            node_ids.tolist(),
            fastcore.dist_to_root(node_ids, parent_ids, weights=weights).tolist(),
        )
    )

    assert np.isclose(dists[int(path[0])], max(dists.values()), rtol=1e-4, atol=1e-4)


@given(forests())
def test_longest_path_is_a_valid_parent_chain(forest):
    node_ids, parent_ids, _ = forest
    parents = parent_map(node_ids, parent_ids)

    path = fastcore.longest_path(node_ids, parent_ids, weights=None)

    for child, parent in zip(path[:-1], path[1:]):
        assert parents[int(child)] == int(parent)
    assert parents[int(path[-1])] == -1


@given(forests())
def test_longest_path_agrees_with_longest_paths_of_one(forest):
    node_ids, parent_ids, weights = forest

    single = fastcore.longest_path(node_ids, parent_ids, weights=weights)
    first = fastcore.longest_paths(node_ids, parent_ids, 1, weights=weights)[0]

    assert single.tolist() == first.tolist()


# ----------------------------------------------------------------- longest_paths


@given(forests(), st.integers(1, 6))
def test_longest_paths_are_disjoint(forest, n):
    node_ids, parent_ids, weights = forest

    paths = fastcore.longest_paths(node_ids, parent_ids, n, weights=weights)

    seen = set()
    for path in paths:
        nodes = set(path.tolist())
        assert not (seen & nodes)
        seen |= nodes
    assert len(paths) <= n


@given(forests(), st.integers(1, 6))
def test_longest_paths_cover_everything_when_exhausted(forest, n):
    """Asking for more paths than exist must consume the whole forest exactly once."""
    node_ids, parent_ids, weights = forest

    paths = fastcore.longest_paths(node_ids, parent_ids, len(node_ids), weights=weights)

    covered = sorted(nid for p in paths for nid in p.tolist())
    assert covered == sorted(node_ids.tolist())


@given(forests())
def test_longest_paths_lengths_are_non_increasing(forest):
    """Each path is the longest of what remains, so lengths cannot go back up.

    A path's cable is the distance from its start to its (live) root, and peeling
    nodes away can only shorten such a distance - a severed branch gets a nearer
    root, never a farther one. So the maximum over a smaller remainder is bounded
    by the maximum over a larger one.
    """
    node_ids, parent_ids, weights = forest
    edge = dict(zip(node_ids.tolist(), weights.tolist()))

    paths = fastcore.longest_paths(node_ids, parent_ids, 4, weights=weights)
    # A path's own cable: every node's edge except the terminal (root-side) one,
    # whose edge leaves the path.
    lengths = [sum(edge[n] for n in p.tolist()[:-1]) for p in paths]

    for earlier, later in zip(lengths, lengths[1:]):
        # float32 accumulation, so allow a relative slack rather than exact >=.
        assert later <= earlier + 1e-4 * max(1.0, abs(earlier))


# ------------------------------------------------------------------- betweenness


@given(forests(), st.booleans())
def test_betweenness_matches_brute_force(forest, directed):
    """Against the definition: count, for every pair, the nodes strictly between."""
    node_ids, parent_ids, _ = forest
    parents = parent_map(node_ids, parent_ids)

    want = {nid: 0 for nid in node_ids.tolist()}
    ids = node_ids.tolist()
    for i, s in enumerate(ids):
        for t in ids[i + 1 :] if not directed else ids:
            if s == t:
                continue
            up_s, up_t = ancestors(parents, s), ancestors(parents, t)
            if directed:
                # A path exists only from a node up to one of its ancestors.
                if t not in up_s:
                    continue
                between = up_s[1 : up_s.index(t)]
            else:
                # Meet at the lowest common ancestor, if they share one at all.
                common = set(up_s) & set(up_t)
                if not common:
                    continue
                lca = next(a for a in up_s if a in common)
                between = up_s[1 : up_s.index(lca)] + [lca] + up_t[1 : up_t.index(lca)]
                between = [b for b in between if b != s and b != t]
            for b in between:
                want[b] += 1

    got = fastcore.betweenness(node_ids, parent_ids, directed=directed)
    assert got.tolist() == [want[nid] for nid in ids]


@given(forests())
def test_betweenness_leafs_and_roots_are_zero(forest):
    node_ids, parent_ids, _ = forest
    types = fastcore.classify_nodes(node_ids, parent_ids)

    bc = fastcore.betweenness(node_ids, parent_ids, directed=True)

    assert (bc[types == 1] == 0).all()
    assert (bc[types == 0] == 0).all()


# -------------------------------------------------------------- descendant_counts


@given(forests())
def test_descendant_counts_matches_descendants(forest):
    node_ids, parent_ids, _ = forest

    counts = fastcore.descendant_counts(node_ids, parent_ids)
    subtrees = fastcore.descendants(node_ids, parent_ids, node_ids)

    assert counts.tolist() == [len(s) - 1 for s in subtrees]


@given(forests(), st.data())
def test_descendant_counts_targets_are_a_subset_of_the_total(forest, data):
    node_ids, parent_ids, _ = forest
    k = data.draw(st.integers(0, len(node_ids)))
    targets = data.draw(
        st.lists(
            st.sampled_from(node_ids.tolist()), min_size=k, max_size=k, unique=True
        )
    )

    total = fastcore.descendant_counts(node_ids, parent_ids)
    subset = fastcore.descendant_counts(node_ids, parent_ids, targets=targets)

    assert (subset <= total).all()
    assert subset.sum() <= total.sum()


@given(forests())
def test_descendant_counts_of_all_targets_equals_the_total(forest):
    node_ids, parent_ids, _ = forest

    total = fastcore.descendant_counts(node_ids, parent_ids)
    explicit = fastcore.descendant_counts(node_ids, parent_ids, targets=node_ids)

    assert total.tolist() == explicit.tolist()


def test_descendant_counts_accepts_no_targets():
    """Counting nothing is a legitimate request, including with uint64 IDs.

    `_ids_to_indices` used to compute `max()` over the target array whenever the
    dtypes differed, which raises on an empty one - and uint64 node IDs alongside
    an int64 target array is navis' normal convention, so this was reachable.
    """
    node_ids = np.array([10, 20, 30], dtype=np.uint64)
    parent_ids = np.array([-1, 10, 20], dtype=np.int64)

    counts = fastcore.descendant_counts(node_ids, parent_ids, targets=[])

    assert counts.tolist() == [0, 0, 0]


# ------------------------------------------------------------------- has_cycles


@given(forests())
def test_has_cycles_is_false_for_a_forest(forest):
    """The generator cannot produce a cycle, so this is the false-positive half."""
    node_ids, parent_ids, _ = forest

    assert not fastcore.has_cycles(node_ids, parent_ids)


@given(forest_and_node(), st.data())
def test_has_cycles_detects_an_edge_closing_a_loop(forest_node, data):
    """Point a node at something below it and the walk up can no longer terminate.

    Every cycle reachable from a valid forest is exactly this edit - including the
    degenerate one where a node becomes its own parent, since `descendants`
    includes its source.
    """
    (node_ids, parent_ids, _), node = forest_node
    below = fastcore.descendants(node_ids, parent_ids, [node])[0]

    cyclic = parent_ids.copy()
    cyclic[node_ids == node] = data.draw(st.sampled_from(below.tolist()))

    assert fastcore.has_cycles(node_ids, cyclic)


@pytest.mark.parametrize("func", [fastcore.geodesic_nearest, fastcore.geodesic_farthest])
def test_geodesic_nearest_farthest_sentinel_survives_uint64_ids(func):
    """A source with no reachable target must report -1, not 2**64 - 1.

    Both functions built their output with `np.full(..., -1, dtype=node_ids.dtype)`,
    which wraps `-1` around to `18446744073709551615` on a uint64 ID column - the
    exact case `_indices_to_ids_sentinel` exists to handle. uint64 IDs are navis'
    normal convention, and unreachable sources are routine as soon as the skeleton
    has more than one component.
    """
    # Two disjoint components, 10 <- 11 and 12 <- 13; only 12 is a target.
    node_ids = np.array([10, 11, 12, 13], dtype=np.uint64)
    parent_ids = np.array([-1, 10, -1, 12], dtype=np.int64)

    dist, found = func(node_ids, parent_ids, sources=[10, 11, 13], targets=[12])

    assert dist.tolist() == [-1, -1, 1]
    assert found.tolist() == [-1, -1, 12]


def test_heal_skeleton_root_sentinel_survives_uint64_ids():
    """The healed skeleton's root parent must be -1, for the same reason."""
    node_ids = np.array([10, 11, 12], dtype=np.uint64)
    parent_ids = np.array([-1, 10, -1], dtype=np.int64)
    coords = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float64)

    healed = fastcore.heal_skeleton(node_ids, parent_ids, coords)

    assert (healed == -1).sum() == 1  # a single root, and it reads as -1
    assert healed.tolist() == [-1, 10, 11]


# ------------------------------------------------------------------ error contracts
#
# These live here rather than in the parity suite: they are about the Python wrapper's
# contract, have nothing to do with igraph, and would be skipped on any machine without
# it if they sat in that file.


def test_descendants_unknown_source_raises():
    node_ids = np.array([1, 2, 3], dtype=np.int64)
    parent_ids = np.array([-1, 1, 2], dtype=np.int64)

    with pytest.raises(ValueError, match="not found"):
        fastcore.descendants(node_ids, parent_ids, [999])


def test_contract_nodes_onto_a_descendant_raises():
    """Collapsing a node onto its own descendant closes a loop - refuse it."""
    node_ids = np.array([0, 1, 2], dtype=np.int64)
    parent_ids = np.array([-1, 0, 1], dtype=np.int64)
    mapping = np.array([2, 1, 2], dtype=np.int64)  # node 0 -> its grandchild

    with pytest.raises(ValueError, match="cycle"):
        fastcore.contract_nodes(node_ids, parent_ids, mapping)


# ---------------------------------------------------------------- mesh simplification
#
# The one place a hand-transcribed port of 500 lines of C++ index arithmetic is most
# likely to be subtly wrong, and the one place a fixed fixture set will not find it.
# Deliberately generated as *triangle soup* rather than as a valid surface: that is the
# input class this function actually meets.


@st.composite
def triangle_soups(draw, max_vertices=25, max_faces=40):
    """An arbitrary triangle soup — emphatically not a manifold.

    Coordinates come off a coarse integer lattice so duplicate vertices, collinear
    triples and zero-area faces all arise naturally, and faces are drawn independently
    so non-manifold edges and bowtie vertices do too. Meshes out of marching cubes look
    like this, which is why every halfedge-based simplifier was ruled out.
    """
    n_verts = draw(st.integers(min_value=0, max_value=max_vertices))
    coord = st.integers(min_value=-3, max_value=3).map(float)
    vertices = np.array(
        [
            [draw(coord), draw(coord), draw(coord)]
            for _ in range(n_verts)
        ],
        dtype=np.float64,
    ).reshape(n_verts, 3)

    n_faces = draw(st.integers(min_value=0, max_value=max_faces)) if n_verts else 0
    vertex = st.integers(min_value=0, max_value=max(n_verts - 1, 0))
    faces = np.array(
        [[draw(vertex), draw(vertex), draw(vertex)] for _ in range(n_faces)],
        dtype=np.uint32,
    ).reshape(n_faces, 3)

    return faces, vertices


@given(triangle_soups(), st.floats(min_value=0.01, max_value=1.0))
def test_simplify_invariants_hold_on_arbitrary_soup(mesh, ratio):
    faces, vertices = mesh
    out = fastcore.simplify_mesh(faces, vertices, ratio=ratio)
    check_simplify_invariants(out, faces, vertices)


@given(triangle_soups())
def test_lossless_invariants_hold_on_arbitrary_soup(mesh):
    faces, vertices = mesh
    out = fastcore.simplify_mesh_lossless(faces, vertices)
    check_simplify_invariants(out, faces, vertices)


@given(triangle_soups(), st.floats(min_value=0.01, max_value=1.0))
def test_simplify_is_deterministic(mesh, ratio):
    faces, vertices = mesh
    a = fastcore.simplify_mesh(faces, vertices, ratio=ratio)
    b = fastcore.simplify_mesh(faces, vertices, ratio=ratio)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)


@given(triangle_soups(), st.floats(min_value=0.01, max_value=1.0), st.integers(0, 2**32 - 1))
def test_locked_vertices_are_never_moved(mesh, ratio, seed):
    """The guarantee pinning exists for, over arbitrary input and arbitrary masks.

    Stated as "never merged into anything, never moved" rather than "always
    survives", because the two come apart on degenerate input: a locked vertex is
    dropped by the final compaction if *every* face touching it was deleted, which a
    mesh of zero-area faces manages. That is the `-1` case and it is honest — there
    is no surface left for the vertex to sit on. What must never happen is a locked
    vertex coming back at a position that is not the one it went in at.
    """
    faces, vertices = mesh
    rng = np.random.default_rng(seed)
    lock = rng.random(len(vertices)) < 0.3

    v, f, vmap = fastcore.simplify_mesh(faces, vertices, ratio=ratio, lock=lock)
    check_simplify_invariants((v, f, vmap), faces, vertices)

    for i in np.flatnonzero(lock):
        if vmap[i] < 0:
            continue
        # Bitwise: a locked position is never recomputed.
        np.testing.assert_array_equal(v[vmap[i]], vertices[i])

    # And no two locked vertices are ever merged into one another: each that
    # survives owns its output slot outright.
    survivors = vmap[lock & (vmap >= 0)]
    assert len(set(survivors.tolist())) == len(survivors)


# ---------------------------------------------------------------- downsampling


@st.composite
def forest_and_coords(draw):
    """A forest plus a coordinate for every node.

    Coordinates are drawn on a coarse grid rather than as arbitrary floats. Two
    reasons: exact ties and exactly-collinear runs then actually occur, which is
    where the geometric methods have to make a *choice* and where an
    order-dependent one would show; and no test below has to reason about what
    happens when a distance is 1e300.
    """
    forest = draw(forests())
    n = len(forest[0])
    coords = np.array(
        draw(
            st.lists(
                st.tuples(*(st.integers(-20, 20),) * 3), min_size=n, max_size=n
            )
        ),
        dtype=np.float64,
    )
    return forest, coords


#: The three methods that drop nodes, as callables over `(forest, coords)`.
DROP_METHODS = [
    lambda ids, parents, xyz, w: fastcore.downsample_skeleton(
        ids, parents, 3, weights=w
    ),
    lambda ids, parents, xyz, w: fastcore.simplify_rdp(ids, parents, xyz, 2.0, weights=w),
    lambda ids, parents, xyz, w: fastcore.simplify_vw(ids, parents, xyz, 5.0, weights=w),
]


@given(forest_and_coords(), st.integers(0, len(DROP_METHODS) - 1))
def test_dropping_preserves_topology_and_cable(forest_coords, which):
    """The invariant all three share: the tree that comes out is the tree that went
    in, at a different sampling density, with the same total cable length."""
    (node_ids, parent_ids, weights), coords = forest_coords

    ids, parents, new_weights = DROP_METHODS[which](
        node_ids, parent_ids, coords, weights
    )

    # The same checker the example-based suite runs, over arbitrary forests.
    check_dropping_invariants(
        (node_ids, parent_ids),
        (ids, parents),
        weights=weights,
        new_weights=new_weights,
    )


@given(forest_and_coords())
def test_resample_is_a_forest_with_the_same_shape(forest_coords):
    (node_ids, parent_ids, _), coords = forest_coords
    spacing = 3.0

    ids, parents, xyz, source, alpha = fastcore.resample_skeleton(
        node_ids, parent_ids, coords, spacing
    )

    check_is_forest(ids, parents)
    # Resampling mints new IDs, so only the counts per class can be compared.
    check_topology_preserved(
        (node_ids, parent_ids), (ids, parents), same_nodes=False
    )

    # The documented interpolation reproduces the coordinates the function chose, so
    # a caller interpolating a radius the same way gets something consistent.
    want = (
        coords[source[:, 0]] * (1 - alpha)[:, None]
        + coords[source[:, 1]] * alpha[:, None]
    )
    np.testing.assert_allclose(xyz, want, atol=1e-9)

    # No edge longer than the even-division rule allows.
    lengths = fastcore.parent_dist(ids, parents, xyz, root_dist=0)
    assert (lengths[parents >= 0] <= spacing * 1.5 + 1e-6).all()


@given(forest_and_coords(), st.booleans())
def test_smoothing_moves_only_slab_nodes(forest_coords, gaussian):
    (node_ids, parent_ids, _), coords = forest_coords

    if gaussian:
        out = fastcore.smooth_skeleton_gaussian(node_ids, parent_ids, coords, 3.0)
    else:
        out = fastcore.smooth_skeleton(node_ids, parent_ids, coords, window=5)

    assert out.shape == coords.shape

    # Roots, branch points and leafs are pinned, bitwise.
    codes = fastcore.classify_nodes(node_ids, parent_ids)
    np.testing.assert_array_equal(out[codes != 3], coords[codes != 3])
    assert np.isfinite(out).all()
