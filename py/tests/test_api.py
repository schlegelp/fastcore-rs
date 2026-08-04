"""Shape of the public interface, as opposed to what any one function computes.

Two rules live here because they are cheap to state and easy to break silently:
what `import *` exports, and what dtype a function hands back for each *kind* of
integer it returns.
"""

import importlib
import pkgutil
import types

import numpy as np
import pytest

import navis_fastcore as fastcore


# ---------------------------------------------------------------------------
# What `import *` exports
# ---------------------------------------------------------------------------

# Not part of the flat namespace: an opt-in interop shim that needs scipy, which
# the package does not otherwise depend on. Imported by path, never star-imported.
NOT_EXPORTED = {"wrappers"}


def _submodules():
    """Every public submodule, discovered rather than listed.

    Discovery is the point: a hardcoded list here would be the same list
    `__init__.py` already has, so adding a submodule and forgetting to wire it in
    would leave both of them agreeing and wrong.
    """
    for info in pkgutil.iter_modules(fastcore.__path__):
        if info.name.startswith("_") or info.name in NOT_EXPORTED:
            continue
        yield importlib.import_module(f"navis_fastcore.{info.name}")


def test_all_covers_every_submodule():
    expected = {"__version__", "__version_vector__"}
    for mod in _submodules():
        assert hasattr(mod, "__all__"), f"{mod.__name__} has no `__all__`"
        expected |= set(mod.__all__)

    assert set(fastcore.__all__) == expected
    assert len(fastcore.__all__) == len(set(fastcore.__all__)), "duplicate entries"


def test_star_import_exports_no_submodules():
    """The interface is one flat namespace, so `import *` hands back callables only.

    This also proves every name in `__all__` resolves: CPython raises
    `AttributeError` from the star-import for any entry that does not.
    """
    ns = {}
    exec("from navis_fastcore import *", ns)
    assert sorted(k for k, v in ns.items() if isinstance(v, types.ModuleType)) == []
    assert ns["__version__"] == fastcore.__version__


# ---------------------------------------------------------------------------
# Integer return dtypes
# ---------------------------------------------------------------------------
#
# The rule and its rationale live in docs/python/index.md under "Integer return
# dtypes". This pins the `mesh` half of it — the index-space API, where a node is
# a position in `0..n_nodes` rather than an ID the caller chose.

FACES = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.uint32)
EDGES = np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.uint32)

_EDGES_U, _INDEX, _INVERSE = fastcore.unique_edges(
    FACES, return_index=True, return_inverse=True
)
_PARENTS, _ORDER = fastcore.parents_from_edges(EDGES, 4)
_MST_ROWS = fastcore.minimum_spanning_tree(EDGES, 4)
_MST_NODES = np.array([0, 2], dtype=np.uint32)
_MST_EDGES, _ = fastcore.geodesic_mst_graph(EDGES, 4, nodes=_MST_NODES)
_GRAPH = fastcore.GeodesicGraph(EDGES, 4)

# `ratio=1.0` is a no-op, which keeps this about the dtypes rather than about what
# the sweep decides to collapse on a four-vertex mesh.
_TETRA = np.eye(4, 3, dtype=np.float64)
_, _SIMP_F, _SIMP_MAP = fastcore.simplify_mesh(FACES, _TETRA, ratio=1.0)
_, _LOSSLESS_F, _LOSSLESS_MAP = fastcore.simplify_mesh_lossless(FACES, _TETRA)

# A node id — an index into the graph.
NODE_IDS = {
    "mesh_connected_components": fastcore.mesh_connected_components(FACES, 4),
    "connected_components_graph": fastcore.connected_components_graph(EDGES, 4),
    "unique_edges[edges]": _EDGES_U,
    "contract_vertices": fastcore.contract_vertices(EDGES, [0, 0, 1, 1]),
    "parents_from_edges[order]": _ORDER,
    "geodesic_path": fastcore.geodesic_path(EDGES, 4, 0, [3])[0],
    "GeodesicGraph.components": _GRAPH.components(),
    "GeodesicGraph.parent_nodes": _GRAPH.subset([0, 1]).parent_nodes,
    "simplify_mesh[faces]": _SIMP_F,
    "simplify_mesh_lossless[faces]": _LOSSLESS_F,
}

# A node id that needs a `-1` sentinel for "none".
SENTINELLED = {
    "parents_from_edges[parents]": _PARENTS,
    "geodesic_predecessors": fastcore.geodesic_predecessors(EDGES, 4, sources=[0])[1],
    "geodesic_nearest_mesh": fastcore.geodesic_nearest_mesh(
        FACES, n_vertices=4, targets=[3]
    )[1],
    # An id into the mesh the function itself returns, with -1 for "did not survive".
    "simplify_mesh[vertex_map]": _SIMP_MAP,
    "simplify_mesh_lossless[vertex_map]": _LOSSLESS_MAP,
}

# A dense label — a cluster or level-set id, not a node id.
LABELS = {
    "geodesic_clusters": fastcore.geodesic_clusters(EDGES, 4, 1.0)[0],
    "level_set_components": fastcore.level_set_components(
        EDGES, 4, np.array([0, 1, 1, 2], dtype=np.int64)
    )[0],
    "GeodesicGraph.clusters": _GRAPH.clusters(1.0)[0],
}

# A position in an array the caller passed in — not a node id.
POSITIONS = {
    "unique_edges[index]": _INDEX,
    "unique_edges[inverse]": _INVERSE,
    "minimum_spanning_tree": _MST_ROWS,
    "geodesic_mst_graph[edges]": _MST_EDGES,
}


@pytest.mark.parametrize(
    "dtype, cases",
    [
        (np.uint32, NODE_IDS),
        (np.int32, SENTINELLED),
        (np.int32, LABELS),
        (np.int64, POSITIONS),
    ],
)
def test_integer_returns_follow_the_dtype_rule(dtype, cases):
    assert {k: v.dtype for k, v in cases.items() if v.dtype != dtype} == {}


def test_every_integer_returning_mesh_function_is_classified():
    """A new `mesh` function has to be filed under one of the four kinds.

    Without this the rule holds only for what someone remembered to type, which is
    how the dtypes drifted apart in the first place.
    """
    classified = {
        name.split("[")[0].split(".")[-1]
        for name in (*NODE_IDS, *SENTINELLED, *LABELS, *POSITIONS)
    }
    # Returns no integers at all: a bool mask, float distances, or a graph object.
    no_integers = {
        "bridges",
        "geodesic_matrix_mesh",
        "geodesic_matrix_graph",
        "geodesic_farthest_mesh",
        "geodesic_mst_mesh",
        "GeodesicGraph",
    }
    from navis_fastcore import mesh

    assert set(mesh.__all__) - classified - no_integers == set()


def test_positions_index_the_array_they_name():
    """Not just the dtype — the values are offsets, and round-trip as offsets."""
    # Rows are positions in `nodes`, so every entry is < len(nodes) even though the
    # node ids themselves go up to 3.
    assert _MST_EDGES.max() < len(_MST_NODES)
    np.testing.assert_array_equal(_MST_NODES[_MST_EDGES], np.array([[0, 2]]))

    assert _MST_ROWS.max() < len(EDGES)  # rows of `edges`, not nodes


def test_labels_are_dense_and_are_not_node_ids():
    """The distinction the `int32` label row makes: contiguous from 0, not indices."""
    labels, n_clusters = fastcore.geodesic_clusters(EDGES, 4, 1.0)
    assert labels.min() == 0
    assert sorted(set(labels.tolist())) == list(range(n_clusters))
