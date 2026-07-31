# `navis-fastcore` (Python)

Python bindings for the [`fastcore`](../rust/index.md) Rust core, built with
[PyO3](https://pyo3.rs). Functions here are generalized and do **not** depend on
[`navis`](https://github.com/navis-org/navis) itself, so third-party libraries can
use them without that rather heavy dependency — `navis-fastcore`'s only runtime
requirement is `numpy`.

## Install

Pre-compiled wheels are on PyPI:

```bash
pip install navis-fastcore
```

See the [GitHub repo](https://github.com/schlegelp/fastcore-rs) if you want to
build from source.

## Usage

`navis` will use `fastcore` automatically where appropriate — you only need this
section if you want to call it directly.

Tree functions take `node_ids` and `parent_ids` arrays; the mapping from arbitrary
IDs onto the core's internal indices happens for you.

```python
import navis_fastcore as fastcore
import numpy as np

node_ids = np.array([1, 2, 3, 4, 5])
parent_ids = np.array([-1, 1, 2, 3, 1])  # -1 marks the root

# All-by-all geodesic distances
dists = fastcore.geodesic_matrix(node_ids, parent_ids)
```

### Integer return dtypes

The [`mesh`](mesh.md) functions and [`GeodesicGraph`](mesh.md#navis_fastcore.GeodesicGraph)
work in *index space* — nodes are `0..n_nodes`, because you handed them an edge
list rather than a set of IDs. Everything they return is one of four things, and
the dtype tells you which:

| dtype | What it holds | Examples |
|---|---|---|
| `uint32` | A **node id** — an index into the graph. | `connected_components_graph`, `mesh_connected_components`, `unique_edges`'s `edges`, `contract_vertices`, `parents_from_edges`'s `order`, `geodesic_path`, `GeodesicGraph.components` / `.parent_nodes` |
| `int32` | A node id that needs a **`-1` sentinel** for "none". | `parents_from_edges`'s `parents`, `geodesic_predecessors`, `geodesic_nearest_mesh` |
| `int32` | A dense **label** — a cluster or level-set id, not a node id. Contiguous from 0, negative where a node is excluded. | `geodesic_clusters`, `level_set_components`, `GeodesicGraph.clusters` |
| `int64` | A **position in an array you passed in** — not a node id. | `minimum_spanning_tree` (rows of `edges`), `geodesic_mst_*` (positions in `nodes`), `unique_edges`'s `index` / `inverse` |

The distinction that matters is the last one: `int64` means the values index
*your* array, so `nodes[out]` is the node-id form and `out` alone is not.
`uint32` caps a graph at 4.29 billion nodes, which is also what the core stores
internally — a wider node id would double the memory every search touches. If you
use a `uint32` result to index a numpy array repeatedly, convert it once with
`.astype(np.intp, copy=False)`: numpy widens `uint32` index arrays on every use.

Two families sit outside this rule.

**The tree functions** ([`dag`](geodesic.md), [`topo`](healing.md)) work in *ID
space*: you give them a `node_ids` array, so what they hand back are values from
it, in **your** dtype — `int64` in, `int64` out; `uint64` in, `uint64` out. Where
a `-1` sentinel is needed and your dtype is unsigned, the result is promoted to
`int64` so the sentinel fits, rather than wrapping to 18446744073709551615.

**Counts** are `int64` regardless
([`betweenness`](morphology.md#navis_fastcore.betweenness),
[`descendant_counts`](morphology.md#navis_fastcore.descendant_counts)): they grow
as the square of the component size and overflow 32 bits on a 100k-node skeleton.

## Available functions

Operations on [rooted trees](../concepts/trees.md):

- [`geodesic_matrix`](geodesic.md#navis_fastcore.geodesic_matrix): geodesic ("along-the-arbor") distances, either all-by-all or between specific sources and targets
- [`geodesic_pairs`](geodesic.md#navis_fastcore.geodesic_pairs): geodesic distances between given pairs of nodes
- [`geodesic_nearest`](geodesic.md#navis_fastcore.geodesic_nearest): distance to the nearest target for each source, without building the full matrix
- [`geodesic_farthest`](geodesic.md#navis_fastcore.geodesic_farthest): distance to the farthest target for each source, without building the full matrix
- [`dist_to_root`](geodesic.md#navis_fastcore.dist_to_root): distance from each node to its root
- [`connected_components`](cc.md#navis_fastcore.connected_components): generate the connected components
- [`classify_nodes`](morphology.md#navis_fastcore.classify_nodes): classify nodes into roots, leaves, branch points and slabs
- [`synapse_flow_centrality`](morphology.md#navis_fastcore.synapse_flow_centrality): synapse flow centrality ([Schneider-Mizell, eLife, 2016](https://elifesciences.org/articles/12059))
- [`strahler_index`](morphology.md#navis_fastcore.strahler_index): calculate Strahler index
- [`subtree_height`](morphology.md#navis_fastcore.subtree_height): distance from each node down to the farthest leaf below it
- [`prune_twigs`](morphology.md#navis_fastcore.prune_twigs): remove terminal twigs below a certain size
- [`break_segments`](segments.md#navis_fastcore.break_segments): break the neuron into the linear segments connecting leafs, branches and roots
- [`generate_segments`](segments.md#navis_fastcore.generate_segments): same as `break_segments` but maximizing segment lengths, i.e. the longest segment goes from the most distal leaf to the root and so on
- [`segment_coords`](segments.md#navis_fastcore.segment_coords): coordinates per linear segment (useful for plotting)
- [`descendants`](topology.md#navis_fastcore.descendants): the sub-tree distal to a node
- [`paths_to_root`](topology.md#navis_fastcore.paths_to_root): the node sequence from a node up to its root
- [`reroot`](topology.md#navis_fastcore.reroot): re-root the skeleton, reversing only the edges that have to move
- [`contract_nodes`](topology.md#navis_fastcore.contract_nodes): collapse groups of nodes onto a representative and rewire
- [`simplify_skeleton`](topology.md#navis_fastcore.simplify_skeleton): keep only roots, leafs and branch points, preserving cable length
- [`adjacency`](topology.md#navis_fastcore.adjacency): the skeleton's adjacency matrix as a CSR triple
- [`longest_path`](topology.md#navis_fastcore.longest_path) / [`longest_paths`](topology.md#navis_fastcore.longest_paths): the longest path from a node to its root, and the `n` longest taken in turn
- [`betweenness`](morphology.md#navis_fastcore.betweenness): betweenness centrality, in O(N) rather than Brandes' O(V*E)
- [`descendant_counts`](morphology.md#navis_fastcore.descendant_counts): how many nodes (or how many of a given set) lie below each node

Repairing fragmented skeletons:

- [`heal_skeleton`](healing.md#navis_fastcore.heal_skeleton): reconnect the fragments of a broken skeleton
- [`stitch_fragments`](healing.md#navis_fastcore.stitch_fragments): find the minimal-length edges that reconnect fragments

Meshes:

- [`mesh_connected_components`](mesh.md#navis_fastcore.mesh_connected_components): connected components of a triangle mesh
- [`geodesic_matrix_mesh`](mesh.md#navis_fastcore.geodesic_matrix_mesh): parallel geodesic distances across a mesh
- [`level_set_components`](mesh.md#navis_fastcore.level_set_components): components of every level set at once (wavefront rings)
- [`contract_vertices`](mesh.md#navis_fastcore.contract_vertices) / [`minimum_spanning_tree`](mesh.md#navis_fastcore.minimum_spanning_tree): collapse a graph onto new nodes, then span it
- [`parents_from_edges`](mesh.md#navis_fastcore.parents_from_edges): orient an edge list into a rooted forest — breaks cycles, and hands back the order that makes parents precede children
- [`bridges`](mesh.md#navis_fastcore.bridges): which edges may not be dropped without disconnecting the graph
- [`geodesic_mst_mesh`](mesh.md#navis_fastcore.geodesic_mst_mesh) / [`geodesic_mst_graph`](mesh.md#navis_fastcore.geodesic_mst_graph): span a *subset* of nodes by geodesic distance, without building the `k x k` matrix
- [`geodesic_path`](mesh.md#navis_fastcore.geodesic_path) / [`geodesic_predecessors`](mesh.md#navis_fastcore.geodesic_predecessors): the shortest *route*, not just its length
- [`geodesic_clusters`](mesh.md#navis_fastcore.geodesic_clusters): greedily partition a graph into clusters of bounded geodesic radius
- [`GeodesicGraph`](mesh.md#navis_fastcore.GeodesicGraph): a graph prepared once for many small geodesic queries — grow fixed-*size* connected regions, and place evenly-spread seeds by farthest-point sampling

[Neuron similarity](../concepts/nblast.md):

- [`dotprops`](nblast.md#navis_fastcore.dotprops) / [`Dotprop.from_points`](nblast.md#navis_fastcore.dotprops): tangent vectors and alpha from a bare point cloud
- [`nblast`](nblast.md#navis_fastcore.nblast.nblast) / [`nblast_allbyall`](nblast.md#navis_fastcore.nblast_allbyall): NBLAST (query-vs-target / all-by-all)
- [`nblast_smart`](nblast.md#navis_fastcore.nblast_smart): two-pass approximate NBLAST for large comparisons
- [`synblast`](nblast.md#navis_fastcore.synblast): synapse-based NBLAST

Interop:

- [`wrappers.csgraph`](wrappers.md): drop-in replacements for some `scipy.sparse.csgraph` routines
