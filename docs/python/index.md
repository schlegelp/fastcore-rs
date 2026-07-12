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

## Available functions

Operations on [rooted trees](../concepts/trees.md):

- [`geodesic_matrix`](geodesic.md#navis_fastcore.geodesic_matrix): geodesic ("along-the-arbor") distances, either all-by-all or between specific sources and targets
- [`geodesic_pairs`](geodesic.md#navis_fastcore.geodesic_pairs): geodesic distances between given pairs of nodes
- [`geodesic_nearest`](geodesic.md#navis_fastcore.geodesic_nearest): distance to the nearest target for each source, without building the full matrix
- [`geodesic_farthest`](geodesic.md#navis_fastcore.geodesic_farthest): distance to the farthest target for each source, without building the full matrix
- [`connected_components`](cc.md#navis_fastcore.connected_components): generate the connected components
- [`classify_nodes`](morphology.md#navis_fastcore.classify_nodes): classify nodes into roots, leaves, branch points and slabs
- [`synapse_flow_centrality`](morphology.md#navis_fastcore.synapse_flow_centrality): synapse flow centrality ([Schneider-Mizell, eLife, 2016](https://elifesciences.org/articles/12059))
- [`strahler_index`](morphology.md#navis_fastcore.strahler_index): calculate Strahler index
- [`prune_twigs`](morphology.md#navis_fastcore.prune_twigs): remove terminal twigs below a certain size
- [`break_segments`](segments.md#navis_fastcore.break_segments): break the neuron into the linear segments connecting leafs, branches and roots
- [`generate_segments`](segments.md#navis_fastcore.generate_segments): same as `break_segments` but maximizing segment lengths, i.e. the longest segment goes from the most distal leaf to the root and so on
- [`segment_coords`](segments.md#navis_fastcore.segment_coords): coordinates per linear segment (useful for plotting)

Repairing fragmented skeletons:

- [`heal_skeleton`](healing.md#navis_fastcore.heal_skeleton): reconnect the fragments of a broken skeleton
- [`stitch_fragments`](healing.md#navis_fastcore.stitch_fragments): find the minimal-length edges that reconnect fragments

Meshes:

- [`mesh_connected_components`](mesh.md#navis_fastcore.mesh_connected_components): connected components of a triangle mesh

[Neuron similarity](../concepts/nblast.md):

- [`nblast`](nblast.md#navis_fastcore.nblast.nblast) / [`nblast_allbyall`](nblast.md#navis_fastcore.nblast_allbyall): NBLAST (query-vs-target / all-by-all)
- [`nblast_smart`](nblast.md#navis_fastcore.nblast_smart): two-pass approximate NBLAST for large comparisons
- [`synblast`](nblast.md#navis_fastcore.synblast): synapse-based NBLAST

Interop:

- [`wrappers.csgraph`](wrappers.md): drop-in replacements for some `scipy.sparse.csgraph` routines
