# `fastcore-rs`

Fast implementations of the algorithms that [`navis`](https://github.com/navis-org/navis)
and [`nat`](https://natverse.org/) need most: traversal and geometry on
[rooted trees](concepts/trees.md) (i.e. neuron skeletons), plus
[NBLAST](concepts/nblast.md) neuron similarity.

The project is a single Rust core with bindings on top:

| Package | Language | |
|---|---|---|
| `fastcore` | Rust | [the core crate](rust/index.md) |
| `navis-fastcore` | Python (PyO3) | [docs](python/index.md) |
| `nat.fastcore` | R (extendr) | [docs](r/index.md) |

The bindings are deliberately generalized: they know nothing about `navis` or
`nat` themselves, so third-party libraries can use them without those rather
heavy dependencies. `navis-fastcore` only requires `numpy`.

## Install

=== "Python"

    ```bash
    pip install navis-fastcore
    ```

    Pre-compiled wheels are on PyPI. `navis` will use `fastcore` automatically
    where appropriate; see the [Python docs](python/index.md) if you want to call
    it directly.

=== "R"

    ```r
    install.packages(
      "nat.fastcore",
      repos = c("https://schlegelp.r-universe.dev", "https://cloud.r-project.org")
    )
    ```

    Pre-compiled binaries for Windows and macOS come from R-universe, so no Rust
    toolchain is needed. On Linux this builds from source. See the
    [R docs](r/index.md).

=== "Rust"

    ```toml
    [dependencies]
    fastcore = { git = "https://github.com/schlegelp/fastcore-rs" }
    ```

    The crate is not published to crates.io — depend on it via git. See the
    [Rust docs](rust/index.md).

## What's available where

Every capability lives in the Rust core; the bindings differ only in what they
surface and how they spell it. `—` means that surface doesn't expose it directly.

### Rooted trees (skeletons)

| Capability | Rust (`fastcore`) | Python (`navis-fastcore`) | R (`nat.fastcore`) |
|---|---|---|---|
| Geodesic distance matrix | `dag::geodesic_distances_all_by_all`, `dag::geodesic_distances_partial` | `geodesic_matrix` | `geodesic_distances` |
| Geodesic distances for given pairs | `dag::geodesic_pairs` | `geodesic_pairs` | `geodesic_pairs` |
| Distance to the nearest target | `dag::geodesic_nearest` | `geodesic_nearest` | `geodesic_nearest` |
| Distance to the farthest target | `dag::geodesic_farthest` | `geodesic_farthest` | `geodesic_farthest` |
| Distances to the root | `dag::all_dists_to_root`, `dag::dist_to_root` | `dist_to_root` | `all_dists_to_root`, `dist_to_root` |
| Subtree height (distance down to the farthest leaf) | `dag::subtree_height` | `subtree_height` | `subtree_height` |
| Connected components | `dag::connected_components` | `connected_components` | `connected_components` |
| Break into linear segments | `dag::generate_segments`, `dag::break_segments` | `generate_segments`, `break_segments`, `segment_coords` | `generate_segments`, `break_segments` |
| Strahler index | `dag::strahler_index` | `strahler_index` | `strahler_index` |
| Prune twigs | `dag::prune_twigs` | `prune_twigs` | `prune_twigs`[^1] |
| Classify nodes (root/leaf/branch/slab) | `dag::classify_nodes` | `classify_nodes` | `classify_nodes` |
| Synapse flow centrality | `dag::synapse_flow_centrality` | `synapse_flow_centrality` | `synapse_flow_centrality` |
| Cycle detection | `dag::has_cycles` | — | `has_cycles` |
| Node ID → index mapping | — | — (done internally) | `node_indices` |
| Child → parent distances | — | — (see `parent_dist`) | `child_to_parent_dists` |

[^1]: R's `prune_twigs` has no `mask` argument — extendr cannot take a `Vec<bool>`.

### Healing fragmented skeletons

| Capability | Rust (`fastcore`) | Python (`navis-fastcore`) | R (`nat.fastcore`) |
|---|---|---|---|
| Reconnect a broken skeleton | (composed in the bindings) | `heal_skeleton` | `heal_skeleton` |
| Minimal reconnecting edges | `topo::stitch_fragments` | `stitch_fragments` | `stitch_fragments` |
| Regenerate the parent vector | `topo::reroot_rewire` | — (used by `heal_skeleton`) | `reroot_rewire` |

### Meshes

| Capability | Rust (`fastcore`) | Python (`navis-fastcore`) | R (`nat.fastcore`) |
|---|---|---|---|
| Connected components of a triangle mesh | `mesh::mesh_connected_components` | `mesh_connected_components` | `mesh_connected_components` |
| Geodesic distances across a mesh | `mesh::geodesic_matrix_mesh` | `geodesic_matrix_mesh` | `geodesic_matrix_mesh` |
| Nearest / farthest target vertex | `mesh::geodesic_nearest_mesh`, `mesh::geodesic_farthest_mesh` | `geodesic_nearest_mesh`, `geodesic_farthest_mesh` | `geodesic_nearest_mesh`, `geodesic_farthest_mesh` |
| Geodesic distances across any graph | `mesh::geodesic_matrix_graph` | `geodesic_matrix_graph` | `geodesic_matrix_graph` |
| Unique undirected edges of a mesh | `mesh::unique_edges` | `unique_edges` | — |
| Connected components of any graph | `mesh::connected_components_graph` | `connected_components_graph` | — |
| Components of every level set at once | `mesh::level_set_components` | `level_set_components` | — |
| Contract nodes and simplify the edge list | `mesh::contract_vertices` | `contract_vertices` | — |
| Minimum / maximum spanning forest | `mesh::minimum_spanning_tree` | `minimum_spanning_tree` | — |
| Shortest paths with the route, not just the length | `mesh::geodesic_predecessors_graph`, `mesh::geodesic_path_graph` | `geodesic_predecessors`, `geodesic_path` | — |
| Greedy clustering by geodesic radius | `mesh::geodesic_clusters` | `geodesic_clusters` | — |

### Neuron similarity

| Capability | Rust (`fastcore`) | Python (`navis-fastcore`) | R (`nat.fastcore`) |
|---|---|---|---|
| Tangent vectors + alpha from a point cloud | `points::dotprops` | `dotprops`, `Dotprop.from_points` | — |
| NBLAST, query vs target | `nblast::nblast_query_target` | `nblast` | `nblast` |
| NBLAST, all-by-all | `nblast::nblast_allbyall` | `nblast_allbyall` | `nblast_allbyall` |
| NBLAST for explicit index pairs | `nblast::nblast_pairs` | — | `nblast_pairs` |
| k nearest neighbours, no score matrix | `nblast_knn::nblast_knn`, `nblast_knn::nblast_knn_query_target` | `nblast_knn` | `nblast_knn` |
| Two-pass approximate NBLAST | — | `nblast_smart` | — |
| synNBLAST (synapse-based) | `synblast::synblast_query_target`, `synblast::synblast_allbyall` | `synblast` | `synblast`, `synblast_allbyall` |
| `limit_dist` heuristic for a scoring matrix | `nblast::Smat` | — (via `limit_dist="auto"`) | `smat_auto_limit` |

### Match extraction

| Capability | Rust (`fastcore`) | Python (`navis-fastcore`) | R (`nat.fastcore`) |
|---|---|---|---|
| Top-N matches per query | `matches::top_matches` | `top_matches` | — |
| Matches above a threshold / percentage band | `matches::matches_above` | `matches_above` | — |
| Count matches without allocating them | `matches::count_matches` | `count_matches` | — |

### Spatial transforms

| Capability | Rust (`fastcore`) | Python (`navis-fastcore`) | R (`nat.fastcore`) |
|---|---|---|---|
| Read a CMTK `*.list` registration | `cmtk::Registration::from_path`, `cmtk::Chain` | `CmtkRegistration`, `load_cmtk_registration` | `cmtk_read` |
| Apply it to points (forward / inverse) | `cmtk::transform_points`, `cmtk::inverse_transform_points` | `CmtkRegistration.xform`, `.xform_inv` | `cmtk_xform`, `cmtk_xform_inv` |
| CMTK registration properties | fields on `Registration` / `SplineWarp` | `.affine`, `.dims`, `.spacing` | `cmtk_affine`, `cmtk_domain`, `cmtk_dims`, `cmtk_spacing` |
| Read an Elastix `TransformParameters` file | `elastix::ElastixTransform::from_path`, `elastix::Chain` | `ElastixTransform`, `load_elastix_transform` | `elastix_read` |
| Apply it to points (forward / inverse) | `elastix::transform_points`, `elastix::inverse_transform_points` | `ElastixTransform.xform`, `.xform_inv` | `elastix_xform`, `elastix_xform_inv` |
| Elastix transform properties | fields on `Linear` / `BSpline` | `.affine`, `.kinds`, `.grid_size`, `.grid_spacing`, `.grid_origin` | `elastix_affine`, `elastix_kinds`, `elastix_grid_size`, `elastix_grid_spacing`, `elastix_grid_origin` |
| Is an Elastix file invertible? (without parsing it) | `elastix::probe_invertible` | `probe_elastix_invertible` | `elastix_probe_invertible` |
| Fit a thin-plate spline to landmarks | `tps::TpsTransform::fit`, `::from_coefs` | `TpsTransform`, `.from_coefs` | `tps_transform`, `tps_coefs` |
| Moving least squares from landmarks | `mls::MlsTransform::new` | `MlsTransform` | `mls_transform` |
| Apply a landmark transform to points | `TpsTransform::xform`, `MlsTransform::xform` | `.xform` | `tps_xform`, `mls_xform` |
| Its global affine | `.matrix_affine` | `.matrix_affine` | `tps_affine`, `mls_affine` |

Neither CMTK nor Elastix needs to be installed — see [CMTK transforms](python/cmtk.md) and
[Elastix transforms](python/elastix.md). The two differ in one important respect: outside the
region it was fitted on, CMTK **fails** a point (we return `NaN`) while Elastix returns it
**unchanged**. Each binding follows its own tool. Elastix additionally gains an **inverse**,
which Elastix itself cannot compute.

Both work the same way: a transform object holds only the *parse*, and **direction is chosen
per call** (`invert=`, or the `_inv` method), so one object serves every direction rather than
re-reading the file to walk it the other way.

When there is no registration at all — only matched landmarks — the thin-plate spline and
moving least squares transforms take over; see
[Landmark transforms](python/landmarks.md). Neither materialises the
`n_points x n_landmarks` matrix its reference implementation does, so there is no batch size
to tune, and moving least squares becomes usable at landmark counts (~3,400) that would
otherwise need tens of GB.

### Python only

`navis_fastcore.wrappers.csgraph` provides drop-in replacements for some
`scipy.sparse.csgraph` routines (`dijkstra`, `connected_components`) that are
faster when your graph happens to be a rooted tree — see
[Scipy wrappers](python/wrappers.md).

## A note on input conventions

The three surfaces take the same tree in different shapes:

- **Rust** is index-based: a parent-index vector in which roots are negative.
- **Python** takes `node_ids` and `parent_ids` arrays with arbitrary IDs and maps
  them to indices for you.
- **R** takes a precomputed `parents` index vector (build one with `node_indices`)
  plus separate `x`/`y`/`z` vectors.

So what reads `geodesic_matrix(node_ids, parent_ids)` in Python reads
`geodesic_distances(parents, ...)` in R, with `node_indices` called first.
