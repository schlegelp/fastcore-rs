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
| Distances to the root | `dag::all_dists_to_root`, `dag::dist_to_root` | — (see `parent_dist`) | `all_dists_to_root`, `dist_to_root` |
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

### Neuron similarity

| Capability | Rust (`fastcore`) | Python (`navis-fastcore`) | R (`nat.fastcore`) |
|---|---|---|---|
| NBLAST, query vs target | `nblast::nblast_query_target` | `nblast` | `nblast` |
| NBLAST, all-by-all | `nblast::nblast_allbyall` | `nblast_allbyall` | `nblast_allbyall` |
| NBLAST for explicit index pairs | `nblast::nblast_pairs` | — | `nblast_pairs` |
| Two-pass approximate NBLAST | — | `nblast_smart` | — |
| synNBLAST (synapse-based) | `synblast::synblast_query_target`, `synblast::synblast_allbyall` | `synblast` | `synblast`, `synblast_allbyall` |
| `limit_dist` heuristic for a scoring matrix | `nblast::Smat` | — (via `limit_dist="auto"`) | `smat_auto_limit` |

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
