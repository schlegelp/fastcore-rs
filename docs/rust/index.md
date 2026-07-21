# `fastcore` (Rust)

The core crate. Everything the [Python](../python/index.md) and
[R](../r/index.md) bindings do is implemented here; they are thin adapters that
translate each language's idioms into the core's index-based API.

## Modules

| Module | What it does |
|---|---|
| `dag` | Traversal and geometry on rooted trees: geodesic distances, linear segments, Strahler index, twig pruning, node classification, connected components, synapse flow centrality, cycle detection. |
| `topo` | Repairing fragmented skeletons: `stitch_fragments` finds the minimal-length edges that reconnect the pieces (optionally preferring fragments of similar calibre), `reroot_rewire` re-derives the parent vector afterwards. |
| `mesh` | Triangle meshes as vertex graphs: `mesh_connected_components` and `unique_edges`, a parallel Dijkstra/BFS behind `geodesic_matrix_mesh`, `geodesic_nearest_mesh`, `geodesic_farthest_mesh` and — for arbitrary (cyclic) graphs given as an edge list — `geodesic_matrix_graph`, plus the traversal primitives that go with them: `connected_components_graph`, `level_set_components`, `contract_vertices` and `minimum_spanning_tree`. |
| `nblast` | The NBLAST pipeline — `build_index`, `score_pair`, `nblast_query_target`, `nblast_allbyall`, `nblast_pairs`, plus the `Smat` scoring matrix and `Opts`. |
| `nblast_knn` | Each neuron's `k` nearest neighbours without the `n x n` matrix — `nblast_knn`, `nblast_knn_query_target`, plus `build_signatures` / `candidate_pairs` and the `Symmetry` combine. |
| `synblast` | Synapse-based NBLAST: `synblast_query_target`, `synblast_allbyall`. |
| `matches` | Pulling the top matches back out of a score matrix — `top_matches` (top-N), `matches_above` (absolute threshold or a percentage band around each group's best), `count_matches` — without copying or transposing a matrix that may be tens of GB. |
| `cmtk` | CMTK spatial transforms: `Registration::from_path` reads a `*.list` registration (12-DOF affine + cubic B-spline warp), `transform_points` / `inverse_transform_points` apply it. Matches CMTK's `streamxform` to ~4e-7 without needing CMTK installed. |
| `elastix` | Elastix spatial transforms: `ElastixTransform::from_path` reads a `TransformParameters` file *and* the initial-transform chain hanging off it (affine / Euler / similarity / translation, plus cubic B-spline warps), `transform_points` / `inverse_transform_points` apply it. Matches `transformix` to 5e-7 without needing Elastix installed — and adds an inverse, which Elastix itself cannot compute. `probe_invertible` answers whether a file inverts without reading its coefficients, ~20x faster than a full parse. |
| `tps` | Thin-plate spline warps from landmark pairs: `TpsTransform::fit` solves for the coefficients (blocked LU, no BLAS dependency), `xform` applies them. The `n_points x n_landmarks` distance matrix is fused into the accumulation rather than built, so peak memory is the output and the landmark count is unbounded. |
| `mls` | Moving least squares warps (Schaefer et al. 2006, affine flavour): `MlsTransform::xform` solves a locally weighted affine per point. No fit step. Same fusion as `tps`, which is what makes landmark counts the reference implementation cannot allocate for tractable here. |

See [Concepts › Rooted trees](../concepts/trees.md) and
[Concepts › NBLAST](../concepts/nblast.md) for the ideas behind them, and the
[capability matrix](../index.md#whats-available-where) for how each module maps
onto the Python and R functions.

## Using the crate

`fastcore` is **not published to crates.io**, so depend on it via git:

```toml
[dependencies]
fastcore = { git = "https://github.com/schlegelp/fastcore-rs" }
```

## API reference

There is no docs.rs page (see above). Build the reference locally:

```sh
git clone https://github.com/schlegelp/fastcore-rs
cd fastcore-rs
cargo doc -p fastcore --open
```

## Shape of the API

The core is index-based and [`ndarray`](https://docs.rs/ndarray)-typed. Where the
Python bindings accept `node_ids` and `parent_ids` with arbitrary IDs and map them
for you, `fastcore` expects the mapping to have happened already:

- A tree is an `ArrayView1<i32>` of **parent indices**, with roots encoded as
  **negative** values.
- Edge weights, coordinates and masks are passed as separate arrays.
- NBLAST takes prepared point clouds (`build_index`) rather than dotprop objects.

That mapping step is exactly what `navis_fastcore`'s internal `_ids_to_indices`
and `nat.fastcore`'s public `node_indices` exist to do.

## Parallelism

`dag` and the NBLAST modules parallelise with [`rayon`](https://docs.rs/rayon).
`Opts::threads` caps the pool for NBLAST; the Python bindings surface this as
`n_cores` and release the GIL around the call.
