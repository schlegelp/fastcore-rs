# `fastcore` (Rust)

The core crate. Everything the [Python](../python/index.md) and
[R](../r/index.md) bindings do is implemented here; they are thin adapters that
translate each language's idioms into the core's index-based API.

## Modules

| Module | What it does |
|---|---|
| `dag` | Traversal and geometry on rooted trees: geodesic distances, linear segments, Strahler index, twig pruning, node classification, connected components, synapse flow centrality, cycle detection. |
| `topo` | Repairing fragmented skeletons: `stitch_fragments` finds the minimal-length edges that reconnect the pieces (optionally preferring fragments of similar calibre), `reroot_rewire` re-derives the parent vector afterwards. |
| `mesh` | `mesh_connected_components` for triangle meshes. |
| `nblast` | The NBLAST pipeline — `build_index`, `score_pair`, `nblast_query_target`, `nblast_allbyall`, `nblast_pairs`, plus the `Smat` scoring matrix and `Opts`. |
| `synblast` | Synapse-based NBLAST: `synblast_query_target`, `synblast_allbyall`. |

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
