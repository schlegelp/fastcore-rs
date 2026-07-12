# nat.fastcore

Re-implementation of `nat` core functions in Rust.

## Installation

### Users (precompiled binary, recommended)

Precompiled binaries for Windows and macOS are served from R-universe, so **no Rust
toolchain is required**:

```r
install.packages(
  "nat.fastcore",
  repos = c("https://schlegelp.r-universe.dev", "https://cloud.r-project.org")
)
```

On Linux this installs from source (compiles the bundled Rust), which needs the
[Rust toolchain](https://rustup.rs/).

### Users (from source)

1. Clone this repository
2. Make sure the [Rust toolchain](https://rustup.rs/) and the R `rextendr` & `devtools` packages are installed
3. In R run:
   ```r
   utils::install.packages("/path/to/repo/fastcore-rs/R/nat.fastcore/", type="source", repos=NULL)
   ```

### Developers

1. Clone this repository
2. Make sure the [Rust toolchain](https://rustup.rs/) and the R `rextendr` & `devtools` packages are installed
3. `cd` into this directory
4. In R run:
   ```r
   library(rextendr)
   library(devtools)
   rextendr::document()  # compiles Rust code and update the R wrappers
   devtools::load_all(".")
   ```

## Examples

```r
> library(nat.fastcore)

> # Load a single skeleton
> s = read.neurons('test.swc')[[1]]

> # Generate node indices from node -> parent IDs
> parents = node_indices(s$d$PointNo, s$d$Parent)

> # Find distances to roots
> all_dists_to_root(parents, sources=NULL, weights=NULL)
[1]  0 47 48 49 50 51 ...

> # Calculate child -> parent distances
> weights = child_to_parent_dists(parents, s$d$X, s$d$Y, s$d$Z)

> # Generate all-by-all geodesic distance matrix
> dists = geodesic_distances(parents, sources=NULL, targets=NULL, weights=weights, directed=F)

> # Heal a fragmented skeleton (reconnect its disconnected fragments)
> healed = heal_skeleton(parents, s$d$X, s$d$Y, s$d$Z, method="ALL",
+                        max_dist=NULL, min_size=NULL, mask=NULL,
+                        radius=NULL, use_radius=FALSE)

> # ... optionally taking node radii into account, which prefers to connect
> # fragments of similar calibre (higher `use_radius` = more influence)
> healed = heal_skeleton(parents, s$d$X, s$d$Y, s$d$Z, method="ALL",
+                        max_dist=NULL, min_size=NULL, mask=NULL,
+                        radius=s$d$W, use_radius=TRUE)
```

Currently the following functions have been wrapped:

Skeleton / tree (DAG):

- `node_indices`: turn node and parent IDs into parent indices
- `geodesic_distances`: calculate geodesic distances between all/subsets of nodes
- `geodesic_pairs`: geodesic distances for explicit pairs of nodes
- `geodesic_nearest`: distance to the nearest target for each source (no full matrix)
- `geodesic_farthest`: distance to the farthest target for each source (no full matrix)
- `strahler_index`: calculate the Strahler index
- `connected_components`: extract connected components
- `classify_nodes`: classify nodes into roots, leaves, branch points and slabs
- `all_dists_to_root`: calculate distances from all/subsets of nodes to the root
- `dist_to_root`: distance from a single node to the root
- `prune_twigs`: prune twigs under a given size threshold
- `generate_segments` / `break_segments`: split the tree into linear segments
- `synapse_flow_centrality`: synapse flow centrality per node
- `has_cycles`: check whether a tree contains cycles
- `child_to_parent_dists`: helper to calculate child -> parent distances
- `heal_skeleton`: reconnect the fragments of a broken skeleton
- `stitch_fragments`: find the minimal-length edges that reconnect fragments
- `reroot_rewire`: regenerate a parent vector after adding edges

Mesh:

- `mesh_connected_components`: connected components of a triangle mesh

Neuron similarity (NBLAST / synNBLAST):

- `nblast` / `nblast_allbyall`: forward NBLAST (query-vs-target / all-by-all)
- `nblast_pairs`: forward NBLAST for a set of `(query, target)` index pairs
- `synblast` / `synblast_allbyall`: synapse-based NBLAST
- `smat_auto_limit`: the `limit_dist="auto"` value for a scoring matrix
