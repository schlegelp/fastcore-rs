# nat.fastcore

Re-implementation of `nat` core functions in Rust.

## Installation

### Users

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

```

Currently the following functions have been wrapped:

- `node_indices`: turn node and parent IDs into parent indices
- `geodesic_distances`: calculate geodesic distances between all/subsets of nodes
- `strahler_index`: calculate the Strahler index
- `connected_components`: extract connected components
- `all_dist_to_root`: calculate distances from all/subsets of nodes to the root
- `prune_twigs`: prune twigs under a given size threshold
- `child_to_parent_dists`: helper to calculate child -> parent distances
