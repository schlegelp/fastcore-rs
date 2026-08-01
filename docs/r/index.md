# `nat.fastcore` (R)

R bindings for the [`fastcore`](../rust/index.md) Rust core, built with
[extendr](https://extendr.github.io/). Intended for use alongside
[`nat`](https://natverse.org/) and the rest of the natverse, though the functions
themselves are generic and don't depend on it.

## Install

Pre-compiled binaries for Windows and macOS are served from R-universe, so **no
Rust toolchain is required**:

```r
install.packages(
  "nat.fastcore",
  repos = c("https://schlegelp.r-universe.dev", "https://cloud.r-project.org")
)
```

On Linux this installs from source (compiling the bundled Rust), which needs the
[Rust toolchain](https://rustup.rs/).

??? note "Building from source"

    1. Clone [the repository](https://github.com/schlegelp/fastcore-rs)
    2. Make sure the [Rust toolchain](https://rustup.rs/) and the R `rextendr` &
       `devtools` packages are installed
    3. In R, run:
       ```r
       utils::install.packages("/path/to/fastcore-rs/R/nat.fastcore/", type = "source", repos = NULL)
       ```

    For development, `cd` into `R/nat.fastcore/` and run:
    ```r
    library(rextendr)
    library(devtools)
    rextendr::document()  # compiles the Rust code and updates the R wrappers
    devtools::load_all(".")
    ```

## Usage

Unlike the Python bindings, `nat.fastcore` works on an explicit **parent-index
vector** rather than node/parent IDs — build one with `node_indices` and pass it
to everything else.

```r
library(nat.fastcore)

# Load a single skeleton
s = read.neurons('test.swc')[[1]]

# Generate node indices from node -> parent IDs
parents = node_indices(s$d$PointNo, s$d$Parent)

# Find distances to roots
all_dists_to_root(parents, sources=NULL, weights=NULL)
#> [1]  0 47 48 49 50 51 ...

# Calculate child -> parent distances
weights = child_to_parent_dists(parents, s$d$X, s$d$Y, s$d$Z)

# Generate all-by-all geodesic distance matrix
dists = geodesic_distances(parents, sources=NULL, targets=NULL, weights=weights, directed=F)
```

Healing a fragmented skeleton (reconnecting its disconnected fragments):

```r
healed = heal_skeleton(parents, s$d$X, s$d$Y, s$d$Z, method="ALL",
                       max_dist=NULL, min_size=NULL, mask=NULL,
                       radius=NULL, use_radius=FALSE)

# ... optionally taking node radii into account, which prefers to connect
# fragments of similar calibre (higher `use_radius` = more influence)
healed = heal_skeleton(parents, s$d$X, s$d$Y, s$d$Z, method="ALL",
                       max_dist=NULL, min_size=NULL, mask=NULL,
                       radius=s$d$W, use_radius=TRUE)
```

## Available functions

**Skeleton / tree (DAG)**

- `node_indices`: turn node and parent IDs into parent indices
- `geodesic_distances`: geodesic distances between all/subsets of nodes
- `geodesic_pairs`: geodesic distances for explicit pairs of nodes
- `geodesic_nearest`: distance to the nearest target for each source (no full matrix)
- `geodesic_farthest`: distance to the farthest target for each source (no full matrix)
- `strahler_index`: calculate the Strahler index
- `subtree_height`: distance from each node down to the farthest leaf below it
- `connected_components`: extract connected components
- `classify_nodes`: classify nodes into roots, leaves, branch points and slabs
- `all_dists_to_root`: distances from all/subsets of nodes to the root
- `dist_to_root`: distance from a single node to the root
- `prune_twigs`: prune twigs under a given size threshold
- `generate_segments` / `break_segments`: split the tree into linear segments
- `synapse_flow_centrality`: synapse flow centrality per node
- `has_cycles`: check whether a tree contains cycles
- `child_to_parent_dists`: helper to calculate child -> parent distances
- `heal_skeleton`: reconnect the fragments of a broken skeleton
- `stitch_fragments`: find the minimal-length edges that reconnect fragments
- `reroot_rewire`: regenerate a parent vector after adding edges
- `descendants` / `paths_to_root`: everything below a node, and everything above it
- `reroot`: re-orient a forest at given nodes, reversing only what has to move
- `contract_nodes`: collapse groups of nodes onto a representative and rewire
- `simplify_skeleton`: keep only roots, leafs and branch points, preserving cable length
- `adjacency`: the skeleton's adjacency matrix, as the three arrays of a CSR matrix
- `longest_path` / `longest_paths`: the longest path to a root, and the `n` longest in turn
- `betweenness`: betweenness centrality in `O(N)` rather than Brandes' `O(V*E)`
- `descendant_counts`: how many nodes lie strictly below each node

**Mesh and graph**

- `mesh_connected_components`: connected components of a triangle mesh
- `unique_edges`: the unique undirected edges of a triangle mesh, with lengths
- `connected_components_graph`: connected components of any graph, from an edge list
- `level_set_components`: the components of *every* level set in one pass (wavefront rings)
- `contract_vertices`: collapse nodes onto new ids and simplify the edge list
- `minimum_spanning_tree`: minimum (or maximum) spanning forest
- `parents_from_edges`: orient an edge list into a rooted forest — breaks cycles, and
  hands back the order that makes parents precede their children
- `bridges`: which edges may not be dropped without disconnecting the graph
- `geodesic_mst_mesh` / `geodesic_mst_graph`: span a *subset* of nodes by geodesic
  distance, without ever building the `k x k` matrix
- `geodesic_predecessors` / `geodesic_path`: the shortest *route*, not just its length
- `geodesic_clusters`: greedily partition a graph into clusters of bounded geodesic radius

!!! note "Indices are 0-based"

    Every function in these two groups speaks in 0-based node indices, matching the
    Rust core and the rest of the DAG family — so add 1 before using a returned index
    to subset an R vector or matrix. Roots and "no such node" are `-1`.

**Neuron similarity (NBLAST / synNBLAST)** — see [Concepts › NBLAST](../concepts/nblast.md)

- `nblast` / `nblast_allbyall`: forward NBLAST (query-vs-target / all-by-all)
- `nblast_knn`: each neuron's `k` nearest neighbours, without the score matrix
- `nblast_pairs`: forward NBLAST for a set of `(query, target)` index pairs
- `synblast` / `synblast_allbyall`: synapse-based NBLAST
- `smat_auto_limit`: the `limit_dist="auto"` value for a scoring matrix

```r
# The 20 nearest neighbours of every neuron, never materialising the n x n matrix.
nn <- nblast_knn(points, vects, k = 20)
nn$idx[1, ]      # 1-based neighbour indices, best first (NA-padded if short)
nn$scores[1, ]   # their exact NBLAST scores

# Query vs target: `idx` then indexes the targets.
nn <- nblast_knn(q_points, q_vects, target = t_points, target_vects = t_vects, k = 5)
```

Only *which* neurons make the shortlist is approximate; every returned score is an
exact NBLAST value. `symmetry` defaults to `"mean"` here (unlike the matrix
functions) because the combine has to happen **before** the top-`k` cut — once
only `k` neighbours per row survive there is no transpose left to symmetrise
against. `n_candidates` (default `200`) trades recall against cost: on 163,976
real neurons recall@20 was 0.91 at 50, 0.97 at 100 and 0.99 at 200. Unlike the
Python bindings, `idx` is **1-based** and short rows are padded with `NA` rather
than `-1` / `-Inf`.

**Clustering**

- `nblast_hclust`: cluster a score matrix, returning an `hclust`
- `nblast_dist`: condensed distances from a score matrix, as a `dist`
- `fast_hclust`: cluster an existing `dist`, without the 65536 limit
- `symmetrize`: combine a score matrix with its transpose, on its own
- `leaf_order`: the drawing order for a merge matrix, i.e. what `hclust$order` holds

```r
scores <- nblast_allbyall(points, vects, ...)   # (n, n) score matrix

# Symmetrise, 1 - score, condense and cluster - in one fused pass.
h <- nblast_hclust(scores, method = "ward")

# A standard hclust object, so the rest of R just works.
groups <- cutree(h, k = 10)
plot(h)
```

Two things this buys you over the idiomatic spelling:

- **No size ceiling.** `stats::hclust` refuses more than 65536 observations —
  a hard blocker at whole-brain scale, where 100k–200k neurons is routine.
  `nblast_hclust` and `fast_hclust` are bounded only by memory.
- **No `n × n` temporaries.** `hclust(as.dist(1 - (m + t(m)) / 2))` materialises
  three more full matrices before clustering starts. Here symmetrising, the
  distance transform and condensing are fused into a single pass, and the
  condensed buffer is then clustered in place.

Method names follow SciPy, so `"ward"` is R's `"ward.D2"` and `"weighted"` is
R's `"mcquitty"`. Note `"centroid"` and `"median"` take **plain** distances here,
whereas `stats::hclust` expects squared ones for those two.

The two steps are also available on their own. `symmetrize(scores)` does the
combine without the rest of the pipeline — for when something *other* than
clustering has to read the matrix — and returns a copy, since R's value semantics
forbid writing to yours. `leaf_order(h)` recomputes the drawing order from a merge
matrix; you do not need it for an untouched `hclust`, which already carries the
same ordering, but you do for one you built or rearranged yourself:

```r
h$merge[nrow(h$merge), ] <- rev(h$merge[nrow(h$merge), ])   # flip the root
h$order <- leaf_order(h)                                    # or it draws crossed
```

**CMTK transforms** — see [CMTK transforms](../python/cmtk.md) for the full story

- `cmtk_read`: read a CMTK `.list` registration (or a chain of them)
- `cmtk_xform` / `cmtk_xform_inv`: apply it to points, forwards / backwards
- `cmtk_affine`, `cmtk_domain`, `cmtk_dims`, `cmtk_spacing`, `cmtk_versions`: properties

CMTK itself does **not** need to be installed — no shelling out to `streamxform`:

```r
reg <- cmtk_read("JFRC2_FCWB.list")

n <- Cell07PNs[[1]]
xyzmatrix(n) <- cmtk_xform(reg, xyzmatrix(n))

# points outside the registration's domain come back as NaN, exactly as CMTK
# reports them as FAILED
```

Direction is chosen per call, so one object serves both ways round and the file is parsed
once. `invert` is *per hop* — unlike `cmtk_xform_inv`, which reverses the whole chain — so
it is the only way to express a mixed-direction traversal:

```r
back  <- cmtk_xform(reg, pts, invert = TRUE)          # same parse, other direction
chain <- cmtk_read(c("A_B.list", "C_B.list"))         # A -> B -> C, 2nd stored as C->B
mixed <- cmtk_xform(chain, pts, invert = c(FALSE, TRUE))
```

**Elastix transforms** — see [Elastix transforms](../python/elastix.md) for the full story

- `elastix_read`: read a `TransformParameters` file (its initial-transform chain is followed
  automatically, however deep)
- `elastix_xform` / `elastix_xform_inv`: apply it to points, forwards / backwards
- `elastix_probe_invertible`: can it be inverted? Answered without reading the coefficients —
  ~20x faster than a full read, for labelling many files at once
- `elastix_affine`, `elastix_kinds`, `elastix_grid_size`, `elastix_grid_spacing`,
  `elastix_grid_origin`: properties

Elastix itself does **not** need to be installed — no shelling out to `transformix`:

```r
xf <- elastix_read("TransformParameters.FixedFANC.txt")
xyzmatrix(n) <- elastix_xform(xf, xyzmatrix(n))

# NB the opposite convention to CMTK: points outside the control-point grid come back
# *unchanged*, which is what Elastix does. Pass out_of_bounds = "nan" to see the boundary.
back <- elastix_xform_inv(xf, xyzmatrix(n))   # Elastix itself cannot invert at all
```

As with CMTK, direction is chosen per call — `elastix_xform(xf, pts, invert = TRUE)` — so a
transform and its inverse share one parse. That matters when the warp is tens of megabytes.

**Landmark transforms** — see [Landmark transforms](../python/landmarks.md) for the full story

When there is no image registration, only matched landmarks:

- `tps_transform` / `tps_xform`: fit a thin-plate spline and apply it. This is `nat`'s
  `tpsreg`, without the fit being repeated on every call.
- `mls_transform` / `mls_xform`: moving least squares — every point gets its own locally
  weighted affine, so there is no fit at all.
- `tps_affine`, `mls_affine`: the global affine each converges to far from the landmarks
- `tps_coefs`: the fit itself (`W` and `A`), if you want to store it rather than repeat it

```r
lm <- read.csv("mirror_landmarks.csv")          # x/y/z columns are found by name
tps <- tps_transform(lm[, 1:3], lm[, 4:6])
xyzmatrix(n) <- tps_xform(tps, xyzmatrix(n))

# moving least squares instead; "inverse" is free here, unlike refitting a spline
mls <- mls_transform(lm[, 1:3], lm[, 4:6], direction = "inverse")
```

Neither builds the `points x landmarks` matrix the reference implementations do, so there is
no batch size to tune and the landmark count is not bounded by memory.

## Function reference

Per-function documentation is generated from the package's roxygen docs and
published by R-universe:

[**nat.fastcore reference on R-universe**](https://schlegelp.r-universe.dev/nat.fastcore){ .md-button }

From R, the usual `?geodesic_distances` works too.

!!! note "Differences from the Python bindings"

    `prune_twigs` has no `mask` argument in R (extendr cannot take a `Vec<bool>`).
    Conversely, R exposes several functions that Python keeps internal or folds
    into keyword arguments — `node_indices` (Python maps IDs to indices for you),
    `all_dists_to_root` and `dist_to_root` (one `dist_to_root(sources=)` there),
    `synblast_allbyall` (`synblast(target=None)`) and `smat_auto_limit`
    (`limit_dist="auto"`). See the
    [capability matrix](../index.md#whats-available-where).
