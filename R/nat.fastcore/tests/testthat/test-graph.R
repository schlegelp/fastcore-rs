# Graph primitives over an edge list.
#
# Where a result has a closed form on the fixture (a unit square, a ring) that is
# the oracle; where it does not, the check is against an invariant the function
# promises — a forest has one root per component, a bridge is an edge whose removal
# raises the component count — rather than against a second implementation.

test_that("unique_edges matches the faces it came from", {
  faces <- .square_faces()
  verts <- .square_verts()

  out <- unique_edges(faces, verts)
  # Five edges: four sides plus the shared 1-2 diagonal, as [min, max] rows sorted
  # ascending by (max, min).
  expect_equal(
    out$edges,
    .edges(0, 1, 0, 2, 1, 2, 1, 3, 2, 3)
  )
  # The diagonal is sqrt(2); the sides are 1.
  expect_equal(out$lengths, c(1, 1, sqrt(2), 1, 1), tolerance = 1e-12)

  # Without coordinates there are no lengths to report.
  expect_null(unique_edges(faces)$lengths)
})

test_that("connected_components_graph labels by lowest node index", {
  # A path 0-1-2, a lone edge 3-4, and an isolated node 5.
  edges <- .edges(0, 1, 1, 2, 3, 4)
  expect_equal(connected_components_graph(edges, 6), c(0, 0, 0, 3, 3, 5))

  # Same mesh, two entry points: the labels themselves must match, not merely the
  # partition.
  faces <- .square_faces()
  expect_equal(
    connected_components_graph(unique_edges(faces)$edges, 4),
    mesh_connected_components(faces, 4)
  )
})

test_that("bridges are exactly the edges holding a component together", {
  # A ring has no bridges; the tail hanging off it is one.
  edges <- .ring_with_tail()
  expect_equal(bridges(edges, 5), c(FALSE, FALSE, FALSE, FALSE, TRUE))

  # Every edge of a tree is a bridge.
  path <- .edges(0, 1, 1, 2, 2, 3)
  expect_true(all(bridges(path, 4)))

  # The definition, done the slow way, on a shape whose answer is not written out
  # above: two triangles joined by a chain, so only the chain's edges are bridges.
  dumbbell <- .edges(0, 1, 1, 2, 2, 0, 2, 3, 3, 4, 4, 5, 5, 6, 6, 4)
  n_comp <- function(e, n) length(unique(connected_components_graph(e, n)))
  got <- bridges(dumbbell, 7)
  base <- n_comp(dumbbell, 7)
  for (i in seq_len(nrow(dumbbell))) {
    raised <- n_comp(dumbbell[-i, , drop = FALSE], 7) > base
    expect_equal(got[i], raised, info = sprintf("edge %d", i))
  }
  expect_true(any(got) && !all(got)) # the fixture exercises both answers

  # Parallel edges are a cycle, so neither of them is a bridge — the case that
  # rules out deduplicating the adjacency first.
  expect_false(any(bridges(.edges(0, 1, 0, 1), 2)))
  expect_true(bridges(.edges(0, 1), 2))
})

test_that("parents_from_edges orients an edge list and breaks cycles", {
  edges <- .ring_with_tail()

  forest <- parents_from_edges(edges, 5)
  # One root, at the lowest node index.
  expect_equal(sum(forest$parents < 0), 1L)
  expect_equal(which(forest$parents < 0) - 1L, 0L)
  # A spanning tree of one component has n - 1 edges, so the ring lost one.
  expect_equal(sum(forest$parents >= 0), 4L)

  # `order` is topological: relabelling by it gives every node a higher id than
  # its parent, which is the SWC requirement.
  new_ids <- integer(length(forest$order))
  new_ids[forest$order + 1L] <- seq_along(forest$order) - 1L
  has_parent <- forest$parents >= 0
  expect_true(all(new_ids[has_parent] > new_ids[forest$parents[has_parent] + 1L]))

  # Rooting elsewhere moves the root and nothing else about the shape.
  rooted <- parents_from_edges(edges, 5, roots = 4L)
  expect_equal(which(rooted$parents < 0) - 1L, 4L)
  expect_equal(sum(rooted$parents >= 0), 4L)

  # One root per component, always.
  two <- .edges(0, 1, 3, 4)
  expect_equal(parents_from_edges(two, 6)$parents, c(-1L, 0L, -1L, -1L, 3L, -1L))
})

test_that("minimum_spanning_tree returns row indices, and maximize inverts it", {
  # A triangle with weights 1, 2, 3.
  edges <- .edges(0, 1, 1, 2, 0, 2)
  w <- c(1, 2, 3)

  keep <- minimum_spanning_tree(edges, 3, w)
  expect_equal(keep, c(0L, 1L)) # 0-based rows: the two cheap edges
  expect_equal(edges[keep + 1L, ], .edges(0, 1, 1, 2))

  # The maximum takes the two expensive ones instead.
  expect_equal(sort(minimum_spanning_tree(edges, 3, w, maximize = TRUE)), c(1L, 2L))

  # Disconnected input yields a forest: n_nodes - n_components edges.
  e2 <- .edges(0, 1, 1, 2, 0, 2, 3, 4)
  expect_equal(length(minimum_spanning_tree(e2, 5, c(1, 2, 3, 1))), 5L - 2L)
})

test_that("level_set_components finds every level's components in one pass", {
  # A path 0-1-2-3-4 labelled 0,0,0,1,1: one run per label.
  edges <- .edges(0, 1, 1, 2, 2, 3, 3, 4)
  out <- level_set_components(edges, 5, c(0L, 0L, 0L, 1L, 1L))
  expect_equal(out$ids, c(0L, 0L, 0L, 1L, 1L))
  expect_equal(out$n_components, 2L)

  # Nodes sharing a label but not touching stay separate.
  out <- level_set_components(edges, 5, c(0L, 1L, 1L, 1L, 0L))
  expect_equal(out$ids, c(0L, 1L, 1L, 1L, 2L))

  # Negative labels are excluded, not fused into one phantom level. This is what
  # lets an unreachable `-1` from geodesic_matrix_* be fed straight in.
  out <- level_set_components(edges, 5, c(-1L, -1L, 5L, 5L, 5L))
  expect_equal(out$ids, c(-1L, -1L, 0L, 0L, 0L))
  expect_equal(out$n_components, 1L)
})

test_that("contract_vertices collapses and simplifies the edge list", {
  edges <- .ring_with_tail()
  # Collapse the ring onto two nodes: {0,1} -> 0 and {2,3} -> 1, tail 4 -> 2.
  out <- contract_vertices(edges, c(0L, 0L, 1L, 1L, 2L))
  # Edges internal to a group are dropped, the rest deduplicated.
  expect_equal(out, .edges(0, 1, 1, 2))

  # An identity mapping is a no-op over an already-unique edge list.
  faces <- .square_faces()
  ue <- unique_edges(faces)$edges
  expect_equal(contract_vertices(ue, 0:3), ue)
})

test_that("geodesic_mst_mesh spans a subset without the k x k matrix", {
  faces <- .square_faces()
  verts <- .square_verts()

  nodes <- c(0L, 1L, 3L)
  out <- geodesic_mst_mesh(faces, 4L, nodes, verts)
  # Rows are positions in `nodes`, not vertex indices, ascending by weight.
  expect_equal(out$edges, .edges(0, 1, 1, 2))
  expect_equal(out$weights, c(1, 1), tolerance = 1e-6)

  # Map back to vertex ids yourself (the +1 is R's 1-based subsetting).
  expect_equal(
    matrix(nodes[out$edges + 1L], ncol = 2),
    .edges(0, 1, 1, 3)
  )

  # Weights are the true geodesic distances, so they are usable as lengths.
  d <- geodesic_matrix_mesh(faces, 4L, verts, nodes, nodes, NULL, NULL)
  expect_equal(
    d[cbind(out$edges[, 1] + 1L, out$edges[, 2] + 1L)],
    out$weights,
    tolerance = 1e-6
  )
})

test_that("geodesic_mst_graph is a forest when the subset cannot be joined", {
  # Two separate paths.
  edges <- .edges(0, 1, 1, 2, 5, 6, 6, 7)
  out <- geodesic_mst_graph(edges, 8L, c(0L, 2L, 5L, 7L))
  # 4 nodes, 2 components -> 2 edges, each two hops.
  expect_equal(nrow(out$edges), 2L)
  expect_equal(out$weights, c(2, 2))

  # A limit shorter than either path leaves nothing to join.
  expect_equal(nrow(geodesic_mst_graph(edges, 8L, c(0L, 2L, 5L, 7L), limit = 1)$edges), 0L)
})

test_that("geodesic_predecessors and geodesic_path give the route, not just the length", {
  # A triangle whose direct 0-2 edge is expensive, so the route goes via 1.
  edges <- .edges(0, 1, 1, 2, 2, 0)
  w <- c(1, 1, 5)

  out <- geodesic_predecessors(edges, 3L, w, sources = 0L)
  expect_equal(as.vector(out$distances), c(0, 1, 2))
  expect_equal(as.vector(out$predecessors), c(-1L, 0L, 1L))

  # The same route, as a node sequence.
  expect_equal(geodesic_path(edges, 3L, 0L, 2L, w)[[1]], c(0L, 1L, 2L))

  # Unreachable comes back empty rather than as an error.
  expect_equal(length(geodesic_path(edges, 4L, 0L, 3L)[[1]]), 0L)
})

test_that("geodesic_clusters partitions by true distance from the seed", {
  # A path 0-1-...-5 with a radius of one hop.
  edges <- .edges(0, 1, 1, 2, 2, 3, 3, 4, 4, 5)
  out <- geodesic_clusters(edges, 6L, 1)
  expect_equal(out$labels, c(0L, 0L, 1L, 1L, 2L, 2L))
  expect_equal(out$n_clusters, 3L)

  # Seeding from the middle gives a different, equally valid partition — but every
  # node is still labelled.
  out <- geodesic_clusters(edges, 6L, 1, seeds = 3L)
  expect_true(all(out$labels >= 0L))
  expect_equal(length(unique(out$labels)), out$n_clusters)
})

# -----------------------------------------------------------------------------
# The binding itself
#
# Everything above tests what the primitives compute, which the Rust unit tests
# and the Python suite also cover. These test what only this layer can get wrong:
# the R <-> Rust conversions, the argument defaults, and what an R caller sees
# when they get it wrong.
# -----------------------------------------------------------------------------

test_that("edge matrices are accepted as integer as well as double", {
  # `robj_to_edges` branches on the R storage mode, and bare literals in the tests
  # above are all doubles -- so without this the integer arm is never exercised.
  edges <- .ring_with_tail()
  storage.mode(edges) <- "integer"
  expect_type(edges, "integer")

  expect_equal(bridges(edges, 5), c(FALSE, FALSE, FALSE, FALSE, TRUE))
  expect_equal(connected_components_graph(edges, 5), rep(0, 5))

  faces <- .square_faces()
  storage.mode(faces) <- "integer"
  expect_equal(unique_edges(faces)$edges, .edges(0, 1, 0, 2, 1, 2, 1, 3, 2, 3))
})

test_that("optional arguments default to their NULL behaviour", {
  edges <- .ring_with_tail()

  # Omitting an argument must equal passing NULL for it explicitly.
  expect_equal(parents_from_edges(edges, 5), parents_from_edges(edges, 5, NULL, NULL))
  expect_equal(
    minimum_spanning_tree(edges, 5),
    minimum_spanning_tree(edges, 5, NULL, FALSE, NULL)
  )
  expect_equal(geodesic_clusters(edges, 5, 1), geodesic_clusters(edges, 5, 1, NULL, NULL))

  # `threads` cannot change an answer, only how it is computed.
  expect_equal(unique_edges(.square_faces(), threads = 1L), unique_edges(.square_faces()))
  expect_equal(
    geodesic_mst_graph(edges, 5, c(0L, 3L), threads = 1L),
    geodesic_mst_graph(edges, 5, c(0L, 3L))
  )
})

test_that("caller mistakes surface as R errors, not as silent nonsense", {
  edges <- .ring_with_tail()

  # A duplicate in the node subset -- the core refuses it rather than renumbering
  # one node twice.
  expect_error(geodesic_mst_graph(edges, 5, c(0L, 3L, 0L)))

  # A node index past the end of the graph.
  expect_error(bridges(edges, 3))
  expect_error(parents_from_edges(edges, 3))

  # A weights vector that does not match the edge count.
  expect_error(minimum_spanning_tree(edges, 5, c(1, 2)))
})

test_that("precision picks the width the search accumulates at", {
  edges <- .ring_with_tail()
  w <- c(1, 1, 1, 1, 1)
  faces <- .square_faces()
  verts <- .square_verts()

  # `geodesic_matrix_*` take their optional arguments positionally; NULL is the
  # "not given" value, as everywhere else in these bindings.
  mg <- function(...) geodesic_matrix_graph(edges, 5, w, FALSE, NULL, NULL, NULL, NULL, ...)
  mm <- function(v, ...) geodesic_matrix_mesh(faces, 4L, v, NULL, NULL, NULL, NULL, ...)

  # Integer weights are exact at both widths, so the two must agree to the bit --
  # otherwise `precision` would be selecting a different algorithm, not a width.
  expect_identical(mg(precision = 32L), mg(precision = 64L))
  expect_identical(
    geodesic_predecessors(edges, 5, w, precision = 32L)$predecessors,
    geodesic_predecessors(edges, 5, w, precision = 64L)$predecessors
  )
  expect_identical(
    geodesic_path(edges, 5, 0L, c(2L, 4L), w, precision = 64L),
    geodesic_path(edges, 5, 0L, c(2L, 4L), w)
  )

  # Unweighted searches are hop counts, which both widths hold exactly.
  expect_identical(mm(NULL, precision = 64L), mm(NULL))

  # 32 is the default, so an explicit 32 changes nothing.
  expect_identical(mm(verts, precision = 32L), mm(verts))

  # Where the two differ is accuracy. The square's diagonal is sqrt(2), which is
  # irrational and so is rounded at both widths -- but only 64 lands within a
  # double's own tolerance of it.
  d32 <- mm(verts)[2, 3]
  d64 <- mm(verts, precision = 64L)[2, 3]
  expect_equal(d64, sqrt(2), tolerance = 1e-15)
  expect_gt(abs(d32 - sqrt(2)), abs(d64 - sqrt(2)))

  # Anything but 32 or 64 is a mistake, not a silent fallback.
  expect_error(mg(precision = 16L))
})

# Mesh simplification.
#
# The numerical behaviour is pinned on the Rust and Python sides (the latter against
# `pyfqmr`, which runs the same algorithm). What is checked here is the R surface:
# that the list comes back with the right shapes and 0-based indices, that the vertex
# map means what the docs say, and that the argument guards fire.

.grid_mesh <- function(n = 6L) {
  # A flat n x n grid, each cell split along its (0,0)-(1,1) diagonal. 0-based.
  idx <- function(i, j) i * n + j
  faces <- do.call(rbind, unlist(
    lapply(0:(n - 2), function(i) lapply(0:(n - 2), function(j) rbind(
      c(idx(i, j), idx(i + 1, j), idx(i + 1, j + 1)),
      c(idx(i, j), idx(i + 1, j + 1), idx(i, j + 1))
    ))),
    recursive = FALSE
  ))
  verts <- as.matrix(expand.grid(j = 0:(n - 1), i = 0:(n - 1))[, c("i", "j")])
  verts <- cbind(verts, 0)
  dimnames(verts) <- NULL
  list(faces = faces, verts = verts)
}

test_that("simplify_mesh returns a mesh and a usable vertex map", {
  m <- .grid_mesh()
  out <- simplify_mesh(m$faces, m$verts, ratio = 0.5)

  expect_named(out, c("vertices", "faces", "vertex_map"))
  expect_equal(ncol(out$vertices), 3L)
  expect_equal(ncol(out$faces), 3L)
  expect_length(out$vertex_map, nrow(m$verts))

  # Fewer faces than we started with, and the target was respected.
  expect_lte(nrow(out$faces), round(0.5 * nrow(m$faces)))

  # Faces are 0-based indices into the returned vertices.
  expect_gte(min(out$faces), 0L)
  expect_lt(max(out$faces), nrow(out$vertices))

  # The map is 0-based into the returned vertices, with -1 for "did not survive",
  # and every surviving vertex has at least one original mapping onto it.
  expect_gte(min(out$vertex_map), -1L)
  expect_lt(max(out$vertex_map), nrow(out$vertices))
  live <- out$vertex_map[out$vertex_map >= 0]
  expect_setequal(unique(live), seq_len(nrow(out$vertices)) - 1L)
})

test_that("simplify_mesh keeps a target of every face", {
  m <- .grid_mesh()
  out <- simplify_mesh(m$faces, m$verts, ratio = 1)

  expect_equal(nrow(out$faces), nrow(m$faces))
  expect_equal(out$vertex_map, seq_len(nrow(m$verts)) - 1L)
})

test_that("locked vertices survive at exactly their input position", {
  m <- .grid_mesh()
  lock <- rep(FALSE, nrow(m$verts))
  lock[c(1L, 8L, 20L)] <- TRUE

  out <- simplify_mesh(m$faces, m$verts, ratio = 0.4, lock = lock)

  kept <- out$vertex_map[lock]
  expect_true(all(kept >= 0))
  # `kept` is 0-based, so +1 to index the R matrix.
  expect_identical(out$vertices[kept + 1L, , drop = FALSE], m$verts[lock, , drop = FALSE])
})

test_that("simplify_mesh_lossless flattens a coplanar interior", {
  m <- .grid_mesh(6L)
  out <- simplify_mesh_lossless(m$faces, m$verts, preserve_border = TRUE)

  expect_lt(nrow(out$faces), nrow(m$faces))
  # Lossless: still flat, and with the border frozen the footprint is unchanged.
  expect_equal(max(abs(out$vertices[, 3])), 0)
  expect_equal(range(out$vertices[, 1]), c(0, 5))
})

test_that("simplify_mesh guards its arguments", {
  m <- .grid_mesh()

  # Bare `expect_error`, as elsewhere in this file: extendr reports a Rust panic to R
  # as "User function panicked: <name>" and drops the message, so there is nothing
  # specific to match on. The messages themselves are pinned on the Python side.
  expect_error(simplify_mesh(m$faces, m$verts))
  expect_error(simplify_mesh(m$faces, m$verts, ratio = 0.5, n_faces = 10L))
  expect_error(simplify_mesh(m$faces, m$verts, ratio = 0))
  expect_error(simplify_mesh(m$faces, m$verts, ratio = 2))
  expect_error(simplify_mesh(m$faces, m$verts, ratio = 0.5, lock = c(TRUE, FALSE)))
  expect_error(simplify_mesh_lossless(m$faces, m$verts, epsilon = -1))
})
