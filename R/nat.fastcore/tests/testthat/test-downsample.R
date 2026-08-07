# Downsampling, resampling and smoothing skeleton geometry.
#
# The invariant all six functions share, and the one worth testing hardest: none of
# them may change the topology. Roots, branch points and leafs survive untouched, and
# only the sampling along each neurite changes. The oracles are hand-computed answers
# on the fixtures in helper-graph.R, plus cross-checks against `simplify_skeleton()`
# and `classify_nodes()`, which pin the new functions to conventions the package
# already follows.

# A straight chain of `n` nodes at unit spacing along x.
.chain_parents <- function(n) c(-1L, seq_len(n - 1L) - 1L)
.chain_x <- function(n) as.numeric(seq_len(n) - 1L)
.zeros <- function(n) rep(0, n)

test_that("downsample_skeleton keeps every Nth node and both ends", {
  p <- .chain_parents(11L) # nodes 0..10, the leaf at 10

  # The segment runs leaf -> root, so positions count down from node 10.
  expect_equal(downsample_skeleton(p, 2L)$nodes, c(0L, 2L, 4L, 6L, 8L, 10L))
  expect_equal(downsample_skeleton(p, 5L)$nodes, c(0L, 5L, 10L))

  # A factor of 1 changes nothing at all.
  out <- downsample_skeleton(p, 1L)
  expect_equal(out$nodes, 0:10)
  expect_equal(out$parents, p)
})

test_that("downsample_skeleton with a huge factor equals simplify_skeleton", {
  p <- .arbor_parents()
  w <- c(0, 1, 2, 4, 8, 16)

  got <- downsample_skeleton(p, 1000L, weights = w)
  want <- simplify_skeleton(p, weights = w)

  expect_equal(got$nodes, want$nodes)
  expect_equal(got$parents, want$parents)
  expect_equal(got$weights, want$weights)

  # Root 0, branch 1, leafs 3 and 5 -- and the cable of the dropped nodes with them.
  expect_equal(got$nodes, c(0L, 1L, 3L, 5L))
  expect_equal(sum(got$weights), sum(w))
})

test_that("downsample_skeleton honours preserve", {
  p <- .chain_parents(11L)
  keep <- rep(FALSE, 11L)
  keep[8] <- TRUE # 1-based R index -> node 7

  expect_equal(downsample_skeleton(p, 5L, preserve = keep)$nodes, c(0L, 5L, 7L, 10L))
})

test_that("node_map sends every node to the nearest survivor", {
  p <- .chain_parents(11L) # nodes 0..10, only the root and the leaf survive

  # Unweighted, so hops: nodes 1-5 are nearer the root (node 5 by the tie-break
  # towards the root), 6-9 nearer the leaf. The map is 0-based *into* `nodes`, so
  # the root is 0 and the leaf is 1.
  map <- downsample_skeleton(p, 100L)$node_map
  expect_equal(map, c(0L, 0L, 0L, 0L, 0L, 0L, 1L, 1L, 1L, 1L, 1L))

  # Weighting the first edge heavily moves the split. Reaching the root now costs 8
  # on top of the walk down the chain, so only node 1 is still nearer to it -- the
  # five nodes that went to the root unweighted are down to one.
  w <- c(0, 8, rep(1, 9))
  expect_equal(
    downsample_skeleton(p, 100L, weights = w)$node_map,
    c(0L, 0L, rep(1L, 9L))
  )
})

test_that("node_map is the same for every dropper, and points at survivors", {
  p <- .arbor_parents()
  x <- c(0, 1, 2, 3, 2, 3) # the two arms run straight, so all three drop the slabs

  maps <- list(
    downsample_skeleton(p, 1000L)$node_map,
    simplify_rdp(p, x, .zeros(6L), .zeros(6L), epsilon = 1e9)$node_map,
    simplify_vw(p, x, .zeros(6L), .zeros(6L), min_area = 1e9)$node_map
  )
  expect_equal(maps[[2]], maps[[1]])
  expect_equal(maps[[3]], maps[[1]])

  # One entry per input node, every entry indexing a node that is actually there,
  # and a survivor mapping to its own slot.
  nodes <- downsample_skeleton(p, 1000L)$nodes
  expect_equal(length(maps[[1]]), length(p))
  expect_true(all(maps[[1]] %in% (seq_along(nodes) - 1L)))
  expect_equal(nodes[maps[[1]][nodes + 1L] + 1L], nodes)
})

test_that("simplify_rdp collapses a straight line and keeps a corner", {
  n <- 9L
  p <- .chain_parents(n)

  # Perfectly straight: everything between the two ends goes.
  out <- simplify_rdp(p, .chain_x(n), .zeros(n), .zeros(n), epsilon = 0.5)
  expect_equal(out$nodes, c(0L, 8L))

  # An L: 5 nodes out along x, then 4 up along y. Only the corner bends the path.
  x <- c(0, 1, 2, 3, 4, 4, 4, 4, 4)
  y <- c(0, 0, 0, 0, 0, 1, 2, 3, 4)
  out <- simplify_rdp(p, x, y, .zeros(n), epsilon = 0.5)
  expect_equal(out$nodes, c(0L, 4L, 8L))
})

test_that("simplify_rdp preserves cable length", {
  n <- 20L
  p <- .chain_parents(n)
  w <- c(0, rep(1, n - 1L))

  out <- simplify_rdp(p, .chain_x(n), .zeros(n), .zeros(n), epsilon = 5, weights = w)

  expect_equal(out$nodes, c(0L, 19L))
  expect_equal(sum(out$weights), sum(w))
})

test_that("simplify_vw drops the flattest node first", {
  p <- .chain_parents(5L)
  # Two bumps off a straight line, one ten times taller than the other.
  x <- c(0, 1, 2, 3, 4)
  y <- c(0, 0.1, 0, 1, 0)

  out <- simplify_vw(p, x, y, .zeros(5L), min_area = 0.5)
  # The small bump (node 1) goes; node 2 survives because losing node 1 widens its
  # triangle past the threshold.
  expect_equal(out$nodes, c(0L, 2L, 3L, 4L))

  # A threshold of zero is a no-op: no triangle is smaller than zero area.
  expect_equal(simplify_vw(p, x, y, .zeros(5L), min_area = 0)$nodes, 0:4)
})

test_that("resample_skeleton places nodes at the requested spacing", {
  n <- 11L # 10 units of cable
  p <- .chain_parents(n)

  out <- resample_skeleton(p, .chain_x(n), .zeros(n), .zeros(n), spacing = 2)

  expect_equal(length(out$parents), 6L)
  expect_equal(sort(out$x), c(0, 2, 4, 6, 8, 10))
  expect_equal(out$y, rep(0, 6L))

  # Exactly one root, and every other node points at a node that exists.
  expect_equal(sum(out$parents < 0), 1L)
  expect_true(all(out$parents[out$parents >= 0] %in% (seq_along(out$parents) - 1L)))
})

test_that("resample_skeleton reports where each node came from", {
  n <- 3L
  p <- .chain_parents(n)

  out <- resample_skeleton(p, .chain_x(n), .zeros(n), .zeros(n), spacing = 0.5)

  # The root and the leaf come first, unmoved, with alpha 0 and themselves as source.
  expect_equal(out$x[1:2], c(0, 2))
  expect_equal(out$source_from[1:2], c(0L, 2L))
  expect_equal(out$source_to[1:2], c(0L, 2L))
  expect_equal(out$alpha[1:2], c(0, 0))

  # The documented interpolation reproduces the coordinates the function chose, so a
  # caller interpolating a radius the same way gets something consistent.
  from <- out$source_from + 1L
  to <- out$source_to + 1L
  expect_equal(out$x, .chain_x(n)[from] * (1 - out$alpha) + .chain_x(n)[to] * out$alpha)
})

test_that("resample_skeleton reports where each node went", {
  n <- 5L # 4 units of cable at x = 0..4
  p <- .chain_parents(n)

  out <- resample_skeleton(p, .chain_x(n), .zeros(n), .zeros(n), spacing = 2)

  # Root and leaf are carried over as slots 0 and 1; the one new node, at x = 2, is
  # slot 2. Nodes at x = 1 and x = 3 are each exactly halfway between two output
  # nodes and both go the proximal way -- towards the root at slot 0.
  expect_equal(length(out$parents), 3L)
  expect_equal(out$x[3], 2)
  expect_equal(out$node_map, c(0L, 0L, 2L, 2L, 1L))

  # Every input node lands on an output node, and the carried-over ones on themselves.
  expect_equal(length(out$node_map), n)
  expect_true(all(out$node_map %in% (seq_along(out$parents) - 1L)))
})

test_that("resample_skeleton collapses a segment shorter than the spacing", {
  n <- 5L
  p <- .chain_parents(n)

  out <- resample_skeleton(p, .chain_x(n), .zeros(n), .zeros(n), spacing = 100)

  expect_equal(length(out$parents), 2L)
  expect_equal(out$parents, c(-1L, 0L))
})

test_that("smoothing flattens a zig-zag but pins the ends", {
  n <- 21L
  p <- .chain_parents(n)
  x <- .chain_x(n)
  y <- rep(c(1, -1), length.out = n)

  for (out in list(
    smooth_skeleton(p, x, y, .zeros(n), window = 5L),
    smooth_skeleton_gaussian(p, x, y, .zeros(n), sigma = 3)
  )) {
    # The root and the leaf do not move...
    expect_equal(out$y[1], y[1])
    expect_equal(out$y[n], y[n])
    # ...and everything in between is much flatter than it was.
    expect_true(all(abs(out$y[3:(n - 2L)]) < 0.5))
  }
})

test_that("smoothing pins branch points and leaves the topology alone", {
  p <- .arbor_parents()
  n <- length(p)
  x <- c(0, 1, 2, 3, 2, 3)
  y <- c(0, 0, 1, 2, -1, -2)
  z <- .zeros(n)

  # Root 0, branch 1, leafs 3 and 5 -- exactly what simplify_skeleton keeps.
  pinned <- simplify_skeleton(p)$nodes + 1L

  for (out in list(
    smooth_skeleton(p, x, y, z, window = 5L),
    smooth_skeleton_gaussian(p, x, y, z, sigma = 2)
  )) {
    expect_equal(length(out$x), n)
    expect_equal(out$x[pinned], x[pinned])
    expect_equal(out$y[pinned], y[pinned])
  }

  # A window of 1 is a no-op.
  out <- smooth_skeleton(p, x, y, z, window = 1L)
  expect_equal(out$x, x)
  expect_equal(out$y, y)
})

test_that("every method leaves the node classes untouched", {
  p <- .forest_parents()
  n <- length(p)
  x <- c(0, 1, 2, 3, 10, 11, 20)
  y <- c(0, 1, 0, 1, 0, 1, 0)
  z <- .zeros(n)
  before <- classify_nodes(p)

  results <- list(
    downsample_skeleton(p, 2L),
    simplify_rdp(p, x, y, z, epsilon = 0.1),
    simplify_vw(p, x, y, z, min_area = 0.1)
  )

  for (out in results) {
    # Every node that carries topology survives...
    expect_true(all((which(before != 3L) - 1L) %in% out$nodes))
    # ...and the roles of the survivors are unchanged.
    expect_equal(classify_nodes(out$parents), before[out$nodes + 1L])
  }
})
