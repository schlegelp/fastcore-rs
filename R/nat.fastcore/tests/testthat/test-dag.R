# Tree traversal and editing.
#
# The oracles are hand-computed answers on the small fixtures in helper-graph.R,
# plus cross-checks against the bindings that were already here (`connected_components`,
# `all_dists_to_root`, `break_segments`) — those pin the new functions to the
# conventions the package already follows rather than to a second opinion of mine.

test_that("descendants walks down and paths_to_root walks up", {
  p <- .forest_parents()

  # Whole path below the root, source first, depth-first pre-order.
  expect_equal(descendants(p, 0L)[[1]], 0:3)
  expect_equal(descendants(p, 2L)[[1]], 2:3)
  expect_equal(descendants(p, 3L)[[1]], 3L) # a leaf is its own sub-tree

  # ...and back up again, source first, root last.
  expect_equal(paths_to_root(p, 3L)[[1]], c(3L, 2L, 1L, 0L))
  expect_equal(paths_to_root(p, 0L)[[1]], 0L) # a root is a single-element path

  # One entry per source, in `sources` order.
  expect_equal(descendants(p, c(0L, 4L, 6L)), list(0:3, 4:5, 6L))

  # An out-of-range source gives an empty vector rather than an error.
  expect_equal(length(descendants(p, 99L)[[1]]), 0L)
  expect_equal(length(paths_to_root(p, -5L)[[1]]), 0L)
})

test_that("descendants agrees with the component labels for a whole component", {
  p <- .forest_parents()
  comp <- connected_components(p)
  for (root in which(p < 0) - 1L) {
    expect_equal(
      sort(descendants(p, root)[[1]]),
      sort(which(comp == comp[root + 1L]) - 1L)
    )
  }
})

test_that("reroot reverses only the path to the old root", {
  p <- .forest_parents()

  # Path 0-1-2-3 rooted at 3: every link on it flips, and the *other* components
  # come back byte-identical.
  out <- reroot(p, 3L)
  expect_equal(out, c(1L, 2L, 3L, -1L, -1L, 4L, -1L))
  expect_equal(out[5:7], p[5:7])

  # Rooting at a node that is already a root is a no-op.
  expect_equal(reroot(p, 0L), p)

  # Several roots at once, one per component.
  expect_equal(reroot(p, c(3L, 5L)), c(1L, 2L, 3L, -1L, 5L, -1L, -1L))
})

test_that("contract_nodes collapses groups and refuses cycles", {
  p <- .forest_parents()

  # Collapse node 1 onto node 0; everything else maps to itself.
  out <- contract_nodes(p, c(0L, 0L, 2L, 3L, 4L, 5L, 6L))
  expect_equal(out$nodes, c(0L, 2L, 3L, 4L, 5L, 6L))
  # Parents index *into* `nodes`, so 2's parent is position 0 (node 0).
  expect_equal(out$parents, c(-1L, 0L, 1L, -1L, 3L, -1L))

  # A mapping that would close a cycle is an error, not a silent non-forest:
  # mapping the root onto its own descendant.
  expect_error(contract_nodes(p, c(3L, 1L, 2L, 3L, 4L, 5L, 6L)))
})

test_that("simplify_skeleton keeps roots, leafs and branch points, preserving length", {
  p <- .arbor_parents()
  w <- c(0, 1, 1, 1, 1, 1) # each child->parent edge is length 1

  out <- simplify_skeleton(p, w)
  # 0 is the root, 1 the branch point, 3 and 5 the leafs; 2 and 4 are pass-through.
  expect_equal(out$nodes, c(0L, 1L, 3L, 5L))
  expect_equal(out$parents, c(-1L, 0L, 1L, 1L))
  # Each surviving edge carries the chain it replaced: 1->0 is one edge, the two
  # leaf edges are two each.
  expect_equal(out$weights, c(0, 1, 2, 2))

  # Without weights there are no lengths to report.
  expect_null(simplify_skeleton(p)$weights)
})

test_that("adjacency returns a well-formed CSR triple", {
  p <- .arbor_parents()
  n <- length(p)

  a <- adjacency(p)
  expect_equal(length(a$indptr), n + 1L)
  expect_equal(a$indptr[1], 0L)
  expect_equal(a$indptr[n + 1L], length(a$indices))
  expect_equal(length(a$indices), length(a$data))
  # Directed: one entry per non-root node, its parent.
  expect_equal(length(a$indices), sum(p >= 0))

  # Undirected has both directions, so twice as many entries...
  u <- adjacency(p, directed = FALSE)
  expect_equal(length(u$indices), 2L * sum(p >= 0))
  # ...and column indices ascend within each row.
  for (i in seq_len(n)) {
    row <- u$indices[seq(u$indptr[i] + 1L, length.out = u$indptr[i + 1L] - u$indptr[i])]
    expect_false(is.unsorted(row))
  }
})

test_that("longest_path is distal-first and longest_paths peels them off in turn", {
  p <- .arbor_parents()

  # Ties break towards the lowest node index, so the 0-1-2-3 branch wins.
  expect_equal(longest_path(p), c(3L, 2L, 1L, 0L))

  paths <- longest_paths(p, 2L)
  expect_equal(length(paths), 2L)
  expect_equal(paths[[1]], c(3L, 2L, 1L, 0L))
  # The second is what is left once the first is removed.
  expect_equal(paths[[2]][1], 5L)
  # Longest first.
  expect_true(length(paths[[1]]) >= length(paths[[2]]))

  # Weights change which branch is longest.
  w <- c(0, 1, 1, 1, 10, 10)
  expect_equal(longest_path(p, w)[1], 5L)
})

test_that("betweenness and descendant_counts are different questions", {
  p <- .forest_parents()

  # descendant_counts: how many nodes lie strictly below each. A leaf scores 0.
  expect_equal(descendant_counts(p), c(3, 2, 1, 0, 1, 0, 0))

  # betweenness: shortest paths through each node, counted within a component.
  # On the 4-node path the two interior nodes each carry 2 pairs; the ends carry
  # none, and the isolated node none.
  expect_equal(betweenness(p, FALSE), c(0, 2, 2, 0, 0, 0, 0))

  # Restricting `targets` counts only those.
  expect_equal(descendant_counts(p, c(3L)), c(1, 1, 1, 0, 0, 0, 0))

  # Doubles, not integers: an undirected 100k skeleton overflows R's 32-bit int.
  expect_type(betweenness(p, FALSE), "double")
  expect_type(descendant_counts(p), "double")
})

test_that("directed and undirected betweenness are genuinely different", {
  # A path cannot tell them apart -- both give (0, 2, 2, 0) -- so use the arbor,
  # where the branch point separates three parts rather than two.
  #
  #   0 - 1 - 2 - 3
  #       |
  #       4 - 5
  p <- .arbor_parents()

  # Directed is descendants x ancestors, so a root and every leaf score 0.
  expect_equal(betweenness(p, TRUE), c(0, 4, 2, 0, 2, 0))

  # Undirected sums products over the parts each node separates. Node 1 splits
  # {0}, {2,3} and {4,5}: 1*2 + 1*2 + 2*2 = 8.
  expect_equal(betweenness(p, FALSE), c(0, 8, 4, 0, 4, 0))

  # ...and neither is the descendant count, which is what navis' `from_` argument
  # actually computes.
  expect_equal(descendant_counts(p), c(5, 4, 1, 0, 1, 0))
})
