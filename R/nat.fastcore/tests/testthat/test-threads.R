# How wide the parallel work runs.
#
# Note what is *not* tested here: that `set_num_threads(n)` actually sizes the
# pool to `n`. The pool is built once per session and testthat runs every file in
# one, so a test that set it would either fail or silently change the size the
# rest of the suite runs at, depending on file order. The Rust side of that
# behaviour (idempotent repeat, refusing to resize) is covered by the Python
# suite, which drives the same code and can afford a fresh interpreter per case.
# What is checked here is the R surface: the arguments exist, reject nonsense,
# and do not change any answer.

test_that("get_num_threads reports a usable pool size", {
  n <- get_num_threads()
  expect_type(n, "integer")
  expect_gte(n, 1L)
})

test_that("set_num_threads rejects a thread count below one", {
  # Rejected before the pool is touched, so this is safe to run in-session.
  # Matched loosely: extendr reports any panic as "User function panicked: <fn>"
  # and prints the reason ("`n` must be >= 1") to stderr instead, so the specific
  # text is not in the condition to assert on.
  expect_error(set_num_threads(0L), "set_num_threads")
  expect_error(set_num_threads(-1L), "set_num_threads")
})

test_that("capping threads does not change what heal_skeleton returns", {
  # Two fragments, {0, 1} and {2, 3}, whose closest pair is node 1 and node 2.
  parents <- c(-1L, 0L, -1L, 2L)
  x <- c(0, 1, 10, 11)
  y <- c(0, 0, 0, 0)
  z <- c(0, 0, 0, 0)

  expected <- heal_skeleton(parents, x, y, z, "ALL", NULL, NULL, NULL, NULL, NULL)
  expect_equal(expected, c(-1L, 0L, 1L, 2L))

  for (threads in c(1L, 2L, 4L)) {
    expect_equal(
      heal_skeleton(parents, x, y, z, "ALL", NULL, NULL, NULL, NULL, NULL, threads),
      expected
    )
  }
})

test_that("capping threads does not change what stitch_fragments returns", {
  components <- c(0L, 0L, 1L, 1L)
  x <- c(0, 1, 10, 11)
  y <- c(0, 0, 0, 0)
  z <- c(0, 0, 0, 0)

  expected <- stitch_fragments(components, x, y, z, NULL, NULL, NULL)
  expect_equal(expected$from, 1L)
  expect_equal(expected$to, 2L)
  expect_equal(expected$dist, 9)

  for (threads in c(1L, 2L, 4L)) {
    expect_equal(
      stitch_fragments(components, x, y, z, NULL, NULL, NULL, threads),
      expected
    )
  }
})

test_that("capping threads does not change what geodesic_pairs returns", {
  # A path 0-1-2-3-4.
  parents <- c(-1L, 0L, 1L, 2L, 3L)
  sources <- c(0L, 0L, 2L)
  targets <- c(1L, 4L, 3L)

  expected <- geodesic_pairs(parents, sources, targets, NULL, FALSE)
  expect_equal(expected, c(1, 4, 1))

  # More threads than pairs is the edge case for the chunking: every pair must
  # still be answered exactly once.
  for (threads in c(1L, 2L, 8L)) {
    expect_equal(
      geodesic_pairs(parents, sources, targets, NULL, FALSE, threads),
      expected
    )
  }
})
