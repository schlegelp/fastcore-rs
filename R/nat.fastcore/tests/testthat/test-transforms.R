# The coordinate coercion shared by every transform wrapper (cmtk.R, elastix.R, warp.R).
#
# This used to be three near-identical private copies. It is now one, so its behaviour is
# worth pinning explicitly rather than only through whichever wrapper happens to exercise it.

test_that("an (N, 3) matrix passes through, coerced to double", {
  m <- matrix(1:6, ncol = 3)
  out <- .xform_xyz(m)
  expect_true(is.matrix(out))
  expect_equal(dim(out), c(2L, 3L))
  expect_type(out, "double")
  expect_equal(out, matrix(as.double(1:6), ncol = 3))
})

test_that("a bare length-3 vector becomes a single row", {
  expect_equal(.xform_xyz(c(1, 2, 3)), matrix(c(1, 2, 3), ncol = 3))
})

test_that("coordinate columns are found by name, in any case", {
  want <- matrix(c(1, 2, 10, 20, 100, 200), ncol = 3)

  lower <- data.frame(x = c(1, 2), y = c(10, 20), z = c(100, 200))
  expect_equal(.xform_xyz(lower), want)

  upper <- data.frame(X = c(1, 2), Y = c(10, 20), Z = c(100, 200))
  expect_equal(.xform_xyz(upper), want)
})

test_that("columns stored out of order are reordered, not silently mis-read", {
  # Positional coercion would hand back (z, y, x) here and no one would notice.
  df <- data.frame(z = c(100, 200), y = c(10, 20), x = c(1, 2))
  expect_equal(.xform_xyz(df), matrix(c(1, 2, 10, 20, 100, 200), ncol = 3))
})

test_that("extra columns are ignored when x/y/z are named", {
  # The shape `read.csv()` gives you for an SWC, or for a landmarks file carrying labels.
  df <- data.frame(
    PointNo = 1:2, Label = c("a", "b"),
    X = c(1, 2), Y = c(10, 20), Z = c(100, 200), Parent = c(-1L, 1L)
  )
  expect_equal(.xform_xyz(df), matrix(c(1, 2, 10, 20, 100, 200), ncol = 3))
})

test_that("a three-column frame without x/y/z names falls back to position", {
  df <- data.frame(a = c(1, 2), b = c(10, 20), c = c(100, 200))
  expect_equal(.xform_xyz(df), matrix(c(1, 2, 10, 20, 100, 200), ncol = 3))
})

test_that("a frame that is neither named nor (N, 3) is an error naming the argument", {
  df <- data.frame(a = 1:2, b = 3:4)
  expect_error(.xform_xyz(df, "landmarks"), "`landmarks` must be an \\(N, 3\\) matrix")
})

test_that("wrong shapes are rejected", {
  expect_error(.xform_xyz(matrix(1:8, ncol = 4)), "\\(N, 3\\)")
  expect_error(.xform_xyz(c(1, 2)), "\\(N, 3\\)")
  expect_error(.xform_xyz("nope"), "\\(N, 3\\)")
})

test_that("every transform wrapper accepts the same shapes", {
  # One landmark pair, three spellings of the same points -- all must agree.
  src <- matrix(runif(12 * 3, 0, 10), ncol = 3)
  trg <- src * 1.5 + 2
  tps <- tps_transform(src, trg)
  mls <- mls_transform(src, trg)

  pts <- matrix(c(1, 2, 10, 20, 100, 200), ncol = 3)
  named <- data.frame(x = c(1, 2), y = c(10, 20), z = c(100, 200))
  swc <- data.frame(PointNo = 1:2, X = c(1, 2), Y = c(10, 20), Z = c(100, 200))

  expect_equal(tps_xform(tps, named), tps_xform(tps, pts))
  expect_equal(tps_xform(tps, swc), tps_xform(tps, pts))
  expect_equal(mls_xform(mls, named), mls_xform(mls, pts))
  expect_equal(mls_xform(mls, swc), mls_xform(mls, pts))
})
