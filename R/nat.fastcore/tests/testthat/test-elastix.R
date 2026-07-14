# Elastix transforms, checked against Elastix's own `transformix` output.
#
# The ground truth in `transformix_golden.csv` was produced by the `transformix` binary itself, so
# these are not self-consistency checks: they pin us to Elastix's behaviour, including its habit of
# returning points outside the control-point grid **unchanged** rather than failing them (which is
# the exact opposite of CMTK, hence the contrast with `test-cmtk.R`).

ex <- function(f) system.file("extdata", "elastix", f, package = "nat.fastcore")

golden <- local({
  g <- utils::read.csv(ex("transformix_golden.csv"))
  split(as.matrix(g[, c("x", "y", "z")]), g$case)[unique(g$case)] |>
    lapply(matrix, ncol = 3)
})

# Measured margin is ~5e-7 (transformix's own print precision, six decimals). If a change lands
# at 1e-5, something is subtly wrong -- do not loosen this.
ATOL <- 1e-4

pts <- golden$input
bspline <- elastix_read(ex("bspline.txt"))

test_that("every transform type matches transformix", {
  for (case in c("affine", "translation", "euler", "euler_zyx", "similarity", "bspline", "add")) {
    xf <- elastix_read(ex(paste0(case, ".txt")))
    expect_equal(elastix_xform(xf, pts), golden[[case]],
                 tolerance = ATOL, ignore_attr = TRUE, info = case)
  }
})

test_that("ComputeZYX selects a different rotation order", {
  # Identical angles; the files differ only in `ComputeZYX`. The default (false) is Rz*Rx*Ry,
  # not the Rz*Ry*Rx its name suggests.
  a <- elastix_xform(elastix_read(ex("euler.txt")), pts)
  b <- elastix_xform(elastix_read(ex("euler_zyx.txt")), pts)
  expect_gt(max(abs(a - b)), 1)
})

test_that("points outside the grid come back unchanged, not NaN", {
  far <- rbind(c(-5000, -5000, -5000))

  # ...but the affine still applies -- it is a separate step of the chain
  affine <- elastix_read(ex("affine.txt"))
  expect_equal(elastix_xform(bspline, far), elastix_xform(affine, far), tolerance = 1e-9)

  # ...unless you ask to see the boundary
  expect_true(all(is.nan(elastix_xform(bspline, far, out_of_bounds = "nan"))))
})

test_that("the inverse is forward-consistent", {
  # The guarantee: whatever comes back really is a preimage. (Round-trip identity is NOT
  # guaranteed -- the forward map is not injective. See `elastix_xform_inv`.)
  fwd <- elastix_xform(bspline, pts)
  back <- elastix_xform_inv(bspline, fwd)
  expect_true(all(is.finite(back)))
  expect_equal(elastix_xform(bspline, back), fwd, tolerance = 1e-6)
})

test_that("an initial guess pins down the answer", {
  fwd <- elastix_xform(bspline, pts)
  expect_equal(elastix_xform_inv(bspline, fwd, initial_guess = pts), pts, tolerance = 1e-6)
})

test_that("Add chains cannot be inverted", {
  # T(x) = T0(x) + T1(x) - x does not decompose into invertible hops. We refuse.
  add <- elastix_read(ex("add.txt"))
  expect_error(elastix_xform_inv(add, pts), "cannot be inverted")
  expect_true(all(is.finite(elastix_xform(add, pts))))  # ...but it still goes forwards
  expect_true(bspline$ptr$invertible())
})

test_that("chains and the invert flag", {
  path <- ex("affine.txt")
  one <- elastix_read(path)
  two <- elastix_read(c(path, path))
  expect_length(elastix_kinds(two), 2)
  expect_equal(elastix_xform(two, pts), elastix_xform(one, elastix_xform(one, pts)),
               tolerance = 1e-9)

  # traversing a transform backwards agrees exactly with inverting it -- from the same parse
  fwd <- elastix_xform(bspline, pts)
  expect_equal(elastix_xform(bspline, fwd, invert = TRUE), elastix_xform_inv(bspline, fwd))
})

test_that("the initial transform resolves from the file's own directory", {
  # `bspline.txt` names `affine.txt` by a bare relative filename.
  expect_equal(elastix_kinds(bspline), "linear+bspline")
  expect_equal(dim(elastix_affine(bspline)), c(4L, 4L))
  expect_equal(as.vector(elastix_grid_size(bspline)), c(10, 10, 8))
  expect_equal(as.vector(elastix_grid_spacing(bspline)), c(14, 14, 13))
})

test_that("input coercion", {
  expect_equal(elastix_xform(bspline, c(30, 25, 20)), elastix_xform(bspline, rbind(c(30, 25, 20))))
  expect_equal(elastix_xform(bspline, as.data.frame(pts)), elastix_xform(bspline, pts))
})

test_that("n_cores does not change the answer", {
  expect_equal(elastix_xform(bspline, pts, n_cores = 1), elastix_xform(bspline, pts))
})

test_that("errors are informative", {
  expect_error(elastix_read("/nonexistent/nope.txt"), "no Elastix transform")
  expect_error(elastix_xform(bspline, pts, invert = c(TRUE, FALSE)), "one flag per transform")
  expect_error(elastix_xform(bspline, matrix(1, 2, 2)), "\\(N, 3\\)")
  expect_error(elastix_xform(bspline, pts, out_of_bounds = "bogus"))
  expect_error(elastix_xform(42, pts), "elastix_transform")
  expect_error(elastix_xform_inv(bspline, pts, initial_guess = pts[1:2, ]),
               "one point per input point")
  expect_output(print(bspline), "elastix_transform")
})

test_that("the header-only probe agrees with a full parse", {
  # That agreement is the probe's whole contract: it reads the same chain and skips only the
  # coefficients, so it must never give a different answer.
  for (name in c("affine.txt", "euler.txt", "similarity.txt", "bspline.txt", "add.txt")) {
    probed <- elastix_probe_invertible(ex(name))
    parsed <- elastix_read(ex(name))$ptr$invertible()
    expect_equal(probed, parsed, info = name)
  }

  expect_true(elastix_probe_invertible(ex("bspline.txt")))
  expect_false(elastix_probe_invertible(ex("add.txt")))
  expect_error(elastix_probe_invertible("/nonexistent/nope.txt"), "no Elastix transform")
})

test_that("a stale absolute initial-transform path falls back to its basename", {
  # Elastix records the path as it was on the author's machine. `copy_files` in navis is
  # precisely this workaround, in disguise.
  d <- tempfile()
  dir.create(d)
  on.exit(unlink(d, recursive = TRUE))
  writeLines(
    c('(Transform "TranslationTransform")', "(TransformParameters 1 2 3)"),
    file.path(d, "parent.txt")
  )
  writeLines(
    c(
      '(Transform "TranslationTransform")', "(TransformParameters 10 20 30)",
      '(InitialTransformParametersFileName "/nowhere/on/this/machine/parent.txt")'
    ),
    file.path(d, "child.txt")
  )

  xf <- elastix_read(file.path(d, "child.txt"))
  expect_equal(elastix_kinds(xf), "linear+linear")
  expect_equal(elastix_xform(xf, rbind(c(0, 0, 0))), rbind(c(11, 22, 33)))
  expect_true(elastix_probe_invertible(file.path(d, "child.txt")))
})
