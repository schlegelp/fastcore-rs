# CMTK transforms, checked against CMTK's own `streamxform` output.
#
# The ground truth in `streamxform_golden.csv` was produced by the CMTK binary itself, so
# these are not self-consistency checks: they pin us to CMTK's behaviour, including its habit
# of *failing* on points it cannot transform (which we return as NaN).

reg_path <- system.file("extdata", "JFRC2_FCWB.list", package = "nat.fastcore")
tiny_path <- system.file("extdata", "tiny_warp.list", package = "nat.fastcore")

golden <- local({
  g <- utils::read.csv(system.file("extdata", "streamxform_golden.csv",
                                   package = "nat.fastcore"))
  split(as.matrix(g[, c("x", "y", "z")]), g$case)[unique(g$case)] |>
    lapply(matrix, ncol = 3)
})

# Measured margin is ~5e-7 (streamxform's own print precision). If a change lands at 1e-5,
# something is subtly wrong -- do not loosen this.
ATOL <- 1e-4

reg <- cmtk_read(reg_path)
pts <- golden$input

test_that("affine and warp match streamxform, forwards", {
  expect_equal(cmtk_xform(reg, pts, transform = "affine"), golden$affine_forward,
               tolerance = ATOL, ignore_attr = TRUE)
  expect_equal(cmtk_xform(reg, pts), golden$warp_forward,
               tolerance = ATOL, ignore_attr = TRUE)
})

test_that("affine and warp match streamxform, backwards", {
  expect_equal(cmtk_xform_inv(reg, pts, transform = "affine"), golden$affine_inverse,
               tolerance = ATOL, ignore_attr = TRUE)
  expect_equal(cmtk_xform_inv(reg, pts), golden$warp_inverse,
               tolerance = ATOL, ignore_attr = TRUE)
})

test_that("points CMTK reports as FAILED come back NaN", {
  # streamxform fails on 2 of the 5 sample points; returning a plausible-looking number
  # instead would silently disagree with every other CMTK-based tool.
  out <- cmtk_xform_inv(reg, pts)
  expect_true(all(is.nan(out[c(1, 5), ])))
  expect_true(all(is.finite(out[2:4, ])))
})

test_that("round-trips", {
  p <- pts[2:4, ]
  expect_equal(cmtk_xform_inv(reg, cmtk_xform(reg, p)), p, tolerance = 1e-4)
  expect_equal(
    cmtk_xform_inv(reg, cmtk_xform(reg, p, transform = "affine"), transform = "affine"),
    p, tolerance = 1e-10
  )
})

test_that("out-of-domain points are NaN by default", {
  far <- rbind(c(-5000, -5000, -5000), c(100, 100, 20))

  strict <- cmtk_xform(reg, far)
  expect_true(all(is.nan(strict[1, ])))
  expect_true(all(is.finite(strict[2, ])))

  # ...unless you opt into extrapolation, which makes you disagree with CMTK
  expect_true(all(is.finite(cmtk_xform(reg, far, allow_extrapolation = TRUE))))

  # ...or fall back to the affine
  filled <- cmtk_xform(reg, far, fallback_to_affine = TRUE)
  expect_equal(filled[1, ], cmtk_xform(reg, far, transform = "affine")[1, ])
})

test_that("input coercion", {
  # a bare length-3 vector, and a data frame, both work
  expect_equal(cmtk_xform(reg, c(50, 50, 50)), cmtk_xform(reg, rbind(c(50, 50, 50))))
  df <- as.data.frame(pts)
  expect_equal(cmtk_xform(reg, df), cmtk_xform(reg, pts))
})

test_that("chains and the invert flag", {
  chain <- cmtk_read(c(reg_path, reg_path))
  expect_length(cmtk_versions(chain), 2)
  p <- pts[2:4, ]
  expect_equal(cmtk_xform(chain, p), cmtk_xform(reg, cmtk_xform(reg, p)))

  # traversing a registration backwards is its inverse
  bwd <- cmtk_read(reg_path, invert = TRUE)
  expect_equal(cmtk_xform(bwd, cmtk_xform(reg, p)), p, tolerance = 1e-4)

  # and inverting the whole chain undoes both hops
  expect_equal(cmtk_xform_inv(chain, cmtk_xform(chain, p)), p, tolerance = 1e-4)
})

test_that("n_cores does not change the answer", {
  expect_equal(cmtk_xform_inv(reg, pts, n_cores = 1), cmtk_xform_inv(reg, pts))
})

test_that("plain and gzipped registrations both load", {
  # JFRC2 is gzipped, the tiny fixture is plain text; gzip is sniffed from magic bytes.
  expect_equal(cmtk_versions(reg), "1.1")
  expect_equal(cmtk_versions(cmtk_read(tiny_path)), "2.4")
  expect_equal(as.vector(cmtk_dims(reg)), c(59, 27, 11))
  expect_equal(as.vector(cmtk_dims(cmtk_read(tiny_path))), c(5, 5, 5))
})

test_that("properties", {
  expect_equal(dim(cmtk_affine(reg)), c(4L, 4L))
  expect_equal(as.vector(cmtk_affine(reg)[4, ]), c(0, 0, 0, 1))
  expect_equal(as.vector(cmtk_spacing(reg)), c(11.3642, 13.2453, 16.8741), tolerance = 1e-4)
  expect_equal(as.vector(cmtk_domain(reg)), c(636.396, 317.887, 134.9931), tolerance = 1e-4)
  expect_output(print(reg), "cmtk_registration")
})

test_that("errors are informative", {
  expect_error(cmtk_read("/nonexistent/nope.list"), "no CMTK registration")
  expect_error(cmtk_read(tempdir()), "neither")
  expect_error(cmtk_read(reg_path, invert = c(TRUE, FALSE)), "one flag per registration")
  expect_error(cmtk_xform(reg, matrix(1, 2, 2)), "\\(N, 3\\)")
  expect_error(cmtk_xform(reg, pts, transform = "bogus"))
  expect_error(cmtk_xform(42, pts), "cmtk_registration")
  expect_error(cmtk_xform_inv(reg, pts, initial_guess = pts[1:2, ]),
               "one point per input point")
})
