# Landmark transforms: thin-plate spline and moving least squares.
#
# There is no external binary to check against here (unlike CMTK/Elastix), so these pin the
# maths on properties that hold for any correct implementation: both transforms interpolate
# their landmarks, and both reproduce a global affine exactly when the landmarks are related
# by one. The Python test suite additionally checks us against `morphops` and `molesq`, the
# reference implementations `navis` delegates to.

set.seed(42)

# A non-degenerate landmark pair with a genuinely non-affine component.
n_lm <- 40L
src <- matrix(runif(n_lm * 3, 0, 100), ncol = 3)
trg <- src %*% diag(c(1.3, 0.8, 1.1)) +
  matrix(c(5, -3, 12), n_lm, 3, byrow = TRUE) +
  matrix(rnorm(n_lm * 3, 0, 4), ncol = 3)

pts <- matrix(runif(200 * 3, 0, 100), ncol = 3)

# Landmarks related by a pure affine, for the exact-recovery checks.
aff_linear <- rbind(c(2, 0.1, 0), c(0, -1, 0.25), c(0.3, 0, 0.75))
aff_offset <- c(3, 4, 5)
src_aff <- matrix(runif(20 * 3, 0, 10), ncol = 3)
apply_affine <- function(m) m %*% aff_linear + matrix(aff_offset, nrow(m), 3, byrow = TRUE)
trg_aff <- apply_affine(src_aff)


# ---------------------------------------------------------------------------
# Thin-plate spline
# ---------------------------------------------------------------------------

test_that("tps interpolates its landmarks", {
  tps <- tps_transform(src, trg)
  expect_equal(tps_xform(tps, src), trg, tolerance = 1e-6)
})

test_that("tps recovers a global affine, including far outside the landmark hull", {
  tps <- tps_transform(src_aff, trg_aff)
  far <- matrix(runif(50 * 3, -500, 500), ncol = 3)
  expect_equal(tps_xform(tps, far), apply_affine(far), tolerance = 1e-6)
})

test_that("tps_affine is the homogeneous form of that affine", {
  m <- tps_affine(tps_transform(src_aff, trg_aff))
  expect_equal(dim(m), c(4L, 4L))
  expect_equal(m[4, ], c(0, 0, 0, 1))
  # The linear part is applied as a row-vector product, so the matrix form is its transpose.
  expect_equal(m[1:3, 1:3], t(aff_linear), tolerance = 1e-8)
  expect_equal(m[1:3, 4], aff_offset, tolerance = 1e-8)
})

test_that("fitting the other way round inverts the mapping", {
  back <- tps_transform(trg, src)
  expect_equal(tps_xform(back, trg), src, tolerance = 1e-6)
})

test_that("tps_coefs returns the fit", {
  co <- tps_coefs(tps_transform(src, trg))
  expect_equal(dim(co$source), c(n_lm, 3L))
  expect_equal(dim(co$W), c(n_lm, 3L))
  expect_equal(dim(co$A), c(4L, 3L))
  expect_equal(co$source, src)
})

test_that("a singular system is an R error, not a panic or nonsense", {
  # Duplicate source landmarks pulled to different targets cannot be interpolated.
  bad_src <- rbind(c(0, 0, 0), c(0, 0, 0), c(1, 0, 0), c(0, 1, 0), c(0, 0, 1))
  bad_trg <- rbind(c(0, 0, 0), c(5, 0, 0), c(1, 0, 0), c(0, 1, 0), c(0, 0, 1))
  expect_error(tps_transform(bad_src, bad_trg), "singular")
})

test_that("tps needs at least four landmarks", {
  p <- rbind(c(0, 0, 0), c(1, 0, 0), c(0, 1, 0))
  expect_error(tps_transform(p, p), "at least 4")
})

test_that("tps print method reports the landmark count", {
  expect_output(print(tps_transform(src, trg)), "40 landmarks")
})


# ---------------------------------------------------------------------------
# Moving least squares
# ---------------------------------------------------------------------------

test_that("mls interpolates its landmarks", {
  mls <- mls_transform(src, trg)
  expect_equal(mls_xform(mls, src), trg, tolerance = 1e-6)
})

test_that("mls recovers a global affine, including far outside the landmark hull", {
  mls <- mls_transform(src_aff, trg_aff)
  far <- matrix(runif(50 * 3, -500, 500), ncol = 3)
  expect_equal(mls_xform(mls, far), apply_affine(far), tolerance = 1e-6)
})

test_that("mls_affine is the homogeneous form of that affine", {
  m <- mls_affine(mls_transform(src_aff, trg_aff))
  expect_equal(dim(m), c(4L, 4L))
  expect_equal(m[4, ], c(0, 0, 0, 1))
  expect_equal(m[1:3, 1:3], t(aff_linear), tolerance = 1e-8)
  expect_equal(m[1:3, 4], aff_offset, tolerance = 1e-8)
})

test_that("direction = 'inverse' maps targets back to sources", {
  back <- mls_transform(src, trg, direction = "inverse")
  expect_equal(mls_xform(back, trg), src, tolerance = 1e-6)
  expect_equal(back$direction, "inverse")
})

test_that("inverting is not the same transform as the forward one", {
  fwd <- mls_transform(src, trg)
  inv <- mls_transform(src, trg, direction = "inverse")
  expect_false(isTRUE(all.equal(mls_xform(fwd, pts), mls_xform(inv, pts))))
})

test_that("mls rejects an unknown direction", {
  expect_error(mls_transform(src, trg, direction = "sideways"))
})

test_that("mls print method reports landmarks and direction", {
  expect_output(print(mls_transform(src, trg)), "40 landmarks, direction 'forward'")
})

test_that("mls handles landmark counts the reference implementation cannot", {
  # molesq builds (3, M, N) intermediates; this shape is out of reach for it but is only
  # the output matrix here.
  big_src <- matrix(runif(2000 * 3, 0, 1000), ncol = 3)
  big_trg <- big_src + matrix(rnorm(2000 * 3, 0, 5), ncol = 3)
  out <- mls_xform(mls_transform(big_src, big_trg), pts)
  expect_equal(dim(out), dim(pts))
  expect_true(all(is.finite(out)))
})


# ---------------------------------------------------------------------------
# Shared input handling
# ---------------------------------------------------------------------------

builders <- list(
  tps = list(build = tps_transform, xform = tps_xform),
  mls = list(build = mls_transform, xform = mls_xform)
)

for (nm in names(builders)) {
  b <- builders[[nm]]

  test_that(paste(nm, "accepts a length-3 vector and data frames"), {
    tr <- b$build(src, trg)
    expect_equal(b$xform(tr, c(50, 50, 50)), b$xform(tr, matrix(c(50, 50, 50), ncol = 3)))

    df_src <- as.data.frame(src)
    names(df_src) <- c("x", "y", "z")
    df_trg <- as.data.frame(trg)
    names(df_trg) <- c("X", "Y", "Z") # capitalised, as `nat` writes them
    expect_equal(b$xform(b$build(df_src, df_trg), pts), b$xform(tr, pts))
  })

  test_that(paste(nm, "rejects mismatched landmark counts"), {
    expect_error(b$build(src, trg[-1, ]), "must match")
  })

  test_that(paste(nm, "rejects non-3-column input"), {
    expect_error(b$build(src[, 1:2], trg[, 1:2]), "\\(N, 3\\)")
  })

  test_that(paste(nm, "rejects non-finite landmarks"), {
    bad <- src
    bad[1, 1] <- NA
    expect_error(b$build(bad, trg), "NA, NaN or Inf")
  })

  test_that(paste(nm, "rejects a foreign object"), {
    expect_error(b$xform(list(), pts), "must be")
  })

  test_that(paste(nm, "thread count does not change the result"), {
    tr <- b$build(src, trg)
    expect_identical(b$xform(tr, pts, n_cores = 1), b$xform(tr, pts))
    expect_identical(b$xform(tr, pts, n_cores = 4), b$xform(tr, pts))
  })
}
