# Landmark-based spatial transforms: thin-plate spline and moving least squares.

# Validate a landmark pair R-side and return them as (M, 3) doubles.
#
# extendr 0.7 loses a panic's payload when it is raised from an *associated* function, so a
# constructor that failed in Rust would reach R as "User function panicked: fit". Everything
# we can check up front therefore gets checked here; the Rust side reports what it can only
# discover by trying (a singular system) through `error()`.
.warp_landmarks <- function(source, target, min_n = 1L, what = "landmark") {
  src <- .xform_xyz(source, "source")
  trg <- .xform_xyz(target, "target")

  if (nrow(src) != nrow(trg)) {
    stop(sprintf(
      "number of source landmarks (%d) must match number of target landmarks (%d)",
      nrow(src), nrow(trg)
    ))
  }
  if (nrow(src) < min_n) {
    stop(sprintf(
      "need at least %d %ss, got %d", min_n, what, nrow(src)
    ))
  }
  if (!all(is.finite(src)) || !all(is.finite(trg))) {
    stop("landmarks must not contain NA, NaN or Inf")
  }
  list(source = src, target = trg)
}

# Raise the Rust-side failure, if any, as a real R condition.
.warp_check <- function(ptr) {
  err <- ptr$error()
  if (nzchar(err)) stop(err, call. = FALSE)
  ptr
}

.tps_ptr <- function(reg) {
  if (!inherits(reg, "tps_transform")) {
    stop("`reg` must be a <tps_transform>, as returned by `tps_transform()`")
  }
  reg$ptr
}

.mls_ptr <- function(reg) {
  if (!inherits(reg, "mls_transform")) {
    stop("`reg` must be an <mls_transform>, as returned by `mls_transform()`")
  }
  reg$ptr
}

#' Fit a thin-plate spline to landmark pairs
#'
#' A thin-plate spline interpolates the source landmarks onto the target landmarks exactly,
#' and between them minimises the integral bending norm -- the smoothest warp consistent
#' with the landmarks. This is the usual fallback when no image registration exists for a
#' pair of template spaces, and what `nat`'s `tpsreg` provides.
#'
#' The fit happens once, here, and the resulting object can then be applied as often as you
#' like with [tps_xform()]. That split matters: the fit is *cubic* in the landmark count
#' while applying it is linear, so refitting per call would dominate any real workload.
#'
#' There is no `batch_size` to tune, unlike the `navis` implementation this mirrors. The
#' `(n_points, n_landmarks)` distance matrix is fused into the accumulation rather than
#' built, so peak memory is the output matrix regardless of how many points or landmarks are
#' involved.
#'
#' @param source An `(M, 3)` matrix of source landmarks. A data frame is also accepted -- if
#'   it has x/y/z columns (in any case) those are used, whatever else it carries, otherwise
#'   the first three columns.
#' @param target An `(M, 3)` matrix of target landmarks, one per source landmark.
#'
#' @return An object of class `tps_transform`.
#'
#' @section Inverting:
#' A thin-plate spline has no closed-form inverse. To map the other way, fit the other way
#' round -- `tps_transform(target, source)` -- which is a fresh fit, not an inversion.
#'
#' @examples
#' src <- rbind(c(0, 0, 0), c(10, 10, 10), c(100, 100, 100), c(80, 10, 30))
#' trg <- rbind(c(1, 15, 5), c(9, 18, 21), c(80, 99, 120), c(5, 10, 80))
#' tps <- tps_transform(src, trg)
#' tps
#'
#' # landmarks are reproduced exactly
#' stopifnot(all.equal(tps_xform(tps, src), trg))
#'
#' tps_xform(tps, rbind(c(0, 0, 0), c(50, 50, 50)))
#'
#' @seealso [tps_xform()], [mls_transform()]
#' @export
tps_transform <- function(source, target) {
  lm <- .warp_landmarks(source, target, min_n = 4L, what = "landmark")
  ptr <- .warp_check(TpsTransformPtr$fit(lm$source, lm$target))
  structure(
    list(ptr = ptr, source = lm$source, target = lm$target),
    class = "tps_transform"
  )
}

#' @export
print.tps_transform <- function(x, ...) {
  cat(sprintf("<tps_transform> %d landmarks\n", x$ptr$n_landmarks()))
  invisible(x)
}

#' Apply a thin-plate spline to points
#'
#' @param reg A `tps_transform`, from [tps_transform()].
#' @param xyz An `(N, 3)` matrix of coordinates. A data frame is also accepted -- if it has
#'   x/y/z columns (in any case) those are used, whatever else it carries, otherwise the
#'   first three columns -- as is a bare length-3 vector. `nat::xyzmatrix()` gives you this
#'   for a neuron.
#' @param n_cores Number of threads. `NULL` (default) uses all cores.
#'
#' @return An `(N, 3)` matrix of transformed coordinates.
#'
#' @examples
#' src <- rbind(c(0, 0, 0), c(10, 10, 10), c(100, 100, 100), c(80, 10, 30))
#' trg <- rbind(c(1, 15, 5), c(9, 18, 21), c(80, 99, 120), c(5, 10, 80))
#' tps <- tps_transform(src, trg)
#' tps_xform(tps, c(50, 50, 50))
#'
#' \dontrun{
#' # transform a neuron
#' n <- nat::Cell07PNs[[1]]
#' nat::xyzmatrix(n) <- tps_xform(tps, nat::xyzmatrix(n))
#' }
#'
#' @seealso [tps_transform()], [tps_affine()]
#' @export
tps_xform <- function(reg, xyz, n_cores = NULL) {
  .tps_ptr(reg)$xform(
    .xform_xyz(xyz),
    if (is.null(n_cores)) NULL else as.integer(n_cores)
  )
}

#' The affine part of a thin-plate spline
#'
#' A thin-plate spline decomposes into an affine part and a non-affine ("bending") part.
#' This returns the affine, which is what the spline converges to far from the landmarks.
#'
#' @param reg A `tps_transform`, from [tps_transform()].
#'
#' @return A `(4, 4)` homogeneous transformation matrix, with last row `c(0, 0, 0, 1)`.
#'
#' @examples
#' # landmarks related by a pure affine give back exactly that affine
#' src <- rbind(c(0, 0, 0), c(1, 0, 0), c(0, 1, 0), c(0, 0, 1), c(1, 1, 1))
#' trg <- src * 2 + matrix(c(3, 4, 5), nrow(src), 3, byrow = TRUE)
#' tps_affine(tps_transform(src, trg))
#'
#' @seealso [tps_transform()], [mls_affine()]
#' @export
tps_affine <- function(reg) {
  .tps_ptr(reg)$matrix_affine()
}

#' Coefficients of a fitted thin-plate spline
#'
#' Returns the fit itself: `W`, the weights of the non-affine part, and `A`, the
#' coefficients of the affine part. Together with the source landmarks these fully determine
#' the transform, so they are what you serialise if you want to store a fit rather than
#' repeat it.
#'
#' @param reg A `tps_transform`, from [tps_transform()].
#'
#' @return A list with elements `source` (`(M, 3)`), `W` (`(M, 3)`) and `A` (`(4, 3)`, whose
#'   first row is the translation).
#'
#' @examples
#' src <- rbind(c(0, 0, 0), c(10, 10, 10), c(100, 100, 100), c(80, 10, 30))
#' trg <- rbind(c(1, 15, 5), c(9, 18, 21), c(80, 99, 120), c(5, 10, 80))
#' co <- tps_coefs(tps_transform(src, trg))
#' dim(co$W)
#' dim(co$A)
#'
#' @seealso [tps_transform()]
#' @export
tps_coefs <- function(reg) {
  ptr <- .tps_ptr(reg)
  list(source = ptr$source(), W = ptr$weights(), A = ptr$affine_coefs())
}

#' Build a moving least squares transform from landmark pairs
#'
#' The affine flavour of the algorithm published in Schaefer et al. (2006). Unlike a
#' thin-plate spline there is no fit: every point gets its *own* affine, solved on the fly
#' from all landmarks weighted by inverse squared distance. Construction is therefore free
#' and [mls_xform()] is the entire cost.
#'
#' Results are similar to but not identical with [tps_transform()]; which suits a given
#' pair of spaces is worth checking empirically.
#'
#' @param source An `(M, 3)` matrix of source landmarks. A data frame is also accepted -- if
#'   it has x/y/z columns (in any case) those are used, whatever else it carries, otherwise
#'   the first three columns.
#' @param target An `(M, 3)` matrix of target landmarks, one per source landmark.
#' @param direction `"forward"` (default), or `"inverse"` to treat the target as the source
#'   and vice versa. Note this fits the warp in the opposite direction; it is not an exact
#'   inverse, which moving least squares does not have. It is free, though -- no refit.
#'
#' @return An object of class `mls_transform`.
#'
#' @references
#' Schaefer S, McPhail T, Warren J (2006). "Image deformation using moving least squares."
#' *ACM Transactions on Graphics* 25(3):533-540. \doi{10.1145/1141911.1141920}
#'
#' @examples
#' src <- rbind(c(0, 0, 0), c(10, 10, 10), c(100, 100, 100), c(80, 10, 30))
#' trg <- rbind(c(1, 15, 5), c(9, 18, 21), c(80, 99, 120), c(5, 10, 80))
#' mls <- mls_transform(src, trg)
#' mls
#'
#' # a point sitting on a landmark maps to its partner
#' stopifnot(all.equal(mls_xform(mls, src), trg, tolerance = 1e-6))
#'
#' # and the inverse direction maps them back
#' back <- mls_transform(src, trg, direction = "inverse")
#' stopifnot(all.equal(mls_xform(back, trg), src, tolerance = 1e-6))
#'
#' @seealso [mls_xform()], [tps_transform()]
#' @export
mls_transform <- function(source, target, direction = c("forward", "inverse")) {
  direction <- match.arg(direction)
  lm <- .warp_landmarks(source, target, min_n = 1L, what = "landmark")
  ptr <- .warp_check(MlsTransformPtr$build(lm$source, lm$target))
  structure(
    list(
      ptr = ptr, source = lm$source, target = lm$target,
      direction = direction, reverse = direction == "inverse"
    ),
    class = "mls_transform"
  )
}

#' @export
print.mls_transform <- function(x, ...) {
  cat(sprintf(
    "<mls_transform> %d landmarks, direction '%s'\n",
    x$ptr$n_landmarks(), x$direction
  ))
  invisible(x)
}

#' Apply a moving least squares transform to points
#'
#' @param reg An `mls_transform`, from [mls_transform()].
#' @param xyz An `(N, 3)` matrix of coordinates. A data frame is also accepted -- if it has
#'   x/y/z columns (in any case) those are used, whatever else it carries, otherwise the
#'   first three columns -- as is a bare length-3 vector. `nat::xyzmatrix()` gives you this
#'   for a neuron.
#' @param n_cores Number of threads. `NULL` (default) uses all cores.
#'
#' @return An `(N, 3)` matrix of transformed coordinates.
#'
#' @examples
#' src <- rbind(c(0, 0, 0), c(10, 10, 10), c(100, 100, 100), c(80, 10, 30))
#' trg <- rbind(c(1, 15, 5), c(9, 18, 21), c(80, 99, 120), c(5, 10, 80))
#' mls <- mls_transform(src, trg)
#' mls_xform(mls, c(50, 50, 50))
#'
#' @seealso [mls_transform()], [mls_affine()]
#' @export
mls_xform <- function(reg, xyz, n_cores = NULL) {
  .mls_ptr(reg)$xform(
    .xform_xyz(xyz),
    isTRUE(reg$reverse),
    if (is.null(n_cores)) NULL else as.integer(n_cores)
  )
}

#' The global affine of a moving least squares transform
#'
#' Moving least squares is *locally* weighted -- every point effectively gets its own affine
#' -- so no single matrix describes it. This returns the *global* affine: the least-squares
#' fit of the source landmarks onto the target landmarks, which is what the warp converges
#' to far from them, where the distance weights even out.
#'
#' @param reg An `mls_transform`, from [mls_transform()].
#'
#' @return A `(4, 4)` homogeneous transformation matrix, with last row `c(0, 0, 0, 1)`.
#'
#' @examples
#' src <- rbind(c(0, 0, 0), c(1, 0, 0), c(0, 1, 0), c(0, 0, 1), c(1, 1, 1))
#' trg <- src * 2 + matrix(c(3, 4, 5), nrow(src), 3, byrow = TRUE)
#' mls_affine(mls_transform(src, trg))
#'
#' @seealso [mls_transform()], [tps_affine()]
#' @export
mls_affine <- function(reg) {
  .mls_ptr(reg)$matrix_affine(isTRUE(reg$reverse))
}
