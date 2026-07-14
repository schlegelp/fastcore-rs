#' Read a CMTK registration
#'
#' Reads a CMTK registration (a 12-DOF affine plus, usually, a cubic B-spline warp)
#' ready to be applied to points with [cmtk_xform()] and [cmtk_xform_inv()].
#'
#' Unlike `nat`'s CMTK support, this does **not** shell out to CMTK's `streamxform`
#' binary: CMTK does not need to be installed. Results match `streamxform` to ~5e-7 --
#' its own printing precision -- including which points it declines to transform.
#'
#' The registration is parsed once, here, and can then be applied as often as you like.
#'
#' @param path Path to a CMTK `.list` registration directory, or to a `registration`
#'   file itself (plain or gzipped). Pass a character vector of several to build a
#'   chain, applied in order: `points -> path[1] -> path[2] -> ... -> output`.
#'
#' Direction is **not** fixed here. It is chosen per call -- see the `invert` argument of
#' [cmtk_xform()], and [cmtk_xform_inv()] -- so one object serves every direction and the
#' registration is parsed only once.
#'
#' @return An object of class `cmtk_registration`.
#'
#' @examples
#' \dontrun{
#' reg <- cmtk_read("JFRC2_FCWB.list")
#' reg
#'
#' # a chain: A -> B -> C, where the second registration is stored as C->B
#' chain <- cmtk_read(c("A_B.list", "C_B.list"))
#' cmtk_xform(chain, pts, invert = c(FALSE, TRUE))
#' }
#'
#' @seealso [cmtk_xform()], [cmtk_xform_inv()]
#' @export
cmtk_read <- function(path) {
  path <- as.character(path)
  if (!length(path)) stop("`path` must name at least one registration")

  # Resolve the paths here rather than in Rust. extendr cannot raise an R condition from a
  # constructor -- a panic there loses its message and R only sees "User function panicked" --
  # so the common failures (missing path, a `.list` dir with no registration in it) have to be
  # caught on this side to produce a message worth reading.
  for (p in path) {
    if (dir.exists(p)) {
      if (!any(file.exists(file.path(p, c("registration", "registration.gz"))))) {
        stop(sprintf(
          "'%s' is a directory but holds neither `registration` nor `registration.gz`", p
        ))
      }
    } else if (!file.exists(p)) {
      stop(sprintf("no CMTK registration at '%s'", p))
    }
  }

  ptr <- CmtkRegistration$load(path)
  structure(list(ptr = ptr, path = path), class = "cmtk_registration")
}

#' @export
print.cmtk_registration <- function(x, ...) {
  kind <- ifelse(x$ptr$has_spline(), "warp", "affine")
  cat(sprintf(
    "<cmtk_registration> %d registration%s: %s\n",
    length(x$path), if (length(x$path) == 1L) "" else "s",
    paste(kind, collapse = " -> ")
  ))
  for (i in seq_along(x$path)) {
    cat(sprintf("  [%d] %s\n", i, x$path[i]))
  }
  invisible(x)
}

# Coerce whatever the user passed into an (N, 3) numeric matrix.
.cmtk_xyz <- function(xyz, arg = "xyz") {
  if (is.data.frame(xyz)) xyz <- as.matrix(xyz)
  if (is.vector(xyz) && length(xyz) == 3L) xyz <- matrix(xyz, ncol = 3L)
  if (!is.matrix(xyz) || ncol(xyz) != 3L) {
    stop(sprintf("`%s` must be an (N, 3) matrix of 3D coordinates", arg))
  }
  storage.mode(xyz) <- "double"
  xyz
}

.cmtk_ptr <- function(reg) {
  if (!inherits(reg, "cmtk_registration")) {
    stop("`reg` must be a <cmtk_registration>, as returned by `cmtk_read()`")
  }
  reg$ptr
}

#' Apply a CMTK registration to points
#'
#' @param reg A `cmtk_registration`, from [cmtk_read()].
#' @param xyz An `(N, 3)` matrix of coordinates (a data frame or a length-3 vector is
#'   also accepted). `nat::xyzmatrix()` gives you this for a neuron.
#' @param transform `"warp"` (default) applies the full transformation; `"affine"`
#'   applies only the affine component. A registration with no spline warp uses its
#'   affine either way.
#' @param allow_extrapolation Evaluate points outside the registration's domain by
#'   clamping to the outermost control points, instead of failing them.
#'
#'   Defaults to `FALSE`, **which is what CMTK does**: `streamxform` reports a point
#'   outside the domain as `FAILED`, and we return `NaN`. Setting this to `TRUE` gives
#'   every point *an* answer, but that answer extrapolates a warp which was never fitted
#'   there, and it will silently disagree with every other CMTK-based tool.
#' @param fallback_to_affine Replace failed rows with the affine result rather than `NaN`.
#'   One of `FALSE` (the default), `TRUE`/`"chain"`, or `"hop"`.
#'
#'   `TRUE` (or `"chain"`) re-runs the **whole chain** affine-only, starting again from the
#'   *original* point. This is what `nat`/`navis` do -- they hand the failed rows back to
#'   `streamxform --affine-only` over the same registration list -- so it is the default
#'   whenever a fallback is asked for.
#'
#'   `"hop"` instead swaps the affine in for **only the hop that failed**, keeping the warps of
#'   the hops that succeeded. Arguably the better answer, but a silent departure from every
#'   other CMTK-based tool, so you have to ask for it. On a single registration the two are
#'   identical; on a chain they differ by a median of ~6 world units.
#'
#'   Works in both directions: a hop travelled backwards falls back to the *inverse* affine,
#'   so the rescued point still lands in the space you asked for.
#' @param invert Logical, length 1 or one flag per registration. `TRUE` traverses that
#'   registration backwards -- what you need when routing through a bridging graph, where an
#'   edge may be walked in either direction, and a chain may need some hops forwards and
#'   others backwards.
#'
#'   This is **not** the same as [cmtk_xform_inv()], which inverts the whole composition
#'   (reversing the order *and* flipping every hop). For a single registration the two agree;
#'   for a chain they do not, and only `invert` can express a mixed-direction traversal.
#' @param n_cores Number of threads. `NULL` (default) uses all cores.
#' @param progress Show a progress bar.
#'
#' @return An `(N, 3)` matrix. Rows that could not be transformed are `NaN`.
#'
#' @examples
#' \dontrun{
#' reg <- cmtk_read("JFRC2_FCWB.list")
#' pts <- rbind(c(50, 50, 50), c(100, 100, 20))
#' cmtk_xform(reg, pts)
#'
#' # transform a neuron
#' n <- nat::Cell07PNs[[1]]
#' nat::xyzmatrix(n) <- cmtk_xform(reg, nat::xyzmatrix(n))
#' }
#'
#' @seealso [cmtk_read()], [cmtk_xform_inv()]
#' @export
cmtk_xform <- function(reg, xyz, transform = c("warp", "affine"),
                       allow_extrapolation = FALSE, fallback_to_affine = FALSE,
                       invert = FALSE, n_cores = NULL, progress = FALSE) {
  transform <- match.arg(transform)
  .cmtk_ptr(reg)$xform(
    .cmtk_xyz(xyz), transform,
    isTRUE(allow_extrapolation), .fallback_mode(fallback_to_affine),
    .invert_flags(invert, length(reg$path), "registration"),
    if (is.null(n_cores)) NULL else as.integer(n_cores),
    isTRUE(progress)
  )
}

#' Apply a CMTK registration to points, backwards
#'
#' The affine part is inverted exactly. The spline warp has no closed-form inverse and
#' is solved per point by damped Gauss-Newton against the analytic Jacobian.
#'
#' Points whose inverse does not converge come back as `NaN`. This is deliberate and
#' matches CMTK, which reports such points as `FAILED` -- a registration is only defined
#' over a finite domain, and some points simply have no preimage inside it.
#'
#' @inheritParams cmtk_xform
#' @param initial_guess An `(N, 3)` matrix of starting points for the solver. Defaults to
#'   `xyz` itself, which is a good guess for any well-behaved registration. In a chain,
#'   this seeds only the first solve.
#' @param max_iter Maximum Gauss-Newton iterations per point.
#' @param tolerance Stop once the step falls below this.
#' @param accuracy Accept a solution only if its residual is within this of the target;
#'   otherwise the row is `NaN`.
#' @param clamp_to_domain Confine the iterate to the registration's domain box.
#'
#'   **This is what makes the result agree with CMTK.** Turning it off finds preimages
#'   that lie outside the image domain, where `streamxform` reports failure -- so you
#'   will get finite numbers where CMTK gives you none.
#' @param fallback_to_affine Replace rows the solver could not land with the *inverse* affine
#'   result rather than `NaN` -- the mirror of the same argument on [cmtk_xform()], with the
#'   same `"chain"` (default) and `"hop"` semantics.
#' @param invert The same per-hop flags as on [cmtk_xform()], composed with this whole-chain
#'   inversion: hop `i` runs *forwards* here exactly when `invert[i]` is `TRUE`.
#'
#' @return An `(N, 3)` matrix. Rows that did not converge are `NaN`.
#'
#' @examples
#' \dontrun{
#' reg <- cmtk_read("JFRC2_FCWB.list")
#' pts <- rbind(c(50, 50, 50), c(100, 100, 20))
#' back <- cmtk_xform_inv(reg, cmtk_xform(reg, pts))
#' stopifnot(all.equal(back, pts, tolerance = 1e-4))
#' }
#'
#' @seealso [cmtk_read()], [cmtk_xform()]
#' @export
cmtk_xform_inv <- function(reg, xyz, transform = c("warp", "affine"),
                           initial_guess = NULL, max_iter = 50L, tolerance = 1e-9,
                           accuracy = 1e-3, clamp_to_domain = TRUE,
                           fallback_to_affine = FALSE, invert = FALSE,
                           n_cores = NULL, progress = FALSE) {
  transform <- match.arg(transform)
  xyz <- .cmtk_xyz(xyz)
  if (!is.null(initial_guess)) {
    initial_guess <- .cmtk_xyz(initial_guess, "initial_guess")
    if (nrow(initial_guess) != nrow(xyz)) {
      stop(sprintf(
        "`initial_guess` must have one point per input point: expected %d, got %d",
        nrow(xyz), nrow(initial_guess)
      ))
    }
  }
  .cmtk_ptr(reg)$xform_inv(
    xyz, transform, initial_guess,
    as.integer(max_iter), as.double(tolerance), as.double(accuracy),
    isTRUE(clamp_to_domain), .fallback_mode(fallback_to_affine),
    .invert_flags(invert, length(reg$path), "registration"),
    if (is.null(n_cores)) NULL else as.integer(n_cores),
    isTRUE(progress)
  )
}

#' Properties of a CMTK registration
#'
#' @param reg A `cmtk_registration`, from [cmtk_read()].
#'
#' @return
#' * `cmtk_affine()`: the 4x4 affine matrix of the first registration, or `NULL`.
#' * `cmtk_domain()`: a `(k, 3)` matrix, one row per spline warp. Points outside
#'   `[0, domain]` cannot be transformed -- CMTK reports them as FAILED.
#' * `cmtk_dims()` / `cmtk_spacing()`: `(k, 3)` matrices of the control-point lattice
#'   dimensions and spacing.
#' * `cmtk_versions()`: the CMTK TypedStream version of each registration.
#'
#' @examples
#' \dontrun{
#' reg <- cmtk_read("JFRC2_FCWB.list")
#' cmtk_affine(reg)
#' cmtk_domain(reg)
#' }
#'
#' @name cmtk_properties
#' @export
cmtk_affine <- function(reg) .cmtk_ptr(reg)$affine()

#' @rdname cmtk_properties
#' @export
cmtk_domain <- function(reg) .cmtk_ptr(reg)$domain()

#' @rdname cmtk_properties
#' @export
cmtk_dims <- function(reg) .cmtk_ptr(reg)$dims()

#' @rdname cmtk_properties
#' @export
cmtk_spacing <- function(reg) .cmtk_ptr(reg)$spacing()

#' @rdname cmtk_properties
#' @export
cmtk_versions <- function(reg) .cmtk_ptr(reg)$versions()
