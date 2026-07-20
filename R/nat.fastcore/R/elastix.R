#' Read an Elastix transform
#'
#' Reads an Elastix `TransformParameters` file, ready to apply to points. **Elastix itself does
#' not need to be installed** -- there is no call out to `transformix`, no subprocess and no
#' temporary directory.
#'
#' A `TransformParameters` file is already a chain: its `InitialTransformParametersFileName` is
#' followed recursively, resolved relative to *that file's own directory*. So a four-deep
#' affine -> affine -> B-spline -> B-spline stack loads from its outermost file alone.
#'
#' The transform is parsed once and can then be applied any number of times, which is what you
#' want when transforming every neuron in a dataset.
#'
#' @param path Path to a `TransformParameters.*.txt` file. Pass several to build a chain, applied
#'   in order: `xyz -> path[1] -> path[2] -> ... -> output`.
#'
#' Direction is **not** fixed here. It is chosen per call -- see the `invert` argument of
#' [elastix_xform()], and [elastix_xform_inv()] -- so one object serves every direction and the
#' file is parsed only once. That is worth caring about: real warps run to tens of megabytes.
#'
#' @return An object of class `elastix_transform`.
#'
#' @seealso [elastix_xform()], [elastix_xform_inv()]
#'
#' @examples
#' \dontrun{
#' xf <- elastix_read("TransformParameters.FixedFANC.txt")
#' xyzmatrix(n) <- elastix_xform(xf, xyzmatrix(n))
#' }
#'
#' @export
elastix_read <- function(path) {
  path <- as.character(path)
  if (!length(path)) stop("`path` must name at least one transform")

  # Validate here rather than in Rust. extendr cannot raise an R condition from a constructor --
  # a panic there loses its message and R only sees "User function panicked" -- so the common
  # failure (a missing file) has to be caught on this side to produce a message worth reading.
  for (p in path) {
    if (!file.exists(p) || dir.exists(p)) {
      stop(sprintf("no Elastix transform file at '%s'", p))
    }
  }

  ptr <- ElastixTransformPtr$load(path)
  structure(list(ptr = ptr, path = path), class = "elastix_transform")
}

#' @export
print.elastix_transform <- function(x, ...) {
  kinds <- x$ptr$kinds()
  cat(sprintf(
    "<elastix_transform> %d transform%s: %s\n",
    length(kinds), if (length(kinds) == 1L) "" else "s", paste(kinds, collapse = " -> ")
  ))
  for (i in seq_along(x$path)) {
    cat(sprintf("  %s\n", x$path[i]))
  }
  invisible(x)
}

.elastix_ptr <- function(xf) {
  if (!inherits(xf, "elastix_transform")) {
    stop("`xf` must be an <elastix_transform>, as returned by `elastix_read()`")
  }
  xf$ptr
}

#' Apply an Elastix transform to points
#'
#' @param xf An `elastix_transform`, from [elastix_read()].
#' @param xyz An `(N, 3)` matrix of coordinates. A data frame is also accepted -- if it has
#'   x/y/z columns (in any case) those are used, whatever else it carries, otherwise the
#'   first three columns -- as is a bare length-3 vector.
#' @param out_of_bounds What to do with points outside a B-spline's control-point grid.
#'   `"identity"` (the default) returns them **unchanged**, which is exactly what `transformix`
#'   does. `"nan"` returns `NaN` instead.
#'
#'   The default is silent by nature: a neuron straddling the grid edge comes back partly
#'   transformed and looks perfectly fine. Use `"nan"` when you would rather see the boundary
#'   than trust it.
#' @param invert Logical, length 1 or one flag per transform. `TRUE` traverses that transform
#'   backwards -- what you need when routing through a bridging graph, where an edge may be walked
#'   in either direction.
#'
#'   This is **not** the same as [elastix_xform_inv()], which inverts the whole composition
#'   (reversing the order *and* flipping every hop). For a single transform the two agree; for a
#'   chain they do not, and only `invert` can express a mixed-direction traversal.
#' @param n_cores Cap the thread pool. `NULL` uses all cores.
#' @param progress Show a progress bar.
#'
#' @return An `(N, 3)` matrix of transformed coordinates.
#'
#' @seealso [elastix_xform_inv()]
#'
#' @examples
#' \dontrun{
#' xf <- elastix_read("TransformParameters.FixedFANC.txt")
#' elastix_xform(xf, cbind(50, 50, 50))
#' }
#'
#' @export
elastix_xform <- function(xf, xyz, out_of_bounds = c("identity", "nan"),
                          invert = FALSE, n_cores = NULL, progress = FALSE) {
  out_of_bounds <- match.arg(out_of_bounds)
  .elastix_ptr(xf)$xform(
    .xform_xyz(xyz),
    out_of_bounds,
    .invert_flags(invert, length(xf$path), "transform"),
    if (is.null(n_cores)) NULL else as.integer(n_cores),
    isTRUE(progress)
  )
}

#' Invert an Elastix transform on points
#'
#' Elastix itself cannot do this: `transformix` only goes forwards, which is why packages that
#' use Elastix ship two separate registration files per brain pair.
#'
#' Linear steps are inverted exactly. Each B-spline warp has no closed-form inverse and is solved
#' per point by damped Gauss-Newton against the analytic Jacobian.
#'
#' What is guaranteed is **forward-consistency**: `elastix_xform(xf, elastix_xform_inv(xf, y))`
#' returns `y`, to within `accuracy`. What is *not* guaranteed is that inverting a forward
#' transform returns the point you started from, because a B-spline warp need not be injective --
#' a strongly folded registration maps several points to the same place, and no inverse can
#' recover which one you meant. Points with no preimage at all come back as `NaN`.
#'
#' @inheritParams elastix_xform
#' @param initial_guess An `(N, 3)` matrix of starting points for the solver, in the *source*
#'   space. Rarely needed -- the solver seeds itself -- but it is what breaks the tie where a
#'   target has more than one preimage.
#' @param max_iter Solver budget per point.
#' @param seed_iter Rounds of the fixed-point pre-seed. Zero starts the solver at the target,
#'   which loses points wherever the deformation is large.
#' @param tolerance Step-size convergence threshold.
#' @param accuracy Accept a solution only if its residual is within this of the target, in world
#'   units. Otherwise the row is `NaN`.
#' @param lattice_points Size of the global seed lattice -- the last-resort start for the few
#'   points the cheap seeds fail on. Built once per call and consulted only by points that have
#'   already failed. Set to 0 to disable.
#' @param invert The same per-hop flags as on [elastix_xform()], composed with this whole-chain
#'   inversion: hop `i` runs *forwards* here exactly when `invert[i]` is `TRUE`.
#'
#' @return An `(N, 3)` matrix of coordinates in the source space. Rows with no preimage are `NaN`.
#'
#' @seealso [elastix_xform()]
#'
#' @export
elastix_xform_inv <- function(xf, xyz, out_of_bounds = c("identity", "nan"),
                              initial_guess = NULL, max_iter = 50L, seed_iter = 8L,
                              tolerance = 1e-9, accuracy = 1e-3, lattice_points = 16000L,
                              invert = FALSE, n_cores = NULL, progress = FALSE) {
  out_of_bounds <- match.arg(out_of_bounds)
  xyz <- .xform_xyz(xyz)

  # Ask before calling in. extendr cannot carry a Rust panic's message across to R -- it arrives
  # as the useless "User function panicked: xform_inv" -- so this has to be checked here.
  if (!.elastix_ptr(xf)$invertible()) {
    stop(
      "this transform cannot be inverted: it contains an `Add` step ",
      "(HowToCombineTransforms \"Add\"), which does not decompose into invertible hops"
    )
  }

  if (!is.null(initial_guess)) {
    initial_guess <- .xform_xyz(initial_guess, "initial_guess")
    if (nrow(initial_guess) != nrow(xyz)) {
      stop(sprintf(
        "`initial_guess` must have one point per input point: expected %d, got %d",
        nrow(xyz), nrow(initial_guess)
      ))
    }
  }

  .elastix_ptr(xf)$xform_inv(
    xyz,
    out_of_bounds,
    initial_guess,
    as.integer(max_iter),
    as.integer(seed_iter),
    as.double(tolerance),
    as.double(accuracy),
    as.integer(lattice_points),
    .invert_flags(invert, length(xf$path), "transform"),
    if (is.null(n_cores)) NULL else as.integer(n_cores),
    isTRUE(progress)
  )
}

#' Properties of an Elastix transform
#'
#' @param xf An `elastix_transform`, from [elastix_read()].
#'
#' @return
#' * `elastix_affine`: the `4 x 4` matrix of the first linear step, or `NULL` if there is none.
#' * `elastix_kinds`: one string per transform in the chain, giving its resolved steps
#'   initial-first (e.g. `"linear+bspline"`).
#' * `elastix_grid_size` / `elastix_grid_spacing` / `elastix_grid_origin`: a `(k, 3)` matrix, one
#'   row per B-spline in the chain, or `NULL` if the chain is purely linear.
#'
#' @name elastix_properties
#' @export
elastix_affine <- function(xf) .elastix_ptr(xf)$affine()

#' @rdname elastix_properties
#' @export
elastix_kinds <- function(xf) .elastix_ptr(xf)$kinds()

#' @rdname elastix_properties
#' @export
elastix_grid_size <- function(xf) .elastix_ptr(xf)$grid_size()

#' @rdname elastix_properties
#' @export
elastix_grid_spacing <- function(xf) .elastix_ptr(xf)$grid_spacing()

#' @rdname elastix_properties
#' @export
elastix_grid_origin <- function(xf) .elastix_ptr(xf)$grid_origin()


#' Can an Elastix transform be inverted?
#'
#' Answered **without reading the transform's coefficients**, and so cheaply enough to ask about
#' many files at once.
#'
#' A transform cannot be inverted exactly when some step in its chain combines via `Add`:
#' `T(x) = T_initial(x) + T_this(x) - x` does not decompose into invertible hops. That fact lives
#' in one short key -- but the key sits *after* a coefficient array that runs to tens of megabytes,
#' so answering it honestly used to mean parsing the whole file.
#'
#' This walks the same chain and applies the same validation, skipping only the numbers: roughly
#' **20x** faster than [elastix_read()] across real registrations, and ~200x on the largest.
#'
#' Use it when you must label *many* files up front and cannot afford to parse them -- building a
#' bridging graph, say, where each edge needs to know whether it can be walked backwards.
#'
#' @param path Path to a `TransformParameters.*.txt` file. Its initial-transform chain is
#'   followed, however deep.
#'
#' @return `TRUE` if [elastix_xform_inv()] would work on it.
#'
#'   Errors if the file would not load at all (missing, not an Elastix transform, an unsupported
#'   transform kind, binary parameters, a circular chain) -- so `TRUE` is a promise, not a guess.
#'
#' @seealso [elastix_read()], [elastix_xform_inv()]
#'
#' @examples
#' \dontrun{
#' elastix_probe_invertible("TransformParameters.FixedFANC.txt")
#' }
#'
#' @export
elastix_probe_invertible <- function(path) {
  path <- as.character(path)
  if (length(path) != 1L) stop("`path` must name exactly one transform")
  if (!file.exists(path) || dir.exists(path)) {
    stop(sprintf("no Elastix transform file at '%s'", path))
  }
  probe_invertible_raw(path)
}
