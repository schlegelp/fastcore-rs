# Mesh smoothing. The Rust side (`smooth_mesh_raw`) does the work; this wrapper holds
# the defaults, the argument matching and the documentation, as `nblast_knn` does for
# `nblast_knn_raw`.
#
# The warning also lives here rather than in Rust, and has to: a warning raised from
# inside `.Call` goes through `R_tryEval`, which opens a new top-level context, so
# `tryCatch` and `withCallingHandlers` in the caller's frame never see it. It prints,
# but it cannot be caught or muffled - which is not a warning, it is a message.

.SMOOTH_METHODS <- c("taubin", "laplacian", "humphrey")
.SMOOTH_WEIGHTS <- c("uniform", "inverse_distance", "cotangent")

# Which of lambda/mu/alpha/beta each method actually reads.
.SMOOTH_PARAMS <- list(
  laplacian = "lambda",
  taubin = c("lambda", "mu"),
  humphrey = c("alpha", "beta")
)

#' Smooth a triangle mesh
#'
#' Moves vertices and touches nothing else: the face matrix, the vertex count and the
#' vertex order all come back unchanged, so anything indexed by vertex - synapses,
#' radii, labels - is still attached to the vertex it was attached to.
#'
#' Three methods:
#' \describe{
#'   \item{`"taubin"` (the default)}{Alternating shrink and inflate passes, tuned so
#'     the two cancel below a cut-off frequency. Removes noise without removing the
#'     shape, and is the default for that reason.}
#'   \item{`"laplacian"`}{The plain diffusion step: simple, effective, and it
#'     *shrinks*. At `lambda = 0.5` and five iterations - what `navis.smooth_mesh`
#'     ships - a neuron mesh loses 88% of its enclosed volume. Reach for it when the
#'     mesh is a means to an end rather than when its volume means something, or pair
#'     it with `volume_correction`.}
#'   \item{`"humphrey"`}{The HC filter of Vollmer et al., which fights shrinkage by
#'     pulling each vertex back towards where it started rather than towards a lower
#'     frequency. The gentler of the two on fine detail worth keeping.}
#' }
#'
#' @section The volume correction:
#'
#' `volume_correction = TRUE` rescales the result **about its centroid** so the
#' enclosed volume matches the input's. About the centroid is the one place this
#' deliberately differs from `trimesh.smoothing.filter_laplacian`, which is what
#' `navis.smooth_mesh` calls: upstream rescales about the *origin*, which is not a
#' shape operation. On the 722817260 test neuron at navis' own defaults it displaces
#' the mesh by 41 um, and the mesh is 19-26 um across; it is also not translation
#' invariant, so the same mesh smoothed at two different offsets comes out two
#' different shapes, and far enough from the origin the volume ratio goes negative and
#' the cube root returns `NaN`.
#'
#' The correction runs once, at the end, which is not an approximation of running it
#' every iteration but exactly equal to it: every filter here is an affine combination
#' of a vertex and a normalised average of its neighbours, and those commute with a
#' uniform scaling.
#'
#' On a closed mesh the correction is exactly what it says. A mesh that is *not*
#' closed still usually gets one, and deliberately: both measurements cone every face
#' back to the same anchor, so their ratio stays a consistent measure of how much the
#' surface shrank even where neither number is an enclosed volume on its own. That
#' matters because meshes worth smoothing are almost never watertight. What is left is
#' the genuinely undecidable case - the ratio is zero, infinite, `NaN` or negative, a
#' flat sheet being the clean example - where the vertices come back smoothed but
#' unscaled and a warning says so.
#'
#' @param faces Integer or numeric `(F, 3)` matrix of triangle vertex indices
#'   (0-based).
#' @param vertices Numeric `(V, 3)` matrix of vertex coordinates. Must be finite.
#' @param method One of `"taubin"`, `"laplacian"` or `"humphrey"`; see Details.
#' @param iterations Integer number of passes. For `"taubin"` one pass is a full
#'   `lambda`-then-`mu` pair, i.e. two sweeps over the mesh - not one, as
#'   `trimesh.smoothing.filter_taubin` counts them. Counting half-steps lets an odd
#'   `iterations` end on a shrink that nothing undoes. Default 10.
#' @param lambda Numeric diffusion speed for `"laplacian"` and `"taubin"`, in
#'   `[0, 1]`. Larger is more aggressive. Default 0.5.
#' @param mu Numeric inflating pass for `"taubin"`. Must be negative and larger in
#'   magnitude than `lambda`. Default -0.53.
#' @param alpha Numeric, `"humphrey"` only: how hard vertices are pulled back towards
#'   their original positions, in `[0, 1]`. Default 0.1.
#' @param beta Numeric, `"humphrey"` only: how much of that pull-back lands on the
#'   vertex itself rather than on its one-ring, in `[0, 1]`. Default 0.5.
#' @param weights One of `"uniform"`, `"inverse_distance"` or `"cotangent"`.
#'   `"uniform"` counts every neighbour equally and also regularises the *sampling*,
#'   which means it slides vertices along the surface where the tessellation is
#'   uneven. `"cotangent"` is the discrete Laplace-Beltrami operator: it depends on
#'   the shape rather than on the triangulation, so it moves vertices along the normal
#'   and leaves them alone within the surface. That is usually what you want on meshes
#'   out of EM segmentation, whose triangles vary wildly in size and aspect.
#' @param preserve_border Logical; pin every vertex on a mesh boundary - an endpoint
#'   of an edge used by exactly one face. Without this an open mesh's rim rolls
#'   inwards under any of these filters, because a boundary vertex's one-ring lies
#'   entirely to one side of it.
#' @param lock Optional logical vector, one entry per vertex. A locked vertex comes
#'   back at bitwise the same coordinates but still pulls on its neighbours, which is
#'   what makes it a boundary condition rather than a hole. Unioned with
#'   `preserve_border`, not an alternative to it. Same name and same meaning as
#'   [simplify_mesh()]'s `lock`.
#' @param volume_correction Logical; see the section above.
#' @param threads Integer cap on the thread count for this call, or `NULL` for the
#'   process-wide pool.
#' @return Numeric `(V, 3)` matrix of new vertex coordinates, in the input's order.
#' @seealso [simplify_mesh()], the other half of mesh cleanup.
#' @examples
#' # A 5x5 grid with its middle vertex lifted out of the plane.
#' n <- 5
#' faces <- do.call(rbind, unlist(lapply(0:(n - 2), function(i) {
#'   lapply(0:(n - 2), function(j) {
#'     rbind(
#'       c(i * n + j, (i + 1) * n + j, (i + 1) * n + j + 1),
#'       c(i * n + j, (i + 1) * n + j + 1, i * n + j + 1)
#'     )
#'   })
#' }), recursive = FALSE))
#' vertices <- cbind(rep(0:(n - 1), each = n), rep(0:(n - 1), n), 0)
#' storage.mode(vertices) <- "double"
#' vertices[13, 3] <- 1
#'
#' # One full-strength Laplacian pass puts it back in the plane its ring spans.
#' out <- smooth_mesh(faces, vertices, method = "laplacian", lambda = 1, iterations = 1)
#' out[13, 3]
#'
#' # Pin the rim so an open mesh does not roll inwards.
#' out <- smooth_mesh(faces, vertices, preserve_border = TRUE)
#' identical(out[1, ], vertices[1, ])
#' @export
smooth_mesh <- function(faces, vertices,
                        method = c("taubin", "laplacian", "humphrey"),
                        iterations = 10L,
                        lambda = NULL, mu = NULL, alpha = NULL, beta = NULL,
                        weights = c("uniform", "inverse_distance", "cotangent"),
                        preserve_border = FALSE,
                        lock = NULL,
                        volume_correction = FALSE,
                        threads = NULL) {
  method <- .match_arg(method, .SMOOTH_METHODS, "method")
  weights <- .match_arg(weights, .SMOOTH_WEIGHTS, "weights")

  # A parameter belonging to another method is an error rather than something
  # quietly dropped: a call that passes `alpha` to Taubin has asked for something,
  # and ignoring it is the one outcome that looks like success.
  given <- c(lambda = lambda, mu = mu, alpha = alpha, beta = beta)
  stray <- setdiff(names(given), .SMOOTH_PARAMS[[method]])
  if (length(stray)) {
    stop(sprintf(
      "`%s` does not apply to method = \"%s\"", stray[1L], method
    ), call. = FALSE)
  }

  iterations <- as.integer(iterations)
  if (is.na(iterations) || iterations < 0L) {
    stop("`iterations` must be a non-negative integer", call. = FALSE)
  }

  out <- smooth_mesh_raw(
    faces, vertices, method, iterations,
    lambda, mu, alpha, beta,
    weights, isTRUE(preserve_border), lock,
    isTRUE(volume_correction),
    if (is.null(threads)) NULL else as.integer(threads)
  )

  # Only ever non-NULL when a correction was asked for and could not be made.
  # Silence would be the failure worth avoiding: the caller asked for a
  # volume-preserving smooth and got a plain one.
  if (!is.null(out$volumes)) {
    warning(sprintf(
      paste0(
        "`volume_correction` was requested but the mesh has no usable enclosed ",
        "volume (signed volume %g before smoothing, %g after), so the vertices ",
        "were returned unscaled. This is expected for a mesh that is not closed."
      ),
      out$volumes[1L], out$volumes[2L]
    ), call. = FALSE)
  }
  out$vertices
}
