# k-nearest-neighbour NBLAST. The Rust side (`nblast_knn_raw`) does the work; this
# wrapper holds the defaults, the coercion and the documentation, as
# `nblast_hclust` does for `nblast_hclust_raw`.

.KNN_SYMMETRIES <- c("mean", "forward", "min", "max")

# Validate a list of (N, 3) numeric matrices and coerce to double storage.
# Deliberately validates rather than silently converting shapes: a transposed
# (3, N) matrix would otherwise be scored as three enormous neurons.
.check_clouds <- function(x, what) {
  if (is.matrix(x)) x <- list(x)
  if (!is.list(x) || !length(x)) {
    stop(sprintf("`%s` must be a non-empty list of (N, 3) numeric matrices", what),
      call. = FALSE
    )
  }
  lapply(seq_along(x), function(i) {
    m <- x[[i]]
    if (is.data.frame(m)) m <- as.matrix(m)
    if (!is.matrix(m) || !is.numeric(m) || ncol(m) != 3L) {
      stop(sprintf(
        "`%s[[%d]]` must be a numeric matrix with 3 columns, got %s",
        what, i, paste(class(m), collapse = "/")
      ), call. = FALSE)
    }
    if (!nrow(m)) {
      stop(sprintf("`%s[[%d]]` has no points", what, i), call. = FALSE)
    }
    if (!is.double(m)) storage.mode(m) <- "double"
    m
  })
}

.check_alphas <- function(alphas, clouds, what) {
  if (is.null(alphas)) {
    return(NULL)
  }
  if (!is.list(alphas) || length(alphas) != length(clouds)) {
    stop(sprintf("`%s` must be a list with one vector per neuron", what), call. = FALSE)
  }
  lapply(seq_along(alphas), function(i) {
    a <- as.double(alphas[[i]])
    if (length(a) != nrow(clouds[[i]])) {
      stop(sprintf(
        "`%s[[%d]]` has %d values but the neuron has %d points",
        what, i, length(a), nrow(clouds[[i]])
      ), call. = FALSE)
    }
    a
  })
}

#' k nearest neighbours under NBLAST, without building the score matrix
#'
#' At connectomics scale an all-by-all is the wrong shape for a k-NN question:
#' 164k neurons is 2.7e10 pairs and a 107 GB matrix, even though the answer
#' wanted from it is a small k-NN graph (typically to feed a UMAP embedding).
#' This computes that graph directly, in three stages: each neuron becomes a
#' coarse voxel-occupancy signature; the `n_candidates` most similar neurons per
#' row are shortlisted from those signatures; and the **exact** NBLAST score is
#' then computed for the shortlisted pairs only.
#'
#' The pre-filter is sound because the FCWB scoring matrix has finite support --
#' beyond its last distance bin (40 um) every cell is about -10, so neurons that
#' do not overlap in space score at that floor whatever their shape. Only *which*
#' neurons make the shortlist is approximate; every returned score is an exact
#' NBLAST value, because a neuron belonging in the true top-`k` outranks the
#' global k-th and so cannot be dropped by the rerank once shortlisted.
#'
#' Measured on 163,976 zebrafish neurons: recall@20 = 0.99 at the default
#' `n_candidates = 200`, scoring 0.16% of the pairs and taking about 5 minutes
#' against an estimated 35 hours for the exact all-by-all.
#'
#' @param points List of `(N, 3)` numeric matrices of query point coordinates,
#'   one per neuron.
#' @param vects List of `(N, 3)` numeric matrices of unit tangent vectors,
#'   aligned with `points`.
#' @param target Optional list of `(N, 3)` target point matrices. When given,
#'   neighbours are searched among the targets and the returned indices address
#'   *them*; this is the k-NN counterpart of [nblast()]. `NULL` (the default)
#'   runs the all-by-all form over `points`, excluding self-matches.
#' @param target_vects Tangent vectors for `target`; required when `target` is
#'   given.
#' @param k Integer; neighbours to return per neuron.
#' @param symmetry How the two directions of a pair are combined *before* the
#'   top-`k` cut. This matters more than for a full matrix: with a matrix you can
#'   symmetrise afterwards against the transpose, but once only `k` neighbours per
#'   row are kept the transpose is gone. The asymmetry is real -- a small neuron
#'   contained in a large one scores high one way and low the other -- so `"mean"`
#'   is the default. `"forward"` keeps each row's own forward score.
#' @param n_candidates Integer; shortlist size per neuron, and the one
#'   recall/cost knob. Measured recall@20 on 163,976 real neurons: 0.91 at 50,
#'   0.97 at 100, 0.99 at 200, 0.996 at 400. The budget needed to hold a given
#'   recall grows only about logarithmically with the number of neurons.
#' @param alphas,target_alphas Optional lists of per-point alpha (anisotropy)
#'   vectors; `NULL` disables alpha weighting.
#' @param smat Optional scoring matrix, or `NULL` for the built-in FCWB matrix.
#' @param dist_edges,dot_edges Bin edges for `smat`; required when it is given.
#' @param normalize Logical; divide each score by the query self-match score, so
#'   a perfect self-match is 1.
#' @param limit_dist Optional numeric distance cut-off; `NULL` disables it and
#'   `"auto"` takes it from the scoring matrix.
#' @param voxel Numeric; signature voxel edge, in the units of `points` (um for
#'   the FCWB matrix). 10-20 measured equivalently.
#' @param n_dirs Integer; tangent-direction bins for the signature (1 disables
#'   them). `3` is the cheap "which axis dominates" case; larger values use a
#'   Fibonacci half-sphere.
#' @param splat Logical; trilinearly spread each point over its 8 surrounding
#'   voxels. Worth about 0.05 recall@20.
#' @param n_cores Optional integer thread count; `NULL` or `<= 0` uses all cores.
#' @param precision Integer; 32 or 64, the width scores are returned at. The
#'   scoring maths always runs in double precision.
#' @param progress Logical; display a progress bar.
#'
#' @return A list with `idx`, an integer `(n_query, k)` matrix of **1-based**
#'   neighbour indices in descending score order, and `scores`, the matching
#'   numeric matrix. Rows with fewer than `k` available neighbours are padded
#'   with `NA` in both.
#'
#' @seealso [nblast_allbyall()] for the full matrix, [nblast_hclust()] to cluster
#'   one.
#'
#' @examples
#' set.seed(1)
#' pts <- lapply(1:6, function(i) {
#'   base <- matrix(rnorm(60), ncol = 3)
#'   base + matrix(rep(c(i, 0, 0), each = 20), ncol = 3)
#' })
#' vects <- lapply(pts, function(p) {
#'   v <- matrix(0, nrow(p), 3)
#'   v[, 1] <- 1
#'   v
#' })
#' nn <- nblast_knn(pts, vects, k = 2)
#' nn$idx
#' nn$scores
#' @export
nblast_knn <- function(points, vects,
                       target = NULL, target_vects = NULL,
                       k = 20L,
                       symmetry = c("mean", "forward", "min", "max"),
                       n_candidates = 200L,
                       alphas = NULL, target_alphas = NULL,
                       smat = NULL, dist_edges = NULL, dot_edges = NULL,
                       normalize = TRUE, limit_dist = NULL,
                       voxel = 20, n_dirs = 3L, splat = TRUE,
                       n_cores = NULL, precision = 32L, progress = FALSE) {
  symmetry <- .match_arg(symmetry, .KNN_SYMMETRIES, "symmetry")

  points <- .check_clouds(points, "points")
  vects <- .check_clouds(vects, "vects")
  if (length(points) != length(vects)) {
    stop("`points` and `vects` must have the same length", call. = FALSE)
  }
  alphas <- .check_alphas(alphas, points, "alphas")

  if (is.null(target) != is.null(target_vects)) {
    stop("`target` and `target_vects` must be given together", call. = FALSE)
  }
  if (!is.null(target)) {
    target <- .check_clouds(target, "target")
    target_vects <- .check_clouds(target_vects, "target_vects")
    if (length(target) != length(target_vects)) {
      stop("`target` and `target_vects` must have the same length", call. = FALSE)
    }
    target_alphas <- .check_alphas(target_alphas, target, "target_alphas")
  }

  k <- as.integer(k)
  if (length(k) != 1L || is.na(k) || k < 1L) {
    stop("`k` must be a single positive integer", call. = FALSE)
  }
  n_candidates <- as.integer(n_candidates)
  if (length(n_candidates) != 1L || is.na(n_candidates) || n_candidates < 1L) {
    stop("`n_candidates` must be a single positive integer", call. = FALSE)
  }
  if (!is.numeric(voxel) || length(voxel) != 1L || is.na(voxel) || voxel <= 0) {
    stop("`voxel` must be a single positive number", call. = FALSE)
  }
  if (!precision %in% c(32L, 64L)) {
    stop("`precision` must be 32 or 64", call. = FALSE)
  }

  # "auto" resolves against whichever matrix will actually be used.
  if (is.character(limit_dist)) {
    if (!identical(limit_dist, "auto")) {
      stop('`limit_dist` must be a number, "auto", or NULL', call. = FALSE)
    }
    limit_dist <- smat_auto_limit(smat, dist_edges, dot_edges, !is.null(alphas))
  }

  nblast_knn_raw(
    points, vects, alphas,
    target, target_vects, target_alphas,
    k, n_candidates, symmetry,
    as.double(voxel), as.integer(n_dirs), isTRUE(splat),
    smat, dist_edges, dot_edges,
    isTRUE(normalize), limit_dist,
    if (is.null(n_cores)) NULL else as.integer(n_cores),
    as.integer(precision), isTRUE(progress)
  )
}
