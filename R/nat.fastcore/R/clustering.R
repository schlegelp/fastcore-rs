# Hierarchical clustering of a score matrix. The Rust side (`*_raw`) does the work;
# these wrappers hold the defaults, the coercion and the documentation.

# Accepted values, first element being the default. The exported functions spell
# these out in their own signatures so `?nblast_hclust` shows the real choices
# rather than a variable name; test-clustering.R pins the two together.
.LINKAGE_METHODS <- c(
  "ward", "single", "complete", "average", "weighted", "centroid", "median"
)
.SYMMETRIES <- c("mean", "min", "max", "none")
.TRANSFORMS <- c("one_minus", "none")

# `match.arg` with a usable message. Base R's hardcodes the literal "'arg'", which
# is no help when a call has three of these to get wrong. Keeps partial matching,
# and treats the untouched default vector as "take the first".
.match_arg <- function(value, choices, what) {
  if (identical(value, choices)) {
    return(choices[1L])
  }
  value <- as.character(value)
  if (length(value) != 1L || is.na(value)) {
    stop(sprintf("`%s` must be a single string", what), call. = FALSE)
  }
  i <- pmatch(value, choices)
  if (is.na(i)) {
    stop(sprintf(
      "`%s` must be one of %s, got \"%s\"",
      what, paste0('"', choices, '"', collapse = ", "), value
    ), call. = FALSE)
  }
  choices[i]
}

# Coerce and check a square score matrix. Deliberately validates rather than
# converts: at 100k neurons a side the matrix is tens of GB, and silently making a
# double copy of it is not a kindness.
.check_score_matrix <- function(scores) {
  if (!is.matrix(scores)) {
    if (is.data.frame(scores)) {
      scores <- as.matrix(scores)
    } else {
      stop("`scores` must be a matrix")
    }
  }
  if (nrow(scores) != ncol(scores)) {
    stop(sprintf(
      "`scores` must be square to be clustered, got %d x %d",
      nrow(scores), ncol(scores)
    ))
  }
  if (nrow(scores) < 2L) stop("clustering needs at least 2 observations")
  if (!is.double(scores)) {
    if (!is.numeric(scores)) stop("`scores` must be numeric")
    storage.mode(scores) <- "double"
  }
  scores
}

.check_labels <- function(labels, n) {
  if (is.null(labels)) {
    return(NULL)
  }
  labels <- as.character(labels)
  if (length(labels) != n) {
    stop(sprintf(
      "`labels` must have one entry per observation: got %d, want %d",
      length(labels), n
    ))
  }
  labels
}

.check_n_cores <- function(n_cores) {
  if (is.null(n_cores)) {
    return(NULL)
  }
  n_cores <- as.integer(n_cores)
  if (length(n_cores) != 1L || is.na(n_cores)) {
    stop("`n_cores` must be a single integer or NULL")
  }
  n_cores
}

#' Hierarchical clustering of a score matrix
#'
#' Clusters a square score matrix -- typically NBLAST output from
#' [nblast_allbyall()] -- and returns a standard [stats::hclust()] object.
#'
#' Unlike [stats::hclust()], which refuses more than 65536 observations, this has
#' no size ceiling beyond available memory. That limit is a hard blocker for
#' whole-brain connectome work, where 100k-200k neurons is routine.
#'
#' Symmetrising, the score-to-distance transform and condensing to the upper
#' triangle are **fused into a single pass**, and the resulting buffer is then
#' clustered in place. The idiomatic R spelling,
#' `hclust(as.dist(1 - (m + t(m)) / 2))`, materialises three further `n x n`
#' matrices before clustering even begins, which at these sizes is where the
#' memory goes.
#'
#' @param scores Square `(n, n)` numeric score matrix. The diagonal is never read,
#'   so self-scores may be left on it.
#' @param method Linkage method: one of `"ward"`, `"single"`, `"complete"`,
#'   `"average"`, `"weighted"`, `"centroid"` or `"median"`. Names follow SciPy, so
#'   `"ward"` is R's `"ward.D2"` and `"weighted"` is R's `"mcquitty"`. Note that
#'   `"centroid"` and `"median"` take **plain** distances here, whereas
#'   [stats::hclust()] expects squared ones for those two; the heights returned
#'   here equal `sqrt()` of what `hclust(d^2, ...)` gives.
#' @param symmetry How to combine cell `(i, j)` with `(j, i)`, since NBLAST is not
#'   symmetric: `"mean"` (the default), `"min"`, `"max"`, or `"none"` to take the
#'   upper triangle as given. Matches the `symmetry` argument of
#'   [nblast_allbyall()]; use `"none"` if the matrix is already symmetric, which is
#'   also the fastest path.
#' @param transform `"one_minus"` (the default) turns a similarity into a distance
#'   via `1 - score`; `"none"` treats the values as distances already.
#' @param labels Optional character vector of `n` names for the observations, or
#'   `NULL`. Falls back to the matrix's row names.
#' @param n_cores Optional integer thread count for the condensing pass; `NULL`
#'   uses all cores.
#'
#' @return An object of class `hclust`, so [stats::cutree()], [plot()] and
#'   [stats::as.dendrogram()] all work on it directly.
#'
#' @seealso [nblast_dist()] for the condensed distances on their own,
#'   [fast_hclust()] to cluster an existing `dist`.
#'
#' @examples
#' # A score matrix with two obvious pairs.
#' m <- matrix(c(
#'   1.00, 0.90, 0.20, 0.10,
#'   0.70, 1.00, 0.30, 0.15,
#'   0.25, 0.35, 1.00, 0.80,
#'   0.05, 0.10, 0.60, 1.00
#' ), 4, 4, byrow = TRUE)
#'
#' h <- nblast_hclust(m, method = "average")
#' cutree(h, 2)
#'
#' @export
nblast_hclust <- function(scores,
                          method = c(
                            "ward", "single", "complete", "average",
                            "weighted", "centroid", "median"
                          ),
                          symmetry = c("mean", "min", "max", "none"),
                          transform = c("one_minus", "none"),
                          labels = NULL,
                          n_cores = NULL) {
  # Validated here rather than in Rust: extendr cannot carry a panic's message
  # across to R, so a Rust-side check would surface as "User function panicked".
  method <- .match_arg(method, .LINKAGE_METHODS, "method")
  symmetry <- .match_arg(symmetry, .SYMMETRIES, "symmetry")
  transform <- .match_arg(transform, .TRANSFORMS, "transform")

  scores <- .check_score_matrix(scores)
  if (is.null(labels)) labels <- rownames(scores)
  labels <- .check_labels(labels, nrow(scores))
  nblast_hclust_raw(
    scores, method, symmetry, transform, labels, .check_n_cores(n_cores)
  )
}

#' Condensed distances from a score matrix
#'
#' Symmetrises a score matrix, converts it to distances and condenses it to the
#' upper triangle in a single fused pass, returning a [stats::dist()] object. The
#' only allocation is the result itself, where `as.dist(1 - (m + t(m)) / 2)` builds
#' several full `n x n` copies on the way.
#'
#' R stores a `dist` as the lower triangle by column, which for a symmetric matrix
#' is element-for-element the same sequence as the upper triangle by row, so no
#' rearranging is needed.
#'
#' @param scores Square `(n, n)` numeric score matrix. The diagonal is not read.
#' @param symmetry How to combine cell `(i, j)` with `(j, i)`: `"mean"` (default),
#'   `"min"`, `"max"`, or `"none"`.
#' @param transform `"one_minus"` (default) for `1 - score`, or `"none"` if the
#'   values are already distances.
#' @param labels Optional character vector of `n` names, or `NULL`. Falls back to
#'   the matrix's row names.
#' @param n_cores Optional integer thread count; `NULL` uses all cores.
#'
#' @return An object of class `dist` with `n * (n - 1) / 2` entries.
#'
#' @seealso [nblast_hclust()] to go straight to a dendrogram without materialising
#'   the distances at all.
#'
#' @examples
#' m <- matrix(c(1, 0.75, 0.25, 1), 2, 2)
#' nblast_dist(m)
#'
#' @export
nblast_dist <- function(scores,
                        symmetry = c("mean", "min", "max", "none"),
                        transform = c("one_minus", "none"),
                        labels = NULL,
                        n_cores = NULL) {
  symmetry <- .match_arg(symmetry, .SYMMETRIES, "symmetry")
  transform <- .match_arg(transform, .TRANSFORMS, "transform")

  scores <- .check_score_matrix(scores)
  if (is.null(labels)) labels <- rownames(scores)
  labels <- .check_labels(labels, nrow(scores))
  nblast_dist_raw(scores, symmetry, transform, labels, .check_n_cores(n_cores))
}

#' Hierarchical clustering without the 65536 observation limit
#'
#' A drop-in replacement for [stats::hclust()] that does not stop at 65536
#' observations, and clusters at a fraction of the memory.
#'
#' Clustering consumes its input as scratch, and R's value semantics forbid writing
#' to the caller's vector, so `d` is copied once. Clustering straight from a score
#' matrix with [nblast_hclust()] avoids that copy entirely, and is the better route
#' when you have the matrix to hand.
#'
#' @param d A `dist` object, or a bare numeric vector of length `n * (n - 1) / 2`
#'   holding the lower triangle by column.
#' @param method Linkage method: one of `"ward"`, `"single"`, `"complete"`,
#'   `"average"`, `"weighted"`, `"centroid"` or `"median"`. Names follow SciPy, so
#'   `"ward"` is R's `"ward.D2"` and `"weighted"` is R's `"mcquitty"`. Note that
#'   `"centroid"` and `"median"` take **plain** distances here, whereas
#'   [stats::hclust()] expects squared ones for those two; the heights returned
#'   here equal `sqrt()` of what `hclust(d^2, ...)` gives.
#' @param labels Optional character vector of `n` names, or `NULL` to use the
#'   `Labels` attribute of `d` when it has one.
#'
#' @return An object of class `hclust`.
#'
#' @examples
#' d <- dist(matrix(rnorm(40), 20, 2))
#' h <- fast_hclust(d, method = "average")
#' # Same tree as stats::hclust for the equivalent method.
#' all.equal(h$height, hclust(d, method = "average")$height)
#'
#' @export
fast_hclust <- function(d,
                        method = c(
                          "ward", "single", "complete", "average",
                          "weighted", "centroid", "median"
                        ),
                        labels = NULL) {
  method <- .match_arg(method, .LINKAGE_METHODS, "method")
  if (!is.numeric(d)) {
    stop("`d` must be a `dist` object or a numeric vector", call. = FALSE)
  }
  if (!is.double(d)) storage.mode(d) <- "double"

  # Resolve the observation count here, not in Rust: extendr cannot carry a panic's
  # message across, so a Rust-side check would surface only as "User function
  # panicked".
  n <- attr(d, "Size")
  if (is.null(n)) {
    n <- (1 + sqrt(1 + 8 * length(d))) / 2
    if (!is.finite(n) || n < 2 || abs(n - round(n)) > 1e-8) {
      stop(sprintf(
        "`d` has %d entries, which is not n(n-1)/2 for any n", length(d)
      ), call. = FALSE)
    }
  }
  n <- as.integer(round(n))
  if (length(d) != n * (n - 1) / 2) {
    stop(sprintf(
      "`d` has %d entries, but Size = %d implies n(n-1)/2 = %.0f",
      length(d), n, n * (n - 1) / 2
    ), call. = FALSE)
  }

  if (!is.null(labels)) labels <- .check_labels(labels, n)
  fast_hclust_raw(d, method, labels)
}

#' Symmetrise a score matrix
#'
#' Combines cell `(i, j)` with `(j, i)`, since NBLAST is not symmetric. The
#' idiomatic `(m + t(m)) / 2` builds two further `n x n` matrices on the way to the
#' answer; this builds one, the result.
#'
#' You do not need this before [nblast_hclust()] or [nblast_dist()] -- both take a
#' `symmetry` argument and fold the same combine into their own fused pass. Reach
#' for it when something *else* has to read the matrix: plotting it, writing it out,
#' or handing it to a function that assumes symmetry.
#'
#' @param scores Square `(n, n)` numeric score matrix. The diagonal is not read, so
#'   self-scores may be left on it.
#' @param symmetry How to combine `(i, j)` with `(j, i)`: `"mean"` (the default),
#'   `"min"`, `"max"`, or `"none"` to mirror the upper triangle onto the lower
#'   instead of combining.
#' @param n_cores Optional integer thread count; `NULL` uses all cores.
#'
#' @return A symmetric `(n, n)` matrix, with `dimnames` carried over. `scores`
#'   itself is unchanged -- R's value semantics forbid writing to it, which is the
#'   one thing this cannot do that the Python version can.
#'
#' @seealso [nblast_dist()] and [nblast_hclust()], which symmetrise as part of a
#'   single pass over the matrix.
#'
#' @examples
#' m <- matrix(c(1, 0.4, 0.8, 1), 2, 2)
#' symmetrize(m)
#'
#' @export
symmetrize <- function(scores,
                       symmetry = c("mean", "min", "max", "none"),
                       n_cores = NULL) {
  symmetry <- .match_arg(symmetry, .SYMMETRIES, "symmetry")
  scores <- .check_score_matrix(scores)

  out <- symmetrize_raw(scores, symmetry, .check_n_cores(n_cores))
  dimnames(out) <- dimnames(scores)
  out
}

#' Dendrogram leaf order
#'
#' The order to place the leaves in so a dendrogram draws without crossing
#' branches -- what [stats::hclust()] returns in its `order` element, computed from
#' a merge matrix alone.
#'
#' You do not need this for an untouched `hclust`: it already carries the same
#' ordering. It is for a merge matrix you built or edited yourself, where `order`
#' is missing or no longer matches -- a dendrogram drawn with an ordering from a
#' *different* child order is the crossing-branches mess this exists to avoid.
#'
#' The walk is iterative, so a 200k-observation chain does not need 200k nested
#' calls.
#'
#' @param x An [stats::hclust()] object, or its `(n - 1, 2)` merge matrix: negative
#'   entries are observations (`-j` is observation `j`), positive entries are the
#'   cluster formed at that earlier step.
#'
#' @return An integer vector of length `n` holding each observation's position in
#'   the drawing order, left to right -- a permutation of `1:n`, in the same form
#'   as `hclust$order`.
#'
#' @examples
#' set.seed(1)
#' h <- fast_hclust(dist(matrix(rnorm(20), 10, 2)), method = "average")
#' identical(leaf_order(h), h$order)
#'
#' @export
leaf_order <- function(x) {
  merge <- if (inherits(x, "hclust")) x$merge else x
  if (!is.matrix(merge) || ncol(merge) != 2L) {
    stop("`x` must be an `hclust` object or an (n - 1, 2) merge matrix",
      call. = FALSE
    )
  }
  if (!is.numeric(merge)) stop("the merge matrix must be numeric", call. = FALSE)
  if (anyNA(merge)) stop("the merge matrix must not contain NA", call. = FALSE)
  if (any(merge != round(merge))) {
    stop("the merge matrix must hold whole numbers", call. = FALSE)
  }
  storage.mode(merge) <- "integer"

  # Validated here rather than in Rust: an entry naming a merge that has not
  # happened yet sends the walk into a loop, and extendr cannot carry a panic's
  # message across to R anyway.
  n <- nrow(merge) + 1L
  ok <- ifelse(merge < 0L, -merge <= n, merge >= 1L & merge < row(merge))
  if (!all(ok)) {
    bad <- which(!ok, arr.ind = TRUE)[1L, ]
    stop(sprintf(
      paste0(
        "row %d of the merge matrix refers to %d, which is neither an ",
        "observation (-1 to -%d) nor a merge before it"
      ),
      bad[["row"]], merge[bad[["row"]], bad[["col"]]], n
    ), call. = FALSE)
  }

  leaf_order_raw(merge)
}
