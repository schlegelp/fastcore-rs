# k-NN NBLAST. The anchor is parity with `nblast_allbyall`: with a signature grid
# coarse enough that every neuron shares one feature, the candidate stage is
# exhaustive and `nblast_knn` must reproduce the dense top-k exactly.

# Coarse-grid options that make candidate generation exhaustive.
EXHAUSTIVE <- list(voxel = 1e6, n_dirs = 1L, splat = FALSE)

# A population of jittered copies of a few random walks, so there are genuine
# near-neighbours rather than uniform noise.
knn_population <- function(n_groups = 4, per_group = 3, n_pts = 25, seed = 1) {
  set.seed(seed)
  pts <- list()
  vects <- list()
  for (g in seq_len(n_groups)) {
    step <- matrix(rnorm(n_pts * 3, sd = 0.5), ncol = 3)
    walk <- apply(step, 2, cumsum) +
      matrix(rep(runif(3, 0, 60), each = n_pts), ncol = 3)
    for (r in seq_len(per_group)) {
      p <- walk + matrix(rnorm(n_pts * 3, sd = 0.4), ncol = 3)
      d <- rbind(diff(p), p[n_pts, ] - p[n_pts - 1, ])
      v <- d / sqrt(rowSums(d^2))
      pts[[length(pts) + 1L]] <- p
      vects[[length(vects) + 1L]] <- v
    }
  }
  list(points = pts, vects = vects)
}

dense_allbyall <- function(pop) {
  nblast_allbyall(
    pop$points, pop$vects, NULL, NULL, NULL, NULL,
    TRUE, NULL, NULL, 64L, FALSE
  )
}

# Top-k of a dense matrix under `symmetry`, self excluded; 1-based indices.
dense_topk <- function(m, k, symmetry = "mean") {
  s <- switch(symmetry,
    mean = (m + t(m)) / 2,
    min = pmin(m, t(m)),
    max = pmax(m, t(m)),
    forward = m
  )
  diag(s) <- -Inf
  idx <- t(apply(s, 1, function(r) order(r, decreasing = TRUE)[seq_len(k)]))
  sc <- t(apply(s, 1, function(r) sort(r, decreasing = TRUE)[seq_len(k)]))
  list(idx = idx, scores = sc)
}

test_that("exhaustive candidates reproduce the dense top-k", {
  pop <- knn_population()
  n <- length(pop$points)
  m <- dense_allbyall(pop)
  k <- 3L
  for (sym in c("mean", "forward", "min", "max")) {
    want <- dense_topk(m, k, sym)
    got <- do.call(nblast_knn, c(
      list(pop$points, pop$vects,
        k = k, symmetry = sym, n_candidates = n - 1L, precision = 64L
      ),
      EXHAUSTIVE
    ))
    expect_equal(got$scores, want$scores, tolerance = 1e-12, info = sym)
    expect_equal(got$idx, want$idx, info = sym)
  }
})

test_that("indices are 1-based and address the right neurons", {
  pop <- knn_population()
  n <- length(pop$points)
  m <- dense_allbyall(pop)
  s <- (m + t(m)) / 2
  got <- do.call(nblast_knn, c(
    list(pop$points, pop$vects, k = 3L, n_candidates = n - 1L, precision = 64L),
    EXHAUSTIVE
  ))
  expect_true(is.integer(got$idx))
  expect_true(all(got$idx >= 1L & got$idx <= n))
  # a returned score must equal the dense cell its index points at
  for (i in seq_len(n)) {
    for (c in seq_len(3L)) {
      expect_equal(got$scores[i, c], s[i, got$idx[i, c]], tolerance = 1e-12)
    }
  }
  # and never itself
  expect_false(any(got$idx == row(got$idx)))
})

test_that("scores are exact even when the shortlist is small", {
  pop <- knn_population()
  m <- dense_allbyall(pop)
  s <- (m + t(m)) / 2
  got <- nblast_knn(pop$points, pop$vects, k = 2L, n_candidates = 2L, precision = 64L)
  for (i in seq_len(nrow(got$idx))) {
    for (c in seq_len(ncol(got$idx))) {
      j <- got$idx[i, c]
      if (!is.na(j)) expect_equal(got$scores[i, c], s[i, j], tolerance = 1e-12)
    }
  }
})

test_that("rows short of k are padded with NA in both matrices", {
  pop <- knn_population(n_groups = 2, per_group = 2)
  n <- length(pop$points)
  k <- n + 2L
  got <- do.call(nblast_knn, c(
    list(pop$points, pop$vects, k = k, n_candidates = n - 1L),
    EXHAUSTIVE
  ))
  expect_equal(dim(got$idx), c(n, k))
  expect_true(all(is.na(got$idx[, n:k])))
  expect_true(all(is.na(got$scores[, n:k])))
  expect_false(any(is.na(got$idx[, seq_len(n - 1L)])))
})

test_that("query -> target indexes the target list and keeps self-matches", {
  pop <- knn_population()
  n <- length(pop$points)
  q <- 1:4
  got <- do.call(nblast_knn, c(
    list(pop$points[q], pop$vects[q],
      target = pop$points, target_vects = pop$vects,
      k = 1L, n_candidates = n, precision = 64L
    ),
    EXHAUSTIVE
  ))
  expect_equal(nrow(got$idx), length(q))
  # unlike the all-by-all form nothing is excluded, so each query matches itself
  expect_equal(as.vector(got$idx[, 1]), q)
  expect_equal(as.vector(got$scores[, 1]), rep(1, length(q)), tolerance = 1e-9)
})

test_that("query -> target matches the dense rectangular scores", {
  pop <- knn_population()
  q <- 1:4
  t_ix <- 5:12
  fwd <- nblast(
    pop$points[q], pop$vects[q], pop$points[t_ix], pop$vects[t_ix],
    NULL, NULL, NULL, NULL, NULL, TRUE, NULL, NULL, 64L, FALSE
  )
  rev <- nblast(
    pop$points[t_ix], pop$vects[t_ix], pop$points[q], pop$vects[q],
    NULL, NULL, NULL, NULL, NULL, TRUE, NULL, NULL, 64L, FALSE
  )
  s <- (fwd + t(rev)) / 2
  k <- 3L
  got <- do.call(nblast_knn, c(
    list(pop$points[q], pop$vects[q],
      target = pop$points[t_ix], target_vects = pop$vects[t_ix],
      k = k, n_candidates = length(t_ix), precision = 64L
    ),
    EXHAUSTIVE
  ))
  want <- t(apply(s, 1, function(r) sort(r, decreasing = TRUE)[seq_len(k)]))
  expect_equal(got$scores, want, tolerance = 1e-12)
})

test_that("limit_dist = 'auto' matches passing the resolved number", {
  pop <- knn_population()
  n <- length(pop$points)
  a <- do.call(nblast_knn, c(
    list(pop$points, pop$vects, k = 2L, n_candidates = n - 1L,
      limit_dist = "auto", precision = 64L
    ), EXHAUSTIVE
  ))
  b <- do.call(nblast_knn, c(
    list(pop$points, pop$vects, k = 2L, n_candidates = n - 1L,
      limit_dist = smat_auto_limit(NULL, NULL, NULL, FALSE), precision = 64L
    ), EXHAUSTIVE
  ))
  expect_equal(a, b)
})

test_that("n_cores does not change the answer", {
  pop <- knn_population()
  n <- length(pop$points)
  args <- c(list(pop$points, pop$vects, k = 3L, n_candidates = n - 1L,
                 precision = 64L), EXHAUSTIVE)
  expect_equal(
    do.call(nblast_knn, args),
    do.call(nblast_knn, c(args, list(n_cores = 1L)))
  )
})

test_that("mean symmetry resolves the containment asymmetry", {
  # A short neuron lying inside a long one: forward rates it a great match one
  # way and a poor one the other; mean must agree in both rows.
  long <- cbind(seq(0, 59.5, by = 0.5), 0, 0)
  short <- long[1:20, , drop = FALSE]
  vl <- matrix(rep(c(1, 0, 0), each = nrow(long)), ncol = 3)
  vs <- matrix(rep(c(1, 0, 0), each = nrow(short)), ncol = 3)
  pts <- list(long, short)
  vects <- list(vl, vs)

  m <- nblast_allbyall(pts, vects, NULL, NULL, NULL, NULL, TRUE, NULL, NULL, 64L, FALSE)
  expect_gt(m[2, 1], m[1, 2] + 0.2) # the asymmetry is real

  fwd <- do.call(nblast_knn, c(list(pts, vects, k = 1L, symmetry = "forward",
                                    n_candidates = 1L, precision = 64L), EXHAUSTIVE))
  mean_ <- do.call(nblast_knn, c(list(pts, vects, k = 1L, symmetry = "mean",
                                      n_candidates = 1L, precision = 64L), EXHAUSTIVE))
  expect_gt(fwd$scores[2, 1] - fwd$scores[1, 1], 0.2)
  expect_equal(mean_$scores[1, 1], mean_$scores[2, 1], tolerance = 1e-12)
  expect_gt(mean_$scores[1, 1], m[1, 2])
  expect_lt(mean_$scores[1, 1], m[2, 1])
})

test_that("bad input is rejected with a useful message", {
  pop <- knn_population(n_groups = 2, per_group = 2)
  expect_error(nblast_knn(pop$points, pop$vects, k = 0L), "`k` must be")
  expect_error(nblast_knn(pop$points, pop$vects, voxel = 0), "`voxel` must be")
  expect_error(nblast_knn(pop$points, pop$vects, precision = 8L), "`precision` must be")
  expect_error(nblast_knn(pop$points, pop$vects, symmetry = "nope"), "`symmetry` must be")
  expect_error(nblast_knn(pop$points, pop$vects[-1]), "same length")
  expect_error(
    nblast_knn(pop$points, pop$vects, target = pop$points),
    "given together"
  )
  expect_error(nblast_knn(list(matrix(1:4, ncol = 2)), pop$vects), "3 columns")
  expect_error(
    nblast_knn(pop$points, pop$vects, alphas = list(1, 2)),
    "one vector per neuron"
  )
  expect_error(nblast_knn(pop$points, pop$vects, limit_dist = "nope"), "auto")
})
