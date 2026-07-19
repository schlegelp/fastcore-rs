test_that("fast_hclust matches stats::hclust", {
  set.seed(42)
  d <- dist(matrix(rnorm(60 * 4), 60, 4))

  # Method names follow SciPy; two of them are spelled differently in R.
  for (p in list(
    c("single", "single"),
    c("complete", "complete"),
    c("average", "average"),
    c("weighted", "mcquitty"),
    c("ward", "ward.D2")
  )) {
    ours <- fast_hclust(d, method = p[1])
    theirs <- stats::hclust(d, method = p[2])
    expect_equal(sort(ours$height), sort(theirs$height),
      tolerance = 1e-9,
      info = sprintf("%s vs %s", p[1], p[2])
    )
    # Same partition at several cut points, ids being arbitrary.
    for (k in c(2, 5, 12)) {
      expect_true(
        .same_partition(cutree(ours, k), cutree(theirs, k)),
        info = sprintf("%s at k=%d", p[1], k)
      )
    }
  }
})

test_that("centroid and median take plain distances, not squared ones", {
  # stats::hclust wants squared distances for these two; we follow SciPy and take
  # plain ones. This pins that documented difference so it cannot drift silently.
  set.seed(1)
  d <- dist(matrix(rnorm(40 * 3), 40, 3))
  for (m in c("centroid", "median")) {
    ours <- fast_hclust(d, method = m)
    theirs <- stats::hclust(d^2, method = m)
    expect_equal(sort(ours$height), sort(sqrt(theirs$height)), tolerance = 1e-9)
  }
})

test_that("the returned object is a well-formed hclust", {
  set.seed(7)
  d <- dist(matrix(rnorm(30 * 2), 30, 2))
  h <- fast_hclust(d, method = "average")

  expect_s3_class(h, "hclust")
  expect_named(h, c("merge", "height", "order", "labels", "method", "dist.method"))
  expect_true(is.matrix(h$merge) && is.integer(h$merge))
  expect_equal(dim(h$merge), c(29L, 2L))
  expect_length(h$height, 29L)
  expect_true(!is.unsorted(h$height), info = "heights must be non-decreasing")
  expect_setequal(h$order, 1:30)

  # Every merge refers to a singleton (negative) or an *earlier* step (positive).
  for (i in seq_len(nrow(h$merge))) {
    for (v in h$merge[i, ]) {
      if (v < 0) expect_true(-v >= 1 && -v <= 30) else expect_true(v >= 1 && v < i)
    }
  }
  # It survives the things people actually do with an hclust.
  expect_silent(stats::as.dendrogram(h))
  expect_length(cutree(h, 3), 30L)
})

test_that("nblast_dist matches the idiomatic R expression", {
  set.seed(3)
  m <- matrix(runif(25 * 25), 25, 25)
  diag(m) <- 1

  expect_equal(
    as.vector(nblast_dist(m, symmetry = "mean")),
    as.vector(as.dist(1 - (m + t(m)) / 2)),
    tolerance = 1e-12
  )
  expect_equal(
    as.vector(nblast_dist(m, symmetry = "min")),
    as.vector(as.dist(1 - pmin(m, t(m)))),
    tolerance = 1e-12
  )
  expect_equal(
    as.vector(nblast_dist(m, symmetry = "max")),
    as.vector(as.dist(1 - pmax(m, t(m)))),
    tolerance = 1e-12
  )
  # transform="none" leaves the values alone.
  s <- (m + t(m)) / 2
  expect_equal(
    as.vector(nblast_dist(s, symmetry = "none", transform = "none")),
    as.vector(as.dist(s)),
    tolerance = 1e-12
  )
})

test_that("nblast_dist returns a real dist object", {
  set.seed(4)
  m <- matrix(runif(12 * 12), 12, 12)
  d <- nblast_dist(m)

  expect_s3_class(d, "dist")
  expect_equal(attr(d, "Size"), 12L)
  expect_length(d, 12 * 11 / 2)
  # Round-tripping through as.matrix is the real test of the storage order.
  expect_equal(dim(as.matrix(d)), c(12L, 12L))
  expect_true(isSymmetric(as.matrix(d)))
})

test_that("nblast_hclust agrees with the two-step route", {
  set.seed(5)
  m <- matrix(runif(40 * 40), 40, 40)
  diag(m) <- 1

  for (method in c("single", "complete", "average", "ward")) {
    a <- nblast_hclust(m, method = method)
    b <- fast_hclust(nblast_dist(m), method = method)
    expect_equal(a$height, b$height, tolerance = 1e-12, info = method)
    expect_equal(a$merge, b$merge, info = method)
  }
})

test_that("the diagonal is never read", {
  # NBLAST leaves self-scores on the diagonal; they must not reach the result.
  set.seed(6)
  m <- matrix(runif(15 * 15), 15, 15)
  base <- nblast_dist(m)
  diag(m) <- NA_real_
  expect_equal(as.vector(nblast_dist(m)), as.vector(base), tolerance = 1e-12)
})

test_that("labels propagate", {
  set.seed(8)
  m <- matrix(runif(6 * 6), 6, 6)
  nm <- paste0("neuron", 1:6)

  expect_equal(nblast_hclust(m, labels = nm)$labels, nm)
  expect_equal(attr(nblast_dist(m, labels = nm), "Labels"), nm)
  # Row names are used when `labels` is not given.
  rownames(m) <- nm
  expect_equal(nblast_hclust(m)$labels, nm)
  # And a dist's own Labels carry through fast_hclust.
  expect_equal(fast_hclust(nblast_dist(m, labels = nm))$labels, nm)
})

test_that("cutree on a clustered matrix recovers the groups", {
  set.seed(9)
  # Three well-separated blocks.
  k <- 3
  centres <- matrix(rnorm(k * 8, sd = 5), k, 8)
  lab <- rep(1:k, each = 15)
  x <- centres[lab, ] + matrix(rnorm(45 * 8, sd = 0.2), 45, 8)
  sim <- 1 - as.matrix(dist(x)) / max(dist(x))

  h <- nblast_hclust(sim, method = "average", symmetry = "none")
  expect_true(.same_partition(cutree(h, k), lab))
})

test_that("invalid input is rejected", {
  m <- matrix(runif(16), 4, 4)

  expect_error(nblast_hclust(matrix(1:12, 3, 4)), "square")
  expect_error(nblast_hclust(matrix(1, 1, 1)), "at least 2")
  expect_error(nblast_hclust(m, method = "nonesuch"), "method")
  expect_error(nblast_hclust(m, symmetry = "nonesuch"), "symmetry")
  expect_error(nblast_hclust(m, transform = "nonesuch"), "transform")
  expect_error(nblast_hclust(m, labels = c("a", "b")), "one entry per observation")
  expect_error(nblast_hclust("not a matrix"), "must be a matrix")
  expect_error(fast_hclust(c(1, 2, 3, 4, 5)), "n\\(n-1\\)/2")
})

test_that("an integer score matrix is accepted", {
  # storage.mode is fixed up rather than refused; the matrix is small enough to
  # coerce, unlike the float32/float64 case Python guards against.
  m <- matrix(0L, 4, 4)
  m[upper.tri(m)] <- 1L
  m[lower.tri(m)] <- 1L
  expect_s3_class(nblast_hclust(m, transform = "none", symmetry = "none"), "hclust")
})

test_that("n_cores does not change the answer", {
  set.seed(10)
  m <- matrix(runif(50 * 50), 50, 50)
  expect_equal(
    nblast_hclust(m, n_cores = 1)$height,
    nblast_hclust(m, n_cores = 4)$height
  )
})

test_that("documented choices match the validated ones", {
  # The exported signatures spell the choices out so `?nblast_hclust` is readable,
  # while validation reads the constants. This keeps the two from drifting apart.
  expect_equal(eval(formals(nblast_hclust)$method), nat.fastcore:::.LINKAGE_METHODS)
  expect_equal(eval(formals(nblast_hclust)$symmetry), nat.fastcore:::.SYMMETRIES)
  expect_equal(eval(formals(nblast_hclust)$transform), nat.fastcore:::.TRANSFORMS)
  expect_equal(eval(formals(nblast_dist)$symmetry), nat.fastcore:::.SYMMETRIES)
  expect_equal(eval(formals(nblast_dist)$transform), nat.fastcore:::.TRANSFORMS)
  expect_equal(eval(formals(fast_hclust)$method), nat.fastcore:::.LINKAGE_METHODS)
})

test_that("partial matching works, as with match.arg", {
  m <- matrix(c(1, 0.75, 0.25, 1), 2, 2)
  expect_equal(nblast_hclust(m, method = "aver")$method, "average")
})
