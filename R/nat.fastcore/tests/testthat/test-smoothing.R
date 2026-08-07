# Mesh smoothing.
#
# The algorithms themselves are tested exhaustively on the Rust side, and against
# `trimesh` from Python. What is worth testing here is the binding: that the argument
# order survives the trip through extendr, that `lock` and `preserve_border` reach the
# core as the flags they are, that the volume correction's warning arrives through R's
# condition system, and that a parameter belonging to another method is refused rather
# than dropped. All indices are 0-based, like the rest of the package.

# An `n x n` grid in the z = 0 plane, split along each cell's (0,0)->(1,1) diagonal,
# displaced out of the plane at the highest frequency the grid can carry. The clean
# surface is `z = 0`, so residual `z` is exactly the noise.
.noisy_grid <- function(n = 5, amplitude = 0.3) {
  faces <- matrix(0L, 0L, 3L)
  for (i in 0:(n - 2)) {
    for (j in 0:(n - 2)) {
      faces <- rbind(
        faces,
        c(i * n + j, (i + 1) * n + j, (i + 1) * n + j + 1),
        c(i * n + j, (i + 1) * n + j + 1, i * n + j + 1)
      )
    }
  }
  verts <- matrix(0, n * n, 3)
  for (i in 0:(n - 1)) {
    for (j in 0:(n - 1)) {
      verts[i * n + j + 1, 1] <- i
      verts[i * n + j + 1, 2] <- j
    }
  }
  verts[, 3] <- ifelse(seq_len(n * n) %% 2 == 0, amplitude, -amplitude)
  list(faces = faces, verts = verts)
}

# A closed UV sphere: valence ~6, no boundary, and a volume worth correcting.
.uv_sphere <- function(n_lat = 16, n_lon = 16) {
  v <- numeric(0)
  for (i in 0:(n_lat - 1)) {
    theta <- pi * (i + 0.5) / n_lat
    for (j in 0:(n_lon - 1)) {
      phi <- 2 * pi * j / n_lon
      v <- c(v, sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta))
    }
  }
  id <- function(i, j) i * n_lon + (j %% n_lon)
  f <- numeric(0)
  for (i in 0:(n_lat - 2)) {
    for (j in 0:(n_lon - 1)) {
      f <- c(
        f,
        id(i, j), id(i + 1, j), id(i + 1, j + 1),
        id(i, j), id(i + 1, j + 1), id(i, j + 1)
      )
    }
  }
  list(
    faces = matrix(f, ncol = 3, byrow = TRUE),
    verts = matrix(v, ncol = 3, byrow = TRUE)
  )
}

# Enclosed volume by the divergence theorem, about the mesh's own centroid.
.volume <- function(faces, verts) {
  p <- sweep(verts, 2, colMeans(verts))
  a <- p[faces[, 1] + 1, ]
  b <- p[faces[, 2] + 1, ]
  cc <- p[faces[, 3] + 1, ]
  cross <- cbind(
    a[, 2] * b[, 3] - a[, 3] * b[, 2],
    a[, 3] * b[, 1] - a[, 1] * b[, 3],
    a[, 1] * b[, 2] - a[, 2] * b[, 1]
  )
  sum(rowSums(cross * cc)) / 6
}


test_that("smooth_mesh returns the same vertices in the same order", {
  m <- .noisy_grid()

  for (method in c("taubin", "laplacian", "humphrey")) {
    for (weights in c("uniform", "inverse_distance", "cotangent")) {
      out <- smooth_mesh(m$faces, m$verts, method = method, weights = weights)
      expect_equal(dim(out), dim(m$verts))
      expect_true(all(is.finite(out)))
      expect_false(isTRUE(all.equal(out, m$verts)))
    }
  }

  # Zero iterations is the identity, whatever else was asked for.
  expect_equal(smooth_mesh(m$faces, m$verts, iterations = 0L), m$verts)
})

test_that("smooth_mesh removes displacement out of the plane", {
  m <- .noisy_grid(9L)
  out <- smooth_mesh(m$faces, m$verts, method = "laplacian", iterations = 20L)
  expect_lt(sum(out[, 3]^2), 0.06 * sum(m$verts[, 3]^2))
})

test_that("a flat grid is an exact fixed point", {
  # Also what pins down what *pinning* means. Not merely that a frozen vertex ends
  # where it started - a filter that let the rim wander mid-iteration and put it back
  # afterwards would satisfy that - but that it never acts on its neighbours from
  # anywhere else. Get that wrong and the interior next to the rim is dragged by an
  # excursion that officially never happened.
  m <- .noisy_grid(7L, amplitude = 0)
  for (method in c("taubin", "laplacian", "humphrey")) {
    out <- smooth_mesh(m$faces, m$verts,
      method = method, iterations = 25L,
      preserve_border = TRUE
    )
    expect_identical(out, m$verts, info = method)
  }
})

test_that("lock and preserve_border reach the core", {
  m <- .noisy_grid(5L)

  # The rim: an endpoint of an edge used by exactly one face.
  out <- smooth_mesh(m$faces, m$verts, preserve_border = TRUE, iterations = 10L)
  expect_identical(out[1, ], m$verts[1, ])
  expect_false(isTRUE(all.equal(out[13, ], m$verts[13, ])))

  # An interior vertex, which `preserve_border` does not cover.
  lock <- rep(FALSE, nrow(m$verts))
  lock[13] <- TRUE
  out <- smooth_mesh(m$faces, m$verts, lock = lock, iterations = 10L)
  expect_identical(out[13, ], m$verts[13, ])

  # The two are a union, not alternatives.
  out <- smooth_mesh(m$faces, m$verts,
    preserve_border = TRUE, lock = lock,
    iterations = 5L
  )
  expect_identical(out[13, ], m$verts[13, ])
  expect_identical(out[1, ], m$verts[1, ])
})

test_that("volume_correction restores the volume without moving the mesh", {
  s <- .uv_sphere()
  v0 <- .volume(s$faces, s$verts)

  plain <- smooth_mesh(s$faces, s$verts, method = "laplacian", iterations = 10L)
  expect_lt(.volume(s$faces, plain), 0.75 * v0)

  fixed <- smooth_mesh(s$faces, s$verts,
    method = "laplacian", iterations = 10L,
    volume_correction = TRUE
  )
  expect_equal(.volume(s$faces, fixed) / v0, 1, tolerance = 1e-9)
  # The correction changes size and provably not position - which is the whole
  # difference from `trimesh`, whose scaling about the origin displaces a neuron by
  # more than twice its own diameter.
  expect_equal(colMeans(fixed), colMeans(plain), tolerance = 1e-9)
})

test_that("taubin holds the volume laplacian loses", {
  s <- .uv_sphere()
  v0 <- .volume(s$faces, s$verts)
  taubin <- smooth_mesh(s$faces, s$verts, method = "taubin", iterations = 20L)
  expect_gt(.volume(s$faces, taubin), 0.95 * v0)
})

test_that("smoothing is translation equivariant", {
  # `trimesh` fails this outright at this offset: its volume constraint scales about
  # the origin, so the ratio goes negative and the cube root returns NaN.
  s <- .uv_sphere()
  offset <- 1e5
  here <- smooth_mesh(s$faces, s$verts, iterations = 10L, volume_correction = TRUE)
  there <- smooth_mesh(s$faces, s$verts + offset,
    iterations = 10L,
    volume_correction = TRUE
  )
  expect_lt(max(abs(here - (there - offset))), 1e-8)
})

test_that("an undecidable volume warns through R's condition system", {
  # A flat sheet: both signed volumes are exactly zero, so the ratio has no cube root.
  m <- .noisy_grid(6L)
  expect_warning(
    out <- smooth_mesh(m$faces, m$verts,
      method = "laplacian", iterations = 5L,
      volume_correction = TRUE
    ),
    "no usable enclosed volume"
  )
  # Undefined means unscaled, not unsmoothed.
  expect_true(all(is.finite(out)))

  # And a closed mesh is silent, so the warning keeps meaning something.
  s <- .uv_sphere()
  expect_no_warning(
    smooth_mesh(s$faces, s$verts, iterations = 5L, volume_correction = TRUE)
  )
})

test_that("the documented choices match the constants", {
  # `clustering.R` explains why the vectors are spelled out twice - so `?smooth_mesh`
  # shows real choices - and notes that a test has to pin the copies together. This is
  # that test.
  expect_equal(eval(formals(smooth_mesh)$method), .SMOOTH_METHODS)
  expect_equal(eval(formals(smooth_mesh)$weights), .SMOOTH_WEIGHTS)
})

test_that("a parameter belonging to another method is refused, not dropped", {
  m <- .noisy_grid()
  expect_error(smooth_mesh(m$faces, m$verts, method = "taubin", alpha = 0.3))
  expect_error(smooth_mesh(m$faces, m$verts, method = "laplacian", mu = -0.53))
  expect_error(smooth_mesh(m$faces, m$verts, method = "humphrey", lambda = 0.5))
  expect_error(smooth_mesh(m$faces, m$verts, method = "nope"))
  expect_error(smooth_mesh(m$faces, m$verts, weights = "nope"))
  expect_error(smooth_mesh(m$faces, m$verts, method = "laplacian", lambda = 1.5))
  expect_error(smooth_mesh(m$faces, m$verts, method = "taubin", mu = -0.4))
  expect_error(smooth_mesh(m$faces, m$verts, iterations = -1L))
})

test_that("degenerate geometry is merely data", {
  # Duplicated faces, a face naming a vertex twice, a zero-area face, and a vertex
  # no face mentions - the shapes EM meshes are actually made of.
  faces <- matrix(
    c(
      0, 1, 2,
      0, 1, 2,
      1, 2, 3,
      4, 4, 5,
      5, 6, 7
    ),
    ncol = 3, byrow = TRUE
  )
  verts <- matrix(
    c(
      0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0.5,
      2, 0, 0, 3, 0, 0, 4, 0, 0, 5, 0, 0,
      9, 9, 9
    ),
    ncol = 3, byrow = TRUE
  )
  for (weights in c("uniform", "inverse_distance", "cotangent")) {
    out <- smooth_mesh(faces, verts, weights = weights, iterations = 5L)
    expect_true(all(is.finite(out)))
    expect_identical(out[9, ], verts[9, ]) # the unreferenced vertex
  }
})

test_that("threads do not change the result", {
  s <- .uv_sphere()
  args <- list(s$faces, s$verts,
    method = "taubin", weights = "cotangent",
    iterations = 8L, volume_correction = TRUE
  )
  expect_identical(
    do.call(smooth_mesh, c(args, list(threads = 1L))),
    do.call(smooth_mesh, c(args, list(threads = 4L)))
  )
})
