# Finding and closing the holes in a mesh.
#
# The algorithms themselves are tested exhaustively on the Rust side and against
# navis' own implementation from Python. What is worth testing here is the binding:
# that faces and logical masks survive the trip through extendr, that `trace_loops`
# hands back a list of two vectors rather than a matrix, and that the offsets it
# returns are the 0-based bounds `triangulate_rings` expects. All indices are
# 0-based, like the rest of the package.

# An `n x n` grid in the z = 0 plane, split along each cell's (0,0)->(1,1) diagonal.
# Its whole outer edge is boundary -- 4 * (n - 1) half-edges -- and nothing inside is.
.grid <- function(n = 5) {
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
  list(faces = faces, verts = verts)
}

# Ring `i` (1-based) out of what `trace_loops` returned.
.ring <- function(loops, i) {
  loops$rings[(loops$offsets[i] + 1):loops$offsets[i + 1]]
}

test_that("a closed mesh has no boundary", {
  tetra <- rbind(c(0, 2, 1), c(0, 1, 3), c(0, 3, 2), c(1, 2, 3))
  expect_equal(nrow(boundary_halfedges(tetra)), 0)
})

test_that("boundary_halfedges finds the grid's outer edge", {
  n <- 6
  g <- .grid(n)
  b <- boundary_halfedges(g$faces)

  expect_equal(nrow(b), 4 * (n - 1))
  expect_equal(ncol(b), 2)
  expect_true(is.integer(b))
  # Every boundary vertex is on the outer edge of the grid.
  v <- unique(as.vector(b))
  i <- v %/% n
  j <- v %% n
  expect_true(all(i == 0 | j == 0 | i == n - 1 | j == n - 1))
})

test_that("boundary_halfedges takes integer and double faces alike", {
  g <- .grid(5)
  expect_equal(
    boundary_halfedges(g$faces),
    boundary_halfedges(matrix(as.numeric(g$faces), ncol = 3))
  )
})

test_that("threads do not change the answer", {
  g <- .grid(7)
  ref <- boundary_halfedges(g$faces)
  expect_equal(boundary_halfedges(g$faces, threads = 1), ref)
  expect_equal(boundary_halfedges(g$faces, threads = 4), ref)
})

test_that("exposed_halfedges reports the edge opposite a dropped corner", {
  # Two triangles sharing (1, 2). Dropping vertex 0 kills the first face and leaves
  # (1, 2) with only the second on it, wound the way that one winds it.
  faces <- rbind(c(0, 1, 2), c(1, 3, 2))

  expect_equal(
    exposed_halfedges(faces, c(TRUE, FALSE, FALSE, FALSE)),
    matrix(c(2L, 1L), nrow = 1)
  )
  expect_equal(
    exposed_halfedges(faces, c(FALSE, FALSE, FALSE, TRUE)),
    matrix(c(1L, 2L), nrow = 1)
  )
})

test_that("exposed_halfedges leaves a pre-existing boundary alone", {
  # A lone triangle: all three edges are boundary already, so dropping a corner
  # exposes nothing new -- there is no face left to expose it.
  expect_equal(
    nrow(exposed_halfedges(matrix(c(0, 1, 2), nrow = 1), c(TRUE, FALSE, FALSE))),
    0
  )
})

test_that("exposed_halfedges is the new part of the subset's boundary", {
  n <- 7
  g <- .grid(n)
  dropped <- rep(FALSE, n * n)
  dropped[3 * n + 3 + 1] <- TRUE # one interior vertex, 0-based -> 1-based

  exposed <- exposed_halfedges(g$faces, dropped)
  expect_true(nrow(exposed) > 0)

  keep <- !apply(g$faces, 1, function(f) any(dropped[f + 1]))
  after <- boundary_halfedges(g$faces[keep, , drop = FALSE])
  before <- boundary_halfedges(g$faces)

  key <- function(m) paste(m[, 1], m[, 2], sep = "-")
  expect_setequal(key(exposed), setdiff(key(after), key(before)))
})

test_that("exposed_halfedges rejects a mask that is too short", {
  # As everywhere else in the package, a bad argument to a directly-exported
  # extendr function surfaces as a panic (the explanation goes to stderr) rather
  # than as an R condition carrying the message -- `unique_edges` and `bridges`
  # behave the same way. What matters is that it is refused, not silently read
  # out of bounds.
  g <- .grid(5)
  expect_error(exposed_halfedges(g$faces, rep(FALSE, 4)), "exposed_halfedges")
})

test_that("trace_loops walks the boundary into one ring", {
  n <- 5
  g <- .grid(n)
  loops <- trace_loops(boundary_halfedges(g$faces))

  expect_named(loops, c("rings", "offsets"))
  expect_equal(length(loops$offsets), 2) # one ring
  expect_equal(loops$offsets, c(0L, 4L * (n - 1L)))
  expect_equal(length(.ring(loops, 1)), 4 * (n - 1))
  expect_equal(length(unique(.ring(loops, 1))), 4 * (n - 1))
})

test_that("trace_loops covers more than a cycle basis", {
  # Two triangles meeting at vertex 0 -- a non-manifold boundary vertex.
  he <- rbind(c(0, 1), c(1, 2), c(2, 0), c(0, 3), c(3, 4), c(4, 0))
  loops <- trace_loops(he)

  expect_equal(length(loops$offsets), 3) # two rings
  expect_equal(length(loops$rings), 6) # every half-edge used exactly once
})

test_that("trace_loops abandons a dead end instead of hanging", {
  loops <- trace_loops(rbind(c(0, 1), c(1, 2), c(2, 3)))

  expect_equal(length(loops$rings), 0)
  expect_equal(loops$offsets, 0L)
})

test_that("triangulate_rings winds the cap against its ring", {
  # A unit square in z = 0, wound counter-clockwise seen from +z, so the cap has
  # to wind the other way.
  verts <- rbind(c(0, 0, 0), c(1, 0, 0), c(1, 1, 0), c(0, 1, 0))
  caps <- triangulate_rings(0:3, c(0L, 4L), verts)

  expect_equal(nrow(caps), 2)
  expect_true(is.integer(caps))
  for (k in seq_len(nrow(caps))) {
    p <- verts[caps[k, ] + 1, ]
    e1 <- p[2, ] - p[1, ]
    e2 <- p[3, ] - p[1, ]
    normal_z <- e1[1] * e2[2] - e1[2] * e2[1]
    expect_lt(normal_z, 0)
  }
})

test_that("capping the grid closes it", {
  n <- 6
  g <- .grid(n)
  loops <- trace_loops(boundary_halfedges(g$faces))
  caps <- triangulate_rings(loops$rings, loops$offsets, g$verts)

  expect_equal(nrow(caps), 4 * (n - 1) - 2) # a ring of k caps to k - 2
  expect_equal(nrow(boundary_halfedges(rbind(g$faces, caps))), 0)
})

test_that("a degenerate ring still closes", {
  # Collinear vertices name no plane at all -- the fan is the last resort.
  verts <- rbind(c(0, 0, 0), c(1, 0, 0), c(2, 0, 0), c(3, 0, 0))
  caps <- triangulate_rings(0:3, c(0L, 4L), verts)

  expect_equal(nrow(caps), 2)
  expect_setequal(as.vector(caps), 0:3)
})

test_that("malformed offsets are refused", {
  # See the note above on how extendr surfaces these.
  verts <- matrix(0, 4, 3)
  expect_error(triangulate_rings(0:2, integer(0), verts), "triangulate_rings")
  expect_error(triangulate_rings(0:2, c(0L, 2L), verts), "triangulate_rings")
  expect_error(triangulate_rings(0:2, c(0L, 2L, 1L, 3L), verts), "triangulate_rings")
  expect_error(triangulate_rings(c(0L, 1L, 99L), c(0L, 3L), verts), "triangulate_rings")
})
