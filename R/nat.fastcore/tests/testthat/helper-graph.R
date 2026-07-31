# Fixtures shared by test-dag.R and test-graph.R.
#
# Everything here is 0-based, like the bindings themselves: a parent vector holds
# 0-based node indices with `< 0` at the roots, and an edge matrix holds 0-based
# node indices. That is the one thing most likely to trip an R caller up, so the
# fixtures are written out explicitly rather than generated.

# A path 0-1-2-3, a lone pair 4-5, and an isolated node 6.
.forest_parents <- function() c(-1L, 0L, 1L, 2L, -1L, 4L, -1L)

# A branching arbor:
#        0
#        |
#        1
#       / \
#      2   4
#      |   |
#      3   5
.arbor_parents <- function() c(-1L, 0L, 1L, 2L, 1L, 4L)

# Two triangles sharing the 1-2 edge, forming a unit square in the z = 0 plane.
.square_faces <- function() matrix(c(0, 1, 2, 1, 2, 3), ncol = 3, byrow = TRUE)

.square_verts <- function() {
  matrix(c(0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0), ncol = 3, byrow = TRUE)
}

# An edge matrix from a flat run of node indices, so the fixtures and the expected
# values below both read as edges rather than as reshape calls.
.edges <- function(...) matrix(c(...), ncol = 2, byrow = TRUE)

# A ring 0-1-2-3-0 with a tail 3-4.
.ring_with_tail <- function() .edges(0, 1, 1, 2, 2, 3, 3, 0, 3, 4)
