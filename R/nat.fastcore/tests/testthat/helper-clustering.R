# Cluster ids are arbitrary labels, so two clusterings agree when they induce the
# same *partition* of the observations, not the same numbers.
.same_partition <- function(a, b) {
  split_by <- function(x) {
    groups <- lapply(unique(x), function(i) sort(which(x == i)))
    sort(vapply(groups, paste, character(1), collapse = ","))
  }
  identical(split_by(a), split_by(b))
}
