# Shared by the spatial-transform wrappers (cmtk.R, elastix.R).

# Coerce `invert` into the 0/1 integer vector the Rust side takes, or `integer(0)` for the
# all-forward default. extendr cannot accept a logical vector as input, hence the integers.
.invert_flags <- function(invert, n, what = "registration") {
  invert <- as.logical(invert)
  if (anyNA(invert)) stop("`invert` must not contain NA")
  if (length(invert) == 1L) {
    if (!invert) {
      return(integer(0))
    }
    invert <- rep(TRUE, n)
  }
  if (length(invert) != n) {
    stop(sprintf(
      "`invert` must be length 1 or one flag per %s (%d), got %d",
      what, n, length(invert)
    ))
  }
  if (!any(invert)) integer(0) else as.integer(invert)
}

# Coerce `fallback_to_affine` into the string the Rust side takes. `TRUE` means "chain" -- the
# nat/navis semantics -- so the plain logical spelling stays the faithful one and the departure
# has to be named.
.fallback_mode <- function(fallback) {
  if (is.logical(fallback) && length(fallback) == 1L && !is.na(fallback)) {
    return(if (fallback) "chain" else "none")
  }
  if (is.character(fallback) && length(fallback) == 1L && fallback %in% c("none", "chain", "hop")) {
    return(fallback)
  }
  stop('`fallback_to_affine` must be FALSE, TRUE, "chain" or "hop"')
}
