# Shared by the spatial-transform wrappers (cmtk.R, elastix.R, warp.R).

# Coerce whatever the user passed into an (N, 3) numeric matrix. Used by every transform
# wrapper in the package.
.xform_xyz <- function(xyz, arg = "xyz") {
  if (is.data.frame(xyz)) {
    # Prefer coordinate columns *by name* where they exist. Points and landmarks are usually
    # read straight out of a CSV or an SWC, where the coordinates are named rather than
    # positional and are rarely the only columns -- and `nat` writes them capitalised.
    # Falling back to positional order keeps a bare three-column frame working.
    nms <- tolower(names(xyz))
    if (all(c("x", "y", "z") %in% nms)) {
      xyz <- xyz[, match(c("x", "y", "z"), nms), drop = FALSE]
    }
    xyz <- as.matrix(xyz)
  }
  if (is.vector(xyz) && length(xyz) == 3L) xyz <- matrix(xyz, ncol = 3L)
  if (!is.matrix(xyz) || ncol(xyz) != 3L) {
    stop(sprintf("`%s` must be an (N, 3) matrix of 3D coordinates", arg))
  }
  storage.mode(xyz) <- "double"
  # `as.matrix()` on a data frame carries the column names through, so without this the
  # helper's return shape would depend on what it was handed. The Rust side reads the data
  # buffer and ignores dimnames either way; this is about the contract being one thing.
  dimnames(xyz) <- NULL
  xyz
}

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
