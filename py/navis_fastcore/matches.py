"""Extracting top matches from a score matrix (e.g. NBLAST output)."""

import numpy as np

from . import _fastcore

__all__ = ["top_matches", "matches_above", "count_matches"]

#: Widths the kernels read natively. A score matrix is *never* cast: at a few 100k on a
#: side, `np.asarray(m, dtype=np.float32)` on a float64 (or memory-mapped) matrix would
#: quietly materialise tens of GB.
_DTYPES = (np.float16, np.float32, np.float64)


def _prep(scores, axis, skip_self):
    """Validate `scores` and turn `skip_self` into the per-group index array Rust wants.

    Deliberately validates rather than coerces: the whole point of this module is that the
    matrix is borrowed, not copied.
    """
    # A DataFrame is a courtesy; we still hand back arrays, never labels.
    scores = getattr(scores, "values", scores)

    if not isinstance(scores, np.ndarray):
        raise TypeError(f"`scores` must be a numpy array, got {type(scores).__name__}")
    if scores.ndim != 2:
        raise ValueError(f"`scores` must be 2-dimensional, got {scores.ndim}D")
    if scores.dtype.type not in _DTYPES:
        raise TypeError(
            f"`scores` must be float16, float32 or float64, got {scores.dtype}. "
            "Cast it yourself if you meant to - fastcore will not copy a matrix "
            "this size on your behalf."
        )
    if not (scores.flags.c_contiguous or scores.flags.f_contiguous):
        raise ValueError(
            "`scores` must be C- or F-contiguous (got a strided view, e.g. from slicing). "
            "Use `np.ascontiguousarray` if you are sure you can afford the copy."
        )
    if axis not in (0, 1):
        raise ValueError(f"`axis` must be 0 or 1, got {axis!r}")

    n_groups = scores.shape[axis]

    if skip_self is False or skip_self is None:
        skip = None
    elif skip_self is True:
        if scores.shape[0] != scores.shape[1]:
            raise ValueError(
                "`skip_self=True` assumes the diagonal is the self-match, which needs a "
                f"square matrix; got {scores.shape}. Pass an explicit index array instead "
                "(-1 where a group has no self-match)."
            )
        skip = np.arange(n_groups, dtype=np.int64)
    else:
        skip = np.ascontiguousarray(skip_self, dtype=np.int64)
        if skip.shape != (n_groups,):
            raise ValueError(
                f"`skip_self` must have one entry per group: expected ({n_groups},), "
                f"got {skip.shape}"
            )

    return scores, skip


def _one_criterion(threshold, percentage):
    if (threshold is None) == (percentage is None):
        raise ValueError("Provide exactly one of `threshold` or `percentage`.")
    return (
        None if threshold is None else float(threshold),
        None if percentage is None else float(percentage),
    )


def top_matches(scores, n, axis=0, distances=False, skip_self=False, n_cores=None,
                progress=False):
    """Extract the `n` best matches for each query (or target).

    The matrix is neither copied nor transposed, whichever `axis` you ask for - matches
    along `axis=1` are served by striding the row-major buffer in column blocks, so it is
    read exactly once either way. That matters: these matrices run to tens of GB.

    Parameters
    ----------
    scores :        (n_query, n_target) np.ndarray
                    Score matrix, float16/32/64. Must be C- or F-contiguous. Not copied.
    n :             int
                    How many matches to extract per group. Must be <= the length of the
                    scanned axis.
    axis :          0 | 1
                    0 = one row of matches per query (per row of `scores`); 1 = per target.
    distances :     bool
                    If True, *lower* is better (a distance matrix rather than similarity).
    skip_self :     bool | (n_groups,) array
                    Exclude each group's self-match. True uses the diagonal (requires a
                    square matrix); an array gives the index to skip per group, -1 for none.
    n_cores :       int, optional
                    Cap the worker count. Default: all cores.
    progress :      bool

    Returns
    -------
    indices :       (n_groups, n) int64
                    Index along the *other* axis, best match first. -1 where the group had
                    fewer than `n` valid (non-NaN) cells.
    values :        (n_groups, n)
                    The matching scores, in the dtype of `scores`. NaN where the paired
                    index is -1.

    Examples
    --------
    >>> import numpy as np
    >>> import navis_fastcore as fastcore
    >>> scores = np.array([[1.0, 0.2, 0.9],
    ...                    [0.2, 1.0, 0.4],
    ...                    [0.9, 0.4, 1.0]], dtype=np.float32)
    >>> idx, val = fastcore.top_matches(scores, 2)
    >>> idx
    array([[0, 2],
           [1, 2],
           [2, 0]])

    Ignore the self-hits on the diagonal:

    >>> idx, val = fastcore.top_matches(scores, 1, skip_self=True)
    >>> idx.ravel()
    array([2, 2, 0])
    >>> val.ravel()
    array([0.9, 0.4, 0.9], dtype=float32)

    """
    scores, skip = _prep(scores, axis, skip_self)
    return _fastcore.top_matches(
        scores,
        int(n),
        int(axis),
        bool(distances),
        skip,
        None if n_cores is None else int(n_cores),
        bool(progress),
    )


def matches_above(scores, threshold=None, percentage=None, axis=0, distances=False,
                  skip_self=False, max_matches=None, n_cores=None, progress=False):
    """Extract every match clearing a cutoff. Ragged, so returned CSR-style.

    Give exactly one of:

    - `threshold` - an absolute cutoff: keep every cell `>= threshold` (`<=` if
      `distances`).
    - `percentage` - a band around each group's *own* best value: `percentage=0.05` keeps
      everything within 5% of that group's top match. Note this is "within X% of the best",
      not "the top X%".

    Parameters
    ----------
    scores :        (n_query, n_target) np.ndarray
                    Score matrix, float16/32/64. Must be C- or F-contiguous. Not copied.
    threshold :     float, optional
    percentage :    float, optional
                    In [0, 1].
    axis :          0 | 1
    distances :     bool
                    If True, *lower* is better.
    skip_self :     bool | (n_groups,) array
                    See [`top_matches`][navis_fastcore.top_matches].
    max_matches :   int, optional
                    Refuse to allocate more than this many matches. The count is known
                    before anything is allocated, so an over-broad cutoff raises instead of
                    taking the machine down with it. See
                    [`count_matches`][navis_fastcore.count_matches] to size a result first.
    n_cores :       int, optional
    progress :      bool

    Returns
    -------
    offsets :       (n_groups + 1,) int64
                    Group `g`'s matches are `indices[offsets[g]:offsets[g + 1]]`.
    indices :       (total,) uint32
                    Index along the other axis, best first within each group.
    values :        (total,)
                    The matching scores, in the dtype of `scores`.

    Examples
    --------
    >>> import numpy as np
    >>> import navis_fastcore as fastcore
    >>> scores = np.array([[1.0, 0.2, 0.9],
    ...                    [0.2, 1.0, 0.4],
    ...                    [0.9, 0.4, 1.0]], dtype=np.float32)
    >>> offsets, indices, values = fastcore.matches_above(scores, threshold=0.5)
    >>> offsets
    array([0, 2, 3, 5])
    >>> indices[offsets[0]:offsets[1]]   # query 0's matches, best first
    array([0, 2], dtype=uint32)

    Expand to a long-format table without a Python loop:

    >>> counts = np.diff(offsets)
    >>> query = np.repeat(np.arange(len(counts)), counts)
    >>> rank = np.arange(len(indices)) - np.repeat(offsets[:-1], counts)
    >>> query
    array([0, 0, 1, 2, 2])
    >>> rank
    array([0, 1, 0, 0, 1])

    """
    threshold, percentage = _one_criterion(threshold, percentage)
    scores, skip = _prep(scores, axis, skip_self)
    return _fastcore.matches_above(
        scores,
        threshold,
        percentage,
        int(axis),
        bool(distances),
        skip,
        None if max_matches is None else int(max_matches),
        None if n_cores is None else int(n_cores),
        bool(progress),
    )


def count_matches(scores, threshold=None, percentage=None, axis=0, distances=False,
                  skip_self=False, n_cores=None):
    """Count the matches each group *would* yield, without materialising them.

    The counting half of [`matches_above`][navis_fastcore.matches_above] on its own. Use it
    to size a result - or to pick a cutoff - on a matrix you cannot afford to guess wrong
    about.

    Parameters
    ----------
    scores :        (n_query, n_target) np.ndarray
                    Score matrix, float16/32/64. Must be C- or F-contiguous. Not copied.
    threshold :     float, optional
    percentage :    float, optional
                    Exactly one of the two, as for `matches_above`.
    axis :          0 | 1
    distances :     bool
    skip_self :     bool | (n_groups,) array
    n_cores :       int, optional

    Returns
    -------
    counts :        (n_groups,) int64

    Examples
    --------
    >>> import numpy as np
    >>> import navis_fastcore as fastcore
    >>> scores = np.array([[1.0, 0.2, 0.9],
    ...                    [0.2, 1.0, 0.4],
    ...                    [0.9, 0.4, 1.0]], dtype=np.float32)
    >>> fastcore.count_matches(scores, threshold=0.5)
    array([2, 1, 2])

    """
    threshold, percentage = _one_criterion(threshold, percentage)
    scores, skip = _prep(scores, axis, skip_self)
    return _fastcore.count_matches(
        scores,
        threshold,
        percentage,
        int(axis),
        bool(distances),
        skip,
        None if n_cores is None else int(n_cores),
    )
