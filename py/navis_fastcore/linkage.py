"""Hierarchical clustering of a score matrix (e.g. NBLAST output)."""

import numpy as np

from . import _fastcore

__all__ = ["linkage", "condensed_distances"]

#: Widths the kernels cluster natively. As in `matches`, a score matrix is *never*
#: cast: at 100k a side that would quietly materialise tens of GB. float16 is absent
#: on purpose — linkage accumulates thousands of updates per cluster and float16 has
#: nowhere near the mantissa for it.
_DTYPES = (np.float32, np.float64)


def _check_scores(scores):
    """Validate a square score matrix. Validates rather than coerces, by design."""
    scores = getattr(scores, "values", scores)

    if not isinstance(scores, np.ndarray):
        raise TypeError(f"`scores` must be a numpy array, got {type(scores).__name__}")
    if scores.ndim != 2:
        raise ValueError(f"`scores` must be 2-dimensional, got {scores.ndim}D")
    if scores.shape[0] != scores.shape[1]:
        raise ValueError(
            f"`scores` must be square to be clustered, got {scores.shape}"
        )
    if scores.dtype.type not in _DTYPES:
        raise TypeError(
            f"`scores` must be float32 or float64, got {scores.dtype}. Cast it "
            "yourself if you meant to - fastcore will not copy a matrix this size "
            "on your behalf."
        )
    if not (scores.flags.c_contiguous or scores.flags.f_contiguous):
        raise ValueError(
            "`scores` must be C- or F-contiguous (got a strided view, e.g. from "
            "slicing). Use `np.ascontiguousarray` if you are sure you can afford "
            "the copy."
        )
    return scores


def condensed_distances(
    scores, symmetry="mean", transform="one_minus", n_cores=None
):
    """Turn a square score matrix into a condensed distance vector.

    Symmetrisation, the similarity-to-distance transform and the condensing all
    happen in a **single fused pass**, so the only allocation is the ``n(n-1)/2``
    output itself. The numpy equivalent::

        cond = squareform(1 - (M + M.T) / 2, checks=False)

    materialises three more ``n x n`` arrays on the way, which at 100k neurons is
    where the memory actually goes.

    The diagonal is never read, so a matrix carrying self-scores rather than zeros
    needs no fixing up first.

    Parameters
    ----------
    scores :    (n, n) float32 or float64 array
                Score matrix, typically from :func:`~navis_fastcore.nblast_allbyall`.
                Must be C- or F-contiguous; it is borrowed, never copied or cast.
    symmetry :  "mean" | "min" | "max" | "none"
                How to combine ``M[i, j]`` with ``M[j, i]``, since NBLAST is not
                symmetric. Mirrors the ``symmetry`` argument of
                :func:`~navis_fastcore.nblast_allbyall`. Use ``"none"`` when the
                matrix is already symmetric — it is also the fastest, since it reads
                the buffer strictly sequentially.
    transform : "one_minus" | "none"
                ``"one_minus"`` gives ``1 - score``, the usual NBLAST convention;
                ``"none"`` passes values through as distances unchanged.
    n_cores :   int, optional
                Thread cap. ``None`` uses all available cores.

    Returns
    -------
    condensed : (n * (n - 1) / 2, ) array
                Upper triangle in row-major order, i.e. the layout
                ``scipy.spatial.distance.squareform`` produces. Same dtype as
                ``scores``.

    Examples
    --------
    >>> import navis_fastcore as fastcore
    >>> import numpy as np
    >>> scores = np.array([[1.00, 0.75],
    ...                    [0.25, 1.00]], dtype=np.float32)
    >>> fastcore.condensed_distances(scores)
    array([0.5], dtype=float32)

    Already-symmetric distances can be passed straight through:

    >>> d = np.array([[0.0, 0.25],
    ...               [0.25, 0.0]], dtype=np.float32)
    >>> fastcore.condensed_distances(d, symmetry="none", transform="none")
    array([0.25], dtype=float32)

    """
    scores = _check_scores(scores)
    return _fastcore.condensed_distances(
        scores, symmetry=symmetry, transform=transform, n_cores=n_cores
    )


def linkage(
    x,
    method="ward",
    symmetry="mean",
    transform="one_minus",
    copy=True,
    n_cores=None,
):
    """Hierarchical clustering, returning a SciPy-compatible linkage matrix.

    Accepts either a square score matrix — in which case the whole pipeline
    (symmetrise, convert to distance, condense, cluster) runs fused, with no
    ``n x n`` temporary ever materialised — or an existing condensed distance
    vector.

    The result is interchangeable with ``scipy.cluster.hierarchy.linkage`` output
    and can be handed straight to ``fcluster``, ``dendrogram`` or ``cut_tree``.

    Two things make this cheaper than the SciPy pipeline at scale:

    - **float32 stays float32.** ``scipy.cluster.hierarchy.linkage`` up-casts its
      input to float64 unconditionally, so handing it a float32 matrix to save
      memory instead costs you a second, doubled copy of it.
    - **Nothing is copied.** The condensed matrix is clustered in place.

    Parameters
    ----------
    x :         (n, n) or (n * (n - 1) / 2, ) float32 or float64 array
                Either a square score matrix (``symmetry`` and ``transform`` apply)
                or a 1-D condensed distance vector (they are ignored).
    method :    str
                Linkage method: "single", "complete", "average", "weighted", "ward",
                "centroid" or "median". Same meanings as in SciPy.
    symmetry :  "mean" | "min" | "max" | "none"
                How to combine ``M[i, j]`` with ``M[j, i]``. Square input only.
    transform : "one_minus" | "none"
                ``"one_minus"`` gives ``1 - score``. Square input only.
    copy :      bool
                Condensed input only. Linkage consumes its input as scratch; with
                ``copy=True`` (the default) a copy is taken so yours survives. Pass
                ``False`` to cluster in place and halve peak memory — your array is
                left in an arbitrary state afterwards. Square input never copies,
                because the condensed buffer it builds is its own.
    n_cores :   int, optional
                Thread cap for the condensing / validation passes. The linkage
                itself is single-threaded.

    Returns
    -------
    Z :         (n - 1, 4) float64 array
                One merge per row as ``[cluster1, cluster2, distance, size]``,
                ordered by increasing distance. Singletons are labelled ``0..n``
                and the cluster formed at step ``i`` is labelled ``n + i``.

    Notes
    -----
    Clustering cannot be interrupted once it starts: it exposes no per-merge hook,
    so ``Ctrl-C`` is only honoured up to the end of the condensing pass. At 100k
    observations the linkage is the part that takes minutes.

    On float32 input the merge *heights* match a float64 run to about 1e-7
    relative. Where distances are very nearly tied, that rounding can also swap the
    order of two otherwise-equivalent merges; on structured data (what a real
    NBLAST matrix looks like) this does not happen, but on near-degenerate input it
    affects a small fraction of merges. Pass float64 if you need merge order to be
    reproducible against SciPy down to the last tie.

    Examples
    --------
    Two obvious pairs, from a score matrix:

    >>> import navis_fastcore as fastcore
    >>> import numpy as np
    >>> scores = np.array([[1.00, 0.90, 0.20, 0.10],
    ...                    [0.70, 1.00, 0.30, 0.15],
    ...                    [0.25, 0.35, 1.00, 0.80],
    ...                    [0.05, 0.10, 0.60, 1.00]], dtype=np.float32)
    >>> Z = fastcore.linkage(scores, method="average")
    >>> Z[:, :2].astype(int)
    array([[0, 1],
           [2, 3],
           [4, 5]])

    Cut it into two clusters, using SciPy as usual:

    >>> from scipy.cluster.hierarchy import fcluster
    >>> fcluster(Z, 2, criterion="maxclust")
    array([1, 1, 2, 2], dtype=int32)

    From an existing condensed distance vector:

    >>> cond = fastcore.condensed_distances(scores)
    >>> Z = fastcore.linkage(cond, method="average")
    >>> Z.shape
    (3, 4)

    """
    x = getattr(x, "values", x)

    if not isinstance(x, np.ndarray):
        raise TypeError(f"`x` must be a numpy array, got {type(x).__name__}")

    if x.ndim == 2:
        x = _check_scores(x)
        return _fastcore.linkage_from_scores(
            x,
            method=method,
            symmetry=symmetry,
            transform=transform,
            n_cores=n_cores,
        )

    if x.ndim != 1:
        raise ValueError(
            f"`x` must be a square score matrix or a condensed distance vector, "
            f"got a {x.ndim}D array"
        )

    if x.dtype.type not in _DTYPES:
        raise TypeError(
            f"`x` must be float32 or float64, got {x.dtype}. Cast it yourself if "
            "you meant to - fastcore will not copy an array this size on your "
            "behalf."
        )

    # Linkage clusters in place, so an un-copied array is consumed. Default to
    # protecting the caller's data; `copy=False` is the memory-minimal path.
    if copy:
        x = np.array(x, order="C", copy=True)
    else:
        # Deliberately no `ascontiguousarray` fallback here: it would copy, which is
        # the one thing `copy=False` exists to avoid.
        if not x.flags.c_contiguous:
            raise ValueError(
                "`copy=False` clusters in place, which needs a C-contiguous array; "
                "got a strided view. Pass `copy=True` if you can afford the copy."
            )
        if not x.flags.writeable:
            raise ValueError(
                "`copy=False` clusters in place but `x` is read-only. Either make "
                "it writeable or pass `copy=True`."
            )

    return _fastcore.linkage_condensed(x, method=method, n_cores=n_cores)
