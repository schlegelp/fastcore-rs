import numpy as np

from collections import namedtuple

from . import _fastcore

__all__ = [
    "nblast_allbyall",
    "nblast",
    "nblast_knn",
    "nblast_smart",
    "synblast",
    "Synapses",
]

#: Minimal dotprop container: `points` (N, 3), unit tangent `vect` (N, 3) and an
#: optional per-point `alpha` (N,) used only when ``use_alpha=True``.
Dotprop = namedtuple("Dotprop", ["points", "vect", "alpha"], defaults=[None])

#: Minimal synapse container for `synblast`: `connectors` is an ``(N, 3)`` or
#: ``(N, 4)`` array of ``[x, y, z, (type)]``. The optional 4th column is a numeric
#: connector type (e.g. ``0`` = presynapse, ``1`` = postsynapse) used when
#: ``by_type=True`` / ``cn_types`` restrict which synapses compare against which.
Synapses = namedtuple("Synapses", ["connectors"])

#: Accepted ``precision`` values -> bit width. The scoring math always runs in
#: float64; ``precision`` only sets the dtype the result is stored at.
_PRECISION = {
    16: 16,
    32: 32,
    64: 64,
    "half": 16,
    "single": 32,
    "double": 64,
    np.float16: 16,
    np.float32: 32,
    np.float64: 64,
}


def _as_clouds(dotprops, want_alpha=False):
    """Validate dotprops and extract contiguous float64 arrays.

    Returns ``(points, vects, alphas)`` where ``alphas`` is ``None`` unless
    ``want_alpha`` is set, in which case each dotprop must expose a per-point
    ``alpha`` array of matching length.
    """
    dps = list(dotprops)
    for n in dps:
        if not hasattr(n, "points") or not hasattr(n, "vect"):
            raise TypeError(
                "Expected an iterable of dotprop-likes with `points` and `vect` "
                "attributes."
            )
    points = [np.ascontiguousarray(n.points, dtype=np.float64) for n in dps]
    vects = [np.ascontiguousarray(n.vect, dtype=np.float64) for n in dps]

    alphas = None
    if want_alpha:
        alphas = []
        for n, pts in zip(dps, points):
            a = getattr(n, "alpha", None)
            if a is None:
                raise ValueError(
                    "use_alpha=True requires every dotprop to expose a per-point "
                    "`alpha` array; got a dotprop without one."
                )
            a = np.ascontiguousarray(np.asarray(a, dtype=np.float64).ravel())
            if a.shape[0] != pts.shape[0]:
                raise ValueError(
                    "`alpha` length must match the number of points for every dotprop."
                )
            alphas.append(a)
    return points, vects, alphas


def _smat_args(smat):
    """Normalize `smat` to `(values, dist_edges, dot_edges)` float64 arrays.

    Accepts:
    - ``None`` -> ``(None, None, None)``; the Rust side uses the embedded FCWB matrix.
    - a navis ``Lookup2d`` (has ``.cells`` and ``.axes`` of ``Digitizer`` with
      ``.boundaries``); the ``n + 1`` boundaries are trimmed to the ``n`` left
      edges we bin against (the leading ``-inf`` is harmless for non-negative
      distances / dot products).
    - a ``(values, dist_edges, dot_edges)`` tuple of left bin edges.
    """
    if smat is None:
        return (None, None, None)

    if hasattr(smat, "cells") and hasattr(smat, "axes"):
        values = np.ascontiguousarray(smat.cells, dtype=np.float64)
        dist_edges = np.asarray(smat.axes[0].boundaries, dtype=np.float64)[:-1]
        dot_edges = np.asarray(smat.axes[1].boundaries, dtype=np.float64)[:-1]
        return (values, np.ascontiguousarray(dist_edges), np.ascontiguousarray(dot_edges))

    values, dist_edges, dot_edges = smat
    return (
        np.ascontiguousarray(values, dtype=np.float64),
        np.ascontiguousarray(dist_edges, dtype=np.float64),
        np.ascontiguousarray(dot_edges, dtype=np.float64),
    )


def _resolve_limit(limit_dist, smat_args, use_alpha):
    """Resolve `limit_dist` (``None`` | number | ``"auto"``) to a float or ``None``.

    ``"auto"`` mirrors navis: ``1.05 *`` the left edge of the last distance bin
    of the (resolved) scoring matrix. With no explicit matrix, ``use_alpha``
    selects the alpha-calibrated default, matching the matrix the scores use.
    """
    if limit_dist is None:
        return None
    if isinstance(limit_dist, str):
        if limit_dist == "auto":
            sv, de, ve = smat_args
            return _fastcore.smat_auto_limit(
                smat_values=sv, dist_edges=de, dot_edges=ve, use_alpha=use_alpha
            )
        raise ValueError(
            f"Unknown limit_dist {limit_dist!r}; expected a number, 'auto', or None."
        )
    return float(limit_dist)


def _resolve_precision(precision):
    """Return ``(user_bits, rust_bits)``.

    Rust emits float32 or float64 directly; a requested 16-bit output is produced
    by down-casting the float32 result in numpy.
    """
    bits = _PRECISION.get(precision)
    if bits is None:
        raise ValueError(
            f"Unknown precision {precision!r}; expected one of 16, 32, 64 "
            "(or 'half' / 'single' / 'double')."
        )
    return bits, (64 if bits == 64 else 32)


#: `symmetry` values that select a combine, mapped to the kernel's name for it.
#: ``True`` is a legacy spelling of ``"mean"``.
_SYMMETRY = {"mean": "mean", True: "mean", "min": "min", "max": "max"}


def _symmetry_name(symmetry):
    """Map a user-facing `symmetry` onto the kernel's name, or raise."""
    try:
        return _SYMMETRY[symmetry]
    except (KeyError, TypeError):
        raise ValueError(
            f"Unknown symmetry {symmetry!r}; expected 'forward', 'mean', 'min', "
            "'max' or None."
        ) from None


def _symmetrize(m, symmetry, n_cores=None):
    """Symmetrise a square score matrix against its own transpose, **in place**.

    Split out from `_combine` because this is the case numpy cannot do cheaply:
    `(M + M.T) / 2` builds two full ``n x n`` temporaries, and even
    `np.add(M, M.T, out=M)` still builds one, because numpy sees the output
    overlapping `M.T` and defensively copies the input first. The Rust kernel writes
    both triangles as it walks the upper one, so it allocates nothing at all — at
    100k neurons that is 80 GB of peak that simply stops existing.

    `m` must be the freshly-created score matrix (float32/float64, C-contiguous);
    it is modified and returned.
    """
    _fastcore.symmetrize(m, symmetry=_symmetry_name(symmetry), n_cores=n_cores)
    return m


def _combine(a, b, symmetry):
    """Combine two equally-shaped score matrices element-wise, **consuming `a`**.

    `a` is written to in place and returned. Every caller creates it immediately
    beforehand and drops its own reference, so nothing else can observe the
    mutation — and writing in place is what makes this free, where `(a + b) / 2`
    would build two full temporaries.

    `b` is a *different* array (typically a transposed view of the reverse-direction
    scores). Passing a view of `a` is not wrong — numpy falls back to buffering the
    overlap — but it silently costs a temporary, which is what `_symmetrize` exists
    to avoid.
    """
    name = _symmetry_name(symmetry)
    if name == "mean":
        np.add(a, b, out=a)
        a *= 0.5  # exact for binary floats, and unlike `/= 2` needs no temporary
    elif name == "min":
        np.minimum(a, b, out=a)
    else:
        np.maximum(a, b, out=a)
    return a


def nblast_allbyall(
    dotprops,
    smat=None,
    normalize=True,
    symmetry=None,
    use_alpha=False,
    limit_dist=None,
    n_cores=None,
    precision=32,
    progress=False,
):
    """All-by-all NBLAST.

    A long run can be interrupted with Ctrl-C / the Jupyter interrupt button;
    it stops promptly and raises ``KeyboardInterrupt``.

    Parameters
    ----------
    dotprops :   iterable of dotprop-likes
                 Each must expose `points` (N, 3) and unit tangent `vect` (N, 3).
                 When ``use_alpha`` is set, each must also expose `alpha` (N,).
    smat :       None | navis Lookup2d | (values, dist_edges, dot_edges)
                 Scoring matrix. ``None`` uses the embedded FCWB matrix.
    normalize :  bool
                 Divide each score by the query's self-hit (self-match == 1.0).
    symmetry :   None | 'forward' | 'mean' | 'min' | 'max'
                 Combine the forward matrix with its transpose. ``None`` /
                 ``'forward'`` returns the raw forward (asymmetric) matrix.
    use_alpha :  bool
                 Weight each dot product by ``sqrt(alpha_query * alpha_target)``,
                 emphasising locally linear (backbone) regions.
    limit_dist : None | float | 'auto'
                 Distance upper bound: a query point whose nearest neighbour is
                 farther than this is scored at the "far + orthogonal" corner of
                 the matrix. ``'auto'`` uses ``1.05 *`` the last distance bin edge.
    n_cores :    int | None
                 Cap the number of worker threads. ``None`` uses all cores.
    precision :  16 | 32 | 64
                 Dtype of the returned matrix. The scoring math is always float64.
    progress :   bool
                 Show progress bars (drawn from Rust to stderr): first over
                 building the neuron indices, then over the scoring pairs.

    Returns
    -------
    np.ndarray
                 (n, n) score matrix; row = query, column = target.

    """
    points, vects, alphas = _as_clouds(dotprops, want_alpha=use_alpha)
    sv, de, ve = _smat_args(smat)
    limit = _resolve_limit(limit_dist, (sv, de, ve), use_alpha)
    user_bits, rust_bits = _resolve_precision(precision)

    M = np.asarray(
        _fastcore.nblast_allbyall(
            points,
            vects,
            alphas=alphas,
            smat_values=sv,
            dist_edges=de,
            dot_edges=ve,
            normalize=normalize,
            limit_dist=limit,
            n_cores=n_cores,
            precision=rust_bits,
            progress=progress,
        )
    )

    if symmetry is not None and symmetry != "forward":
        M = _symmetrize(M, symmetry, n_cores)
    if user_bits == 16:
        M = M.astype(np.float16)
    return M


def nblast(
    query,
    target,
    smat=None,
    normalize=True,
    symmetry=None,
    use_alpha=False,
    limit_dist=None,
    n_cores=None,
    precision=32,
    progress=False,
):
    """NBLAST every query neuron against every target neuron.

    A long run can be interrupted with Ctrl-C / the Jupyter interrupt button;
    it stops promptly and raises ``KeyboardInterrupt``.

    Parameters
    ----------
    query, target : iterable of dotprop-likes
                    Each must expose `points` (N, 3) and unit tangent `vect`
                    (N, 3); also `alpha` (N,) when ``use_alpha`` is set.
    smat :          None | navis Lookup2d | (values, dist_edges, dot_edges)
                    Scoring matrix. ``None`` uses the embedded FCWB matrix.
    normalize :     bool
                    Divide each score by the query's self-hit.
    symmetry :      None | 'forward' | 'mean' | 'min' | 'max'
                    ``None`` / ``'forward'`` returns the raw forward matrix.
                    The others combine it with the reverse (target-vs-query)
                    NBLAST, matching navis' ``scores`` argument.
    use_alpha :     bool
                    Weight each dot product by ``sqrt(alpha_query * alpha_target)``.
    limit_dist :    None | float | 'auto'
                    Distance upper bound (see ``nblast_allbyall``).
    n_cores :       int | None
                    Cap the number of worker threads. ``None`` uses all cores.
    precision :     16 | 32 | 64
                    Dtype of the returned matrix. Math is always float64.
    progress :      bool
                    Show progress bars over index building and scoring.

    Returns
    -------
    np.ndarray
                    (n_query, n_target) score matrix; row = query, column = target.

    """
    q_points, q_vects, q_alphas = _as_clouds(query, want_alpha=use_alpha)
    t_points, t_vects, t_alphas = _as_clouds(target, want_alpha=use_alpha)
    sv, de, ve = _smat_args(smat)
    limit = _resolve_limit(limit_dist, (sv, de, ve), use_alpha)
    user_bits, rust_bits = _resolve_precision(precision)

    def _forward(qp, qv, qa, tp, tv, ta):
        return np.asarray(
            _fastcore.nblast(
                qp,
                qv,
                tp,
                tv,
                q_alphas=qa,
                t_alphas=ta,
                smat_values=sv,
                dist_edges=de,
                dot_edges=ve,
                normalize=normalize,
                limit_dist=limit,
                n_cores=n_cores,
                precision=rust_bits,
                progress=progress,
            )
        )

    M = _forward(q_points, q_vects, q_alphas, t_points, t_vects, t_alphas)

    if symmetry is not None and symmetry != "forward":
        # Reverse NBLAST (target-as-query); its transpose aligns with M cell-wise.
        R = _forward(t_points, t_vects, t_alphas, q_points, q_vects, q_alphas)
        M = _combine(M, R.T, symmetry)
    if user_bits == 16:
        M = M.astype(np.float16)
    return M


#: `symmetry` values accepted by `nblast_knn`, mapped to the kernel's name.
_KNN_SYMMETRY = {
    None: "forward",
    "forward": "forward",
    "mean": "mean",
    True: "mean",
    "min": "min",
    "max": "max",
}


def nblast_knn(
    dotprops,
    target=None,
    k=20,
    symmetry="mean",
    n_candidates=200,
    voxel=20.0,
    n_dirs=3,
    splat=True,
    smat=None,
    normalize=True,
    use_alpha=False,
    limit_dist=None,
    n_cores=None,
    precision=32,
    progress=False,
):
    """The `k` nearest neighbours of every neuron, without the ``n x n`` matrix.

    An all-by-all is the wrong shape for a k-NN question at scale: 164k neurons is
    2.7e10 pairs and a 107 GB matrix, when the answer wanted from it is a 26 MB
    k-NN graph (typically to feed a UMAP embedding). This computes that graph
    directly, in three stages:

    1. each neuron becomes a coarse voxel-occupancy signature;
    2. the ``n_candidates`` most similar neurons per row are shortlisted from
       those signatures;
    3. the **exact** NBLAST score is computed for the shortlisted pairs only.

    The pre-filter is sound because the FCWB scoring matrix has finite support —
    beyond its last distance bin (40 um) every cell is ~ -10, so neurons that do
    not overlap in space score at that floor regardless of shape. Only *which*
    neurons make the shortlist is approximate; every returned score is an exact
    NBLAST value, because a neuron that belongs in the true top-`k` outranks the
    global k-th and so cannot be dropped by the rerank once shortlisted.

    Measured on 163,976 zebrafish neurons: recall@20 = 0.990 at the default
    ``n_candidates=200``, scoring 0.16% of pairs.

    A long run can be interrupted with Ctrl-C / the Jupyter interrupt button.

    Parameters
    ----------
    dotprops :     iterable of dotprop-likes
                   The queries. Each must expose `points` (N, 3) and unit tangent
                   `vect` (N, 3); also `alpha` (N,) when ``use_alpha`` is set.
    target :       iterable of dotprop-likes | None
                   If given, neighbours are searched among these instead of among
                   `dotprops`, and the returned `idx` indexes **`target`** — the
                   k-NN counterpart of ``nblast(query, target)``. Two things
                   differ from the ``None`` (all-by-all) case: nothing is excluded
                   from a row, so a neuron appearing in both sets matches itself
                   at 1.0; and the candidate shortlist is not symmetrically closed
                   (there is no "target proposes the query" side to close over),
                   which makes a given `n_candidates` slightly less generous than
                   in the all-by-all case — raise it by ~40% to match.
    k :            int
                   Neighbours to return per neuron.
    symmetry :     'mean' | 'forward' | 'min' | 'max' | None
                   How the two directions of a pair are combined **before** the
                   top-`k` cut. This matters more here than for a full matrix:
                   with a matrix you can symmetrise afterwards against the
                   transpose, but once only `k` neighbours per row are kept the
                   transpose is gone. The asymmetry is real — a small neuron
                   contained in a large one scores high one way and low the other
                   — so ``'mean'`` is the default. ``None`` / ``'forward'`` keeps
                   each row's own forward score.
    n_candidates : int
                   Shortlist size per neuron; the one recall/cost knob. Measured
                   recall@20 on 163,976 real neurons: 0.911 at 50, 0.969 at 100,
                   0.990 at 200, 0.996 at 400. The budget needed for a given
                   recall grows only about logarithmically with the number of
                   neurons.
    voxel :        float
                   Signature voxel edge, in the units of `points` (um for the
                   FCWB matrix). 10-20 measured equivalently.
    n_dirs :       int
                   Tangent-direction bins for the signature; 1 disables them.
    splat :        bool
                   Trilinearly spread each point over its 8 surrounding voxels
                   (worth ~0.05 recall@20).
    smat, normalize, use_alpha, limit_dist, n_cores, precision, progress :
                   As in `nblast` / `nblast_allbyall`.

    Returns
    -------
    idx :          np.ndarray
                   (n_query, k) int64 neighbour indices, descending by score —
                   into `target` if given, else into `dotprops`. Rows with fewer
                   than `k` candidates are padded with ``-1``.
    scores :       np.ndarray
                   (n_query, k) NBLAST similarities aligned to `idx`; padding is
                   ``-inf``. For UMAP's ``precomputed_knn`` pass ``1.0 - scores``
                   as the distances.

    Examples
    --------
    >>> idx, scores = nblast_knn(dotprops, k=20)                 # doctest: +SKIP
    >>> reducer = umap.UMAP(precomputed_knn=(idx, 1.0 - scores)) # doctest: +SKIP

    Search a small query set against a large reference set:

    >>> idx, scores = nblast_knn(queries, target=library, k=5)   # doctest: +SKIP

    """
    try:
        sym = _KNN_SYMMETRY[symmetry]
    except (KeyError, TypeError):
        raise ValueError(
            f"Unknown symmetry {symmetry!r}; expected 'mean', 'forward', 'min', "
            "'max' or None."
        ) from None

    points, vects, alphas = _as_clouds(dotprops, want_alpha=use_alpha)
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}.")

    t_points = t_vects = t_alphas = None
    if target is not None:
        t_points, t_vects, t_alphas = _as_clouds(target, want_alpha=use_alpha)

    if len(points) == 0:
        return (
            np.zeros((0, k), dtype=np.int64),
            np.zeros((0, k), dtype=np.float32 if precision != 64 else np.float64),
        )

    sv, de, ve = _smat_args(smat)
    limit = _resolve_limit(limit_dist, (sv, de, ve), use_alpha)
    user_bits, rust_bits = _resolve_precision(precision)

    idx, scores = _fastcore.nblast_knn(
        points,
        vects,
        alphas=alphas,
        t_points=t_points,
        t_vects=t_vects,
        t_alphas=t_alphas,
        k=k,
        n_candidates=n_candidates,
        symmetry=sym,
        voxel=float(voxel),
        n_dirs=n_dirs,
        splat=splat,
        smat_values=sv,
        dist_edges=de,
        dot_edges=ve,
        normalize=normalize,
        limit_dist=limit,
        n_cores=n_cores,
        precision=rust_bits,
        progress=progress,
    )
    idx = np.asarray(idx)
    scores = np.asarray(scores)
    if user_bits == 16:
        scores = scores.astype(np.float16)
    return idx, scores


def _clouds_to_dotprops(points, vects, alphas):
    """Wrap parallel ``points`` / ``vects`` (/ ``alphas``) arrays as `Dotprop`s."""
    if alphas is None:
        return [Dotprop(p, v) for p, v in zip(points, vects)]
    return [Dotprop(p, v, a) for p, v, a in zip(points, vects, alphas)]


def _downsample_clouds(points, vects, alphas, factor):
    """Keep every ``factor``-th point of each cloud (navis' ``Dotprops.downsample``).

    Slicing a non-empty cloud always keeps at least one point; ``build_index``'s
    complete-graph fallback keeps such tiny clouds valid.
    """
    step = max(int(factor), 1)
    dp = [p[::step] for p in points]
    dv = [v[::step] for v in vects]
    da = None if alphas is None else [a[::step] for a in alphas]
    return dp, dv, da


def _select_mask(coarse, t, criterion):
    """Boolean (n_query, n_target) mask of the best targets per query row.

    `criterion` mirrors navis: ``'percentile'`` keeps cells at/above the ``t``-th
    percentile of their row, ``'score'`` keeps cells ``>= t``, ``'N'`` keeps the top
    ``t`` targets per row.
    """
    if criterion == "percentile":
        sel = np.percentile(coarse, t, axis=1)
        return coarse >= sel[:, None]
    if criterion == "score":
        return coarse >= t
    if criterion == "N":
        n_target = coarse.shape[1]
        k = int(min(max(t, 0), n_target))
        mask = np.zeros(coarse.shape, dtype=bool)
        if k > 0:
            top = np.argsort(coarse, axis=1)[:, ::-1][:, :k]
            np.put_along_axis(mask, top, True, axis=1)
        return mask
    raise ValueError(
        f"Unknown criterion {criterion!r}; expected 'percentile', 'score' or 'N'."
    )


def _nblast_pairs(
    q_points, q_vects, q_alphas, t_points, t_vects, t_alphas, q_idx, t_idx,
    smat, normalize, use_alpha, limit_dist, n_cores, precision, progress,
):
    """Forward NBLAST of the selected `(q_idx[k], t_idx[k])` pairs; returns a 1-D array."""
    sv, de, ve = _smat_args(smat)
    limit = _resolve_limit(limit_dist, (sv, de, ve), use_alpha)
    user_bits, rust_bits = _resolve_precision(precision)
    scores = np.asarray(
        _fastcore.nblast_pairs(
            q_points,
            q_vects,
            t_points,
            t_vects,
            np.ascontiguousarray(q_idx, dtype=np.int64),
            np.ascontiguousarray(t_idx, dtype=np.int64),
            q_alphas=q_alphas,
            t_alphas=t_alphas,
            smat_values=sv,
            dist_edges=de,
            dot_edges=ve,
            normalize=normalize,
            limit_dist=limit,
            n_cores=n_cores,
            precision=rust_bits,
            progress=progress,
        )
    )
    if user_bits == 16:
        scores = scores.astype(np.float16)
    return scores


def nblast_smart(
    query,
    target=None,
    t=90,
    criterion="percentile",
    downsample=10,
    smat=None,
    normalize=True,
    symmetry=None,
    use_alpha=False,
    limit_dist=None,
    n_cores=None,
    precision=32,
    progress=False,
    return_mask=False,
):
    """Smart(er) NBLAST: a fast two-pass approximation of an all-by-all / query-target.

    A cheap "pre-NBLAST" is run on **downsampled** dotprops; for each query the
    best-scoring targets are then kept (per ``criterion``) and the **full-resolution**
    NBLAST is computed only for those query-target pairs. Unselected cells keep their
    coarse pre-pass score. This matches navis' ``nblast_smart`` (its ``scores``
    argument is spelled ``symmetry`` here, as in `nblast`/`nblast_allbyall`).

    A long run can be interrupted with Ctrl-C / the Jupyter interrupt button
    (during either pass); it stops promptly and raises ``KeyboardInterrupt``.

    Parameters
    ----------
    query :      iterable of dotprop-likes
                 Each must expose `points` (N, 3) and unit tangent `vect` (N, 3);
                 also `alpha` (N,) when ``use_alpha`` is set.
    target :     iterable of dotprop-likes | None
                 Targets to compare against. ``None`` runs an all-by-all
                 (``target = query``).
    t :          int | float
                 Threshold for ``criterion``: the percentile (default, ``90`` keeps
                 the top 10% per query), an absolute score, or the number of targets.
    criterion :  'percentile' | 'score' | 'N'
                 How ``t`` selects the candidate targets kept for the full pass.
    downsample : int
                 Pre-pass keeps every ``downsample``-th point (default ``10``).
    smat, normalize, symmetry, use_alpha, limit_dist, n_cores, precision, progress :
                 As in `nblast` / `nblast_allbyall`.
    return_mask : bool
                 If ``True``, also return the boolean mask of cells that were
                 recomputed at full resolution.

    Returns
    -------
    np.ndarray | (np.ndarray, np.ndarray)
                 The (n_query, n_target) score matrix, or ``(scores, mask)`` when
                 ``return_mask`` is set.

    """
    if criterion not in ("percentile", "score", "N"):
        raise ValueError(
            f"Unknown criterion {criterion!r}; expected 'percentile', 'score' or 'N'."
        )

    aba = target is None
    if aba:
        target = query

    q_points, q_vects, q_alphas = _as_clouds(query, want_alpha=use_alpha)
    if aba:
        t_points, t_vects, t_alphas = q_points, q_vects, q_alphas
    else:
        t_points, t_vects, t_alphas = _as_clouds(target, want_alpha=use_alpha)

    # --- pre-pass: coarse NBLAST on downsampled dotprops ---
    dq_p, dq_v, dq_a = _downsample_clouds(q_points, q_vects, q_alphas, downsample)
    q_simp = _clouds_to_dotprops(dq_p, dq_v, dq_a)
    # navis' pre-pass rule: an all-by-all 'mean' selects on the forward score.
    pre_sym = "forward" if (aba and symmetry == "mean") else symmetry
    if aba:
        coarse = nblast_allbyall(
            q_simp, smat=smat, normalize=normalize, symmetry=pre_sym,
            use_alpha=use_alpha, limit_dist=limit_dist, n_cores=n_cores,
            precision=precision, progress=progress,
        )
    else:
        dt_p, dt_v, dt_a = _downsample_clouds(t_points, t_vects, t_alphas, downsample)
        t_simp = _clouds_to_dotprops(dt_p, dt_v, dt_a)
        coarse = nblast(
            q_simp, t_simp, smat=smat, normalize=normalize, symmetry=pre_sym,
            use_alpha=use_alpha, limit_dist=limit_dist, n_cores=n_cores,
            precision=precision, progress=progress,
        )
    coarse = np.asarray(coarse)

    # --- select candidate pairs and recompute them at full resolution ---
    mask = _select_mask(coarse, t, criterion)
    qi, tj = np.where(mask)

    out = coarse.copy()
    if qi.size:
        full = _nblast_pairs(
            q_points, q_vects, q_alphas, t_points, t_vects, t_alphas, qi, tj,
            smat=smat, normalize=normalize, use_alpha=use_alpha,
            limit_dist=limit_dist, n_cores=n_cores, precision=precision,
            progress=progress,
        )
        if symmetry is not None and symmetry != "forward":
            # Reverse direction for the same selected cells (target-as-query).
            rev = _nblast_pairs(
                t_points, t_vects, t_alphas, q_points, q_vects, q_alphas, tj, qi,
                smat=smat, normalize=normalize, use_alpha=use_alpha,
                limit_dist=limit_dist, n_cores=n_cores, precision=precision,
                progress=progress,
            )
            full = _combine(full, rev, symmetry)
        out[qi, tj] = full

    if return_mask:
        return out, mask
    return out


def _as_connectors(neurons, by_type, cn_types):
    """Validate synapse-bearing neurons and extract `(points, types)` per neuron.

    Each item is either an object exposing a ``connectors`` attribute or a raw
    array-like of shape ``(N, 3)`` / ``(N, 4)`` — columns ``[x, y, z, (type)]``.
    The optional 4th column is a numeric connector type. Returns two parallel
    lists: contiguous float64 ``(N, 3)`` coordinates and int64 ``(N,)`` type ids.

    ``by_type`` / ``cn_types`` mirror navis: ``cn_types`` (if given) first keeps
    only connectors whose type is in that set; ``by_type`` then decides whether
    the type column groups the search (only like-typed synapses compare) or is
    collapsed to a single group. Either requires a type column.
    """
    needs_type = by_type or (cn_types is not None)
    keep = None if cn_types is None else np.asarray(list(cn_types))

    out_pts, out_types = [], []
    for n in neurons:
        cn = getattr(n, "connectors", n)
        cn = np.ascontiguousarray(cn, dtype=np.float64)
        if cn.ndim != 2 or cn.shape[1] < 3:
            raise ValueError(
                "each neuron's `connectors` must be a 2-D array with >= 3 columns "
                "[x, y, z, (type)]."
            )
        if needs_type and cn.shape[1] < 4:
            raise ValueError(
                "by_type=True / cn_types require a 4th connector `type` column."
            )

        pts = cn[:, :3]
        if cn.shape[1] >= 4:
            ty = np.rint(cn[:, 3]).astype(np.int64)
        else:
            ty = np.zeros(cn.shape[0], dtype=np.int64)

        if keep is not None:
            sel = np.isin(ty, keep)
            pts, ty = pts[sel], ty[sel]
        if not by_type:
            ty = np.zeros(pts.shape[0], dtype=np.int64)
        if pts.shape[0] == 0:
            raise ValueError(
                "a neuron has no connectors (after cn_types filtering); synblast "
                "requires at least one per neuron."
            )
        out_pts.append(np.ascontiguousarray(pts, dtype=np.float64))
        out_types.append(np.ascontiguousarray(ty, dtype=np.int64))
    return out_pts, out_types


def synblast(
    query,
    target=None,
    by_type=False,
    cn_types=None,
    smat=None,
    normalize=True,
    symmetry=None,
    n_cores=None,
    precision=32,
    progress=False,
):
    """Synapse-based NBLAST (syNBLAST).

    Compares neurons by their **synapses** (connectors) rather than their skeleton
    points: for every query connector the nearest target connector *of the same
    type* is found, and the euclidean distance is scored through the NBLAST lookup
    matrix with the dot product fixed at 1 (synapses have no tangent vector). This
    matches navis' `synblast` (navis' ``scores`` argument is spelled ``symmetry``
    here, as in `nblast` / `nblast_allbyall`).

    A long run can be interrupted with Ctrl-C / the Jupyter interrupt button;
    it stops promptly and raises ``KeyboardInterrupt``.

    Parameters
    ----------
    query :      iterable of synapse-bearing neurons
                 Each item exposes ``connectors`` (or is itself an array) of shape
                 ``(N, 3)`` or ``(N, 4)`` — ``[x, y, z, (type)]``. The 4th column is
                 a numeric connector type, required when ``by_type`` / ``cn_types``
                 are set. See `Synapses`.
    target :     iterable of synapse-bearing neurons | None
                 Targets to compare against. ``None`` runs an all-by-all
                 (``target = query``).
    by_type :    bool
                 If ``True``, only connectors of the same type compare against each
                 other (navis' ``by_type``); requires a type column. Default
                 ``False`` treats all connectors as one group.
    cn_types :   iterable | None
                 If given, keep only connectors whose type is in this set before
                 scoring (navis' ``cn_types``); requires a type column.
    smat :       None | navis Lookup2d | (values, dist_edges, dot_edges)
                 Scoring matrix. ``None`` uses the embedded FCWB matrix (navis'
                 ``smat="auto"``); only its last (aligned) dot-product column is
                 used.
    normalize :  bool
                 Divide each score by the query's self-hit (self-match == 1.0).
    symmetry :   None | 'forward' | 'mean' | 'min' | 'max'
                 ``None`` / ``'forward'`` returns the raw forward matrix; the others
                 combine it with the reverse (target-vs-query) syNBLAST.
    n_cores :    int | None
                 Cap the number of worker threads. ``None`` uses all cores.
    precision :  16 | 32 | 64
                 Dtype of the returned matrix. The scoring math is always float64.
    progress :   bool
                 Show progress bars over index building and scoring.

    Returns
    -------
    np.ndarray
                 (n_query, n_target) score matrix; row = query, column = target.

    """
    aba = target is None
    if aba:
        target = query

    q_pts, q_types = _as_connectors(query, by_type, cn_types)
    if aba:
        t_pts, t_types = q_pts, q_types
    else:
        t_pts, t_types = _as_connectors(target, by_type, cn_types)

    sv, de, ve = _smat_args(smat)
    user_bits, rust_bits = _resolve_precision(precision)

    if aba:
        M = np.asarray(
            _fastcore.synblast_allbyall(
                q_pts, q_types, smat_values=sv, dist_edges=de, dot_edges=ve,
                normalize=normalize, n_cores=n_cores, precision=rust_bits,
                progress=progress,
            )
        )
    else:
        M = np.asarray(
            _fastcore.synblast(
                q_pts, q_types, t_pts, t_types, smat_values=sv, dist_edges=de,
                dot_edges=ve, normalize=normalize, n_cores=n_cores,
                precision=rust_bits, progress=progress,
            )
        )

    if symmetry is not None and symmetry != "forward":
        if aba:
            M = _symmetrize(M, symmetry, n_cores)
        else:
            # Reverse syNBLAST (target-as-query); its transpose aligns with M.
            R = np.asarray(
                _fastcore.synblast(
                    t_pts, t_types, q_pts, q_types, smat_values=sv, dist_edges=de,
                    dot_edges=ve, normalize=normalize, n_cores=n_cores,
                    precision=rust_bits, progress=progress,
                )
            )
            M = _combine(M, R.T, symmetry)
    if user_bits == 16:
        M = M.astype(np.float16)
    return M


def _make_dotprop(points, k=5):
    """Create a `Dotprop` (points + tangent vectors + alpha) from a point cloud.

    Convenience helper only; not on the NBLAST scoring path. Requires scipy.
    """
    vect, alpha = _tangents_and_alpha(points, k)
    return Dotprop(np.asarray(points, dtype=np.float64), vect, alpha)


def _calculate_tangent_vectors(points, k):
    """Tangent vectors as the first principal axis of each point's k-neighborhood.

    Requires scipy (imported lazily).
    """
    return _tangents_and_alpha(points, k)[0]


def _tangents_and_alpha(points, k):
    """Per-point tangent vector and alpha from the local k-neighborhood SVD.

    `alpha = (s0 - s1) / (s0 + s1 + s2)` over the neighborhood scatter matrix's
    singular values, matching navis' dotprops. Requires scipy (lazy import).
    """
    from scipy.spatial import cKDTree as KDTree

    points = np.asarray(points, dtype=np.float64)
    _, ix = KDTree(points).query(points, k=k)

    # (N, k, 3) neighborhoods, centered, then SVD of the local inertia tensor.
    pt = points[ix]
    centers = np.mean(pt, axis=1)
    cpt = pt - centers.reshape((pt.shape[0], 1, 3))
    inertia = cpt.transpose((0, 2, 1)) @ cpt
    _, s, vh = np.linalg.svd(inertia)
    vect = vh[:, 0, :]
    alpha = (s[:, 0] - s[:, 1]) / s.sum(axis=1)

    return vect, alpha
