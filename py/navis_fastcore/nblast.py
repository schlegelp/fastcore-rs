import numpy as np

from collections import namedtuple

from . import _fastcore

__all__ = ["nblast_allbyall", "nblast"]

#: Minimal dotprop container: `points` (N, 3), unit tangent `vect` (N, 3) and an
#: optional per-point `alpha` (N,) used only when ``use_alpha=True``.
Dotprop = namedtuple("Dotprop", ["points", "vect", "alpha"], defaults=[None])

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


def _combine(a, b, symmetry):
    """Combine two equally-shaped score matrices element-wise."""
    if symmetry in ("mean", True):
        return (a + b) / 2
    if symmetry == "min":
        return np.minimum(a, b)
    if symmetry == "max":
        return np.maximum(a, b)
    raise ValueError(
        f"Unknown symmetry {symmetry!r}; expected 'forward', 'mean', 'min', "
        "'max' or None."
    )


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
                 Show a progress bar over the scoring pairs (drawn from Rust to
                 stderr).

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
        M = _combine(M, M.T, symmetry)
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
                    Show a progress bar over the scoring pairs.

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
