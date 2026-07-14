"""Point coercion shared by the spatial-transform modules."""

import numpy as np


def _prep_points(points, name="points"):
    """Coerce to a C-contiguous (N, 3) float64 array.

    A bare `(3,)` point is promoted to `(1, 3)`; the caller un-promotes the result so a
    single point in gives a single point out.
    """
    pts = np.asarray(points, dtype=np.float64)
    was_1d = pts.ndim == 1
    if was_1d:
        pts = pts[None, :]
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(
            f"`{name}` must be an (N, 3) array of 3D coordinates, got shape "
            f"{np.shape(points)}"
        )
    return np.ascontiguousarray(pts), was_1d


def _prep_fallback(fallback):
    """Coerce `fallback_to_affine` to "none" | "chain" | "hop".

    `True` means "chain" - the nat/navis semantics - so that the plain boolean spelling stays
    the faithful one and the departure has to be named.
    """
    if fallback is False or fallback is None:
        return "none"
    if fallback is True:
        return "chain"
    if fallback in ("none", "chain", "hop"):
        return fallback
    raise ValueError(
        "`fallback_to_affine` must be False, True, 'chain' or 'hop', got "
        f"{fallback!r}"
    )


def _prep_invert(invert, n, what="registration"):
    """Coerce `invert` to a list of `n` bools, or None for the all-forward default.

    `None` rather than `[False] * n` so the core can skip the per-hop machinery entirely on
    the common path.
    """
    if invert is None or invert is False:
        return None
    if isinstance(invert, bool):
        return [invert] * n
    flags = [bool(i) for i in invert]
    if len(flags) != n:
        raise ValueError(
            f"`invert` must have one flag per {what}: expected {n}, got {len(flags)}"
        )
    return flags if any(flags) else None
