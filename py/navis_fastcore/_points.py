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
