import numpy as np

from . import _fastcore

__all__ = ["nblast_allbyall"]


def nblast_allbyall(x):
    """All-by-all NBLAST.

    Parameters
    ----------
    x :     list-like with dotprop-likes

    """
    if not hasattr(x, "__iter__"):
        raise TypeError("x must be iterable")

    for n in x:
        if not hasattr(n, "points") or not hasattr(n, "vect"):
            raise TypeError("x must be iterable of dotprop-likes")

    # Collect points and vectors
    points = [n.points.astype(np.float64) for n in x]
    vects = [n.vect.astype(np.float64) for n in x]

    # Calculate all-by-all NBLAST
    scores = _fastcore.nblast_allbyall(points, vects)

    return scores