import numpy as np

from collections import namedtuple

from . import _fastcore

__all__ = ["nblast_allbyall"]

Dotprop = namedtuple("Dotprop", ["points", "vect"])


def nblast_allbyall(x, backend="bosque"):
    """All-by-all NBLAST.

    Parameters
    ----------
    x :     list-like with dotprop-likes
            Must have attributes `points` and `vect` that are numpy arrays.

    """
    # TODO:
    # - add support for NBLAST parameters (query, target, scores, precision, etc.)
    # - add support for progress bar
    if not hasattr(x, "__iter__"):
        raise TypeError("x must be iterable")

    for n in x:
        if not hasattr(n, "points") or not hasattr(n, "vect"):
            raise TypeError("x must be iterable of dotprop-likes")

    # Collect points and vectors
    points = [n.points.astype(np.float64, copy=False) for n in x]
    vects = [n.vect.astype(np.float64, copy=False) for n in x]

    # Calculate all-by-all NBLAST
    scores = _fastcore.nblast_allbyall(points, vects, backend=backend)

    return scores


def _make_dotprop(points, k=5):
    """Create a Dotprop object.

    Parameters
    ----------
    points :    np.ndarray
                Array of points.
    k :         int
                Number of nearest neighbours to use for tangent vector
                calculation.

    Returns
    -------
    Dotprop
                Namedtuple with attributes `points` and `vect`.

    """
    vect = _calculate_tangent_vectors(points, k)
    return Dotprop(points, vect)


def _calculate_tangent_vectors(points, k):
    """Calculate tangent vectors.

    Parameters
    ----------
    k :         int
                Number of nearest neighbours to use for tangent vector
                calculation.

    Returns
    -------
    Dotprops
                Only if ``inplace=False``.

    """
    # Create the KDTree and get the k-nearest neighbors for each point
    from scipy.spatial import cKDTree as KDTree

    dist, ix = KDTree(points).query(points, k=k)

    # Get points: array of (N, k, 3)
    pt = points[ix]

    # Generate centers for each cloud of k nearest neighbors
    centers = np.mean(pt, axis=1)

    # Generate vector from center
    cpt = pt - centers.reshape((pt.shape[0], 1, 3))

    # Get inertia (N, 3, 3)
    inertia = cpt.transpose((0, 2, 1)) @ cpt

    # Extract vector and alpha
    u, s, vh = np.linalg.svd(inertia)
    vect = vh[:, 0, :]
    # alpha = (s[:, 0] - s[:, 1]) / np.sum(s, axis=1)

    return vect
