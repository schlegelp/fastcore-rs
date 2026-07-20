"""Landmark-based spatial transforms: thin-plate spline and moving least squares.

These are the fallback when no image registration exists for a pair of template spaces -
you have matched landmarks and want a warp that interpolates between them.

Both are drop-in replacements for their `navis` counterparts
(``navis.transforms.TPStransform`` and ``navis.transforms.MovingLeastSquaresTransform``)
and agree with them to ~1e-14 relative. The difference is that the ``(n_points,
n_landmarks)`` matrix the reference implementations build is never materialised here, so
there is no ``batch_size`` to tune and peak memory does not depend on the landmark count.
"""

import numpy as np

from . import _fastcore
from ._points import _prep_points

__all__ = ["TpsTransform", "MlsTransform"]


def _prep_xyz(points, name="points"):
    """Coerce to a C-contiguous (N, 3) float64 array, accepting a DataFrame.

    `navis`' landmark transforms take x/y/z-columned DataFrames, and landmarks are
    routinely read straight out of a CSV, so accepting one here is what makes these classes
    drop-in. Duck-typed on `.columns` rather than importing pandas, which is not a
    dependency of this package.
    """
    cols = getattr(points, "columns", None)
    if cols is not None:
        missing = [c for c in ("x", "y", "z") if c not in cols]
        if missing:
            raise ValueError(
                f"`{name}` DataFrame must have x/y/z columns, missing: "
                f"{', '.join(missing)}"
            )
        points = points[["x", "y", "z"]].values
    return _prep_points(points, name=name)


def _prep_landmarks(source, target):
    """Coerce a landmark pair to matching C-contiguous (M, 3) float64 arrays."""
    src, _ = _prep_xyz(source, name="landmarks_source")
    trg, _ = _prep_xyz(target, name="landmarks_target")
    if src.shape[0] != trg.shape[0]:
        raise ValueError(
            "number of source landmarks must match number of target landmarks: got "
            f"{src.shape[0]} and {trg.shape[0]}"
        )
    return src, trg


class TpsTransform:
    """A thin-plate spline transform, fitted to landmark pairs.

    The spline interpolates the source landmarks onto the target landmarks exactly, and
    between them it minimises the integral bending norm - the smoothest warp consistent
    with the landmarks.

    The fit happens once, here, and the transform can then be applied any number of times.

    Parameters
    ----------
    landmarks_source :  (M, 3) array-like
                        Source landmarks as x/y/z coordinates. A pandas ``DataFrame`` with
                        x/y/z columns is also accepted.
    landmarks_target :  (M, 3) array-like
                        Target landmarks, one per source landmark.

    Attributes
    ----------
    source :            (M, 3) np.ndarray
                        The source landmarks.
    W :                 (M, 3) np.ndarray
                        Weights of the non-affine part of the spline.
    A :                 (4, 3) np.ndarray
                        Coefficients of the affine part; row 0 is the translation.
    matrix_affine :     (4, 4) np.ndarray
                        The affine part as a homogeneous matrix - what the spline
                        converges to far from the landmarks.

    Notes
    -----
    Unlike ``navis.transforms.TPStransform`` there is no ``batch_size``: the distance
    matrix is fused into the accumulation rather than built, so peak memory is the output
    array regardless of how many points or landmarks are involved.

    The fit is cubic in the landmark count and runs through a blocked LU. At a few thousand
    landmarks it is somewhat slower than ``numpy.linalg.solve`` (which reaches hardware
    LAPACK); :meth:`from_coefs` exists if you would rather fit with numpy and only use this
    class to apply the result.

    Examples
    --------
    >>> import navis_fastcore as fastcore
    >>> import numpy as np
    >>> src = np.array([[0, 0, 0], [10, 10, 10], [100, 100, 100], [80, 10, 30]])
    >>> trg = np.array([[1, 15, 5], [9, 18, 21], [80, 99, 120], [5, 10, 80]])
    >>> tr = fastcore.TpsTransform(src, trg)
    >>> tr.xform(np.array([[0, 0, 0], [50, 50, 50]]))
    array([[ 1.        , 15.        ,  5.        ],
           [40.55555556, 54.        , 65.        ]])
    >>> # Landmarks are reproduced exactly
    >>> np.allclose(tr.xform(src), trg)
    True
    >>> # Negation refits in the opposite direction
    >>> np.allclose((-tr).xform(trg), src)
    True

    """

    def __init__(self, landmarks_source, landmarks_target):
        src, trg = _prep_landmarks(landmarks_source, landmarks_target)
        if src.shape[0] < 4:
            raise ValueError(
                "need at least 4 landmarks to fit a 3D thin-plate spline, got "
                f"{src.shape[0]}"
            )
        self._target = trg
        self._tr = _fastcore.TpsTransform(src, trg)

    @classmethod
    def from_coefs(cls, landmarks_source, W, A, landmarks_target=None):
        """Build from coefficients fitted elsewhere, skipping the fit.

        Parameters
        ----------
        landmarks_source :  (M, 3) array-like
                            The source landmarks the coefficients belong to.
        W :                 (M, 3) array-like
                            Weights of the non-affine part.
        A :                 (4, 3) array-like
                            Coefficients of the affine part.
        landmarks_target :  (M, 3) array-like, optional
                            The target landmarks. Not needed to transform points, but
                            without them :meth:`__neg__` cannot refit the inverse.

        Returns
        -------
        TpsTransform

        Examples
        --------
        >>> import navis_fastcore as fastcore
        >>> import numpy as np
        >>> src = np.array([[0, 0, 0], [10, 10, 10], [100, 100, 100], [80, 10, 30]])
        >>> trg = np.array([[1, 15, 5], [9, 18, 21], [80, 99, 120], [5, 10, 80]])
        >>> tr = fastcore.TpsTransform(src, trg)
        >>> same = fastcore.TpsTransform.from_coefs(tr.source, tr.W, tr.A)
        >>> np.allclose(same.xform(src), trg)
        True

        """
        obj = cls.__new__(cls)
        src, _ = _prep_xyz(landmarks_source, name="landmarks_source")
        w = np.ascontiguousarray(np.asarray(W, dtype=np.float64))
        a = np.ascontiguousarray(np.asarray(A, dtype=np.float64))
        if w.shape != src.shape:
            raise ValueError(f"`W` must be (M, 3) matching the landmarks, got {w.shape}")
        if a.shape != (4, 3):
            raise ValueError(f"`A` must be (4, 3), got {a.shape}")
        obj._tr = _fastcore.TpsTransform.from_coefs(src, w, a)
        obj._target = (
            None
            if landmarks_target is None
            else _prep_xyz(landmarks_target, name="landmarks_target")[0]
        )
        return obj

    def xform(self, points, n_cores=None):
        """Transform points.

        Parameters
        ----------
        points :    (N, 3) array-like
                    Coordinates to transform. A single ``(3,)`` point is accepted and
                    returns a ``(3,)`` point; a ``DataFrame`` with x/y/z columns also works.
        n_cores :   int, optional
                    Number of threads. ``None`` (default) uses all cores.

        Returns
        -------
        (N, 3) np.ndarray
                    Transformed coordinates.

        """
        pts, was_1d = _prep_xyz(points)
        out = self._tr.xform(pts, None if n_cores is None else int(n_cores))
        return out[0] if was_1d else out

    @property
    def source(self):
        """The source landmarks, as an (M, 3) array."""
        return self._tr.source

    @property
    def target(self):
        """The target landmarks, or ``None`` if built from coefficients without them."""
        return None if self._target is None else self._target.copy()

    @property
    def W(self):
        """Weights of the non-affine part of the spline, as an (M, 3) array."""
        return self._tr.weights

    @property
    def A(self):
        """Coefficients of the affine part, as a (4, 3) array."""
        return self._tr.affine_coefs

    @property
    def matrix_affine(self):
        """The affine part as a (4, 4) homogeneous matrix."""
        return self._tr.matrix_affine

    def copy(self):
        """Return a copy. Shares the fit rather than repeating it."""
        obj = self.__class__.__new__(self.__class__)
        obj._tr = self._tr
        obj._target = None if self._target is None else self._target.copy()
        return obj

    def __neg__(self):
        """Fit the spline in the opposite direction.

        This is a fresh fit of target onto source, not an inversion of this one - a thin
        plate spline has no closed-form inverse. Requires the target landmarks.
        """
        if self._target is None:
            raise ValueError(
                "cannot invert a TpsTransform built via `from_coefs` without "
                "`landmarks_target`"
            )
        return self.__class__(self._target, self.source)

    def __eq__(self, other):
        if not isinstance(other, TpsTransform):
            return NotImplemented
        if not np.array_equal(self.source, other.source):
            return False
        if self._target is None or other._target is None:
            # No targets to compare - fall back to the coefficients, which determine the
            # transform just as completely.
            return np.array_equal(self.W, other.W) and np.array_equal(self.A, other.A)
        return np.array_equal(self._target, other._target)

    def __len__(self):
        return self._tr.n_landmarks

    def __reduce__(self):
        # Ship the coefficients, not the landmarks alone: refitting in every
        # multiprocessing worker would cost more than the transforms they were forked to
        # perform.
        return (
            _tps_from_state,
            (self.source, self.W, self.A, self._target),
        )

    def __repr__(self):
        return f"<TpsTransform(landmarks={len(self)})>"


def _tps_from_state(source, W, A, target):
    """Unpickle helper - module-level so it is importable by name."""
    return TpsTransform.from_coefs(source, W, A, landmarks_target=target)


class MlsTransform:
    """A moving least squares transform, defined by landmark pairs.

    The affine flavour of the algorithm published in
    [Schaefer et al. (2006)](https://dl.acm.org/doi/pdf/10.1145/1179352.1141920). Unlike a
    thin-plate spline there is no fit: every point gets its *own* affine, solved on the fly
    from all landmarks weighted by inverse squared distance. That makes construction free
    and :meth:`xform` the entire cost.

    Parameters
    ----------
    landmarks_source :  (M, 3) array-like
                        Source landmarks as x/y/z coordinates. A pandas ``DataFrame`` with
                        x/y/z columns is also accepted.
    landmarks_target :  (M, 3) array-like
                        Target landmarks, one per source landmark.
    direction :         "forward" | "inverse"
                        ``"inverse"`` treats the target as the source and vice versa.
                        Note this fits the warp in the opposite direction; it is not an
                        exact inverse, which moving least squares does not have.

    Notes
    -----
    ``navis.transforms.MovingLeastSquaresTransform`` (via `molesq`) builds
    ``(3, M, N)``-shaped intermediates, which is why it takes a ``batch_size`` - and why in
    practice it cannot run at the landmark counts real registrations use: 3400 landmarks at
    the default batch size needs ~23 GB. Here everything but the result is a reduction over
    landmarks, so peak memory is the output array and the landmark count is unbounded.

    Examples
    --------
    >>> import navis_fastcore as fastcore
    >>> import numpy as np
    >>> src = np.array([[0, 0, 0], [10, 10, 10], [100, 100, 100], [80, 10, 30]])
    >>> trg = np.array([[1, 15, 5], [9, 18, 21], [80, 99, 120], [5, 10, 80]])
    >>> tr = fastcore.MlsTransform(src, trg)
    >>> # A point sitting on a landmark maps to its partner
    >>> np.allclose(tr.xform(src), trg)
    True
    >>> # Negation swaps the direction
    >>> np.allclose((-tr).xform(trg), src)
    True

    """

    def __init__(self, landmarks_source, landmarks_target, direction="forward"):
        if direction not in ("forward", "inverse"):
            raise ValueError(
                f'`direction` must be "forward" or "inverse", got {direction!r}'
            )
        src, trg = _prep_landmarks(landmarks_source, landmarks_target)
        self._reverse = direction == "inverse"
        self._tr = _fastcore.MlsTransform(src, trg)

    def xform(self, points, n_cores=None, reverse=None):
        """Transform points.

        Parameters
        ----------
        points :    (N, 3) array-like
                    Coordinates to transform. A single ``(3,)`` point is accepted and
                    returns a ``(3,)`` point; a ``DataFrame`` with x/y/z columns also works.
        n_cores :   int, optional
                    Number of threads. ``None`` (default) uses all cores.
        reverse :   bool, optional
                    Override this transform's ``direction`` for this call only:
                    ``False`` maps source -> target, ``True`` target -> source.
                    ``None`` (default) uses ``direction``. The underlying fit is
                    direction-agnostic, so this saves rebuilding the object just
                    to run it the other way.

        Returns
        -------
        (N, 3) np.ndarray
                    Transformed coordinates.

        """
        pts, was_1d = _prep_xyz(points)
        rev = self._reverse if reverse is None else bool(reverse)
        out = self._tr.xform(pts, rev, None if n_cores is None else int(n_cores))
        return out[0] if was_1d else out

    @property
    def source(self):
        """The landmarks this transform maps *from*, honouring ``direction``."""
        return self._tr.target if self._reverse else self._tr.source

    @property
    def target(self):
        """The landmarks this transform maps *to*, honouring ``direction``."""
        return self._tr.source if self._reverse else self._tr.target

    @property
    def direction(self):
        """``"forward"`` or ``"inverse"``."""
        return "inverse" if self._reverse else "forward"

    @property
    def matrix_affine(self):
        """The *global* affine as a (4, 4) homogeneous matrix.

        Moving least squares is *locally* weighted - every point effectively gets its own
        affine - so there is no single matrix describing it. This is the least-squares fit
        of source onto target landmarks, which is what the warp converges to far from them.
        """
        return self._tr.matrix_affine(self._reverse)

    def copy(self):
        """Return a copy."""
        obj = self.__class__.__new__(self.__class__)
        obj._tr = self._tr
        obj._reverse = self._reverse
        return obj

    def __neg__(self):
        """Flip the direction."""
        obj = self.copy()
        obj._reverse = not self._reverse
        return obj

    def __eq__(self, other):
        if not isinstance(other, MlsTransform):
            return NotImplemented
        return (
            np.array_equal(self.source, other.source)
            and np.array_equal(self.target, other.target)
        )

    def __len__(self):
        return self._tr.n_landmarks

    def __reduce__(self):
        return (
            _mls_from_state,
            (self._tr.source, self._tr.target, self.direction),
        )

    def __repr__(self):
        return f"<MlsTransform(landmarks={len(self)}, direction='{self.direction}')>"


def _mls_from_state(source, target, direction):
    """Unpickle helper - module-level so it is importable by name."""
    return MlsTransform(source, target, direction=direction)
