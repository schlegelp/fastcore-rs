"""Applying CMTK transformations to points."""

import os

import numpy as np

from . import _fastcore

__all__ = ["CmtkRegistration", "load_cmtk_registration"]


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


class CmtkRegistration:
    """A CMTK registration - or a chain of them - ready to apply to points.

    A registration is parsed once, on construction, and can then be applied any number of
    times. It consists of a 12-DOF affine and, usually, a cubic B-spline warp on a
    control-point lattice.

    Unlike `nat`/`navis`, this does not shell out to CMTK's ``streamxform`` binary: CMTK
    does not need to be installed. Results match ``streamxform`` to ~1e-7, including its
    convention of failing (here: returning ``NaN``) on points whose inverse does not
    converge.

    Parameters
    ----------
    path :      str | pathlib.Path | list thereof
                Path to a CMTK ``*.list`` registration directory, or to a ``registration``
                file itself (plain or gzipped). Pass a list to build a chain, applied in
                order: ``points -> path[0] -> path[1] -> ... -> output``.
    invert :    bool | list of bool
                Traverse a registration backwards. A single bool applies to every entry;
                pass a list to set them per registration. Useful when routing through a
                bridging graph, where an edge may be traversed in either direction.

    Attributes
    ----------
    affine :    (4, 4) np.ndarray
                The affine matrix of the first registration in the chain.
    dims :      (k, 3) np.ndarray
                Control-point lattice dimensions of each spline warp in the chain.
    spacing :   (k, 3) np.ndarray
                Control-point spacing of each spline warp in the chain.
    version :   list of str
                CMTK TypedStream version of each registration.

    Examples
    --------
    >>> import navis_fastcore as fastcore
    >>> import numpy as np
    >>> reg = fastcore.CmtkRegistration("JFRC2_FCWB.list")     # doctest: +SKIP
    >>> pts = np.array([[50.0, 50.0, 50.0], [100.0, 100.0, 20.0]])
    >>> xf = reg.xform(pts)                                    # doctest: +SKIP
    >>> back = reg.xform_inv(xf)                               # doctest: +SKIP
    >>> np.allclose(back, pts, atol=1e-4)                      # doctest: +SKIP
    True

    """

    def __init__(self, path, invert=False):
        if isinstance(path, (str, os.PathLike)):
            paths = [os.fspath(path)]
        else:
            paths = [os.fspath(p) for p in path]
        if not paths:
            raise ValueError("`path` must name at least one registration")

        if isinstance(invert, bool):
            inv = [invert] * len(paths)
        else:
            inv = [bool(i) for i in invert]
            if len(inv) != len(paths):
                raise ValueError(
                    f"`invert` must have one flag per registration: expected "
                    f"{len(paths)}, got {len(inv)}"
                )

        self._paths = paths
        self._invert = inv
        self._reg = _fastcore.CmtkRegistration(paths, inv)

    def xform(
        self,
        points,
        transform="warp",
        allow_extrapolation=False,
        fallback_to_affine=False,
        n_cores=None,
        progress=False,
    ):
        """Transform points forward through the registration.

        Parameters
        ----------
        points :                (N, 3) array-like
                                Coordinates to transform. A single ``(3,)`` point is
                                accepted and returns a ``(3,)`` point.
        transform :             "warp" | "affine"
                                ``"warp"`` (default) applies the full transformation.
                                ``"affine"`` applies only the affine component. A
                                registration with no spline warp uses its affine either way.
        allow_extrapolation :   bool
                                Evaluate points outside the registration's domain box by
                                clamping to the outermost control points, instead of failing
                                them.

                                Defaults to ``False``, **which is what CMTK does**:
                                ``streamxform`` reports a point outside the domain as
                                ``FAILED``, and we return ``NaN``. Setting this to ``True``
                                gives every point *an* answer, but that answer extrapolates a
                                warp that was never fitted there, and it will silently
                                disagree with every other CMTK-based tool.
        fallback_to_affine :    bool
                                Replace failed rows with the affine result rather than
                                ``NaN``. Only reachable when ``allow_extrapolation=False``,
                                since extrapolation otherwise never fails.
        n_cores :               int, optional
                                Number of threads. ``None`` (default) uses all cores.
        progress :              bool
                                Show a progress bar.

        Returns
        -------
        (N, 3) np.ndarray
                                Transformed coordinates. Rows that could not be transformed
                                are ``NaN``.

        """
        pts, was_1d = _prep_points(points)
        out = self._reg.xform(
            pts,
            str(transform),
            bool(allow_extrapolation),
            bool(fallback_to_affine),
            None if n_cores is None else int(n_cores),
            bool(progress),
        )
        return out[0] if was_1d else out

    def xform_inv(
        self,
        points,
        transform="warp",
        initial_guess=None,
        max_iter=50,
        tolerance=1e-9,
        accuracy=1e-3,
        clamp_to_domain=True,
        n_cores=None,
        progress=False,
    ):
        """Transform points backwards through the registration.

        The affine part is inverted exactly. The spline warp has no closed-form inverse and
        is solved per point by damped Gauss-Newton against the analytic Jacobian. Points
        whose residual does not converge come back as ``NaN`` - this is deliberate, and
        matches CMTK's ``streamxform``, which reports such points as ``FAILED``.

        Parameters
        ----------
        points :            (N, 3) array-like
                            Coordinates to invert.
        transform :         "warp" | "affine"
                            Which forward transform to invert.
        initial_guess :     (N, 3) array-like, optional
                            Starting points for the solver. Defaults to ``points`` itself,
                            which is a good guess for any well-behaved registration. In a
                            chain, this seeds only the first solve.
        max_iter :          int
                            Maximum Gauss-Newton iterations per point.
        tolerance :         float
                            Stop once the step falls below this.
        accuracy :          float
                            Accept a solution only if its residual is within this of the
                            target; otherwise the row is ``NaN``.
        clamp_to_domain :   bool
                            Confine the iterate to the spline's domain box. **This is what
                            makes the result agree with CMTK.** Turning it off finds
                            preimages that lie outside the image domain, where ``streamxform``
                            reports failure - so you will get finite numbers where CMTK gives
                            you none.
        n_cores :           int, optional
                            Number of threads. ``None`` (default) uses all cores.
        progress :          bool
                            Show a progress bar.

        Returns
        -------
        (N, 3) np.ndarray
                            Inverse-transformed coordinates. Rows that did not converge are
                            ``NaN``.

        """
        pts, was_1d = _prep_points(points)

        guess = None
        if initial_guess is not None:
            guess, _ = _prep_points(initial_guess, name="initial_guess")
            if guess.shape != pts.shape:
                raise ValueError(
                    "`initial_guess` must have one point per input point: expected "
                    f"{pts.shape}, got {guess.shape}"
                )

        out = self._reg.xform_inv(
            pts,
            str(transform),
            guess,
            int(max_iter),
            float(tolerance),
            float(accuracy),
            bool(clamp_to_domain),
            None if n_cores is None else int(n_cores),
            bool(progress),
        )
        return out[0] if was_1d else out

    @property
    def affine(self):
        """The (4, 4) affine matrix of the first registration in the chain."""
        return self._reg.affine

    @property
    def dims(self):
        """Control-point lattice dimensions of each spline warp, as a (k, 3) array."""
        return self._reg.dims

    @property
    def spacing(self):
        """Control-point spacing of each spline warp, as a (k, 3) array."""
        return self._reg.spacing

    @property
    def version(self):
        """CMTK TypedStream version of each registration in the chain."""
        return self._reg.version

    @property
    def has_spline(self):
        """Whether each registration carries a spline warp (vs. affine only)."""
        return self._reg.has_spline

    @property
    def paths(self):
        """The paths this registration was loaded from."""
        return list(self._paths)

    def __len__(self):
        return self._reg.n_registrations

    def __reduce__(self):
        # Re-load from disk in the child rather than pickling ~420 KB of coefficients
        # through every multiprocessing/joblib fan-out.
        return (CmtkRegistration, (self._paths, self._invert))

    def __repr__(self):
        return self._reg.__repr__()


def load_cmtk_registration(path, invert=False):
    """Load one or more CMTK registrations.

    A convenience wrapper around :class:`~navis_fastcore.CmtkRegistration`.

    Parameters
    ----------
    path :      str | pathlib.Path | list thereof
                Path to a CMTK ``*.list`` directory, or to a ``registration`` file itself
                (plain or gzipped). A list builds a chain, applied in order.
    invert :    bool | list of bool
                Traverse a registration backwards.

    Returns
    -------
    CmtkRegistration

    Examples
    --------
    >>> import navis_fastcore as fastcore
    >>> reg = fastcore.load_cmtk_registration("JFRC2_FCWB.list")   # doctest: +SKIP

    """
    return CmtkRegistration(path, invert=invert)
