"""Applying CMTK transformations to points."""

import os

import numpy as np

from . import _fastcore
from ._points import _prep_fallback, _prep_invert, _prep_points

__all__ = ["CmtkRegistration", "load_cmtk_registration"]


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

    Direction is **not** fixed here. It is chosen per call - see the ``invert`` argument on
    :meth:`xform`, and :meth:`xform_inv` - so one instance serves every direction and the
    files are parsed only once.

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

    def __init__(self, path):
        if isinstance(path, (str, os.PathLike)):
            paths = [os.fspath(path)]
        else:
            paths = [os.fspath(p) for p in path]
        if not paths:
            raise ValueError("`path` must name at least one registration")

        self._paths = paths
        self._reg = _fastcore.CmtkRegistration(paths)

    def xform(
        self,
        points,
        transform="warp",
        allow_extrapolation=False,
        fallback_to_affine=False,
        invert=False,
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
        fallback_to_affine :    bool | "chain" | "hop"
                                Replace failed rows with the affine result rather than
                                ``NaN``. Only reachable when ``allow_extrapolation=False``,
                                since extrapolation otherwise never fails.

                                ``True`` (or ``"chain"``) re-runs the **whole chain**
                                affine-only, starting again from the *original* point. This is
                                what ``nat``/``navis`` do - they hand the failed rows back to
                                ``streamxform --affine-only`` over the same registration list -
                                so it is the default whenever a fallback is asked for.

                                ``"hop"`` instead swaps the affine in for **only the hop that
                                failed**, keeping the warps of the hops that succeeded.
                                Arguably the better answer, since discarding a perfectly good
                                hop-1 warp because hop 2 ran out of domain is crude - but it is
                                a silent departure from every other CMTK-based tool, so you
                                have to ask for it. On a single registration the two are
                                identical; on a chain they differ by a median of ~6 world
                                units.

                                Works in both directions: a hop travelled backwards falls back
                                to the *inverse* affine, so the rescued point still lands in
                                the space you asked for.
        invert :                bool | list of bool
                                Traverse a registration backwards. A single bool applies to
                                every hop; pass a list to set them per registration. This is
                                what you need when routing through a bridging graph, where an
                                edge may be walked in either direction - and a chain may need
                                some hops forwards and others backwards.

                                Note this is **not** the same as :meth:`xform_inv`, which
                                inverts the whole composition (reversing the order *and*
                                flipping every hop). For a single registration the two agree;
                                for a chain they do not, and only ``invert`` can express a
                                mixed-direction traversal.
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
            _prep_fallback(fallback_to_affine),
            _prep_invert(invert, len(self._paths)),
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
        fallback_to_affine=False,
        invert=False,
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
        fallback_to_affine : bool | "chain" | "hop"
                            Replace rows the solver could not land with the *inverse* affine
                            result rather than ``NaN`` - the mirror of the same argument on
                            :meth:`xform`, with the same ``"chain"`` (default) and ``"hop"``
                            semantics.
        invert :            bool | list of bool
                            The same per-hop flags as on :meth:`xform`, composed with this
                            whole-chain inversion: hop ``i`` runs *forwards* here exactly when
                            ``invert[i]`` is set.
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
            _prep_fallback(fallback_to_affine),
            _prep_invert(invert, len(self._paths)),
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
        return (CmtkRegistration, (self._paths,))

    def __repr__(self):
        return self._reg.__repr__()


def load_cmtk_registration(path):
    """Load one or more CMTK registrations.

    A convenience wrapper around :class:`~navis_fastcore.CmtkRegistration`.

    Parameters
    ----------
    path :      str | pathlib.Path | list thereof
                Path to a CMTK ``*.list`` directory, or to a ``registration`` file itself
                (plain or gzipped). A list builds a chain, applied in order.

    Returns
    -------
    CmtkRegistration

    Examples
    --------
    >>> import navis_fastcore as fastcore
    >>> reg = fastcore.load_cmtk_registration("JFRC2_FCWB.list")   # doctest: +SKIP

    """
    return CmtkRegistration(path)
