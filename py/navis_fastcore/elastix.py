"""Applying Elastix transformations to points."""

import os

import numpy as np

from . import _fastcore
from ._points import _prep_points

__all__ = ["ElastixTransform", "load_elastix_transform"]


class ElastixTransform:
    """An Elastix transform - or a chain of them - ready to apply to points.

    The transform is parsed once, on construction, and can then be applied any number of
    times. A ``TransformParameters`` file is *already* a chain: its
    ``InitialTransformParametersFileName`` is followed recursively (resolved relative to
    that file's own directory), so a four-deep affine -> affine -> B-spline -> B-spline
    stack loads from its outermost file alone.

    Unlike `navis`, this does not shell out to the ``transformix`` binary: **Elastix does
    not need to be installed**, and there is no subprocess, no temporary directory and no
    ``copy_files`` dance. Results match ``transformix`` to ~5e-7 - its own print precision.

    It also gives you an **inverse**, which Elastix has no way to compute. See
    :meth:`~navis_fastcore.ElastixTransform.xform_inv`.

    Parameters
    ----------
    path :      str | pathlib.Path | list thereof
                Path to a ``TransformParameters.*.txt`` file. Pass a list to build a chain,
                applied in order: ``points -> path[0] -> path[1] -> ... -> output``.
    invert :    bool | list of bool
                Traverse a transform backwards. A single bool applies to every entry; pass
                a list to set them per transform. Useful when routing through a bridging
                graph, where an edge may be traversed in either direction.

    Attributes
    ----------
    affine :        (4, 4) np.ndarray
                    The matrix of the first linear step of the first transform, if it has
                    one; ``None`` otherwise.
    kinds :         list of list of str
                    The resolved step kinds of each transform, initial first - e.g.
                    ``[["linear", "bspline"]]``.
    grid_size :     (k, 3) np.ndarray
                    Control-point grid size of every B-spline in the chain.
    grid_spacing :  (k, 3) np.ndarray
                    Control-point spacing of every B-spline in the chain.
    grid_origin :   (k, 3) np.ndarray
                    Control-point grid origin of every B-spline in the chain.

    Examples
    --------
    >>> import navis_fastcore as fastcore
    >>> import numpy as np
    >>> xf = fastcore.ElastixTransform("TransformParameters.FixedFANC.txt")  # doctest: +SKIP
    >>> pts = np.array([[50.0, 50.0, 50.0], [100.0, 200.0, 40.0]])
    >>> out = xf.xform(pts)                                                  # doctest: +SKIP
    >>> back = xf.xform_inv(out)                                             # doctest: +SKIP
    >>> np.allclose(xf.xform(back), out, atol=1e-3)                          # doctest: +SKIP
    True

    """

    def __init__(self, path, invert=False):
        if isinstance(path, (str, os.PathLike)):
            paths = [os.fspath(path)]
        else:
            paths = [os.fspath(p) for p in path]
        if not paths:
            raise ValueError("`path` must name at least one transform")

        if isinstance(invert, bool):
            inv = [invert] * len(paths)
        else:
            inv = [bool(i) for i in invert]
            if len(inv) != len(paths):
                raise ValueError(
                    f"`invert` must have one flag per transform: expected "
                    f"{len(paths)}, got {len(inv)}"
                )

        self._paths = paths
        self._invert = inv
        self._xf = _fastcore.ElastixTransform(paths, inv)

    def xform(self, points, out_of_bounds="identity", n_cores=None, progress=False):
        """Transform points forward.

        Parameters
        ----------
        points :            (N, 3) array | (3,) array
                            Coordinates to transform.
        out_of_bounds :     "identity" | "nan"
                            What to do with points that fall outside a B-spline's
                            control-point grid. ``"identity"`` (the default) returns them
                            **unchanged**, which is exactly what ``transformix`` does.
                            ``"nan"`` returns ``NaN`` instead.

                            The default is silent by nature: a neuron straddling the grid
                            edge comes back partly transformed and looks perfectly fine.
                            Use ``"nan"`` when you would rather see the boundary than trust
                            it.
        n_cores :           int, optional
                            Cap the thread pool. ``None`` uses all cores.
        progress :          bool
                            Show a progress bar.

        Returns
        -------
        (N, 3) np.ndarray
                    Transformed coordinates. A ``(3,)`` input gives a ``(3,)`` output.

        """
        pts, was_1d = _prep_points(points)
        out = self._xf.xform(
            pts,
            str(out_of_bounds),
            None if n_cores is None else int(n_cores),
            bool(progress),
        )
        return out[0] if was_1d else out

    def xform_inv(
        self,
        points,
        out_of_bounds="identity",
        initial_guess=None,
        max_iter=50,
        seed_iter=8,
        tolerance=1e-9,
        accuracy=1e-3,
        lattice_points=16_000,
        n_cores=None,
        progress=False,
    ):
        """Transform points backwards - something Elastix itself cannot do.

        Linear steps are inverted exactly. Each B-spline warp has no closed-form inverse
        and is solved per point by damped Gauss-Newton against the analytic Jacobian.

        What is guaranteed is **forward-consistency**: ``xform(xform_inv(y)) == y``, to
        within ``accuracy``. What is *not* guaranteed is that ``xform_inv(xform(p)) == p``,
        because a B-spline warp need not be injective - a strongly folded registration maps
        several points to the same place, and no inverse can recover which one you meant.
        Points with no preimage at all come back as ``NaN``.

        Parameters
        ----------
        points :            (N, 3) array | (3,) array
                            Coordinates to transform.
        out_of_bounds :     "identity" | "nan"
                            See :meth:`~navis_fastcore.ElastixTransform.xform`.
        initial_guess :     (N, 3) array, optional
                            Starting points for the solver. Rarely needed: it seeds itself
                            with a fixed-point iteration, which is what makes it converge
                            even where the deformation is large.
        max_iter :          int
                            Solver budget per point.
        seed_iter :         int
                            Rounds of the fixed-point pre-seed. Zero starts the solver at
                            the target, which fails wherever the deformation is large.
        tolerance :         float
                            Step-size convergence threshold.
        accuracy :          float
                            Accept a solution only if its residual is within this of the
                            target, in world units. Otherwise the row is ``NaN``.
        lattice_points :    int
                            Size of the global seed lattice - the last-resort start for the
                            few points the cheap seeds fail on. Built once per call, and only
                            consulted by points that have already failed, so it costs almost
                            nothing on a well-behaved registration. Set to 0 to disable.
        n_cores :           int, optional
                            Cap the thread pool. ``None`` uses all cores.
        progress :          bool
                            Show a progress bar.

        Returns
        -------
        (N, 3) np.ndarray
                    Coordinates in the source space. Rows with no preimage are ``NaN``.

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
        out = self._xf.xform_inv(
            pts,
            str(out_of_bounds),
            guess,
            int(max_iter),
            int(seed_iter),
            float(tolerance),
            float(accuracy),
            int(lattice_points),
            None if n_cores is None else int(n_cores),
            bool(progress),
        )
        return out[0] if was_1d else out

    @property
    def affine(self):
        """The (4, 4) matrix of the first linear step, or `None`."""
        return self._xf.affine

    @property
    def kinds(self):
        """The step kinds of each transform in the chain, initial first."""
        return self._xf.kinds

    @property
    def invertible(self):
        """Whether `xform_inv` can run. False only for a chain carrying an ``Add`` step."""
        return self._xf.invertible

    @property
    def grid_size(self):
        """Control-point grid size of every B-spline in the chain, `(k, 3)`."""
        return self._xf.grid_size

    @property
    def grid_spacing(self):
        """Control-point spacing of every B-spline in the chain, `(k, 3)`."""
        return self._xf.grid_spacing

    @property
    def grid_origin(self):
        """Control-point grid origin of every B-spline in the chain, `(k, 3)`."""
        return self._xf.grid_origin

    @property
    def paths(self):
        """The files this transform was loaded from."""
        return list(self._paths)

    def __len__(self):
        return self._xf.n_transforms

    def __reduce__(self):
        # Re-load from disk in the child rather than pickling the coefficients through every
        # multiprocessing/joblib fan-out. BANC's `BANC_to_template.txt` is 56 MB.
        return (ElastixTransform, (self._paths, self._invert))

    def __repr__(self):
        return self._xf.__repr__()


def load_elastix_transform(path, invert=False):
    """Load one or more Elastix transforms.

    A convenience wrapper around :class:`~navis_fastcore.ElastixTransform`.

    Parameters
    ----------
    path :      str | pathlib.Path | list thereof
                Path to a ``TransformParameters.*.txt`` file, or several to chain.
    invert :    bool | list of bool
                Traverse a transform backwards.

    Returns
    -------
    ElastixTransform

    """
    return ElastixTransform(path, invert=invert)
