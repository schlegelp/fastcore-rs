"""Process-wide control over how many threads fastcore uses."""

from . import _fastcore

__all__ = [
    "set_num_threads",
    "get_num_threads",
]


def set_num_threads(n):
    """Set the number of threads used for parallel work in this process.

    By default fastcore uses every core it can see, which is the right answer for
    a single call in a single process and the wrong one when the *caller* is
    already spreading work over processes. Nothing tells a worker process that it
    is one of twenty, so it has to be told.

    Call this once, before any other fastcore function.

    Parameters
    ----------
    n :         int
                Number of threads. Must be >= 1.

    Raises
    ------
    ValueError
                If ``n < 1``.
    RuntimeError
                If the thread pool already exists at a different size. It is
                built once per process by whichever comes first: an earlier
                ``set_num_threads``, the ``RAYON_NUM_THREADS`` environment
                variable, a call to :func:`get_num_threads`, or simply the first
                parallel fastcore call.

    See Also
    --------
    :func:`get_num_threads`
                Read the current thread count back.

    Examples
    --------
    >>> import navis_fastcore as fastcore
    >>> fastcore.set_num_threads(1)                       # doctest: +SKIP

    Under navis' multiprocessing, set it in each worker as it starts. The hook is
    shipped to the workers by pickle, so it has to be picklable — a
    ``functools.partial`` of this function is, a lambda is not:

    >>> import navis, functools                           # doctest: +SKIP
    >>> navis.compute.worker_init_hooks.append(           # doctest: +SKIP
    ...     functools.partial(fastcore.set_num_threads, 1)
    ... )
    >>> navis.heal_skeleton(nl, parallel=True, n_cores=20)  # doctest: +SKIP

    Calling it repeatedly with the same value is fine, which is what makes it
    safe in a hook that fires more than once. Calling it with a *different* value
    raises, because the pool cannot be resized once built.

    """
    _fastcore.set_num_threads(int(n))


def get_num_threads():
    """Number of threads available for parallel work in this process.

    Note that asking builds the thread pool if it does not exist yet — which is
    then exactly what makes a subsequent :func:`set_num_threads` fail. Set first,
    ask second.

    Returns
    -------
    int

    See Also
    --------
    :func:`set_num_threads`
                Set the thread count.

    """
    return _fastcore.get_num_threads()
