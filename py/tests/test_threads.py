"""Tests for process-wide and per-call thread control.

The thread pool is built at most once per process, so every test that *sets* it
has to run in a fresh interpreter — pytest's own process has already built the
pool by the time these run (any earlier parallel call does it). Only the checks
that leave the pool alone can run in-process.
"""

import os
import subprocess
import sys
import tempfile
import textwrap

import numpy as np
import pytest

import navis_fastcore as fastcore


#: Emscripten (Pyodide) has no `fork`/`exec`, so a fresh interpreter cannot be
#: spawned there and `subprocess.run` raises `OSError(ENOTSUP)`. That platform is
#: also single-threaded, so what these tests pin - the size of the rayon pool and
#: when it gets built - is not a thing that varies there anyway. The in-process
#: tests below still run.
needs_subprocess = pytest.mark.skipif(
    sys.platform in ("emscripten", "wasi"),
    reason="wasm platforms cannot spawn processes",
)


def run_snippet(body, env=None):
    """Run `body` in a fresh interpreter; return its completed process.

    Runs from a scratch directory rather than inheriting pytest's: `python -c`
    puts the working directory first on `sys.path`, and pytest's is the source
    tree, whose `navis_fastcore/` shadows the installed package but carries no
    compiled `_fastcore` unless it happens to have been built in place.
    """
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(body)],
        capture_output=True,
        text=True,
        cwd=tempfile.gettempdir(),
        env=env,
    )


def fragmented(n=20_000, n_frags=200, seed=0):
    """A chain skeleton cut into `n_frags` disconnected pieces."""
    node_ids = np.arange(n, dtype=np.int64)
    parent_ids = node_ids - 1
    parent_ids[:: max(1, n // n_frags)] = -1
    coords = np.cumsum(
        np.random.default_rng(seed).normal(size=(n, 3)), axis=0
    ) * 10
    return node_ids, parent_ids, coords


# ---------------------------------------------------------------------------
# set_num_threads / get_num_threads
# ---------------------------------------------------------------------------


@needs_subprocess
def test_set_num_threads_sizes_the_pool():
    proc = run_snippet("""
        import navis_fastcore as fastcore
        fastcore.set_num_threads(2)
        print(fastcore.get_num_threads())
    """)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "2"


@needs_subprocess
def test_set_num_threads_is_idempotent():
    """Safe to call from a worker-init hook, which fires once per chunk."""
    proc = run_snippet("""
        import navis_fastcore as fastcore
        fastcore.set_num_threads(2)
        fastcore.set_num_threads(2)
        fastcore.set_num_threads(2)
        print(fastcore.get_num_threads())
    """)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "2"


@needs_subprocess
def test_set_num_threads_refuses_to_resize():
    proc = run_snippet("""
        import navis_fastcore as fastcore
        fastcore.set_num_threads(2)
        try:
            fastcore.set_num_threads(4)
        except RuntimeError as e:
            print("raised:", e)
    """)
    assert proc.returncode == 0, proc.stderr
    assert "raised:" in proc.stdout
    assert "already running with 2 thread(s)" in proc.stdout


@needs_subprocess
def test_set_num_threads_after_parallel_work_raises():
    """The first parallel call builds the pool, so setting it afterwards fails."""
    proc = run_snippet("""
        import numpy as np
        import navis_fastcore as fastcore
        ids = np.arange(1000, dtype=np.int64)
        par = ids - 1
        par[::100] = -1
        fastcore.heal_skeleton(ids, par, np.zeros((1000, 3)))
        # `n + 1` rather than a literal: any literal could coincide with this
        # machine's default pool size, and then the call would legitimately
        # succeed as a no-op and the test would pass for the wrong reason.
        n = fastcore.get_num_threads()
        try:
            fastcore.set_num_threads(n + 1)
            print("no error")
        except RuntimeError:
            print("raised")
    """)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "raised"


@needs_subprocess
def test_get_num_threads_before_set_blocks_the_set():
    """Documents the footgun the docstring warns about: asking builds the pool."""
    proc = run_snippet("""
        import navis_fastcore as fastcore
        n = fastcore.get_num_threads()
        try:
            fastcore.set_num_threads(n + 1)
            print("no error")
        except RuntimeError:
            print("raised")
    """)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "raised"


def test_set_num_threads_rejects_zero():
    """Rejected before the pool is touched, so this is safe in-process."""
    with pytest.raises(ValueError):
        fastcore.set_num_threads(0)


@needs_subprocess
def test_respects_rayon_env_var():
    """`RAYON_NUM_THREADS` is the no-code-change version of the same lever."""
    proc = run_snippet(
        "import navis_fastcore as f; print(f.get_num_threads())",
        env={**os.environ, "RAYON_NUM_THREADS": "3"},
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "3"


# ---------------------------------------------------------------------------
# Per-call `threads=`
# ---------------------------------------------------------------------------
# Capping must not change the answer. It is worth pinning for healing in
# particular: the bridge search shares a per-component bound across threads, so
# how many threads participate decides the order in which candidates are
# rejected - and a tie between two equally short bridges has to resolve the same
# way regardless.


@pytest.mark.parametrize("threads", [1, 2, 4])
def test_heal_skeleton_threads_agree(threads):
    node_ids, parent_ids, coords = fragmented()
    expected = fastcore.heal_skeleton(node_ids, parent_ids, coords)
    got = fastcore.heal_skeleton(node_ids, parent_ids, coords, threads=threads)
    assert np.array_equal(got, expected)
    assert (expected < 0).sum() == 1, "should have healed to a single root"


@pytest.mark.parametrize("threads", [1, 2, 4])
def test_stitch_fragments_threads_agree(threads):
    node_ids, parent_ids, coords = fragmented()
    edges, dists = fastcore.stitch_fragments(node_ids, parent_ids, coords)
    got_edges, got_dists = fastcore.stitch_fragments(
        node_ids, parent_ids, coords, threads=threads
    )
    assert np.array_equal(got_edges, edges)
    assert np.array_equal(got_dists, dists)


@pytest.mark.parametrize("threads", [1, 2, 4])
def test_geodesic_pairs_threads_agree(threads):
    node_ids = np.arange(7)
    parent_ids = np.array([-1, 0, 1, 2, 1, 4, 5])
    pairs = np.array([(0, 1), (0, 2), (3, 6), (5, 5)])

    expected = fastcore.geodesic_pairs(node_ids, parent_ids, pairs)
    got = fastcore.geodesic_pairs(node_ids, parent_ids, pairs, threads=threads)
    assert np.array_equal(got, expected)


def test_geodesic_pairs_threads_chunk_correctly():
    """More threads than pairs must not drop or duplicate any of them.

    The pairs are split into one chunk per worker, so a cap that exceeds the
    number of pairs is the edge case where an off-by-one in the chunk size
    shows up.
    """
    node_ids = np.arange(5)
    parent_ids = np.array([-1, 0, 1, 2, 3])
    pairs = np.array([(0, 1), (0, 4)])

    expected = fastcore.geodesic_pairs(node_ids, parent_ids, pairs)
    assert np.array_equal(expected, [1.0, 4.0])
    for threads in (1, 2, 8):
        got = fastcore.geodesic_pairs(node_ids, parent_ids, pairs, threads=threads)
        assert np.array_equal(got, expected), threads
