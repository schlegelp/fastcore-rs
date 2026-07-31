"""Performance regression guards.

Opt-in: the default `addopts` deselects the `benchmark` marker, so a normal test
run skips everything here. Run with::

    pytest -m benchmark                 # check against the committed baseline
    pytest -m benchmark --baseline      # record a new baseline
    pytest -m benchmark -s              # ...and see the igraph comparison table

Two things are being measured, and they are deliberately treated differently.

**Regression** (`test_no_regression`) is a gate, but a loose one. CI runners are
noisy and machine speeds differ by more than 2x, so the threshold is 2x *after*
normalising by a calibration workload timed in the same session. A tight
threshold on shared CI produces flakes, and flaky gates get ignored.

**Complexity class** (`test_scaling_is_subquadratic`) is the assertion that
actually catches an accidental O(N^2): a fixed threshold at one size cannot tell
a slow linear implementation from a fast quadratic one, but the ratio between
`N` and `10N` can.

**Speedup vs igraph** (`test_report_speedup_vs_igraph`) is reporting only, never
a gate - it is the number that goes in the changelog.
"""

import json
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pytest

import navis_fastcore as fastcore
import topologies

pytestmark = pytest.mark.benchmark

BASELINE_PATH = Path(__file__).parent / "perf_baseline.json"

#: Sizes for the scaling check. Both are far enough above per-call overhead that
#: the ratio between them measures the algorithm and not the binding layer.
N, N10 = 100_000, 1_000_000

#: A case may be 2x slower than baseline before it fails. See module docstring.
REGRESSION_FACTOR = 2.0

#: `t(10N) / t(N)`. Linear is 10 and O(N log N) is ~11.7, so 20 passes both while
#: rejecting quadratic (which would be 100).
MAX_SCALING_RATIO = 20.0


def cases(topo):
    """The operations under guard, as zero-argument callables.

    Anything set up here (the source samples, the contraction mapping) is built
    once, outside the timed callable, so the measurement is of the function and
    not of the fixture.
    """
    nid, pid, w = topo.node_ids, topo.parent_ids, topo.weights
    sources = nid[:200]
    targets = nid[200:400]
    # Collapse every node onto its component's root: an O(N) contraction that
    # actually removes nodes, rather than an identity mapping that does nothing.
    to_roots = fastcore.connected_components(nid, pid)
    return {
        "classify_nodes": lambda: fastcore.classify_nodes(nid, pid),
        "connected_components": lambda: fastcore.connected_components(nid, pid),
        "dist_to_root": lambda: fastcore.dist_to_root(nid, pid, weights=w),
        "break_segments": lambda: fastcore.break_segments(nid, pid),
        "generate_segments": lambda: fastcore.generate_segments(nid, pid, weights=w),
        "geodesic_matrix_partial": lambda: fastcore.geodesic_matrix(
            nid, pid, sources=sources, targets=targets, weights=w
        ),
        # Fewer sources than the rest: cost here is proportional to *total sub-tree
        # size*, and `nid[:200]` all sit near the top of the backbone, so each one
        # spans almost the whole tree. Asking for 200 of those measures numpy
        # building 20M node IDs rather than the traversal. Both navis call sites
        # (`cut_skeleton`, `split_into_fragments`) pass a handful.
        "descendants": lambda: fastcore.descendants(nid, pid, sources[:20]),
        "paths_to_root": lambda: fastcore.paths_to_root(nid, pid, sources),
        "reroot": lambda: fastcore.reroot(nid, pid, nid[-1:]),
        "contract_nodes": lambda: fastcore.contract_nodes(nid, pid, to_roots),
        "simplify_skeleton": lambda: fastcore.simplify_skeleton(nid, pid, weights=w),
        "adjacency": lambda: fastcore.adjacency(nid, pid, weights=w, directed=False),
        "longest_path": lambda: fastcore.longest_path(nid, pid, weights=w),
        # 5 paths, which is the scale `split_into_fragments` works at. Each round
        # re-scans what is left, so this is where an accidental O(n * N^2) would show.
        "longest_paths": lambda: fastcore.longest_paths(nid, pid, 5, weights=w),
        # The whole point of the closed form is that this is O(N), not Brandes'
        # O(V*E) - which at 1M nodes would not finish.
        "betweenness": lambda: fastcore.betweenness(nid, pid, directed=False),
        "descendant_counts": lambda: fastcore.descendant_counts(nid, pid),
    }


#: `as_graph` results, keyed by the topology it was derived from. The transform costs
#: ~70 ms at 1M nodes and every parametrisation of the graph tests asks for the same
#: handful of graphs, so without this the suite rebuilds them a dozen times over. Safe
#: to memoise because the derivation is deterministic and nothing here mutates it.
_GRAPH_CACHE = {}


def as_graph(topo):
    """The same skeleton as a cyclic, undirected edge list.

    `cases` above works in parent arrays; the primitives below work in edge lists,
    so they need the skeleton transposed into one. Chords are added on purpose: over
    a plain tree `bridges` would return all-true without ever walking a back edge,
    and `parents_from_edges` would have no cycle to break — so a tree would time the
    two functions on the paths they are *not* being guarded for.
    """
    if topo.name in _GRAPH_CACHE:
        return _GRAPH_CACHE[topo.name]

    ids = np.asarray(topo.node_ids)
    parents = np.asarray(topo.parent_ids)
    n = len(ids)

    # `synthetic_neuron` and `fragmented_neuron` both number nodes 0..n-1, so a
    # parent id is already a parent index.
    child = np.flatnonzero(parents >= 0)
    tree = np.column_stack((child, parents[child]))

    # One chord per 20 nodes, within the same component so the graph does not gain
    # connectivity the parent array did not have.
    rng = np.random.default_rng(0)
    comp = fastcore.connected_components(ids, parents)
    order = np.argsort(comp, kind="stable")
    shifted = np.roll(order, -7)
    ok = comp[order] == comp[shifted]
    chords = np.column_stack((order[ok], shifted[ok]))
    chords = chords[rng.permutation(len(chords))[: max(1, n // 20)]]

    edges = np.vstack((tree, chords)).astype(np.uint32)
    # `random()` is [0, 1), so this is [0.1, 1.1) — positive by construction.
    weights = rng.random(len(edges)).astype(np.float32) + np.float32(0.1)

    _GRAPH_CACHE[topo.name] = (edges, weights, n)
    return _GRAPH_CACHE[topo.name]


def graph_cases(topo):
    """The edge-list primitives, as zero-argument callables."""
    edges, weights, n = as_graph(topo)
    # 1000 nodes to span: the scale skeletor's `mst_over_mesh` works at, and enough
    # that the k x k route it replaces would be a million-cell matrix.
    span = np.linspace(0, n - 1, 1000, dtype=np.uint32)
    return {
        "parents_from_edges": lambda: fastcore.parents_from_edges(edges, n),
        "bridges": lambda: fastcore.bridges(edges, n),
        "geodesic_mst_graph": lambda: fastcore.geodesic_mst_graph(
            edges, n, span, weights=weights
        ),
    }


def best_of(fn, repeats=5):
    """Minimum of `repeats` runs, in seconds.

    The minimum, not the mean: wall-clock noise is one-sided (scheduling and
    cache eviction can only ever make a run slower), so the fastest observed run
    is the stable statistic.
    """
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def calibrate():
    """A fixed workload whose timing stands in for "how fast is this machine".

    Normalising by this lets one committed baseline serve machines of different
    speeds - which is the difference between a gate that travels and one that has
    to be re-recorded per runner.
    """
    a = np.arange(4_000_000, dtype=np.float64)
    return best_of(lambda: np.sqrt(a).sum(), repeats=5)


#: The two shapes every case is measured against.
#:
#: "dense" is one arbor. "fragmented" is what a segmentation-derived skeleton actually
#: looks like - one large component plus several thousand small ones - and is the shape
#: that catches work scaling with the number of *components* rather than nodes: root
#: sweeps, per-component bookkeeping, anything that re-seeds a traversal per root.
SHAPES = ("dense", "fragmented")


# ------------------------------------------------------------------------ fixtures


@pytest.fixture(scope="session")
def topos():
    """Session-cached topology builder, keyed by (shape, size).

    Built lazily so a `-k` selection only pays for the shapes it actually runs; cached
    because the 1M-node builds are shared by several tests.
    """
    cache = {}

    def get(shape, n):
        if (shape, n) not in cache:
            cache[shape, n] = (
                topologies.synthetic_neuron(n)
                if shape == "dense"
                else topologies.fragmented_neuron(n)
            )
        return cache[shape, n]

    return get


@pytest.fixture(scope="session")
def big(topos):
    return topos("dense", N)


@pytest.fixture(scope="session")
def calibration():
    return calibrate()


@pytest.fixture(scope="session")
def baseline(request, calibration):
    """The committed baseline, or a recorder when `--baseline` was passed."""
    recording = request.config.getoption("--baseline")

    if recording:
        recorded = {"n": N, "calibration": calibration, "cases": {}}
        yield recorded
        BASELINE_PATH.write_text(json.dumps(recorded, indent=2, sort_keys=True) + "\n")
        print(f"\nWrote baseline to {BASELINE_PATH}")
        return

    if not BASELINE_PATH.exists():
        pytest.skip(f"no baseline at {BASELINE_PATH}; record one with --baseline")
    yield json.loads(BASELINE_PATH.read_text())


#: Every guarded operation, as `(baseline key prefix, builder, case name)`.
#:
#: The two suites are parametrised together rather than given a test apiece: they gate
#: on exactly the same thing and differ only in which builder turns a topology into
#: callables. `graph_cases` still runs only for its own cases, which is what the
#: separation was ever for — it transposes the skeleton into an edge list, and the
#: parent-array cases would pay for that without looking at it.
SUITES = [("", cases), ("graph/", graph_cases)]
GUARDED = [(prefix, build, case)
           for prefix, build in SUITES
           for case in sorted(build(topologies.synthetic_neuron(64)))]

#: Readable ids, so a failure names the case rather than a tuple.
GUARDED_IDS = [f"{prefix}{case}" for prefix, _, case in GUARDED]


# --------------------------------------------------------------------- regression


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("prefix,build,case", GUARDED, ids=GUARDED_IDS)
def test_no_regression(prefix, build, case, shape, topos, baseline, calibration, request):
    """Each operation stays within `REGRESSION_FACTOR` of its recorded time."""
    key = f"{prefix}{shape}/{case}"
    elapsed = best_of(build(topos(shape, N))[case])

    if request.config.getoption("--baseline"):
        baseline["cases"][key] = elapsed
        pytest.skip("recording baseline")

    if key not in baseline["cases"]:
        pytest.skip(f"{key} is not in the baseline; re-record with --baseline")
    if baseline["n"] != N:
        pytest.skip(f"baseline was recorded at n={baseline['n']}, now n={N}")

    # Scale the recorded time by how much slower/faster this machine is.
    speed = calibration / baseline["calibration"]
    budget = baseline["cases"][key] * speed * REGRESSION_FACTOR

    assert elapsed < budget, (
        f"{key}: {elapsed * 1e3:.1f} ms exceeds budget {budget * 1e3:.1f} ms "
        f"(baseline {baseline['cases'][key] * 1e3:.1f} ms, "
        f"machine factor {speed:.2f}x)"
    )


# --------------------------------------------------------------------- complexity


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("prefix,build,case", GUARDED, ids=GUARDED_IDS)
def test_scaling_is_subquadratic(prefix, build, case, shape, topos):
    """10x the nodes must not cost ~100x the time.

    Run against the fragmented shape too: there 10x the nodes is also ~10x the
    *components*, so anything quadratic in component count — a per-component search,
    a root sweep — shows up here and nowhere else.
    """
    t_n = best_of(build(topos(shape, N))[case], repeats=3)
    t_10n = best_of(build(topos(shape, N10))[case], repeats=3)

    ratio = t_10n / t_n
    assert ratio < MAX_SCALING_RATIO, (
        f"{prefix}{shape}/{case}: {N} -> {N10} nodes cost {ratio:.1f}x more time "
        f"({t_n * 1e3:.1f} -> {t_10n * 1e3:.1f} ms); "
        f"that is worse than O(N log N) and suggests a quadratic path"
    )


# ------------------------------------------------------------------------- memory


def test_partial_geodesic_memory_is_block_sized(big):
    """A sources x targets query must allocate the block, not an N x N matrix.

    This is the failure mode that has actually bitten navis in production: asking
    for a small block from a backend that answers it by building the full matrix.

    Caveat: `tracemalloc` only sees Python-level allocations, so this proves the
    *returned* array is block-sized - it cannot see a transient allocation inside
    Rust. Guarding that needs a counting global allocator behind a cargo feature.
    """
    sources, targets = big.node_ids[:200], big.node_ids[200:400]

    tracemalloc.start()
    fastcore.geodesic_matrix(
        big.node_ids, big.parent_ids, sources=sources, targets=targets,
        weights=big.weights,
    )
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    block = len(sources) * len(targets) * 4  # float32
    assert peak < 100 * block, (
        f"peak {peak / 1e6:.1f} MB for a {len(sources)}x{len(targets)} block "
        f"({block / 1e6:.3f} MB) - is the full matrix being built?"
    )


# -------------------------------------------------------------------- vs igraph


def test_report_speedup_vs_igraph(big, capsys):
    """Print the fastcore-vs-oracle table. Reports only; never fails.

    Read these numbers carefully - they are **not** publishable "vs igraph"
    speedups:

    - The reference is `igraph_oracle.py`, which is written to be obviously
      correct, not fast. Where it transcribes navis' actual fallback
      (`classify_nodes`, `break_segments`, `connected_components`) the ratio is
      meaningful; where it does not, it is measuring the oracle's naivety.
    - `dist_to_root` is the clearest example: the oracle asks igraph for an
      `N x roots` matrix through the Python layer, whereas navis' real fallback
      builds a scipy CSR and runs `csgraph.dijkstra(min_only=True)`. The five-digit
      ratio below is an artefact of the oracle, not a property of fastcore.
    - `generate_segments` likewise pays for the oracle's per-node
      `g.successors()` calls.

    Treat this as a smoke signal ("is fastcore in the right ballpark") and
    benchmark against the real fallback before quoting anything.
    """
    igraph = pytest.importorskip("igraph")  # noqa: F841
    import igraph_oracle as oracle

    g = oracle.as_igraph(big)
    #: True where the oracle is a faithful transcription of navis' igraph
    #: fallback, so the ratio says something about fastcore rather than about
    #: the oracle.
    theirs = {
        "classify_nodes": (lambda: oracle.classify_nodes(g), True),
        "connected_components": (lambda: oracle.connected_components(g), True),
        "break_segments": (lambda: oracle.break_segments(g), True),
        "generate_segments": (lambda: oracle.generate_segments(g), False),
        "dist_to_root": (lambda: oracle.dist_to_root(g), False),
    }
    ours = cases(big)

    with capsys.disabled():
        print(f"\n  {N:,}-node synthetic skeleton")
        print("  (* = oracle is not navis' real fallback; ratio is not meaningful)\n")
        print(f"  {'case':<24}{'fastcore':>12}{'oracle':>12}{'ratio':>10}")
        for name, (fn, faithful) in theirs.items():
            t_ours = best_of(ours[name], repeats=3)
            t_theirs = best_of(fn, repeats=3)
            mark = "" if faithful else " *"
            print(
                f"  {name:<24}{t_ours * 1e3:>10.1f}ms{t_theirs * 1e3:>10.1f}ms"
                f"{t_theirs / t_ours:>9.1f}x{mark}"
            )


def test_report_graph_primitives_vs_igraph(topos, capsys):
    """Print the edge-list primitives against igraph. Reports only; never fails.

    Unlike the table above these *are* like-for-like: `bridges` is igraph's own
    `Graph.bridges()`, and the spanning-forest reference is the per-component
    `Graph.bfs()` loop skeletor's `edges_to_graph` is a networkx spelling of. The
    fragmented shape is included because that is where the two diverge in kind
    rather than in constant: igraph pays per component, fastcore does not.
    """
    igraph = pytest.importorskip("igraph")

    with capsys.disabled():
        print(f"\n  {'shape':<12}{'case':<20}{'fastcore':>12}{'igraph':>12}{'ratio':>10}")
        for shape in SHAPES:
            # One topology per shape, so both columns time the same graph.
            topo = topos(shape, N)
            edges, _, n = as_graph(topo)
            g = igraph.Graph(n=n, edges=edges.tolist(), directed=False)

            theirs = {
                "bridges": lambda g=g: g.bridges(),
                # One BFS per component, which is what any graph library forces —
                # and the reason `parents_from_edges` sweeps them into one column.
                "parents_from_edges": lambda g=g: [
                    g.bfs(int(c[0])) for c in g.connected_components()
                ],
            }
            ours = graph_cases(topo)
            for name, fn in theirs.items():
                t_ours = best_of(ours[name], repeats=3)
                t_theirs = best_of(fn, repeats=1)
                print(
                    f"  {shape:<12}{name:<20}{t_ours * 1e3:>10.1f}ms"
                    f"{t_theirs * 1e3:>10.1f}ms{t_theirs / t_ours:>9.1f}x"
                )


def test_report_geodesic_mst_vs_dense_matrix(big, capsys):
    """Print the matrix-free MST against the route it replaces. Reports only.

    The reference is exactly what a caller would otherwise write: ask for the
    `k x k` geodesic matrix, hand it to scipy's matrix MST. The point of the table
    is the *shape* of the two columns — ours is flat in `k` because it is one
    sweep whatever `k` is, theirs is quadratic.
    """
    from scipy.sparse.csgraph import minimum_spanning_tree as scipy_mst

    edges, weights, n = as_graph(big)

    with capsys.disabled():
        print(f"\n  {'k':>7}{'fastcore':>12}{'k x k + MST':>15}{'ratio':>10}{'matrix':>10}")
        for k in (250, 1_000, 4_000):
            span = np.linspace(0, n - 1, k, dtype=np.uint32)

            def dense(span=span):
                d = fastcore.geodesic_matrix_graph(
                    edges, n, weights=weights, sources=span, targets=span
                )
                d = np.where(d < 0, np.inf, d).astype(np.float64)
                return scipy_mst(d, overwrite=True)

            t_ours = best_of(
                lambda span=span: fastcore.geodesic_mst_graph(
                    edges, n, span, weights=weights
                ),
                repeats=3,
            )
            t_theirs = best_of(dense, repeats=1)
            print(
                f"  {k:>7,}{t_ours * 1e3:>10.1f}ms{t_theirs * 1e3:>13.1f}ms"
                f"{t_theirs / t_ours:>9.1f}x{k * k * 4 / 1e6:>8.0f} MB"
            )


def test_report_betweenness_vs_brandes(capsys):
    """Print the betweenness comparison. Reports only; never fails.

    Kept out of the table above because it cannot be run at the same size. igraph
    computes betweenness with Brandes' algorithm, which is O(V*E) - measurably
    quadratic on a tree, where E ~ V. Extrapolating the numbers below, a
    like-for-like run at `N` = 100,000 would take minutes per repeat, so the
    comparison is made where Brandes is still tractable and the *trend* is the
    point: fastcore is O(N), so its column barely moves.
    """
    igraph = pytest.importorskip("igraph")  # noqa: F841
    import igraph_oracle as oracle

    with capsys.disabled():
        print(f"\n  {'nodes':>8}{'fastcore':>12}{'igraph (Brandes)':>20}{'ratio':>10}")
        for n in (2_000, 5_000, 10_000):
            topo = topologies.synthetic_neuron(n)
            g = oracle.as_igraph(topo)

            ours = best_of(
                lambda: fastcore.betweenness(
                    topo.node_ids, topo.parent_ids, directed=False
                ),
                repeats=3,
            )
            theirs = best_of(lambda: g.betweenness(directed=False), repeats=1)
            print(
                f"  {n:>8,}{ours * 1e3:>10.2f}ms{theirs * 1e3:>18.1f}ms"
                f"{theirs / ours:>9.0f}x"
            )
