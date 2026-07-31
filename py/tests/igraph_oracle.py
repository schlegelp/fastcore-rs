"""igraph reference implementations of the tree algorithms fastcore provides.

These exist so fastcore can be checked against an *independent* implementation
rather than against itself. igraph is a test-only dependency: nothing here is
imported by the package, and the suites that use it are skipped when it is
absent (see `test_igraph_parity.py`).

Where navis has an igraph fallback for the same operation, the function below is
a transcription of it, cited by file and line. Those fallbacks are exactly the
code navis deletes once it moves onto fastcore, so pinning fastcore against them
is what makes that deletion safe.

Two conventions, both matching navis' `neuron2igraph`
(navis: `graph/converters.py:524-568`):

- the graph is **directed child -> parent**, so a node's route to its root
  follows edges forwards (igraph `mode="OUT"`) and a root has out-degree 0;
- `weight` is an *edge* attribute, whereas fastcore takes a per-**node** weight
  vector indexed by the child. `as_igraph` does that translation.
"""

import warnings

import igraph
import numpy as np

__all__ = [
    "as_igraph",
    "classify_nodes",
    "connected_components",
    "canonical_labels",
    "dist_to_root",
    "geodesic_matrix",
    "break_segments",
    "generate_segments",
    "descendants",
    "paths_to_root",
    "reroot",
    "simplify_skeleton",
    "adjacency",
    "vertex_of",
    "longest_path",
    "betweenness",
    "descendant_counts",
]


def as_igraph(topo, weighted=True):
    """Build the igraph a `Topology` describes, the way navis would build it.

    Vertex order is row order, so vertex index == row index throughout; the
    `node_id` attribute carries the original IDs.
    """
    node_ids = topo.node_ids.tolist()
    n = len(node_ids)
    id2ix = {nid: i for i, nid in enumerate(node_ids)}

    # A node is a root if its parent is -1 *or* names an ID that isn't present.
    # `_ids_to_indices` treats both as -1, so the oracle has to agree.
    children, parents = [], []
    for child_ix, pid in enumerate(topo.parent_ids.tolist()):
        parent_ix = id2ix.get(pid)
        if parent_ix is not None:
            children.append(child_ix)
            parents.append(parent_ix)

    g = igraph.Graph(list(zip(children, parents)), n=n, directed=True)
    g.vs["node_id"] = node_ids
    if weighted:
        # fastcore indexes weights by child; igraph by edge. Edges were added in
        # `children` order, so this line is the whole translation. Assigned even
        # when there are no edges: igraph raises on `weights="weight"` if the
        # attribute was never created, and `single_node` has an empty edge set.
        g.es["weight"] = topo.weights[children].tolist()
    return g


def _w(weighted):
    """igraph's `weights` argument for a weighted/unweighted query."""
    return "weight" if weighted else None


# ------------------------------------------------------------------ classification

#: fastcore's node type codes (`dag.classify_nodes`).
ROOT, LEAF, BRANCH, SLAB = 0, 1, 2, 3


def classify_nodes(g):
    """Node type per vertex: root / leaf / branch / slab.

    Precedence matters and is not symmetric: **root wins**. A root with two
    children is a root, not a branch point, and an isolated node is a root, not
    a leaf. igraph's own `vs.select(_indegree_gt=1, _outdegree=1)` encodes the
    same precedence by requiring out-degree 1 on a branch point
    (navis: `graph/graph_utils.py:383-385`).
    """
    indeg = np.asarray(g.indegree())
    outdeg = np.asarray(g.outdegree())

    out = np.full(g.vcount(), SLAB, dtype=np.int32)
    out[indeg == 0] = LEAF
    out[indeg > 1] = BRANCH
    out[outdeg == 0] = ROOT  # last: overrides both of the above
    return out


# -------------------------------------------------------------- connected components


def connected_components(g):
    """Weakly connected component membership per vertex."""
    return np.asarray(g.components(mode="WEAK").membership)


def canonical_labels(labels):
    """Relabel so a partition can be compared regardless of the label values.

    fastcore labels a component by its root's *node ID*; igraph labels by an
    arbitrary running integer. Mapping both to "row index of the first member of
    this component" makes them comparable while still catching a node placed in
    the wrong component.
    """
    labels = np.asarray(labels)
    first = {}
    out = np.empty(len(labels), dtype=np.int64)
    for i, lab in enumerate(labels.tolist()):
        out[i] = first.setdefault(lab, i)
    return out


# --------------------------------------------------------------------- distances


def dist_to_root(g, weighted=True):
    """Distance from every vertex to its own root.

    Every node in a forest reaches exactly one root, so taking the minimum over
    all roots gives the distance to *its* root - the trick navis relies on to
    keep this O(N) instead of building a roots x N matrix
    (navis: `graph/graph_utils.py:460-475`).
    """
    roots = np.flatnonzero(np.asarray(g.outdegree()) == 0)
    if not g.vcount():
        return np.empty(0, dtype=np.float64)

    # Edges run child -> parent, so a node reaches its root going "OUT".
    d = np.asarray(g.distances(target=roots.tolist(), weights=_w(weighted), mode="OUT"))
    return d.min(axis=1)


def geodesic_matrix(g, sources=None, targets=None, directed=False, weighted=True):
    """All- or partial-pairs geodesic distances, in fastcore's output convention.

    fastcore reports unreachable pairs as `-1`; igraph as `inf`.
    `directed=True` means "towards the root only", i.e. igraph `mode="OUT"`.
    """
    d = np.asarray(
        g.distances(
            source=sources,
            target=targets,
            weights=_w(weighted),
            mode="OUT" if directed else "ALL",
        ),
        dtype=np.float64,
    )
    d[~np.isfinite(d)] = -1
    return d


# ---------------------------------------------------------------------- segments


def break_segments(g):
    """Linear segments between ends, branch points and roots, as node IDs.

    Transcribed from navis' igraph fallback
    (navis: `graph/graph_utils.py:382-411`). The `sorted(seeds)` is load-bearing
    and is called out as such there: without it the walk order is a Python set's
    hash order, which is arbitrary and does not match what fastcore returns.
    """
    end = g.vs.select(_indegree=0).indices
    branch = g.vs.select(_indegree_gt=1, _outdegree=1).indices
    root = g.vs.select(_outdegree=0).indices

    # Seeds are ends and branch points; isolated nodes are both "end" and "root"
    # and drop out here.
    seeds = set(end + branch) - set(root)
    stops = set(branch + root)

    seg_list = []
    for s in sorted(seeds):
        parent = g.successors(s)[0]
        seg = [s, parent]
        while parent not in stops:
            parent = g.successors(parent)[0]
            seg.append(parent)
        seg_list.append(seg)

    ids = np.asarray(g.vs["node_id"])
    return [ids[s] for s in seg_list]


def generate_segments(g, weighted=True):
    """Maximal linear segments, longest first, as (segments, lengths).

    Transcribed from navis' igraph fallback
    (navis: `graph/graph_utils.py:135-172`): walk root-ward from every leaf,
    longest-leaf-first, stopping at the first node another walk already claimed.

    Two documented divergences from navis' version, both deliberate:

    1. navis appends isolated nodes as single-node segments *after* sorting, but
       re-sorts only `lengths` - so on a neuron with isolated nodes its
       `segments` and `lengths` come back misaligned. We emit isolated nodes in
       the sort like any other segment.
    2. On tied lengths navis falls through to comparing the node lists
       themselves. That ordering is arbitrary, so callers compare tie-robustly
       (see `test_igraph_parity.py`).
    """
    d = dist_to_root(g, weighted)
    indeg = np.asarray(g.indegree())
    outdeg = np.asarray(g.outdegree())

    # Leaves only - an isolated node is a root, not a leaf.
    ends = np.flatnonzero((indeg == 0) & (outdeg > 0)).tolist()
    ends.sort(key=lambda v: d[v], reverse=True)

    seen = set()
    sequences = []
    for v in ends:
        seq = [v]
        parents = g.successors(v)
        while parents:
            p = parents[0]
            seq.append(p)
            if p in seen:
                break
            seen.add(p)
            parents = g.successors(p)
        if len(seq) > 1:
            sequences.append(seq)

    # Isolated nodes form their own zero-length segment (see divergence 1).
    sequences += [[v] for v in np.flatnonzero((indeg == 0) & (outdeg == 0)).tolist()]

    lengths = [d[s[0]] - d[s[-1]] for s in sequences]
    order = sorted(range(len(sequences)), key=lambda i: lengths[i], reverse=True)

    ids = np.asarray(g.vs["node_id"])
    return (
        [ids[sequences[i]] for i in order],
        np.array([lengths[i] for i in order], dtype=np.float64),
    )


# ------------------------------------------------------------- sub-trees and paths


def vertex_of(g):
    """`{node_id: vertex index}` - the inverse of the `node_id` attribute.

    The oracle speaks node IDs at both ends so callers never have to build this
    themselves; `as_igraph` keeps vertex index == row index.
    """
    return {nid: i for i, nid in enumerate(g.vs["node_id"])}


def descendants(g, sources):
    """Vertices of the sub-tree distal to each source, by node ID.

    Edges run child -> parent, so "everything below `v`" is everything that reaches
    `v` going *against* the arrows - igraph's `subcomponent(v, mode="IN")`.
    """
    ids = np.asarray(g.vs["node_id"])
    ix = vertex_of(g)
    return [np.sort(ids[g.subcomponent(ix[v], mode="IN")]) for v in sources]


def paths_to_root(g, sources):
    """Node sequence from each source up to its root, source first. By node ID."""
    roots = np.flatnonzero(np.asarray(g.outdegree()) == 0).tolist()
    ids = np.asarray(g.vs["node_id"])
    ix = vertex_of(g)

    out = []
    for source in sources:
        v = ix[source]
        # Only one root is reachable, so exactly one of these paths is non-empty -
        # except for a source that *is* a root, where igraph returns [v] itself.
        # On a fragmented skeleton the other roots are genuinely unreachable, and
        # igraph warns about that; it is the expected answer here, not a problem.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            paths = g.get_shortest_paths(int(v), to=roots, mode="OUT")
        best = max(paths, key=len)
        out.append(ids[best])
    return out


def reroot(g, new_roots):
    """Parent ID per node after re-rooting, by node ID.

    Re-rooting is just "orient every edge away from the new root", so a BFS over the
    *undirected* graph gives it directly - which is also how navis' `edges2neuron`
    orients a raw edge list (navis: `graph/converters.py:812-818`).
    """
    ids = np.asarray(g.vs["node_id"])
    ix = vertex_of(g)
    new_roots = [ix[r] for r in new_roots]
    und = g.as_undirected()

    parent_ix = np.full(g.vcount(), -1, dtype=np.int64)
    claimed = np.zeros(g.vcount(), dtype=bool)

    def orient(start):
        # igraph's `bfs` returns (vids, layer starts, parents). `vids` holds only the
        # start's own component - it does not restart in the others - and the start's
        # own parent comes back as -1, which is already fastcore's root marker.
        vids, _, parents = und.bfs(int(start))
        for v in vids:
            claimed[v] = True
            parent_ix[v] = parents[v]

    for root in new_roots:
        if not claimed[int(root)]:
            orient(root)

    # Components nobody named keep their original orientation.
    for v in range(g.vcount()):
        if not claimed[v] and g.outdegree(v) == 0:
            orient(v)

    return np.where(parent_ix < 0, -1, ids[parent_ix])


# ------------------------------------------------------------------ simplification


def simplify_skeleton(g, weighted=True):
    """Keep only roots, leafs and branch points, preserving path lengths.

    Transcribed from navis' `simplify_graph` igraph branch
    (navis: `graph/converters.py:403-431`): the kept set is exactly its
    `roots | leafs | branches`, and each survivor is walked up to the next survivor.

    The replacement edge's weight is asked of igraph (`distances` along the chain)
    rather than accumulated here, so the number is arrived at independently.
    """
    ids = np.asarray(g.vs["node_id"])
    indeg = np.asarray(g.indegree())
    outdeg = np.asarray(g.outdegree())

    leafs = set(np.flatnonzero((indeg == 0) & (outdeg != 0)).tolist())
    branches = set(np.flatnonzero((indeg > 1) & (outdeg != 0)).tolist())
    roots = set(np.flatnonzero(outdeg == 0).tolist())
    keep = sorted(roots | leafs | branches)

    parents, weights = [], []
    for v in keep:
        if v in roots:
            parents.append(-1)
            weights.append(0.0)
            continue

        # Walk up to the next kept node.
        node = g.successors(v)[0]
        while node not in roots and node not in leafs and node not in branches:
            node = g.successors(node)[0]

        parents.append(ids[node])
        weights.append(
            g.distances(int(v), int(node), mode="OUT", weights=_w(weighted))[0][0]
        )

    return ids[keep], np.asarray(parents), np.asarray(weights, dtype=np.float64)


# ---------------------------------------------------------------------- adjacency


def adjacency(g, weighted=True, directed=True, transpose=False):
    """Dense adjacency, in fastcore's row/column convention.

    Transcribed from navis' `_igraph_to_sparse`
    (navis: `graph/graph_utils.py:2433-2450`), densified so a test can compare it
    without depending on scipy's sparse internals. Small fixtures only.
    """
    n = g.vcount()
    edges = np.asarray(g.get_edgelist(), dtype=np.int64).reshape(-1, 2)
    w = (
        np.asarray(g.es["weight"], dtype=np.float64)
        if weighted and g.ecount()
        else np.ones(len(edges))
    )

    rows, cols = edges[:, 0], edges[:, 1]
    if not directed:
        rows, cols = np.concatenate([rows, cols]), np.concatenate([cols, rows])
        w = np.concatenate([w, w])
    if transpose:
        rows, cols = cols, rows

    out = np.zeros((n, n), dtype=np.float64)
    out[rows, cols] = w
    return out


# ------------------------------------------------------------------- longest path


def longest_path(g, weighted=True):
    """The longest weighted path from a node to its root, as node IDs.

    Transcribed from navis' `_longest_weighted_path`
    (navis: `graph/graph_utils.py:1741-1773`): join every root to one virtual
    super-sink at zero cost, run a *single* search to find the node farthest from
    its own root, then ask igraph for the route back.

    Its `np.argmax` breaks ties towards the lowest vertex index, which is the
    behaviour fastcore has to reproduce for a stable answer.
    """
    n = g.vcount()
    sinks = np.flatnonzero(np.asarray(g.outdegree()) == 0)
    if not len(sinks):
        return np.empty(0, dtype=np.int64)

    h = g.copy()
    h.add_vertices(1)
    h.add_edges([(int(s), n) for s in sinks])
    if weighted:
        h.es["weight"] = np.concatenate(
            [np.asarray(g.es["weight"]), np.zeros(len(sinks))]
        ).tolist()

    dists = np.asarray(
        h.distances(source=[n], weights=_w(weighted), mode="IN")[0][:n], dtype=np.float64
    )
    dists[~np.isfinite(dists)] = -1
    start = int(np.argmax(dists))

    path = h.get_shortest_paths(start, to=n, weights=_w(weighted), mode="OUT")[0]

    ids = np.asarray(g.vs["node_id"])
    return ids[path[:-1]]  # drop the virtual super-sink


# ------------------------------------------------------------------- betweenness


def betweenness(g, directed=True):
    """igraph's own betweenness centrality.

    Unlike most of this module this is not a transcription of navis - navis'
    `betweeness_centrality(from_=None)` simply forwards to `g.betweenness()`, so
    this *is* the reference.
    """
    return np.asarray(g.betweenness(directed=directed))


def descendant_counts(g, targets=None):
    """For each vertex, how many `targets` lie strictly below it.

    This is what navis' `betweeness_centrality(from_=...)` branch actually
    computes (navis: `morpho/mmetrics.py:1741-1757`), despite the name: it walks
    root -> source paths and tallies every node *except* the source itself, which
    counts a node once per target below it.
    """
    ix = vertex_of(g)
    targets = range(g.vcount()) if targets is None else [ix[t] for t in targets]

    out = np.zeros(g.vcount(), dtype=np.int64)
    for t in targets:
        # Every strict ancestor of `t` gains one.
        node = g.successors(int(t))
        while node:
            out[node[0]] += 1
            node = g.successors(node[0])
    return out
