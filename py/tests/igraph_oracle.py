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
