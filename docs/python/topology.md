# Topology

Functions that walk or rewire the tree itself, rather than measuring distances
along it.

These exist because the operations look like general graph algorithms but are not:
on a rooted forest, "everything below this node", "re-root here" or "collapse these
nodes" are all linear passes over the parent vector. Building a graph object to
answer them costs more than the answer does.

## Traversal

[`descendants`](#navis_fastcore.descendants) and
[`paths_to_root`](#navis_fastcore.paths_to_root) are the two directions of the same
walk: everything below a node, and everything above it.

```python
import navis_fastcore as fastcore
import numpy as np

node_ids = np.array([0, 1, 2, 3, 4])
parent_ids = np.array([-1, 0, 1, 2, 1])

# Cutting the skeleton at node 2 = splitting it into this and everything else
distal = fastcore.descendants(node_ids, parent_ids, [2])[0]
```

::: navis_fastcore.descendants

::: navis_fastcore.paths_to_root

## Editing

[`reroot`](#navis_fastcore.reroot) reverses the edges between a new root and the
old one and leaves the rest of the tree alone.
[`contract_nodes`](#navis_fastcore.contract_nodes) merges groups of nodes onto a
representative. [`simplify_skeleton`](#navis_fastcore.simplify_skeleton) throws
away the slab nodes that carry no topological information, keeping total cable
length intact.

!!! note

    `contract_nodes` does not re-root; chain it with `reroot` if you need the
    result rooted somewhere specific.

::: navis_fastcore.reroot

::: navis_fastcore.contract_nodes

::: navis_fastcore.simplify_skeleton

## Adjacency

[`adjacency`](#navis_fastcore.adjacency) hands back the three arrays of a CSR
matrix, so a skeleton can be fed to `scipy.sparse` (or anything else) without going
through an edge list.

```python
from scipy.sparse import csr_matrix

n = len(node_ids)
indptr, indices, data = fastcore.adjacency(node_ids, parent_ids)
# N.B. scipy takes the three arrays in the opposite order
A = csr_matrix((data, indices, indptr), shape=(n, n))
```

::: navis_fastcore.adjacency

## Longest paths

[`longest_path`](#navis_fastcore.longest_path) finds the longest path from a node
to its root; [`longest_paths`](#navis_fastcore.longest_paths) repeats that,
removing each path before looking for the next, which is how a neuron gets split
into its *n* longest neurites.

This is not the (NP-hard) general longest-path problem. In a rooted forest every
maximal path is fixed by its start node — just follow the parents up — so the
longest one starts at whichever node is farthest from its own root.

!!! warning "`min_length` measures the catchment, not the path"

    Every edge whose *parent* lies on the path counts towards `min_length`, so
    each twig hanging off the path contributes its first edge too. The comparison
    is `<=` and hitting it stops the search rather than skipping one path. This
    is inherited from navis so that results do not shift.

::: navis_fastcore.longest_path

::: navis_fastcore.longest_paths

## Validity

Everything on this page assumes the parent vector describes a rooted forest:
follow parents from any node and you arrive at a root.
[`has_cycles`](#navis_fastcore.has_cycles) is that assumption, checked in a
linear pass — worth doing once on data of unknown provenance, since a cycle is
malformed input rather than an unusual shape. The functions here are written so
that a cycle cannot hang them, but on one they return a truncated walk, not an
answer.

```python
if fastcore.has_cycles(node_ids, parent_ids):
    raise ValueError("not a skeleton")
```

::: navis_fastcore.has_cycles
