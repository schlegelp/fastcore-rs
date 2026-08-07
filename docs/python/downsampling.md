# Downsampling

Functions that change how densely a skeleton is sampled, without changing what it
is.

All six work on the skeleton's *linear segments* — the runs between roots, branch
points and leafs — and none of them ever moves or drops the nodes at the ends of
those runs. That is what keeps them topology-preserving: the leaf count, the branch
count and the shape of the tree come out the other side untouched, and only the
sampling *along* each neurite changes. It is also what makes them parallel, since
segments meet only at the nodes nothing here is allowed to touch.

They come in three families:

| | Node count | Node IDs | Coordinates |
|---|---|---|---|
| [`downsample_skeleton`](#navis_fastcore.downsample_skeleton), [`simplify_rdp`](#navis_fastcore.simplify_rdp), [`simplify_vw`](#navis_fastcore.simplify_vw) | falls | a subset of the originals | unchanged |
| [`resample_skeleton`](#navis_fastcore.resample_skeleton) | either way | new ones for the new nodes | interpolated |
| [`smooth_skeleton`](#navis_fastcore.smooth_skeleton), [`smooth_skeleton_gaussian`](#navis_fastcore.smooth_skeleton_gaussian) | unchanged | unchanged | moved |

## Taking your data with you

A skeleton rarely travels alone. Synapses, soma tags and manual annotations all hang
off particular nodes, and renumbering the nodes strands them — so every function here
that changes the node table also says where each input node's data should go, as a
`node_map`:

```python
ids, parents, weights, node_map = fastcore.simplify_rdp(
    node_ids, parent_ids, coords, epsilon=100
)

# node_map is indexed like node_ids and valued in the returned ids, so this is a
# lookup table from old node to new.
lookup = pd.Series(node_map, index=node_ids)
synapses["node_id"] = lookup[synapses["node_id"]].values
```

It is *total*: every input node names exactly one output node — the nearest one along
the neurite, ties going towards the root — so there is no sentinel to mask off. Nodes
that survive map to themselves.

The two smoothers have no `node_map` and need none: they move coordinates and nothing
else, so anything attached to a node is still attached to it afterwards. The one thing
that does go stale is a *copy* of a node's position taken beforehand.

## Dropping nodes

The three thinning methods share an output contract with
[`simplify_skeleton`](topology.md#navis_fastcore.simplify_skeleton) — surviving IDs,
their new parents, the edge weights that replace the dropped chains, and the
`node_map` — so they are interchangeable at the call site. Because the replacement
edges carry the *summed* length of the chains they stand in for, total cable length
and geodesic distances survive exactly, even where the geometry has been cut across.

All three take a `preserve` list of node IDs that must survive whatever the rule
decides — nodes carrying synapses, say. They differ in what they spend the node budget
on:

- [`downsample_skeleton`](#navis_fastcore.downsample_skeleton) counts. Every Nth
  node of every segment, geometry ignored. Cheapest, and the right answer when the
  skeleton is already evenly sampled. This is `navis.downsample_neuron`.
- [`simplify_rdp`](#navis_fastcore.simplify_rdp) asks how far the path would *move*.
  Straight stretches collapse to their two ends, tight curves keep every node they
  need. One tolerance, in the units of your coordinates.
- [`simplify_vw`](#navis_fastcore.simplify_vw) asks how much *area* each node
  contributes, and removes the smallest first. Under aggressive simplification RDP
  will keep one spike and flatten everything around it; Visvalingam-Whyatt sheds
  detail evenly and so keeps a neurite looking like itself.

```python
import navis_fastcore as fastcore

# Same skeleton, three ways to make it a fifth of the size.
ids, parents, weights, node_map = fastcore.downsample_skeleton(
    node_ids, parent_ids, 5, weights=weights
)
ids, parents, weights, node_map = fastcore.simplify_rdp(
    node_ids, parent_ids, coords, epsilon=100, weights=weights
)
ids, parents, weights, node_map = fastcore.simplify_vw(
    node_ids, parent_ids, coords, min_area=1e4, weights=weights
)
```

!!! warning "RDP is quadratic in the worst case"

    Its worst case is a segment on which it keeps almost everything: each split
    then peels off one node and re-scans a span one shorter. An `epsilon` well
    below the tracing jitter on a very long unbranched neurite is how to hit it —
    but an RDP that keeps every node is not buying anything anyway, so the fix is a
    larger `epsilon` or `downsample_skeleton`.

::: navis_fastcore.downsample_skeleton

::: navis_fastcore.simplify_rdp

::: navis_fastcore.simplify_vw

## Resampling

[`resample_skeleton`](#navis_fastcore.resample_skeleton) is the inverse problem:
rather than thinning what is there, it re-samples each segment from scratch, so a
skeleton whose node density varies tenfold between neurites comes out evenly sampled
throughout. Anything that averages a quantity *per node* wants this in front of it —
otherwise the average is weighted by how finely each neurite happened to be traced.

Because it creates nodes, it is the one function here that returns a new node table
rather than a subset — and the only one that reports both directions. `source` and
`alpha` name the input edge each output node sits on and how far along it, so any
per-node *column* interpolates in one expression; `node_map` points the other way,
for whatever is *attached* to a node.

```python
ids, parents, xyz, source, alpha, node_map = fastcore.resample_skeleton(
    node_ids, parent_ids, coords, spacing=1000
)

# Per-node columns interpolate onto the new nodes...
new_radius = radius[source[:, 0]] * (1 - alpha) + radius[source[:, 1]] * alpha

# ...and per-node attachments follow node_map to their new home.
synapses["node_id"] = pd.Series(node_map, index=node_ids)[synapses["node_id"]].values
```

You need both because neither derives from the other: an input node that fell between
two output nodes has no output row of its own, so `source`/`alpha` does not invert.

::: navis_fastcore.resample_skeleton

## Smoothing

The two smoothers move coordinates and nothing else — every node keeps its ID and
its parent. Roots, branch points and leafs are pinned, since a branch point that
drifted would drag three neurites apart, so this is safe to run before measuring
angles, tortuosity or tangent vectors, all of which a raw traced skeleton
overstates.

Pick by what you want the amount of smoothing to be tied to:
[`smooth_skeleton`](#navis_fastcore.smooth_skeleton)'s window is a count of nodes,
[`smooth_skeleton_gaussian`](#navis_fastcore.smooth_skeleton_gaussian)'s `sigma` is
a distance. The distance is usually the better choice — it does not change meaning
when the skeleton is resampled.

!!! note "The kernel measures distance along the neurite"

    Not between the points. Node spacing in a traced skeleton varies by an order of
    magnitude, and a kernel over straight-line distance would let the far arm of a
    hairpin pull on the near one.

::: navis_fastcore.smooth_skeleton

::: navis_fastcore.smooth_skeleton_gaussian
