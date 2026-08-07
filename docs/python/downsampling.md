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

## Dropping nodes

The three thinning methods share an output contract with
[`simplify_skeleton`](topology.md#navis_fastcore.simplify_skeleton) — surviving IDs,
their new parents, and the edge weights that replace the dropped chains — so they
are interchangeable at the call site. Because the replacement edges carry the
*summed* length of the chains they stand in for, total cable length and geodesic
distances survive exactly, even where the geometry has been cut across.

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
ids, parents, weights = fastcore.downsample_skeleton(
    node_ids, parent_ids, 5, weights=weights
)
ids, parents, weights = fastcore.simplify_rdp(
    node_ids, parent_ids, coords, epsilon=100, weights=weights
)
ids, parents, weights = fastcore.simplify_vw(
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
rather than a subset. `source` and `alpha` are how the rest of your data follows:
they name the input edge each output node sits on and how far along it, so any
per-node column interpolates in one expression.

```python
ids, parents, xyz, source, alpha = fastcore.resample_skeleton(
    node_ids, parent_ids, coords, spacing=1000
)

# ...and everything else you track per node comes along the same way
new_radius = radius[source[:, 0]] * (1 - alpha) + radius[source[:, 1]] * alpha
```

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
