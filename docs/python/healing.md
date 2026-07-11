# Healing fragmented skeletons

Reconstructed skeletons are often broken into several disconnected fragments.
Healing means finding the shortest set of new edges that stitches those fragments
back into a single rooted tree.

[`heal_skeleton`](#navis_fastcore.heal_skeleton) does the whole job: it finds the
minimal-length bridges between connected components and regenerates the parent
vector. [`stitch_fragments`](#navis_fastcore.stitch_fragments) is the lower-level
half, returning just the bridging edges if you want to inspect or filter them
before rewiring.

```python
import navis_fastcore as fastcore

new_parent_ids = fastcore.heal_skeleton(node_ids, parent_ids, coords)
```

Two options are worth knowing about:

- `max_dist` caps how long a single new edge may be. Gaps wider than that are left
  alone, so the result can still be fragmented — which is usually what you want, as
  bridging a huge gap is more likely to be wrong than right.
- `use_radius` takes node radii into account when measuring distances, which
  prefers connecting fragments of similar calibre over merely nearby ones. Pass a
  float rather than `True` to weight how much influence radius gets. Note that
  `max_dist` is then measured in that augmented space too.

## API

::: navis_fastcore.heal_skeleton

::: navis_fastcore.stitch_fragments
