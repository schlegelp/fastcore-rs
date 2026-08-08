# Packing

Arranging shapes on a page without letting any of them touch — what a collage of neurons
needs, and the kind of thing that is easy to write in numpy and slow to run.

Three primitives, meant to be used together:

```python
masks = fastcore.rasterize_segments(coords, edges, scale=40, pad=1)
positions, variant, grid = fastcore.pack_masks(
    masks, page_shape=(1300, 980)
)
```

`rasterize_segments` turns line work — a skeleton's node-to-parent edges, a mesh's face
edges — into binary masks. `pack_masks` lays those out so that no two share a pixel.
`pack_rectangles` is the cheaper bounding-box alternative, useful on its own or as a
starting point for the mask packing.

## Packing shapes, not boxes

A bounding box is mostly empty for anything branching, so packing boxes wastes most of the
page. Packing the shapes themselves lets one reach into another's empty space — even sit
inside a loop of it — as long as no cable actually meets:

```python
ring = np.zeros((9, 9), dtype=bool)
ring[0, :] = ring[-1, :] = ring[:, 0] = ring[:, -1] = True
small = np.ones((3, 3), dtype=bool)

fastcore.pack_masks([[ring], [small]], (9, 9))[0]     # -> [[0, 0], [1, 1]]
fastcore.pack_rectangles([[9, 9], [3, 3]], (9, 9))[0] # -> None, no room
```

## Why this is not a cross correlation

The obvious way to test every position of a mask against the page at once is a
correlation: `fftconvolve(page, mask[::-1, ::-1], mode="valid")` gives the number of
shared pixels at each of the ~1.3M positions of a 1300x980 page, and the free ones are the
zeros. One vectorised call — which is why it is what you write in numpy.

It is also the wrong computation. It produces an exact overlap *count* at every position,
in floating point, when the question is boolean and the answer is wanted at exactly one
position: the best one. Measured on 200 synthetic arbors at 100 px per page unit, on a
1300x980 page:

| step | numpy/scipy | fastcore |
|---|---|---|
| rasterise 200 arbors | 69 ms | 3.9 ms |
| ...with `fill=True` | 89 ms | 4.2 ms |
| pack, bottom-up, 1 variant | 1.46 s | 56 ms |
| pack, bottom-up, 2 variants | 4.18 s | 64 ms |
| pack, under a cost surface | 4.40 s | 234 ms |
| pack 200 bounding boxes | 19 ms | 3.4 ms |

The correlation is about 70% of the layout, and it grows as `O(N² res²)` — doubling the
resolution quadruples it. What replaces it:

- **Positions are visited in the order you score them.** Both cost models — bottom-up, and
  a cost surface — define a total order on positions that does not depend on the shape, so
  the best free position is the *first* free one in that order. The scan stops there. A
  correlation cannot stop early; that is the whole difference.
- **Collisions are bit operations.** A page row is `u64` words, a mask row is `u64` words,
  and the test is an `AND` — 64 pixels per instruction, no allocation, and a page small
  enough to stay in cache (160 kB packed against 1.3 MB as bytes).
- **The densest rows are tried first.** A position that collides usually collides on the
  shape's heaviest row, so most rejections cost one or two words.
- **Provably empty space is skipped.** Everything above the highest occupied row is free,
  so a bottom-up scan is bounded by the fill line rather than by the page.

Rasterising is the same story on a smaller scale. Interpolating every edge and scattering
the result materialises one element per *pixel-step of every edge*: on a mesh with 300k
edges averaging five pixels each that is a 1.5M-element index array, plus the same again
for the parameter and both coordinates, to set a few tens of thousands of distinct pixels.
Here the walk writes straight into the mask and allocates nothing.

## Cost surfaces and masks

By default shapes go as far down, and then as far left, as they will go, which fills a
rectangle neatly. Pass `cost` to say what a good position is instead — each shape lands
where the surface is lowest under its centre:

```python
rows, cols = np.ogrid[0:height, 0:width]
cost = np.hypot(rows - height / 2, cols - width / 2)   # fill from the middle outwards
```

To confine shapes to a silhouette, mark everything outside it as already taken and hand
that in as `grid`. Combining the two — a `grid` that blocks the outside and a `cost` that
grows from the middle of the shape — is what fills an arbitrary outline, since bottom-up
would pile everything into the bottom of it.

`grid` also serves the two-pass case: pack one set of shapes, then hand the page back to
squeeze a second set into whatever room is left.

## Reference

::: navis_fastcore.rasterize_segments

::: navis_fastcore.pack_masks

::: navis_fastcore.pack_rectangles
