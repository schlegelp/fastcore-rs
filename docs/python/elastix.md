# Elastix transforms

[Elastix](https://github.com/SuperElastix/elastix) registrations bridge the *Drosophila* VNC
and whole-CNS template spaces — FANC → JRCVNC2018F, BANC → JRC2018F, and friends. A
`TransformParameters` file holds a linear transform, a cubic B-spline warp, or a chain of both.

[`ElastixTransform`](#navis_fastcore.ElastixTransform) reads one and applies it to points.
**Elastix itself does not need to be installed**: unlike `navis`, this does not shell out to
the `transformix` binary — so there is no subprocess, no temporary directory, no
`LD_LIBRARY_PATH` to set, and no `copy_files` to marshal.

```python
import navis_fastcore as fastcore

xf = fastcore.ElastixTransform("TransformParameters.FixedFANC.txt")

out  = xf.xform(points)          # (N, 3) -> (N, 3)
back = xf.xform_inv(out)         # ...and back, which Elastix cannot do at all
```

A `TransformParameters` file is **already a chain**: its `InitialTransformParametersFileName`
is followed recursively, resolved relative to *that file's own directory*. BANC's is four deep
(`3_Bspline_fine → 2_Bspline_coarse → 1_affine → 0_manual_affine`) and loads from the outermost
file alone.

Elastix records that name **as it was on the machine that ran the registration** — routinely an
absolute path like `/home/someone/scratch/TransformParameters.0.txt`, which of course is not
there when you receive the files. `transformix` simply fails. If the recorded path does not
resolve, we look for its **basename in the naming file's own directory** instead, which is what
`navis`'s `copy_files` achieves by copying everything into one place. A path that *does* resolve
still wins, so this can only rescue a lookup that would otherwise have failed.

Supported transform types: `AffineTransform`, `TranslationTransform`, `EulerTransform`,
`SimilarityTransform`, and `BSplineTransform` / `RecursiveBSplineTransform`. Both
`HowToCombineTransforms` modes (`Compose` and `Add`) work going forwards.

## Chaining, and direction

Pass a list to compose transforms, applied left to right.

**Direction is chosen per call, not per object.** The transform you load holds only the
*parse*, so one instance serves every direction. That is worth caring about here: BANC's
`BANC_to_template.txt` is **56 MB**, and re-reading it just to walk it backwards would be
absurd.

```python
chain = fastcore.ElastixTransform(["A_to_B.txt", "B_to_C.txt"])   # A -> B -> C

chain.xform(points)                            # forwards
chain.xform_inv(points)                        # the whole composition, backwards
chain.xform(points, invert=[False, True])      # hop 0 forwards, hop 1 backwards
```

Those last two are **not** the same knob, and the difference bites on a chain. `xform_inv`
inverts the whole composition — it reverses the order *and* flips every hop. `invert` flips
hops in place, keeping the order. For a single transform they agree; for a chain they do not,
and only `invert` can express a **mixed-direction** traversal — which is exactly what a
bridging graph hands you, since an edge may be stored in either direction:

```python
# A -> B -> C, where the second transform happens to be stored as C->B
chain = fastcore.ElastixTransform(["A_to_B.txt", "C_to_B.txt"])
chain.xform(points, invert=[False, True])
```

## Outside the grid, points come back *unchanged*

A B-spline warp is only defined over its control-point grid. **Outside it, Elastix returns the
point untouched** — the identity, never a failure. That is the exact opposite of CMTK, which
reports such points as `FAILED` (and which [`CmtkRegistration`](cmtk.md) surfaces as `NaN`).

We reproduce Elastix's behaviour by default, because swapping `fastcore` in must not silently
change anyone's existing results. But it is worth knowing what that default costs you: it is
**silent**. A neuron straddling the edge of the registered volume comes back *partly*
transformed and looks perfectly fine.

```python
faithful = xf.xform(points)                          # unchanged outside the grid, like transformix
strict   = xf.xform(points, out_of_bounds="nan")     # NaN instead, so you can see the boundary
```

## The inverse

Elastix has no inverse — `transformix` only goes forwards. That is precisely why
`navis-flybrains` ships *two* separate registration files per brain pair and registers four
one-way BANC edges.

`xform_inv` provides one. Linear steps are inverted exactly; each B-spline is solved per point
by damped Gauss-Newton against the analytic Jacobian.

**What is guaranteed** is *forward-consistency*: `xform(xform_inv(y)) == y`.

**What is not guaranteed** is that `xform_inv(xform(p)) == p`. A B-spline warp need not be
injective: because the forward map warps inside the grid and is the identity outside it, it is
discontinuous at the boundary, and a strongly deforming registration also folds. Several points
can map to the same place, and no inverse can recover which one you meant. Pass `initial_guess`
if you know which preimage you want — it breaks the tie. Points with no recoverable preimage at
all come back as `NaN`.

Measured on the six real B-spline registrations `navis-flybrains` ships, over points spanning
each fixed image, `f(xform_inv(y)) == y` holds to ~1e-13 and **four of the six invert with zero
failures**. The exception is BANC's `BANC_to_template.txt`, whose median displacement is 163 µm
— ten grid cells — and which folds: ~0.5% of its points have no recoverable preimage. (Nothing
needs *its* inverse in practice: flybrains ships a dedicated reverse file, which inverts
cleanly.)

## Accuracy

Validated against the **`transformix` binary itself** (Elastix 5.2.0) on 5000 points per file,
spanning each control-point grid and well beyond it, for **all seven** registration files
`navis-flybrains` ships — including both four-deep BANC chains. Agreement is **5e-7**, which is
`transformix`'s own print precision, and the set of points it leaves untouched is reproduced
*exactly*.

## Speed

Measured against `transformix` on the same machine (14 cores), FANC's warp:

| | `transformix` | fastcore (1 core) | fastcore (all cores) |
|---|---|---|---|
| Forward, 100k points | 955 ms | 5.4 ms (**177×**) | 1.2 ms (**796×**) |
| Inverse, 100k points | *not possible* | 1975 ms | 179 ms |

`transformix` is a separate process that writes the points to a temp file, spawns, and parses
text back — so it also pays a fixed ~0.9 s per call regardless of how few points you give it.
Parsing the registration costs us 38 ms **once**; every subsequent transform is free of it.

## Asking whether a file inverts, without parsing it

A transform is invertible unless some step in its chain combines via `Add`. That fact lives in
one short key — but the key sits *after* the coefficient array, which is 56 MB in BANC's warp,
so reading it honestly used to mean parsing the whole file.

`probe_elastix_invertible` walks the same chain and skips only the numbers:

```python
fastcore.probe_elastix_invertible("TransformParameters.FixedFANC.txt")   # -> True
```

| | full parse | probe |
|---|---|---|
| All 13 registrations `navis-flybrains` ships | 648 ms | **31 ms** (21×) |
| BANC's `BANC_to_template.txt` (54 MB) | 409 ms | **19.6 ms** (21×) |

That is the difference between a bridging graph that can afford to label its edges honestly and
one that has to guess per *backend*. It raises rather than returning `True` for anything that
would not load at all — missing, not elastix, unsupported kind, binary parameters, circular
chain — so a `True` is a promise.

## API

::: navis_fastcore.ElastixTransform

::: navis_fastcore.load_elastix_transform

::: navis_fastcore.probe_elastix_invertible
