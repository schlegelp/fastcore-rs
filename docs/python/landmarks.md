# Landmark transforms

When no image registration exists between two template spaces, what you usually have
instead is a set of **matched landmarks** — points a human (or another algorithm) has
identified as corresponding. Two classical methods turn those into a warp:

- [`TpsTransform`](#navis_fastcore.TpsTransform) — a **thin-plate spline**. Interpolates the
  landmarks exactly and, between them, minimises the integral bending norm: the smoothest
  warp consistent with the data.
- [`MlsTransform`](#navis_fastcore.MlsTransform) — **moving least squares** (the affine
  flavour of [Schaefer et al. 2006](https://dl.acm.org/doi/pdf/10.1145/1179352.1141920)).
  Gives every point its *own* affine, solved on the fly from all landmarks weighted by
  inverse squared distance.

These mirror `navis.transforms.TPStransform` and
`navis.transforms.MovingLeastSquaresTransform`, and agree with them to ~1e-14 relative.

```python
import navis_fastcore as fastcore

tps = fastcore.TpsTransform(landmarks_source, landmarks_target)
xf = tps.xform(points)          # (N, 3) -> (N, 3)

mls = fastcore.MlsTransform(landmarks_source, landmarks_target)
xf = mls.xform(points)
```

Landmarks and points may be `(N, 3)` arrays or `DataFrame`s with x/y/z columns. A single
`(3,)` point is accepted and returns a `(3,)` point.

## Which one?

They give similar but **not** identical results, and which suits a given pair of spaces is
worth checking empirically. As a rule of thumb:

|  | thin-plate spline | moving least squares |
|---|---|---|
| Landmarks | reproduced exactly | reproduced exactly |
| Cost | expensive **fit**, cheap to apply | no fit, ~4× more expensive to apply |
| Far from landmarks | converges to the global affine | converges to the global affine |
| Reusing across many point sets | strongly favours it — fit once | no advantage |

Both fall back to a sensible global affine far outside the landmark hull, exposed as
`.matrix_affine`.

## Direction

Neither method has a closed-form inverse, so "backwards" means *fitting the other way
round*, not inverting. Both spell it `-transform`:

```python
back = (-tps).xform(points)     # refits target -> source; costs another fit
back = (-mls).xform(points)     # free - MLS has no fit to redo
```

For MLS you can also ask at construction: `MlsTransform(src, trg, direction="inverse")`.

## No `batch_size`

The `navis` versions take a `batch_size` because both reference implementations build an
intermediate sized by *points × landmarks*. That is what makes them expensive, and for MLS
it is a hard ceiling: `molesq` allocates `(3, M, N)` arrays, so 3,400 landmarks at the
default batch size wants **~23 GB** — which means
`MovingLeastSquaresTransform` cannot run at the landmark counts real registrations use.

Here that intermediate never exists. Each output row depends only on its own row of the
distance matrix, so the distance and its contribution to the result are fused into one
streaming pass over landmarks. Peak memory is the output array, independent of the landmark
count, and there is nothing to tune.

Transforming 1M points through a 3,390-landmark spline:

| | peak memory |
|---|---|
| `navis` (already batched) | 5.6 GB |
| fastcore | 231 MB |

## Speed

Measured on 14 cores against the reference implementations (`morphops` for TPS, `molesq`
for MLS — these are what `navis` calls), 1M points:

### 500 landmarks

| | reference | fastcore (1 core) | fastcore (all cores) |
|---|---|---|---|
| TPS | 0.56 µs/pt | 0.405 µs/pt (**1.4×**) | 0.039 µs/pt (**14×**) |
| MLS | 7.03 µs/pt | 1.902 µs/pt (**3.7×**) | 0.186 µs/pt (**38×**) |

### 3,390 landmarks (a real `flybrains` mirroring registration)

| | reference | fastcore (1 core) | fastcore (all cores) |
|---|---|---|---|
| TPS | 3.67 µs/pt | 2.793 µs/pt (**1.3×**) | 0.270 µs/pt (**14×**) |
| MLS | out of memory (~23 GB) | — | 1.269 µs/pt |

Worth reading honestly: **TPS's win is almost entirely parallelism.** Single-threaded we are
only ~1.3× ahead, because `cdist` plus a BLAS matmul is already good code. What we add is
scaling across cores and the removal of the memory ceiling.

**MLS is a genuine algorithmic win** — 3.7× before any parallelism — because `molesq`
expresses the reduction as a chain of `einsum`s over materialised per-landmark arrays.

## The one place we are slower: fitting a spline

The TPS *fit* solves an `(M+4)` square system, so it is cubic in the landmark count. `numpy`
sends that to hardware LAPACK (on Apple silicon, the AMX coprocessor); we use a blocked LU
in portable Rust, which cannot match it:

| landmarks | numpy (LAPACK) | fastcore |
|---|---|---|
| 500 | 1 ms | 13 ms |
| 1,793 | 29 ms | 174 ms |
| 3,390 | 141 ms | 553 ms |

This is a **one-off** cost per registration, against an `xform` that is 14× faster and can
be called thousands of times, so it is very rarely the thing to optimise. If it is, fit with
numpy and hand the coefficients over:

```python
tps = fastcore.TpsTransform.from_coefs(source, W, A)
```

`W` and `A` follow the same convention as `morphops.tps_coefs` and
`navis.transforms.TPStransform.W` / `.A`, so a `navis` transform converts directly:

```python
fast = fastcore.TpsTransform.from_coefs(navis_tps.source, navis_tps.W, navis_tps.A)
```

MLS has no fit at all, so none of this applies to it.

## Pickling

Both transforms pickle. `TpsTransform` ships its **coefficients** rather than just the
landmarks, so unpickling in a `multiprocessing` worker does not repeat the fit.

## API

::: navis_fastcore.TpsTransform

::: navis_fastcore.MlsTransform
