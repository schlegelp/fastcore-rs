# CMTK transforms

[CMTK](https://www.nitrc.org/projects/cmtk) registrations are how *Drosophila*
connectomics bridges between template brain spaces — JFRC2 → FCWB and friends. A
registration is a 12-DOF affine, usually followed by a cubic B-spline warp on a
control-point lattice.

[`CmtkRegistration`](#navis_fastcore.CmtkRegistration) reads one and applies it to points.
**CMTK itself does not need to be installed**: unlike `nat` and `navis`, this does not shell
out to the `streamxform` binary.

```python
import navis_fastcore as fastcore

reg = fastcore.CmtkRegistration("JFRC2_FCWB.list")

xf = reg.xform(points)          # (N, 3) -> (N, 3)
back = reg.xform_inv(xf)        # and back again
```

The path may be a `*.list` directory or a `registration` file itself, plain or gzipped.

## Chaining, and direction

Pass a list to compose registrations, applied left to right.

**Direction is chosen per call, not per object.** The registration you load holds only the
*parse*, so one instance serves every direction — you never pay to read a file twice just to
walk it the other way.

```python
chain = fastcore.CmtkRegistration(["A_B.list", "B_C.list"])   # A -> B -> C

chain.xform(points)                            # forwards
chain.xform_inv(points)                        # the whole composition, backwards
chain.xform(points, invert=[False, True])      # hop 0 forwards, hop 1 backwards
```

Those last two are **not** the same knob, and the difference bites on a chain. `xform_inv`
inverts the whole composition — it reverses the order *and* flips every hop. `invert` flips
hops in place, keeping the order. For a single registration they agree; for a chain they do
not, and only `invert` can express a **mixed-direction** traversal — which is exactly what a
bridging graph hands you, since an edge may be stored in either direction:

```python
# A -> B -> C, where the second registration happens to be stored as C->B
chain = fastcore.CmtkRegistration(["A_B.list", "C_B.list"])
chain.xform(points, invert=[False, True])
```

## Failed points are `NaN`, and that is deliberate

A registration is only defined over a finite **domain box** — the volume the template brain
occupies. CMTK reports any point outside it as `FAILED`; we return `NaN`.

This applies in both directions:

- **Forward**: points outside the domain box are `NaN`. Pass `allow_extrapolation=True` to
  get a value anyway, by clamping to the outermost control points.
- **Inverse**: some points have no preimage *inside* the domain. Pass
  `clamp_to_domain=False` to find preimages outside it.

Both escape hatches make you **disagree with CMTK**, so use them knowingly. The default in
each case is to return `NaN`, because returning a plausible-looking number from a warp that
was never fitted at that location is worse than returning nothing — it would silently
diverge from every other CMTK-based tool.

```python
faithful = reg.xform(points)                              # NaN outside the domain
loose    = reg.xform(points, allow_extrapolation=True)    # extrapolates instead

faithful = reg.xform_inv(points)                          # NaN where CMTK says FAILED
loose    = reg.xform_inv(points, clamp_to_domain=False)   # finds out-of-domain preimages
```

`fallback_to_affine=True` is a middle road: points the warp cannot place fall back to the
affine component rather than becoming `NaN`. It works in **both** directions, and on either
method — a hop travelled backwards (`invert=True`, or `xform_inv`) falls back to the
*inverse* affine, so the rescued point still lands in the space you asked for.

```python
reg.xform(points, fallback_to_affine=True)                     # out-of-domain -> affine
reg.xform(points, invert=True, fallback_to_affine=True)        # ...-> inverse affine
reg.xform_inv(points, fallback_to_affine=True)                 # ...same
```

### On a chain, *what* falls back matters

Say a point clears hop 1 but its image lands outside hop 2's domain. There are two defensible
things to do, and they are **not** close together — on a two-hop JFRC2→FCWB chain they differ
by a median of 6.4 and up to 18 world units:

| | what it does |
|---|---|
| `fallback_to_affine=True` (= `"chain"`) | Re-runs the **whole chain** affine-only, from the *original* point. The good hop-1 warp is discarded along with the hop-2 failure. |
| `fallback_to_affine="hop"` | Keeps the hop-1 warp and swaps the affine in for **only** the hop that failed. |

`"chain"` is the default because it is what `nat` and `navis` do: they hand the failed rows
straight back to `streamxform --affine-only` over the same registration list, and (verified
against the binary) `--affine-only` composes the affine of *every* registration in that list.

`"hop"` is arguably the better answer — throwing away a perfectly good hop-1 warp because hop
2 ran out of domain is crude. But it is a **silent** departure from every other CMTK-based
tool, so you have to name it. On a single registration the two are identical.

## Accuracy

Validated against the **`streamxform` binary itself** (CMTK 3.3.1) on 5000 points spanning
the domain and well beyond it. All four paths — affine and warp, forward and inverse — agree
to **5e-7**, which is `streamxform`'s own print precision, and the set of points it reports
as `FAILED` is reproduced *exactly*.

## Speed

Measured against `streamxform` on the same machine (14 cores), 100k points:

| | `streamxform` | fastcore (1 core) | fastcore (all cores) |
|---|---|---|---|
| Affine | 8.07 µs/pt | 0.006 µs/pt (**1300×**) | 0.005 µs/pt (**1600×**) |
| Warp, forward | 8.28 µs/pt | 0.061 µs/pt (**135×**) | 0.013 µs/pt (**644×**) |
| Warp, inverse | 50.8 µs/pt | 1.58 µs/pt (**32×**) | 0.157 µs/pt (**323×**) |

`streamxform` is a separate process, so it also pays ~15 ms of startup per call — which
dominates for small point sets. Transforming 10 points costs it 15.6 ms and us 0.03 ms.

## API

::: navis_fastcore.CmtkRegistration

::: navis_fastcore.load_cmtk_registration
