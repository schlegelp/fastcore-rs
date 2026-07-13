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

## Chaining

Pass a list to compose registrations, applied left to right. `invert` traverses one
backwards — which is what you need when routing through a bridging graph, where an edge may
be walked in either direction.

```python
chain = fastcore.CmtkRegistration(["A_B.list", "B_C.list"])   # A -> B -> C
mixed = fastcore.CmtkRegistration(["A_B.list", "C_B.list"], invert=[False, True])  # A -> B -> C
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

`fallback_to_affine=True` is a middle road for the forward direction: out-of-domain points
fall back to the affine component rather than becoming `NaN`.

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
