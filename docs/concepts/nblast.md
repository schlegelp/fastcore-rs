# NBLAST

NBLAST scores the similarity of two neurons by walking one neuron's points,
finding the nearest point on the other, and looking the resulting
*(distance, dot-product)* pair up in a scoring matrix. The dot product compares
the two points' tangent vectors, so a pair of neurons scores well when they run
close together *and* in the same direction.

`fastcore` implements this as a pure-Rust pipeline: a Delaunay neighbourhood
graph is built per neuron (via [`shull`](https://github.com/schlegelp/shull)),
nearest neighbours are found by graph descent (via
[`aann`](https://github.com/schlegelp/aann)), and the scoring runs in Rust. The
whole all-by-all releases the GIL and parallelises across cores, streaming each
pair's nearest-neighbour result so peak memory stays low. That parallelism is the
main reason to reach for it from Python, where multi-processing would otherwise
dominate the cost.

## Scoring matrices

The lookup table that turns a *(distance, dot-product)* pair into a score is
calibrated against a reference dataset. `fastcore` embeds the two standard FCWB
matrices: a plain one, and an alpha-weighted one used when per-point `alpha`
values (a measure of how locally linear the neuron is at that point) are supplied.
You can substitute your own.

## syNBLAST

A variant that compares neurons by their **synapses** rather than their skeleton
points: for each query connector it finds the nearest target connector, optionally
requiring the same connector type, and scores that distance through the same
lookup matrix with the dot product fixed at `1` — synapses carry no tangent vector.

## Using it

- **Python**: [`nblast`, `nblast_allbyall`, `nblast_smart`, `synblast`](../python/nblast.md)
- **R**: [`nblast`, `nblast_allbyall`, `nblast_pairs`, `synblast`, `synblast_allbyall`](../r/index.md)
- **Rust**: the [`nblast` and `synblast` modules](../rust/index.md)
