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

## Nearest neighbours without the matrix

Often the all-by-all is not the answer you want but a step towards one: a UMAP
embedding, or a clustering, needs each neuron's ~20 nearest neighbours, not every
pairwise score. At scale that framing is expensive in a way that stops being an
optimisation problem and becomes a feasibility one — 164k neurons is 2.7e10 pairs
and a 107 GB score matrix, to extract a k-NN graph of a few tens of MB.

`nblast_knn` computes that graph directly, in three stages:

1. **Signature.** Each neuron becomes a sparse, L2-normalised vector of voxel
   occupancy, binned by tangent direction, with each point trilinearly splatted
   over its eight surrounding voxels.
2. **Shortlist.** The `n_candidates` most similar neurons per row by cosine
   similarity of those signatures, via an inverted index over voxels.
3. **Rerank.** The **exact** NBLAST score, using the same kernel as the dense
   paths, for the shortlisted pairs only.

The pre-filter is sound rather than merely plausible, because the scoring matrix
has **finite support**: beyond its last distance bin (40 µm for FCWB, where
`limit_dist="auto"` is 42) every cell sits at about −10. Two neurons that do not
overlap in space therefore score at that floor and cannot be pulled apart by
shape, which makes NBLAST similarity a strictly *local* function of geometry — and
a coarse voxel signature the same information at lower resolution, not an
arbitrary proxy.

Only the shortlist is approximate. The rerank is lossless by construction: a
neuron belonging in the true top-`k` outranks the global k-th, so once it is
shortlisted the exact rerank must rank it in. Every returned score is an exact
NBLAST value; only *which* neurons appear can differ.

Measured on 163,976 zebrafish neurons, `n_candidates=200`: recall@20 = 0.99,
scoring 0.16% of the pairs, in about 5 minutes against an estimated 35 hours for
the exact all-by-all. The shortlist needed to hold a given recall grows only
about logarithmically with the number of neurons, which is what keeps the rerank
`O(n log n)` rather than `O(n²)`.

### Symmetry has to be decided up front

With a full matrix you can symmetrise afterwards — `(M + Mᵀ) / 2`. Once only `k`
neighbours per row are kept the transpose is gone, so the combine has to happen
*before* the top-`k` cut or not at all.

That matters because the asymmetry is real. A small neuron contained in a large
one scores high in one direction (all of it is matched) and low in the other (it
covers a fraction of the big one), so a forward-only k-NN makes such a pair look
like neighbours in one row and strangers in the other. On real data this is not a
rounding effect: forward-only neighbours overlapped the mean-symmetrised ones only
60% of the time. `nblast_knn` therefore defaults to `mean`, and also offers
`min` (a pair must match well *both* ways), `max` and `forward`.

## syNBLAST

A variant that compares neurons by their **synapses** rather than their skeleton
points: for each query connector it finds the nearest target connector, optionally
requiring the same connector type, and scores that distance through the same
lookup matrix with the dot product fixed at `1` — synapses carry no tangent vector.

## Using it

- **Python**: [`nblast`, `nblast_allbyall`, `nblast_knn`, `nblast_smart`, `synblast`](../python/nblast.md)
- **R**: [`nblast`, `nblast_allbyall`, `nblast_knn`, `nblast_pairs`, `synblast`, `synblast_allbyall`](../r/index.md)
- **Rust**: the [`nblast`, `nblast_knn` and `synblast` modules](../rust/index.md)
