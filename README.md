[![docs](https://github.com/schlegelp/fastcore-rs/actions/workflows/docs.yaml/badge.svg)](https://schlegelp.github.io/fastcore-rs/)

# fastcore-rs
Rust re-implementation of `navis` and `nat` core functions.

## TO-DOs
- [x] geodesic distances
- [x] connected components
- [x] generation of linear segments
- [x] synapse flow centrality
- [x] Strahler index
- [x] classify nodes
- [ ] flow centrality
- [ ] CI tests
- [ ] NBLAST (started a prototype)
- [ ] shortest paths
- [ ] cater for `i32` node IDs which are currently cast up to `i64`
- [ ] faster version of `navis.connected_subgraph`

### Additional Notes
- internally, we use `i32` to represent node indices which means we can't
  process neurons with more than 2,147,483,647 nodes (should be fine though)

## Usage

See the README for the [navis](./py) and [nat](./R/nat.fastcore/) wrappers for instructions on installation and usage.
