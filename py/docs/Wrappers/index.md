# Scipy CSGraph Wrappers

If you are working with graphs (representing neurons or similar objects) you might
already be using routines implemented in `scipy.csgraph` - for example, to compute
distances or extract connected components.

Assuming your graphs are directed acyclic graphs (DAGs), you can use `fastcore` as
drop-in replacement for some of these scipy functions:

```python
>>> from scipy.sparse import csr_array
>>> from navis_fastcore.wrappers.csgraph import dijkstra
>>> graph = [
... [0, 1, 0, 0],
... [0, 0, 0, 1],
... [0, 0, 0, 3],
... [0, 0, 0, 0]
... ]
>>> graph = csr_array(graph)
>>> dist_matrix = dijkstra(csgraph=graph, directed=False)
>>> dist_matrix
array([[0., 1., 5., 2.],
       [1., 0., 4., 1.],
       [5., 4., 0., 3.],
       [2., 1., 3., 0.]], dtype=float32)
```

`fastcore` currently implements the following `scipy.csgraph` functions:

- `dijkstra`
- `connected_components`

## Notes
1. Not all arguments are supported. For example, `dijkstra` currently does not
   support `return_predecessors=True`. See the docstrings for details!
2. By default, these functions will perform a check to make sure the input graph
   is actually directed and acyclic. These are generally fairly fast but if you
   are confident in your graphs and want to save the odd millisecond, you can
   set `checks=False`.
