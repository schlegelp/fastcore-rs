# Segments

For some operations (e.g. plotting) it is useful to break the graph into its linear components. You can do so by
simply "breaking" the graph at each branch point, effectively producing linear segments that connect leafs, branch points
and roots. Alternatively, you can try to make as few cuts as possible resulting in fewer and longer linear segments.

![Breaking neurons into linear components with as few cuts as possible. The alternative would be to introduce a break at node 2 resulting in 3 separate segments. This is the same toy skeleton used in the code examples below.](../_static/segments.png)

## Generate Segments

::: navis_fastcore.generate_segments
::: navis_fastcore.break_segments

## Segment Coordinates

::: navis_fastcore.segment_coords