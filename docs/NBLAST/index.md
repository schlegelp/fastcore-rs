# NBLAST

There is an experimental implementation but this is very much still
work in progress. In particular, we still need to:

1. Properly test which Rust library to use for nearest-neighbor look-ups
2. Match features in the pure Python implementation in `navis`

At this point, there is only a single function for a simple all-by-all
NBLAST:

```python
import fastcore

import numpy as np

from collections import namedtuple

# fastcore expects dotprops to be named tuple
Dotprop = namedtuple("Dotprop", ["points", "vect"])

# Generate 10 random dotprops
dps = [
    Dotprop(
        np.random.rand(100, 3),
        np.random.rand(100, 3)
        )
        for _ in range(10)
        ]

scores = fastcore.nblast_allbyall(dps)
```


