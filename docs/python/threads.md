# Threads

By default `navis-fastcore` uses every core it can see. That is the right answer
for one call in one process, and the wrong one when *you* are already spreading
work over processes — because nothing tells a worker process that it is one of
twenty, so every worker claims the whole machine.

```python
# 20 worker processes on a 224-core node, each building a 224-thread pool:
# 4480 threads over 224 cores.
navis.heal_skeleton(nl, parallel=True, n_cores=20)
```

Measured on exactly that node, healing 40 skeletons of 200k nodes each, this was
**slower than running the same work on a single worker** (6.71 s vs 5.10 s) while
burning 2.3x the CPU. Capping each worker to one thread brought it to 3.60 s at a
sixth of the CPU.

## Setting it

[`set_num_threads`](#navis_fastcore.set_num_threads) sizes the pool for the whole
process. Call it once, before any other fastcore function:

```python
import navis_fastcore as fastcore

fastcore.set_num_threads(1)
```

Under `navis`' multiprocessing, set it in each worker as it starts. The hook is
shipped to the workers by pickle, so it has to be picklable — a
`functools.partial` of this function is, a lambda is not:

```python
import functools
import navis
import navis_fastcore as fastcore

navis.compute.worker_init_hooks.append(
    functools.partial(fastcore.set_num_threads, 1)
)
navis.heal_skeleton(nl, parallel=True, n_cores=20)
```

`RAYON_NUM_THREADS` does the same thing without a code change, and is inherited by
spawned workers if you set it in the parent before the pool starts:

```bash
RAYON_NUM_THREADS=1 python my_script.py
```

## What "once per process" means

The pool is built at most once, by whichever of these comes first: an earlier
`set_num_threads`, `RAYON_NUM_THREADS`, a call to
[`get_num_threads`](#navis_fastcore.get_num_threads), or simply the first parallel
fastcore call. After that it cannot be resized: calling `set_num_threads` again
with the *same* value is a no-op — which is what makes it safe in a worker-init
hook that fires more than once — and with a *different* value raises
`RuntimeError`.

Note that `get_num_threads` builds the pool as a side effect, so asking before
setting is what makes the set fail. Set first, ask second.

## Per-call caps

Most functions also take a `threads` (or `n_cores`) argument, which caps that call
alone:

```python
fastcore.heal_skeleton(node_ids, parent_ids, coords, threads=4)
```

This builds a fresh thread pool for the duration of the call, so it costs `n`
thread spawns every time — fine for one big all-by-all NBLAST, wasteful for a
small call in a loop. Prefer `set_num_threads` when the cap is a property of the
process rather than of the call.

## Is this your problem?

`scripts/profile-heal-parallel.py` in the repository measures it, separating
oversubscription from the two things that look identical from the outside:
serialising neurons to the workers, and healing simply not having enough parallel
work in it to fill the cores it claims.

```bash
python scripts/profile-heal-parallel.py --workers 20
```

## API

::: navis_fastcore.set_num_threads

::: navis_fastcore.get_num_threads
