# Tree utilities

`anytensor.tree` follows the
[JAX pytree](https://docs.jax.dev/en/latest/pytrees.html) API
(`flatten` → `(leaves, treedef)`, `map`, `unflatten`, …). The
implementation is **pure Python**; the only binary dependency is **NumPy**.
No JAX, no C++ pytree extension.

`None` is an **empty pytree** (zero leaves), matching jraph — not a leaf.
NumPy arrays and framework tensors are leaves. Dicts flatten by **sorted
keys**; `OrderedDict` keeps insertion order. `str` / `bytes` / sets / mapping
views are leaves.

Runnable recipes: [Examples](examples.md). Generated API: [API](api.md).
Graphs use this for nested node/edge/global features
([Jraph](../jraph/index.md)), but the same helpers apply to any nested
numeric record.

!!! note "Stability"

    Public functions (`map`, `flatten`, `unflatten`, `batch`, `unbatch`, and
    the `tree_*` aliases) and the built-in walking rules are **stable**.

    **Custom-type registration is beta.** `__tree_flatten__` /
    `__tree_unflatten__` and consulting already-imported `jax.tree_util`,
    `torch.utils._pytree`, or `optree` registries may change. Do not depend on
    undocumented registry details. `__tree_batch__` / `__tree_unbatch__` are
    part of the stable batch/unbatch API.

## Why tree

Nested dicts and tuples of arrays show up everywhere: a simulation step
(`{"pos": …, "vel": …}`), a lab run (`{"temp": …, "ph": …, "notes": None}`),
a minibatch of observations, checkpoint blobs, and — yes — GNN node/edge
features. The useful operations are the same: apply `f` to every array,
flatten to a list of leaves, stack two records, split a stack back into
rows. Writing those walks by hand is where `None` vs missing keys vs list
vs tuple quietly diverges.

That problem already has good libraries:

| Library | What it is good at |
|---|---|
| [`jax.tree`](https://docs.jax.dev/en/latest/pytrees.html) / [`jax.tree_util`](https://docs.jax.dev/en/latest/jax.tree_util.html) | The API this module follows. `None` is empty. Built for `jit` / `vmap` over nested parameters. Needs JAX (and optree). |
| [`dm-tree`](https://github.com/google-deepmind/tree) | `map_structure` / `flatten` for TensorFlow and JAX-era nests. C++ extension. Treats `None` as a **leaf** (wrong for jraph). |
| [`optree`](https://github.com/metaopt/optree) | Fast C++ pytrees; JAX uses it under `jax.tree`. |
| [`torch.utils._pytree`](https://pytorch.org/docs/stable/pytree.html) | Nested tensors for `torch.compile` / `export`. Ships with PyTorch. |

The **key value of `anytensor.tree`** is that same `jax.tree` contract as
**pure Python + NumPy**. No JAX install, no dm-tree / optree shared library.
Leaves may still be JAX / Torch / TF arrays when those are present;
`tree.map(fn, nest)` does not care. One implementation for a structured
record — GNN features, a physics state, or a table of experimental traces.

`batch` / `unbatch` are the extra that those libs do not standardize: stack
nests along an axis, and let an object own join/partition
(`__tree_batch__` / `__tree_unbatch__`) when fieldwise concat would be
wrong. These are the **same functions** as `jraph.batch` / `jraph.unbatch`.
GraphsTuple uses magic for sender offsets; a packed buffer or a ragged
container can do the same.

Use upstream `jax.tree` when you are JAX-only and do not need batch/unbatch.
Use this module when you want the same API with only NumPy as a binary
dep, when the helper must run on Torch / TF too, or when `None` must mean
“no arrays here” like jraph.

## Walking rules

| Kind | Treatment |
|---|---|
| `None` | Empty pytree (jraph / `jax.tree`) |
| `list` / `tuple` / namedtuple | Node (tuple vs list must match) |
| `dict` | Node, keys sorted |
| `OrderedDict` | Node, insertion order |
| arrays / tensors | Leaf |
| `str` / `bytes` / sets / mapping views | Leaf |

`flatten(tree)` returns `(leaves, treedef)`. `unflatten(treedef, leaves)`
rebuilds. `jax.tree_util` aliases (`tree_map`, `tree_flatten`, …) are provided
for jraph-style call sites.

## Batch and unbatch

`tree.batch` / `tree.unbatch` join or partition along an axis. They are the
same functions as `jraph.batch` / `jraph.unbatch` (sequence in; unbatch
returns unit slices, or whatever `__tree_unbatch__` yields). All-`None`
stays `None`. Mixing `None` with arrays is a structure error (same as JAX).

If the object defines `__tree_batch__(xs, axis=0)` /
`__tree_unbatch__(axis=0)`, those win **before** walking children — so a
feature container or `GraphsTuple` can own join/partition. GraphsTuple
offsets senders/receivers; it does not take sizes because `n_node` /
`n_edge` already know the grouping.

## Custom types (beta)

Until registration stabilizes, prefer:

1. **Stable batch hooks** — `__tree_batch__` / `__tree_unbatch__` when the
   object must own join/partition.
2. **Built-in containers** — dicts, lists, tuples, namedtuples.
3. **Beta flatten hooks** — `__tree_flatten__` / `__tree_unflatten__` (JAX
   child/aux convention), or a type already registered with JAX / Torch /
   optree. Those modules are consulted **only if already imported**; this
   library never imports them as a side effect, and does not ship a local
   registry.
