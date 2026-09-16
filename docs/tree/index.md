# Tree utilities

`anytensor.tree` follows the
[JAX pytree](https://docs.jax.dev/en/latest/pytrees.html) API
(`flatten` → `(leaves, treedef)`, `map`, `unflatten`, …) in
pure Python. JAX is **not** a runtime dependency.

`None` is an **empty pytree** (zero leaves), matching jraph — not a leaf.
NumPy arrays and framework tensors are leaves. Dicts flatten by **sorted
keys**; `OrderedDict` keeps insertion order. `str` / `bytes` / sets / mapping
views are leaves.

Runnable recipes: [Examples](examples.md). Generated API: [API](api.md).
Jraph uses this module for nested node/edge/global features — see
[Jraph](../jraph/index.md).

!!! note "Stability"

    Public functions (`map`, `flatten`, `unflatten`, `concat`, `split`, and
    the `tree_*` aliases) and the built-in walking rules are **stable**.

    **Custom-type registration is beta.** `__tree_flatten__` /
    `__tree_unflatten__` and consulting already-imported `jax.tree_util`,
    `torch.utils._pytree`, or `optree` registries may change. Do not depend on
    undocumented registry details. `__tree_concat__` / `__tree_split__` are
    part of the stable concat/split API.

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

## Concat and split

`tree.concat` / `tree.split` join or partition along an axis. All-`None` stays
`None`. Mixing `None` with arrays is a structure error (same as JAX).

If the object defines `__tree_concat__(xs, axis=0)` /
`__tree_split__(sizes, axis=0)`, those win **before** walking children — so a
feature container or `GraphsTuple` can own join/partition. Jraph `batch` /
`unbatch` go through this path.

## Custom types (beta)

Until registration stabilizes, prefer:

1. **Stable concat hooks** — `__tree_concat__` / `__tree_split__` when the
   object must own join/partition.
2. **Built-in containers** — dicts, lists, tuples, namedtuples.
3. **Beta flatten hooks** — `__tree_flatten__` / `__tree_unflatten__` (JAX
   child/aux convention), or a type already registered with JAX / Torch /
   optree. Those modules are consulted **only if already imported**; this
   library never imports them as a side effect, and does not ship a local
   registry.
