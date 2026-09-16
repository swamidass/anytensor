# Tree API

Nested-structure helpers with the public API of
[`jax.tree`](https://docs.jax.dev/en/latest/pytrees.html). JAX is not a
runtime dependency. ``None`` is an empty pytree.

Public functions and built-in walking rules are **stable**. Custom-type
registration (``__tree_flatten__`` / ``__tree_unflatten__`` and JAX / Torch /
optree registries) is **beta** — see [Overview](index.md#custom-types-beta).
Why this exists (and how it relates to ``jax.tree`` / dm-tree / optree):
[Why tree](index.md#why-tree).

::: anytensor.tree
    options:
      members_order: source
      filters:
        - "!^_"
