# Tree utilities

Nested-structure helpers with the public API of
[`jax.tree`](https://docs.jax.dev/en/latest/pytrees.html) (`flatten`, `map`,
`unflatten`, …), implemented in pure Python. ``None`` is an empty pytree.
Custom objects opt in with `__tree_flatten__` / `__tree_unflatten__`,
`__tree_concat__` / `__tree_split__`, or an already-imported JAX / PyTorch /
optree pytree registry.

::: anytensor.tree
    options:
      members_order: source
      filters:
        - "!^_"
