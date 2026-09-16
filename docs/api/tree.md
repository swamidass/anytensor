# Tree utilities

Nested-structure helpers with the public API of
[dm-tree](https://tree.readthedocs.io/) (`import tree`), implemented in
pure Python. Custom objects opt in with `__tree_flatten__` /
`__tree_unflatten__`, or via an already-imported JAX / PyTorch / optree
pytree registry.

::: anytensor.tree
    options:
      members_order: source
      filters:
        - "!^_"
