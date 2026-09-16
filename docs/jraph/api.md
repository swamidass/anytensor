# Jraph API

Portable [jraph](https://github.com/google-deepmind/jraph): GraphsTuple,
batching/padding, and GraphNetwork models on NumPy / JAX / PyTorch / TF.

Segment ops on this module require ``num_segments`` (AnyTensor contract).
``None`` node/edge/global features are empty pytrees (jraph / ``jax.tree``).

Narrative docs: [Overview](index.md). Recipes: [Examples](examples.md).

::: anytensor.jraph
    options:
      members_order: source
      filters:
        - "!^_"
