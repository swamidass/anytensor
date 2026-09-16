# Jraph (portable)

[jraph](https://github.com/google-deepmind/jraph)-compatible GraphsTuple,
batching/padding, and GraphNetwork models on NumPy / JAX / PyTorch / TF.

Segment ops on this module require ``num_segments`` (AnyTensor contract).
``None`` node/edge/global features are empty (jraph/JAX), not dm-tree leaves.

::: anytensor.jraph
    options:
      members_order: source
      filters:
        - "!^_"
