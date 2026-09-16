Lock `torch.compile` coverage of the public API (`fullgraph=False` always; `fullgraph=True` except `partition_softmax`). Fix `mean` so Dynamo does not compare Torch `Tensor.size` (a method) to `0`.
