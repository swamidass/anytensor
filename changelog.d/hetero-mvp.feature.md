### Added

- ``anytensor.split`` / backend ``split`` (NumPy cut-index semantics; Torch via
  ``tensor_split``), plus ``anytensor.tree.split`` / ``tree.partition`` /
  ``tree.match_sizes`` for nest-aware leading-axis splits.
- ``anytensor.hetero`` MVP with ``HeteroGraphsTuple`` / ``SendRecvTuple`` and
  ``__tree_batch__`` / ``__tree_unbatch__``. Batching requires matching keys;
  unbatch uses ``tree.partition`` (also used by ``GraphsTuple`` unbatch).
