### Added

- ``anytensor.hetero`` MVP: ``HeteroGraphsTuple``, ``SendRecvTuple``, with
  ``__tree_batch__`` / ``__tree_unbatch__`` hooks into :func:`anytensor.tree.batch`
  / ``unbatch``. Batching requires identical ntype/etype keys; callers pad
  missing types with empty features themselves.
