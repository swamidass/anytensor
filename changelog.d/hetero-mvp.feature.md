### Added

- ``anytensor.hetero`` MVP: ``HeteroGraphsTuple``, ``SendRecvTuple``, and
  ``HeteroBatch`` with ``__tree_batch__`` / ``__tree_unbatch__`` hooks into
  :func:`anytensor.tree.batch` / ``unbatch`` (Policy-C ``filled_*`` metadata for
  round-trip when merging unequal schemas).
