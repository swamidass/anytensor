### Added

- ``anytensor.split`` / backend ``split`` (NumPy cut-index semantics; Torch via
  ``tensor_split``), plus ``anytensor.tree.split``, ``tree.match_sizes``, and
  ``tree.lengths_to_cuts`` / ``cuts_to_lengths`` / ``lengths_to_ids``. Ragged
  unbatch is flatten → map ``split`` by cuts → zip → ``unflatten`` per part.
- ``anytensor.hetero`` MVP with ``HeteroGraphsTuple`` / ``SendRecvTuple`` and
  ``__tree_batch__`` / ``__tree_unbatch__``. Batching requires matching keys;
  ``GraphsTuple`` and hetero share the same length-tree split path.
