### Added

- ``anytensor.split`` / backend ``split``, plus ``anytensor.lengths`` helpers
  (``lengths_to_ids`` / ``lengths_to_splits`` / ``batch_ids`` / ``unbatch_ids`` /
  ``split_by_lengths``) used by graph batch/unbatch — not part of ``tree``.
  Length/id arithmetic stays on the caller's array backend (no NumPy casts).
- Graph batch: fieldwise concat, then offset send/recv ids from length
  vectors. Unbatch: ``split_by_lengths`` / ``unbatch_ids``, then zip into
  graphs.
- ``anytensor.hetero`` MVP with ``HeteroGraphsTuple`` / ``SendRecvTuple``;
  batching requires matching keys (empty arrays, not ``None``).
