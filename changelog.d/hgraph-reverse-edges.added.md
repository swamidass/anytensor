### Added

- ``add_reverse_edges`` / ``reverse_canonical_etype`` to materialize reverse
  relations as stored etypes (``relation_view(..., reverse=True)`` remains a
  zero-copy alias only). Docs clarify the portable-hetero niche vs DGL/PyG
  and ``anytensor.jraph``.
