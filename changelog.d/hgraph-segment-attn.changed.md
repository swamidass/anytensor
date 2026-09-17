### Changed

- Hetero neighborhood attention goes through ``segment_attention`` (vectorized
  per etype; no edge unroll, no interleaved multi-relation edge tensor).
- HAN **semantic** attention stays a dense softmax on stacked ``(n, R, d)``
  path embeddings — forcing that through ``segment_attention`` would
  ``repeat`` path ids and scatter, which is a copy-heavy detour for fixed
  schema-sized ``R``.
- Source-only zoo linears use ``RelationSpec.src_apply`` (map on ``N_src``)
  before gather; CompGCN stays after gather (needs edge features). Compile
  tests cover ``jax.jit`` on the mailbox path.
