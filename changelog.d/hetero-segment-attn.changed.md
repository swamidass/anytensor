### Changed

- Hetero neighborhood attention goes through ``segment_attention`` (vectorized
  per etype; no edge unroll, no interleaved multi-relation edge tensor).
- HAN **semantic** attention stays a dense softmax on stacked ``(n, R, d)``
  path embeddings — forcing that through ``segment_attention`` would
  ``repeat`` path ids and scatter, which is a copy-heavy detour for fixed
  schema-sized ``R``. Compile tests cover ``jax.jit`` on the mailbox path.
