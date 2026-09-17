### Changed

- Hetero neighborhood attention now goes through
  ``segment_attention`` (not a hand-rolled ``segment_softmax`` + multiply +
  ``segment_sum``). HAN semantic attention uses the same helper over stacked
  meta-path embeddings. Compile tests cover ``jax.jit`` / ``tf.function`` on
  the vectorized message path (no edge unroll).
