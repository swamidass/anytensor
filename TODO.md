# TODO

- jraph-style GraphsTuple API (after segment primitives settle)
- RaggedTensor / richer partition helpers
- Make `repeat` / `partition_softmax` graph-safe under `tf.function` (drop from skip list)
- CI matrix for numpy/jax/torch/tf version floors (incl. int32 vs int64 index paths)
- Publish to PyPI when the public API settles
- Torch `sorted=` fast path (currently no-op; documented as unsorted-safe)
- Optional TF extra in default CI once wheel story is settled
- Keep non-fuzz coverage at 100% (`pytest -m "not fuzz" --cov=anytensor`); backends.py omitted from the gate
