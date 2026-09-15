# TODO

- jraph-style GraphsTuple API (after segment primitives settle)
- RaggedTensor / richer partition helpers
- CI matrix for numpy/jax/torch/tf version floors (incl. int32 vs int64 index paths)
- Raise coverage on backends.py (TF install in CI; trim unused AbstractBackend debug APIs)
- Publish to PyPI when the public API settles
- Torch `sorted=` fast path (currently no-op; documented as unsorted-safe)
- Optional TF extra in default CI once wheel story is settled
