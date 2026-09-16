# TODO

- jraph-style GraphsTuple API (after segment primitives settle)
- RaggedTensor / richer partition helpers
- Publish to PyPI when the public API settles
- Torch `sorted=` fast path (currently no-op; documented as unsorted-safe)
- Keep non-fuzz coverage at 100% (`pytest -m "not fuzz" --cov=anytensor`); backends.py omitted from the gate
- Optional `@pytest.mark.gpu` smoke (skip unless CUDA); not required in CI
- Expand CI fuzz budget beyond `--fuzz-examples=200` once runtime is acceptable
- Enable `mkdocs build --strict` once griffe is happy with remaining public APIs
