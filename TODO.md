# TODO

- Graduate `ragged_poc/` → `anytensor.ragged` once API settles (structure peel already in `anytensor.structure`)
- Ragged follow-ups: denser NumPy `__array_function__` allowlist; document JAX `jnp.*` gap; to_dense/from_dense
- jraph-style GraphsTuple API (after segment primitives settle)
- Configure PyPI Trusted Publishing (GitHub env `pypi` + PyPI pending publisher) then cut first `v*` release
- Torch `sorted=` fast path (currently no-op; documented as unsorted-safe)
- Keep non-fuzz coverage at 100% (`pytest -m "not fuzz" --cov=anytensor`); `backends.py` / `torchscript.py` omitted from the gate
- Optional `@pytest.mark.gpu` smoke (skip unless CUDA); not required in CI
- Expand CI fuzz budget beyond `--fuzz-examples=200` once runtime is acceptable
- If `min-backends` CI fails: bisect `ci/min-versions.txt` upward and bump `pyproject.toml` floors
