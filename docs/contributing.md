# Contributing

PRs welcome. Prefer portable helpers in `anytensor/core.py` or
`anytensor/segment.py`; only extend `backends.py` when the Array API cannot
express the op (e.g. `segment_reduce`).

## Dev setup

```bash
uv sync --extra all --group dev --group docs
```

Hypothesis, pytest, and optional backends (JAX / Torch / TF) are **dev / extras**
only. A plain `pip install .` (NumPy + core deps) is enough to import and run
NumPy ops — see the `minimal-numpy` CI job. Pytest assumes the `dev` group
(including Hypothesis and beartype) is installed.

## Tests

```bash
# Coverage gate: unit/contract only (no fuzz); backends.py omitted; fail_under=100
uv run pytest -m "not fuzz" --cov=anytensor --cov-report=term-missing

# Full suite (fuzz uses fuzz_examples from pyproject, default 1000)
uv run pytest

# Long fuzz — CLI override (preferred) or env
uv run pytest -m fuzz --fuzz-examples=10000 --hypothesis-show-statistics
ANYTENSOR_FUZZ_EXAMPLES=20000 uv run pytest -m fuzz
```

Pytest installs a jaxtyping → beartype import hook for the `anytensor` package
(see `test/conftest.py`), so shape/dtype annotations are checked during the
suite. Disable with `ANYTENSOR_TYPECHECK=0`. Dedicated checks live in
`test/test_typecheck.py`.

CI (`.github/workflows/test.yml`):

- `minimal-numpy` — install the package alone; assert hypothesis / jax / torch /
  tensorflow are absent; smoke NumPy segment ops (runtime deploy surface)
- `test` — `uv sync --extra all --group dev`; coverage gate + bounded fuzz
  (typecheck hook on by default)

Docs deploy to GitHub Pages on pushes to `main` (`.github/workflows/docs.yml`):
`uv sync --group docs` then `mkdocs build` (no Hypothesis / optional backends).
Site: <https://swamidass.github.io/anytensor/>.

`fuzz_examples` lives under `[tool.pytest.ini_options]` in `pyproject.toml`.
Override order: `--fuzz-examples` > `ANYTENSOR_FUZZ_EXAMPLES` > pyproject.

Add coverage in `test/test_ops.py`, boundaries in `test/test_boundaries.py`,
fuzz registrations in `test/test_cross_backend_fuzz.py`, contracts in
`test/test_backend_contracts.py`, symbolic/compiled coverage in
`test/test_symbolic_fuzz.py`.

## Docs

```bash
uv run mkdocs serve    # http://127.0.0.1:8000
uv run mkdocs build    # site/ (gitignored)
```

API pages are generated from docstrings via
[mkdocstrings](https://mkdocstrings.github.io/). Keep public function docstrings
short and accurate; surprising behavior belongs in
[Surprising differences](semantics.md).

## Changelog fragments

User-facing changes need a towncrier fragment in `changelog.d/` (see
[Release](release.md)).
