# Lab log

## 2026-09-18

- ONNX recipes live in the opt-in `anytensor.export` subpackage (not in
  `anytensor.__all__`; `to_onnx_torch` / `to_onnx_tensorflow` /
  `numpy_leaves`). ONNX is the recommended deploy target (ORT is well
  tested); AnyTensor does not run ops on ORT. Weights embed as ONNX
  initializers via
  `as_torch_module(fn, params)` (`nn.Parameter`, best names) or
  `as_tensorflow_fn` (named constants created *inside* the TF trace). Outer
  `tf.constant` / extra args leak as graph inputs; `assert_embedded_weights`
  catches that. Lightning is an `nn.Module`; Keras uses `tf.function` +
  tf2onnx (not `model.export`); Flax rebinds numpy params — jax2tf is a dead
  end (`XlaCallModule`). Torch `segment_reduce` no longer `int()`s
  `num_segments`. Public-op coverage via TF tf2onnx in CI; Torch dynamo ONNX
  skipped on CI.

## 2026-09-16

- Folded `anytensor.jraph` into the 100% coverage gate (None connectivity,
  padding without senders, dynamically_batch flush/split, 1-d zero-out,
  `_flip0` fallbacks, extra jraph pad parity). Minimal-numpy CI smokes
  GraphsTuple batch + GraphNetwork. Fuzz inventory allowlists `tree` /
  `jraph` (modules, not ops).
- Docs: dedicated Tree and Jraph sections (examples moved off usage /
  worked-examples). Tree marks flatten-style registration as beta; public
  tree/jraph API is stable. Jraph overview links upstream and discusses
  why the library’s data model is worth following. Tree overview links
  jax.tree / dm-tree / optree / Torch pytree and positions the module as
  a nest walker for any structured record, not only GNN features. Tree is
  pure Python; NumPy is the only binary dependency. Official-jraph public
  `__all__` is a subset of `anytensor.jraph` (full public-API coverage);
  Hypothesis parity fuzz vs upstream jraph when JAX is installed, including
  GAT (self-edges added, not skipped) and the rest of the model zoo.
  `jraph.batch` / `unbatch` are `tree.batch` / `tree.unbatch`; GraphsTuple
  implements `__tree_batch__` / `__tree_unbatch__` (no sized-split API).
  Magic-method examples use AnyTensor ops so they stay portable.

## 2026-09-15

- Require current array-api-compat (≥1.15); fix coverage test that passed raw ``numpy`` into ``_pad_or_slice_leading`` (needs AAC ``concat`` under numpy 1.24).
- GHA 35053243764: main test matrix green; min-backends still red on AAC 1.6 (no `cumulative_sum` / `concat` / Array-API `clip`) → raise AAC floor (now current 1.15).
- GHA 35052980138: coverage fixed; min-backends hit AAC 1.4 `asarray(copy=False)` NotImplementedError → catch it in `_asarray` + floor AAC→1.6; fuzz SIGSEGV in `torch.compile` symbolic → skip on CI (same as Sybil).
- GHA 35052455676 red: (1) min-backends jaxtyping 0.2.28 `Shaped[ArrayT,…]` → TypeError; floor →0.2.34 (+ beartype 0.18.2, yanked 0.18.0). (2) coverage miss `enable_torchscript` return-False — pytest `filterwarnings` with `torch.jit.TracerWarning` imported torch before anytensor; switched to message-only filters + explicit unit test.
- CI min-backends: raised Torch→2.1 / TF→2.13 after first GHA pass (TF 2.10≠numpy 1.24; export needs ≥2.1). Skip torch.compile Sybil on CI (dynamo/triton SIGSEGV). minimal-numpy uses `.venv/bin/python` to avoid uv syncing hypothesis.
- CI: `min-backends` job on Python 3.10 installs tensor libs at declared floors (`ci/min-versions.txt`); failure means raise floors after bisect. Documented in contributing.
- Docs: expanded `design.md` into a principles/edge-case explainer (hybrid array-api-compat vs segment backends, promote kinds, shape-sizes, empty identities, TorchScript divert, typing policy, intentional non-portables); links from home.
- Docs: Sybil executes fenced examples in `docs/examples.md` (GAT helper + jax.jit / torch.compile / tf.function / TorchScript / trace); root `conftest.py` collects them. Home page links to the compile/script gotchas.
- Docs home: expanded motivation (framework lock-in / segment ops) + case study porting PyG/GAT-style neighbor softmax to a single `neighbor_attention` that runs on NumPy/JAX/Torch/TF; verified numerically across four backends; `mkdocs build --strict` green.
- Typing: public APIs annotated with jaxtyping (`Shaped` / `Integer` / `SegmentValues`…) + `ArrayT` TypeVar (no backend imports for typing). Runtime checks **off by default**; pytest installs `jaxtyping.install_import_hook` on selected submodules (not `torchscript`) in `test/conftest.py` (disable `ANYTENSOR_TYPECHECK=0`). Opt-in for apps: `enable_typecheck()`. Divert wrappers live in `torchscript.py` so `torch.jit.script` still compiles under the typecheck hook. `mkdocs build --strict` passes; enabled in docs CI.
- CI: test job enforces `--cov-fail-under=100` + uploads `coverage.xml`; `release.yml` on `v*` tags (clean version gate, build, GitHub Release, PyPI Trusted Publishing via environment `pypi` — configure when ready).
- Docs: GitHub Pages via Actions (`.github/workflows/docs.yml`); Pages `build_type=workflow` enabled for `swamidass/anytensor`. Dropped test-side hypothesis shim; pytest assumes `dev`. Runtime without hypothesis covered by `minimal-numpy` CI.
- ``enable_torchscript()``: ``is_scripting()`` divert so libraries can call ``at.segment_*`` and end users can ``torch.jit.script`` through them; eager NumPy/JAX/TF/Torch dispatch unchanged. Kernels live in ``anytensor.torchscript``. Fuzz: scripted vs eager Torch for sum/min/max (`test_fuzz_torchscript_matches_eager`).
- Expanded Google-style docstrings on segment ops / `repeat` / `*_like` / reductions / `empty_segment_identity`, with per-function Notes for TF NaN·±inf, required `num_segments`, `sorted=` no-op, jit/`total_repeat_length`, TorchScript (JAX docs as model).
- Fuzz: XLA ``prod`` with ``inf`` × float32-min → eager ``inf`` vs XLA ``nan``; finite samples now ``0`` or ``|x|>=1e-3`` (float32-exact), XLA skips ``fuzz_prod`` with non-finite inputs. Long fuzz `--fuzz-examples=5000`: 13 passed (log `artifacts/fuzz-long.out`).
- Documented “Surprising differences” in `docs/semantics.md` + `semantics.py` (TF scatter/unsorted NaN·±inf, XLA min/max, jax.jit repeat static length, shape-kind `num_segments`, TF `*_like` under polymorphic graph, index/float width, GPU notes).
- GPU notes in docs + `semantics.py` (no GPU CI): int64 Torch ids, device matching, float32 compares, nondeterministic ties, empty CUDA / XLA caveats; optional `@pytest.mark.gpu` smoke parked in TODO.
- Added promote kind `shape` (`_normalize_shape_dim`); `num_segments` is a shape-size (required, JAX-style), not inferred. Wired through segment + `total_repeat_length`.
- Symbolic fuzz green for `jax.jit` / `tf.function` / TF XLA / `torch.compile`: `*_like` uses `shape(x)`; partition fuzz passes static `sum_partitions`; TF scatter min/max OR-in segment NaNs; XLA skips min/max when inputs contain NaN (eager NaN vs XLA ±inf).
- `num_segments` always required (JAX convention); may be Python int, jit symbolic constant, or tensor scalar — never inferred from ids.
- Graph-safe `repeat` / `partition_softmax`: TF shim uses `tf.repeat` / `tf.range`; Python scalar repeats not promoted; `shape()` returns backend symbolic dims under trace; `total_repeat_length` pads/slices when lengths are not concrete.
- Added TensorFlow to the local matrix (`uv sync --extra tensorflow`). Ordinary ops via `tf.experimental.numpy` shim (`anytensor.namespace`); segment min/max use scatter so ±inf semantics match. Fuzz peers: numpy × {jax, torch, tensorflow}.
- Fuzz budget: `fuzz_examples = 1000` in pyproject `[tool.pytest.ini_options]`; override with `--fuzz-examples=N` or `ANYTENSOR_FUZZ_EXAMPLES`.
- Coverage gate: `pytest -m "not fuzz" --cov=anytensor` with `fail_under=100`; `backends.py` omitted. Non-fuzz total at 100%.
- Public specials are thin functions over `get_backend(x)` attrs: `inf(x)` / `ninf(x)` / `nan(x)` / `pi(x)` / `e(x)` / `dtype(name, like=x)` / `finfo` / `iinfo`. Backend objects stay internal; `newaxis` is `None`.
- Cross-backend fuzz registers essentially all public ops/helpers (`@fuzz_op`); inventory test guards gaps. `max_examples=1000` (Hypothesis default is 100).
- NaN utilities: `is_nan` / `is_finite` / `is_inf` (aliases `isnan` / `isfinite` / `isinf`), `fill_nan` (alias `nan_fill`), `fill_nan_mask` → `(filled, mask)` with True=was-NaN, Array API `nan_to_num`, element-wise `equal_nan`. Chose verb-first `fill_nan` over `nan_fill`; kept the latter as alias.
- Hybrid architecture: ordinary ops via `array-api-compat`; segment ops keep einops-style input-adaptive backends (NumPy / JAX / Torch / TF).
- Scalar policy: reductions return 0-d arrays; `@as_array_result` / `@promote_scalars` centralize wrapping (NumPy `np.generic` is not usable as a stable array type for methods / `type(x) is type(y)`).
- Mixed operands: prefer non-NumPy namespace; upcast NumPy by **reference** when possible (`copy=False`), with `fallback="copy"` (warn) or `"error"`. `@promote(x="data", indices="index")` applies `result_type` only to data args so ints widen beside floats without corrupting segment ids.
- Index **width** left backend-local (Torch→int64 at scatter; JAX/TF often int32). Do not force `xp.int64` in portable helpers.
- Standardized empty-segment identities in `anytensor.semantics`: float min/max use ±inf (not finfo); ints use iinfo; sum uses 0. Contract tests pin all backends.
- `promote_options(copy=True)` / decorator `copy=True` when host NumPy buffers mutate.
- Added boundary tests + Hypothesis fuzz; coverage ~74% (backends TF paths / unused AbstractBackend methods dominate miss). `segment_normalize` avoids 0/0 warnings via safe denominator.
- Backend patches: Torch `from_numpy` no longer sets `requires_grad`; dtype-safe min/max fills + int64 segment ids; JAX `jax.Array` detection; TF sorted `segment_*` arity (no `num_segments`); version floors at backend init.
- Jraph: portable GraphsTuple / GraphNetwork (`anytensor.jraph`) on AnyTensor
  segment ops; `anytensor.tree` follows the `jax.tree` API in pure Python with
  `__tree_flatten__` / `__tree_unflatten__` and JAX/Torch/optree registry hooks.
  `tree.batch` / `tree.unbatch` dispatch to `__tree_batch__` / `__tree_unbatch__`
  so GraphsTuple and custom feature objects own batch/unbatch. Dev extra
  installs `jraph` for parity tests. `anytensor.jraph` is in the 100%
  coverage gate (only `backends.py` / `torchscript.py` remain omitted).
