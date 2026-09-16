# Lab log

## 2026-09-15

- Added TensorFlow to the local matrix (`uv sync --extra tensorflow`). Ordinary ops via `tf.experimental.numpy` shim (`anytensor.namespace`); segment min/max use scatter so ±inf semantics match. Fuzz peers: numpy × {jax, torch, tensorflow}.
- TF graph coverage: Hypothesis fuzz compares eager vs `tf.function` on FUZZ_OPS (skip `repeat`, `partition_softmax` for now — Python `int(tensor)` / NumPy conversion under trace).
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
- Deferred: GraphsTuple / RaggedTensor; ORT not a v1 backend.
