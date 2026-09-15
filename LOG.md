# Lab log

## 2026-09-15

- Hybrid architecture: ordinary ops via `array-api-compat`; segment ops keep einops-style input-adaptive backends (NumPy / JAX / Torch / TF).
- Scalar policy: reductions return 0-d arrays; `@as_array_result` / `@promote_scalars` centralize wrapping (NumPy `np.generic` is not usable as a stable array type for methods / `type(x) is type(y)`).
- Backend patches: Torch `from_numpy` no longer sets `requires_grad`; dtype-safe min/max fills + int64 segment ids; JAX `jax.Array` detection; TF sorted `segment_*` arity (no `num_segments`); version floors at backend init.
- Deferred: GraphsTuple / RaggedTensor; ORT not a v1 backend.
