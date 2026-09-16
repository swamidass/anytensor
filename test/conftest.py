"""Pytest config: fuzz budget + opt-in jaxtyping runtime checks for tests.

Library code keeps jaxtyping annotations but does **not** enable runtime
checking by default. Tests install the import hook here (before test modules
import ``anytensor``) so shape/dtype annotations are verified during the suite.
"""

from __future__ import annotations

import os

# Import Dynamo before TensorFlow. ``import tensorflow`` then ``torch._dynamo``
# SIGSEGVs in triton on this stack (same crash as CI inductor/triton). Test
# modules load TF via ``helpers``; compiling must not be the first Dynamo
# import after that.
try:
    import importlib

    importlib.import_module("torch._dynamo")
except ImportError:
    pass

# Install before Hypothesis / any test module imports ``anytensor``.
# Skip if explicitly disabled (e.g. debugging) or beartype is missing.
# Hook specific submodules — not ``anytensor.torchscript`` — so TorchScript
# can compile divert wrappers (jaxtyped wrappers hide free vars like ``torch``).
_TYPECHECK_MODULES = (
    "anytensor.core",
    "anytensor.segment",
    "anytensor.namespace",
    "anytensor.semantics",
    "anytensor.backends",
    "anytensor.typing",
)
if os.environ.get("ANYTENSOR_TYPECHECK", "1") not in ("0", "false", "False"):
    try:
        from jaxtyping import install_import_hook

        install_import_hook(list(_TYPECHECK_MODULES), "beartype.beartype")
        _TYPECHECK_HOOK = True
    except Exception as exc:  # pragma: no cover - misconfigured env
        _TYPECHECK_HOOK = False
        _TYPECHECK_HOOK_ERROR = exc
else:
    _TYPECHECK_HOOK = False
    _TYPECHECK_HOOK_ERROR = None

from hypothesis import HealthCheck, settings

_FUZZ_HEALTH = (HealthCheck.too_slow, HealthCheck.filter_too_much)


def pytest_addoption(parser):
    # ``fuzz_examples`` ini is registered in root ``conftest.py`` (docs collection).
    group = parser.getgroup("anytensor")
    group.addoption(
        "--fuzz-examples",
        type=int,
        default=None,
        metavar="N",
        help="Override Hypothesis max_examples for fuzz (else ini/env)",
    )


def _resolve_fuzz_examples(config) -> int:
    cli = config.getoption("--fuzz-examples")
    if cli is not None:
        n = int(cli)
    else:
        env = os.environ.get("ANYTENSOR_FUZZ_EXAMPLES")
        if env is not None and env.strip() != "":
            n = int(env)
        else:
            n = int(config.getini("fuzz_examples"))
    if n < 1:
        raise ValueError(f"fuzz examples must be >= 1, got {n}")
    return n


def pytest_configure(config):
    n = _resolve_fuzz_examples(config)
    settings.register_profile(
        "default",
        max_examples=n,
        deadline=None,
        suppress_health_check=_FUZZ_HEALTH,
    )
    # Only load if the user did not explicitly pick another Hypothesis profile.
    if not config.getoption("--hypothesis-profile", default=None):
        settings.load_profile("default")
    config._anytensor_fuzz_examples = n  # type: ignore[attr-defined]
    config._anytensor_typecheck = _TYPECHECK_HOOK  # type: ignore[attr-defined]
    if not _TYPECHECK_HOOK and os.environ.get("ANYTENSOR_TYPECHECK", "1") not in (
        "0",
        "false",
        "False",
    ):
        # Fail loudly in CI/dev if typecheck was expected but beartype/jaxtyping missing.
        err = globals().get("_TYPECHECK_HOOK_ERROR")
        raise RuntimeError(
            "jaxtyping runtime typecheck hook failed to install "
            f"(install beartype / jaxtyping, or set ANYTENSOR_TYPECHECK=0). "
            f"Original error: {err!r}"
        )


def pytest_report_header(config):
    lines = []
    n = getattr(config, "_anytensor_fuzz_examples", None)
    if n is not None:
        lines.append(f"anytensor fuzz_examples: {n}")
    tc = getattr(config, "_anytensor_typecheck", None)
    if tc is not None:
        lines.append(f"anytensor jaxtyping typecheck: {'on' if tc else 'off'}")
    return lines
