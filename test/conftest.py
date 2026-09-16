"""Fuzz example budget: pyproject default, CLI, or env.

Priority (highest wins)::

1. ``--fuzz-examples=N``
2. ``ANYTENSOR_FUZZ_EXAMPLES=N``
3. ``[tool.pytest.ini_options] fuzz_examples`` in ``pyproject.toml`` (default 1000)

Examples::

    uv run pytest -m fuzz
    uv run pytest -m fuzz --fuzz-examples=10000
    ANYTENSOR_FUZZ_EXAMPLES=20000 uv run pytest -m fuzz
"""

from __future__ import annotations

import os

from hypothesis import HealthCheck, settings

_FUZZ_HEALTH = (HealthCheck.too_slow, HealthCheck.filter_too_much)


def pytest_addoption(parser):
    parser.addini(
        "fuzz_examples",
        default="1000",
        help="Hypothesis max_examples for @pytest.mark.fuzz tests",
    )
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


def pytest_report_header(config):
    n = getattr(config, "_anytensor_fuzz_examples", None)
    if n is not None:
        return [f"anytensor fuzz_examples: {n}"]
    return []
