"""Sybil collector for fenced Python examples in ``docs/*.md``.

Kept separate from Hypothesis / typecheck setup so doc failures are obvious.
Registered from ``conftest.py`` (pytest only loads ``pytest_collect_file`` from
conftest / plugins, not from ordinary test modules).
"""

from __future__ import annotations

from pathlib import Path

from sybil import Sybil
from sybil.parsers.markdown import PythonCodeBlockParser, SkipParser

_DOCS = Path(__file__).resolve().parents[1] / "docs"


def _setup(namespace: dict) -> None:
    import numpy as np
    import pytest

    import anytensor as at

    namespace.update(np=np, pytest=pytest, at=at)


docs_sybil = Sybil(
    parsers=[SkipParser(), PythonCodeBlockParser()],
    path=str(_DOCS),
    filenames=["examples.md"],
    setup=_setup,
    name="docs",
)
