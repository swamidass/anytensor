"""Root pytest hooks — Sybil must live here so ``docs/`` collection sees it.

``test/conftest.py`` only applies under ``test/``; markdown examples live in
``docs/``, whose ancestors are the repo root.
"""

from __future__ import annotations

import sys
from pathlib import Path

_TEST = Path(__file__).resolve().parent / "test"
if str(_TEST) not in sys.path:
    sys.path.insert(0, str(_TEST))

from docs_sybil import docs_sybil  # noqa: E402

pytest_collect_file = docs_sybil.pytest()
