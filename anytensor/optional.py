"""Peek at optional libraries that are already imported — never import them.

AnyTensor does not import JAX / Torch / TensorFlow unless the caller already
did. Use :func:`loaded` instead of ``try: import …`` when a missing extra
must stay unloaded (broken installs, memory, import order).

A callback can run now if the module is present, or later when it is first
imported in this process — the pattern used by TorchScript divert and by
JAX pytree registration on structured types.
"""

from __future__ import annotations

import sys
import threading
from collections.abc import Callable
from types import ModuleType
from typing import Any, Optional

__all__ = ["loaded"]

_Callback = Callable[[ModuleType], Any]
_pending: dict[str, list[_Callback]] = {}
_lock = threading.Lock()
_finder: _PendingFinder | None = None


def _module(name: str) -> ModuleType | None:
    """``sys.modules`` entry if it is a real loaded module (not a failed-import ``None``)."""
    mod = sys.modules.get(name)
    return mod if mod is not None else None


def _fire_ready() -> None:
    """Invoke callbacks for names that have appeared in ``sys.modules``."""
    while True:
        with _lock:
            ready = [n for n in _pending if _module(n) is not None]
            if not ready:
                return
            batch = [(n, _pending.pop(n)) for n in ready]
        for name, callbacks in batch:
            mod = _module(name)
            if mod is None:
                continue
            for callback in callbacks:
                callback(mod)


class _NotifyLoader:
    """Delegates to a real loader, then drains pending :func:`loaded` callbacks."""

    def __init__(self, loader, name: str):
        self._loader = loader
        self._name = name

    def create_module(self, spec):
        create = getattr(self._loader, "create_module", None)
        if create is None:
            return None
        return create(spec)

    def exec_module(self, module):
        exec_fn = getattr(self._loader, "exec_module", None)
        if exec_fn is not None:
            exec_fn(module)
        else:
            self._loader.load_module(self._name)
        if _pending:
            _fire_ready()

    def __getattr__(self, item):
        return getattr(self._loader, item)


class _PendingFinder:
    """Wrap loaders for watched names only; never imports anything itself."""

    def __init__(self):
        self._busy: set[str] = set()

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in _pending or fullname in self._busy:
            return None
        self._busy.add(fullname)
        try:
            spec = None
            for finder in sys.meta_path:
                if finder is self:
                    continue
                find = getattr(finder, "find_spec", None)
                if find is None:
                    continue
                spec = find(fullname, path, target)
                if spec is not None:
                    break
            if spec is not None and spec.loader is not None:
                if not isinstance(spec.loader, _NotifyLoader):
                    spec.loader = _NotifyLoader(spec.loader, fullname)
            return spec
        finally:
            self._busy.discard(fullname)


def _install_hook() -> None:
    """Observe watched imports via ``sys.meta_path`` (no global ``__import__`` wrap)."""
    global _finder
    if _finder is None:
        _finder = _PendingFinder()
    try:
        idx = sys.meta_path.index(_finder)
    except ValueError:
        sys.meta_path.insert(0, _finder)
        return
    if idx != 0:
        sys.meta_path.remove(_finder)
        sys.meta_path.insert(0, _finder)


def loaded(
    name: str, callback: Optional[_Callback] = None
) -> ModuleType | None:
    """Return ``name`` if it is already imported, else ``None``.

    Never imports ``name``. If ``callback`` is given and the module is
    already loaded, it is invoked immediately with the module. If not,
    ``callback`` is invoked later when that module is imported in this
    process (including when a submodule import loads the parent).

    Args:
        name: Absolute module name (``"torch"``, ``"jax"``, ``"tensorflow"``).
        callback: Optional ``callback(module)`` run now or on a future import.

    Returns:
        The loaded module, or ``None`` if it is not in ``sys.modules``.

    Examples:
        Check without importing::

            torch = loaded("torch")
            if torch is None:
                return

        Register a side effect for now-or-later (JAX pytree, TorchScript)::

            def _register_jax_pytree(jax):
                jax.tree_util.register_pytree_node(Ragged, flatten, unflatten)

            loaded("jax", _register_jax_pytree)
    """
    mod = _module(name)
    if callback is None:
        return mod
    if mod is not None:
        callback(mod)
        return mod
    with _lock:
        _pending.setdefault(name, []).append(callback)
    _install_hook()
    # A concurrent import may have finished while we registered.
    _fire_ready()
    return _module(name)
