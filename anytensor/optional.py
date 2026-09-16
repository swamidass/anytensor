"""Peek at optional libraries that are already imported — never import them.

AnyTensor does not import JAX / Torch / TensorFlow unless the caller already
did. Use :func:`loaded` instead of ``try: import …`` when a missing extra
must stay unloaded (broken installs, memory, import order).

A callback can run now if the module is present, or later when it is first
imported in this process — the pattern used by TorchScript divert and by
JAX pytree registration on structured types.
"""

from __future__ import annotations

import builtins
import importlib
import sys
import threading
from collections.abc import Callable
from types import ModuleType
from typing import Any, Optional

__all__ = ["loaded"]

_Callback = Callable[[ModuleType], Any]
_pending: dict[str, list[_Callback]] = {}
_lock = threading.Lock()
_hook_installed = False


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


def _install_hook() -> None:
    """Drain pending callbacks after imports complete.

    CPython's frozen importlib keeps a private ``_find_and_load``; wrapping
    ``builtins.__import__`` and :func:`importlib.import_module` observes both
    ``import`` statements and ``import_module`` without importing extras
    ourselves.
    """
    global _hook_installed
    if _hook_installed:
        return
    orig_import = builtins.__import__
    orig_import_module = importlib.import_module

    def _import(name, globals=None, locals=None, fromlist=(), level=0):
        try:
            return orig_import(name, globals, locals, fromlist, level)
        finally:
            if _pending:
                _fire_ready()

    def _import_module(name, package=None):
        try:
            return orig_import_module(name, package)
        finally:
            if _pending:
                _fire_ready()

    builtins.__import__ = _import
    importlib.import_module = _import_module
    _hook_installed = True


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
