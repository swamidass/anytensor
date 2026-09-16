"""``loaded()`` never imports extras; callbacks run now or on a later import."""

from __future__ import annotations

import importlib
import sys
import types

import pytest

import anytensor as at
from anytensor import optional as opt
from anytensor.optional import _install_hook, loaded


@pytest.fixture(autouse=True)
def _isolate_pending():
    snapshot = {k: list(v) for k, v in opt._pending.items()}
    yield
    with opt._lock:
        opt._pending.clear()
        opt._pending.update(snapshot)


def test_loaded_is_public():
    import json

    assert at.loaded is loaded
    assert at.loaded("json") is json


def test_loaded_returns_existing_module():
    import json

    assert loaded("json") is json


def test_loaded_returns_none_without_importing():
    name = "anytensor_definitely_missing_module_xyz"
    assert name not in sys.modules
    assert loaded(name) is None
    assert name not in sys.modules


def test_loaded_treats_failed_import_none_as_absent():
    name = "anytensor_failed_import_marker_xyz"
    sys.modules[name] = None
    try:
        assert loaded(name) is None
    finally:
        del sys.modules[name]


def test_callback_runs_immediately_when_already_loaded():
    import json

    seen = []
    assert loaded("json", seen.append) is json
    assert seen == [json]


def test_callback_runs_on_future_import(tmp_path, monkeypatch):
    name = "anytensor_loaded_future_cb"
    sys.modules.pop(name, None)
    (tmp_path / f"{name}.py").write_text("VALUE = 7\n")
    monkeypatch.syspath_prepend(str(tmp_path))

    seen = []
    assert loaded(name, seen.append) is None
    assert seen == []

    mod = importlib.import_module(name)
    assert seen == [mod]
    assert loaded(name) is mod
    assert mod.VALUE == 7
    sys.modules.pop(name, None)


def test_callback_survives_failed_import():
    name = "anytensor_loaded_missing_never"
    seen = []
    loaded(name, seen.append)
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(name)
    assert seen == []
    assert loaded(name) is None


def test_callback_on_module_stuffed_then_any_import():
    name = "anytensor_loaded_stuffed"
    sys.modules.pop(name, None)
    seen = []
    loaded(name, seen.append)
    sys.modules[name] = types.ModuleType(name)
    import json  # noqa: F401  — any import drains pending names

    assert seen == [sys.modules[name]]
    sys.modules.pop(name, None)


def test_fire_ready_drains_chained_pending():
    a, b = "anytensor_loaded_chain_a", "anytensor_loaded_chain_b"
    sys.modules.pop(a, None)
    sys.modules.pop(b, None)
    seen = []

    def on_a(_mod):
        seen.append("a")
        sys.modules[b] = types.ModuleType(b)

    loaded(a, on_a)
    loaded(b, lambda _m: seen.append("b"))
    sys.modules[a] = types.ModuleType(a)
    import json  # noqa: F401

    assert seen == ["a", "b"]
    sys.modules.pop(a, None)
    sys.modules.pop(b, None)


def test_parent_import_from_submodule_fires_parent_callback(tmp_path, monkeypatch):
    pkg = "anytensor_loaded_pkg"
    sys.modules.pop(pkg, None)
    sys.modules.pop(f"{pkg}.child", None)
    root = tmp_path / pkg
    root.mkdir()
    (root / "__init__.py").write_text("PARENT = 1\n")
    (root / "child.py").write_text("CHILD = 2\n")
    monkeypatch.syspath_prepend(str(tmp_path))

    seen = []
    loaded(pkg, seen.append)
    child = importlib.import_module(f"{pkg}.child")
    assert seen and seen[0].PARENT == 1
    assert child.CHILD == 2
    sys.modules.pop(f"{pkg}.child", None)
    sys.modules.pop(pkg, None)


def test_race_loaded_between_check_and_fire(monkeypatch):
    """If the module appears after the first peek, trailing ``_fire_ready`` runs."""
    import json

    target = "json"
    seen = []
    n = {"checks": 0}
    real = opt._module

    def fake(name):
        if name == target:
            n["checks"] += 1
            if n["checks"] == 1:
                return None
        return real(name)

    monkeypatch.setattr(opt, "_module", fake)
    assert loaded(target, seen.append) is json
    assert seen == [json]


def test_fire_ready_skips_if_unloaded_after_snapshot(monkeypatch):
    name = "anytensor_loaded_vanish"
    sys.modules.pop(name, None)
    fake = types.ModuleType(name)
    seen = []
    real = opt._module
    phases = {name: iter((fake, None))}

    def gated(n):
        if n == name:
            return next(phases[name])
        return real(n)

    monkeypatch.setattr(opt, "_module", gated)
    with opt._lock:
        opt._pending.setdefault(name, []).append(seen.append)
    opt._fire_ready()
    assert seen == []


def test_install_hook_is_idempotent():
    _install_hook()
    assert opt._hook_installed is True
    _install_hook()
    assert opt._hook_installed is True


def test_import_wrappers_noop_when_nothing_pending():
    _install_hook()
    with opt._lock:
        saved = {k: list(v) for k, v in opt._pending.items()}
        opt._pending.clear()
    try:
        import json as json_mod
        assert importlib.import_module("json") is json_mod
    finally:
        with opt._lock:
            opt._pending.update(saved)


def test_callback_on_import_statement(tmp_path, monkeypatch):
    name = "anytensor_loaded_stmt"
    sys.modules.pop(name, None)
    (tmp_path / f"{name}.py").write_text("VALUE = 3\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    seen = []
    loaded(name, seen.append)
    mod = __import__(name)
    assert seen == [mod]
    assert mod.VALUE == 3
    sys.modules.pop(name, None)


def test_namespace_tf_check_does_not_need_tensorflow():
    from anytensor.namespace import _is_tensorflow_tensor
    from anytensor import namespace

    real = namespace.loaded

    def fake(name, callback=None):
        if name == "tensorflow":
            return None
        return real(name, callback)

    namespace.loaded = fake
    try:
        assert _is_tensorflow_tensor(object()) is False
    finally:
        namespace.loaded = real


def test_namespace_tf_check_uses_already_loaded_module(monkeypatch):
    from anytensor import namespace

    tensor_cls = type("TfTensor", (), {})
    variable_cls = type("TfVariable", (), {})
    fake = types.SimpleNamespace(Tensor=tensor_cls, Variable=variable_cls)
    monkeypatch.setattr(
        namespace, "loaded", lambda name, callback=None: fake if name == "tensorflow" else None
    )
    assert namespace._is_tensorflow_tensor(tensor_cls()) is True
    assert namespace._is_tensorflow_tensor(variable_cls()) is True
    assert namespace._is_tensorflow_tensor(object()) is False
