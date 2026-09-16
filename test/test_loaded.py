"""``loaded()`` never imports extras; callbacks run now or on a later import."""

from __future__ import annotations

import importlib
import importlib.machinery
import sys
import types

import pytest

import anytensor as at
from anytensor import optional as opt
from anytensor.optional import _NotifyLoader, _install_hook, loaded


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


def test_callback_on_module_stuffed_then_fire_ready():
    name = "anytensor_loaded_stuffed"
    sys.modules.pop(name, None)
    seen = []
    loaded(name, seen.append)
    sys.modules[name] = types.ModuleType(name)
    opt._fire_ready()
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
    opt._fire_ready()
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


def test_install_hook_is_idempotent_and_reinserts():
    _install_hook()
    assert opt._finder is not None
    assert sys.meta_path[0] is opt._finder
    _install_hook()
    assert sys.meta_path[0] is opt._finder

    sentinel = object()
    sys.meta_path.insert(0, sentinel)
    try:
        _install_hook()
        assert sys.meta_path[0] is opt._finder
    finally:
        if sentinel in sys.meta_path:
            sys.meta_path.remove(sentinel)

    sys.meta_path.remove(opt._finder)
    _install_hook()
    assert sys.meta_path[0] is opt._finder


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


def test_finder_skips_when_busy_or_unwatched():
    _install_hook()
    finder = opt._finder
    assert finder.find_spec("json", None) is None
    name = "anytensor_loaded_busy"
    with opt._lock:
        opt._pending.setdefault(name, []).append(lambda _m: None)
    finder._busy.add(name)
    try:
        assert finder.find_spec(name, None) is None
    finally:
        finder._busy.discard(name)


def test_finder_wraps_loader_once_and_skips_finders_without_find_spec():
    name = "anytensor_loaded_wrap_once"
    seen = []
    loaded(name, seen.append)

    class NoSpec:
        pass

    class DummyLoader:
        def create_module(self, spec):
            return None

        def exec_module(self, module):
            module.ok = True

    loader = DummyLoader()
    spec = importlib.machinery.ModuleSpec(name, loader)

    class DummyFinder:
        def find_spec(self, fullname, path=None, target=None):
            if fullname == name:
                return spec
            return None

    dummy = DummyFinder()
    nospec = NoSpec()
    sys.meta_path.insert(0, dummy)
    sys.meta_path.insert(0, nospec)
    # Finder must sit in front so it wraps DummyFinder after skipping NoSpec.
    _install_hook()
    try:
        wrapped = opt._finder.find_spec(name, None)
        assert wrapped is spec
        assert isinstance(spec.loader, _NotifyLoader)
        again = opt._finder.find_spec(name, None)
        assert again is spec
        assert again.loader is spec.loader

        module = types.ModuleType(name)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        assert module.ok is True
        assert seen == [module]
        loader.marker = 1
        assert spec.loader.marker == 1
    finally:
        sys.meta_path.remove(dummy)
        sys.meta_path.remove(nospec)
        sys.modules.pop(name, None)


def test_notify_loader_create_module_and_legacy_load_module():
    name = "anytensor_loaded_legacy"

    class LegacyLoader:
        def load_module(self, n):
            mod = types.ModuleType(n)
            sys.modules[n] = mod
            mod.legacy = True
            return mod

    wrap_legacy = _NotifyLoader(LegacyLoader(), name)
    assert wrap_legacy.create_module(None) is None
    with opt._lock:
        opt._pending.clear()
    wrap_legacy.exec_module(types.ModuleType(name))
    assert sys.modules[name].legacy is True
    sys.modules.pop(name, None)

    created = types.ModuleType(name)

    class Creates:
        def create_module(self, spec):
            return created

        def exec_module(self, module):
            module.created = True

    wrap_create = _NotifyLoader(Creates(), name)
    assert wrap_create.create_module(None) is created
    wrap_create.exec_module(created)
    assert created.created is True


def test_finder_namespace_spec_without_loader():
    name = "anytensor_loaded_noloader"
    loaded(name, lambda _m: None)
    spec = importlib.machinery.ModuleSpec(name, None, is_package=True)

    class NSFinder:
        def find_spec(self, fullname, path=None, target=None):
            if fullname == name:
                return spec
            return None

    ns = NSFinder()
    sys.meta_path.insert(0, ns)
    _install_hook()
    try:
        found = opt._finder.find_spec(name, None)
        assert found is spec
        assert found.loader is None
    finally:
        sys.meta_path.remove(ns)


def test_finder_returns_none_when_no_spec():
    name = "anytensor_loaded_nospec_anywhere"
    loaded(name, lambda _m: None)
    assert opt._finder.find_spec(name, None) is None


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


def test_numpy_type_checks_skip_when_not_loaded(monkeypatch):
    import numpy as np
    from anytensor import core, namespace, semantics
    from anytensor.semantics import empty_segment_identity

    monkeypatch.setattr(namespace, "loaded", lambda name, callback=None: None)
    assert namespace._is_numpy_ndarray(np.array([1.0])) is False
    assert namespace._is_scalar(1.5) is True
    assert namespace._is_scalar(np.int64(1)) is False

    monkeypatch.setattr(namespace, "loaded", at.loaded)
    assert namespace._is_scalar(np.int64(1)) is True
    assert namespace._is_numpy_ndarray(np.array([1.0])) is True

    monkeypatch.setattr(core, "loaded", lambda name, callback=None: None)
    assert core._normalize_shape_dim(4) == 4

    monkeypatch.setattr(semantics, "loaded", lambda name, callback=None: None)

    class KindF:
        kind = "f"

    assert empty_segment_identity(KindF(), "min", xp=np) == np.inf


def test_tf_namespace_init_and_isdtype_require_loaded_modules(monkeypatch):
    from anytensor import namespace

    monkeypatch.setattr(namespace, "loaded", lambda name, callback=None: None)
    with pytest.raises(RuntimeError, match="tensorflow is not imported"):
        namespace._TensorflowNumpyNamespace()

    ns = namespace._TensorflowNumpyNamespace.__new__(namespace._TensorflowNumpyNamespace)
    with pytest.raises(RuntimeError, match="numpy is not imported"):
        ns.isdtype(object(), "bool")
