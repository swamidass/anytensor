"""Context-scoped namespaced cache (a dict of dicts).

The partition→segment-id map lives at ``cache["partition"]``. Other
namespaces can be added the same way if later helpers need the same
opt-in reuse. Not a process-wide ``id()`` cache: tensors are unhashable,
in-place edits would stale entries, and tracers wrap a new object every
compile.
"""

from __future__ import annotations

import weakref
from contextvars import ContextVar
from functools import wraps


class _WeakMap(dict):
    """Weakref-able map (builtin ``dict`` cannot take a weakref)."""


class _CacheRoot(_WeakMap):
    """Root dict-of-dicts: namespace name → per-kind map."""

    def __init__(self):
        super().__init__()
        self.depth = 0
        self.sticky = False
        self.token = None

    def namespace(self, name: str) -> _WeakMap:
        ns = dict.get(self, name)
        if ns is None:
            ns = _WeakMap()
            dict.__setitem__(self, name, ns)
        return ns


_CACHE: ContextVar[_CacheRoot | None] = ContextVar("anytensor_cache", default=None)


def _activate(*, sticky: bool = False) -> _CacheRoot:
    root = _CACHE.get()
    if root is not None:
        if sticky:
            root.sticky = True
        return root
    root = _CacheRoot()
    root.sticky = sticky
    root.namespace("partition")
    root.token = _CACHE.set(root)
    return root


def _drop(root: _CacheRoot) -> None:
    for ns in root.values():
        ns.clear()
    root.clear()
    root.sticky = False
    root.depth = 0
    token = root.token
    root.token = None
    _CACHE.reset(token)


def _namespace(name: str):
    """Return the map for ``name``, or ``None`` if the cache is off."""
    root = _CACHE.get()
    if root is None:
        return None
    return root.namespace(name)


def _cache_lookup(ns, obj, extra=()):
    """One cached value per ``obj`` (and optional ``extra`` key parts).

    Partition totals are ``shape(ids)[0]`` of the cached ids. GraphConvolution
    passes ``(add_self_edges, symmetric_normalization)`` so stacked layers
    with different flags do not share structure.
    """
    key = (id(obj),) + tuple(extra)
    hit = ns.get(key)
    if hit is not None:
        held_ref, value = hit
        if held_ref() is obj:
            return key, value
        ns.pop(key, None)
    return key, None


class _StrongRef:
    """Pin ``obj`` when it cannot take a :class:`weakref.ref`."""

    __slots__ = ("_obj",)

    def __init__(self, obj):
        self._obj = obj

    def __call__(self):
        return self._obj


def _purge_cache_entry(ns_ref, key):
    held = ns_ref()
    if held is None:
        return
    held.pop(key, None)


def _cache_store(ns, key, obj, value):
    def _drop_entry(_ref, ns_ref=weakref.ref(ns), key=key):
        _purge_cache_entry(ns_ref, key)

    held_ref = _ref_obj(obj, _drop_entry)
    ns[key] = (held_ref, value)


def _ref_obj(obj, callback):
    try:
        return weakref.ref(obj, callback)
    except TypeError:
        return _StrongRef(obj)


class _Cache:
    """Opt-in namespaced cache: a dict of dicts.

    Graph helpers often expand the same ``n_node`` / ``n_edge`` vector more
    than once in a call. A process-wide ``id()`` cache is wrong (unhashable
    tensors, in-place edits, tracers). This object is opt-in:

    * **Decorator** (preferred on library apply functions; **sticky** so
      repeated calls reuse the map)::

          @cache
          def apply(graph):
              ...

    * **Context** (reentrant; drops on exit unless already sticky)::

          with cache():
              ...

    * :meth:`lookup` / :meth:`store` for derived structure (same pair
      :func:`~anytensor.partition_ids` and GraphConvolution use).
    * :meth:`enable` / :meth:`disable` for a ContextVar-scoped cache outside
      a block (``disable`` clears and turns it off).
    * :meth:`purge` drops cached values for one object in one namespace.

    While the cache is on it behaves as a dict of dicts. The
    partition→segment-id map lives at ``"partition"``; other namespaces can
    be added the same way (``cache["other"][key] = value``). Subscript the
    cache while it is on: ``cache["partition"]``.

    Entries in tensor-keyed namespaces are weak: when the tensor is
    collected, the cached value drops. A context that owns the root also
    clears on exit so it does not pin. GC callbacks hold only a weakref to
    the **namespace** map, so a long-lived tensor cannot keep the root alive
    after the block and cannot form a callback→map→entry cycle that would
    pin cached values. :func:`~anytensor.partition_ids` is the only
    partition helper that consults ``"partition"``; other partition
    functions call ``partition_ids`` so a cache hit is shared. One entry
    per partition tensor: the ids' leading size *is* the flattened total
    (no separate ``sum(partitions)`` cache; on export that length is a
    ``dim_param``). If a cached expansion's length does not match
    ``total_length`` (host Python ints), that entry is purged, a warning
    is issued, and ids are recomputed; tracing skips the check.
    Callers do not thread ids through the stack.

    **Pattern** (library apply and user apply are the same)::

        @cache
        def apply(graph):
            ids = partition_ids(graph.n_node, shape(graph.nodes)[0])
            extra = (flag,)
            packed = cache.lookup("structure", graph.senders, extra)
            if packed is None:
                packed = build(graph)
                cache.store("structure", graph.senders, packed, extra)
            return ...

    :func:`~anytensor.partition_ids` is :meth:`lookup` / :meth:`store` on
    ``"partition"``. GraphConvolution uses the same pair on ``"gcn"``.
    GraphNetwork, GraphConvolution, GAT, GraphMapFeatures, and hetero
    ``multi_update_all`` are ``@cache`` so stacked applies share the map.

    Tensor-keyed namespaces use ``key[0] == id(obj)`` so :meth:`purge` can
    drop every entry for one object.
    """

    def __call__(self, fn=None):
        if fn is None:
            return self

        @wraps(fn)
        def wrapped(*args, **kwargs):
            self.enable()
            return fn(*args, **kwargs)

        return wrapped

    def __enter__(self):
        root = _activate()
        root.depth += 1
        return self

    def __exit__(self, *exc):
        root = _CACHE.get()
        if root is None:
            return False
        root.depth = max(0, root.depth - 1)
        if root.depth == 0 and not root.sticky:
            _drop(root)
        return False

    def enable(self):
        """Turn the cache on until :meth:`disable` (no surrounding block)."""
        _activate(sticky=True)

    def disable(self):
        """Clear every namespace and turn the cache off."""
        root = _CACHE.get()
        if root is None:
            return
        _drop(root)

    def lookup(self, namespace, obj, extra=()):
        """Return the cached value for ``obj`` (+ ``extra``), or ``None``.

        No-op miss when the cache is off. The key is
        ``(id(obj),) + tuple(extra)``. This is the same helper
        :func:`~anytensor.partition_ids` and GraphConvolution use.
        """
        ns = _namespace(namespace)
        if ns is None:
            return None
        _, value = _cache_lookup(ns, obj, extra)
        return value

    def store(self, namespace, obj, value, extra=()):
        """Cache ``value`` for ``obj`` (+ ``extra``). No-op when the cache is off."""
        ns = _namespace(namespace)
        if ns is None:
            return
        key, _ = _cache_lookup(ns, obj, extra)
        _cache_store(ns, key, obj, value)

    def purge(self, namespace, obj):
        """Drop cached entries for ``obj`` in ``namespace`` (no-op if off)."""
        root = _CACHE.get()
        if root is None:
            return
        ns = root.get(namespace)
        if not ns:
            return
        oid = id(obj)
        for key in [k for k in ns if isinstance(k, tuple) and k[:1] == (oid,)]:
            ns.pop(key, None)

    purge_cache = purge

    def __getitem__(self, name):
        root = _CACHE.get()
        if root is None:
            raise KeyError(name)
        return root.namespace(name)

    def __contains__(self, name):
        root = _CACHE.get()
        return root is not None and name in root


cache = _Cache()
