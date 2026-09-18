"""ONNX recipes: symbolic lengths, public-op coverage, Flax / Lightning / Keras."""

from __future__ import annotations

import os
from collections.abc import Callable

import numpy as np
import pytest

import anytensor as at
from anytensor import export
from anytensor.export import _bind, _graph, _serialize

pytest.importorskip("onnx")

_SKIP_CI_TORCH = pytest.mark.skipif(
    bool(os.environ.get("CI")),
    reason="torch.onnx dynamo disabled on CI runners (dynamo/triton SIGSEGV)",
)


@pytest.fixture(autouse=True)
def _disable_jaxtyping_for_onnx():
    try:
        from jaxtyping import config
    except ImportError:  # pragma: no cover
        yield
        return
    prev = config.jaxtyping_disable
    config.update("jaxtyping_disable", True)
    try:
        yield
    finally:
        config.update("jaxtyping_disable", prev)


def neighbor_from_nodes(messages, scores, dst_index, nodes):
    num_nodes = at.shape(nodes)[0]
    alpha = at.where(scores > 0, scores, scores * 0.2)
    alpha = at.segment_softmax(alpha, dst_index, num_nodes)
    weighted = messages * alpha[:, None]
    return at.segment_sum(weighted, dst_index, num_nodes)


_MESSAGES = np.array(
    [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.0]], dtype=np.float32
)
_SCORES = np.array([1.0, 1.0, 0.5, 2.0], dtype=np.float32)
_DST = np.array([0, 0, 1, 2], dtype=np.int64)
_NODES = np.zeros((3, 2), dtype=np.float32)
_EAGER = neighbor_from_nodes(_MESSAGES, _SCORES, _DST, _NODES)


def _ort(model, arrays):
    ort = pytest.importorskip("onnxruntime")
    proto = _graph._model_proto(model)
    sess = ort.InferenceSession(
        proto.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    feeds = {inp.name: np.asarray(a) for inp, a in zip(sess.get_inputs(), arrays)}
    return sess.run(None, feeds)[0]


def test_export_is_opt_in_unstable_subpackage():
    assert "export" not in at.__all__
    with pytest.raises(ModuleNotFoundError):
        import anytensor.onnx  # noqa: F401
    assert export.__all__ == [
        "as_tensorflow_fn",
        "as_torch_module",
        "assert_embedded_weights",
        "assert_symbolic_lengths",
        "numpy_leaves",
        "to_onnx_tensorflow",
        "to_onnx_torch",
    ]
    nested = {"w": np.arange(3, dtype=np.float32), "inner": (np.ones((2,)), None), "n": 3}
    out = export.numpy_leaves(nested)
    assert out["w"].dtype == np.float32
    assert list(out["inner"][0]) == [1.0, 1.0]
    assert out["inner"][1] is None
    assert out["n"] == 3

    class Frozen(dict):
        pass

    frozen = Frozen(w=np.array([1.0], dtype=np.float32))
    assert export.numpy_leaves(frozen)["w"][0] == 1.0
    listed = export.numpy_leaves([np.array([2.0]), None])
    assert listed[0][0] == 2.0
    assert listed[1] is None
    from collections import namedtuple

    Pair = namedtuple("Pair", "a b")
    pair = export.numpy_leaves(Pair(np.array([1.0]), np.array([2.0])))
    assert pair.a[0] == 1.0


def test_symbolic_dims_and_assert_on_hand_built_graph():
    from onnx import TensorProto, helper

    graph = helper.make_graph(
        [helper.make_node("Identity", ["messages"], ["out"])],
        "id",
        [helper.make_tensor_value_info("messages", TensorProto.FLOAT, ["E", 2])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, ["E", 2])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    dims = _graph._symbolic_dims(model)
    assert dims["messages"] == ("E", 2)
    assert dims["out"] == ("E", 2)
    export.assert_symbolic_lengths(model, inputs={"messages": (0,)}, outputs={"out": (0,)})
    export.assert_symbolic_lengths(model)
    with pytest.raises(AssertionError, match="axis 1"):
        export.assert_symbolic_lengths(model, inputs={"messages": (1,)})
    with pytest.raises(KeyError):
        export.assert_symbolic_lengths(model, inputs={"nope": (0,)})
    export.assert_symbolic_lengths(model, inputs={"mess": (0,)})  # unique prefix
    export.assert_symbolic_lengths(model, outputs={"ut": (0,)})  # unique suffix


def test_assert_symbolic_lengths_skips_rank0_and_rejects_ambiguous():
    from onnx import TensorProto, helper

    graph = helper.make_graph(
        [helper.make_node("Identity", ["scale"], ["graph_out"])],
        "id",
        [helper.make_tensor_value_info("scale", TensorProto.FLOAT, [])],
        [helper.make_tensor_value_info("graph_out", TensorProto.FLOAT, ["N"])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    export.assert_symbolic_lengths(model)  # rank-0 input has no axes
    export.assert_symbolic_lengths(model, outputs={"out": (0,)})

    graph2 = helper.make_graph(
        [
            helper.make_node("Identity", ["a_out"], ["left"]),
            helper.make_node("Identity", ["b_out"], ["right"]),
        ],
        "id",
        [
            helper.make_tensor_value_info("a_out", TensorProto.FLOAT, ["E"]),
            helper.make_tensor_value_info("b_out", TensorProto.FLOAT, ["E"]),
        ],
        [
            helper.make_tensor_value_info("left", TensorProto.FLOAT, ["E"]),
            helper.make_tensor_value_info("right", TensorProto.FLOAT, ["E"]),
        ],
    )
    model2 = helper.make_model(graph2, opset_imports=[helper.make_opsetid("", 18)])
    with pytest.raises(KeyError, match="not in"):
        export.assert_symbolic_lengths(model2, inputs={"out": (0,)})


def test_assert_symbolic_lengths_rejects_static_rank1():
    from onnx import TensorProto, helper

    graph = helper.make_graph(
        [helper.make_node("Identity", ["x"], ["y"])],
        "id",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [4])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    with pytest.raises(AssertionError, match="no symbolic"):
        export.assert_symbolic_lengths(model)


def _weight_params():
    return {
        "Dense_0": {
            "kernel": np.array([[1.0, 0.5], [0.25, 2.0]], dtype=np.float32),
            "bias": np.array([0.5, -0.25], dtype=np.float32),
        }
    }


def _apply_dense(x, *, params):
    layer = params["Dense_0"]
    return x @ layer["kernel"] + layer["bias"]


def test_named_weight_leaves_paths_and_duplicates():
    from collections import namedtuple

    W = np.eye(2, dtype=np.float32)
    B = np.array([1.0, 2.0], dtype=np.float32)
    _td, slots, weights = _bind._named_weight_leaves({"W": W, "n": 3})
    assert "W" in weights and "n" not in weights
    assert ("static", 3) in slots
    assert _bind._path_name(("plain",)) == "plain"
    seq = _bind._named_weight_leaves((W, B))[2]
    assert "p_0" in seq and "p_1" in seq
    NT = namedtuple("NT", "kernel bias")
    named = _bind._named_weight_leaves(NT(W, B))[2]
    assert set(named) == {"kernel", "bias"}
    from anytensor.tree import DictKey

    assert _bind._path_name((DictKey("..."),)) == "p"
    with pytest.raises(ValueError, match="duplicate"):
        _bind._named_weight_leaves({"a-b": W, "a_b": B})
    assert _bind._forward_arg_names(lambda *args: args) is None
    assert _bind._forward_arg_names(42) is None
    assert _bind._forward_arg_names(lambda x, **kwargs: x) == ["x"]
    mixed = _bind._named_weight_leaves({"W": W, "n": 3, "s": "hi"})
    assert mixed[2].keys() == {"W"}


def test_as_torch_module_varargs_and_static_slots():
    torch = pytest.importorskip("torch")
    mod = export.as_torch_module(lambda *args: args[0])
    x = torch.tensor([1.0])
    assert torch.equal(mod(x), x)
    W = np.eye(2, dtype=np.float32)

    def apply(x, *, params):
        return x @ params["W"]

    bound = export.as_torch_module(apply, {"W": W, "n": 3})
    y = bound(torch.ones(2, 2)).detach().cpu().numpy()
    np.testing.assert_allclose(y, np.ones((2, 2)), atol=1e-5)
    d = _serialize._torch_dim("E")
    assert d.min == 1


def test_assert_embedded_weights_hand_built():
    from onnx import TensorProto, helper, numpy_helper

    W = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["x", "W"], ["y"])],
        "m",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["N", 2])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, ["N", 2])],
        [numpy_helper.from_array(W, name="W")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    assert export.assert_embedded_weights(model, {"W": W}) == {"W": "W"}
    assert _graph._initializer_arrays(model)["W"].shape == (2, 2)

    graph_tf = helper.make_graph(
        [helper.make_node("MatMul", ["x", "W:0"], ["y"])],
        "m",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["N", 2])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, ["N", 2])],
        [numpy_helper.from_array(W, name="W:0")],
    )
    model_tf = helper.make_model(graph_tf, opset_imports=[helper.make_opsetid("", 18)])
    assert export.assert_embedded_weights(model_tf, {"W": W})["W"] == "W:0"

    graph_anon = helper.make_graph(
        [helper.make_node("MatMul", ["x", "Const"], ["y"])],
        "m",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["N", 2])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, ["N", 2])],
        [numpy_helper.from_array(W, name="Const")],
    )
    model_anon = helper.make_model(graph_anon, opset_imports=[helper.make_opsetid("", 18)])
    with pytest.raises(AssertionError, match="not an embedded"):
        export.assert_embedded_weights(model_anon, {"W": W})
    assert export.assert_embedded_weights(model_anon, {"W": W}, require_names=False)["W"] == "Const"

    graph_in = helper.make_graph(
        [helper.make_node("MatMul", ["x", "W"], ["y"])],
        "m",
        [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, ["N", 2]),
            helper.make_tensor_value_info("W", TensorProto.FLOAT, [2, 2]),
        ],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, ["N", 2])],
    )
    model_in = helper.make_model(graph_in, opset_imports=[helper.make_opsetid("", 18)])
    with pytest.raises(AssertionError, match="leaked"):
        export.assert_embedded_weights(model_in, {"W": W})
    with pytest.raises(AssertionError, match="no array"):
        export.assert_embedded_weights(model, {"n": 3})
    with pytest.raises(AssertionError, match="not an embedded"):
        export.assert_embedded_weights(model, {"W": np.zeros_like(W)})

    extra = numpy_helper.from_array(W, name="W2")
    graph_dup = helper.make_graph(
        [helper.make_node("Add", ["W", "W2"], ["y"])],
        "m",
        [],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 2])],
        [numpy_helper.from_array(W, name="W"), extra],
    )
    model_dup = helper.make_model(graph_dup, opset_imports=[helper.make_opsetid("", 18)])
    matched = export.assert_embedded_weights(
        model_dup, {"u": W, "v": W.copy()}, require_names=False
    )
    assert set(matched.values()) == {"W", "W2"}

    wrong = numpy_helper.from_array(np.zeros_like(W), name="Z")
    graph_skip = helper.make_graph(
        [helper.make_node("Identity", ["W"], ["y"])],
        "m",
        [],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 2])],
        [wrong, numpy_helper.from_array(W, name="Keep")],
    )
    model_skip = helper.make_model(graph_skip, opset_imports=[helper.make_opsetid("", 18)])
    assert export.assert_embedded_weights(model_skip, {"W": W}, require_names=False)["W"] == "Keep"


def test_to_onnx_rejects_numpy_only():
    with pytest.raises(RuntimeError, match="NumPy-only"):
        _serialize._to_onnx(at.exp, (np.array([1.0], dtype=np.float32),))


def test_to_onnx_rejects_jax_arrays():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    with pytest.raises(RuntimeError, match="jax2tf"):
        _serialize._to_onnx(at.exp, ([jnp.asarray([1.0], dtype=jnp.float32)],))
    del jax


def test_to_onnx_backend_jax_without_arrays():
    with pytest.raises(RuntimeError, match="jax2tf"):
        _serialize._to_onnx(at.exp, (np.array([1.0], dtype=np.float32),), backend="jax")


def test_to_onnx_dispatches_tensorflow():
    tf = pytest.importorskip("tensorflow")
    pytest.importorskip("tf2onnx")
    proto = _serialize._to_onnx(
        at.exp,
        (np.array([0.5, 1.5], dtype=np.float32),),
        backend="tensorflow",
        input_signature=[tf.TensorSpec((None,), tf.float32, name="x")],
    )
    export.assert_symbolic_lengths(proto, inputs={"x": (0,)})
    y = _ort(proto, (np.array([0.5, 1.5], dtype=np.float32),))
    np.testing.assert_allclose(y, np.exp([0.5, 1.5]).astype(np.float32), atol=1e-5)


def test_to_onnx_dispatches_torch(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(_serialize, "to_onnx_torch", lambda *a, **k: "torch-ok")
    x = torch.tensor([1.0])
    assert _serialize._to_onnx(at.exp, (x,)) == "torch-ok"
    assert _serialize._to_onnx(at.exp, (x,), backend="torch") == "torch-ok"


def test_to_onnx_torch_defaults_dynamo_and_wraps(monkeypatch):
    torch = pytest.importorskip("torch")
    captured = {}

    def fake_export(model, args, **kwargs):
        captured["model"] = model
        captured["kwargs"] = kwargs

        class _Prog:
            model_proto = "proto"

        return _Prog()

    monkeypatch.setattr(torch.onnx, "export", fake_export)
    x = torch.tensor([1.0, 2.0])
    prog = export.to_onnx_torch(at.exp, (x,))
    assert _graph._model_proto(prog) == "proto"
    assert captured["kwargs"]["dynamo"] is True
    assert isinstance(captured["model"], torch.nn.Module)
    y = captured["model"](x).detach().cpu().numpy()
    np.testing.assert_allclose(y, np.exp([1.0, 2.0]).astype(np.float32), atol=1e-5)


def test_to_onnx_torch_binds_params(monkeypatch):
    torch = pytest.importorskip("torch")
    captured = {}

    def fake_export(model, args, **kwargs):
        captured["model"] = model
        return model

    monkeypatch.setattr(torch.onnx, "export", fake_export)
    params = {"W": np.eye(2, dtype=np.float32)}

    def apply(x, *, params):
        return x @ params["W"]

    export.to_onnx_torch(apply, (torch.ones(2, 2),), params=params)
    assert "W" in dict(captured["model"].named_parameters())


def test_to_onnx_torch_passes_existing_module(monkeypatch):
    torch = pytest.importorskip("torch")
    seen = {}

    def fake_export(model, args, **kwargs):
        seen["model"] = model
        return model

    monkeypatch.setattr(torch.onnx, "export", fake_export)
    mod = export.as_torch_module(at.exp)
    assert export.to_onnx_torch(mod, (torch.tensor([1.0]),)) is mod
    assert seen["model"] is mod


def test_torch_dim_min_max():
    torch = pytest.importorskip("torch")
    d = _serialize._torch_dim("E", min=2, max=8)
    assert d.min == 2
    assert d.max == 8
    del torch


def test_to_onnx_tensorflow_requires_signature():
    tf = pytest.importorskip("tensorflow")
    x = tf.constant([1.0, 2.0])
    with pytest.raises(TypeError, match="input_signature"):
        _serialize._to_onnx(at.exp, (x,))


def test_tf2onnx_missing_raises(monkeypatch):
    tf = pytest.importorskip("tensorflow")

    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "tf2onnx" or name.startswith("tf2onnx."):
            raise ImportError("hidden")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(RuntimeError, match="tf2onnx"):
        export.to_onnx_tensorflow(
            at.exp, [tf.TensorSpec((None,), tf.float32)], opset=18
        )


def _tf_neighbor_signature():
    tf = pytest.importorskip("tensorflow")
    pytest.importorskip("tf2onnx")
    return [
        tf.TensorSpec((None, 2), tf.float32, name="messages"),
        tf.TensorSpec((None,), tf.float32, name="scores"),
        tf.TensorSpec((None,), tf.int64, name="dst"),
        tf.TensorSpec((None, 2), tf.float32, name="nodes"),
    ]


def test_tensorflow_neighbor_attention_symbolic_lengths():
    tf = pytest.importorskip("tensorflow")
    pytest.importorskip("tf2onnx")
    args = (
        tf.constant(_MESSAGES),
        tf.constant(_SCORES),
        tf.constant(_DST),
        tf.constant(_NODES),
    )
    proto = export.to_onnx_tensorflow(neighbor_from_nodes, _tf_neighbor_signature())
    out_name = proto.graph.output[0].name
    export.assert_symbolic_lengths(
        proto,
        inputs={"messages": (0,), "scores": (0,), "dst": (0,), "nodes": (0,)},
        outputs={out_name: (0,)},
    )
    y = _ort(proto, args)
    np.testing.assert_allclose(y, _EAGER, equal_nan=True, atol=1e-5)
    # Different E and N still run.
    m2 = np.vstack([_MESSAGES, [[0.5, 0.5]]]).astype(np.float32)
    s2 = np.append(_SCORES, 0.3).astype(np.float32)
    d2 = np.append(_DST, 1).astype(np.int64)
    n4 = np.zeros((4, 2), dtype=np.float32)
    y2 = _ort(proto, (m2, s2, d2, n4))
    assert y2.shape == (4, 2)


_ONNX_OP_NAMES = (
    "arange",
    "astype",
    "clip",
    "concatenate",
    "cumsum",
    "einsum",
    "equal_nan",
    "exp",
    "fill_nan",
    "fill_nan_mask",
    "full",
    "full_like",
    "is_finite",
    "is_inf",
    "is_nan",
    "log",
    "matmul",
    "max",
    "maximum",
    "mean",
    "min",
    "minimum",
    "nan_to_num",
    "ones",
    "ones_like",
    "partition_softmax",
    "prod",
    "rearrange",
    "reduce",
    "repeat",
    "reshape",
    "rsqrt",
    "segment_count",
    "segment_max",
    "segment_max_or_constant",
    "segment_mean",
    "segment_min",
    "segment_min_or_constant",
    "segment_normalize",
    "segment_softmax",
    "segment_sum",
    "segment_variance",
    "split",
    "sqrt",
    "stack",
    "sum",
    "take",
    "transpose",
    "where",
    "zeros",
    "zeros_like",
)
_ONNX_SKIP = frozenset(
    {
        "backends",
        "hetero",
        "jraph",
        "tree",
        "get_backend",
        "module_if_loaded",
        "promote",
        "promote_scalars",
        "promote_options",
        "align_arrays",
        "newaxis",
        "__version__",
        "empty_segment_identity",
        "enable_torchscript",
        "enable_typecheck",
        "ArrayT",
        "Axes",
        "Bool",
        "DtypeLike",
        "Float",
        "FloatArray",
        "Inexact",
        "Int",
        "IntArray",
        "Integer",
        "Num",
        "Real",
        "SegmentIds",
        "SegmentOut",
        "SegmentValues",
        "ShapeLike",
        "ShapeSize",
        "Shaped",
        "ShapedArray",
        # Metadata / host specials, not a tensor graph.
        "inf",
        "ninf",
        "nan",
        "pi",
        "e",
        "dtype",
        "finfo",
        "iinfo",
        "shape",
        # Aliases covered by the primary name.
        "cast",
        "isnan",
        "isfinite",
        "isinf",
        "nan_fill",
        "nan_fill_mask",
        # Einops pack/unpack are layout helpers with Python nested sizes.
        "pack",
        "unpack",
    }
)


def _tf_cases() -> dict[str, tuple[Callable, list, tuple]]:
    tf = pytest.importorskip("tensorflow")
    x = tf.constant([0.5, 1.5, 2.5], tf.float32)
    y = tf.constant([1.0, 0.0, 2.0], tf.float32)
    c = tf.constant([True, False, True])
    ids = tf.constant([0, 0, 1], tf.int64)
    idx = tf.constant([0, 2], tf.int64)
    x23 = tf.reshape(tf.range(6, dtype=tf.float32), (2, 3))
    nodes2 = tf.zeros((2, 1), tf.float32)
    vec = tf.TensorSpec((None,), tf.float32)
    vec_i = tf.TensorSpec((None,), tf.int64)
    vec_b = tf.TensorSpec((None,), tf.bool)
    mat = tf.TensorSpec((None, 3), tf.float32)
    col = tf.TensorSpec((None, 1), tf.float32)

    def seg(op):
        return lambda a, i, n: op(a, i, at.shape(n)[0])

    return {
        "exp": (lambda a: at.exp(a), [vec], (x,)),
        "log": (lambda a: at.log(a), [vec], (x,)),
        "sqrt": (lambda a: at.sqrt(a), [vec], (x,)),
        "rsqrt": (lambda a: at.rsqrt(a), [vec], (x,)),
        "sum": (lambda a: at.sum(a), [vec], (x,)),
        "min": (lambda a: at.min(a), [vec], (x,)),
        "max": (lambda a: at.max(a), [vec], (x,)),
        "mean": (lambda a: at.mean(a), [vec], (x,)),
        "prod": (lambda a: at.prod(a), [vec], (x,)),
        "cumsum": (lambda a: at.cumsum(a), [vec], (x,)),
        "maximum": (lambda a, b: at.maximum(a, b), [vec, vec], (x, y)),
        "minimum": (lambda a, b: at.minimum(a, b), [vec, vec], (x, y)),
        "where": (lambda m, a, b: at.where(m, a, b), [vec_b, vec, vec], (c, x, y)),
        "take": (lambda a, i: at.take(a, i), [vec, vec_i], (x, idx)),
        "reshape": (lambda a: at.reshape(a, (-1, 1)), [vec], (x,)),
        "transpose": (lambda a: at.transpose(a, (1, 0)), [mat], (x23,)),
        "concatenate": (lambda a, b: at.concatenate([a, b], axis=0), [vec, vec], (x, y)),
        "stack": (lambda a, b: at.stack([a, b], axis=0), [vec, vec], (x, y)),
        "clip": (lambda a: at.clip(a, 0.0, 2.0), [vec], (x,)),
        "astype": (lambda a: at.astype(a, tf.float64), [vec], (x,)),
        "zeros_like": (lambda a: at.zeros_like(a), [vec], (x,)),
        "ones_like": (lambda a: at.ones_like(a), [vec], (x,)),
        "full_like": (lambda a: at.full_like(a, 3.0), [vec], (x,)),
        # Sizes from at.shape so the constructor introduces a graph symbol
        # (equated with the operand's dim by where it sits in the graph).
        "zeros": (lambda a: at.zeros(at.shape(a), dtype=tf.float32, like=a), [vec], (x,)),
        "ones": (lambda a: at.ones(at.shape(a), dtype=tf.float32, like=a), [vec], (x,)),
        "full": (lambda a: at.full(at.shape(a), 3.0, dtype=tf.float32, like=a), [vec], (x,)),
        "arange": (
            lambda a: at.arange(at.shape(a)[0], dtype=tf.int64, like=a),
            [vec],
            (x,),
        ),
        "split": (
            lambda a: at.concatenate(at.split(a, 3, axis=1), axis=1),
            [mat],
            (x23,),
        ),
        "partition_softmax": (
            lambda a, p: at.partition_softmax(a, p, sum_partitions=at.shape(a)[0]),
            [vec, vec_i],
            (x, tf.constant([2, 1], tf.int64)),
        ),
        "matmul": (
            lambda a: at.matmul(a, tf.constant([[1.0, 0.0], [0.5, 1.0], [0.0, 0.5]], tf.float32)),
            [mat],
            (x23,),
        ),
        "repeat": (lambda a: at.repeat(a, 2), [vec], (x,)),
        "is_nan": (lambda a: at.is_nan(a), [vec], (x,)),
        "is_finite": (lambda a: at.is_finite(a), [vec], (x,)),
        "is_inf": (lambda a: at.is_inf(a), [vec], (x,)),
        "fill_nan": (lambda a: at.fill_nan(a, 0.0), [vec], (x,)),
        "fill_nan_mask": (lambda a: at.fill_nan_mask(a, 0.0)[0], [vec], (x,)),
        "nan_to_num": (lambda a: at.nan_to_num(a), [vec], (x,)),
        "equal_nan": (lambda a, b: at.equal_nan(a, b), [vec, vec], (x, y)),
        "segment_sum": (seg(at.segment_sum), [vec, vec_i, col], (x, ids, nodes2)),
        "segment_min": (seg(at.segment_min), [vec, vec_i, col], (x, ids, nodes2)),
        "segment_max": (seg(at.segment_max), [vec, vec_i, col], (x, ids, nodes2)),
        "segment_mean": (seg(at.segment_mean), [vec, vec_i, col], (x, ids, nodes2)),
        "segment_count": (
            lambda i, n: at.segment_count(i, at.shape(n)[0]),
            [vec_i, col],
            (ids, nodes2),
        ),
        "segment_variance": (seg(at.segment_variance), [vec, vec_i, col], (x, ids, nodes2)),
        "segment_normalize": (seg(at.segment_normalize), [vec, vec_i, col], (x, ids, nodes2)),
        "segment_softmax": (seg(at.segment_softmax), [vec, vec_i, col], (x, ids, nodes2)),
        "segment_min_or_constant": (
            lambda a, i, n: at.segment_min_or_constant(a, i, at.shape(n)[0], 0.0),
            [vec, vec_i, col],
            (x, ids, nodes2),
        ),
        "segment_max_or_constant": (
            lambda a, i, n: at.segment_max_or_constant(a, i, at.shape(n)[0], 0.0),
            [vec, vec_i, col],
            (x, ids, nodes2),
        ),
        "rearrange": (lambda a: at.rearrange(a, "a b -> a b"), [mat], (x23,)),
        "reduce": (lambda a: at.reduce(a, "a b -> a", "sum"), [mat], (x23,)),
        "einsum": (
            lambda a: at.einsum(a, at.transpose(a, (1, 0)), "i j, j k -> i k"),
            [mat],
            (x23,),
        ),
    }


def test_all_public_ops_have_onnx_case_or_skip():
    covered = set(_ONNX_OP_NAMES)
    required = set(at.__all__) - _ONNX_SKIP
    missing = sorted(required - covered)
    extra = sorted(covered - required)
    unknown = sorted(_ONNX_SKIP - set(at.__all__))
    assert not missing, f"public ops missing ONNX cases: {missing}"
    assert not extra, f"ONNX cases without public name: {extra}"
    assert not unknown, f"stale ONNX skip entries: {unknown}"
    pytest.importorskip("tensorflow")
    assert set(_tf_cases()) == covered


@pytest.mark.parametrize("onnx_op_name", list(_ONNX_OP_NAMES))
def test_tensorflow_public_ops_export_and_match(onnx_op_name):
    pytest.importorskip("tensorflow")
    pytest.importorskip("tf2onnx")
    fn, signature, args = _tf_cases()[onnx_op_name]
    proto = export.to_onnx_tensorflow(fn, signature)
    proto_g = _graph._model_proto(proto)
    dims = _graph._symbolic_dims(proto)
    eager = fn(*args)
    y = _ort(proto, args)
    eager_np = np.asarray(eager)
    if eager_np.ndim == 0:
        np.testing.assert_allclose(np.reshape(y, ()), eager_np, equal_nan=True, atol=1e-5)
        return
    out_axes = dims[proto_g.graph.output[0].name]
    if out_axes:
        assert any(isinstance(d, str) and d for d in out_axes), (onnx_op_name, out_axes)
    np.testing.assert_allclose(y, eager_np, equal_nan=True, atol=1e-4)


@_SKIP_CI_TORCH
def test_torch_neighbor_symbolic_e_and_n():
    torch = pytest.importorskip("torch")
    E = _serialize._torch_dim("E")
    N = _serialize._torch_dim("N")
    args = (
        torch.as_tensor(_MESSAGES),
        torch.as_tensor(_SCORES),
        torch.as_tensor(_DST),
        torch.as_tensor(_NODES),
    )
    prog = export.to_onnx_torch(
        neighbor_from_nodes,
        args,
        dynamic_shapes={
            "messages": {0: E},
            "scores": {0: E},
            "dst_index": {0: E},
            "nodes": {0: N},
        },
        input_names=["messages", "scores", "dst_index", "nodes"],
        output_names=["out"],
    )
    dims = export.assert_symbolic_lengths(
        prog,
        inputs={"messages": (0,), "scores": (0,), "dst_index": (0,), "nodes": (0,)},
        outputs={"out": (0,)},
    )
    assert dims["messages"][0] == "E"
    assert dims["nodes"][0] == "N"
    assert dims["out"][0] == "N"
    y = _ort(prog, args)
    np.testing.assert_allclose(y, _EAGER, equal_nan=True, atol=1e-5)
    n4 = torch.zeros((4, 2), dtype=torch.float32)
    y4 = _ort(prog, (args[0], args[1], args[2], n4))
    assert y4.shape == (4, 2)


@_SKIP_CI_TORCH
def test_lightning_module_exports_like_nn_module():
    L = pytest.importorskip("lightning")
    torch = pytest.importorskip("torch")

    class LitNeighbor(L.LightningModule):
        def forward(self, messages, scores, dst_index, nodes):
            return neighbor_from_nodes(messages, scores, dst_index, nodes)

    E, N = _serialize._torch_dim("E"), _serialize._torch_dim("N")
    args = (
        torch.as_tensor(_MESSAGES),
        torch.as_tensor(_SCORES),
        torch.as_tensor(_DST),
        torch.as_tensor(_NODES),
    )
    prog = export.to_onnx_torch(
        LitNeighbor(),
        args,
        dynamic_shapes={
            "messages": {0: E},
            "scores": {0: E},
            "dst_index": {0: E},
            "nodes": {0: N},
        },
    )
    out_name = _graph._model_proto(prog).graph.output[0].name
    export.assert_symbolic_lengths(
        prog, inputs={"messages": (0,), "nodes": (0,)}, outputs={out_name: (0,)}
    )
    y = _ort(prog, args)
    np.testing.assert_allclose(y, _EAGER, equal_nan=True, atol=1e-5)


@_SKIP_CI_TORCH
def test_torch_parameters_embed_named_weights():
    torch = pytest.importorskip("torch")
    params = _weight_params()
    x = torch.ones(3, 2)
    E = _serialize._torch_dim("B")
    prog = export.to_onnx_torch(
        _apply_dense,
        (x,),
        params=params,
        dynamic_shapes={"x": {0: E}},
        input_names=["x"],
        output_names=["y"],
    )
    matched = export.assert_embedded_weights(prog, params)
    assert matched["Dense_0__kernel"] == "Dense_0__kernel"
    assert matched["Dense_0__bias"] == "Dense_0__bias"
    feeds = [i.name for i in _graph._model_proto(prog).graph.input]
    assert "Dense_0__kernel" not in feeds
    y = _ort(prog, (x,))
    np.testing.assert_allclose(
        y, _apply_dense(np.ones((3, 2), dtype=np.float32), params=params), atol=1e-5
    )


def test_flax_params_rebind_to_tensorflow_onnx():
    jax = pytest.importorskip("jax")
    flax = pytest.importorskip("flax")
    tf = pytest.importorskip("tensorflow")
    pytest.importorskip("tf2onnx")
    from flax import linen as nn

    class FlaxAttn(nn.Module):
        @nn.compact
        def __call__(self, messages, scores, dst_index, nodes):
            w = self.param("W", nn.initializers.ones, (2, 2))
            b = self.param("b", nn.initializers.constant(0.5), (2,))
            return neighbor_from_nodes(messages @ w + b, scores, dst_index, nodes)

    mj, sj, dj, nj = (
        jax.numpy.asarray(_MESSAGES),
        jax.numpy.asarray(_SCORES),
        jax.numpy.asarray(_DST),
        jax.numpy.asarray(_NODES),
    )
    module = FlaxAttn()
    variables = module.init(jax.random.key(0), mj, sj, dj, nj)
    params = export.numpy_leaves(variables["params"])
    eager = np.asarray(module.apply(variables, mj, sj, dj, nj))

    def tf_apply(messages, scores, dst, nodes, *, params):
        return neighbor_from_nodes(
            messages @ params["W"] + params["b"], scores, dst, nodes
        )

    proto = export.to_onnx_tensorflow(
        tf_apply, _tf_neighbor_signature(), params=params
    )
    out_name = proto.graph.output[0].name
    export.assert_symbolic_lengths(
        proto,
        inputs={"messages": (0,), "nodes": (0,)},
        outputs={out_name: (0,)},
    )
    matched = export.assert_embedded_weights(proto, params)
    assert matched["W"].startswith("W")
    assert matched["b"].startswith("b")
    y = _ort(
        proto,
        (
            tf.constant(_MESSAGES),
            tf.constant(_SCORES),
            tf.constant(_DST),
            tf.constant(_NODES),
        ),
    )
    np.testing.assert_allclose(y, eager, equal_nan=True, atol=1e-5)
    del flax


def test_tensorflow_named_constants_embed_weights():
    tf = pytest.importorskip("tensorflow")
    pytest.importorskip("tf2onnx")
    params = _weight_params()
    x = np.ones((4, 2), dtype=np.float32)
    proto = _serialize._to_onnx(
        _apply_dense,
        (tf.constant(x),),
        backend="tensorflow",
        params=params,
        input_signature=[tf.TensorSpec((None, 2), tf.float32, name="x")],
    )
    export.assert_symbolic_lengths(proto, inputs={"x": (0,)})
    matched = export.assert_embedded_weights(proto, params)
    assert "Dense_0__kernel" in matched["Dense_0__kernel"]
    y = _ort(proto, (x,))
    np.testing.assert_allclose(y, _apply_dense(x, params=params), atol=1e-5)

    def apply_static(x, *, params):
        return x @ params["Dense_0"]["kernel"] + params["n"]

    proto_s = export.to_onnx_tensorflow(
        apply_static,
        [tf.TensorSpec((None, 2), tf.float32, name="x")],
        params={"Dense_0": params["Dense_0"], "n": 0.0},
    )
    export.assert_embedded_weights(
        proto_s, {"Dense_0": {"kernel": params["Dense_0"]["kernel"]}}
    )


def test_as_tensorflow_fn_eager_static_and_weights():
    tf = pytest.importorskip("tensorflow")
    params = {"W": np.eye(2, dtype=np.float32), "n": 1.0}

    def apply(x, *, params):
        return x @ params["W"] + params["n"]

    fn = export.as_tensorflow_fn(apply, params)
    y = np.asarray(fn(tf.ones((2, 2))))
    np.testing.assert_allclose(y, np.ones((2, 2)) + 1.0, atol=1e-5)


def test_tensorflow_outer_tensors_do_not_embed():
    """Closing over outer constants leaks weights as ONNX inputs."""
    tf = pytest.importorskip("tensorflow")
    pytest.importorskip("tf2onnx")
    params = _weight_params()
    W = params["Dense_0"]["kernel"]
    B = params["Dense_0"]["bias"]
    W_c, B_c = tf.constant(W), tf.constant(B)

    def leaked(x):
        return x @ W_c + B_c

    proto = export.to_onnx_tensorflow(
        leaked, [tf.TensorSpec((None, 2), tf.float32, name="x")]
    )
    with pytest.raises(AssertionError, match="leaked|not an embedded"):
        export.assert_embedded_weights(
            proto, {"kernel": W, "bias": B}, require_names=False
        )


def test_keras_tf_function_recipe():
    """Keras 3 ``model.export(format='onnx')`` is not the supported path."""
    keras = pytest.importorskip("keras")
    tf = pytest.importorskip("tensorflow")
    pytest.importorskip("tf2onnx")
    del keras

    class NeighborLayer(tf.keras.layers.Layer):
        def call(self, inputs):
            messages, scores, dst, nodes = inputs
            return neighbor_from_nodes(messages, scores, dst, nodes)

    layer = NeighborLayer()

    def call(messages, scores, dst, nodes):
        return layer([messages, scores, dst, nodes])

    proto = export.to_onnx_tensorflow(call, _tf_neighbor_signature())
    out_name = proto.graph.output[0].name
    export.assert_symbolic_lengths(
        proto, inputs={"messages": (0,)}, outputs={out_name: (0,)}
    )
    y = _ort(
        proto,
        (
            tf.constant(_MESSAGES),
            tf.constant(_SCORES),
            tf.constant(_DST),
            tf.constant(_NODES),
        ),
    )
    np.testing.assert_allclose(y, _EAGER, equal_nan=True, atol=1e-5)


def test_as_torch_module_forward():
    torch = pytest.importorskip("torch")
    mod = export.as_torch_module(at.exp)
    x = torch.tensor([1.0, 0.0])
    np.testing.assert_allclose(
        mod(x).detach().cpu().numpy(), np.exp(np.array([1.0, 0.0], dtype=np.float32)), atol=1e-5
    )

    class _Fn:
        def __call__(self, x):
            return x

    unnamed = export.as_torch_module(_Fn())
    assert type(unnamed).__name__ == "AnyTensorModule"

    params = _weight_params()
    bound = export.as_torch_module(_apply_dense, params)
    assert set(dict(bound.named_parameters())) == {"Dense_0__kernel", "Dense_0__bias"}
    x = torch.ones(3, 2)
    y = bound(x).detach().cpu().numpy()
    eager = _apply_dense(np.ones((3, 2), dtype=np.float32), params=params)
    np.testing.assert_allclose(y, eager, atol=1e-5)

    buf = export.as_torch_module(_apply_dense, params, buffers=True)
    assert set(dict(buf.named_buffers())) >= {"Dense_0__kernel", "Dense_0__bias"}
    np.testing.assert_allclose(buf(x).detach().cpu().numpy(), eager, atol=1e-5)

    with pytest.raises(TypeError, match="params="):
        export.to_onnx_torch(bound, (x,), params=params)


_WRITES = ("author", "writes", "paper")
_CITES = ("paper", "cites", "paper")
_HETERO_WEIGHTS = {
    "Ww": np.eye(2, dtype=np.float32) * 1.1,
    "Wc": np.eye(2, dtype=np.float32) * 0.9,
    "Sa": np.eye(2, dtype=np.float32) * 0.2,
    "Sp": np.eye(2, dtype=np.float32) * 0.3,
    "C": np.concatenate(
        [np.eye(2, dtype=np.float32), np.eye(2, dtype=np.float32)], axis=0
    ),
    "attn": np.ones((4, 1), dtype=np.float32) * 0.5,
    "q": np.array([0.7, -0.2], dtype=np.float32),
}
_JRAPH_WEIGHTS = {"W": np.eye(2, dtype=np.float32) * 1.2}


def _lin(w):
    return lambda x: x @ w


def _size1(x):
    """Turn a leading length into a length-1 vector (one graph in the batch)."""
    return at.reshape(at.shape(x)[0:1], (1,))


def _dot_logit(a, b):
    return at.reshape(at.sum(a * b, axes=-1), (at.shape(a)[0], 1))


def _hetero_pack(author, paper, w_send, w_recv, c_send, c_recv, w_e, c_e):
    from anytensor.hetero import HeteroGraphsTuple

    return HeteroGraphsTuple(
        nodes={"author": author, "paper": paper},
        edges={_WRITES: w_e, _CITES: c_e},
        senders={_WRITES: w_send, _CITES: c_send},
        receivers={_WRITES: w_recv, _CITES: c_recv},
        n_node={"author": _size1(author), "paper": _size1(paper)},
        n_edge={_WRITES: _size1(w_send), _CITES: _size1(c_send)},
    )


def _hetero_signature(tf):
    return [
        tf.TensorSpec((None, 2), tf.float32, name="author"),
        tf.TensorSpec((None, 2), tf.float32, name="paper"),
        tf.TensorSpec((None,), tf.int32, name="w_send"),
        tf.TensorSpec((None,), tf.int32, name="w_recv"),
        tf.TensorSpec((None,), tf.int32, name="c_send"),
        tf.TensorSpec((None,), tf.int32, name="c_recv"),
        tf.TensorSpec((None, 2), tf.float32, name="w_e"),
        tf.TensorSpec((None, 2), tf.float32, name="c_e"),
    ]


def _hetero_args(tf):
    return (
        tf.constant([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        tf.constant([[0.5, 0.5], [2.0, 0.0]]),
        tf.constant([0, 1, 2], tf.int32),
        tf.constant([0, 0, 1], tf.int32),
        tf.constant([0], tf.int32),
        tf.constant([1], tf.int32),
        tf.ones((3, 2), tf.float32),
        tf.ones((1, 2), tf.float32),
    )


def _rgcn_fn(author, paper, w_send, w_recv, c_send, c_recv, w_e, c_e, *, params):
    from anytensor.hetero import relational_graph_convolution

    g = _hetero_pack(author, paper, w_send, w_recv, c_send, c_recv, w_e, c_e)
    return relational_graph_convolution(
        g,
        {_WRITES: _lin(params["Ww"]), _CITES: _lin(params["Wc"])},
        {"author": _lin(params["Sa"]), "paper": _lin(params["Sp"])},
        activation=lambda x: x,
    ).nodes["paper"]


def _sage_fn(author, paper, w_send, w_recv, c_send, c_recv, w_e, c_e, *, params):
    from anytensor.hetero import hetero_sage

    g = _hetero_pack(author, paper, w_send, w_recv, c_send, c_recv, w_e, c_e)
    return hetero_sage(
        g,
        {_WRITES: _lin(params["Ww"]), _CITES: _lin(params["Wc"])},
        {"paper": _lin(params["C"]), "author": _lin(params["C"])},
        activation=lambda x: x,
    ).nodes["paper"]


def _compgcn_fn(author, paper, w_send, w_recv, c_send, c_recv, w_e, c_e, *, params):
    from anytensor.hetero import comp_gcn

    g = _hetero_pack(author, paper, w_send, w_recv, c_send, c_recv, w_e, c_e)
    return comp_gcn(
        g,
        {_WRITES: _lin(params["Ww"]), _CITES: _lin(params["Wc"])},
        {"author": _lin(params["Sa"]), "paper": _lin(params["Sp"])},
        composition="mult",
        activation=lambda x: x,
    ).nodes["paper"]


def _hgt_fn(author, paper, w_send, w_recv, c_send, c_recv, w_e, c_e, *, params):
    from anytensor.hetero import gat_attention_logit, hgt

    g = _hetero_pack(author, paper, w_send, w_recv, c_send, c_recv, w_e, c_e)
    attn = _lin(params["attn"])
    return hgt(
        g,
        {_WRITES: _lin(params["Ww"]), _CITES: _lin(params["Wc"])},
        {
            _WRITES: lambda s, d, e: gat_attention_logit(s, d, attn),
            _CITES: lambda s, d, e: gat_attention_logit(s, d, attn),
        },
        {"paper": _lin(params["Sp"])},
    ).nodes["paper"]


def _han_fn(author, paper, w_send, w_recv, c_send, c_recv, w_e, c_e, *, params):
    from anytensor.hetero import gat_attention_logit, han

    g = _hetero_pack(author, paper, w_send, w_recv, c_send, c_recv, w_e, c_e)
    attn = _lin(params["attn"])
    return han(
        g,
        [_WRITES, _CITES],
        {_WRITES: _lin(params["Ww"]), _CITES: _lin(params["Wc"])},
        {
            _WRITES: lambda s, d, e: gat_attention_logit(s, d, attn),
            _CITES: lambda s, d, e: gat_attention_logit(s, d, attn),
        },
        _lin(params["Sp"]),
        params["q"],
        node_activation=lambda x: x,
        semantic_activation=lambda x: x,
    ).nodes["paper"]


def _pick(keys):
    return {k: _HETERO_WEIGHTS[k] for k in keys}


@pytest.mark.parametrize(
    "zoo_fn, weight_keys",
    [
        pytest.param(_rgcn_fn, ("Ww", "Wc", "Sa", "Sp"), id="rgcn"),
        pytest.param(_sage_fn, ("Ww", "Wc", "C"), id="hetero_sage"),
        pytest.param(_compgcn_fn, ("Ww", "Wc", "Sa", "Sp"), id="comp_gcn"),
        pytest.param(_hgt_fn, ("Ww", "Wc", "attn", "Sp"), id="hgt"),
        pytest.param(_han_fn, ("Ww", "Wc", "attn", "Sp", "q"), id="han"),
    ],
)
def test_hetero_zoo_tensorflow_onnx(zoo_fn, weight_keys):
    tf = pytest.importorskip("tensorflow")
    pytest.importorskip("tf2onnx")
    params = _pick(weight_keys)
    args = _hetero_args(tf)
    proto = export.to_onnx_tensorflow(zoo_fn, _hetero_signature(tf), params=params)
    export.assert_symbolic_lengths(proto, inputs={"author": (0,), "paper": (0,)})
    export.assert_embedded_weights(proto, params)
    y = _ort(proto, args)
    eager = np.asarray(zoo_fn(*args, params=params))
    np.testing.assert_allclose(y, eager, equal_nan=True, atol=1e-4)


def _jraph_pack(nodes, edges, senders, receivers, n_node, n_edge, globals_):
    from anytensor.jraph.graph import GraphsTuple

    return GraphsTuple(
        nodes=nodes,
        edges=edges,
        senders=senders,
        receivers=receivers,
        globals=globals_,
        n_node=n_node,
        n_edge=n_edge,
    )


def _jraph_signature(tf):
    return [
        tf.TensorSpec((None, 2), tf.float32, name="nodes"),
        tf.TensorSpec((None, 2), tf.float32, name="edges"),
        tf.TensorSpec((None,), tf.int32, name="senders"),
        tf.TensorSpec((None,), tf.int32, name="receivers"),
        tf.TensorSpec((None,), tf.int32, name="n_node"),
        tf.TensorSpec((None,), tf.int32, name="n_edge"),
        tf.TensorSpec((None, 2), tf.float32, name="globals"),
    ]


def _jraph_args(tf):
    return (
        tf.constant(np.arange(6.0, dtype=np.float32).reshape(3, 2)),
        tf.constant(np.arange(10.0, dtype=np.float32).reshape(5, 2)),
        tf.constant([0, 0, 1, 1, 2], tf.int32),
        tf.constant([1, 2, 0, 2, 1], tf.int32),
        tf.constant([3], tf.int32),
        tf.constant([5], tf.int32),
        tf.constant([[1.0, 0.0]], tf.float32),
    )


def _jraph_gat(nodes, edges, senders, receivers, n_node, n_edge, globals_, *, params):
    from anytensor import jraph as atj

    g = _jraph_pack(nodes, edges, senders, receivers, n_node, n_edge, globals_)
    return atj.GAT(
        lambda x: x @ params["W"],
        lambda a, b, _e: _dot_logit(a, b),
        lambda x: x,
    )(g).nodes


def _jraph_gcn(nodes, edges, senders, receivers, n_node, n_edge, globals_, *, params):
    from anytensor import jraph as atj

    g = _jraph_pack(nodes, edges, senders, receivers, n_node, n_edge, globals_)
    return atj.GraphConvolution(lambda x: x @ params["W"], add_self_edges=True)(g).nodes


def _jraph_gn(nodes, edges, senders, receivers, n_node, n_edge, globals_, *, params):
    from anytensor import jraph as atj

    g = _jraph_pack(nodes, edges, senders, receivers, n_node, n_edge, globals_)
    return atj.GraphNetwork(
        lambda e, s, r, _g: e,
        lambda n, s, r, _g: n @ params["W"],
        None,
    )(g).nodes


def _jraph_interaction(
    nodes, edges, senders, receivers, n_node, n_edge, globals_, *, params
):
    from anytensor import jraph as atj

    g = _jraph_pack(nodes, edges, senders, receivers, n_node, n_edge, globals_)
    return atj.InteractionNetwork(
        lambda e, s, r: e,
        lambda n, r: n @ params["W"],
    )(g).nodes


def _jraph_map(nodes, edges, senders, receivers, n_node, n_edge, globals_, *, params):
    from anytensor import jraph as atj

    g = _jraph_pack(nodes, edges, senders, receivers, n_node, n_edge, globals_)
    embed = atj.GraphMapFeatures(lambda e: e, lambda n: n @ params["W"], lambda gl: gl)
    return embed(g).nodes


def _jraph_deepsets(
    nodes, edges, senders, receivers, n_node, n_edge, globals_, *, params
):
    from anytensor import jraph as atj

    g = _jraph_pack(nodes, edges, senders, receivers, n_node, n_edge, globals_)
    # Node update reads repeated globals, so arange/repeat stay live in the graph.
    return atj.DeepSets(lambda n, gl: n @ params["W"] + gl, lambda n: n)(g).nodes


def _jraph_relation(
    nodes, edges, senders, receivers, n_node, n_edge, globals_, *, params
):
    from anytensor import jraph as atj

    g = _jraph_pack(nodes, edges, senders, receivers, n_node, n_edge, globals_)
    return atj.RelationNetwork(
        lambda s, r: s + r,
        lambda e: e @ params["W"],
    )(g).globals


def _jraph_gngat(nodes, edges, senders, receivers, n_node, n_edge, globals_, *, params):
    from anytensor import jraph as atj

    g = _jraph_pack(nodes, edges, senders, receivers, n_node, n_edge, globals_)
    return atj.GraphNetGAT(
        lambda e, s, r, gl: e,
        lambda n, s, r, gl: n @ params["W"],
        lambda e, s, r, gl: _dot_logit(s, r),
        lambda e, w: e * w,
    )(g).nodes


@pytest.mark.parametrize(
    "zoo_fn",
    [
        pytest.param(_jraph_gat, id="gat"),
        pytest.param(_jraph_gcn, id="gcn"),
        pytest.param(_jraph_gn, id="graph_network"),
        pytest.param(_jraph_interaction, id="interaction_network"),
        pytest.param(_jraph_map, id="graph_map_features"),
        pytest.param(_jraph_deepsets, id="deep_sets"),
        pytest.param(_jraph_relation, id="relation_network"),
        pytest.param(_jraph_gngat, id="graphnet_gat"),
    ],
)
def test_jraph_zoo_tensorflow_onnx(zoo_fn):
    tf = pytest.importorskip("tensorflow")
    pytest.importorskip("tf2onnx")
    params = dict(_JRAPH_WEIGHTS)
    args = _jraph_args(tf)
    proto = export.to_onnx_tensorflow(zoo_fn, _jraph_signature(tf), params=params)
    export.assert_symbolic_lengths(proto, inputs={"nodes": (0,), "edges": (0,)})
    export.assert_embedded_weights(proto, params)
    y = _ort(proto, args)
    eager = np.asarray(zoo_fn(*args, params=params))
    np.testing.assert_allclose(y, eager, equal_nan=True, atol=1e-4)

