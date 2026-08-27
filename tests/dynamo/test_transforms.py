# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Unit tests for the dynamo-specific transforms and context managers introduced
in the enable_dynamo_for_causallm branch.

Covered:
  - qeff_nested_compile_region
  - PreserveNestedCacheRetainedStateTransform
  - RenameRepeatedSubgraphTransform
  - PruneFakeInitializersTransform

CPU-only. No QAIC hardware required.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import torch
from onnx import TensorProto, helper
from transformers import LlamaConfig, LlamaForCausalLM

from QEfficient.base.onnx_transforms import (
    PreserveNestedCacheRetainedStateTransform,
    PruneFakeInitializersTransform,
    RenameRepeatedSubgraphTransform,
)
from QEfficient.transformers.models.llama.modeling_llama import QEffLlamaDecoderLayer
from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM
from QEfficient.utils.torch_patches import qeff_nested_compile_region

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_tiny_llama():
    cfg = LlamaConfig(
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        hidden_size=64,
        intermediate_size=128,
        vocab_size=500,
        max_position_embeddings=32,
    )
    model = LlamaForCausalLM(cfg).eval()
    return model, cfg


def _make_minimal_onnx_with_repeated_subgraphs(num_layers: int = 2, scatter_count_per_fn: int = 2):
    """
    Build a minimal ONNX ModelProto that mimics dynamo's repeated-subgraph output:
      - graph has num_layers call nodes (one per layer), each referencing repeated_subgraphN
      - each function contains scatter_count_per_fn CtxScatter nodes
      - graph outputs include past_key/value _RetainedState placeholders (dangling)
    """
    functions = []
    call_nodes = []
    graph_outputs = []
    graph_inputs = []

    for i in range(num_layers):
        fn_name = f"repeated_subgraph{i}"

        # Scatter nodes inside the function
        scatter_nodes = []
        fn_outputs = []
        for j in range(scatter_count_per_fn):
            kind = "key" if j == 0 else "value"
            scatter_out = f"scatter_{kind}_{i}"
            scatter_node = helper.make_node(
                "CtxScatter",
                inputs=[f"past_{kind}.{i}", f"new_{kind}_{i}", "position_ids"],
                outputs=[scatter_out],
                domain="qti.aisw",
            )
            scatter_nodes.append(scatter_node)
            fn_outputs.append(scatter_out)

        fn = helper.make_function(
            domain="",
            fname=fn_name,
            inputs=[f"past_key.{i}", f"past_value.{i}", f"hidden_{i}", "position_ids"],
            outputs=fn_outputs,
            nodes=scatter_nodes,
            opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("qti.aisw", 1)],
        )
        functions.append(fn)

        # Call node in the main graph — outputs start EMPTY so that the
        # _RetainedState names in graph.output are dangling (not produced by any
        # node). PreserveNestedCacheRetainedStateTransform must wire them up.
        retained_key = f"past_key.{i}_RetainedState"
        retained_val = f"past_value.{i}_RetainedState"
        call_node = helper.make_node(
            fn_name,
            inputs=[f"past_key.{i}", f"past_value.{i}", f"hidden_{i}", "position_ids"],
            outputs=[],
            domain="",
        )
        call_nodes.append(call_node)

        graph_outputs.append(helper.make_tensor_value_info(retained_key, TensorProto.FLOAT, None))
        graph_outputs.append(helper.make_tensor_value_info(retained_val, TensorProto.FLOAT, None))

        graph_inputs.append(helper.make_tensor_value_info(f"past_key.{i}", TensorProto.FLOAT, None))
        graph_inputs.append(helper.make_tensor_value_info(f"past_value.{i}", TensorProto.FLOAT, None))
        graph_inputs.append(helper.make_tensor_value_info(f"hidden_{i}", TensorProto.FLOAT, None))

    graph_inputs.append(helper.make_tensor_value_info("position_ids", TensorProto.INT64, None))

    graph = helper.make_graph(call_nodes, "test_graph", graph_inputs, graph_outputs)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    for fn in functions:
        model.functions.append(fn)
    return model


# ---------------------------------------------------------------------------
# TestQEffNestedCompileRegionDecorator
# ---------------------------------------------------------------------------


class TestQEffNestedCompileRegionDecorator:
    def test_decoder_layer_forward_is_decorated(self):
        model_hf, _ = make_tiny_llama()
        qeff_model = QEFFAutoModelForCausalLM(model_hf)
        decoder_layers = [m for m in qeff_model.model.modules() if isinstance(m, QEffLlamaDecoderLayer)]

        assert decoder_layers
        assert all(getattr(layer.forward, "__marked_compile_region_fn__", None) is not None for layer in decoder_layers)

    def test_decorator_uses_torch_nested_compile_region_when_available(self, monkeypatch):
        calls = []

        def fake_nested_compile_region(fn):
            calls.append(fn)

            def wrapped(*args, **kwargs):
                return fn(*args, **kwargs) + 1

            return wrapped

        monkeypatch.setattr(torch.compiler, "nested_compile_region", fake_nested_compile_region)

        @qeff_nested_compile_region
        def add_one(value):
            return value + 1

        assert add_one(1) == 3
        assert len(calls) == 1

    def test_decorator_falls_back_when_torch_nested_compile_region_is_unavailable(self, monkeypatch):
        monkeypatch.delattr(torch.compiler, "nested_compile_region", raising=False)

        @qeff_nested_compile_region
        def add_one(value):
            return value + 1

        assert add_one(1) == 2


# ---------------------------------------------------------------------------
# TestPreserveNestedCacheRetainedStateTransform
# ---------------------------------------------------------------------------


class TestPreserveNestedCacheRetainedStateTransform:
    def test_adds_retained_state_outputs_to_call_nodes(self):
        model = _make_minimal_onnx_with_repeated_subgraphs(num_layers=2, scatter_count_per_fn=2)

        # Initially the call nodes have outputs but the functions don't expose scatter outputs
        changed = PreserveNestedCacheRetainedStateTransform.apply(model)
        assert changed, "Transform should have modified the model (dangling _RetainedState outputs)"

        # After transform: function outputs should include scatter node outputs
        for i, fn in enumerate(model.functions):
            assert len(fn.output) >= 2, (
                f"Function '{fn.name}' should have at least 2 outputs after transform, got {list(fn.output)}"
            )

    def test_noop_when_no_dangling_retained_states(self):
        model = _make_minimal_onnx_with_repeated_subgraphs(num_layers=2, scatter_count_per_fn=2)

        # Remove all _RetainedState outputs from the graph — nothing is dangling
        for out in list(model.graph.output):
            if out.name.endswith("_RetainedState"):
                model.graph.output.remove(out)

        changed = PreserveNestedCacheRetainedStateTransform.apply(model)
        assert not changed, "Transform should be a no-op when there are no dangling _RetainedState outputs"

    def test_noop_when_scatter_count_not_two(self):
        # Build model where function has only 1 scatter node
        model = _make_minimal_onnx_with_repeated_subgraphs(num_layers=1, scatter_count_per_fn=1)
        PreserveNestedCacheRetainedStateTransform.apply(model)
        # The key invariant: no crash; the function with only 1 scatter is skipped
        fn = model.functions[0]
        assert len(fn.output) == 1, f"Function with 1 scatter should not have outputs added, got {list(fn.output)}"


# ---------------------------------------------------------------------------
# TestRenameRepeatedSubgraphTransform
# ---------------------------------------------------------------------------


class TestRenameRepeatedSubgraphTransform:
    def test_renames_repeated_subgraph_functions(self):
        model = _make_minimal_onnx_with_repeated_subgraphs(num_layers=2)
        changed = RenameRepeatedSubgraphTransform.apply(model, target_classnames=["QEffLlamaDecoderLayer"])
        assert changed

        fn_names = [fn.name for fn in model.functions]
        assert "QEffLlamaDecoderLayer" in fn_names, f"Expected 'QEffLlamaDecoderLayer' in {fn_names}"
        assert "QEffLlamaDecoderLayer_1" in fn_names, f"Expected 'QEffLlamaDecoderLayer_1' in {fn_names}"

        # Call-site op_type must also be updated
        node_op_types = [n.op_type for n in model.graph.node]
        assert "QEffLlamaDecoderLayer" in node_op_types
        assert "QEffLlamaDecoderLayer_1" in node_op_types

    def test_noop_on_empty_classnames(self):
        model = _make_minimal_onnx_with_repeated_subgraphs(num_layers=2)
        changed = RenameRepeatedSubgraphTransform.apply(model, target_classnames=[])
        assert not changed

    def test_noop_when_no_repeated_subgraph_functions(self):
        # Build a model with a non-repeated_subgraph function name
        fn = helper.make_function(
            domain="",
            fname="SomeOtherFunction",
            inputs=[],
            outputs=[],
            nodes=[],
            opset_imports=[helper.make_opsetid("", 17)],
        )
        graph = helper.make_graph([], "g", [], [])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        model.functions.append(fn)

        changed = RenameRepeatedSubgraphTransform.apply(model, target_classnames=["QEffLlamaDecoderLayer"])
        assert not changed

    def test_handles_alternative_subgraph_pattern(self):
        # torch < 2.5 naming: subgraph_0, subgraph_1
        fn0 = helper.make_function(
            domain="",
            fname="subgraph_0",
            inputs=[],
            outputs=[],
            nodes=[],
            opset_imports=[helper.make_opsetid("", 17)],
        )
        fn1 = helper.make_function(
            domain="",
            fname="subgraph_1",
            inputs=[],
            outputs=[],
            nodes=[],
            opset_imports=[helper.make_opsetid("", 17)],
        )
        call0 = helper.make_node("subgraph_0", inputs=[], outputs=[])
        call1 = helper.make_node("subgraph_1", inputs=[], outputs=[])
        graph = helper.make_graph([call0, call1], "g", [], [])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        model.functions.extend([fn0, fn1])

        changed = RenameRepeatedSubgraphTransform.apply(model, target_classnames=["MyDecoderLayer"])
        assert changed
        fn_names = {fn.name for fn in model.functions}
        assert "MyDecoderLayer" in fn_names


# ---------------------------------------------------------------------------
# TestPruneFakeInitializersTransform
# ---------------------------------------------------------------------------


class TestPruneFakeInitializersTransform:
    def _make_mock_onnx_program(self, initializer_names, used_names, fake_initializers):
        """Build a mock onnx_program object matching PruneFakeInitializersTransform's API."""
        from torch._subclasses.fake_tensor import FakeTensor

        initializers = {}
        for name in initializer_names:
            mock_init = MagicMock()
            if name in fake_initializers:
                fake_tensor = MagicMock(spec=FakeTensor)
                mock_init.const_value.raw = fake_tensor
            else:
                mock_init.const_value.raw = torch.zeros(2)
            initializers[name] = mock_init

        mock_graph = MagicMock()
        mock_graph.initializers = initializers

        # Simulate used_names via graph nodes + outputs
        mock_node = MagicMock()
        mock_node.inputs = list(used_names)
        mock_graph.__iter__ = lambda self: iter([mock_node])
        mock_graph.outputs = []

        mock_program = MagicMock()
        mock_program.model.graph = mock_graph
        return mock_program

    def test_prunes_fake_tensor_initializers(self):
        program = self._make_mock_onnx_program(
            initializer_names=["weight_a", "weight_b"],
            used_names=set(),  # neither is used
            fake_initializers={"weight_a"},
        )
        changed = PruneFakeInitializersTransform.apply(program)
        assert changed
        assert "weight_a" not in program.model.graph.initializers

    def test_preserves_used_fake_initializers(self):
        program = self._make_mock_onnx_program(
            initializer_names=["weight_a"],
            used_names={"weight_a"},  # it is used
            fake_initializers={"weight_a"},
        )
        changed = PruneFakeInitializersTransform.apply(program)
        assert not changed
        assert "weight_a" in program.model.graph.initializers

    def test_preserves_non_fake_initializers(self):
        program = self._make_mock_onnx_program(
            initializer_names=["real_weight"],
            used_names=set(),
            fake_initializers=set(),  # not a FakeTensor
        )
        changed = PruneFakeInitializersTransform.apply(program)
        assert not changed
        assert "real_weight" in program.model.graph.initializers
