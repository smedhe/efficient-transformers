# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""CPU-only weight-free blocking export structure tests."""

from __future__ import annotations

from collections import Counter

import onnx
import pytest

from .._helpers import (
    CTX_LEN,
    PROMPT_LEN,
    WEIGHT_FREE_CAUSAL_LM_MODEL_IDS,
    build_meta_qeff_model,
    exported_onnx_path,
    skip_on_model_fetch_error,
)

# Same supported model set used by tests/dynamo/unit_tests/test_blocking.py.
_BLOCKING_SUPPORTED_TYPES = {
    "gemma",
    "gemma2",
    "glm4_moe",
    "gpt_oss",
    "granite",
    "granitemoe",
    "llama",
    "mistral",
    "mixtral",
    "qwen2",
    "qwen3",
    "qwen3_moe",
    "starcoder2",
}
BLOCKING_MODEL_IDS = {k: v for k, v in WEIGHT_FREE_CAUSAL_LM_MODEL_IDS.items() if k in _BLOCKING_SUPPORTED_TYPES}

HEAD_BLOCK_SIZE = 2
NUM_KV_BLOCKS = 2
NUM_Q_BLOCKS = 2

# Modes with a reliable CtxGatherBlockedKV export marker.
BLOCKING_MODE_CASES = {
    "kv": dict(enable_blocking=True, blocking_mode="kv", num_kv_blocks=NUM_KV_BLOCKS),
    "qkv": dict(enable_blocking=True, blocking_mode="qkv", num_kv_blocks=NUM_KV_BLOCKS, num_q_blocks=NUM_Q_BLOCKS),
    "hkv": dict(
        enable_blocking=True, blocking_mode="hkv", head_block_size=HEAD_BLOCK_SIZE, num_kv_blocks=NUM_KV_BLOCKS
    ),
    "hqkv": (
        dict(
            enable_blocking=True,
            blocking_mode="hqkv",
            head_block_size=HEAD_BLOCK_SIZE,
            num_kv_blocks=NUM_KV_BLOCKS,
            num_q_blocks=NUM_Q_BLOCKS,
        )
    ),
}


@pytest.mark.weight_free
@pytest.mark.weight_free_export
@pytest.mark.parametrize("blocking_key", list(BLOCKING_MODE_CASES))
@pytest.mark.parametrize("model_type,model_id", sorted(BLOCKING_MODEL_IDS.items()), ids=sorted(BLOCKING_MODEL_IDS))
def test_weight_free_blocking_export_gather_ops(model_type, model_id, blocking_key, tmp_export_dir):
    """Verify KV-blocking modes emit CtxGatherBlockedKV in decoder subfunctions."""
    if model_type == "gpt_oss":
        pytest.xfail("gpt_oss forward() disables blocking whenever self.sliding_window is not None")
    if blocking_key == "hkv":
        pytest.xfail(f"blocked_hqkv_attention_forward crashes on mode='{blocking_key}' (None block count)")

    # transform() mutates qaic_config.
    qaic_config = dict(BLOCKING_MODE_CASES[blocking_key])

    try:
        qeff_model = build_meta_qeff_model(model_id)
    except Exception as exc:
        skip_on_model_fetch_error(exc, model_id)
    qeff_model.transform(
        ctx_len=CTX_LEN,
        seq_len=PROMPT_LEN,
        bs=1,
        qaic_config=qaic_config,
    )

    onnx_path = exported_onnx_path(
        qeff_model.export(
            tmp_export_dir,
            use_weight_free_export=True,
            use_onnx_subfunctions=True,
            offload_pt_weights=False,
        )
    )

    onnx_model = onnx.load(str(onnx_path), load_external_data=False)
    get_submodules = getattr(qeff_model.model, "get_submodules_for_export", None)
    decoder_names = {cls.__name__ for cls in get_submodules()} if callable(get_submodules) else set()
    decoder_functions = [fn for fn in onnx_model.functions if any(name in fn.name for name in decoder_names)]
    assert decoder_functions, (
        f"Expected decoder-block subfunctions ({decoder_names}) but found none. "
        f"Functions present: {[fn.name for fn in onnx_model.functions]}"
    )

    for function_proto in decoder_functions:
        op_counts = Counter(node.op_type for node in function_proto.node)
        blocked_kv_count = op_counts["CtxGatherBlockedKV"]
        assert blocked_kv_count > 0, (
            f"Expected CtxGatherBlockedKV ops in {function_proto.name} for mode '{blocking_key}' "
            f"but found none. Ops present: {dict(op_counts)}"
        )
