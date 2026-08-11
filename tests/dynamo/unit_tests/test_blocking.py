# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Dynamo + attention blocking (BlockedKV) export structure tests.

All tests run with dynamo=True and use_onnx_subfunctions=True.
CPU-only. No QAIC hardware required.

Blocking requires an explicit QEFFAutoModelForCausalLM.transform(qaic_config=...)
call before export() -- export() alone never applies it (attn_blocking_config
defaults to BlockingMode.NONE on every attention module until transform()
attaches a real config; see tests/dynamo/test_blocking.py for why compile()
is the one that calls transform() internally when no onnx_path is given).
Here we call transform() explicitly since we only export, never compile.

Verifies the exported ONNX graph actually contains CtxGatherBlockedKV custom
ops -- the same structural marker tests/transformers/models/test_moe_prefill_blocked.py
uses (test_glm4_moe_kv_blocking_transform_and_prefill_export) -- rather than
tests/transformers/subfunction/test_causal_lm_blocking_subfunction.py's
function-count comparison, which is vacuous: it never calls transform(), so
both its "blocked" and "unblocked" exports are actually unblocked.
"""

from __future__ import annotations

from collections import Counter

import onnx
import pytest
import torch

from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM

from .._helpers import (
    CTX_LEN,
    DYNAMO_CAUSAL_LM_MODEL_IDS,
    PROMPT_LEN,
    exported_onnx_path,
    load_hf_model,
    skip_on_model_fetch_error,
)

# Same set used in tests/dynamo/test_blocking.py -- architectures whose
# modeling_*.py forward() reads attn_blocking_config directly (unlike
# QEffGPT2Attention, which is mapped in KVCacheTransform._module_mapping but
# never checks the config, so a blocked export of gpt2 is silently identical
# to an unblocked one). Confirmed via
# `grep -rl attn_blocking_config QEfficient/transformers/models/<arch>/`.
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
    "mpt",
    "qwen2",
    "qwen3",
    "qwen3_moe",
    "starcoder2",
}
BLOCKING_MODEL_IDS = {k: v for k, v in DYNAMO_CAUSAL_LM_MODEL_IDS.items() if k in _BLOCKING_SUPPORTED_TYPES}


@pytest.mark.dynamo
@pytest.mark.dynamo_export
@pytest.mark.parametrize("model_type,model_id", sorted(BLOCKING_MODEL_IDS.items()), ids=sorted(BLOCKING_MODEL_IDS))
def test_dynamo_kv_blocking_export_has_blocked_gather_ops(model_type, model_id, tmp_export_dir):
    """transform(qaic_config={'enable_blocking': True, 'num_kv_blocks': 2}) -> export(dynamo=True,
    use_onnx_subfunctions=True) must produce CtxGatherBlockedKV ops in every decoder-block subfunction."""
    if model_type == "gpt_oss":
        pytest.xfail("gpt_oss forward() disables blocking whenever self.sliding_window is not None")

    try:
        model_hf = load_hf_model(model_id, torch_dtype=torch.float32)
    except Exception as exc:
        skip_on_model_fetch_error(exc, model_id)

    qeff_model = QEFFAutoModelForCausalLM(model_hf)
    qeff_model.transform(
        ctx_len=CTX_LEN,
        seq_len=PROMPT_LEN,
        bs=1,
        qaic_config={"enable_blocking": True, "num_kv_blocks": 2},
    )

    onnx_path = exported_onnx_path(
        qeff_model.export(
            tmp_export_dir,
            dynamo=True,
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
        assert op_counts["CtxGatherBlockedKV"] > 0, (
            f"Expected CtxGatherBlockedKV ops in {function_proto.name} but found none. Ops present: {dict(op_counts)}"
        )


@pytest.mark.dynamo
@pytest.mark.dynamo_export
@pytest.mark.parametrize("model_type,model_id", sorted(BLOCKING_MODEL_IDS.items()), ids=sorted(BLOCKING_MODEL_IDS))
def test_dynamo_export_without_blocking_has_no_blocked_gather_ops(model_type, model_id, tmp_export_dir):
    """Sanity check: without transform(), the plain CtxGather op is used, never CtxGatherBlockedKV --
    confirms the previous test's assertion is actually discriminating, not trivially true."""
    try:
        model_hf = load_hf_model(model_id, torch_dtype=torch.float32)
    except Exception as exc:
        skip_on_model_fetch_error(exc, model_id)

    qeff_model = QEFFAutoModelForCausalLM(model_hf)
    onnx_path = exported_onnx_path(
        qeff_model.export(
            tmp_export_dir,
            dynamo=True,
            use_onnx_subfunctions=True,
            offload_pt_weights=False,
        )
    )

    onnx_model = onnx.load(str(onnx_path), load_external_data=False)
    op_counts = Counter(node.op_type for fn in onnx_model.functions for node in fn.node)
    assert op_counts["CtxGatherBlockedKV"] == 0
