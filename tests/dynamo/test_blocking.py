# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Dynamo + attention blocking (BlockedKV) compile/generate tests.

Requires QAIC hardware (marked @pytest.mark.on_qaic) -- qaic-compile is not
installed on GitHub-hosted CI runners, only on the Jenkins QAIC node.

Blocking is enabled purely via qaic_config passed to compile() -- see
examples/text_generation/batch_blocking_example.py. Unlike test_on_qaic.py/
test_ccl.py, no onnx_path is pre-exported here: leaving onnx_path=None lets
QEFFBaseModel._compile() -> get_onnx_path() call self.transform(...) (which
derives an AttentionBlockingConfig from ctx_len/seq_len/batch_size/num_devices
and attaches it to every supported attention module) and then self.export(...)
internally, with the exact ctx_len/seq_len/batch_size that specializations
was built from. Each blocking mode/CB combination still needs its own
compile() call -- since transform() depends on those same values, there is
no export to cache/share across the different blocking_key parametrizations
here (unlike get_dynamo_export() in test_on_qaic.py/test_ccl.py).

Matches how tests/transformers/models/causal_lm_models/test_causal_lm_blocking_hqkv.py
exercises the non-dynamo path: head-only, kv-only, q-only, qkv, and head+qkv
blocking, each with and without continuous batching.

Head/head+qkv blocking split attention heads across devices (num_devices=4)
and, matching test_dynamo_multi_device_compile in test_on_qaic.py, only verify
compile success -- no generate() is attempted for multi-device configs
anywhere else in tests/dynamo/, and those two blocking_keys carry
@pytest.mark.dynamo_multi_device so they're auto-skipped on non-MDP-capable
nodes (see conftest.py::skip_if_no_mdp_setup).
"""

from __future__ import annotations

import pytest
import torch

from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM

from ._helpers import (
    BATCH_SIZE,
    CTX_LEN,
    DYNAMO_CAUSAL_LM_MODEL_IDS,
    FULL_BATCH_SIZE,
    PROMPT_LEN,
    load_tokenizer,
    skip_on_model_fetch_error,
)

# Restricted to architectures whose modeling_*.py forward() actually reads
# attn_blocking_config. BlockingAttentionTransform.apply() attaches the config
# to any module in KVCacheTransform._module_mapping, but not every mapped
# class's forward() checks it (e.g. QEffGPT2Attention doesn't -- blocking
# would silently no-op and this test would pass vacuously). Confirmed via
# `grep -rl attn_blocking_config QEfficient/transformers/models/<arch>/`.
# NOTE: gpt_oss references attn_blocking_config but its forward() disables
# blocking whenever self.sliding_window is not None, so gpt_oss compiles and
# generates fine here but is silently running unblocked -- this compile/
# generate-only test can't detect that (see tests/dynamo/unit_tests/
# test_blocking.py, which does assert on ONNX op structure and xfails gpt_oss).
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

HEAD_BLOCK_SIZE = 2
NUM_KV_BLOCKS = 2
NUM_Q_BLOCKS = 2
# head blocking splits attention heads across devices/SoCs (mdp_ts_num_devices
# feeds the blocking-config computation), so it needs num_devices > 1 to mean
# anything -- matches test_causal_lm_blocking_hqkv.py's num_devices=4.
HEAD_BLOCKING_NUM_DEVICES = 4
MULTI_DEVICE_BLOCKING_KEYS = {"head", "hqkv"}

BLOCKING_QAIC_CONFIGS = {
    "head": dict(enable_blocking=True, head_block_size=HEAD_BLOCK_SIZE),
    "kv": dict(enable_blocking=True, num_kv_blocks=NUM_KV_BLOCKS),
    "q": dict(enable_blocking=True, num_q_blocks=NUM_Q_BLOCKS),
    "qkv": dict(enable_blocking=True, num_kv_blocks=NUM_KV_BLOCKS, num_q_blocks=NUM_Q_BLOCKS),
    "hqkv": dict(
        enable_blocking=True,
        head_block_size=HEAD_BLOCK_SIZE,
        num_kv_blocks=NUM_KV_BLOCKS,
        num_q_blocks=NUM_Q_BLOCKS,
    ),
}

BLOCKING_KEY_PARAMS = [
    pytest.param(key, marks=pytest.mark.dynamo_multi_device) if key in MULTI_DEVICE_BLOCKING_KEYS else key
    for key in BLOCKING_QAIC_CONFIGS
]


@pytest.mark.dynamo
@pytest.mark.on_qaic
@pytest.mark.xdist_group(name="qaic-runtime")
@pytest.mark.llm_model
@pytest.mark.parametrize("blocking_key", BLOCKING_KEY_PARAMS)
@pytest.mark.parametrize("model_type,model_id", sorted(BLOCKING_MODEL_IDS.items()), ids=sorted(BLOCKING_MODEL_IDS))
def test_dynamo_blocking_compile_and_generate(model_type, model_id, blocking_key, tmp_export_dir):
    """compile(qaic_config=<blocking mode>, dynamo=True, use_onnx_subfunctions=True) -> generate.

    Multi-device blocking_keys (head, hqkv) only verify compile success; no generate()."""
    qaic_config = BLOCKING_QAIC_CONFIGS[blocking_key]
    is_multi_device = blocking_key in MULTI_DEVICE_BLOCKING_KEYS
    num_devices = HEAD_BLOCKING_NUM_DEVICES if is_multi_device else 1

    try:
        qeff_model = QEFFAutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, torch_dtype=torch.float16
        )
    except Exception as exc:
        skip_on_model_fetch_error(exc, model_id)

    compile_dir = tmp_export_dir / f"{blocking_key}_compile"
    qeff_model.compile(
        compile_dir=str(compile_dir),
        prefill_seq_len=PROMPT_LEN,
        ctx_len=CTX_LEN,
        num_cores=16,
        num_devices=num_devices,
        batch_size=BATCH_SIZE,
        qaic_config=qaic_config,
        dynamo=True,
        use_onnx_subfunctions=True,
    )

    if is_multi_device:
        assert compile_dir.is_dir()
        return

    tokenizer = load_tokenizer(model_id, torch_dtype=torch.float16)
    output = qeff_model.generate(
        tokenizer=tokenizer,
        prompts=["hello world"],
        device_id=[0],
    )
    assert output is not None
    assert output.generated_texts is not None


@pytest.mark.dynamo
@pytest.mark.on_qaic
@pytest.mark.xdist_group(name="qaic-runtime")
@pytest.mark.llm_model
@pytest.mark.parametrize("blocking_key", BLOCKING_KEY_PARAMS)
@pytest.mark.parametrize("model_type,model_id", sorted(BLOCKING_MODEL_IDS.items()), ids=sorted(BLOCKING_MODEL_IDS))
def test_dynamo_cb_blocking_compile_and_generate(model_type, model_id, blocking_key, tmp_export_dir):
    """Continuous-batching + blocking: compile(qaic_config=<blocking mode>, dynamo=True,
    use_onnx_subfunctions=True) -> generate.

    Multi-device blocking_keys (head, hqkv) only verify compile success; no generate()."""
    # TODO: fix gpt_oss CB scatter op shape mismatch with dynamo subfunctions (see test_dynamo_cb_generate).
    if model_type == "gpt_oss":
        pytest.skip("gpt_oss CB scatter op has shape mismatch with dynamo subfunctions — pending fix")

    qaic_config = BLOCKING_QAIC_CONFIGS[blocking_key]
    is_multi_device = blocking_key in MULTI_DEVICE_BLOCKING_KEYS
    num_devices = HEAD_BLOCKING_NUM_DEVICES if is_multi_device else 1

    try:
        qeff_model = QEFFAutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, torch_dtype=torch.float16, continuous_batching=True
        )
    except Exception as exc:
        skip_on_model_fetch_error(exc, model_id)

    compile_dir = tmp_export_dir / f"cb_{blocking_key}_compile"
    qeff_model.compile(
        compile_dir=str(compile_dir),
        prefill_seq_len=PROMPT_LEN,
        ctx_len=CTX_LEN,
        num_cores=16,
        num_devices=num_devices,
        batch_size=BATCH_SIZE,
        full_batch_size=FULL_BATCH_SIZE,
        qaic_config=qaic_config,
        dynamo=True,
        use_onnx_subfunctions=True,
    )

    if is_multi_device:
        assert compile_dir.is_dir()
        return

    tokenizer = load_tokenizer(model_id, torch_dtype=torch.float16)
    prompts = ["hello world"] * FULL_BATCH_SIZE
    output = qeff_model.generate(
        tokenizer=tokenizer,
        prompts=prompts,
        device_id=[0],
    )
    assert output is not None
    assert output.generated_texts is not None
    assert len(output.generated_texts) == FULL_BATCH_SIZE
