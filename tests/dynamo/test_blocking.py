# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Dynamo + attention blocking compile/generate tests."""

from __future__ import annotations

import pytest

from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM

from ._helpers import (
    BATCH_SIZE,
    DTYPE,
    DYNAMO,
    DYNAMO_CAUSAL_LM_MODEL_IDS,
    FULL_BATCH_SIZE,
    assert_hf_hw_parity,
    get_hf_tokens,
    load_hf_model,
    load_tokenizer,
    skip_on_hf_model_load_error,
)

# Only include architectures whose forward path consumes attn_blocking_config.
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
BLOCKING_MODEL_IDS = {k: v for k, v in DYNAMO_CAUSAL_LM_MODEL_IDS.items() if k in _BLOCKING_SUPPORTED_TYPES}

HEAD_BLOCK_SIZE = 2
NUM_KV_BLOCKS = 8
NUM_Q_BLOCKS = 8
NUM_BATCH_BLOCKS = 2
PROMPT_LEN_BLOCKING = 32
CTX_LEN_BLOCKING = 128
PROMPT = "hello world"
# Head blocking is meaningful only with multiple devices.
HEAD_BLOCKING_NUM_DEVICES = 4
MULTI_DEVICE_BLOCKING_KEYS = {"head", "hq", "hkv", "hqkv", "bhqkv"}
# hq/hkv/bhqkv currently crash in blocked_hqkv_attention_forward.
XFAIL_BLOCKING_KEYS = {"hq", "hkv", "bhqkv"}

BLOCKING_QAIC_CONFIGS = {
    "head": dict(enable_blocking=True, blocking_mode="h", head_block_size=HEAD_BLOCK_SIZE),
    "kv": dict(enable_blocking=True, blocking_mode="kv", num_kv_blocks=NUM_KV_BLOCKS),
    "q": dict(enable_blocking=True, blocking_mode="q", num_kv_blocks=NUM_KV_BLOCKS),
    "qkv": dict(enable_blocking=True, blocking_mode="qkv", num_kv_blocks=NUM_KV_BLOCKS, num_q_blocks=NUM_Q_BLOCKS),
    "hq": dict(enable_blocking=True, blocking_mode="hq", head_block_size=HEAD_BLOCK_SIZE, num_q_blocks=NUM_Q_BLOCKS),
    "hkv": dict(
        enable_blocking=True, blocking_mode="hkv", head_block_size=HEAD_BLOCK_SIZE, num_kv_blocks=NUM_KV_BLOCKS
    ),
    "hqkv": dict(
        enable_blocking=True,
        blocking_mode="hqkv",
        head_block_size=HEAD_BLOCK_SIZE,
        num_kv_blocks=NUM_KV_BLOCKS,
        num_q_blocks=NUM_Q_BLOCKS,
    ),
}


BHQKV_KEY = "bhqkv"
BHQKV_CONFIG = dict(
    enable_blocking=True,
    blocking_mode="bhqkv",
    head_block_size=HEAD_BLOCK_SIZE,
    num_kv_blocks=NUM_KV_BLOCKS,
    num_q_blocks=NUM_Q_BLOCKS,
    num_batch_blocks=NUM_BATCH_BLOCKS,
)


def _key_params(keys):
    params = []
    for key in keys:
        marks = []
        if key in MULTI_DEVICE_BLOCKING_KEYS:
            marks.append(pytest.mark.dynamo_multi_device)
        if key in XFAIL_BLOCKING_KEYS:
            marks.append(pytest.mark.xfail(reason=f"blocked_hqkv_attention_forward crashes on mode='{key}'"))
        params.append(pytest.param(key, marks=marks) if marks else key)
    return params


BLOCKING_KEY_PARAMS = _key_params(BLOCKING_QAIC_CONFIGS)
CB_BLOCKING_QAIC_CONFIGS = {**BLOCKING_QAIC_CONFIGS, BHQKV_KEY: BHQKV_CONFIG}
CB_BLOCKING_KEY_PARAMS = _key_params(CB_BLOCKING_QAIC_CONFIGS)


@pytest.mark.dynamo
@pytest.mark.on_qaic
@pytest.mark.xdist_group(name="qaic-runtime")
@pytest.mark.llm_model
@pytest.mark.parametrize("blocking_key", BLOCKING_KEY_PARAMS)
@pytest.mark.parametrize("model_type,model_id", sorted(BLOCKING_MODEL_IDS.items()), ids=sorted(BLOCKING_MODEL_IDS))
def test_dynamo_blocking_compile_and_generate(model_type, model_id, blocking_key, tmp_export_dir):
    """Compile blocking modes; single-device modes also validate parity."""
    if model_type == "gpt_oss" and blocking_key == "qkv":
        pytest.xfail("gpt_oss qkv blocking export fails with fake-tensor broadcast mismatch")

    # compile()/transform() mutates qaic_config.
    qaic_config = dict(BLOCKING_QAIC_CONFIGS[blocking_key])
    is_multi_device = blocking_key in MULTI_DEVICE_BLOCKING_KEYS
    num_devices = HEAD_BLOCKING_NUM_DEVICES if is_multi_device else 1

    try:
        qeff_model = QEFFAutoModelForCausalLM.from_pretrained(model_id, torch_dtype=DTYPE)
    except Exception as exc:
        skip_on_hf_model_load_error(exc, model_id)

    compile_dir = tmp_export_dir / f"{blocking_key}_compile"
    qeff_model.compile(
        compile_dir=str(compile_dir),
        prefill_seq_len=PROMPT_LEN_BLOCKING,
        ctx_len=CTX_LEN_BLOCKING,
        num_cores=16,
        num_devices=num_devices,
        batch_size=BATCH_SIZE,
        qaic_config=qaic_config,
        user_tiled=True,
        dynamo=DYNAMO,
        use_onnx_subfunctions=True,
    )

    if is_multi_device:
        assert compile_dir.is_dir()
        return

    try:
        tokenizer = load_tokenizer(model_id, torch_dtype=DTYPE)
        model_hf = load_hf_model(model_id, torch_dtype=DTYPE)
    except Exception as exc:
        skip_on_hf_model_load_error(exc, model_id)

    prompts = [PROMPT]
    hf_tokens = get_hf_tokens(tokenizer, model_hf, prompts, prompt_len=PROMPT_LEN_BLOCKING, ctx_len=CTX_LEN_BLOCKING)
    output = qeff_model.generate(
        tokenizer=tokenizer,
        prompts=prompts,
        device_id=[0],
    )
    assert output.generated_texts is not None
    assert_hf_hw_parity(
        model_id,
        hf_tokens,
        output,
        gen_len=CTX_LEN_BLOCKING - PROMPT_LEN_BLOCKING,
        context="blocking",
    )


@pytest.mark.dynamo
@pytest.mark.on_qaic
@pytest.mark.xdist_group(name="qaic-runtime")
@pytest.mark.llm_model
@pytest.mark.parametrize("blocking_key", CB_BLOCKING_KEY_PARAMS)
@pytest.mark.parametrize("model_type,model_id", sorted(BLOCKING_MODEL_IDS.items()), ids=sorted(BLOCKING_MODEL_IDS))
def test_dynamo_cb_blocking_compile_and_generate(model_type, model_id, blocking_key, tmp_export_dir):
    """Compile continuous-batching blocking modes; single-device modes also validate parity."""
    if model_type == "gpt_oss":
        pytest.skip("gpt_oss CB scatter op has shape mismatch with dynamo subfunctions; pending fix")

    qaic_config = dict(CB_BLOCKING_QAIC_CONFIGS[blocking_key])
    is_multi_device = blocking_key in MULTI_DEVICE_BLOCKING_KEYS
    num_devices = HEAD_BLOCKING_NUM_DEVICES if is_multi_device else 1

    try:
        qeff_model = QEFFAutoModelForCausalLM.from_pretrained(model_id, torch_dtype=DTYPE, continuous_batching=True)
    except Exception as exc:
        skip_on_hf_model_load_error(exc, model_id)

    compile_dir = tmp_export_dir / f"cb_{blocking_key}_compile"
    qeff_model.compile(
        compile_dir=str(compile_dir),
        prefill_seq_len=PROMPT_LEN_BLOCKING,
        ctx_len=CTX_LEN_BLOCKING,
        num_cores=16,
        num_devices=num_devices,
        batch_size=BATCH_SIZE,
        full_batch_size=FULL_BATCH_SIZE,
        qaic_config=qaic_config,
        user_tiled=True,
        dynamo=DYNAMO,
        use_onnx_subfunctions=True,
    )

    if is_multi_device:
        assert compile_dir.is_dir()
        return

    try:
        tokenizer = load_tokenizer(model_id, torch_dtype=DTYPE)
        model_hf = load_hf_model(model_id, torch_dtype=DTYPE)
    except Exception as exc:
        skip_on_hf_model_load_error(exc, model_id)

    prompts = [PROMPT] * FULL_BATCH_SIZE
    hf_tokens = get_hf_tokens(
        tokenizer,
        model_hf,
        prompts,
        prompt_len=PROMPT_LEN_BLOCKING,
        ctx_len=CTX_LEN_BLOCKING,
        full_batch_size=FULL_BATCH_SIZE,
    )
    output = qeff_model.generate(
        tokenizer=tokenizer,
        prompts=prompts,
        device_id=[0],
    )
    assert output.generated_texts is not None
    assert len(output.generated_texts) == FULL_BATCH_SIZE
    assert_hf_hw_parity(
        model_id,
        hf_tokens,
        output,
        gen_len=CTX_LEN_BLOCKING - PROMPT_LEN_BLOCKING,
        full_batch_size=FULL_BATCH_SIZE,
        context="blocking",
    )
