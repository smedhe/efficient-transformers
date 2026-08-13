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
examples/text_generation/batch_blocking_example.py. Unlike test_ccl.py, no
onnx_path is pre-exported here: leaving onnx_path=None lets
QEFFBaseModel._compile() -> get_onnx_path() call self.transform(...) (which
derives an AttentionBlockingConfig from ctx_len/seq_len/batch_size/num_devices
and attaches it to every supported attention module) and then self.export(...)
internally, with the exact ctx_len/seq_len/batch_size that specializations
was built from. Each blocking mode/CB combination gets its own independent
export -- no export caching/reuse anywhere in tests/dynamo/ (continuous_batching
and ccl_enabled/blocking config are baked into the exported graph at export
time, so an export built for one configuration can never be recompiled under
a different one; sharing an export cache across configurations would be
incorrect, not just redundant).

Covers all 7 non-batch BlockingMode values (head, kv, q, qkv, hq, hkv, hqkv) in
the basic test, plus bhqkv (batch+head+q+kv) added only in the
continuous-batching test -- broader than
tests/transformers/models/causal_lm_models/test_causal_lm_blocking_hqkv.py's
non-dynamo path (head-only, kv-only, q-only, qkv, and head+qkv blocking).

Head/hq/hkv/hqkv blocking split attention heads across devices (num_devices=4)
and, matching test_dynamo_multi_device_compile in test_on_qaic.py, only verify
compile success -- no generate() is attempted for multi-device configs
anywhere else in tests/dynamo/, and those blocking_keys carry
@pytest.mark.dynamo_multi_device so they're auto-skipped on non-MDP-capable
nodes (see conftest.py::skip_if_no_mdp_setup). bhqkv (also head-blocked) gets
the same multi-device treatment in the continuous-batching test.

TODO: hq and hkv are marked xfail everywhere -- blocked_hqkv_attention_forward
(QEfficient/blocking/blocked_attention_forwards.py, handles hq/hkv/hqkv) does
unconditional max(1, num_kv_blocks) and max(1, num_q_blocks); hq's config
leaves num_kv_blocks=None and hkv's leaves num_q_blocks=None, so both raise
TypeError: '>' not supported between 'NoneType' and 'int' during export, even
in plain eager Python. Fix the source, then remove the xfail marks here.
"""

from __future__ import annotations

import pytest
import torch

from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM

from ._helpers import (
    BATCH_SIZE,
    DYNAMO_CAUSAL_LM_MODEL_IDS,
    FULL_BATCH_SIZE,
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
    "qwen2",
    "qwen3",
    "qwen3_moe",
    "starcoder2",
}
BLOCKING_MODEL_IDS = {k: v for k, v in DYNAMO_CAUSAL_LM_MODEL_IDS.items() if k in _BLOCKING_SUPPORTED_TYPES}

HEAD_BLOCK_SIZE = 2
NUM_KV_BLOCKS = 2
NUM_Q_BLOCKS = 2
NUM_BATCH_BLOCKS = 2
PROMPT_LEN_BLOCKING = 32
CTX_LEN_BLOCKING = 128
# head blocking splits attention heads across devices/SoCs (mdp_ts_num_devices
# feeds the blocking-config computation), so it needs num_devices > 1 to mean
# anything -- matches test_causal_lm_blocking_hqkv.py's num_devices=4.
HEAD_BLOCKING_NUM_DEVICES = 4
MULTI_DEVICE_BLOCKING_KEYS = {"head", "hq", "hkv", "hqkv", "bhqkv"}
# See module docstring TODO: blocked_hqkv_attention_forward crashes for hq/hkv.
XFAIL_BLOCKING_KEYS = {"hq", "hkv", "bhqkv"}

# All 7 BlockingMode values reachable from single-attention-call configs (NONE
# is the no-blocking baseline, covered by unit_tests/test_blocking.py's
# sanity-check test). build_transformer_blocking_config_for_transform derives
# the mode string purely from which qaic_config keys are set -- see
# QEfficient/blocking/blocking_configurator.py -- num_kv_blocks -> "kv",
# num_q_blocks -> "q", head_block_size -> "h", combined in h+q+kv order.
# BLOCKING_QAIC_CONFIGS = {
#     "head": dict(enable_blocking=True, head_block_size=HEAD_BLOCK_SIZE),
#     "kv": dict(enable_blocking=True, num_kv_blocks=NUM_KV_BLOCKS),
#     "q": dict(enable_blocking=True, num_q_blocks=NUM_Q_BLOCKS),
#     "qkv": dict(enable_blocking=True, num_kv_blocks=NUM_KV_BLOCKS, num_q_blocks=NUM_Q_BLOCKS),
#     "hq": dict(enable_blocking=True, head_block_size=HEAD_BLOCK_SIZE, num_q_blocks=NUM_Q_BLOCKS),
#     "hkv": dict(enable_blocking=True, head_block_size=HEAD_BLOCK_SIZE, num_kv_blocks=NUM_KV_BLOCKS),
#     "hqkv": dict(
#         enable_blocking=True,
#         head_block_size=HEAD_BLOCK_SIZE,
#         num_kv_blocks=NUM_KV_BLOCKS,
#         num_q_blocks=NUM_Q_BLOCKS,
#     ),
# }
BLOCKING_QAIC_CONFIGS = {
    "head": dict(enable_blocking=True, blocking_mode="h"),
    "kv": dict(enable_blocking=True, blocking_mode="kv"),
    "q": dict(enable_blocking=True, blocking_mode="q"),
    "qkv": dict(enable_blocking=True, blocking_mode="qkv"),
    "hq": dict(enable_blocking=True, blocking_mode="hq"),
    "hkv": dict(enable_blocking=True, blocking_mode="hkv"),
    "hqkv": dict(enable_blocking=True, blocking_mode="hqkv"),
}

# BHQKV (batch+head+q+kv) additionally requires num_batch_blocks <= batch_size
# to mean anything -- with the basic (non-CB) test's batch_size=1,
# num_batch_blocks would clamp to 1 and silently no-op (same failure shape as
# the gpt_oss issue found in tests/dynamo/unit_tests/test_blocking.py), so
# bhqkv is only exercised in the continuous-batching test below, where
# full_batch_size=FULL_BATCH_SIZE=4 >= NUM_BATCH_BLOCKS=2. "b" is only
# recognized when blocking_mode is explicitly set to a string containing "b"
# (default blocking_mode is "hqkv", which has no "b") -- see
# build_transformer_blocking_config_for_transform's `"b" in blocking_mode` gate.
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
    """compile(qaic_config=<blocking mode>, dynamo=True, use_onnx_subfunctions=True) -> generate.

    Multi-device blocking_keys (head, hq, hkv, hqkv) only verify compile success; no generate()."""
    # Copy -- compile()/transform() mutates qaic_config in place (e.g. stamping
    # num_replicate_kv_heads), and BLOCKING_QAIC_CONFIGS[blocking_key] is a single
    # dict object shared across every model parametrized under this blocking_key.
    qaic_config = dict(BLOCKING_QAIC_CONFIGS[blocking_key])
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
        prefill_seq_len=PROMPT_LEN_BLOCKING,
        ctx_len=CTX_LEN_BLOCKING,
        num_cores=16,
        num_devices=num_devices,
        batch_size=BATCH_SIZE,
        qaic_config=qaic_config,
        user_tiled=True,
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
@pytest.mark.parametrize("blocking_key", CB_BLOCKING_KEY_PARAMS)
@pytest.mark.parametrize("model_type,model_id", sorted(BLOCKING_MODEL_IDS.items()), ids=sorted(BLOCKING_MODEL_IDS))
def test_dynamo_cb_blocking_compile_and_generate(model_type, model_id, blocking_key, tmp_export_dir):
    """Continuous-batching + blocking: compile(qaic_config=<blocking mode>, dynamo=True,
    use_onnx_subfunctions=True) -> generate.

    Multi-device blocking_keys (head, hq, hkv, hqkv, bhqkv) only verify compile success; no generate()."""
    # TODO: fix gpt_oss CB scatter op shape mismatch with dynamo subfunctions (see test_dynamo_cb_generate).
    if model_type == "gpt_oss":
        pytest.skip("gpt_oss CB scatter op has shape mismatch with dynamo subfunctions — pending fix")

    qaic_config = dict(CB_BLOCKING_QAIC_CONFIGS[blocking_key])
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
        prefill_seq_len=PROMPT_LEN_BLOCKING,
        ctx_len=CTX_LEN_BLOCKING,
        num_cores=16,
        num_devices=num_devices,
        batch_size=BATCH_SIZE,
        full_batch_size=FULL_BATCH_SIZE,
        qaic_config=qaic_config,
        user_tiled=True,
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
