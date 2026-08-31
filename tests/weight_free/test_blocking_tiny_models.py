# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Weight-free blocking QAIC tests using explicit tiny model configs."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pytest
import torch
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, LlamaConfig

from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM
from QEfficient.utils.device_utils import get_available_device_id

from ._helpers import assert_blocked_kv_ops_for_mode, assert_hf_hw_parity, exported_onnx_path, get_hf_tokens

VOCAB_SIZE_FLOOR = 512
BATCH_SIZE = 1
FULL_BATCH_SIZE = 4
PROMPT_LEN_BLOCKING = 32
CTX_LEN_BLOCKING = 128
HEAD_BLOCK_SIZE = 2
NUM_KV_BLOCKS = 2
NUM_Q_BLOCKS = 2
NUM_BATCH_BLOCKS = 2
HEADPAR_SPLIT = 4
TOKENIZER_MODEL_ID = "hf-internal-testing/tiny-random-LlamaForCausalLM"
PROMPT = "hello world"
CB_PROMPTS = ["hello world", "quick brown fox", "machine learning", "open source"]


@dataclass(frozen=True)
class BlockingQaicCase:
    model_label: str
    config_factory: Callable[[int], object]
    qaic_config: dict
    num_devices: int = 1
    batch_size: int = BATCH_SIZE
    prompt_len: int = PROMPT_LEN_BLOCKING
    ctx_len: int = CTX_LEN_BLOCKING
    xfail_reason: str | None = None


def _make_tiny_llama_config(vocab_size: int):
    return LlamaConfig(
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        hidden_size=128,
        intermediate_size=256,
        vocab_size=vocab_size,
        max_position_embeddings=CTX_LEN_BLOCKING,
        pad_token_id=0,
    )


def _make_tiny_glm4_moe_config(vocab_size: int):
    return AutoConfig.for_model(
        "glm4_moe",
        max_position_embeddings=CTX_LEN_BLOCKING,
        num_hidden_layers=2,
        num_attention_heads=4,
        hidden_size=128,
        intermediate_size=256,
        moe_intermediate_size=32,
        vocab_size=vocab_size,
        num_key_value_heads=2,
        n_routed_experts=4,
        num_experts_per_tok=2,
        first_k_dense_replace=0,
        n_group=1,
        topk_group=1,
        head_dim=32,
        pad_token_id=0,
    )


def _make_tiny_gpt_oss_config(vocab_size: int):
    pytest.importorskip("transformers")
    from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig

    return GptOssConfig(
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        hidden_size=128,
        intermediate_size=128,
        head_dim=32,
        vocab_size=vocab_size,
        max_position_embeddings=CTX_LEN_BLOCKING,
        num_local_experts=4,
        num_experts_per_tok=2,
        sliding_window=32,
        layer_types=["full_attention", "full_attention"],
        rope_parameters={"rope_type": "default"},
    )


def _make_kimi_k25_language_config(vocab_size: int):
    from tests.utils.load_kimi_utils import KIMI_K25_MODEL_NAME, get_kimi_k25_test_config

    config_path = Path(__file__).parents[1] / "configs" / "image_text_model_configs.json"
    model_configs = json.loads(config_path.read_text())["image_text_models"]
    model_config_dict = {model["model_name"]: model for model in model_configs}
    config = get_kimi_k25_test_config(KIMI_K25_MODEL_NAME, model_config_dict)
    config.text_config.vocab_size = vocab_size
    config.text_config.max_position_embeddings = CTX_LEN_BLOCKING
    return config.text_config


def _make_tiny_qwen3_vl_moe_config(vocab_size: int):
    from transformers.models.qwen3_vl_moe.configuration_qwen3_vl_moe import (
        Qwen3VLMoeConfig,
        Qwen3VLMoeTextConfig,
        Qwen3VLMoeVisionConfig,
    )

    text_config = Qwen3VLMoeTextConfig(
        vocab_size=vocab_size,
        hidden_size=128,
        intermediate_size=256,
        moe_intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=512,
        num_experts=4,
        num_experts_per_tok=2,
        decoder_sparse_step=1,
        mlp_only_layers=[],
        rope_scaling={"rope_type": "default", "mrope_section": [11, 11, 10]},
        pad_token_id=0,
        dtype="float32",
    )
    vision_config = Qwen3VLMoeVisionConfig(
        depth=1,
        hidden_size=16,
        intermediate_size=32,
        num_heads=2,
        patch_size=4,
        temporal_patch_size=1,
        spatial_merge_size=1,
        out_hidden_size=16,
        num_position_embeddings=64,
        deepstack_visual_indexes=[],
        dtype="float32",
    )
    return Qwen3VLMoeConfig(
        text_config=text_config,
        vision_config=vision_config,
        image_token_id=3,
        video_token_id=4,
        vision_start_token_id=5,
        vision_end_token_id=6,
    )


def _qaic_config(mode: str, **kwargs) -> dict:
    return {"blocking_mode": mode, **kwargs}


def _case_id(case: BlockingQaicCase) -> str:
    mode = case.qaic_config["blocking_mode"]
    suffix = "-mdp" if case.num_devices > 1 else ""
    return f"{case.model_label}-{mode}{suffix}"


SKIP_CASE_IDS = {
    "llama-hq-mdp": "num_kv_blocks is None in HQ blocked export path",
    "gpt_oss-hq-mdp": "num_kv_blocks is None in HQ blocked export path",
}


def _with_marks(case: BlockingQaicCase):
    marks = []
    if case.num_devices > 1:
        marks.append(pytest.mark.weight_free_multi_device)
    if skip_reason := SKIP_CASE_IDS.get(_case_id(case)):
        marks.append(pytest.mark.skip(reason=skip_reason))
    if case.xfail_reason:
        marks.append(pytest.mark.xfail(reason=case.xfail_reason))
    return pytest.param(case, marks=marks, id=_case_id(case))


@dataclass(frozen=True)
class BlockingModeSpec:
    mode: str
    kwargs: dict
    num_devices: int = 1
    batch_size: int = BATCH_SIZE


@dataclass(frozen=True)
class BlockingModelSpec:
    label: str
    config_factory: Callable[[int], object]


def _mode(mode: str, *, num_devices: int = 1, batch_size: int = BATCH_SIZE, **kwargs) -> BlockingModeSpec:
    return BlockingModeSpec(mode, kwargs, num_devices=num_devices, batch_size=batch_size)


STANDARD_BLOCKING_MODES = (
    _mode("kv", num_kv_blocks=NUM_KV_BLOCKS),
    _mode("h", head_block_size=HEAD_BLOCK_SIZE, num_devices=4),
    _mode("q", num_q_blocks=NUM_Q_BLOCKS),
    _mode("qkv", num_q_blocks=NUM_Q_BLOCKS, num_kv_blocks=NUM_KV_BLOCKS),
    _mode("hq", head_block_size=HEAD_BLOCK_SIZE, num_q_blocks=NUM_Q_BLOCKS, num_devices=4),
    _mode("hkv", head_block_size=HEAD_BLOCK_SIZE, num_kv_blocks=NUM_KV_BLOCKS, num_devices=4),
    _mode(
        "hqkv",
        head_block_size=HEAD_BLOCK_SIZE,
        num_q_blocks=NUM_Q_BLOCKS,
        num_kv_blocks=NUM_KV_BLOCKS,
        num_devices=4,
    ),
    _mode(
        "bhqkv",
        head_block_size=HEAD_BLOCK_SIZE,
        num_q_blocks=NUM_Q_BLOCKS,
        num_kv_blocks=NUM_KV_BLOCKS,
        num_batch_blocks=NUM_BATCH_BLOCKS,
        num_devices=4,
        batch_size=2,
    ),
    _mode("kv_headpar", num_kv_blocks=NUM_KV_BLOCKS, headpar_split=HEADPAR_SPLIT, num_devices=4),
)

MLA_BLOCKING_MODES = (
    _mode(
        "kv",
        num_kv_blocks=NUM_KV_BLOCKS,
        mla_absorption={"absorption": False, "online": False, "cache_compressed": True},
    ),
    _mode(
        "h",
        head_block_size=HEAD_BLOCK_SIZE,
        mla_absorption={"absorption": True, "online": False, "cache_compressed": True},
        num_devices=4,
    ),
)

QWEN3_VL_MOE_SPECIAL_MODES = (
    _mode("prefill_q", num_q_blocks=NUM_Q_BLOCKS),
    _mode("prefill_kv", num_kv_blocks=NUM_KV_BLOCKS, headpar_split=HEADPAR_SPLIT, num_devices=4),
    _mode(
        "prefill_qkv",
        num_q_blocks=NUM_Q_BLOCKS,
        num_kv_blocks=NUM_KV_BLOCKS,
        headpar_split=HEADPAR_SPLIT,
        num_devices=4,
    ),
    _mode("prefill_online", num_q_blocks=NUM_Q_BLOCKS, num_kv_blocks=NUM_KV_BLOCKS, n_rep_chunk=2),
    _mode("kv_batch_fold", num_kv_blocks=NUM_KV_BLOCKS, batch_size=FULL_BATCH_SIZE),
)

QWEN3_VL_MOE_CB_SPECIAL_MODES = (_mode("kv_batch_fold", num_kv_blocks=NUM_KV_BLOCKS, batch_size=FULL_BATCH_SIZE),)

STANDARD_MODEL_SPECS = (
    BlockingModelSpec("llama", _make_tiny_llama_config),
    BlockingModelSpec("gpt_oss", _make_tiny_gpt_oss_config),
)

MLA_MODEL_SPECS = (BlockingModelSpec("kimi_k25_language", _make_kimi_k25_language_config),)

QWEN3_VL_MOE_MODEL_SPEC = BlockingModelSpec("qwen3_vl_moe_text", _make_tiny_qwen3_vl_moe_config)


def _expand_cases(
    model_specs: tuple[BlockingModelSpec, ...],
    mode_specs: tuple[BlockingModeSpec, ...],
    *,
    prompt_len: int = PROMPT_LEN_BLOCKING,
    ctx_len: int = CTX_LEN_BLOCKING,
) -> list:
    cases = []
    for model_spec in model_specs:
        for mode_spec in mode_specs:
            cases.append(
                _with_marks(
                    BlockingQaicCase(
                        model_spec.label,
                        model_spec.config_factory,
                        _qaic_config(mode_spec.mode, **mode_spec.kwargs),
                        num_devices=mode_spec.num_devices,
                        batch_size=mode_spec.batch_size,
                        prompt_len=prompt_len,
                        ctx_len=ctx_len,
                    )
                )
            )
    return cases


BLOCKING_QAIC_CASES = [
    *_expand_cases(STANDARD_MODEL_SPECS, STANDARD_BLOCKING_MODES),
    *_expand_cases(MLA_MODEL_SPECS, MLA_BLOCKING_MODES),
    *_expand_cases((QWEN3_VL_MOE_MODEL_SPEC,), QWEN3_VL_MOE_SPECIAL_MODES, prompt_len=64, ctx_len=512),
]

CB_BLOCKING_QAIC_CASES = [
    *_expand_cases(STANDARD_MODEL_SPECS, STANDARD_BLOCKING_MODES),
    *_expand_cases(MLA_MODEL_SPECS, MLA_BLOCKING_MODES),
    *_expand_cases((QWEN3_VL_MOE_MODEL_SPEC,), QWEN3_VL_MOE_CB_SPECIAL_MODES, prompt_len=64, ctx_len=512),
]


def _load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _save_tiny_checkpoint(case: BlockingQaicCase, tmp_path: Path):
    tokenizer = _load_tokenizer()
    config = case.config_factory(max(len(tokenizer), VOCAB_SIZE_FLOOR))
    config.torch_dtype = torch.float32

    model_hf = AutoModelForCausalLM.from_config(config, trust_remote_code=True, attn_implementation="eager").eval()
    checkpoint_dir = tmp_path / f"{_case_id(case)}_checkpoint"
    model_hf.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    return checkpoint_dir, model_hf, tokenizer


def _build_meta_qeff_from_checkpoint(checkpoint_dir: Path, *, continuous_batching: bool):
    config = AutoConfig.from_pretrained(checkpoint_dir, trust_remote_code=True)
    with init_empty_weights():
        meta_model = AutoModelForCausalLM.from_config(config, trust_remote_code=True, attn_implementation="eager")
    return QEFFAutoModelForCausalLM(
        meta_model,
        pretrained_model_name_or_path=str(checkpoint_dir),
        continuous_batching=continuous_batching,
    )


def _compile_and_check_blocking(qeff_model, case: BlockingQaicCase, tmp_export_dir: Path, *, continuous_batching: bool):
    compile_dir = tmp_export_dir / f"{_case_id(case)}_{'cb' if continuous_batching else 'decode'}"
    compile_kwargs = {}
    if continuous_batching:
        compile_kwargs["full_batch_size"] = FULL_BATCH_SIZE

    qeff_model.compile(
        compile_dir=str(compile_dir),
        prefill_seq_len=case.prompt_len,
        ctx_len=case.ctx_len,
        num_cores=16,
        num_devices=case.num_devices,
        batch_size=case.batch_size,
        qaic_config=copy.deepcopy(case.qaic_config),
        user_tiled=True,
        use_weight_free_export=True,
        use_onnx_subfunctions=True,
        **compile_kwargs,
    )
    assert_blocked_kv_ops_for_mode(
        exported_onnx_path(qeff_model.onnx_path),
        qeff_model,
        case.qaic_config["blocking_mode"],
        continuous_batching=continuous_batching,
    )
    return compile_dir


def _assert_generate_parity(
    case: BlockingQaicCase, qeff_model, model_hf, tokenizer, prompts, *, full_batch_size: int | None = None
):
    hf_tokens = get_hf_tokens(
        tokenizer,
        model_hf,
        prompts,
        prompt_len=case.prompt_len,
        ctx_len=case.ctx_len,
        full_batch_size=full_batch_size,
    )
    output = qeff_model.generate(
        tokenizer=tokenizer,
        prompts=prompts,
        device_id=get_available_device_id(),
    )
    assert output.generated_texts is not None
    if full_batch_size is not None:
        assert len(output.generated_texts) == full_batch_size
    assert_hf_hw_parity(
        str(getattr(model_hf.config, "model_type", "tiny")),
        hf_tokens,
        output,
        gen_len=case.ctx_len - case.prompt_len,
        full_batch_size=full_batch_size,
        context="tiny blocking",
    )


@pytest.mark.weight_free
@pytest.mark.on_qaic
@pytest.mark.xdist_group(name="qaic-runtime")
@pytest.mark.llm_model
@pytest.mark.parametrize("case", BLOCKING_QAIC_CASES)
def test_weight_free_tiny_blocking_compile_and_generate(case, tmp_export_dir, tmp_path):
    checkpoint_dir, model_hf, tokenizer = _save_tiny_checkpoint(case, tmp_path)
    qeff_model = _build_meta_qeff_from_checkpoint(checkpoint_dir, continuous_batching=False)
    compile_dir = _compile_and_check_blocking(qeff_model, case, tmp_export_dir, continuous_batching=False)

    if case.num_devices > 1:
        assert compile_dir.is_dir()
        return

    _assert_generate_parity(case, qeff_model, model_hf, tokenizer, [PROMPT])


@pytest.mark.weight_free
@pytest.mark.on_qaic
@pytest.mark.xdist_group(name="qaic-runtime")
@pytest.mark.llm_model
@pytest.mark.parametrize("case", CB_BLOCKING_QAIC_CASES)
def test_weight_free_tiny_cb_blocking_compile_and_generate(case, tmp_export_dir, tmp_path):
    checkpoint_dir, model_hf, tokenizer = _save_tiny_checkpoint(case, tmp_path)
    qeff_model = _build_meta_qeff_from_checkpoint(checkpoint_dir, continuous_batching=True)
    compile_dir = _compile_and_check_blocking(qeff_model, case, tmp_export_dir, continuous_batching=True)

    if case.num_devices > 1:
        assert compile_dir.is_dir()
        return

    _assert_generate_parity(case, qeff_model, model_hf, tokenizer, CB_PROMPTS, full_batch_size=FULL_BATCH_SIZE)
