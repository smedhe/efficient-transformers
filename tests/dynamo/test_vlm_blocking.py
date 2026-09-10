# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Dynamo blocking QAIC tests for image-text-to-text blocking paths."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pytest
import torch
from transformers import AutoModelForImageTextToText

from QEfficient.blocking.attention_blocking import BlockingMode
from QEfficient.generation.cloud_infer import QAICInferenceSession, is_retained_state_name
from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForImageTextToText

from ._helpers import DYNAMO, assert_blocked_kv_ops_for_mode, exported_onnx_path
from .test_blocking_tiny_models import (
    FULL_BATCH_SIZE,
    HEAD_BLOCK_SIZE,
    HEADPAR_SPLIT,
    NUM_KV_BLOCKS,
    NUM_Q_BLOCKS,
    VOCAB_SIZE_FLOOR,
    _get_device_ids,
    _make_tiny_qwen3_vl_moe_config,
    _qaic_config,
)

QWEN3_VL_MOE_PREFILL_LEN = 64
QWEN3_VL_MOE_CTX_LEN = 512
KIMI_K25_CTX_LEN = 512
IMAGE_SIZE = (536, 354)


@dataclass(frozen=True)
class VlmBlockingQaicCase:
    model_label: str
    model_factory: Callable[[], torch.nn.Module]
    qaic_config: dict
    expected_mode: BlockingMode
    prompt_len: int
    ctx_len: int
    batch_size: int = 1
    num_devices: int = 1
    prefill_only: bool = False
    enable_chunking: bool = False
    retain_full_kv: bool = False
    expected_onnx_marker_key: str | None = None


def _case_id(case: VlmBlockingQaicCase) -> str:
    mode = case.qaic_config["blocking_mode"]
    suffix = "-mdp" if case.num_devices > 1 else ""
    return f"{case.model_label}-{mode}{suffix}"


def _make_kimi_k25_vlm_model():
    from tests.utils.load_kimi_utils import (
        KIMI_K25_MODEL_NAME,
        get_kimi_k25_test_config,
        load_kimi_k25_model_from_config,
    )

    config_path = Path(__file__).parents[1] / "configs" / "image_text_model_configs.json"
    model_configs = json.loads(config_path.read_text())["image_text_models"]
    model_config_dict = {model["model_name"]: model for model in model_configs}
    config = get_kimi_k25_test_config(KIMI_K25_MODEL_NAME, model_config_dict)
    config.text_config.vocab_size = VOCAB_SIZE_FLOOR
    config.text_config.max_position_embeddings = KIMI_K25_CTX_LEN
    model_hf, _, _ = load_kimi_k25_model_from_config(config)
    return model_hf.eval()


def _make_qwen3_vl_moe_vlm_model():
    pytest.importorskip("qwen_vl_utils")
    config = _make_tiny_qwen3_vl_moe_config(VOCAB_SIZE_FLOOR)
    config._attn_implementation = "eager"
    config.text_config._attn_implementation = "eager"
    config.text_config.max_position_embeddings = QWEN3_VL_MOE_CTX_LEN
    torch.manual_seed(42)
    return AutoModelForImageTextToText.from_config(config, attn_implementation="eager").eval()


def _with_marks(case: VlmBlockingQaicCase):
    marks = []
    if case.num_devices > 1:
        marks.append(pytest.mark.dynamo_multi_device)
    return pytest.param(case, marks=marks, id=_case_id(case))


def _compile_kwargs(case: VlmBlockingQaicCase, compile_dir: Path) -> dict:
    kwargs = {
        "compile_dir": str(compile_dir),
        "batch_size": case.batch_size,
        "prefill_seq_len": case.prompt_len,
        "ctx_len": case.ctx_len,
        "height": IMAGE_SIZE[1],
        "width": IMAGE_SIZE[0],
        "image_height": IMAGE_SIZE[1],
        "image_width": IMAGE_SIZE[0],
        "num_cores": 16,
        "num_devices": case.num_devices,
        "split_model_io": True,
        "mos": 1,
        "aic_enable_depth_first": True,
        "skip_vision": True,
        "skip_lang": False,
        "use_onnx_subfunctions": True,
        "layerwise": False,
        "dynamo": DYNAMO,
        "qaic_config": copy.deepcopy(case.qaic_config),
        "prefill_only": case.prefill_only,
        "enable_chunking": case.enable_chunking,
        "retain_full_kv": case.retain_full_kv,
    }
    if case.model_label.startswith("kimi"):
        kwargs.pop("height")
        kwargs.pop("width")
    else:
        kwargs.pop("image_height")
        kwargs.pop("image_width")
    return kwargs


def _assert_blocking_config(qeff_model: QEFFAutoModelForImageTextToText, case: VlmBlockingQaicCase):
    blocking_config = qeff_model.lang_model.hash_params.get("blocking_kwargs")
    assert blocking_config is not None
    assert blocking_config.mode == case.expected_mode

    attached_configs = [
        module.attn_blocking_config
        for module in qeff_model.lang_model.model.modules()
        if hasattr(module, "attn_blocking_config")
    ]
    assert attached_configs, "Expected blocking config on at least one language attention module"
    assert all(config is blocking_config for config in attached_configs)

    expected_mla_absorption = case.qaic_config.get("mla_absorption")
    if expected_mla_absorption is not None:
        language_model = getattr(qeff_model.lang_model.model, "language_model", None)
        assert language_model is not None
        assert getattr(language_model, "mla_absorption", None) == expected_mla_absorption


def _clone_tensor_tree(value):
    if torch.is_tensor(value):
        return value.clone()
    if isinstance(value, list):
        return [_clone_tensor_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_tensor_tree(item) for item in value)
    if isinstance(value, dict):
        return {key: _clone_tensor_tree(item) for key, item in value.items()}
    return copy.deepcopy(value)


def _get_output_logits(outputs):
    if hasattr(outputs, "logits"):
        return outputs.logits
    if isinstance(outputs, dict):
        return outputs["logits"]
    return outputs[0]


def _make_dummy_lang_inputs(qeff_model: QEFFAutoModelForImageTextToText, case: VlmBlockingQaicCase):
    dummy_inputs = qeff_model.model.get_dummy_inputs(
        kv_offload=True,
        batch_size=case.batch_size,
        prefill_seq_len=case.prompt_len,
    )
    return dummy_inputs["lang"]


def _flatten_qaic_inputs(inputs: dict) -> dict[str, np.ndarray]:
    flat_inputs = {}
    for name, value in inputs.items():
        if name == "past_key_values":
            for layer_idx, layer_cache in enumerate(value):
                flat_inputs[f"past_key.{layer_idx}"] = layer_cache[0].detach().cpu().numpy()
                flat_inputs[f"past_value.{layer_idx}"] = layer_cache[1].detach().cpu().numpy()
        elif name == "compressed_kvs":
            for layer_idx, layer_cache in enumerate(value):
                flat_inputs[f"compressed_kv.{layer_idx}"] = layer_cache[0].detach().cpu().numpy()
                flat_inputs[f"k_pe.{layer_idx}"] = layer_cache[1].detach().cpu().numpy()
        elif torch.is_tensor(value):
            flat_inputs[name] = value.detach().cpu().numpy()
    return flat_inputs


def _session_input_names(session: QAICInferenceSession) -> set[str]:
    input_names = set(session.input_names)
    input_names.update(name.rsplit("/", 1)[-1] for name in session.input_names)
    return input_names


def _cast_for_session(session: QAICInferenceSession, name: str, value: np.ndarray) -> np.ndarray:
    binding_index = session.binding_index_map.get(name)
    if binding_index is None:
        return value
    dtype = session.aic_to_np_dtype_mapping[session.bindings[binding_index].type]
    return value.astype(dtype, copy=False)


def _filter_session_inputs(session: QAICInferenceSession, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    input_names = _session_input_names(session)
    return {
        name: _cast_for_session(session, name, value)
        for name, value in inputs.items()
        if name in input_names and not is_retained_state_name(name)
    }


def _set_logits_buffer(session: QAICInferenceSession, logits: np.ndarray):
    binding_index = session.binding_index_map.get("logits")
    if binding_index is None:
        return
    dtype = session.aic_to_np_dtype_mapping[session.bindings[binding_index].type]
    session.set_buffers({"logits": np.zeros(logits.shape, dtype=dtype)})


def _assert_logits_close(label: str, expected: np.ndarray, actual: np.ndarray, *, atol: float) -> None:
    assert expected.shape == actual.shape, (
        f"{label} logits shape mismatch: expected={expected.shape}, actual={actual.shape}"
    )
    diff = np.abs(expected - actual)
    max_diff = float(diff.max())
    if max_diff >= atol:
        max_idx = np.unravel_index(np.argmax(diff), diff.shape)
        assert False, (
            f"{label} logits diverged: shape={expected.shape}, max_abs_diff={max_diff}, "
            f"mean_abs_diff={float(diff.mean())}, max_diff_index={max_idx}, atol={atol}"
        )


@torch.no_grad()
def _assert_lang_qpc_logits_parity(qeff_model: QEFFAutoModelForImageTextToText, case: VlmBlockingQaicCase):
    lang_inputs = _make_dummy_lang_inputs(qeff_model, case)
    qeff_outputs = qeff_model.lang_model.model(**_clone_tensor_tree(lang_inputs))
    qeff_logits = _get_output_logits(qeff_outputs).detach().float().cpu().numpy()

    session = QAICInferenceSession(str(qeff_model.lang_model.qpc_path), _get_device_ids(case))
    session.skip_buffers(
        [
            name
            for name in session.input_names + session.output_names
            if is_retained_state_name(name) or name.endswith("_RetainedState")
        ]
    )
    try:
        _set_logits_buffer(session, qeff_logits)
        qaic_outputs = session.run(_filter_session_inputs(session, _flatten_qaic_inputs(lang_inputs)))
    finally:
        session.deactivate()

    _assert_logits_close(f"{_case_id(case)} QAIC vs QEff PyTorch", qeff_logits, qaic_outputs["logits"], atol=5e-2)


QWEN3_VL_MOE_CASES = [
    _with_marks(
        VlmBlockingQaicCase(
            "qwen3_vl_moe",
            _make_qwen3_vl_moe_vlm_model,
            _qaic_config("prefill_q", num_q_blocks=NUM_Q_BLOCKS),
            BlockingMode.PREFILL_Q,
            prompt_len=QWEN3_VL_MOE_PREFILL_LEN,
            ctx_len=QWEN3_VL_MOE_CTX_LEN,
            prefill_only=True,
            enable_chunking=True,
            retain_full_kv=True,
        )
    ),
    _with_marks(
        VlmBlockingQaicCase(
            "qwen3_vl_moe",
            _make_qwen3_vl_moe_vlm_model,
            _qaic_config("prefill_kv", num_kv_blocks=NUM_KV_BLOCKS, headpar_split=HEADPAR_SPLIT),
            BlockingMode.PREFILL_KV,
            prompt_len=QWEN3_VL_MOE_PREFILL_LEN,
            ctx_len=QWEN3_VL_MOE_CTX_LEN,
            num_devices=4,
            prefill_only=True,
            enable_chunking=True,
            retain_full_kv=True,
            expected_onnx_marker_key="kv",
        )
    ),
    _with_marks(
        VlmBlockingQaicCase(
            "qwen3_vl_moe",
            _make_qwen3_vl_moe_vlm_model,
            _qaic_config(
                "prefill_qkv",
                num_q_blocks=NUM_Q_BLOCKS,
                num_kv_blocks=NUM_KV_BLOCKS,
                headpar_split=HEADPAR_SPLIT,
            ),
            BlockingMode.PREFILL_QKV,
            prompt_len=QWEN3_VL_MOE_PREFILL_LEN,
            ctx_len=QWEN3_VL_MOE_CTX_LEN,
            num_devices=4,
            prefill_only=True,
            enable_chunking=True,
            retain_full_kv=True,
            expected_onnx_marker_key="kv",
        )
    ),
    _with_marks(
        VlmBlockingQaicCase(
            "qwen3_vl_moe",
            _make_qwen3_vl_moe_vlm_model,
            _qaic_config("prefill_online", num_q_blocks=NUM_Q_BLOCKS, num_kv_blocks=NUM_KV_BLOCKS, n_rep_chunk=2),
            BlockingMode.PREFILL_ONLINE,
            prompt_len=QWEN3_VL_MOE_PREFILL_LEN,
            ctx_len=QWEN3_VL_MOE_CTX_LEN,
            prefill_only=True,
            enable_chunking=True,
            retain_full_kv=True,
            expected_onnx_marker_key="kv",
        )
    ),
    _with_marks(
        VlmBlockingQaicCase(
            "qwen3_vl_moe",
            _make_qwen3_vl_moe_vlm_model,
            _qaic_config("kv_batch_fold", num_kv_blocks=NUM_KV_BLOCKS),
            BlockingMode.KV_BATCH_FOLD,
            prompt_len=1,
            ctx_len=QWEN3_VL_MOE_CTX_LEN,
            batch_size=FULL_BATCH_SIZE,
            expected_onnx_marker_key="kv_batch_fold",
        )
    ),
]

KIMI_K25_MLA_CASES = [
    _with_marks(
        VlmBlockingQaicCase(
            "kimi_k25",
            _make_kimi_k25_vlm_model,
            _qaic_config(
                "kv",
                num_kv_blocks=NUM_KV_BLOCKS,
                mla_absorption={"absorption": False, "online": False, "cache_compressed": True},
            ),
            BlockingMode.KV,
            prompt_len=1,
            ctx_len=KIMI_K25_CTX_LEN,
        )
    ),
    _with_marks(
        VlmBlockingQaicCase(
            "kimi_k25",
            _make_kimi_k25_vlm_model,
            _qaic_config(
                "h",
                head_block_size=HEAD_BLOCK_SIZE,
                mla_absorption={"absorption": True, "online": False, "cache_compressed": True},
            ),
            BlockingMode.H,
            prompt_len=1,
            ctx_len=KIMI_K25_CTX_LEN,
            num_devices=4,
        )
    ),
]


@pytest.mark.dynamo
@pytest.mark.on_qaic
@pytest.mark.multimodal
@pytest.mark.xdist_group(name="qaic-runtime")
@pytest.mark.parametrize("case", QWEN3_VL_MOE_CASES + KIMI_K25_MLA_CASES)
def test_dynamo_vlm_blocking_compile(case: VlmBlockingQaicCase, tmp_export_dir):
    qeff_model = QEFFAutoModelForImageTextToText(
        case.model_factory(),
        kv_offload=True,
        qaic_config=copy.deepcopy(case.qaic_config),
        torch_dtype=torch.float32,
        layerwise=False,
    )

    qpc_paths = qeff_model.compile(**_compile_kwargs(case, tmp_export_dir / _case_id(case)))
    assert qpc_paths
    assert Path(qpc_paths["lang_prefill_qpc_path" if case.prefill_only else "lang_decode_qpc_path"]).is_dir()

    _assert_blocking_config(qeff_model, case)
    onnx_path = exported_onnx_path(qeff_model.lang_model.onnx_path)
    if case.expected_onnx_marker_key is not None:
        assert_blocked_kv_ops_for_mode(onnx_path, qeff_model.lang_model, case.expected_onnx_marker_key)
    _assert_lang_qpc_logits_parity(qeff_model, case)
