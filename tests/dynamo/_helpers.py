# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Shared helpers, model registry, and constants for tests/dynamo/.

Model IDs follow the same plain-dict pattern as CAUSAL_RUNTIME_MODEL_IDS
in tests/unit_test/models/test_model_quickcheck.py.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict, Tuple

import onnx
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM

# ---------------------------------------------------------------------------
# Worker-level model cache — from_pretrained runs once per (model_id, dtype)
# per worker. Tests receive a deepcopy so weight offload or transforms in one
# test do not affect other tests that share the same cached instance.
# ---------------------------------------------------------------------------
_HF_MODEL_CACHE: Dict[Tuple[str, torch.dtype], Tuple[AutoModelForCausalLM, AutoTokenizer]] = {}

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

DYNAMO_CAUSAL_LM_MODEL_IDS = {
    "codegen": "hf-internal-testing/tiny-random-CodeGenForCausalLM",
    # deepseek_v3: transformers version in this env is missing is_torch_fx_available;
    # excluded until the env is updated to a compatible transformers release.
    "falcon": "hf-internal-testing/tiny-random-FalconForCausalLM",
    "gemma": "Xenova/tiny-random-GemmaForCausalLM",
    "gemma2": "hf-internal-testing/tiny-random-Gemma2ForCausalLM",
    "glm4_moe": "tiny-random/glm-4-moe",
    "gpt2": "hf-internal-testing/tiny-random-GPT2LMHeadModel",
    "gpt_bigcode": "hf-internal-testing/tiny-random-GPTBigCodeForCausalLM",
    "gpt_oss": "tiny-random/gpt-oss-bf16",
    "gptj": "hf-internal-testing/tiny-random-GPTJForCausalLM",
    "granite": "hf-internal-testing/tiny-random-GraniteForCausalLM",
    "granitemoe": "hf-internal-testing/tiny-random-GraniteMoeForCausalLM",
    # grok_1: tiny random model config is malformed for QEff (AttributeError on keys());
    # excluded until a valid tiny checkpoint is available.
    "llama": "hf-internal-testing/tiny-random-LlamaForCausalLM",
    # llama_swiftkv requires QEffLlamaSwiftKVConfig + QEffLlamaSwiftKVForCausalLM,
    # not AutoModelForCausalLM. No tiny random model exists on HF Hub.
    # Tested via check_causal_models.py with the full Snowflake model.
    "mistral": "hf-internal-testing/tiny-random-MistralForCausalLM",
    "mixtral": "hf-internal-testing/tiny-random-MixtralForCausalLM",
    "mpt": "hf-internal-testing/tiny-random-MptForCausalLM",
    "olmo2": "hf-internal-testing/tiny-random-Olmo2ForCausalLM",
    "phi": "hf-internal-testing/tiny-random-PhiForCausalLM",
    # "phi3": "tiny-random/phi-4", #TODO: need to fix the SplitToSequence issue
    "qwen2": "yujiepan/qwen2-tiny-random",
    "qwen3": "tiny-random/qwen3",
    "qwen3_moe": "tiny-random/qwen3-moe",
    "starcoder2": "hf-internal-testing/tiny-random-Starcoder2ForCausalLM",
}

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROMPT_LEN = 8
CTX_LEN = 16
BATCH_SIZE = 1
FULL_BATCH_SIZE = 4
MODEL_KWARGS = {"attn_implementation": "eager", "low_cpu_mem_usage": False}

# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------


def skip_on_model_fetch_error(exc: Exception, model_id: str) -> None:
    pytest.skip(
        f"Skipping {model_id}: model unavailable or unsupported in this environment ({type(exc).__name__}: {exc})"
    )


def load_hf_model(model_id: str, torch_dtype: torch.dtype) -> AutoModelForCausalLM:
    key = (model_id, torch_dtype)
    if key not in _HF_MODEL_CACHE:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            **MODEL_KWARGS,
        )
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if not hasattr(tokenizer, "pad_token") or tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        _HF_MODEL_CACHE[key] = (model, tokenizer)
    model, _ = _HF_MODEL_CACHE[key]
    return copy.deepcopy(model)


def load_tokenizer(model_id: str, torch_dtype: torch.dtype) -> AutoTokenizer:
    key = (model_id, torch_dtype)
    if key not in _HF_MODEL_CACHE:
        load_hf_model(model_id, torch_dtype=torch_dtype)
    _, tokenizer = _HF_MODEL_CACHE[key]
    return tokenizer


def exported_onnx_path(export_result) -> Path:
    if isinstance(export_result, (list, tuple)):
        export_result = export_result[-1]
    onnx_path = Path(export_result)
    assert onnx_path.is_file(), f"Expected ONNX file at {onnx_path}"
    return onnx_path


# ---------------------------------------------------------------------------
# Dynamo ONNX export cache — shared across all tests/dynamo/ test modules.
#
# dynamo=True + use_onnx_subfunctions=True export is expensive (torch.export
# graph capture + subfunction extraction), and multiple test modules
# (test_on_qaic.py, test_ccl.py) each need the same basic/continuous_batching
# export per model to then compile it under different flags (single vs
# multi-device, with vs without CCL specializations). Key on
# (model_id, torch_dtype, continuous_batching, ccl_enabled) -- the only
# export-time knobs that change the resulting graph/weights. Compile-time-only
# flags (mxfp6_matmul, num_devices, comp_ctx_lengths_prefill/decode, ...) must
# NOT be part of this key since they don't affect the exported ONNX. Note the
# QAIC compiler itself never emits an fp32 QPC -- CUSTOM_IO_DTYPE_MAP maps
# torch.float32 to "float16" same as torch.float16 (see modeling_auto.py), so
# convert_to_fp16 is derived from the PyTorch model's dtype, not requested at
# compile(); torch_dtype here changes the exported ONNX *weights*, not what
# the compiler outputs.
# ---------------------------------------------------------------------------
_DYNAMO_ONNX_CACHE: Dict[Tuple[str, torch.dtype, bool, bool], Tuple[str, "QEFFAutoModelForCausalLM"]] = {}


def get_dynamo_export(
    model_id: str,
    tmp_path_factory,
    *,
    torch_dtype: torch.dtype,
    continuous_batching: bool = False,
    ccl_enabled: bool = False,
) -> Tuple[str, QEFFAutoModelForCausalLM]:
    """Export once per (model_id, torch_dtype, continuous_batching, ccl_enabled) combination
    and cache the (onnx_path, qeff_model) pair for reuse by every test that needs that exact export.

    Callers still perform their own compile()/generate() against the returned qeff_model --
    only the (expensive) export step is shared.
    """
    key = (model_id, torch_dtype, continuous_batching, ccl_enabled)
    if key in _DYNAMO_ONNX_CACHE:
        return _DYNAMO_ONNX_CACHE[key]

    model_hf = load_hf_model(model_id, torch_dtype=torch_dtype)
    kwargs: Dict[str, object] = {}
    if continuous_batching:
        kwargs["continuous_batching"] = True
    if ccl_enabled:
        kwargs["qaic_config"] = {"ccl_enabled": True}

    qeff_model = QEFFAutoModelForCausalLM(model_hf, **kwargs)

    export_dir = tmp_path_factory.mktemp("dynamo_export", numbered=True)
    onnx_path = exported_onnx_path(
        qeff_model.export(
            export_dir,
            dynamo=True,
            use_onnx_subfunctions=True,
        )
    )
    _DYNAMO_ONNX_CACHE[key] = (str(onnx_path), qeff_model)
    return _DYNAMO_ONNX_CACHE[key]


# ---------------------------------------------------------------------------
# ONNX assertion helpers
# ---------------------------------------------------------------------------


def assert_has_subfunctions(onnx_path: Path, qeff_model: QEFFAutoModelForCausalLM) -> None:
    """Assert the ONNX graph contains at least one decoder-block subfunction.

    CtxScatter/CtxGather/CustomRMSNorm always appear as functions regardless of
    use_onnx_subfunctions, so checking len(model.functions) > 0 is not sufficient.
    We require at least one function whose name contains a decoder class name from
    get_submodules_for_export(), matching the main suite's approach.
    """
    get_submodules = getattr(qeff_model.model, "get_submodules_for_export", None)
    if not callable(get_submodules):
        return  # Model doesn't declare submodule boundaries — skip check

    submodule_classes = get_submodules()
    if not submodule_classes:
        return

    decoder_names = {
        cls.__name__
        for cls in (submodule_classes if isinstance(submodule_classes, (set, list, tuple)) else [submodule_classes])
    }

    model = onnx.load(str(onnx_path), load_external_data=False)
    found = [fn.name for fn in model.functions if any(d in fn.name for d in decoder_names)]
    assert found, (
        f"Expected decoder-block subfunctions ({decoder_names}) in {onnx_path.name} but found none. "
        f"Functions present: {[fn.name for fn in model.functions]}"
    )


def assert_subfunction_names_match_decoder_class(onnx_path: Path, qeff_model: QEFFAutoModelForCausalLM) -> None:
    """Verify RenameRepeatedSubgraphTransform renamed functions to decoder class names."""
    get_submodules = getattr(qeff_model.model, "get_submodules_for_export", None)
    if not callable(get_submodules):
        return  # Model doesn't declare submodule boundaries — skip name check

    submodule_classes = get_submodules()
    if not submodule_classes:
        return

    expected_names = {
        cls.__name__
        for cls in (submodule_classes if isinstance(submodule_classes, (set, list, tuple)) else [submodule_classes])
    }

    model = onnx.load(str(onnx_path), load_external_data=False)
    for fn in model.functions:
        assert not any(fn.name.startswith(pat) for pat in ("repeated_subgraph", "subgraph_", "invoke_subgraph_")), (
            f"Function '{fn.name}' still has raw dynamo name — "
            f"RenameRepeatedSubgraphTransform did not rename it. "
            f"Expected a name derived from {expected_names}."
        )


def assert_retained_state_outputs(onnx_path: Path, expected_count: int) -> None:
    """Assert that the ONNX graph has the expected number of _RetainedState outputs."""
    model = onnx.load(str(onnx_path), load_external_data=False)
    retained = [o for o in model.graph.output if o.name.endswith("_RetainedState")]
    assert len(retained) == expected_count, (
        f"Expected {expected_count} _RetainedState outputs, got {len(retained)}: {[o.name for o in retained]}"
    )
