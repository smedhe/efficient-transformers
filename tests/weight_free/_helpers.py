# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Shared helpers, model registry, and constants for tests/weight_free/.

Model IDs are the same tiny-random checkpoints used by tests/dynamo/ so
both suites exercise the same model families.
"""

from __future__ import annotations

import copy
from collections import Counter
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import onnx
import onnxruntime
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from QEfficient.exporter.weight_free import load_weight_free_ort_inputs
from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM

# ---------------------------------------------------------------------------
# Worker-level model cache
# ---------------------------------------------------------------------------
_HF_MODEL_CACHE: Dict[str, Tuple[AutoModelForCausalLM, AutoTokenizer]] = {}
_HF_TOKEN_CACHE: Dict[Tuple[str, str, Tuple[str, ...], int, int, int, int | None], object] = {}

# Export cache shared by QAIC weight-free tests. The key only includes export-time
# knobs; compile-time options still get their own QPCs per test.
_WEIGHT_FREE_ONNX_CACHE: Dict[Tuple[str, int, bool, bool, bool], Tuple[str, "QEFFAutoModelForCausalLM"]] = {}

# ---------------------------------------------------------------------------
# Model registry — same tiny-random models as tests/dynamo/
# ---------------------------------------------------------------------------

WEIGHT_FREE_CAUSAL_LM_MODEL_IDS = {
    "codegen": "hf-internal-testing/tiny-random-CodeGenForCausalLM",
    # deepseek_v3 needs a newer compatible transformers environment.
    "falcon": "hf-internal-testing/tiny-random-FalconForCausalLM",
    "gemma": "Xenova/tiny-random-GemmaForCausalLM",
    "gemma2": "hf-internal-testing/tiny-random-Gemma2ForCausalLM",
    "glm4_moe": "tiny-random/glm-4-moe",
    "gpt2": "hf-internal-testing/tiny-random-GPT2LMHeadModel",
    "gpt_bigcode": "hf-internal-testing/tiny-random-GPTBigCodeForCausalLM",
    "gpt_oss": "tiny-random/gpt-oss-mxfp4",
    "gptj": "hf-internal-testing/tiny-random-GPTJForCausalLM",
    "granite": "hf-tiny-v2/tiny-random-GraniteForCausalLM",
    "granitemoe": "hf-tiny-v2/tiny-random-GraniteMoeForCausalLM",
    # grok_1 tiny config is not supported in legacy.
    "llama": "hf-internal-testing/tiny-random-LlamaForCausalLM",
    # llama_swiftkv is not AutoModelForCausalLM-compatible.
    "mistral": "hf-internal-testing/tiny-random-MistralForCausalLM",
    "mixtral": "hf-internal-testing/tiny-random-MixtralForCausalLM",
    "mpt": "hf-internal-testing/tiny-random-MptForCausalLM",
    "olmo2": "hf-internal-testing/tiny-random-Olmo2ForCausalLM",
    "phi": "hf-internal-testing/tiny-random-PhiForCausalLM",
    # phi3 is disabled until SplitToSequence is fixed.
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
MODEL_KWARGS = {"attn_implementation": "eager", "low_cpu_mem_usage": False, "dtype": torch.float32}

# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------


def load_hf_model(model_id: str) -> AutoModelForCausalLM:
    if model_id not in _HF_MODEL_CACHE:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            **MODEL_KWARGS,
        )
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if not hasattr(tokenizer, "pad_token") or tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        _HF_MODEL_CACHE[model_id] = (model, tokenizer)
    model, _ = _HF_MODEL_CACHE[model_id]
    return copy.deepcopy(model)


def load_tokenizer(model_id: str) -> AutoTokenizer:
    if model_id not in _HF_MODEL_CACHE:
        load_hf_model(model_id)
    _, tokenizer = _HF_MODEL_CACHE[model_id]
    return tokenizer


def _hf_token_cache_key(tokenizer, model_hf, prompts, prompt_len, ctx_len, batch_size, full_batch_size):
    model_name = getattr(model_hf.config, "_name_or_path", "") or getattr(model_hf.config, "name_or_path", "")
    tokenizer_name = getattr(tokenizer, "name_or_path", "")
    return (
        model_name,
        tokenizer_name,
        tuple(prompts),
        prompt_len,
        ctx_len,
        batch_size,
        full_batch_size,
    )


def get_hf_tokens(
    tokenizer,
    model_hf,
    prompts,
    *,
    prompt_len: int,
    ctx_len: int,
    batch_size: int = BATCH_SIZE,
    full_batch_size: int | None = None,
):
    from QEfficient.utils.run_utils import ApiRunner

    key = _hf_token_cache_key(tokenizer, model_hf, prompts, prompt_len, ctx_len, batch_size, full_batch_size)
    if key in _HF_TOKEN_CACHE:
        return copy.deepcopy(_HF_TOKEN_CACHE[key])

    api_runner = ApiRunner(
        batch_size=batch_size,
        tokenizer=tokenizer,
        config=model_hf.config,
        prompt=prompts,
        prompt_len=prompt_len,
        ctx_len=ctx_len,
        full_batch_size=full_batch_size,
    )
    if full_batch_size is None:
        hf_tokens = api_runner.run_hf_model_on_pytorch(model_hf)
        assert hf_tokens is not None, "HF PT inference returned None"
        _HF_TOKEN_CACHE[key] = copy.deepcopy(hf_tokens)
        return copy.deepcopy(hf_tokens)

    hf_tokens = api_runner.run_hf_model_on_pytorch_CB(model_hf)
    assert hf_tokens is not None, "HF PT CB inference returned None"
    _HF_TOKEN_CACHE[key] = copy.deepcopy(hf_tokens)
    return copy.deepcopy(hf_tokens)


def assert_hf_hw_parity(
    model_id: str,
    hf_tokens,
    qaic_output,
    *,
    gen_len: int,
    full_batch_size: int | None = None,
    context: str = "",
) -> None:
    assert qaic_output is not None, "QAIC generate returned None"
    assert hasattr(qaic_output, "generated_ids"), "QAIC generate did not return generated_ids"
    assert qaic_output.generated_ids is not None, "QAIC generate returned generated_ids=None"

    label = f" {context}" if context else ""
    if full_batch_size is None:
        qaic_tokens = qaic_output.generated_ids[0].flatten()[:gen_len]
        if not np.array_equal(hf_tokens, qaic_tokens):
            assert False, (
                f"HF AIC HW{label} parity failed for {model_id}: HF={hf_tokens.tolist()}, QAIC={qaic_tokens.tolist()}"
            )
        return

    assert len(hf_tokens) == full_batch_size
    for batch_idx in range(full_batch_size):
        hf_batch_tokens = np.asarray(hf_tokens[batch_idx]).flatten()[:gen_len]
        qaic_batch_tokens = qaic_output.generated_ids[batch_idx].flatten()[:gen_len]
        if not np.array_equal(hf_batch_tokens, qaic_batch_tokens):
            assert False, (
                f"HF AIC HW{label} CB parity failed for {model_id} batch {batch_idx}: "
                f"HF={hf_batch_tokens.tolist()}, QAIC={qaic_batch_tokens.tolist()}"
            )


def get_weight_free_export(
    model_id: str,
    tmp_path_factory,
    *,
    num_hidden_layers: int = 2,
    continuous_batching: bool = False,
    qaic_config: dict | None = None,
    model: AutoModelForCausalLM | None = None,
) -> Tuple[str, QEFFAutoModelForCausalLM]:
    """Export once per compatible weight-free graph shape and cache the ONNX path."""
    key = (model_id, num_hidden_layers, continuous_batching, bool(qaic_config), model is not None)
    if key in _WEIGHT_FREE_ONNX_CACHE:
        return _WEIGHT_FREE_ONNX_CACHE[key]

    kwargs: Dict[str, object] = {}
    if continuous_batching:
        kwargs["continuous_batching"] = True
    if qaic_config:
        kwargs["qaic_config"] = qaic_config

    export_dir = tmp_path_factory.mktemp("weight_free_export", numbered=True)
    qeff_model = build_meta_qeff_model(
        model_id,
        num_hidden_layers=num_hidden_layers,
        checkpoint_dir=export_dir / "checkpoint",
        model=model,
        **kwargs,
    )

    onnx_path = exported_onnx_path(
        qeff_model.export(
            export_dir,
            use_onnx_subfunctions=True,
            offload_pt_weights=False,
        )
    )
    _WEIGHT_FREE_ONNX_CACHE[key] = (str(onnx_path), qeff_model)
    return _WEIGHT_FREE_ONNX_CACHE[key]


def _set_num_hidden_layers_for_test(config, num_hidden_layers: int) -> None:
    config.num_hidden_layers = num_hidden_layers
    layer_types = getattr(config, "layer_types", None)
    if layer_types is not None:
        if len(layer_types) < num_hidden_layers:
            layer_types = list(layer_types) + ["full_attention"] * (num_hidden_layers - len(layer_types))
        config.layer_types = list(layer_types)[:num_hidden_layers]


def write_local_weight_free_checkpoint(
    model_id: str,
    checkpoint_dir: Path,
    *,
    num_hidden_layers: int = 2,
    model: AutoModelForCausalLM | None = None,
) -> Path:
    """Write a test-local safetensors checkpoint that matches the exported config."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if model is None:
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        _set_num_hidden_layers_for_test(config, num_hidden_layers)
        config.dtype = torch.float32
        config.torch_dtype = torch.float32
        model = AutoModelForCausalLM.from_config(
            config,
            trust_remote_code=True,
            attn_implementation="eager",
        )
    model.eval()
    if getattr(model.config, "pad_token_id", None) is None or model.config.pad_token_id < 0:
        model.config.pad_token_id = 0
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.pad_token_id = model.config.pad_token_id
    model.save_pretrained(checkpoint_dir, safe_serialization=True)
    return checkpoint_dir


def build_meta_qeff_model(
    model_id: str,
    num_hidden_layers: int = 2,
    *,
    checkpoint_dir: Path | None = None,
    model: AutoModelForCausalLM | None = None,
    **qeff_kwargs,
) -> QEFFAutoModelForCausalLM:
    """Build a weight-free QEff model from config, with no weights loaded into memory.

    The current weight-free API is from_pretrained(..., weight_free=True). It
    constructs the HF module on meta tensors and records the checkpoint location
    for weight_spec.json generation during export.

    num_hidden_layers limits the layer count so tests run quickly, matching the
    --layers flag used by the weight-free example scripts. Extra kwargs, such as
    continuous_batching=True, are forwarded to QEFFAutoModelForCausalLM.
    """
    model_ref = model_id
    if checkpoint_dir is not None:
        model_ref = str(
            write_local_weight_free_checkpoint(
                model_id,
                checkpoint_dir,
                num_hidden_layers=num_hidden_layers,
                model=model,
            )
        )

    config = AutoConfig.from_pretrained(model_ref, trust_remote_code=True)
    _set_num_hidden_layers_for_test(config, num_hidden_layers)
    config.dtype = torch.float32
    config.torch_dtype = torch.float32
    return QEFFAutoModelForCausalLM.from_pretrained(
        model_ref,
        config=config,
        weight_free=True,
        dtype=torch.float32,
        trust_remote_code=True,
        **qeff_kwargs,
    )


def exported_onnx_path(export_result) -> Path:
    if isinstance(export_result, (list, tuple)):
        export_result = export_result[-1]
    onnx_path = Path(export_result)
    assert onnx_path.is_file(), f"Expected ONNX file at {onnx_path}"
    return onnx_path


# ---------------------------------------------------------------------------
# Shared ONNX structure assertions (same as tests/dynamo/_helpers.py)
# ---------------------------------------------------------------------------


def assert_has_subfunctions(onnx_path: Path, qeff_model: QEFFAutoModelForCausalLM) -> None:
    """Assert the ONNX contains at least one decoder-block subfunction."""
    get_submodules = getattr(qeff_model.model, "get_submodules_for_export", None)
    if not callable(get_submodules):
        return
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


_BLOCKED_KV_MARKER_MODES = {"kv", "qkv", "hkv", "hqkv", "bhqkv", "kv_headpar", "kv_batch_fold"}


def assert_blocked_kv_ops_for_mode(
    onnx_path: Path,
    qeff_model: QEFFAutoModelForCausalLM,
    blocking_key: str,
    *,
    continuous_batching: bool = False,
) -> None:
    """Assert stable blocked-KV custom op markers for KV-bearing blocking modes.

    Pure Q/H/HQ modes do not have a small reliable graph marker, so they are
    covered by dispatch tests plus export/compile/generation parity.
    """
    if blocking_key not in _BLOCKED_KV_MARKER_MODES:
        return

    model = onnx.load(str(onnx_path), load_external_data=False)
    get_submodules = getattr(qeff_model.model, "get_submodules_for_export", None)
    decoder_names = set()
    if callable(get_submodules):
        submodule_classes = get_submodules()
        decoder_names = {
            cls.__name__
            for cls in (submodule_classes if isinstance(submodule_classes, (set, list, tuple)) else [submodule_classes])
        }

    function_nodes = [
        node
        for fn in model.functions
        if not decoder_names or any(decoder_name in fn.name for decoder_name in decoder_names)
        for node in fn.node
    ]
    op_counts = Counter(node.op_type for node in list(model.graph.node) + function_nodes)

    expected_ops = {"CtxGatherBlockedKV", "CtxGatherBlockedKVBatch"}
    if continuous_batching:
        expected_ops.add("CtxGatherBlockedKVCB")

    found_ops = {op_name: op_counts[op_name] for op_name in sorted(expected_ops) if op_counts[op_name]}
    assert found_ops, (
        f"Expected blocked KV custom op marker for mode '{blocking_key}' in {onnx_path.name}, "
        f"but none of {sorted(expected_ops)} were present. "
        f"Ops present: {dict(op_counts)}"
    )


def assert_subfunction_names_match_decoder_class(onnx_path: Path, qeff_model: QEFFAutoModelForCausalLM) -> None:
    """Verify RenameRepeatedSubgraphTransform renamed functions to decoder class names."""
    get_submodules = getattr(qeff_model.model, "get_submodules_for_export", None)
    if not callable(get_submodules):
        return
    submodule_classes = get_submodules()
    if not submodule_classes:
        return
    model = onnx.load(str(onnx_path), load_external_data=False)
    for fn in model.functions:
        assert not any(fn.name.startswith(pat) for pat in ("repeated_subgraph", "subgraph_", "invoke_subgraph_")), (
            f"Function '{fn.name}' still has raw dynamo name — RenameRepeatedSubgraphTransform did not rename it."
        )


def assert_retained_state_outputs(onnx_path: Path, expected_count: int) -> None:
    """Assert the ONNX graph has the expected number of _RetainedState outputs."""
    model = onnx.load(str(onnx_path), load_external_data=False)
    retained = [o for o in model.graph.output if o.name.endswith("_RetainedState")]
    assert len(retained) == expected_count, (
        f"Expected {expected_count} _RetainedState outputs, got {len(retained)}: {[o.name for o in retained]}"
    )


# ---------------------------------------------------------------------------
# Weight-free-specific ONNX assertions
# ---------------------------------------------------------------------------

# ONNX elem_type constants
_ONNX_INT64 = 7


def assert_unique_graph_input_names(onnx_path: Path) -> None:
    """Assert no ONNX graph input name appears twice.

    Guards the regression where position_ids is mislabeled as past_key.0 due to
    a dict-order vs input_names-order mismatch in the weight-free export path,
    producing a duplicate graph input that causes a compiler error.
    """
    model = onnx.load(str(onnx_path), load_external_data=False)
    names = [i.name for i in model.graph.input]
    duplicates = [n for n in set(names) if names.count(n) > 1]
    assert not duplicates, (
        f"Duplicate ONNX graph inputs found: {duplicates}. "
        "position_ids was likely mislabeled as a KV cache input during weight-free export."
    )


def assert_no_int64_kv_cache_inputs(onnx_path: Path) -> None:
    """Assert no past_key.X / past_value.X graph input has dtype int64.

    Guards the same regression: if position_ids (int64) is aliased to past_key.0,
    that KV cache slot will have the wrong dtype. A valid KV cache tensor is always
    a floating-point type (float16 or float32), never int64.
    """
    model = onnx.load(str(onnx_path), load_external_data=False)
    for inp in model.graph.input:
        if inp.name.startswith(("past_key.", "past_value.")):
            dtype = inp.type.tensor_type.elem_type
            assert dtype != _ONNX_INT64, (
                f"Graph input '{inp.name}' has dtype int64 — this is position_ids mislabeled "
                "as a KV cache tensor due to a weight-free export input-naming mismatch."
            )


# ---------------------------------------------------------------------------
# Weight-free ORT generation loop
# ---------------------------------------------------------------------------


def run_weight_free_ort(api_runner, onnx_path: Path, weight_spec_path: Path) -> np.ndarray:
    """Run token generation on a weight-free ONNX using ORT, injecting real weights
    from the checkpoint at every step via load_weight_free_ort_inputs.

    This mirrors the loop in examples/text_generation/compare.py.local and is needed
    because weight-free ONNX has no embedded weights — they appear as extra ORT inputs
    that must be populated from the HF cache safetensors files each inference step.

    Returns generated token IDs with shape (1, gen_len) matching run_hf_model_on_pytorch.
    """
    session = onnxruntime.InferenceSession(str(onnx_path))

    # Prepare runtime inputs (input_ids, position_ids, initial KV cache zeros)
    inputs = api_runner.input_handler.prepare_ort_inputs()
    # Inject model weights from HF cache
    inputs = load_weight_free_ort_inputs(weight_spec_path, inputs)

    ort_outputs_raw = api_runner.run_ort_session(inputs, session)
    ort_outputs = api_runner.input_handler.update_ort_outputs(ort_outputs_raw)

    generated_ids = []
    for _ in range(1, api_runner.gen_len):
        generated_ids.append(ort_outputs["logits"].argmax(-1).reshape(-1, 1))
        inputs = api_runner.input_handler.update_ort_inputs(inputs, ort_outputs)
        # Re-inject weights: update_ort_inputs only updates runtime tensors (input_ids,
        # position_ids, KV cache) and does not carry weights forward.
        inputs = load_weight_free_ort_inputs(weight_spec_path, inputs)
        ort_outputs_raw = api_runner.run_ort_session(inputs, session)
        ort_outputs = api_runner.input_handler.update_ort_outputs(ort_outputs_raw)

    generated_ids.append(ort_outputs["logits"].argmax(-1).reshape(-1, 1))
    return np.concatenate(generated_ids, axis=1)
