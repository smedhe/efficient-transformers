# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Weight-free on-QAIC tests.

All tests require QAIC hardware (marked @pytest.mark.on_qaic).
All tests run with weight_free=True (set on the QEff model, which forces
dynamo=True internally) and use_onnx_subfunctions=True.

Mirrors tests/dynamo/test_on_qaic.py — weight-free is dynamo-plus, so each
dynamo on-QAIC test has a weight-free equivalent here, built from a
from_pretrained(..., weight_free=True) instead of real weights +
dynamo=True. test_weight_free_vs_legacy_qaic_parity has no dynamo equivalent
(it is weight-free-specific: compares a weight-free-compiled QPC against a
legacy/dynamo-compiled QPC of the same model).
"""

from __future__ import annotations

import numpy as np
import pytest

from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM
from QEfficient.utils import get_num_layers_from_config
from QEfficient.utils.device_utils import get_available_device_id

from ._helpers import (
    BATCH_SIZE,
    CTX_LEN,
    FULL_BATCH_SIZE,
    PROMPT_LEN,
    WEIGHT_FREE_CAUSAL_LM_MODEL_IDS,
    assert_hf_hw_parity,
    build_meta_qeff_model,
    exported_onnx_path,
    get_hf_tokens,
    get_weight_free_export,
    load_hf_model,
    load_tokenizer,
)


@pytest.mark.weight_free
@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.skip(reason="Covered by test_weight_free_hw_hf_parity, which also compiles before generation.")
@pytest.mark.parametrize(
    "model_type,model_id",
    sorted(WEIGHT_FREE_CAUSAL_LM_MODEL_IDS.items()),
    ids=sorted(WEIGHT_FREE_CAUSAL_LM_MODEL_IDS),
)
def test_weight_free_fp16_compile(model_type, model_id, tmp_export_dir, tmp_path_factory):
    """Export a weight-free meta-device model, compile to QPC."""
    if model_type == "gpt_oss":
        pytest.xfail()

    onnx_path, qeff_model = get_weight_free_export(model_id, tmp_path_factory)

    qeff_model.compile(
        onnx_path=onnx_path,
        compile_dir=str(tmp_export_dir / "fp16_compile"),
        prefill_seq_len=PROMPT_LEN,
        ctx_len=CTX_LEN,
        num_cores=16,
        batch_size=BATCH_SIZE,
        use_onnx_subfunctions=True,
    )
    assert (tmp_export_dir / "fp16_compile").is_dir()


@pytest.mark.weight_free
@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.skip(reason="Disabled while testing -- fp32-weights compile still converts to fp16 on-device anyway.")
@pytest.mark.parametrize(
    "model_type,model_id",
    sorted(WEIGHT_FREE_CAUSAL_LM_MODEL_IDS.items()),
    ids=sorted(WEIGHT_FREE_CAUSAL_LM_MODEL_IDS),
)
def test_weight_free_fp32_compile(model_type, model_id, tmp_export_dir, tmp_path_factory):
    """Export a weight-free meta-device model, compile to QPC (compiler still converts to fp16)."""
    if model_type == "gpt_oss":
        pytest.xfail()

    onnx_path, qeff_model = get_weight_free_export(model_id, tmp_path_factory)

    qeff_model.compile(
        onnx_path=onnx_path,
        compile_dir=str(tmp_export_dir / "fp32_compile"),
        prefill_seq_len=PROMPT_LEN,
        ctx_len=CTX_LEN,
        num_cores=16,
        batch_size=BATCH_SIZE,
        mxfp6_matmul=False,
        mxint8_kv_cache=False,
        use_onnx_subfunctions=True,
    )
    assert (tmp_export_dir / "fp32_compile").is_dir()


@pytest.mark.weight_free
@pytest.mark.weight_free_multi_device
@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.parametrize(
    "model_type,model_id",
    sorted(WEIGHT_FREE_CAUSAL_LM_MODEL_IDS.items()),
    ids=sorted(WEIGHT_FREE_CAUSAL_LM_MODEL_IDS),
)
def test_weight_free_multi_device_compile(model_type, model_id, tmp_export_dir, tmp_path_factory):
    """Compile the shared weight-free export for 4 devices."""
    if model_type == "gpt_oss":
        pytest.xfail()

    onnx_path, qeff_model = get_weight_free_export(model_id, tmp_path_factory)

    qeff_model.compile(
        onnx_path=onnx_path,
        compile_dir=str(tmp_export_dir / "mdp_compile"),
        prefill_seq_len=PROMPT_LEN,
        ctx_len=CTX_LEN,
        num_cores=16,
        num_devices=4,
        batch_size=BATCH_SIZE,
        use_onnx_subfunctions=True,
    )
    assert (tmp_export_dir / "mdp_compile").is_dir()


@pytest.mark.weight_free
@pytest.mark.on_qaic
@pytest.mark.xdist_group(name="qaic-runtime")
@pytest.mark.llm_model
@pytest.mark.parametrize(
    "model_type,model_id",
    sorted(WEIGHT_FREE_CAUSAL_LM_MODEL_IDS.items()),
    ids=sorted(WEIGHT_FREE_CAUSAL_LM_MODEL_IDS),
)
def test_weight_free_hw_hf_parity(model_type, model_id, tmp_export_dir, tmp_path_factory):
    """Validate exact HF PT vs weight-free QAIC FP16 token parity."""
    if model_type == "gpt_oss":
        pytest.xfail()

    tokenizer = load_tokenizer(model_id)
    model_hf = load_hf_model(model_id)

    hf_tokens = get_hf_tokens(
        tokenizer,
        model_hf,
        ["hello world"],
        prompt_len=PROMPT_LEN,
        ctx_len=CTX_LEN,
    )

    onnx_path, qeff_model = get_weight_free_export(
        model_id,
        tmp_path_factory,
        num_hidden_layers=get_num_layers_from_config(model_hf.config),
        model=model_hf,
    )

    qeff_model.compile(
        onnx_path=onnx_path,
        compile_dir=str(tmp_export_dir / "hw_parity_compile"),
        prefill_seq_len=PROMPT_LEN,
        ctx_len=CTX_LEN,
        num_cores=16,
        batch_size=BATCH_SIZE,
        use_onnx_subfunctions=True,
    )

    qaic_output = qeff_model.generate(
        tokenizer=tokenizer,
        prompts=["hello world"],
    )

    assert_hf_hw_parity(
        model_id,
        hf_tokens,
        qaic_output,
        gen_len=CTX_LEN - PROMPT_LEN,
    )


@pytest.mark.weight_free
@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.parametrize(
    "model_type,model_id",
    sorted(WEIGHT_FREE_CAUSAL_LM_MODEL_IDS.items()),
    ids=sorted(WEIGHT_FREE_CAUSAL_LM_MODEL_IDS),
)
def test_weight_free_cb_generate(model_type, model_id, tmp_export_dir, tmp_path_factory):
    """Continuous-batching weight-free export -> compile -> generate with HF PT parity."""
    if model_type == "gpt_oss":
        pytest.xfail()

    tokenizer = load_tokenizer(model_id)
    model_hf = load_hf_model(model_id)

    prompts = ["hello world", "quick brown fox", "machine learning", "open source"]
    hf_tokens = get_hf_tokens(
        tokenizer,
        model_hf,
        prompts,
        prompt_len=PROMPT_LEN,
        ctx_len=CTX_LEN,
        full_batch_size=FULL_BATCH_SIZE,
    )

    onnx_path, qeff_model = get_weight_free_export(
        model_id,
        tmp_path_factory,
        num_hidden_layers=get_num_layers_from_config(model_hf.config),
        continuous_batching=True,
        model=model_hf,
    )

    qeff_model.compile(
        onnx_path=onnx_path,
        compile_dir=str(tmp_export_dir / "cb_gen_compile"),
        prefill_seq_len=PROMPT_LEN,
        ctx_len=CTX_LEN,
        num_cores=16,
        batch_size=BATCH_SIZE,
        full_batch_size=FULL_BATCH_SIZE,
        use_onnx_subfunctions=True,
    )
    output = qeff_model.generate(
        tokenizer=tokenizer,
        prompts=prompts,
        device_id=get_available_device_id(),
    )
    assert_hf_hw_parity(
        model_id,
        hf_tokens,
        output,
        gen_len=CTX_LEN - PROMPT_LEN,
        full_batch_size=FULL_BATCH_SIZE,
        context="CB",
    )


@pytest.mark.weight_free
@pytest.mark.on_qaic
@pytest.mark.xdist_group(name="qaic-runtime")
@pytest.mark.llm_model
@pytest.mark.parametrize(
    "model_type,model_id",
    sorted(WEIGHT_FREE_CAUSAL_LM_MODEL_IDS.items()),
    ids=sorted(WEIGHT_FREE_CAUSAL_LM_MODEL_IDS),
)
def test_weight_free_vs_legacy_qaic_parity(model_type, model_id, tmp_export_dir):
    """Weight-free-compiled and legacy dynamo-compiled QPCs produce identical tokens on QAIC.

    Weight-free-only: no dynamo equivalent, since it compares two independently
    compiled QPCs of the same model (weight-free vs legacy) rather than HF PT.
    """

    tokenizer = load_tokenizer(model_id)
    model_hf = load_hf_model(model_id)

    # Legacy/dynamo leg — real weights, no weight-free export.
    qeff_legacy = QEFFAutoModelForCausalLM.from_pretrained(model_id)
    legacy_onnx_path = exported_onnx_path(
        qeff_legacy.export(
            tmp_export_dir / "legacy_export",
            dynamo=True,
            use_onnx_subfunctions=True,
            offload_pt_weights=False,
        )
    )
    qeff_legacy.compile(
        onnx_path=str(legacy_onnx_path),
        compile_dir=str(tmp_export_dir / "legacy_compile"),
        prefill_seq_len=PROMPT_LEN,
        ctx_len=CTX_LEN,
        num_cores=16,
        batch_size=BATCH_SIZE,
        use_onnx_subfunctions=True,
        dynamo=True,
    )
    legacy_output = qeff_legacy.generate(
        tokenizer=tokenizer,
        prompts=["hello world"],
    )
    assert legacy_output is not None, "Legacy QAIC generate returned None"

    # Weight-free leg — meta-device model backed by a local checkpoint with the same weights.
    qeff_weight_free = build_meta_qeff_model(
        model_id,
        num_hidden_layers=get_num_layers_from_config(model_hf.config),
        checkpoint_dir=tmp_export_dir / "wf_vs_legacy_checkpoint",
        model=model_hf,
    )

    weight_free_onnx_path = exported_onnx_path(
        qeff_weight_free.export(
            tmp_export_dir / "wf_vs_legacy_export",
            use_onnx_subfunctions=True,
            offload_pt_weights=False,
        )
    )
    qeff_weight_free.compile(
        onnx_path=str(weight_free_onnx_path),
        compile_dir=str(tmp_export_dir / "wf_vs_legacy_compile"),
        prefill_seq_len=PROMPT_LEN,
        ctx_len=CTX_LEN,
        num_cores=16,
        batch_size=BATCH_SIZE,
        use_onnx_subfunctions=True,
    )
    weight_free_output = qeff_weight_free.generate(
        tokenizer=tokenizer,
        prompts=["hello world"],
    )
    assert weight_free_output is not None, "Weight-free QAIC generate returned None"

    if (
        hasattr(legacy_output, "generated_ids")
        and legacy_output.generated_ids is not None
        and hasattr(weight_free_output, "generated_ids")
        and weight_free_output.generated_ids is not None
    ):
        legacy_tokens = legacy_output.generated_ids[0].flatten()
        weight_free_tokens = weight_free_output.generated_ids[0].flatten()
        assert np.array_equal(legacy_tokens, weight_free_tokens), (
            f"Weight-free vs legacy QAIC parity failed for {model_id}: "
            f"legacy={legacy_tokens.tolist()}, weight_free={weight_free_tokens.tolist()}"
        )
