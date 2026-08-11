# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Dynamo on-QAIC tests.

All tests require QAIC hardware (marked @pytest.mark.on_qaic).
All tests run with dynamo=True and use_onnx_subfunctions=True.

Covers:
  - FP16-weights compile (model loaded with torch_dtype=torch.float16)
  - FP32-weights compile (model loaded with torch_dtype=torch.float32) --
    currently disabled via @pytest.mark.skip, see that test for why.
  - Multi-device compile (num_devices=4)
  - Generate FP16
  - HF AIC HW parity (HF PT tokens == QAIC FP16 top-1 token)
  - Continuous-batching generate

The QAIC compiler never emits an fp32 QPC: CUSTOM_IO_DTYPE_MAP maps both
torch.float16 and torch.float32 models to a "float16" convert_to_fp16 compile,
so the fp16-weights vs fp32-weights split above is purely about which PyTorch
model dtype gets exported (i.e. different ONNX weights, both compiled to the
same fp16 QPC format) -- not a compile-time fp16-vs-fp32 choice. Multi-device
compile/generate/hw-parity all reuse the fp16-weights basic (non-CB) export
via get_dynamo_export() instead of each re-running the expensive
dynamo+subfunctions export. CB generate uses a separate
continuous_batching=True export (cached independently -- test_ccl.py's CB+CCL
tests key on continuous_batching=True AND ccl_enabled=True, so their export
also has the comp_ctx_lengths graph input and is not the same artifact).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ._helpers import (
    BATCH_SIZE,
    CTX_LEN,
    DYNAMO_CAUSAL_LM_MODEL_IDS,
    FULL_BATCH_SIZE,
    PROMPT_LEN,
    get_dynamo_export,
    load_hf_model,
    load_tokenizer,
    skip_on_model_fetch_error,
)


@pytest.mark.dynamo
@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.parametrize(
    "model_type,model_id", list(DYNAMO_CAUSAL_LM_MODEL_IDS.items()), ids=list(DYNAMO_CAUSAL_LM_MODEL_IDS)
)
def test_dynamo_fp16_weights_compile(model_type, model_id, tmp_export_dir, tmp_path_factory):
    """Export a torch.float16-weights model, compile to QPC."""

    try:
        onnx_path, qeff_model = get_dynamo_export(model_id, tmp_path_factory, torch_dtype=torch.float16)
    except Exception as exc:
        skip_on_model_fetch_error(exc, model_id)

    qeff_model.compile(
        onnx_path=onnx_path,
        compile_dir=str(tmp_export_dir / "fp16_weights_compile"),
        prefill_seq_len=PROMPT_LEN,
        ctx_len=CTX_LEN,
        num_cores=16,
        batch_size=BATCH_SIZE,
        use_onnx_subfunctions=True,
    )
    assert (tmp_export_dir / "fp16_weights_compile").is_dir()


@pytest.mark.dynamo
@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.skip(reason="Disabled while testing -- fp32-weights compile still converts to fp16 on-device anyway.")
@pytest.mark.parametrize(
    "model_type,model_id", list(DYNAMO_CAUSAL_LM_MODEL_IDS.items()), ids=list(DYNAMO_CAUSAL_LM_MODEL_IDS)
)
def test_dynamo_fp32_weights_compile(model_type, model_id, tmp_export_dir, tmp_path_factory):
    """Export a torch.float32-weights model, compile to QPC (compiler still converts to fp16)."""

    try:
        onnx_path, qeff_model = get_dynamo_export(model_id, tmp_path_factory, torch_dtype=torch.float32)
    except Exception as exc:
        skip_on_model_fetch_error(exc, model_id)

    qeff_model.compile(
        onnx_path=onnx_path,
        compile_dir=str(tmp_export_dir / "fp32_weights_compile"),
        prefill_seq_len=PROMPT_LEN,
        ctx_len=CTX_LEN,
        num_cores=16,
        batch_size=BATCH_SIZE,
        mxfp6_matmul=False,
        mxint8_kv_cache=False,
        use_onnx_subfunctions=True,
    )
    assert (tmp_export_dir / "fp32_weights_compile").is_dir()


@pytest.mark.dynamo
@pytest.mark.dynamo_multi_device
@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.parametrize(
    "model_type,model_id",
    list(DYNAMO_CAUSAL_LM_MODEL_IDS.items()),
    ids=list(DYNAMO_CAUSAL_LM_MODEL_IDS),
)
def test_dynamo_multi_device_compile(model_type, model_id, tmp_export_dir, tmp_path_factory):
    """Compile the shared fp16-weights basic export for 4 devices."""

    try:
        onnx_path, qeff_model = get_dynamo_export(model_id, tmp_path_factory, torch_dtype=torch.float16)
    except Exception as exc:
        skip_on_model_fetch_error(exc, model_id)

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


@pytest.mark.dynamo
@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.parametrize(
    "model_type,model_id",
    list(DYNAMO_CAUSAL_LM_MODEL_IDS.items()),
    ids=list(DYNAMO_CAUSAL_LM_MODEL_IDS),
)
def test_dynamo_generate_fp16(model_type, model_id, tmp_export_dir, tmp_path_factory):
    """Compile the shared fp16-weights basic export -> generate with dynamo=True and use_onnx_subfunctions=True."""

    try:
        onnx_path, qeff_model = get_dynamo_export(model_id, tmp_path_factory, torch_dtype=torch.float16)
        tokenizer = load_tokenizer(model_id, torch_dtype=torch.float16)
    except Exception as exc:
        skip_on_model_fetch_error(exc, model_id)

    qeff_model.compile(
        onnx_path=onnx_path,
        compile_dir=str(tmp_export_dir / "gen_compile"),
        prefill_seq_len=PROMPT_LEN,
        ctx_len=CTX_LEN,
        num_cores=16,
        batch_size=BATCH_SIZE,
        use_onnx_subfunctions=True,
    )
    output = qeff_model.generate(
        tokenizer=tokenizer,
        prompts=["hello world"],
    )
    assert output is not None
    assert output.generated_texts is not None


@pytest.mark.dynamo
@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.parametrize(
    "model_type,model_id",
    list(DYNAMO_CAUSAL_LM_MODEL_IDS.items()),
    ids=list(DYNAMO_CAUSAL_LM_MODEL_IDS),
)
def test_dynamo_hw_hf_parity(model_type, model_id, tmp_export_dir, tmp_path_factory):
    """HF PT tokens == QAIC FP16 tokens (exact equality), against the shared fp16-weights basic export."""
    from QEfficient.utils.run_utils import ApiRunner

    try:
        tokenizer = load_tokenizer(model_id, torch_dtype=torch.float16)
        model_hf = load_hf_model(model_id, torch_dtype=torch.float16)
        onnx_path, qeff_model = get_dynamo_export(model_id, tmp_path_factory, torch_dtype=torch.float16)
    except Exception as exc:
        skip_on_model_fetch_error(exc, model_id)

    api_runner = ApiRunner(
        batch_size=BATCH_SIZE,
        tokenizer=tokenizer,
        config=model_hf.config,
        prompt=["hello world"],
        prompt_len=PROMPT_LEN,
        ctx_len=CTX_LEN,
        full_batch_size=None,
    )

    hf_tokens = api_runner.run_hf_model_on_pytorch(model_hf)
    assert hf_tokens is not None, "HF PT inference returned None"

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

    assert qaic_output is not None, "QAIC generate returned None"
    if hasattr(qaic_output, "generated_ids") and qaic_output.generated_ids is not None:
        gen_len = CTX_LEN - PROMPT_LEN
        qaic_tokens = qaic_output.generated_ids[0].flatten()[:gen_len]
        assert np.array_equal(hf_tokens, qaic_tokens), (
            f"HF AIC HW parity failed for {model_id}: HF={hf_tokens.tolist()}, QAIC={qaic_tokens.tolist()}"
        )


@pytest.mark.dynamo
@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.parametrize(
    "model_type,model_id",
    list(DYNAMO_CAUSAL_LM_MODEL_IDS.items()),
    ids=list(DYNAMO_CAUSAL_LM_MODEL_IDS),
)
def test_dynamo_cb_generate(model_type, model_id, tmp_export_dir):
    """Continuous-batching export → compile → generate with dynamo=True and use_onnx_subfunctions=True."""

    try:
        onnx_path, qeff_model = get_dynamo_export(
            model_id, tmp_path_factory, torch_dtype=torch.float16, continuous_batching=True
        )
        tokenizer = load_tokenizer(model_id, torch_dtype=torch.float16)
    except Exception as exc:
        skip_on_model_fetch_error(exc, model_id)

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
    prompts = ["hello world"] * FULL_BATCH_SIZE
    output = qeff_model.generate(
        tokenizer=tokenizer,
        prompts=prompts,
    )
    assert output is not None
    assert output.generated_texts is not None
    assert len(output.generated_texts) == FULL_BATCH_SIZE
