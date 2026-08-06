# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Dynamo + Compute-Context-Length (CCL) export/compile tests.

All tests run with dynamo=True and use_onnx_subfunctions=True, plus
qaic_config={"ccl_enabled": True} (basic CCL) and continuous_batching=True
combined with CCL (cb_ccl).

CPU-only tests validate ONNX export structure. on_qaic tests additionally
compile with comp_ctx_lengths_prefill/decode specializations and generate.
"""

from __future__ import annotations

import pytest

from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM

from ._helpers import (
    BATCH_SIZE,
    DYNAMO_CAUSAL_LM_MODEL_IDS,
    FULL_BATCH_SIZE,
    assert_has_subfunctions,
    assert_retained_state_outputs,
    exported_onnx_path,
    load_hf_model,
    load_tokenizer,
    skip_on_model_fetch_error,
)

# CCL's specialization validation floors small comp_ctx_lengths values up to
# CCL_MIN_CTX_LEN (1024) then clamps back down to ctx_len; with the shared tiny
# CTX_LEN=16 used elsewhere in tests/dynamo/, both prefill and decode collapse to
# the same value and the collision-repair step (walking down by CCL_UNIQNE_STEP=32)
# lands on 0, which the compiler rejects. Use ctx_len/prefill_seq_len large enough
# to avoid that collision (matches the values validated in manual CCL automation runs).
CCL_PREFILL_SEQ_LEN = 32
CCL_CTX_LEN = 128
CCL_LENGTHS = [1024, 2048]


@pytest.mark.dynamo
@pytest.mark.dynamo_export
@pytest.mark.parametrize(
    "model_type,model_id", sorted(DYNAMO_CAUSAL_LM_MODEL_IDS.items()), ids=sorted(DYNAMO_CAUSAL_LM_MODEL_IDS)
)
def test_dynamo_ccl_export(model_type, model_id, tmp_export_dir):
    """qaic_config={'ccl_enabled': True} + dynamo=True and use_onnx_subfunctions=True."""
    try:
        model_hf = load_hf_model(model_id)
    except Exception as exc:
        skip_on_model_fetch_error(exc, model_id)

    qeff_model = QEFFAutoModelForCausalLM(model_hf, qaic_config={"ccl_enabled": True})
    onnx_path = exported_onnx_path(
        qeff_model.export(
            tmp_export_dir,
            dynamo=True,
            use_onnx_subfunctions=True,
            offload_pt_weights=False,
        )
    )

    num_layers = model_hf.config.num_hidden_layers
    assert_retained_state_outputs(onnx_path, expected_count=2 * num_layers)
    assert_has_subfunctions(onnx_path, qeff_model)


@pytest.mark.dynamo
@pytest.mark.dynamo_export
@pytest.mark.parametrize(
    "model_type,model_id", sorted(DYNAMO_CAUSAL_LM_MODEL_IDS.items()), ids=sorted(DYNAMO_CAUSAL_LM_MODEL_IDS)
)
def test_dynamo_cb_ccl_export(model_type, model_id, tmp_export_dir):
    """continuous_batching=True + qaic_config={'ccl_enabled': True} + dynamo=True and use_onnx_subfunctions=True."""
    try:
        model_hf = load_hf_model(model_id)
    except Exception as exc:
        skip_on_model_fetch_error(exc, model_id)

    qeff_model = QEFFAutoModelForCausalLM(model_hf, continuous_batching=True, qaic_config={"ccl_enabled": True})
    onnx_path = exported_onnx_path(
        qeff_model.export(
            tmp_export_dir,
            dynamo=True,
            use_onnx_subfunctions=True,
            offload_pt_weights=False,
        )
    )

    num_layers = model_hf.config.num_hidden_layers
    assert_retained_state_outputs(onnx_path, expected_count=2 * num_layers)
    assert_has_subfunctions(onnx_path, qeff_model)


@pytest.mark.dynamo
@pytest.mark.on_qaic
@pytest.mark.xdist_group(name="qaic-runtime")
@pytest.mark.llm_model
@pytest.mark.parametrize(
    "model_type,model_id", list(DYNAMO_CAUSAL_LM_MODEL_IDS.items()), ids=list(DYNAMO_CAUSAL_LM_MODEL_IDS)
)
def test_dynamo_ccl_generate(model_type, model_id, tmp_export_dir):
    """Export -> compile with comp_ctx_lengths -> generate, dynamo=True + use_onnx_subfunctions=True + ccl_enabled=True."""
    try:
        model_hf = load_hf_model(model_id)
        tokenizer = load_tokenizer(model_id)
    except Exception as exc:
        skip_on_model_fetch_error(exc, model_id)

    qeff_model = QEFFAutoModelForCausalLM(model_hf, qaic_config={"ccl_enabled": True})
    onnx_path = exported_onnx_path(
        qeff_model.export(
            tmp_export_dir / "ccl_export",
            dynamo=True,
            use_onnx_subfunctions=True,
        )
    )
    qeff_model.compile(
        onnx_path=str(onnx_path),
        compile_dir=str(tmp_export_dir / "ccl_compile"),
        prefill_seq_len=CCL_PREFILL_SEQ_LEN,
        ctx_len=CCL_CTX_LEN,
        comp_ctx_lengths_prefill=CCL_LENGTHS,
        comp_ctx_lengths_decode=CCL_LENGTHS,
        num_cores=16,
        batch_size=BATCH_SIZE,
        use_onnx_subfunctions=True,
    )
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
@pytest.mark.parametrize(
    "model_type,model_id", list(DYNAMO_CAUSAL_LM_MODEL_IDS.items()), ids=list(DYNAMO_CAUSAL_LM_MODEL_IDS)
)
def test_dynamo_cb_ccl_generate(model_type, model_id, tmp_export_dir):
    """Continuous-batching + CCL: export -> compile -> generate, dynamo=True + use_onnx_subfunctions=True."""
    # TODO: fix gpt_oss CB scatter op shape mismatch with dynamo subfunctions (see test_dynamo_cb_generate).
    if model_type == "gpt_oss":
        pytest.skip("gpt_oss CB scatter op has shape mismatch with dynamo subfunctions — pending fix")

    try:
        model_hf = load_hf_model(model_id)
        tokenizer = load_tokenizer(model_id)
    except Exception as exc:
        skip_on_model_fetch_error(exc, model_id)

    qeff_model = QEFFAutoModelForCausalLM(model_hf, continuous_batching=True, qaic_config={"ccl_enabled": True})
    onnx_path = exported_onnx_path(
        qeff_model.export(
            tmp_export_dir / "cb_ccl_export",
            dynamo=True,
            use_onnx_subfunctions=True,
        )
    )
    qeff_model.compile(
        onnx_path=str(onnx_path),
        compile_dir=str(tmp_export_dir / "cb_ccl_compile"),
        prefill_seq_len=CCL_PREFILL_SEQ_LEN,
        ctx_len=CCL_CTX_LEN,
        comp_ctx_lengths_prefill=CCL_LENGTHS,
        comp_ctx_lengths_decode=CCL_LENGTHS,
        num_cores=16,
        batch_size=BATCH_SIZE,
        full_batch_size=FULL_BATCH_SIZE,
        use_onnx_subfunctions=True,
    )
    prompts = ["hello world"] * FULL_BATCH_SIZE
    output = qeff_model.generate(
        tokenizer=tokenizer,
        prompts=prompts,
        device_id=[0],
    )
    assert output is not None
    assert output.generated_texts is not None
    assert len(output.generated_texts) == FULL_BATCH_SIZE
