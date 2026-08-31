# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Weight-free + Compute-Context-Length (CCL) compile/generate tests.
"""

from __future__ import annotations

import pytest

from QEfficient.utils import get_num_layers_from_config

from ._helpers import (
    BATCH_SIZE,
    FULL_BATCH_SIZE,
    WEIGHT_FREE_CAUSAL_LM_MODEL_IDS,
    assert_hf_hw_parity,
    get_hf_tokens,
    get_weight_free_export,
    load_hf_model,
    load_tokenizer,
)

# CCL lengths need a larger ctx_len than the tiny default used by basic tests.
CCL_PREFILL_SEQ_LEN = 32
CCL_CTX_LEN = 128
CCL_LENGTHS = [1024, 2048]
PROMPT = "hello world"


def _generate(qeff_model, tokenizer, prompts, model_id, hf_tokens, full_batch_size=None):
    """Generate from the QPC produced by the preceding compile() call."""
    output = qeff_model.generate(
        tokenizer=tokenizer,
        prompts=prompts,
    )
    assert output.generated_texts is not None
    assert_hf_hw_parity(
        model_id,
        hf_tokens,
        output,
        gen_len=CCL_CTX_LEN - CCL_PREFILL_SEQ_LEN,
        full_batch_size=full_batch_size,
    )
    return output


@pytest.mark.weight_free
@pytest.mark.on_qaic
@pytest.mark.xdist_group(name="qaic-runtime")
@pytest.mark.llm_model
@pytest.mark.parametrize(
    "model_type,model_id",
    sorted(WEIGHT_FREE_CAUSAL_LM_MODEL_IDS.items()),
    ids=sorted(WEIGHT_FREE_CAUSAL_LM_MODEL_IDS),
)
def test_weight_free_ccl_compile_and_generate(model_type, model_id, tmp_export_dir, tmp_path_factory):
    """Compile and generate with default and explicit CCL lengths."""
    if model_type == "gpt_oss":
        pytest.xfail("gpt_oss CCL compile fails with ONNX broadcast shape mismatch")

    tokenizer = load_tokenizer(model_id)
    model_hf = load_hf_model(model_id)

    onnx_path, qeff_model = get_weight_free_export(
        model_id,
        tmp_path_factory,
        num_hidden_layers=get_num_layers_from_config(model_hf.config),
        qaic_config={"ccl_enabled": True},
        model=model_hf,
    )

    prompts = [PROMPT]
    hf_tokens = get_hf_tokens(tokenizer, model_hf, prompts, prompt_len=CCL_PREFILL_SEQ_LEN, ctx_len=CCL_CTX_LEN)

    # Compile with auto-generated CCL lists.
    qeff_model.compile(
        onnx_path=onnx_path,
        compile_dir=str(tmp_export_dir / "normal_compile"),
        prefill_seq_len=CCL_PREFILL_SEQ_LEN,
        ctx_len=CCL_CTX_LEN,
        num_cores=16,
        batch_size=BATCH_SIZE,
        use_onnx_subfunctions=True,
    )
    _generate(qeff_model, tokenizer, prompts, model_id, hf_tokens)

    # Compile with explicit CCL lists.
    qeff_model.compile(
        onnx_path=onnx_path,
        compile_dir=str(tmp_export_dir / "ccl_compile"),
        prefill_seq_len=CCL_PREFILL_SEQ_LEN,
        ctx_len=CCL_CTX_LEN,
        comp_ctx_lengths_prefill=CCL_LENGTHS,
        comp_ctx_lengths_decode=CCL_LENGTHS,
        num_cores=16,
        batch_size=BATCH_SIZE,
        use_onnx_subfunctions=True,
    )
    _generate(qeff_model, tokenizer, prompts, model_id, hf_tokens)


@pytest.mark.weight_free
@pytest.mark.on_qaic
@pytest.mark.xdist_group(name="qaic-runtime")
@pytest.mark.llm_model
@pytest.mark.parametrize(
    "model_type,model_id",
    sorted(WEIGHT_FREE_CAUSAL_LM_MODEL_IDS.items()),
    ids=sorted(WEIGHT_FREE_CAUSAL_LM_MODEL_IDS),
)
def test_weight_free_cb_ccl_compile_and_generate(model_type, model_id, tmp_export_dir, tmp_path_factory):
    """Compile and generate with continuous batching and CCL."""
    if model_type == "gpt_oss":
        pytest.skip("gpt_oss CB scatter op has shape mismatch with dynamo subfunctions; pending fix")

    tokenizer = load_tokenizer(model_id)
    model_hf = load_hf_model(model_id)

    onnx_path, qeff_model = get_weight_free_export(
        model_id,
        tmp_path_factory,
        num_hidden_layers=get_num_layers_from_config(model_hf.config),
        continuous_batching=True,
        qaic_config={"ccl_enabled": True},
        model=model_hf,
    )

    prompts = [PROMPT] * FULL_BATCH_SIZE
    hf_tokens = get_hf_tokens(
        tokenizer,
        model_hf,
        prompts,
        prompt_len=CCL_PREFILL_SEQ_LEN,
        ctx_len=CCL_CTX_LEN,
        full_batch_size=FULL_BATCH_SIZE,
    )

    # Compile with auto-generated CCL lists.
    qeff_model.compile(
        onnx_path=onnx_path,
        compile_dir=str(tmp_export_dir / "cb_normal_compile"),
        prefill_seq_len=CCL_PREFILL_SEQ_LEN,
        ctx_len=CCL_CTX_LEN,
        num_cores=16,
        batch_size=BATCH_SIZE,
        full_batch_size=FULL_BATCH_SIZE,
        use_onnx_subfunctions=True,
    )
    _generate(qeff_model, tokenizer, prompts, model_id, hf_tokens, full_batch_size=FULL_BATCH_SIZE)

    # Compile with explicit CCL lists.
    qeff_model.compile(
        onnx_path=onnx_path,
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
    _generate(qeff_model, tokenizer, prompts, model_id, hf_tokens, full_batch_size=FULL_BATCH_SIZE)
