# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------


import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM
from QEfficient.utils.run_utils import ApiRunner

torch.manual_seed(42)
# model_name = "openai/gpt-oss-20b"
# model_name = "meta-llama/Llama-3.2-1B"
# model_name = "gpt2"
# model_name = "hf-internal-testing/tiny-random-Olmo2ForCausalLM"
# model_name = "tiny-random/gpt-oss-bf16"
# model_name = "hf-internal-testing/tiny-random-LlamaForCausalLM"
# model_name = "ibm-granite/granite-3.1-3b-a800m-instruct"
model_name = "hf-tiny-v2/tiny-random-GraniteForCausalLM"

config = AutoConfig.from_pretrained(model_name)
# config.num_hidden_layers = 8
config.dtype = torch.float16
tokenizer = AutoTokenizer.from_pretrained(model_name, config=config)
print(config)
runner = ApiRunner(
    batch_size=1,
    tokenizer=tokenizer,
    config=config,
    prompt=["My name is"],
    prompt_len=32,
    ctx_len=128,
    dtype=torch.float16,
)

# PyTorch (KV) output
hf_model = AutoModelForCausalLM.from_pretrained(model_name, config=config, dtype=config.dtype)
hf_tokens = runner.run_hf_model_on_pytorch(hf_model)
print(hf_tokens)

qeff_model = QEFFAutoModelForCausalLM.from_pretrained(model_name, config=config, dtype=config.dtype)
pt_tokens = runner.run_kv_model_on_pytorch(qeff_model.model)
print(pt_tokens)

onnx_path = qeff_model.export(dynamo=False, use_onnx_subfunctions=True)
ort_inputs = runner.input_handler.prepare_ort_inputs()
ort_tokens = runner.run_kv_model_on_ort(onnx_path)
print(ort_tokens)

qeff_model.compile(
    onnx_path=onnx_path, prefill_seq_len=32, ctx_len=128, use_onnx_subfunctions=True, dynamo=False, num_devices=1
)
print("compile done")
print("QEff Transformed Onnx Model Outputs(AIC Backend)")
output = qeff_model.generate(prompts=["My name is"], tokenizer=tokenizer, automation=True)
print(output)
print(output.generated_ids)
qeff_tokens = output.generated_ids[0][:, : pt_tokens.shape[-1]]

# assert np.allclose(hf_tokens, pt_tokens), "HF and PT outputs do not match"

assert np.allclose(pt_tokens, ort_tokens), "PT and ORT outputs do not match"
assert np.allclose(qeff_tokens, pt_tokens), "PT and QEff outputs do not match"
