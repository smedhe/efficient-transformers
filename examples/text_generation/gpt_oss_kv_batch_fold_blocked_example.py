# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""GPT-OSS kv_batch_fold decode logits comparison.

This exercises ``blocked_kv_attention_forward_decode_headpar_batch`` via
``qaic_config={"blocking_mode": "kv_batch_fold"}`` and compares its one-step
decode logits against a non-blocked continuous-batching decode graph.
"""

import argparse
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM
from QEfficient.generation.cloud_infer import QAICInferenceSession


def parse_args():
    parser = argparse.ArgumentParser(description="Compare GPT-OSS kv_batch_fold decode logits to non-blocked logits")
    parser.add_argument("--model-name", type=str, default="openai/gpt-oss-20b", help="HuggingFace model ID")
    parser.add_argument("--prompt", action="append", dest="prompts", help="Prompt text. Repeat for multiple slots.")
    parser.add_argument("--ctx-len", type=int, default=2048, help="Decode KV cache context length")
    parser.add_argument("--num-kv-blocks", type=int, default=2, help="Number of KV blocks for kv_batch_fold")
    parser.add_argument("--full-batch-size", type=int, default=4, help="Continuous-batching decode width")
    parser.add_argument("--num-cores", type=int, default=16, help="Number of cores")
    parser.add_argument("--num-devices", type=int, default=4, help="Number of devices")
    parser.add_argument("--num-layers", type=int, default=None, help="Override number of layers for quick testing")
    parser.add_argument("--subf", action="store_true", help="Use ONNX subfunctions during export")
    parser.add_argument("--atol", type=float, default=1e-2, help="Absolute tolerance for logits comparison")
    parser.add_argument("--rtol", type=float, default=1e-2, help="Relative tolerance for logits comparison")
    parser.add_argument("--top-k", type=int, default=5, help="Print top-k logits for the first batch row")
    return parser.parse_args()


def _from_pretrained(model_name, num_layers):
    kwargs = {"continuous_batching": True}
    if num_layers is not None:
        kwargs["num_hidden_layers"] = num_layers
    return QEFFAutoModelForCausalLM.from_pretrained(model_name, **kwargs)


def _binding_shape_and_dtype(session, name):
    binding = session.bindings[session.binding_index_map[name]]
    return tuple(binding.dims), session.aic_to_np_dtype_mapping[binding.type]


def _zeros_for_past_inputs(session):
    inputs = {}
    for name in session.input_names:
        basename = name.rsplit("/", 1)[-1]
        if basename.startswith(("past_key.", "past_value.")):
            shape, dtype = _binding_shape_and_dtype(session, basename)
            inputs[basename] = np.zeros(shape, dtype=dtype)
    return inputs


def _normalize_logits(logits):
    logits = np.asarray(logits)
    if logits.ndim == 2:
        return logits
    if logits.ndim == 3:
        return logits[:, -1, :]
    raise ValueError(f"Unsupported logits shape: {logits.shape}")


def _prepare_decode_inputs(session, tokenizer, prompts, full_batch_size):
    prompts = list(prompts or ["Hello"] * full_batch_size)
    if len(prompts) == 1:
        prompts = prompts * full_batch_size
    if len(prompts) != full_batch_size:
        raise ValueError(f"Expected 1 or {full_batch_size} prompts, got {len(prompts)}")

    encoded = tokenizer(prompts, return_tensors="np", padding=True)
    attention_mask = encoded["attention_mask"]
    last_indices = attention_mask.sum(axis=1) - 1
    input_ids = encoded["input_ids"][np.arange(full_batch_size), last_indices].reshape(full_batch_size, 1)
    position_ids = last_indices.reshape(full_batch_size, 1).astype(np.int64)

    inputs = _zeros_for_past_inputs(session)
    _, input_ids_dtype = _binding_shape_and_dtype(session, "input_ids")
    _, position_ids_dtype = _binding_shape_and_dtype(session, "position_ids")
    inputs["input_ids"] = input_ids.astype(input_ids_dtype)
    inputs["position_ids"] = position_ids.astype(position_ids_dtype)
    if "batch_index" in session.binding_index_map:
        _, batch_index_dtype = _binding_shape_and_dtype(session, "batch_index")
        inputs["batch_index"] = np.arange(full_batch_size).reshape(full_batch_size, 1).astype(batch_index_dtype)
    return inputs


def _compile_decode_model(model_name, num_layers, compile_kwargs, qaic_config=None):
    model = _from_pretrained(model_name, num_layers)
    return model.compile(
        prefill_seq_len=1,
        qaic_config=qaic_config,
        user_tiled=qaic_config is not None,
        **compile_kwargs,
    )


def _compare_logits(reference, blocked, atol, rtol, top_k):
    reference = _normalize_logits(reference)
    blocked = _normalize_logits(blocked)
    if reference.shape != blocked.shape:
        raise AssertionError(f"Logits shape mismatch: non-blocked={reference.shape}, kv_batch_fold={blocked.shape}")

    diff = np.abs(reference.astype(np.float32) - blocked.astype(np.float32))
    max_abs = float(diff.max())
    mean_abs = float(diff.mean())
    passed = bool(np.allclose(reference, blocked, atol=atol, rtol=rtol))

    reference_tokens = np.argmax(reference, axis=-1)
    blocked_tokens = np.argmax(blocked, axis=-1)
    tokens_match = bool(np.array_equal(reference_tokens, blocked_tokens))

    print(f"Logits allclose: {passed} (atol={atol}, rtol={rtol})")
    print(f"Logits max_abs_diff={max_abs:.6f}, mean_abs_diff={mean_abs:.6f}")
    print(f"Argmax token match: {tokens_match}")
    print(f"Non-blocked argmax: {reference_tokens.tolist()}")
    print(f"kv_batch_fold argmax: {blocked_tokens.tolist()}")

    if top_k > 0:
        ref_top = np.argsort(reference[0])[-top_k:][::-1]
        blk_top = np.argsort(blocked[0])[-top_k:][::-1]
        print(f"Top-{top_k} non-blocked row 0: {ref_top.tolist()}")
        print(f"Top-{top_k} kv_batch_fold row 0: {blk_top.tolist()}")

    if not passed:
        mismatch = np.unravel_index(int(diff.argmax()), diff.shape)
        raise AssertionError(
            "kv_batch_fold logits differ from non-blocked logits beyond tolerance; "
            f"max mismatch at {mismatch}: non-blocked={reference[mismatch]}, kv_batch_fold={blocked[mismatch]}"
        )


def main():
    args = parse_args()

    if args.full_batch_size < 1:
        raise ValueError("--full-batch-size must be >= 1")
    if args.num_layers is not None and args.num_layers < 1:
        raise ValueError("--num-layers must be >= 1 when provided; omit it for the full model")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    compile_kwargs = {
        "ctx_len": args.ctx_len,
        "full_batch_size": args.full_batch_size,
        "num_cores": args.num_cores,
        "num_devices": args.num_devices,
        "mxfp6_matmul": True,
        "mxint8_kv_cache": True,
        "retain_full_kv": True,
        "use_onnx_subfunctions": args.subf,
    }

    print("[1/2] Compiling non-blocked continuous-batching decode model...")
    reference_qpc = _compile_decode_model(args.model_name, args.num_layers, compile_kwargs)
    print(f"  -> {Path(reference_qpc)}")

    print("\n[2/2] Compiling kv_batch_fold blocked decode model...")
    blocked_qpc = _compile_decode_model(
        args.model_name,
        args.num_layers,
        compile_kwargs,
        qaic_config={"blocking_mode": "kv_batch_fold", "num_kv_blocks": args.num_kv_blocks},
    )
    print(f"  -> {Path(blocked_qpc)}")

    reference_session = QAICInferenceSession(reference_qpc)
    blocked_session = QAICInferenceSession(blocked_qpc)

    reference_inputs = _prepare_decode_inputs(reference_session, tokenizer, args.prompts, args.full_batch_size)
    blocked_inputs = _prepare_decode_inputs(blocked_session, tokenizer, args.prompts, args.full_batch_size)

    print("\nRunning one decode step...")
    reference_out = reference_session.run(reference_inputs)
    blocked_out = blocked_session.run(blocked_inputs)
    _compare_logits(reference_out["logits"], blocked_out["logits"], args.atol, args.rtol, args.top_k)


if __name__ == "__main__":
    main()
