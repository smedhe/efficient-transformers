#!/usr/bin/env python3
# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM
from QEfficient.utils.run_utils import ApiRunner
from scripts.memory_profiling import QEffMemoryProfiler


def _add_repo_root_to_path() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


_add_repo_root_to_path()


def _str_to_bool(value: str) -> bool:
    value = str(value).strip().lower()
    if value in {"true", "1", "yes", "y", "on"}:
        return True
    if value in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value!r}. Use true/false.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run export/ORT/compile flow with timing and export RAM profiling.")
    parser.add_argument(
        "--model-name",
        type=str,
        default="meta-llama/Llama-3.2-1B",
        help="Hugging Face model id.",
    )
    parser.add_argument(
        "--use-dynamo",
        type=_str_to_bool,
        default=True,
        metavar="{true,false}",
        help="Whether to enable dynamo during export (true/false).",
    )
    parser.add_argument(
        "--use-onnx-subfunctions",
        type=_str_to_bool,
        default=True,
        metavar="{true,false}",
        help="Whether to enable ONNX subfunctions during export/compile (true/false).",
    )
    parser.add_argument(
        "--num-hidden-layers",
        type=int,
        default=4,
        help="Override config.num_hidden_layers for quick experiments.",
    )
    parser.add_argument(
        "--torch-dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float32",
        help="Dtype to set in config.torch_dtype.",
    )
    parser.add_argument("--prompt", type=str, default="My name is")
    parser.add_argument("--prompt-len", type=int, default=8)
    parser.add_argument("--ctx-len", type=int, default=32)
    parser.add_argument(
        "--profile-output",
        type=Path,
        default=Path(__file__).resolve().parent / "export_memory_profile.png",
        help="Output path for export RAM profile graph.",
    )
    return parser.parse_args()


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    return getattr(torch, dtype_name)


def _to_1d_tensor(x) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        t = x.detach().cpu()
    elif isinstance(x, np.ndarray):
        t = torch.from_numpy(x)
    elif isinstance(x, (list, tuple)):
        # Handle objects like [array(...)] from some backends
        if len(x) == 1 and isinstance(x[0], (np.ndarray, torch.Tensor, list, tuple)):
            return _to_1d_tensor(x[0])
        t = torch.tensor(x)
    else:
        t = torch.as_tensor(x)
    return t.reshape(-1)


def main() -> None:
    args = _parse_args()
    args.profile_output.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    config = AutoConfig.from_pretrained(args.model_name)
    # config.num_hidden_layers = 4
    config.torch_dtype = torch.float32
    # print(config)

    runner = ApiRunner(
        batch_size=1,
        tokenizer=tokenizer,
        config=config,
        prompt=[args.prompt],
        prompt_len=args.prompt_len,
        ctx_len=args.ctx_len,
    )

    # PyTorch (KV) output
    hf_model = AutoModelForCausalLM.from_pretrained(args.model_name, config=config)
    hf_tokens = runner.run_hf_model_on_pytorch(hf_model)
    print(hf_tokens)

    qeff_model = QEFFAutoModelForCausalLM.from_pretrained(args.model_name, config=config)
    pt_tokens = runner.run_kv_model_on_pytorch(qeff_model.model)
    print(pt_tokens)

    # Export timing + RAM profiling
    profiler = QEffMemoryProfiler(output_file=str(args.profile_output), verbose=True)
    profiler.start_monitoring()
    profiler.mark_operation("Export")
    export_start = time.perf_counter()
    try:
        onnx_path = qeff_model.export(
            use_dynamo=args.use_dynamo,
            use_onnx_subfunctions=args.use_onnx_subfunctions,
        )
    finally:
        profiler.stop_monitoring()

    export_elapsed = time.perf_counter() - export_start
    print(f"[TIMING] qeff_model.export: {export_elapsed:.3f} seconds")
    print(f"[MEMORY] export peak RSS: {profiler.peak_rss:.2f} MB")
    print(f"[ARTIFACT] onnx_path={onnx_path}")
    print(profiler.get_memory_report())
    profiler.generate_memory_graph(str(args.profile_output))
    print(f"[MEMORY] export profile graph saved to: {args.profile_output}")

    ort_tokens = runner.run_kv_model_on_ort(onnx_path)
    print(ort_tokens)

    compile_start = time.perf_counter()
    qpc_path = qeff_model.compile(
        prefill_seq_len=args.prompt_len,
        ctx_len=args.ctx_len,
        use_onnx_subfunctions=args.use_onnx_subfunctions,
        use_dynamo=args.use_dynamo,
        num_devices=4,
        mxfp6_matmul=True,
    )
    compile_elapsed = time.perf_counter() - compile_start
    print(f"[TIMING] qeff_model.compile: {compile_elapsed:.3f} seconds")
    print(f"[ARTIFACT] qpc_path={qpc_path}")
    print("compile done")

    print("QEff Transformed Onnx Model Outputs(AIC Backend)")
    output = qeff_model.generate(prompts=[args.prompt], tokenizer=tokenizer, automation=True,       # 👈 This dumps .raw files + for_qaic.json
    write_io=True     )
    print(output)
    print(output.generated_ids)

    # Compare all token streams with length-safe trimming
    hf_t = _to_1d_tensor(hf_tokens)
    pt_t = _to_1d_tensor(pt_tokens)
    ort_t = _to_1d_tensor(ort_tokens)
    aic_t = _to_1d_tensor(output.generated_ids)

    lengths = {
        "hf_tokens": hf_t.numel(),
        "pt_tokens": pt_t.numel(),
        "ort_tokens": ort_t.numel(),
        "aic_generated_ids": aic_t.numel(),
    }
    min_len = min(lengths.values()) if lengths else 0
    print(f"[COMPARE] original lengths: {lengths}")

    if min_len == 0:
        print("[COMPARE] Cannot compare tokens because at least one output is empty.")
        return

    hf_trim = hf_t[:min_len]
    pt_trim = pt_t[:min_len]
    ort_trim = ort_t[:min_len]
    aic_trim = aic_t[:min_len]

    # Use allclose with exact tolerance for token ids
    hf_pt_match = torch.allclose(hf_trim.float(), pt_trim.float(), rtol=0.0, atol=0.0)
    hf_ort_match = torch.allclose(hf_trim.float(), ort_trim.float(), rtol=0.0, atol=0.0)
    hf_aic_match = torch.allclose(hf_trim.float(), aic_trim.float(), rtol=0.0, atol=0.0)
    all_4_match = hf_pt_match and hf_ort_match and hf_aic_match

    print(f"[COMPARE] trimmed length used: {min_len}")
    if all_4_match:
        print("[COMPARE] All 4 outputs match.")
    else:
        print("[COMPARE] All 4 outputs do NOT match.")

    print("[COMPARE] trimmed outputs together:")
    print(f"  hf_tokens: {hf_trim.tolist()}")
    print(f"  pt_tokens: {pt_trim.tolist()}")
    print(f"  ort_tokens: {ort_trim.tolist()}")
    print(f"  aic_generated_ids: {aic_trim.tolist()}")


if __name__ == "__main__":
    main()
