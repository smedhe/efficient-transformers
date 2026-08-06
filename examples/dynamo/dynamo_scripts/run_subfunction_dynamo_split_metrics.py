# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import argparse
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM
from scripts.memory_profiling import QEffMemoryProfiler


def find_repo_root(start: Path) -> Path:
    for parent in (start, *start.parents):
        if (parent / "QEfficient").is_dir() and (parent / "scripts" / "memory_profiling" / "__init__.py").is_file():
            return parent
    raise RuntimeError(f"Could not find efficient-transformers repo root from {start}")


REPO_ROOT = find_repo_root(Path(__file__).resolve().parent)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_PROMPTS = ["My name is"]
DEFAULT_CB_PROMPTS = [
    "My name is",
    "Explain quantum computing",
    "Where is Taj Mahal located?",
    "Hey, are you conscious? Can you talk to me?",
]
DEFAULT_CB_CCL_PROMPTS = ["My name is", "Explain quantum computing"]
DEFAULT_CCL_LENGTHS = [1024, 2048]
DEFAULT_AIC_HW_VERSION = "ai100"
DEFAULT_HF_HUB_CACHE = "/home/huggingface_hub"
DEFAULT_TMPDIR = "/home/vtirumal/tmp_dir/"
DEFAULT_QEFF_HOME = "../qeff_dynamo"


torch.manual_seed(42)


@dataclass(frozen=True)
class RunMode:
    continuous_batching: bool = False
    ccl_enabled: bool = False

    @property
    def name(self) -> str:
        if self.continuous_batching and self.ccl_enabled:
            return "cb_ccl"
        if self.continuous_batching:
            return "cb"
        if self.ccl_enabled:
            return "ccl"
        return "basic"


@dataclass(frozen=True)
class RunConfig:
    case_name: Optional[str]
    export_dtype: str
    model_name: str
    num_hidden_layers: Optional[int]
    full_model: bool
    use_dynamo: bool
    use_onnx_subfunctions: bool
    from_config: bool
    trust_remote_code: bool
    prefill_seq_len: int
    ctx_len: int
    generation_len: int
    num_devices: int
    num_cores: int
    mxfp6_matmul: bool
    mxint8_kv_cache: bool
    full_batch_size: Optional[int]
    comp_ctx_lengths_prefill: Optional[List[int]]
    comp_ctx_lengths_decode: Optional[List[int]]
    output_dir: Path
    profiler_sampling_interval: float
    generate: bool


@dataclass(frozen=True)
class PhaseMetric:
    name: str
    duration_seconds: float
    start_rss_mb: Optional[float]
    end_rss_mb: Optional[float]
    peak_rss_mb: Optional[float]
    peak_vms_mb: Optional[float]
    rss_delta_mb: Optional[float]
    sample_count: int


@dataclass(frozen=True)
class RunMetrics:
    mode: str
    export_dtype: str
    model_name: str
    artifact_dir: str
    onnx_path: Optional[str]
    qpc_path: Optional[str]
    qpc_dir_size_gb: Optional[str]
    onnx_preexisting_before_export: bool
    qpc_preexisting_before_compile: bool
    metrics_path: str
    profile_graph_path: Optional[str]
    export_wall_time_seconds: float
    compile_wall_time_seconds: float
    global_peak_rss_mb: float
    global_peak_operation: Optional[str]
    phases: List[PhaseMetric]
    generated_ids: Optional[str]


def ensure_env() -> None:
    os.environ.setdefault("HF_HUB_CACHE", DEFAULT_HF_HUB_CACHE)
    os.environ.setdefault("TMPDIR", DEFAULT_TMPDIR)
    os.environ.setdefault("QEFF_HOME", DEFAULT_QEFF_HOME)
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    Path(os.environ["TMPDIR"]).mkdir(parents=True, exist_ok=True)


def resolve_mode(args: argparse.Namespace) -> RunMode:
    continuous_batching = args.cb or args.cb_ccl
    ccl_enabled = args.ccl or args.cb_ccl
    return RunMode(continuous_batching=continuous_batching, ccl_enabled=ccl_enabled)


def resolve_prompts(mode: RunMode) -> List[str]:
    if mode.continuous_batching and mode.ccl_enabled:
        return DEFAULT_CB_CCL_PROMPTS
    if mode.continuous_batching:
        return DEFAULT_CB_PROMPTS
    return DEFAULT_PROMPTS


def resolve_run_config(args: argparse.Namespace, mode: RunMode) -> RunConfig:
    ctx_len = args.ctx_len if args.ctx_len is not None else 128
    if mode.continuous_batching or mode.ccl_enabled:
        default_prefill_seq_len = 32
        default_num_devices = 4
    else:
        default_prefill_seq_len = 1
        default_num_devices = 1

    prompts = resolve_prompts(mode)
    full_batch_size = len(prompts) if mode.continuous_batching else None

    return RunConfig(
        case_name=args.case_name,
        export_dtype=args.export_dtype,
        model_name=args.model_name or DEFAULT_MODEL_NAME,
        num_hidden_layers=args.num_hidden_layers,
        full_model=args.full_model,
        use_dynamo=args.enable_dynamo,
        use_onnx_subfunctions=args.enable_subfun,
        from_config=args.from_config,
        trust_remote_code=args.trust_remote_code,
        prefill_seq_len=args.prefill_seq_len if args.prefill_seq_len is not None else default_prefill_seq_len,
        ctx_len=ctx_len,
        generation_len=args.generation_len,
        num_devices=args.num_devices if args.num_devices is not None else default_num_devices,
        num_cores=args.num_cores,
        mxfp6_matmul=args.enable_mx,
        mxint8_kv_cache=args.enable_mx,
        full_batch_size=full_batch_size,
        comp_ctx_lengths_prefill=DEFAULT_CCL_LENGTHS if mode.ccl_enabled else None,
        comp_ctx_lengths_decode=DEFAULT_CCL_LENGTHS if mode.ccl_enabled else None,
        output_dir=Path(args.output_dir),
        profiler_sampling_interval=args.profiler_sampling_interval,
        generate=not args.skip_generate,
    )


def safe_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    return normalized.strip("._") or None


def torch_dtype_from_export_dtype(export_dtype: str):
    if export_dtype == "fp16":
        return torch.float16
    if export_dtype == "fp32":
        return torch.float32
    raise ValueError(f"unsupported export dtype: {export_dtype!r}")


def load_tokenizer_and_config(run_config: RunConfig):
    tokenizer = AutoTokenizer.from_pretrained(run_config.model_name, trust_remote_code=run_config.trust_remote_code)
    config = AutoConfig.from_pretrained(run_config.model_name, trust_remote_code=run_config.trust_remote_code)
    if not run_config.full_model and run_config.num_hidden_layers is not None:
        config.num_hidden_layers = run_config.num_hidden_layers
    if not hasattr(config, "max_position_embeddings"):
        config.max_position_embeddings = getattr(config, "n_positions", 2048)

    if not hasattr(config, "ffn_hidden_size"):
        config.ffn_hidden_size = 4 * config.hidden_size
    if not hasattr(config, "activation"):
        config.activation = getattr(config, "hidden_act", "gelu")
    config.torch_dtype = torch_dtype_from_export_dtype(run_config.export_dtype)
    return tokenizer, config


def build_qeff_model(run_config: RunConfig, config, mode: RunMode):
    kwargs: Dict[str, Any] = {}
    if mode.continuous_batching:
        kwargs["continuous_batching"] = True
    if mode.ccl_enabled:
        kwargs["qaic_config"] = {"ccl_enabled": True}
    if run_config.from_config:
        if hasattr(config, "tie_word_embeddings"):
            config.tie_word_embeddings = False
        hf_model = AutoModelForCausalLM.from_config(config, trust_remote_code=run_config.trust_remote_code)
        return QEFFAutoModelForCausalLM(hf_model, **kwargs)
    return QEFFAutoModelForCausalLM.from_pretrained(
        run_config.model_name,
        config=config,
        trust_remote_code=run_config.trust_remote_code,
        **kwargs,
    )


def build_compile_kwargs(run_config: RunConfig, mode: RunMode, onnx_path: str) -> Dict[str, Any]:
    compile_kwargs: Dict[str, Any] = {
        "onnx_path": onnx_path,
        "prefill_seq_len": run_config.prefill_seq_len,
        "ctx_len": run_config.ctx_len,
        "dynamo": run_config.use_dynamo,
        "use_onnx_subfunctions": run_config.use_onnx_subfunctions,
        "num_devices": run_config.num_devices,
        "num_cores": run_config.num_cores,
        "mxfp6_matmul": run_config.mxfp6_matmul,
        "mxint8_kv_cache": run_config.mxint8_kv_cache,
        "aic_hw_version": DEFAULT_AIC_HW_VERSION,
    }
    if mode.continuous_batching:
        compile_kwargs["full_batch_size"] = run_config.full_batch_size
    if mode.ccl_enabled:
        compile_kwargs["comp_ctx_lengths_prefill"] = run_config.comp_ctx_lengths_prefill
        compile_kwargs["comp_ctx_lengths_decode"] = run_config.comp_ctx_lengths_decode
    return compile_kwargs


def iter_phase_windows(profiler: QEffMemoryProfiler) -> Sequence[Tuple[str, datetime, datetime]]:
    operations = list(profiler.operations)
    windows = []
    for index, (start_time, operation_name) in enumerate(operations):
        if operation_name in {"Initialization", "Completion"}:
            continue
        if index + 1 < len(operations):
            end_time = operations[index + 1][0]
        elif profiler.samples:
            end_time = profiler.samples[-1].timestamp
        else:
            end_time = start_time
        windows.append((operation_name, start_time, end_time))
    return windows


def phase_metric_from_samples(
    profiler: QEffMemoryProfiler, name: str, start_time: datetime, end_time: datetime
) -> PhaseMetric:
    samples = [sample for sample in profiler.samples if start_time <= sample.timestamp <= end_time]
    duration_seconds = max(0.0, (end_time - start_time).total_seconds())
    if not samples:
        return PhaseMetric(
            name=name,
            duration_seconds=duration_seconds,
            start_rss_mb=None,
            end_rss_mb=None,
            peak_rss_mb=None,
            peak_vms_mb=None,
            rss_delta_mb=None,
            sample_count=0,
        )

    start_rss_mb = samples[0].rss_mb
    end_rss_mb = samples[-1].rss_mb
    return PhaseMetric(
        name=name,
        duration_seconds=duration_seconds,
        start_rss_mb=start_rss_mb,
        end_rss_mb=end_rss_mb,
        peak_rss_mb=max(sample.rss_mb for sample in samples),
        peak_vms_mb=max(sample.vms_mb for sample in samples),
        rss_delta_mb=end_rss_mb - start_rss_mb,
        sample_count=len(samples),
    )


def collect_phase_metrics(profiler: QEffMemoryProfiler) -> List[PhaseMetric]:
    return [
        phase_metric_from_samples(profiler, operation_name, start_time, end_time)
        for operation_name, start_time, end_time in iter_phase_windows(profiler)
    ]


def serialize_generated_ids(generated_ids: Any) -> str:
    return repr(generated_ids)


def dir_size_gb(path_value: Optional[str]) -> Optional[str]:
    if not path_value:
        return None
    path = Path(path_value)
    if not path.is_dir():
        return None
    total_bytes = 0
    try:
        for entry in path.rglob("*"):
            if entry.is_file():
                total_bytes += entry.stat().st_size
    except OSError:
        return None
    return f"{total_bytes / (1024**3):.3f}"


def run_export_compile_metrics(run_config: RunConfig, mode: RunMode, prompts: List[str]) -> RunMetrics:
    ensure_env()
    run_config.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer, config = load_tokenizer_and_config(run_config)
    qeff_model = build_qeff_model(run_config, config, mode)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifact_dir = None
    case_label = safe_name(run_config.case_name)
    if case_label:
        metrics_stem = f"phase_metrics_{case_label}_{timestamp}"
        profile_stem = f"memory_profile_{case_label}_{timestamp}"
    else:
        metrics_stem = f"phase_metrics_{mode.name}_{timestamp}"
        profile_stem = f"memory_profile_{mode.name}_{timestamp}"
    metrics_path = run_config.output_dir / f"{metrics_stem}.json"
    profile_graph_path = run_config.output_dir / f"{profile_stem}.png"

    profiler = QEffMemoryProfiler(
        sampling_interval=run_config.profiler_sampling_interval,
        output_file=str(profile_graph_path),
        verbose=False,
        track_child_processes=True,
        child_scan_interval=min(run_config.profiler_sampling_interval, 1.0),
    )

    profiler.start_monitoring()
    generated_ids = None
    onnx_path: Optional[str] = None
    qpc_path: Optional[str] = None
    qpc_size_gb: Optional[str] = None
    onnx_preexisting_before_export = False
    qpc_preexisting_before_compile = False
    export_wall_time_seconds = 0.0
    compile_wall_time_seconds = 0.0
    run_error: Optional[BaseException] = None
    try:
        profiler.mark_operation("Export")
        export_start = time.perf_counter()
        export_start_epoch = time.time()
        onnx_path = qeff_model.export(
            dynamo=run_config.use_dynamo,
            use_onnx_subfunctions=run_config.use_onnx_subfunctions,
            offload_pt_weights=False,
        )
        artifact_dir = str(Path(onnx_path).parent)
        onnx_path_obj = Path(onnx_path)
        onnx_preexisting_before_export = onnx_path_obj.is_file() and onnx_path_obj.stat().st_mtime < (
            export_start_epoch - 1e-3
        )
        export_wall_time_seconds = time.perf_counter() - export_start
        print(f"ONNX exported to: {onnx_path}")
        print(f"ONNX preexisting before export: {onnx_preexisting_before_export}")
        print(f">>>>>>>>  export time : {export_wall_time_seconds:.2f} secs")

        profiler.mark_operation("Compile")
        compile_start = time.perf_counter()
        compile_start_epoch = time.time()
        qpc_path = qeff_model.compile(**build_compile_kwargs(run_config, mode, onnx_path))
        qpc_program_path = Path(qpc_path) / "programqpc.bin"
        qpc_preexisting_before_compile = qpc_program_path.is_file() and qpc_program_path.stat().st_mtime < (
            compile_start_epoch - 1e-3
        )
        compile_wall_time_seconds = time.perf_counter() - compile_start
        qpc_size_gb = dir_size_gb(qpc_path)
        print(f"QPC compiled to: {qpc_path}")
        print(f"QPC dir size GB: {qpc_size_gb if qpc_size_gb is not None else 'NA'}")
        print(f"QPC preexisting before compile: {qpc_preexisting_before_compile}")
        print(f">>>>>>>> compile time : {compile_wall_time_seconds:.2f} secs")

        if run_config.generate:
            profiler.mark_operation("Generate")
            try:
                output = qeff_model.generate(
                    prompts=prompts,
                    tokenizer=tokenizer,
                    generation_len=run_config.generation_len,
                    automation=True,
                )
            except BaseException as generate_error:
                print(f"Generate failed on first attempt: {generate_error}. Retrying after 15 seconds...")
                time.sleep(15)
                output = qeff_model.generate(
                    prompts=prompts,
                    tokenizer=tokenizer,
                    generation_len=run_config.generation_len,
                    automation=True,
                )
            generated_ids = output.generated_ids
            print(generated_ids)
    except BaseException as error:
        run_error = error
    finally:
        profiler.stop_monitoring()

    phases = collect_phase_metrics(profiler)
    profile_graph_output_path: Optional[str] = str(profile_graph_path)
    try:
        profiler.generate_memory_graph(str(profile_graph_path))
        print(f"Memory profile graph: {profile_graph_path}")
    except (ImportError, ModuleNotFoundError) as error:
        profile_graph_output_path = None
        print(f"Memory profile graph skipped: {error}")

    metrics = RunMetrics(
        mode=mode.name,
        export_dtype=run_config.export_dtype,
        model_name=run_config.model_name,
        artifact_dir=artifact_dir or "",
        onnx_path=str(onnx_path) if onnx_path is not None else None,
        qpc_path=str(qpc_path) if qpc_path is not None else None,
        qpc_dir_size_gb=qpc_size_gb,
        onnx_preexisting_before_export=onnx_preexisting_before_export,
        qpc_preexisting_before_compile=qpc_preexisting_before_compile,
        metrics_path=str(metrics_path),
        profile_graph_path=profile_graph_output_path,
        export_wall_time_seconds=export_wall_time_seconds,
        compile_wall_time_seconds=compile_wall_time_seconds,
        global_peak_rss_mb=profiler.peak_rss,
        global_peak_operation=profiler.peak_operation,
        phases=phases,
        generated_ids=serialize_generated_ids(generated_ids) if generated_ids is not None else None,
    )

    metrics_path.write_text(json.dumps(asdict(metrics), indent=2), encoding="utf-8")
    print(f"Metrics JSON: {metrics_path}")
    print(profiler.get_memory_report())
    if run_error is not None:
        raise run_error
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run run_subfunction_dynamo with separate export/compile phases and memory metrics."
    )
    parser.add_argument("--cb", action="store_true", help="Enable continuous batching mode.")
    parser.add_argument("--ccl", action="store_true", help="Enable compute-context-length mode.")
    parser.add_argument(
        "--cb-ccl",
        "--cb_ccl",
        "-cb_ccl",
        dest="cb_ccl",
        action="store_true",
        help="Enable both continuous batching and CCL mode.",
    )
    parser.add_argument("--enable-dynamo", "--enable_dynamo", action="store_true", help="Pass dynamo=True.")
    parser.add_argument(
        "--enable-subfun", "--enable_subfun", action="store_true", help="Pass use_onnx_subfunctions=True."
    )
    parser.add_argument(
        "--enable-mx", "--enable_mx", action="store_true", help="Enable mxfp6_matmul and mxint8_kv_cache."
    )
    parser.add_argument(
        "--from-config",
        action="store_true",
        help="Build model from HF config (no pretrained weights) and wrap it with QEFFAutoModelForCausalLM.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to HF loaders.",
    )
    parser.add_argument(
        "--export-dtype",
        choices=("fp16", "fp32"),
        default="fp16",
        help="Dtype for model loading/export. Default: fp16.",
    )
    parser.add_argument(
        "--case-name", type=str, default=None, help="Optional case label for output metric/profile filenames."
    )
    parser.add_argument("--model-name", type=str, default=None, help="Override the default model.")
    parser.add_argument("--num-hidden-layers", type=int, default=None, help="Override config.num_hidden_layers.")
    parser.add_argument(
        "--full_model", action="store_true", help="Use the full model config without overriding layers."
    )
    parser.add_argument("--prefill-seq-len", type=int, default=None, help="Override prefill_seq_len.")
    parser.add_argument("--ctx-len", type=int, default=None, help="Override ctx_len.")
    parser.add_argument("--generation-len", type=int, default=100, help="Generation length for AIC generate.")
    parser.add_argument("--num-devices", type=int, default=None, help="Override compile num_devices.")
    parser.add_argument("--num-cores", type=int, default=16, help="Override compile num_cores.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="run_logs/subfunction_dynamo_metrics",
        help="Directory for metrics JSON and memory profile graph.",
    )
    parser.add_argument(
        "--profiler-sampling-interval",
        type=float,
        default=0.1,
        help="Memory profiler sampling interval in seconds.",
    )
    parser.add_argument("--skip-generate", action="store_true", help="Only export and compile; skip AIC generation.")
    args = parser.parse_args()
    if args.full_model and args.num_hidden_layers is not None:
        parser.error("--full_model and --num-hidden-layers cannot be used together")
    return args


def main() -> None:
    args = parse_args()
    mode = resolve_mode(args)
    run_config = resolve_run_config(args, mode)
    prompts = resolve_prompts(mode)
    print(
        f">>>>>> mode: {mode.name} <<<<<<< export_dtype: {run_config.export_dtype} "
        f"<<<<<< dynamo: {run_config.use_dynamo} "
        f"<<<<<< use_onnx_subfunctions: {run_config.use_onnx_subfunctions} "
        f"<<<<<< from_config: {run_config.from_config} "
        f"<<<<<< trust_remote_code: {run_config.trust_remote_code} "
        f"<<<<<< enable_mx: {run_config.mxfp6_matmul} "
        f"<<<<<< aic_hw_version: {DEFAULT_AIC_HW_VERSION} <<<<<<<<<<<"
    )
    if mode.continuous_batching:
        print(f"Continuous batching full_batch_size: {run_config.full_batch_size}")
    if mode.ccl_enabled:
        print(f"CCL prefill lengths: {run_config.comp_ctx_lengths_prefill}")
        print(f"CCL decode lengths: {run_config.comp_ctx_lengths_decode}")
    run_export_compile_metrics(run_config, mode, prompts)


if __name__ == "__main__":
    main()
