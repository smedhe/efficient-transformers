# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import argparse
import csv
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


def find_repo_root(start: Path) -> Path:
    for parent in (start, *start.parents):
        if (parent / "QEfficient").is_dir() and (parent / "scripts" / "memory_profiling" / "__init__.py").is_file():
            return parent
    raise RuntimeError(f"Could not find efficient-transformers repo root from {start}")


REPO_ROOT = find_repo_root(Path(__file__).resolve().parent)
TARGET_SCRIPT = Path(__file__).resolve().with_name("run_subfunction_dynamo_split_metrics.py")
DEFAULT_HF_HUB_CACHE = "/home/huggingface_hub"
DEFAULT_TMPDIR = "/home/vtirumal/tmp_dir/"
DEFAULT_QEFF_HOME = "../qeff_dynamo"
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_GENERATION_LEN = 100
DEFAULT_NUM_CORES = 16
DEFAULT_SKIP_GENERATE = False
DEFAULT_PROFILER_SAMPLING_INTERVAL = 0.1
VALID_MODES = {"basic", "cb", "ccl", "cb_ccl"}
VALID_EXPORT_DTYPES = {"fp16", "fp32"}
MODE_FLAGS = {"basic": (), "cb": ("--cb",), "ccl": ("--ccl",), "cb_ccl": ("--cb_ccl",)}
NA_VALUE = "NA"
ERROR_EXCERPT_MAX_LINES = 40
CSV_ERROR_MAX_CHARS = 1000

TIME_METRIC_KEYS = {
    "User time (seconds)": "user_time_seconds",
    "System time (seconds)": "system_time_seconds",
    "Percent of CPU this job got": "cpu_percent",
    "Elapsed (wall clock) time (h:mm:ss or m:ss)": "wall_time",
    "Maximum resident set size (kbytes)": "max_rss_kb",
    "Average resident set size (kbytes)": "avg_rss_kb",
    "Major (requiring I/O) page faults": "major_page_faults",
    "Minor (reclaiming a frame) page faults": "minor_page_faults",
    "Exit status": "time_exit_status",
}


@dataclass(frozen=True)
class CsvCase:
    row_number: int
    architecture: str
    model_family: str
    name: str
    mode: str
    enable_dynamo: bool
    enable_subfun: bool
    enable_mx: bool
    from_config: bool
    trust_remote_code: bool
    export_dtype: str
    model_name: Optional[str]
    num_hidden_layers: Optional[int]
    full_model: bool
    prefill_seq_len: Optional[int]
    ctx_len: Optional[int]
    generation_len: Optional[int]
    num_devices: Optional[int]
    output_dir: Optional[str]
    extra_args: Sequence[str]


@dataclass
class CaseResult:
    name: str
    row_number: int
    exit_code: int
    command: str
    log_file: str
    time_metrics_file: str
    case_output_dir: str
    phase_metrics_file: Optional[str]
    profile_graph_file: Optional[str]
    config: Dict[str, object]
    process_metrics: Dict[str, str]
    phase_metrics: Dict[str, object]
    error: Optional[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run export/compile memory metrics cases from a CSV file.")
    parser.add_argument("--csv", required=True, help="CSV file with one split metrics run configuration per row.")
    parser.add_argument(
        "--row",
        action="append",
        type=int,
        default=[],
        help="Run only the given physical CSV row number. Repeat to run multiple rows.",
    )
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
        help="Python executable for launching the target script. Default: current interpreter.",
    )
    parser.add_argument(
        "--export-dtype",
        choices=("fp16", "fp32"),
        default=None,
        help="Override CSV export_dtype for all selected rows.",
    )
    parser.add_argument(
        "--from-config",
        action="store_true",
        help="Build model via AutoModelForCausalLM.from_config for all selected rows (no pretrained weights).",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True for all selected rows.",
    )
    parser.add_argument(
        "--log-dir",
        default=None,
        help=(
            "Directory for per-case logs and summaries. Default: run_logs/subfunction_dynamo_metrics_csv_<timestamp>."
        ),
    )
    parser.add_argument(
        "--cleanup-qeff-home-after-row",
        action="store_true",
        help="Remove all files/directories inside QEFF_HOME after each completed row.",
    )
    return parser.parse_args()


def clean_cell(row: Dict[str, str], key: str) -> str:
    value = row.get(key, "")
    return value.strip() if value is not None else ""


def parse_bool(raw_value: str, default: bool = False) -> bool:
    value = raw_value.strip().lower()
    if not value:
        return default
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"invalid boolean value: {raw_value!r}")


def parse_optional_int(raw_value: str, key: str) -> Optional[int]:
    if not raw_value:
        return None
    try:
        return int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{key} must be an integer, got {raw_value!r}") from exc


def parse_optional_float(raw_value: str, key: str) -> Optional[float]:
    if not raw_value:
        return None
    try:
        return float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{key} must be a float, got {raw_value!r}") from exc


def parse_case(row: Dict[str, str], row_number: int) -> Optional[CsvCase]:
    enabled = parse_bool(clean_cell(row, "enabled"), default=True)
    if not enabled:
        return None

    name = clean_cell(row, "name") or f"row_{row_number}"
    if name.startswith("#"):
        return None

    mode = clean_cell(row, "mode") or "basic"
    if mode not in VALID_MODES:
        raise ValueError(f"row {row_number}: mode must be one of {sorted(VALID_MODES)}, got {mode!r}")
    export_dtype = (clean_cell(row, "export_dtype") or "fp16").lower()
    if export_dtype not in VALID_EXPORT_DTYPES:
        raise ValueError(
            f"row {row_number}: export_dtype must be one of {sorted(VALID_EXPORT_DTYPES)}, got {export_dtype!r}"
        )

    num_hidden_layers = parse_optional_int(clean_cell(row, "num_hidden_layers"), "num_hidden_layers")
    full_model = parse_bool(clean_cell(row, "full_model"))
    if full_model and num_hidden_layers is not None:
        raise ValueError(f"row {row_number}: full_model and num_hidden_layers cannot both be set")

    return CsvCase(
        row_number=row_number,
        architecture=clean_cell(row, "Architecture"),
        model_family=clean_cell(row, "Model Family"),
        name=name,
        mode=mode,
        enable_dynamo=parse_bool(clean_cell(row, "enable_dynamo")),
        enable_subfun=parse_bool(clean_cell(row, "enable_subfun")),
        enable_mx=parse_bool(clean_cell(row, "enable_mx")),
        from_config=parse_bool(clean_cell(row, "from_config")),
        trust_remote_code=parse_bool(clean_cell(row, "trust_remote_code")),
        export_dtype=export_dtype,
        model_name=clean_cell(row, "model_name") or None,
        num_hidden_layers=num_hidden_layers,
        full_model=full_model,
        prefill_seq_len=parse_optional_int(clean_cell(row, "prefill_seq_len"), "prefill_seq_len"),
        ctx_len=parse_optional_int(clean_cell(row, "ctx_len"), "ctx_len"),
        generation_len=parse_optional_int(clean_cell(row, "generation_len"), "generation_len"),
        num_devices=parse_optional_int(clean_cell(row, "num_devices"), "num_devices"),
        output_dir=clean_cell(row, "output_dir") or None,
        extra_args=tuple(shlex.split(clean_cell(row, "extra_args"))),
    )


def read_cases(csv_path: Path) -> List[CsvCase]:
    cases: List[CsvCase] = []
    with csv_path.open(newline="", encoding="utf-8") as csv_handle:
        reader = csv.DictReader(csv_handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header: {csv_path}")
        for row_number, row in enumerate(reader, start=2):
            case = parse_case(row, row_number)
            if case is not None:
                cases.append(case)
    if not cases:
        raise ValueError(f"CSV file has no enabled cases: {csv_path}")
    return cases


def filter_cases_by_row(cases: Sequence[CsvCase], row_numbers: Sequence[int]) -> List[CsvCase]:
    if not row_numbers:
        return list(cases)
    requested_rows = set(row_numbers)
    selected_cases = [case for case in cases if case.row_number in requested_rows]
    selected_rows = {case.row_number for case in selected_cases}
    missing_rows = sorted(requested_rows - selected_rows)
    if missing_rows:
        raise ValueError(f"requested CSV row(s) not found or not enabled: {missing_rows}")
    return selected_cases


def default_prefill_seq_len(mode: str) -> int:
    return 32 if mode in {"cb", "ccl", "cb_ccl"} else 1


def default_num_devices(mode: str) -> int:
    return 4 if mode in {"cb", "ccl", "cb_ccl"} else 1


def effective_model_name(case: CsvCase) -> str:
    return case.model_name or DEFAULT_MODEL_NAME


def effective_export_dtype(case: CsvCase, export_dtype_override: Optional[str]) -> str:
    return export_dtype_override or case.export_dtype


def effective_prefill_seq_len(case: CsvCase) -> int:
    return case.prefill_seq_len if case.prefill_seq_len is not None else default_prefill_seq_len(case.mode)


def effective_generation_len(case: CsvCase) -> int:
    return case.generation_len if case.generation_len is not None else DEFAULT_GENERATION_LEN


def effective_num_devices(case: CsvCase) -> int:
    return case.num_devices if case.num_devices is not None else default_num_devices(case.mode)


def effective_num_cores() -> int:
    return DEFAULT_NUM_CORES


def effective_skip_generate() -> bool:
    return DEFAULT_SKIP_GENERATE


def effective_profiler_sampling_interval() -> float:
    return DEFAULT_PROFILER_SAMPLING_INTERVAL


def add_optional_arg(command: List[str], flag: str, value: Optional[object]) -> None:
    if value is not None:
        command.extend([flag, str(value)])


def build_case_args(
    case: CsvCase,
    case_output_dir: Path,
    export_dtype_override: Optional[str],
    force_from_config: bool,
    force_trust_remote_code: bool,
) -> List[str]:
    command_args = list(MODE_FLAGS[case.mode])
    command_args.extend(["--case-name", case.name])
    command_args.extend(["--export-dtype", effective_export_dtype(case, export_dtype_override)])
    if case.enable_dynamo:
        command_args.append("--enable_dynamo")
    if case.enable_subfun:
        command_args.append("--enable_subfun")
    if case.enable_mx:
        command_args.append("--enable_mx")
    if case.from_config or force_from_config:
        command_args.append("--from-config")
    if case.trust_remote_code or force_trust_remote_code:
        command_args.append("--trust-remote-code")
    add_optional_arg(command_args, "--model-name", case.model_name)
    if case.full_model:
        command_args.append("--full_model")
    else:
        add_optional_arg(command_args, "--num-hidden-layers", case.num_hidden_layers)
    add_optional_arg(command_args, "--prefill-seq-len", case.prefill_seq_len)
    add_optional_arg(command_args, "--ctx-len", case.ctx_len)
    add_optional_arg(command_args, "--generation-len", case.generation_len)
    add_optional_arg(command_args, "--num-devices", case.num_devices)
    command_args.extend(["--num-cores", str(effective_num_cores())])
    command_args.extend(["--profiler-sampling-interval", str(effective_profiler_sampling_interval())])
    command_args.extend(["--output-dir", str(case_output_dir)])
    if effective_skip_generate():
        command_args.append("--skip-generate")
    command_args.extend(case.extra_args)
    return command_args


def case_config(
    case: CsvCase, export_dtype_override: Optional[str], force_from_config: bool, force_trust_remote_code: bool
) -> Dict[str, object]:
    return {
        "row_number": case.row_number,
        "Architecture": case.architecture,
        "Model Family": case.model_family,
        "mode": case.mode,
        "export_dtype": effective_export_dtype(case, export_dtype_override),
        "enable_dynamo": case.enable_dynamo,
        "enable_subfun": case.enable_subfun,
        "enable_mx": case.enable_mx,
        "from_config": case.from_config or force_from_config,
        "trust_remote_code": case.trust_remote_code or force_trust_remote_code,
        "model_name": effective_model_name(case),
        "num_hidden_layers": case.num_hidden_layers,
        "full_model": case.full_model,
        "prefill_seq_len": effective_prefill_seq_len(case),
        "ctx_len": case.ctx_len if case.ctx_len is not None else 128,
        "generation_len": effective_generation_len(case),
        "num_devices": effective_num_devices(case),
        "num_cores": effective_num_cores(),
        "skip_generate": effective_skip_generate(),
        "profiler_sampling_interval": effective_profiler_sampling_interval(),
        "output_dir": case.output_dir,
        "extra_args": list(case.extra_args),
    }


def ensure_env() -> Dict[str, str]:
    env = os.environ.copy()
    env.setdefault("HF_HUB_CACHE", DEFAULT_HF_HUB_CACHE)
    env.setdefault("TMPDIR", DEFAULT_TMPDIR)
    env.setdefault("QEFF_HOME", DEFAULT_QEFF_HOME)
    env.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    return env


def collect_case_artifact_dirs(result: CaseResult) -> List[Path]:
    phase_metrics = result.phase_metrics or {}
    onnx_path = phase_metrics.get("onnx_path")
    qpc_path = phase_metrics.get("qpc_path")
    candidates: List[Path] = []

    if isinstance(qpc_path, str) and qpc_path.strip():
        candidates.append(Path(qpc_path))

    if isinstance(onnx_path, str) and onnx_path.strip():
        onnx_file = Path(onnx_path)
        if onnx_file.parent != onnx_file:
            candidates.append(onnx_file.parent)

    # Deduplicate while preserving order.
    unique: List[Path] = []
    seen = set()
    for path in candidates:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def cleanup_case_artifacts(result: CaseResult) -> List[str]:
    removed: List[str] = []
    # Remove deeper paths first to avoid parent-before-child conflicts.
    for artifact_dir in sorted(collect_case_artifact_dirs(result), key=lambda p: len(p.parts), reverse=True):
        if not artifact_dir.exists():
            continue
        if artifact_dir.is_dir():
            shutil.rmtree(artifact_dir)
            removed.append(str(artifact_dir))
    return removed


def safe_name(name: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    return normalized.strip("._") or "case"


def parse_time_metrics(metrics_file: Path) -> Dict[str, str]:
    if not metrics_file.is_file():
        return {}
    metrics = {}
    for line in metrics_file.read_text(encoding="utf-8", errors="replace").splitlines():
        if ":" not in line:
            continue
        key, value = line.strip().split(":", 1)
        if key in TIME_METRIC_KEYS:
            metrics[TIME_METRIC_KEYS[key]] = value.strip()
    return metrics


def find_latest_file(directory: Path, pattern: str) -> Optional[Path]:
    if not directory.is_dir():
        return None
    files = sorted(directory.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    return files[0] if files else None


def load_phase_metrics(case_output_dir: Path) -> Dict[str, object]:
    metrics_file = find_latest_file(case_output_dir, "phase_metrics_*.json")
    if metrics_file is None:
        metrics_file = find_latest_file(case_output_dir, "split_metrics_*.json")
    if metrics_file is None:
        return {}
    try:
        return json.loads(metrics_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def extract_error_from_log(log_file: Path, max_lines: int = ERROR_EXCERPT_MAX_LINES) -> Optional[str]:
    if not log_file.is_file():
        return f"log file not found: {log_file}"

    lines = log_file.read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines:
        return "log file is empty"

    traceback_start = None
    for index, line in enumerate(lines):
        if "Traceback (most recent call last):" in line:
            traceback_start = index

    if traceback_start is not None:
        excerpt = lines[traceback_start:]
    else:
        non_empty_lines = [line.rstrip() for line in lines if line.strip()]
        excerpt = non_empty_lines[-max_lines:]

    if len(excerpt) > max_lines:
        excerpt = ["..."] + excerpt[-max_lines:]

    error_text = "\n".join(line.rstrip() for line in excerpt).strip()
    return error_text or "no error text found in log"


def format_csv_error(error: Optional[str]) -> str:
    if not error:
        return ""
    formatted = re.sub(r"\s+", " ", error).strip()
    if len(formatted) <= CSV_ERROR_MAX_CHARS:
        return formatted
    return f"{formatted[: CSV_ERROR_MAX_CHARS - 3]}..."


SUMMARY_CSV_FIELDS = [
    "Architecture",
    "Model Family",
    "name",
    "row_number",
    "exit_code",
    "mode",
    "export_dtype",
    "enable_dynamo",
    "enable_subfun",
    "enable_mx",
    "from_config",
    "trust_remote_code",
    "model_name",
    "num_hidden_layers",
    "full_model",
    "prefill_seq_len",
    "ctx_len",
    "generation_len",
    "num_devices",
    "num_cores",
    "onnx_path",
    "qpc_path",
    "onnx_preexisting_before_export",
    "qpc_preexisting_before_compile",
    "onnx_reused_across_cases",
    "qpc_reused_across_cases",
    "qpc_dir_size_gb",
    "export_peak_rss_mb",
    "export_duration_seconds",
    "export_rss_delta_mb",
    "compile_peak_rss_mb",
    "compile_duration_seconds",
    "compile_rss_delta_mb",
    "error",
]


def get_phase_metrics(phase_metrics: Dict[str, object], phase_name: str) -> Dict[str, object]:
    for phase in phase_metrics.get("phases", []) or []:
        if isinstance(phase, dict) and phase.get("name") == phase_name:
            return phase
    return {}


def add_phase_columns(
    row: Dict[str, object], phase_metrics: Dict[str, object], phase_name: str, prefix: str, use_na: bool
) -> None:
    phase = {} if use_na else get_phase_metrics(phase_metrics, phase_name)
    row[f"{prefix}_peak_rss_mb"] = phase.get("peak_rss_mb", NA_VALUE)
    row[f"{prefix}_duration_seconds"] = phase.get("duration_seconds", NA_VALUE)
    row[f"{prefix}_rss_delta_mb"] = phase.get("rss_delta_mb", NA_VALUE)


def bool_or_na(value: Optional[object], use_na: bool) -> object:
    if use_na:
        return NA_VALUE
    if value is None:
        return NA_VALUE
    if isinstance(value, bool):
        return value
    return str(value)


def build_path_counts(results: Iterable[CaseResult], key: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for result in results:
        phase_metrics = result.phase_metrics or {}
        path_value = phase_metrics.get(key)
        if isinstance(path_value, str) and path_value.strip():
            counts[path_value] = counts.get(path_value, 0) + 1
    return counts


def qpc_dir_size_gb(qpc_path: object) -> object:
    if not isinstance(qpc_path, str) or not qpc_path.strip():
        return NA_VALUE
    path = Path(qpc_path)
    if not path.is_dir():
        return NA_VALUE
    total_bytes = 0
    try:
        for entry in path.rglob("*"):
            if entry.is_file():
                total_bytes += entry.stat().st_size
    except OSError:
        return NA_VALUE
    return f"{total_bytes / (1024**3):.3f}"


def persist_phase_metrics(result: CaseResult) -> None:
    if not result.phase_metrics_file or not result.phase_metrics:
        return
    metrics_file = Path(result.phase_metrics_file)
    if not metrics_file.is_absolute():
        metrics_file = REPO_ROOT / metrics_file
    if not metrics_file.parent.is_dir():
        return
    metrics_file.write_text(json.dumps(result.phase_metrics, indent=2), encoding="utf-8")


def build_summary_csv_row(
    result: CaseResult, onnx_counts: Dict[str, int], qpc_counts: Dict[str, int]
) -> Dict[str, object]:
    phase_metrics = result.phase_metrics or {}
    onnx_path = phase_metrics.get("onnx_path", NA_VALUE)
    qpc_path = phase_metrics.get("qpc_path", NA_VALUE)
    row: Dict[str, object] = {
        "Architecture": result.config.get("Architecture", ""),
        "Model Family": result.config.get("Model Family", ""),
        "name": result.name,
        "row_number": result.row_number,
        "exit_code": result.exit_code,
        "mode": result.config.get("mode"),
        "export_dtype": result.config.get("export_dtype"),
        "enable_dynamo": result.config.get("enable_dynamo"),
        "enable_subfun": result.config.get("enable_subfun"),
        "enable_mx": result.config.get("enable_mx"),
        "from_config": result.config.get("from_config"),
        "trust_remote_code": result.config.get("trust_remote_code"),
        "model_name": result.config.get("model_name"),
        "num_hidden_layers": result.config.get("num_hidden_layers"),
        "full_model": result.config.get("full_model"),
        "prefill_seq_len": result.config.get("prefill_seq_len"),
        "ctx_len": result.config.get("ctx_len"),
        "generation_len": result.config.get("generation_len"),
        "num_devices": result.config.get("num_devices"),
        "num_cores": result.config.get("num_cores"),
    }
    has_metrics = bool(phase_metrics)
    row["onnx_path"] = onnx_path if has_metrics else NA_VALUE
    row["qpc_path"] = qpc_path if has_metrics else NA_VALUE
    row["onnx_preexisting_before_export"] = bool_or_na(
        phase_metrics.get("onnx_preexisting_before_export"), not has_metrics
    )
    row["qpc_preexisting_before_compile"] = bool_or_na(
        phase_metrics.get("qpc_preexisting_before_compile"), not has_metrics
    )
    row["onnx_reused_across_cases"] = (
        bool_or_na(onnx_counts.get(onnx_path, 0) > 1, not has_metrics)
        if isinstance(onnx_path, str) and onnx_path != NA_VALUE
        else NA_VALUE
    )
    row["qpc_reused_across_cases"] = (
        bool_or_na(qpc_counts.get(qpc_path, 0) > 1, not has_metrics)
        if isinstance(qpc_path, str) and qpc_path != NA_VALUE
        else NA_VALUE
    )
    if has_metrics and isinstance(phase_metrics.get("qpc_dir_size_gb"), str):
        row["qpc_dir_size_gb"] = phase_metrics["qpc_dir_size_gb"]
    else:
        row["qpc_dir_size_gb"] = qpc_dir_size_gb(qpc_path) if has_metrics else NA_VALUE
    add_phase_columns(row, phase_metrics, "Export", "export", not has_metrics)
    add_phase_columns(row, phase_metrics, "Compile", "compile", not has_metrics)
    row["error"] = format_csv_error(result.error)
    return row


def write_csv_summary(results: Iterable[CaseResult], summary_file: Path) -> None:
    results_list = list(results)
    onnx_counts = build_path_counts(results_list, "onnx_path")
    qpc_counts = build_path_counts(results_list, "qpc_path")
    with summary_file.open("w", newline="", encoding="utf-8") as csv_handle:
        writer = csv.DictWriter(csv_handle, fieldnames=SUMMARY_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for result in results_list:
            writer.writerow(build_summary_csv_row(result, onnx_counts, qpc_counts))


def write_markdown_summary(results: Iterable[CaseResult], summary_file: Path) -> None:
    lines = ["# Subfunction Dynamo Export/Compile Metrics CSV Summary", ""]
    for result in results:
        lines.extend(
            [
                f"## {result.name}",
                "",
                f"- row_number: {result.row_number}",
                f"- exit_code: {result.exit_code}",
                f"- log_file: {result.log_file}",
                f"- time_metrics_file: {result.time_metrics_file}",
                f"- case_output_dir: {result.case_output_dir}",
                f"- phase_metrics_file: {result.phase_metrics_file}",
                f"- profile_graph_file: {result.profile_graph_file}",
            ]
        )
        if result.error:
            lines.append("- error:")
            lines.extend(f"  {line}" for line in result.error.splitlines())
        else:
            lines.append("- error: None")
        lines.extend(
            [
                f"- command: `{result.command}`",
                "- config:",
            ]
        )
        for key, value in result.config.items():
            lines.append(f"  - {key}: {value}")
        lines.append("- process_metrics:")
        if result.process_metrics:
            for key, value in result.process_metrics.items():
                lines.append(f"  - {key}: {value}")
        else:
            lines.append("  - none found")
        lines.append("- phase_metrics:")
        if result.phase_metrics:
            for key in (
                "export_wall_time_seconds",
                "compile_wall_time_seconds",
                "global_peak_rss_mb",
                "global_peak_operation",
                "onnx_path",
                "qpc_path",
                "qpc_dir_size_gb",
                "onnx_preexisting_before_export",
                "qpc_preexisting_before_compile",
            ):
                if key in result.phase_metrics:
                    lines.append(f"  - {key}: {result.phase_metrics[key]}")
            phases = result.phase_metrics.get("phases") or []
            if phases:
                lines.append("  - phases:")
                for phase in phases:
                    lines.append(
                        "    - "
                        f"{phase.get('name')}: peak_rss_mb={phase.get('peak_rss_mb')}, "
                        f"duration_seconds={phase.get('duration_seconds')}, "
                        f"rss_delta_mb={phase.get('rss_delta_mb')}"
                    )
        else:
            lines.append("  - none found")
        lines.append("")
    summary_file.write_text("\n".join(lines), encoding="utf-8")


def run_case(args: argparse.Namespace, case: CsvCase, env: Dict[str, str], log_dir: Path) -> CaseResult:
    case_log_name = safe_name(case.name)
    case_export_dtype = effective_export_dtype(case, args.export_dtype)
    dtype_root_dir = log_dir / f"{case_export_dtype}_export"
    model_dir = dtype_root_dir / safe_name(effective_model_name(case))
    case_dir = model_dir / case_log_name
    case_dir.mkdir(parents=True, exist_ok=True)
    case_output_dir = Path(case.output_dir) if case.output_dir else case_dir
    case_output_dir.mkdir(parents=True, exist_ok=True)
    log_file = case_dir / f"{case_log_name}.log"
    time_metrics_file = case_dir / f"{case_log_name}.time.txt"
    command = [
        args.python_bin,
        str(TARGET_SCRIPT),
        *build_case_args(case, case_output_dir, args.export_dtype, args.from_config, args.trust_remote_code),
    ]
    command_text = shlex.join(command)
    config = case_config(case, args.export_dtype, args.from_config, args.trust_remote_code)

    time_bin = shutil.which("time")
    timed_command = [time_bin, "-v", "-o", str(time_metrics_file), *command] if time_bin else command

    with log_file.open("w", encoding="utf-8") as log_handle:
        log_handle.write(f"===== {case.name} =====\n")
        log_handle.write(f"CSV row: {case.row_number}\n")
        log_handle.write(f"Command: {command_text}\n")
        log_handle.write(f"Time metrics file: {time_metrics_file}\n")
        log_handle.write(f"Case output dir: {case_output_dir}\n")
        log_handle.write(f"Working directory: {REPO_ROOT}\n")
        log_handle.write(f"HF_HUB_CACHE: {env['HF_HUB_CACHE']}\n")
        log_handle.write(f"TMPDIR: {env['TMPDIR']}\n")
        log_handle.write(f"QEFF_HOME: {env['QEFF_HOME']}\n")
        log_handle.write("\n")
        log_handle.flush()
        start_time = time.perf_counter()
        process = subprocess.run(
            timed_command,
            cwd=REPO_ROOT,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        elapsed_seconds = time.perf_counter() - start_time

    process_metrics = parse_time_metrics(time_metrics_file)
    if not process_metrics:
        process_metrics = {"wall_time_seconds": f"{elapsed_seconds:.3f}"}
    phase_metrics = load_phase_metrics(case_output_dir)
    phase_metrics_file = phase_metrics.get("metrics_path") if phase_metrics else None
    profile_graph_file = phase_metrics.get("profile_graph_path") if phase_metrics else None
    log_error = extract_error_from_log(log_file) if process.returncode != 0 else None
    error = f"exit_code={process.returncode}\n{log_error}" if log_error else None

    return CaseResult(
        name=case.name,
        row_number=case.row_number,
        exit_code=process.returncode,
        command=command_text,
        log_file=str(log_file),
        time_metrics_file=str(time_metrics_file),
        case_output_dir=str(case_output_dir),
        phase_metrics_file=phase_metrics_file,
        profile_graph_file=profile_graph_file,
        config=config,
        process_metrics=process_metrics,
        phase_metrics=phase_metrics,
        error=error,
    )


def main() -> int:
    args = parse_args()
    csv_path = Path(args.csv)
    cases = filter_cases_by_row(read_cases(csv_path), args.row)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_log_dir = REPO_ROOT / "run_logs" / f"subfunction_dynamo_metrics_csv_{timestamp}"
    log_dir = Path(args.log_dir) if args.log_dir else default_log_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    env = ensure_env()
    results: List[CaseResult] = []
    summary_json = log_dir / "summary.json"
    summary_md = log_dir / "summary.md"
    summary_csv = log_dir / "summary.csv"

    print(f"Using CSV: {csv_path}")
    print(f"Using Python: {args.python_bin}")
    print(f"Using logs: {log_dir}")
    print(f"Using HF_HUB_CACHE={env['HF_HUB_CACHE']}")
    print(f"Using TMPDIR={env['TMPDIR']}")
    print(f"Using QEFF_HOME={env['QEFF_HOME']}")
    if args.cleanup_qeff_home_after_row:
        print("Artifact cleanup per row: enabled (onnx/qpc dirs only)")

    for case in cases:
        print(f"\n===== running {case.name} row={case.row_number} =====")
        result = run_case(args, case, env, log_dir)
        results.append(result)
        # Keep summary.csv current after every row to make progress visible
        # during long multi-case runs.
        write_csv_summary(results, summary_csv)
        print(f"exit_code={result.exit_code} log={result.log_file}")
        if result.process_metrics:
            wall_time = result.process_metrics.get("wall_time", result.process_metrics.get("wall_time_seconds"))
            print(f"  wall_time: {wall_time}")
            if "max_rss_kb" in result.process_metrics:
                print(f"  process_max_rss_kb: {result.process_metrics['max_rss_kb']}")
        if result.phase_metrics:
            print(f"  export_wall_time_seconds: {result.phase_metrics.get('export_wall_time_seconds')}")
            print(f"  compile_wall_time_seconds: {result.phase_metrics.get('compile_wall_time_seconds')}")
            print(f"  global_peak_rss_mb: {result.phase_metrics.get('global_peak_rss_mb')}")
            print(f"  onnx_path: {result.phase_metrics.get('onnx_path')}")
            print(f"  qpc_path: {result.phase_metrics.get('qpc_path')}")
            print(
                "  preexisting: "
                f"onnx={result.phase_metrics.get('onnx_preexisting_before_export')} "
                f"qpc={result.phase_metrics.get('qpc_preexisting_before_compile')}"
            )
            for phase in result.phase_metrics.get("phases", []):
                print(f"  {phase.get('name')}_peak_rss_mb: {phase.get('peak_rss_mb')}")
        if result.error:
            print(f"  error: {format_csv_error(result.error)}")
        if args.cleanup_qeff_home_after_row:
            try:
                # Capture qpc directory size before deleting artifacts so summary.csv
                # remains stable across final summary regeneration.
                qpc_path = (result.phase_metrics or {}).get("qpc_path")
                if result.phase_metrics and isinstance(qpc_path, str):
                    result.phase_metrics["qpc_dir_size_gb"] = qpc_dir_size_gb(qpc_path)
                    persist_phase_metrics(result)
                    write_csv_summary(results, summary_csv)
                removed_paths = cleanup_case_artifacts(result)
                if removed_paths:
                    print(f"  cleaned_artifacts: {', '.join(removed_paths)}")
                else:
                    print("  cleaned_artifacts: none")
            except Exception as cleanup_error:
                print(f"  cleanup_artifacts_error: {cleanup_error}")

    summary_json.write_text(
        json.dumps([asdict(result) for result in results], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_markdown_summary(results, summary_md)
    write_csv_summary(results, summary_csv)

    print("\nExport/compile metrics CSV run complete")
    print(f"Summary JSON: {summary_json}")
    print(f"Summary MD: {summary_md}")
    print(f"Summary CSV: {summary_csv}")
    return 1 if any(result.exit_code != 0 for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
