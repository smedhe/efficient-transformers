#!/usr/bin/env python3
"""Generate HTML reports comparing with/without dynamo and include prompt/output text."""

from __future__ import annotations

import argparse
import ast
import csv
import html
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

MODES = ["basic", "cb", "ccl", "cb_ccl"]
METRIC_KEYS = [
    "exit_code",
    "export_peak_rss_mb",
    "export_duration_seconds",
    "compile_peak_rss_mb",
    "compile_duration_seconds",
    "qpc_dir_size_gb",
]


def safe_name(name: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    return normalized.strip("._") or "case"


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "model"


def parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def parse_log_prompts(log_file: Path) -> Dict[str, str]:
    if not log_file.is_file():
        return {}

    prompt_to_completion: Dict[str, str] = {}
    current_prompt: Optional[str] = None
    collecting_completion = False
    completion_lines: List[str] = []
    last_inputs: List[str] = []
    last_outputs: List[str] = []

    def finalize_completion() -> None:
        nonlocal current_prompt, completion_lines, collecting_completion
        if not current_prompt:
            completion_lines = []
            collecting_completion = False
            return
        completion = "\n".join(completion_lines).strip()
        completion = completion.split("input=", 1)[0].strip()
        completion = completion.split("output=", 1)[0].strip()
        prompt_to_completion[current_prompt] = completion
        current_prompt = None
        completion_lines = []
        collecting_completion = False

    for raw_line in log_file.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if stripped.startswith("input="):
            payload = stripped.split("=", 1)[1].strip()
            try:
                values = ast.literal_eval(payload)
                if isinstance(values, list):
                    last_inputs = [str(v) for v in values]
            except Exception:
                pass
        if stripped.startswith("output="):
            payload = stripped.split("=", 1)[1].strip()
            try:
                values = ast.literal_eval(payload)
                if isinstance(values, list):
                    last_outputs = [str(v) for v in values]
            except Exception:
                pass
        if collecting_completion:
            if (
                stripped.startswith("Prompt")
                or stripped.startswith("input=")
                or stripped.startswith("output=")
                or stripped.startswith("--------------------")
            ):
                finalize_completion()
                if stripped.startswith("Prompt") and ":" in stripped:
                    current_prompt = stripped.split(":", 1)[1].strip()
                continue
            completion_lines.append(line)
            continue
        if stripped.startswith("Prompt") and ":" in stripped:
            current_prompt = stripped.split(":", 1)[1].strip()
            continue
        if stripped.startswith("Completion") and ":" in stripped:
            completion_lines = [stripped.split(":", 1)[1].strip()]
            collecting_completion = True
            continue
    if collecting_completion:
        finalize_completion()

    # Some runs emit empty "Completion :" lines and only provide final input/output arrays.
    # Fill missing entries from that structured summary.
    if last_inputs and last_outputs:
        for prompt, out in zip(last_inputs, last_outputs):
            existing = prompt_to_completion.get(prompt, "").strip()
            if not existing:
                prompt_to_completion[prompt] = out if out.strip() else "EMPTY_OUTPUT"
    return prompt_to_completion


def parse_csv(summary_csv: Path) -> List[Dict[str, str]]:
    with summary_csv.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def infer_log_path(base_dir: Path, row: Dict[str, str]) -> Path:
    dtype = row.get("export_dtype", "fp16")
    model_name = row.get("model_name", "")
    case_name = row.get("name", "")
    model_dir = safe_name(model_name)
    return base_dir / f"{dtype}_export" / model_dir / case_name / f"{case_name}.log"


def html_escape(value: object) -> str:
    if value is None:
        return "-"
    text = str(value)
    if not text:
        return "-"
    return html.escape(text)


def is_failed(row: Optional[Dict[str, str]]) -> bool:
    if not row:
        return False
    exit_code = str(row.get("exit_code", "")).strip()
    error = str(row.get("error", "")).strip()
    return (exit_code not in {"", "0"}) or bool(error)


def row_key(model_name: str, dtype: str, mode: str, enable_dynamo: bool) -> Tuple[str, str, str, bool]:
    return (model_name, dtype, mode, enable_dynamo)


def load_error_details(summary_json: Path) -> Dict[Tuple[str, str, str, bool], str]:
    details: Dict[Tuple[str, str, str, bool], str] = {}
    if not summary_json.is_file():
        return details
    try:
        entries = json.loads(summary_json.read_text(encoding="utf-8"))
    except Exception:
        return details
    if not isinstance(entries, list):
        return details

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        cfg = entry.get("config", {}) or {}
        model_name = str(cfg.get("model_name", "")).strip()
        dtype = str(cfg.get("export_dtype", "")).strip()
        mode = str(cfg.get("mode", "")).strip()
        enable_dynamo = bool(cfg.get("enable_dynamo", False))
        if not model_name or not dtype or not mode:
            continue
        error_text = str(entry.get("error", "") or "").strip()
        if error_text:
            details[row_key(model_name, dtype, mode, enable_dynamo)] = error_text
    return details


def render_metrics_table(without_row: Optional[Dict[str, str]], with_row: Optional[Dict[str, str]]) -> str:
    rows = ["<table><tr><th>metric</th><th>without_dynamo</th><th>with_dynamo</th></tr>"]
    for key in METRIC_KEYS:
        left = without_row.get(key, "-") if without_row else "-"
        right = with_row.get(key, "-") if with_row else "-"
        rows.append(f"<tr><td>{html_escape(key)}</td><td>{html_escape(left)}</td><td>{html_escape(right)}</td></tr>")
    rows.append("</table>")
    return "".join(rows)


def render_prompt_table(
    without_prompts: Dict[str, str],
    with_prompts: Dict[str, str],
) -> str:
    prompt_order = list(without_prompts.keys())
    for prompt in with_prompts:
        if prompt not in without_prompts:
            prompt_order.append(prompt)

    if not prompt_order:
        return "<p>No prompt/completion text found in logs for this mode.</p>"

    out = ["<table><tr><th>#</th><th>Prompt</th><th>without_dynamo completion</th><th>with_dynamo completion</th></tr>"]
    for idx, prompt in enumerate(prompt_order, start=1):
        out.append(
            "<tr>"
            f"<td>{idx}</td>"
            f"<td><pre>{html_escape(prompt)}</pre></td>"
            f"<td><pre>{html_escape(without_prompts.get(prompt, '-'))}</pre></td>"
            f"<td><pre>{html_escape(with_prompts.get(prompt, '-'))}</pre></td>"
            "</tr>"
        )
    out.append("</table>")
    return "".join(out)


def render_error_block(label: str, text: str) -> str:
    return f"<p><strong>[ERROR] {html_escape(label)}</strong></p><pre>{html_escape(text)}</pre>"


def build_model_page(model_name: str, by_dtype_and_mode: Dict[str, Dict[str, Dict[str, object]]]) -> str:
    blocks = []
    for dtype in ["fp32", "fp16"]:
        if dtype not in by_dtype_and_mode:
            continue
        section = [f"<section><h2>{dtype.upper()} export</h2>"]
        for mode in MODES:
            mode_bucket = by_dtype_and_mode[dtype].get(mode)
            if not mode_bucket:
                continue
            without_row = mode_bucket.get("without_row")
            with_row = mode_bucket.get("with_row")
            without_prompts = mode_bucket.get("without_prompts", {})
            with_prompts = mode_bucket.get("with_prompts", {})
            without_error_detail = str(mode_bucket.get("without_error_detail", "") or "")
            with_error_detail = str(mode_bucket.get("with_error_detail", "") or "")

            cfg_source = with_row or without_row or {}
            failed = is_failed(without_row) or is_failed(with_row)
            status_text = "FAIL" if failed else "OK"
            status_cls = "fail" if failed else "ok"
            cfg_table = (
                "<table><tr><th>num_devices</th><th>prefill_seq_len</th><th>ctx_len</th><th>generation_len</th><th>enable_mx</th></tr>"
                f"<tr><td>{html_escape(cfg_source.get('num_devices', '-'))}</td>"
                f"<td>{html_escape(cfg_source.get('prefill_seq_len', '-'))}</td>"
                f"<td>{html_escape(cfg_source.get('ctx_len', '-'))}</td>"
                f"<td>{html_escape(cfg_source.get('generation_len', '-'))}</td>"
                f"<td>{html_escape(cfg_source.get('enable_mx', '-'))}</td></tr></table>"
            )

            section.append("<div class='mode'>")
            section.append(f"<h3>Mode: {html_escape(mode)} <span class='badge {status_cls}'>{status_text}</span></h3>")
            section.append("<h5>Config</h5>")
            section.append(cfg_table)
            section.append("<h5>Metrics Comparison</h5>")
            section.append(render_metrics_table(without_row, with_row))
            if failed:
                section.append("<h5>Failure Details</h5>")
                if without_error_detail:
                    section.append(render_error_block("without_dynamo", without_error_detail))
                if with_error_detail:
                    section.append(render_error_block("with_dynamo", with_error_detail))
            section.append("<h5>Prompt/Completion Comparison</h5>")
            section.append(render_prompt_table(without_prompts, with_prompts))
            section.append("</div>")
        section.append("</section>")
        blocks.append("".join(section))

    return (
        "<!doctype html><html><head><meta charset='utf-8'/><meta name='viewport' content='width=device-width, initial-scale=1'/>"
        f"<title>{html_escape(model_name)}</title>"
        "<style>"
        "body{font-family:Arial,sans-serif;margin:20px;background:#f7f7f9;color:#111}"
        "a{color:#0a4fbf}"
        ".mode{background:#fff;border:1px solid #d9dde5;border-radius:10px;padding:12px;margin:12px 0}"
        ".badge{display:inline-block;padding:3px 8px;border-radius:999px;font-weight:700;font-size:12px;vertical-align:middle}"
        ".ok{background:#d7f5df;color:#0a5f22}.fail{background:#ffe2e2;color:#8b0000}"
        "table{border-collapse:collapse;width:100%;background:#fff;margin:8px 0 12px 0}"
        "th,td{border:1px solid #e1e4eb;padding:7px;font-size:13px;vertical-align:top;text-align:left}"
        "th{background:#f1f4fa}"
        "pre{margin:0;white-space:pre-wrap;word-break:break-word;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}"
        "h1,h2,h3,h5{margin:.4rem 0}"
        "</style></head><body>"
        "<p><a href='index.html'>Back to index</a></p>"
        f"<h1>{html_escape(model_name)}</h1>" + "".join(blocks) + "</body></html>"
    )


def build_index(models: List[Tuple[str, str]], aggregate: Dict[str, Dict[str, Dict[str, Dict[str, object]]]]) -> str:
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    def get_preferred_bucket(model_name: str, mode: str) -> Dict[str, object]:
        by_dtype_and_mode = aggregate.get(model_name, {})
        for dtype in ("fp16", "fp32"):
            mode_bucket = by_dtype_and_mode.get(dtype, {}).get(mode)
            if mode_bucket:
                return mode_bucket
        return {}

    def get_model_meta(model_name: str) -> Tuple[str, str]:
        by_dtype_and_mode = aggregate.get(model_name, {})
        for dtype in ("fp16", "fp32"):
            for mode in MODES:
                mode_bucket = by_dtype_and_mode.get(dtype, {}).get(mode, {})
                for key in ("with_row", "without_row"):
                    row = mode_bucket.get(key)
                    if not row:
                        continue
                    arch = str(row.get("Architecture", "") or "").strip()
                    family = str(row.get("Model Family", "") or "").strip()
                    if arch or family:
                        return arch or "-", family or "-"
        return "-", "-"

    def render_mode_cell(mode_bucket: Dict[str, object], with_dynamo: bool) -> Tuple[str, str]:
        row_key_name = "with_row" if with_dynamo else "without_row"
        err_key_name = "with_error_detail" if with_dynamo else "without_error_detail"
        row = mode_bucket.get(row_key_name)
        if not row:
            return "missing", "<span class='badge missing'>missing</span>"
        row_data = row if isinstance(row, dict) else None
        if not row_data:
            return "missing", "<span class='badge missing'>missing</span>"
        if not is_failed(row_data):
            return "passed", "<span class='badge ok'>PASSED</span>"
        error_text = str(mode_bucket.get(err_key_name, "") or row_data.get("error", "") or "").strip()
        if not error_text:
            error_text = f"exit_code={row_data.get('exit_code', '-')}"
        return "failed", f"<span class='badge fail'>FAILED</span><pre>{html_escape(error_text)}</pre>"

    def overall_status(values: List[str]) -> Tuple[str, str]:
        if values and all(v == "passed" for v in values):
            return "passed", "ok"
        if values and all(v == "failed" for v in values):
            return "failed", "fail"
        if values and all(v == "missing" for v in values):
            return "missing", "missing"
        return "mixed", "mixed"

    cards = "".join(
        f"<div class='card'><a href='{html_escape(filename)}'>{html_escape(model_name)}</a></div>"
        for model_name, filename in models
    )
    rows = []
    for model_name, filename in models:
        basic_bucket = get_preferred_bucket(model_name, "basic")
        cb_bucket = get_preferred_bucket(model_name, "cb")
        basic_without_status, basic_without_html = render_mode_cell(basic_bucket, with_dynamo=False)
        basic_with_status, basic_with_html = render_mode_cell(basic_bucket, with_dynamo=True)
        cb_without_status, cb_without_html = render_mode_cell(cb_bucket, with_dynamo=False)
        cb_with_status, cb_with_html = render_mode_cell(cb_bucket, with_dynamo=True)
        arch, family = get_model_meta(model_name)
        overall_text, overall_class = overall_status(
            [basic_without_status, basic_with_status, cb_without_status, cb_with_status]
        )

        rows.append(
            "<tr>"
            f"<td><a href='{html_escape(filename)}'>{html_escape(model_name)}</a></td>"
            f"<td>{html_escape(arch)}</td>"
            f"<td>{html_escape(family)}</td>"
            f"<td><span class='badge {overall_class}'>{html_escape(overall_text)}</span></td>"
            f"<td>{basic_without_html}</td>"
            f"<td>{basic_with_html}</td>"
            f"<td>{cb_without_html}</td>"
            f"<td>{cb_with_html}</td>"
            "</tr>"
        )
    matrix_rows = "".join(rows)
    return (
        "<!doctype html><html><head><meta charset='utf-8'/>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'/>"
        "<title>Combined FP16 Dynamo Report</title>"
        "<style>"
        "body{font-family:Arial,sans-serif;margin:20px;background:#f7f7f9;color:#111}"
        "a{color:#0a4fbf;text-decoration:none;font-weight:600}"
        ".card,.case{background:#fff;border:1px solid #d9dde5;border-radius:10px;padding:12px;margin:12px 0}"
        ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px}"
        ".metric{font-size:28px;font-weight:700;margin-top:4px}"
        ".badge{display:inline-block;padding:3px 8px;border-radius:999px;font-weight:700;font-size:12px;vertical-align:middle}"
        ".ok{background:#d7f5df;color:#0a5f22}.fail{background:#ffe2e2;color:#8b0000}"
        ".mixed{background:#fff0c2;color:#7a4b00}.missing{background:#e7e9ef;color:#4a4f5f}"
        "table{border-collapse:collapse;width:100%;background:#fff;margin:8px 0 12px 0}"
        "th,td{border:1px solid #e1e4eb;padding:7px;font-size:13px;vertical-align:top;text-align:left}"
        "th{background:#f1f4fa}"
        "pre{margin:6px 0 0 0;white-space:pre-wrap;word-break:break-word;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12px}"
        "h1,h2,h3,h4{margin:.45rem 0}"
        ".small{font-size:12px;color:#555}"
        "</style></head><body>"
        "<h1>Combined FP16 Dynamo Report</h1>"
        f"<p class='small'>Generated {html_escape(generated_at)}.</p>"
        "<p class='small'>Each model page includes metrics and prompt/completion comparisons for with/without dynamo.</p>"
        "<div class='grid'>"
        f"<div class='card'><div class='small'>Models</div><div class='metric'>{len(models)}</div></div>"
        f"<div class='card'><div class='small'>Model Pages</div><div class='metric'>{len(models)}</div></div>"
        "</div>"
        "<h2>Model Cards</h2>"
        f"{cards}"
        "<h2>Model Matrix</h2>"
        "<table><tr>"
        "<th>Model</th>"
        "<th>Architecture</th>"
        "<th>Family</th>"
        "<th>Overall</th>"
        "<th>Basic without Dynamo</th>"
        "<th>Basic with Dynamo</th>"
        "<th>CB without Dynamo</th>"
        "<th>CB with Dynamo</th>"
        "</tr>"
        f"{matrix_rows}"
        "</table>"
        "</body></html>"
    )


def load_bucket(
    summary_csv: Path,
    base_dir: Path,
    error_details: Dict[Tuple[str, str, str, bool], str],
    aggregate: Dict[str, Dict[str, Dict[str, Dict[str, object]]]],
) -> None:
    for row in parse_csv(summary_csv):
        # Keep MX rows for cb_ccl mode (requested for report visibility); skip other MX rows
        # to avoid clobbering non-MX rows in the same mode/dynamo bucket.
        if parse_bool(row.get("enable_mx", "False")) and row.get("mode", "") != "cb_ccl":
            continue
        model_name = row.get("model_name", "")
        dtype = row.get("export_dtype", "fp16")
        mode = row.get("mode", "")
        if not model_name or mode not in MODES:
            continue

        enable_dynamo = parse_bool(row.get("enable_dynamo", "False"))
        prompt_map = parse_log_prompts(infer_log_path(base_dir, row))
        error_detail = error_details.get(
            row_key(model_name, dtype, mode, enable_dynamo), str(row.get("error", "") or "")
        )

        slot = aggregate.setdefault(model_name, {}).setdefault(dtype, {}).setdefault(mode, {})
        if enable_dynamo:
            slot["with_row"] = row
            slot["with_prompts"] = prompt_map
            slot["with_error_detail"] = error_detail
        else:
            slot["without_row"] = row
            slot["without_prompts"] = prompt_map
            slot["without_error_detail"] = error_detail


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate dynamo comparison HTML report including prompt/output text.")
    parser.add_argument("--fp16-summary", type=Path, required=True)
    parser.add_argument("--fp32-summary", type=Path, required=True)
    parser.add_argument("--fp16-base-dir", type=Path, required=True)
    parser.add_argument("--fp32-base-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    aggregate: Dict[str, Dict[str, Dict[str, Dict[str, object]]]] = {}
    fp16_errors = load_error_details(args.fp16_summary.with_name("summary.json"))
    fp32_errors = load_error_details(args.fp32_summary.with_name("summary.json"))
    load_bucket(args.fp16_summary, args.fp16_base_dir, fp16_errors, aggregate)
    load_bucket(args.fp32_summary, args.fp32_base_dir, fp32_errors, aggregate)

    model_links: List[Tuple[str, str]] = []
    for model_name in sorted(aggregate.keys(), key=str.lower):
        filename = f"{slugify(model_name)}.html"
        page = build_model_page(model_name, aggregate[model_name])
        (args.out_dir / filename).write_text(page, encoding="utf-8")
        model_links.append((model_name, filename))

    (args.out_dir / "index.html").write_text(build_index(model_links, aggregate), encoding="utf-8")
    print(f"Report generated at: {args.out_dir}")
    print(f"Models: {len(model_links)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
