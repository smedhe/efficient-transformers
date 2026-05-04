# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Generate a ready-to-paste REFERENCE_DATA block from captured JSONL step history."""

import argparse
import json
from pathlib import Path
from pprint import pformat


def load_capture_records(capture_file: str):
    """Load JSONL records and keep the latest record per scenario_key."""
    latest_by_scenario = {}
    with open(capture_file, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {capture_file} at line {line_no}: {exc}") from exc

            scenario_key = record.get("scenario_key")
            if not scenario_key:
                raise ValueError(f"Missing 'scenario_key' in {capture_file} at line {line_no}.")

            latest_by_scenario[scenario_key] = record
    return latest_by_scenario


def build_reference_data(records_by_scenario):
    """Build REFERENCE_DATA structure from captured records."""
    reference_data = {}
    for scenario_key in sorted(records_by_scenario):
        record = records_by_scenario[scenario_key]
        reference_data[scenario_key] = {
            "description": f"Baseline regenerated from captured run for {record.get('config_name', scenario_key)}",
            "train_step_losses": record.get("train_step_losses", []),
            "eval_step_losses": record.get("eval_step_losses", []),
            "train_step_metrics": record.get("train_step_metrics", []),
            "eval_step_metrics": record.get("eval_step_metrics", []),
        }
    return reference_data


def render_reference_block(reference_data):
    """Render a ready-to-paste Python block for reference_data.py."""
    return "REFERENCE_DATA = " + pformat(reference_data, width=120, sort_dicts=False) + "\n"


def _resolve_capture_file(capture_file: str) -> str:
    """Resolve capture file against CWD, then script directory, with clear diagnostics."""
    direct = Path(capture_file)
    if direct.is_file():
        return str(direct)

    script_relative = Path(__file__).resolve().parent / capture_file
    if script_relative.is_file():
        return str(script_relative)

    # Common case: user passes a repo-relative path while running from repo root.
    repo_relative = Path(__file__).resolve().parents[4] / capture_file
    if repo_relative.is_file():
        return str(repo_relative)

    raise FileNotFoundError(
        "Capture file not found. Checked:\n"
        f"  - {direct.resolve()}\n"
        f"  - {script_relative}\n"
        f"  - {repo_relative}\n"
        "Run the integrated test first with capture enabled, e.g.:\n"
        "  QEFF_ALLOW_REFERENCE_MISMATCH=1 QEFF_REFERENCE_CAPTURE_FILE=QEfficient/finetune/experimental/tests/reference_capture.jsonl "
        "pytest -s -q QEfficient/finetune/experimental/tests/test_integrated.py -k llama_alpaca --maxfail=1"
    )


def _resolve_reference_file(reference_file: str) -> str:
    """Resolve target reference_data.py path for --apply mode."""
    direct = Path(reference_file)
    if direct.is_file():
        return str(direct)

    script_relative = Path(__file__).resolve().parent / reference_file
    if script_relative.is_file():
        return str(script_relative)

    repo_relative = Path(__file__).resolve().parents[4] / reference_file
    if repo_relative.is_file():
        return str(repo_relative)

    raise FileNotFoundError(
        "Reference file not found. Checked:\n"
        f"  - {direct.resolve()}\n"
        f"  - {script_relative}\n"
        f"  - {repo_relative}"
    )


def apply_reference_block(reference_file: str, content: str):
    """Replace existing REFERENCE_DATA block in reference_data.py in-place."""
    target = Path(reference_file)
    original = target.read_text(encoding="utf-8")

    marker = "REFERENCE_DATA ="
    marker_idx = original.find(marker)
    if marker_idx == -1:
        raise ValueError(f"Could not locate '{marker}' in {reference_file}")

    # Keep file header/content before REFERENCE_DATA and replace the block till EOF.
    prefix = original[:marker_idx]
    new_text = prefix + content
    target.write_text(new_text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--capture-file",
        default="reference_capture.jsonl",
        help="Path to JSONL file produced by test_integrated.py capture hook.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional output file for generated block. If omitted, prints to stdout.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply generated REFERENCE_DATA block directly to reference_data.py.",
    )
    parser.add_argument(
        "--reference-file",
        default="QEfficient/finetune/experimental/tests/reference_data.py",
        help="Path to reference_data.py used with --apply.",
    )
    args = parser.parse_args()

    resolved_capture_file = _resolve_capture_file(args.capture_file)
    records = load_capture_records(resolved_capture_file)
    if not records:
        raise ValueError(f"No capture records found in: {args.capture_file}")

    reference_data = build_reference_data(records)
    content = render_reference_block(reference_data)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(content)
        print(f"Generated REFERENCE_DATA block written to: {args.output}")

    if args.apply:
        resolved_reference_file = _resolve_reference_file(args.reference_file)
        apply_reference_block(resolved_reference_file, content)
        print(f"Applied REFERENCE_DATA block to: {resolved_reference_file}")

    if not args.output and not args.apply:
        print(content)


if __name__ == "__main__":
    main()