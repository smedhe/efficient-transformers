# Subfunction Dynamo Export/Compile Metrics Automation

This folder contains automation for running `run_subfunction_dynamo.py`-style cases while measuring **Export** and **Compile** as separate phases.

## Files

- `run_subfunction_dynamo_split_metrics.py`: runs one case, calls `export()`, then `compile(onnx_path=...)`, and records phase metrics.
- `run_subfunction_dynamo_split_metrics_csv.py`: reads a CSV matrix and runs the target script for each enabled row.
- `fp16_all_combinations.csv`: sample full-model CSV matrix for FP16 runs.
- `fp32_all_combinations.csv`: sample full-model CSV matrix for FP32 runs.

## What The Automation Does

For each enabled CSV row, the CSV runner:

1. Builds command-line args for `run_subfunction_dynamo_split_metrics.py`.
2. Uses environment variables with env-first behavior:
   - `HF_HUB_CACHE` (default: `/home/huggingface_hub`)
   - `TMPDIR` (default: `/home/vtirumal/tmp_dir/`)
   - `QEFF_HOME` (default: `../qeff_dynamo`)
   - `HF_HUB_ENABLE_HF_TRANSFER` (default: `1`)
3. Runs from the efficient-transformers repo root.
4. Captures stdout/stderr to per-case `.log`.
5. Captures `/usr/bin/time -v` to per-case `.time.txt`.
6. Loads per-case `phase_metrics_*.json` (fallback-compatible with older `split_metrics_*.json`).
7. Writes `summary.json`, `summary.md`, and `summary.csv` under the selected log directory.

The target script marks operations:

1. `Export`
2. `Compile`
3. `Generate` (only when generation is not skipped)

## Log And Output Layout

### Default per-case layout from CSV runner

Under `--log-dir`, files are grouped by export dtype, model, then test case:

```text
<log-dir>/
  fp16_export/
    <model_name_sanitized>/
      <case_name>/
        <case_name>.log
        <case_name>.time.txt
        phase_metrics_<case_name>_<timestamp>.json
        memory_profile_<case_name>_<timestamp>.png
  fp32_export/
    ...
  summary.json
  summary.md
  summary.csv
```

### ONNX and QPC artifact location

ONNX and QPC are not written to the log folder by default. They follow QEff cache behavior under `QEFF_HOME` (same style as direct `test_subfn_cb.py` runs).

## Example Commands

Run one row:

```bash
cd /home/vtirumal/dynamo/efficient-transformers

/home/vtirumal/dynamo/qenv/bin/python3 \
  examples/text_generation/dynamo_scripts/run_subfunction_dynamo_split_metrics_csv.py \
  --csv examples/text_generation/dynamo_scripts/fp16_all_combinations.csv \
  --row 2 \
  --log-dir /home/vtirumal/dynamo/run_logs/subfunction_dynamo_metrics_csv_row2
```

Run all enabled rows:

```bash
/home/vtirumal/dynamo/qenv/bin/python3 \
  examples/text_generation/dynamo_scripts/run_subfunction_dynamo_split_metrics_csv.py \
  --csv examples/text_generation/dynamo_scripts/fp16_all_combinations.csv \
  --log-dir /home/vtirumal/dynamo/run_logs/subfunction_dynamo_metrics_csv_all
```

## Input CSV Columns

Sample CSVs:

- `fp16_all_combinations.csv`
- `fp32_all_combinations.csv`

| Column | Meaning |
| --- | --- |
| `name` | Case name used for log and metric/profile filename stems. |
| `enabled` | If false, row is skipped. Defaults to true if blank. |
| `mode` | One of `basic`, `cb`, `ccl`, `cb_ccl`. Defaults to `basic` if blank. |
| `enable_dynamo` | Adds `--enable_dynamo` when true. |
| `enable_subfun` | Adds `--enable_subfun` when true. |
| `enable_mx` | Adds `--enable_mx` when true (enables both `mxfp6_matmul` and `mxint8_kv_cache`). |
| `export_dtype` | Export/load dtype: `fp16` or `fp32`. Defaults to `fp16` if blank. |
| `model_name` | Hugging Face model id/path. Defaults to `Qwen/Qwen2.5-3B-Instruct` if blank. |
| `num_hidden_layers` | Overrides `config.num_hidden_layers`. Leave blank with `full_model=true`. |
| `full_model` | Use full model config without overriding layer count. |
| `prefill_seq_len` | Compile prefill seq len. Defaults: `1` for `basic`, `32` for `cb/ccl/cb_ccl`. |
| `ctx_len` | Compile context length. Defaults to `128` if blank. |
| `generation_len` | Generation length. Defaults to `100` if blank. |
| `num_devices` | Compile device count. Defaults: `1` for `basic`, `4` for `cb/ccl/cb_ccl`. |
| `output_dir` | Optional per-row override for the target script output directory. |
| `extra_args` | Extra CLI args forwarded to the target script. |

Common runner settings are constants in `run_subfunction_dynamo_split_metrics_csv.py` (not CSV columns):

- `num_cores=16`
- `skip_generate=False`
- `profiler_sampling_interval=0.1`

## Output CSV (`summary.csv`)

`summary.csv` includes run identity/config plus key phase metrics and reuse diagnostics:

```csv
name,row_number,exit_code,mode,export_dtype,enable_dynamo,enable_subfun,enable_mx,model_name,num_hidden_layers,full_model,prefill_seq_len,ctx_len,generation_len,num_devices,num_cores,onnx_path,qpc_path,onnx_preexisting_before_export,qpc_preexisting_before_compile,onnx_reused_across_cases,qpc_reused_across_cases,qpc_dir_size_gb,export_peak_rss_mb,export_duration_seconds,export_rss_delta_mb,compile_peak_rss_mb,compile_duration_seconds,compile_rss_delta_mb,error
```

### Key path/reuse columns

- `onnx_path`: ONNX path used by compile.
- `qpc_path`: compiled QPC path.
- `onnx_preexisting_before_export`: `True` if ONNX already existed before export call.
- `qpc_preexisting_before_compile`: `True` if QPC already existed before compile call.
- `onnx_reused_across_cases`: `True` if same ONNX path appears in more than one case in that CSV run.
- `qpc_reused_across_cases`: `True` if same QPC path appears in more than one case in that CSV run.
- `qpc_dir_size_gb`: compiled QPC directory size in GB (when available).

For failed rows, metric/path columns are `NA` only when no phase metrics JSON is available; otherwise collected values are still reported.

## Notes

- Memory profile PNG generation is optional. If `matplotlib` is missing, JSON/MD/CSV summaries are still produced.
- CSV automation continues to next row when a row fails.
- Exit code is non-zero if any selected row fails.

## Why Dynamo Export Can Use More Time/RAM

`enable_dynamo=true` does not only "switch ONNX export on". It changes the export pipeline to include PyTorch graph capture (`torch.export`) before ONNX translation.

Typical extra work in dynamo export:

1. Capturing Python model execution into an intermediate graph (FX/exported program).
2. Building guards for shapes/constraints to ensure graph validity.
3. Decomposing/normalizing ops before ONNX conversion.
4. Tracking aliasing/mutations more strictly when stateful behavior exists.

This can increase both:

- `export_duration_seconds`
- `export_peak_rss_mb`

even when final compile/generation behavior is valid.

### KV-cache side-effect warnings and slowdown

If logs contain warnings such as:

- `side effects happened in model.forward`
- references like `past_key_value.layers[...]`

then export detected mutable/stateful behavior (common in decode KV-cache updates). In this case, capture tends to be more conservative (fewer aggressive simplifications), which can further increase export time and memory.

### Practical interpretation

- Dynamo export can be faster for some models/configs and slower for others.
- Compare **like-for-like** rows (same model/mode/dtype/seq lengths/devices) using:
  - `export_duration_seconds`
  - `export_peak_rss_mb`
- Do not infer runtime decode speed from export time alone.

## HTML Report Automation

Use `generate_dynamo_comparison_report.py` to generate per-model HTML comparison pages (without-dynamo vs with-dynamo) from run outputs.

- Inputs:
  - `summary.csv` for FP16 run directory
  - `summary.csv` for FP32 run directory
  - corresponding run base dirs (used to parse per-case `.log` prompt/completion text)
- Error details:
  - reads `summary.json` (same folder as each `summary.csv`) and adds failure details (including compiler errors) in failed sections.

Example:

```bash
/home/vtirumal/dynamo/qenv/bin/python3 \
  /home/vtirumal/dynamo/efficient-transformers/examples/text_generation/dynamo_scripts/generate_dynamo_comparison_report.py \
  --fp16-summary /home/vtirumal/dynamo/fp16_all_models/summary.csv \
  --fp32-summary /home/vtirumal/dynamo/fp32_all_models/summary.csv \
  --fp16-base-dir /home/vtirumal/dynamo/fp16_all_models \
  --fp32-base-dir /home/vtirumal/dynamo/fp32_all_models \
  --out-dir /home/vtirumal/dynamo/reports_fp32_fp16_all_models
```

Generated output:

```text
<out-dir>/
  index.html
  <model-1>.html
  <model-2>.html
  ...
```
