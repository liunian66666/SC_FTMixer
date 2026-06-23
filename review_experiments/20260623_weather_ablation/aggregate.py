#!/usr/bin/env python3
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def trunc3(value):
    return math.trunc(float(value) * 1000.0) / 1000.0


def fmt_raw(value):
    return f"{float(value):.10f}"


def fmt_trunc(value):
    return f"{trunc3(value):.3f}"


root = Path(sys.argv[1]).resolve()
jobs_root = root / "jobs"
out_root = root / "summary"
out_root.mkdir(parents=True, exist_ok=True)
queue_path = root / "p0_jobs.tsv"
if queue_path.exists():
    expected_runs = max(
        0, sum(1 for line in queue_path.open(encoding="utf-8") if line.strip()) - 1
    )
else:
    expected_runs = 0

rows = []
failures = []
for job_dir in sorted(jobs_root.iterdir()):
    if not job_dir.is_dir():
        continue
    meta_path = job_dir / "meta.json"
    metrics_path = job_dir / "metrics.json"
    if not meta_path.exists():
        continue
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("exit_code") != 0 or not metrics_path.exists():
        failures.append(meta)
        continue
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    cfg = meta.get("config", {})
    rows.append({
        "job_id": meta["job_id"],
        "dataset": meta["dataset"],
        "input_len": meta["input_len"],
        "pred_len": meta["pred_len"],
        "variant": meta["variant"],
        "seed": meta["seed"],
        "mse": metrics["mse"],
        "mae": metrics["mae"],
        "mse_trunc3": trunc3(metrics["mse"]),
        "mae_trunc3": trunc3(metrics["mae"]),
        "duration_seconds": meta["duration_seconds"],
        "peak_gpu_memory_mib": meta["peak_gpu_memory_mib"],
        "parameters_total": meta["parameters_total"],
        "batch_size": cfg.get("batch_size"),
        "hidden": cfg.get("hidden"),
        "cycle_len": cfg.get("cycle_len"),
        "phase_mode": cfg.get("phase_mode"),
        "gate_init": cfg.get("gate_init"),
        "use_sde": cfg.get("use_sde"),
        "use_global_sde": cfg.get("use_global_sde"),
        "use_calendar_sde": cfg.get("use_calendar_sde"),
        "use_dynamic_filter": cfg.get("use_dynamic_filter"),
        "fix_calendar_gate": cfg.get("fix_calendar_gate"),
        "checkpoint_path": meta.get("checkpoint_path"),
        "stdout_log": meta.get("stdout_log"),
        "gpu_log": meta.get("gpu_log"),
    })

fieldnames = list(rows[0].keys()) if rows else [
    "job_id", "dataset", "input_len", "pred_len", "variant", "seed",
    "mse", "mae", "mse_trunc3", "mae_trunc3"
]
with (out_root / "p0_per_run.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
(out_root / "p0_per_run.json").write_text(
    json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
)
(out_root / "p0_failures.json").write_text(
    json.dumps(failures, indent=2, ensure_ascii=False), encoding="utf-8"
)

groups = defaultdict(list)
for row in rows:
    groups[(row["dataset"], row["pred_len"], row["variant"])].append(row)

summary = []
for (dataset, pred_len, variant), group in sorted(groups.items()):
    mses = [float(x["mse"]) for x in group]
    maes = [float(x["mae"]) for x in group]
    durations = [float(x["duration_seconds"]) for x in group]
    memories = [float(x["peak_gpu_memory_mib"]) for x in group]
    mse_std = statistics.stdev(mses) if len(mses) > 1 else 0.0
    mae_std = statistics.stdev(maes) if len(maes) > 1 else 0.0
    item = {
        "dataset": dataset,
        "pred_len": pred_len,
        "variant": variant,
        "n": len(group),
        "seeds": ",".join(str(x["seed"]) for x in sorted(group, key=lambda x: x["seed"])),
        "mse_mean": statistics.mean(mses),
        "mse_std": mse_std,
        "mae_mean": statistics.mean(maes),
        "mae_std": mae_std,
        "mse_mean_trunc3": trunc3(statistics.mean(mses)),
        "mse_std_trunc3": trunc3(mse_std),
        "mae_mean_trunc3": trunc3(statistics.mean(maes)),
        "mae_std_trunc3": trunc3(mae_std),
        "duration_mean_seconds": statistics.mean(durations),
        "peak_gpu_memory_max_mib": max(memories),
        "parameters_total": group[0]["parameters_total"],
    }
    summary.append(item)

summary_fields = list(summary[0].keys()) if summary else [
    "dataset", "pred_len", "variant", "n", "seeds",
    "mse_mean", "mse_std", "mae_mean", "mae_std"
]
with (out_root / "p0_summary.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=summary_fields)
    writer.writeheader()
    writer.writerows(summary)
(out_root / "p0_summary.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
)

latex = [
    r"\begin{tabular}{llrcc}",
    r"\toprule",
    r"Dataset & Variant & $H$ & MSE & MAE \\",
    r"\midrule",
]
last_dataset = None
for item in summary:
    if last_dataset is not None and item["dataset"] != last_dataset:
        latex.append(r"\midrule")
    last_dataset = item["dataset"]
    if item["n"] > 1:
        mse = f'{fmt_trunc(item["mse_mean"])} $\\pm$ {fmt_trunc(item["mse_std"])}'
        mae = f'{fmt_trunc(item["mae_mean"])} $\\pm$ {fmt_trunc(item["mae_std"])}'
    else:
        mse = fmt_trunc(item["mse_mean"])
        mae = fmt_trunc(item["mae_mean"])
    latex.append(
        f'{item["dataset"]} & {item["variant"]} & {item["pred_len"]} & {mse} & {mae} \\\\'
    )
latex.extend([r"\bottomrule", r"\end{tabular}"])
(out_root / "p0_ablation_table.tex").write_text(
    "\n".join(latex) + "\n", encoding="utf-8"
)

status = {
    "completed_runs": len(rows),
    "failed_runs": len(failures),
    "expected_runs": expected_runs,
    "completion_fraction": (
        len(rows) / expected_runs if expected_runs else 0.0
    ),
}
(out_root / "status.json").write_text(
    json.dumps(status, indent=2), encoding="utf-8"
)
print(json.dumps(status))
