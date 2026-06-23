#!/usr/bin/env python3
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for path in sorted(root.glob("*.json")):
    data = json.loads(path.read_text(encoding="utf-8"))
    rows.append({
        "model": data["model"],
        "parameters": data["parameters_trainable"],
        "epoch_seconds": data["one_epoch_seconds"],
        "train_ms_iter": data["stable_train"]["mean_ms"],
        "inference_ms_batch": data["inference_batch"]["mean_ms"],
        "single_sample_ms": data["single_sample_latency"]["mean_ms"],
        "peak_train_mib": data["peak_train_memory_mib"],
        "peak_inference_mib": data["peak_inference_memory_mib"],
        "batch_size": data["batch_size"],
        "precision": data["precision"],
        "gpu": data["gpu"],
    })

fields = list(rows[0]) if rows else [
    "model", "parameters", "epoch_seconds", "train_ms_iter",
    "inference_ms_batch", "single_sample_ms", "peak_train_mib",
    "peak_inference_mib", "batch_size", "precision", "gpu"
]
with (root / "p2_efficiency.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

tex = [
    r"\begin{tabular}{lrrrrrr}",
    r"\toprule",
    r"Model & Params & Epoch (s) & Train (ms/iter) & Infer (ms/batch) & Train Mem. (MiB) & Infer Mem. (MiB) \\",
    r"\midrule",
]
for row in rows:
    tex.append(
        f'{row["model"]} & {row["parameters"]:,} & '
        f'{row["epoch_seconds"]:.3f} & {row["train_ms_iter"]:.3f} & '
        f'{row["inference_ms_batch"]:.3f} & {row["peak_train_mib"]:.1f} & '
        f'{row["peak_inference_mib"]:.1f} \\\\'
    )
tex.extend([r"\bottomrule", r"\end{tabular}"])
(root / "p2_efficiency_table.tex").write_text(
    "\n".join(tex) + "\n", encoding="utf-8"
)
print(json.dumps({"models_completed": len(rows)}))
