#!/usr/bin/env python3
import csv
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
pattern = re.compile(
    r"mse:([0-9.eE+-]+),\s*mae:([0-9.eE+-]+)"
)
for log in sorted((root / "logs").glob("PEMS*_96_96_PemsSearchV2_*.log")):
    text = log.read_text(encoding="utf-8", errors="replace")
    matches = pattern.findall(text)
    if not matches:
        continue
    mse, mae = map(float, matches[-1])
    name = log.stem
    match = re.match(r"(PEMS\d+)_96_96_PemsSearchV2_(.+)", name)
    rows.append({
        "dataset": match.group(1),
        "pred_len": 96,
        "tag": match.group(2),
        "mse": mse,
        "mae": mae,
        "log": str(log),
        "command": str(root / "commands" / f"{name}.sh"),
    })

rows.sort(key=lambda x: (x["dataset"], x["mse"], x["mae"]))
with (root / "summary.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys() if rows else [
        "dataset", "pred_len", "tag", "mse", "mae", "log", "command"
    ])
    writer.writeheader()
    writer.writerows(rows)
(root / "summary.json").write_text(
    json.dumps(rows, indent=2), encoding="utf-8"
)
for dataset in ("PEMS07", "PEMS08"):
    group = [row for row in rows if row["dataset"] == dataset]
    if group:
        best = min(group, key=lambda x: (x["mse"], x["mae"]))
        print(dataset, json.dumps(best))
