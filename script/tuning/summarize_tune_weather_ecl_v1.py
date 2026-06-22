#!/usr/bin/env python3
import glob
import os
import re

import numpy as np

root = "/home/DM24/workspace/Time_Series_Forecasting/SC_FTMixer_SDE_Unified"
pattern = os.path.join(
    root,
    "results/log_mse_mae/npy_results/"
    "long_term_forecast_*_96_*_TuneWE1_*/metrics.npy",
)
rx = re.compile(
    r"forecast_(Weather|ECL)_96_(96|336|720)_TuneWE1_"
    r"(W[1-4]|E[1-5])_(R25S75|R50S50)_(LR003|LR005)_"
    r"(G1|G2|G3)_H(192|256)"
)

rows = []
for path in glob.glob(pattern):
    match = rx.search(path)
    if not match:
        continue
    dataset, horizon, tag, ratio, lr, gate, hidden = match.groups()
    mae, mse, *_ = np.load(path).tolist()
    rows.append(
        (
            dataset,
            int(horizon),
            tag,
            ratio,
            lr,
            gate,
            int(hidden),
            float(mse),
            float(mae),
        )
    )

print(f"TuneWE1 completed: {len(rows)}/18")
print("dataset horizon tag ratio   lr     gate hidden      MSE       MAE")
for row in sorted(rows):
    dataset, horizon, tag, ratio, lr, gate, hidden, mse, mae = row
    print(
        f"{dataset:7s} {horizon:7d} {tag:3s} {ratio:7s} {lr:6s} "
        f"{gate:4s} {hidden:6d}  {mse:.6f}  {mae:.6f}"
    )
