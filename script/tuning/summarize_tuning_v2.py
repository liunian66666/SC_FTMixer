#!/usr/bin/env python3
import glob
import os
import re

import numpy as np


ROOT = "/home/DM24/workspace/Time_Series_Forecasting/SC_FTMixer_SDE_Unified"
PATTERN = os.path.join(
    ROOT,
    "results/log_mse_mae/npy_results/"
    "long_term_forecast_ETTh[12]_96_*_TuneV2_*"
    "/metrics.npy",
)
RX = re.compile(
    r"forecast_(ETTh[12])_96_(96|720)_TuneV2_"
    r"(BaselineRepeat|Gate1|Gate3|LR003)_"
    r"(R25S75|R50S50)_(LR003|LR005)_(G1|G2|G3)_H192"
)

rows = []
for path in glob.glob(PATTERN):
    match = RX.search(path)
    if not match:
        continue
    dataset, horizon, experiment, ratio, lr, gate = match.groups()
    mae, mse, *_ = np.load(path).tolist()
    rows.append(
        (dataset, int(horizon), experiment, ratio, lr, gate, float(mse), float(mae))
    )

print(f"TuneV2 completed: {len(rows)}/10")
print("dataset horizon experiment      ratio   lr     gate      MSE       MAE")
for row in sorted(rows):
    dataset, horizon, experiment, ratio, lr, gate, mse, mae = row
    print(
        f"{dataset:6s} {horizon:7d} {experiment:15s} {ratio:7s} "
        f"{lr:6s} {gate:4s}  {mse:.6f}  {mae:.6f}"
    )
