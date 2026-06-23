#!/usr/bin/env python3
import glob
import os
import re

import numpy as np


ROOT = "/home/DM24/workspace/Time_Series_Forecasting/SC_FTMixer_SDE_Unified"
PATTERN = os.path.join(
    ROOT,
    "results/log_mse_mae/npy_results/"
    "long_term_forecast_PEMS0[34]_96_*_PemsTuneV1_*/metrics.npy",
)
RX = re.compile(
    r"forecast_(PEMS0[34])_96_(12|96)_PemsTuneV1_(P[1-4])_"
)

rows = []
for path in glob.glob(PATTERN):
    match = RX.search(path)
    if not match:
        continue
    dataset, horizon, tag = match.groups()
    mae, mse, *_ = np.load(path).tolist()
    rows.append((dataset, int(horizon), tag, float(mse), float(mae)))

print(f"PemsTuneV1 completed: {len(rows)}/16")
print("dataset horizon tag      MSE       MAE")
for dataset, horizon, tag, mse, mae in sorted(rows):
    print(f"{dataset:6s} {horizon:7d} {tag:3s}  {mse:.6f}  {mae:.6f}")
