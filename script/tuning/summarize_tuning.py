#!/usr/bin/env python3
import glob
import os
import re

import numpy as np


ROOT = "/home/DM24/workspace/Time_Series_Forecasting/SC_FTMixer_SDE_Unified"
PATTERN = os.path.join(
    ROOT,
    "results/log_mse_mae/npy_results/"
    "long_term_forecast_ETTh[12]_96_*_TuneV1_*"
    "/metrics.npy",
)
RX = re.compile(
    r"forecast_(ETTh[12])_96_(96|720)_TuneV1_"
    r"(R50S50|R75S25|R100S0)_LR005_G2_H192"
)

rows = []
for path in glob.glob(PATTERN):
    match = RX.search(path)
    if not match:
        continue
    dataset, horizon, ratio = match.groups()
    mae, mse, *_ = np.load(path).tolist()
    rows.append((dataset, int(horizon), ratio, float(mse), float(mae)))

print(f"TuneV1 completed: {len(rows)}/12")
print("dataset horizon ratio      MSE       MAE")
for dataset, horizon, ratio, mse, mae in sorted(rows):
    print(f"{dataset:6s} {horizon:7d} {ratio:7s} {mse:.6f}  {mae:.6f}")
