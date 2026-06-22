#!/usr/bin/env python3
import glob
import os
import re

import numpy as np

root = "/home/DM24/workspace/Time_Series_Forecasting/SC_FTMixer_SDE_Unified"
pattern = os.path.join(
    root,
    "results/log_mse_mae/npy_results/"
    "long_term_forecast_ETTh[12]_96_*_TuneV3_*/metrics.npy",
)
rx = re.compile(
    r"forecast_(ETTh[12])_96_(96|192|336)_TuneV3_"
    r"(MidConfirm|Gate1Confirm)_(R50S50|R25S75)_LR005_(G1|G2)_H192"
)

rows = []
for path in glob.glob(pattern):
    match = rx.search(path)
    if not match:
        continue
    dataset, horizon, tag, ratio, gate = match.groups()
    mae, mse, *_ = np.load(path).tolist()
    rows.append((dataset, int(horizon), tag, ratio, gate, float(mse), float(mae)))

print(f"TuneV3 completed: {len(rows)}/5")
print("dataset horizon tag           ratio   gate      MSE       MAE")
for dataset, horizon, tag, ratio, gate, mse, mae in sorted(rows):
    print(
        f"{dataset:6s} {horizon:7d} {tag:13s} {ratio:7s} "
        f"{gate:4s}  {mse:.6f}  {mae:.6f}"
    )
