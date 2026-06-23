#!/usr/bin/env python3
import glob
import os
import re

import numpy as np

root = "/home/DM24/workspace/Time_Series_Forecasting/SC_FTMixer_SDE_Unified"
pattern = os.path.join(
    root,
    "results/log_mse_mae/npy_results/"
    "long_term_forecast_ECL_96_720_TuneECL2_*/metrics.npy",
)
rx = re.compile(
    r"forecast_ECL_96_720_TuneECL2_([A-F])_"
    r"(R25S75|R35S65|R50S50)_(LR004|LR005)_G3_"
    r"H(256|320)_B(32|64)"
)

rows = []
for path in glob.glob(pattern):
    match = rx.search(path)
    if not match:
        continue
    tag, ratio, lr, hidden, batch = match.groups()
    mae, mse, *_ = np.load(path).tolist()
    rows.append(
        (tag, ratio, lr, int(hidden), int(batch), float(mse), float(mae))
    )

print(f"TuneECL2 completed: {len(rows)}/6")
print("tag ratio   lr     hidden batch      MSE       MAE")
for tag, ratio, lr, hidden, batch, mse, mae in sorted(rows):
    print(
        f"{tag:3s} {ratio:7s} {lr:6s} {hidden:6d} {batch:5d}  "
        f"{mse:.6f}  {mae:.6f}"
    )
