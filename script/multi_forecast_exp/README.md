# Per-horizon experiment scripts

These scripts are generated from the best currently verified result for each
dataset and prediction length. Every prediction length is represented by a
separate `python3 -u main_sde.py` invocation.

Existing files under `script/multi_forecast/` are not modified.

Experiment results are written by `main_sde.py` to:

```text
results/log_mse_mae/All/
```

The MSE and MAE comments above each command describe the result used to choose
that command's parameters; they are not a guarantee that a future rerun on a
different environment will be bitwise identical.

Solar and Exchange are intentionally excluded from this experiment set.
