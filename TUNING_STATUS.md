# SDE_Unified Targeted Tuning — Final Status

Date: 2026-06-22

## Scope

Targeted tuning was limited to ETTh1 and ETTh2. The model structure was not
changed. All runs used `sde_hidden=192`, seed 2021, and the `TuneV1`–`TuneV3`
tags.

## Tested factors

- Loss ratios: `0.25/0.75`, `0.50/0.50`, `0.75/0.25`, `1.00/0.00`
- Calendar gate initialization: `1.0`, `2.0`, `3.0`
- Learning rate: `0.003`, `0.005`

## ETTh1

Candidate: `rec/spectral=0.50/0.50`, `lr=0.005`, `gate_init=2.0`.

| Horizon | Original 0.25/0.75 | Candidate 0.50/0.50 | Delta MSE |
|---:|---:|---:|---:|
| 96 | 0.364535 | 0.365627 | +0.001092 |
| 192 | 0.418192 | 0.417710 | -0.000482 |
| 336 | 0.458581 | 0.460174 | +0.001593 |
| 720 | 0.460478 | 0.455883 | -0.004595 |
| Average | 0.425446 | 0.424848 | -0.000598 |

The candidate improves horizon 720 but worsens 96 and 336. Its average
improvement is below the predefined `0.003` acceptance threshold.

## ETTh2

Candidate: `rec/spectral=0.25/0.75`, `lr=0.005`, `gate_init=1.0`.

| Horizon | Original gate=2 | Candidate gate=1 | Delta MSE |
|---:|---:|---:|---:|
| 96 | 0.279427 | 0.279003 | -0.000424 |
| 192 | 0.361048 | 0.360996 | -0.000052 |
| 336 | 0.400094 | 0.401116 | +0.001022 |
| 720 | 0.417867 | 0.415363 | -0.002504 |
| Average | 0.364609 | 0.364119 | -0.000490 |

Gate initialization 1.0 improves three horizons slightly, but worsens 336.
The average gain is too small to justify a dataset-specific configuration.

## Other findings

- `rec/spectral=1.00/0.00` was clearly worse, confirming that the spectral
  objective is necessary.
- `lr=0.003` did not improve either dataset.
- `gate_init=3.0` did not outperform `gate_init=1.0`.
- No hidden-size sweep is warranted based on these results.

## Best configuration by prediction length

The final main-table policy allows each prediction length to use the best
configuration found in the tested `TuneV1`–`TuneV3` search space. Selection is
based on the lowest MSE; MAE is reported as the secondary metric.

### ETTh1

| Pred len | rec | spectral | lr | gate_init | hidden | MSE | MAE | MSE change vs original |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 96 | 0.25 | 0.75 | 0.005 | 2.0 | 192 | 0.364535 | 0.388060 | baseline |
| 192 | 0.50 | 0.50 | 0.005 | 2.0 | 192 | 0.417710 | 0.417323 | -0.000482 |
| 336 | 0.25 | 0.75 | 0.005 | 2.0 | 192 | 0.458581 | 0.438760 | baseline |
| 720 | 0.50 | 0.50 | 0.005 | 1.0 | 192 | 0.455284 | 0.455420 | -0.005194 |

ETTh1 average with per-horizon selection:

```text
MSE = 0.424027
MAE = 0.424891
```

### ETTh2

| Pred len | rec | spectral | lr | gate_init | hidden | MSE | MAE | MSE change vs original |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 96 | 0.25 | 0.75 | 0.005 | 1.0 | 192 | 0.279003 | 0.329730 | -0.000424 |
| 192 | 0.25 | 0.75 | 0.005 | 1.0 | 192 | 0.360996 | 0.381147 | -0.000052 |
| 336 | 0.25 | 0.75 | 0.005 | 2.0 | 192 | 0.400094 | 0.415764 | baseline |
| 720 | 0.25 | 0.75 | 0.005 | 1.0 | 192 | 0.415363 | 0.435264 | -0.002504 |

ETTh2 average with per-horizon selection:

```text
MSE = 0.363864
MAE = 0.390476
```

## Final decision

Use the per-prediction-length configurations above for the main result table.
All selections stay within the same model architecture and the small,
predefined tuning grid. The default/fallback configuration remains:

```text
sde_rec_weight=0.25
sde_spectral_weight=0.75
learning_rate=0.005
sde_calendar_gate_init=2.0
sde_hidden=192
```

For reproducibility, report the horizon-specific loss ratio and gate
initialization in the implementation details or appendix.

## Related scripts

- `script/tuning/tune_loss_ETTh1_ETTh2.sh`
- `script/tuning/tune_v2_ETTh1_ETTh2.sh`
- `script/tuning/tune_v3_confirm.sh`
