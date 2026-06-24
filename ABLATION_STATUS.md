# SDE_Unified Ablation — Current Status

Date: 2026-06-22

## Repository state

- **Active backup**: `/home/DM24/workspace/Time_Series_Forecasting/SC_FTMixer_SDE_Unified/`
- **Final model**: `SC_FTMixer_SDE_Unified` (models/SC_FTMixer_SDE_Unified.py)
- **Ablation scripts**: `script/ablation/` (4 scripts, 6 variants × 2 horizons = 48 runs total)
- **Ablation version tag**: `AblFixStatic` (embedded in model_id and des)

## Model modifications (vs original SDE_Unified)

Added three switches to `models/SC_FTMixer_SDE_Unified.py`:

| Switch | Default | Purpose |
|---|---|---|
| `use_global_sde` | 1 | Controls `global_spectrum` in static prior |
| `use_calendar_sde` | 1 | Controls `gate * calendar_residual[phase]` in static prior |
| `use_dynamic_filter` | 1 | Controls dynamic residual filtering |
| `fix_calendar_gate` | 0 | When 1, gate = 1.0 strictly (not learned) |

Static prior construction (forecast method):
```python
static = zeros
if use_global_sde:   static += global_spectrum
if use_calendar_sde: static += gate * calendar_residual[phase]
```

## Ablation variants

| # | Variant | use_sde | global | calendar | gate | dyn_filt |
|---|---------|---------|--------|----------|------|----------|
| 1 | Full | 1 | 1 | 1 | learned | 1 |
| 2 | NoStatic | 0 | 1 | 1 | learned | 1 |
| 3 | GlobalOnly | 1 | 1 | 0 | learned | 1 |
| 4 | CalendarOnly | 1 | 0 | 1 | learned | 1 |
| 5 | NoGate | 1 | 1 | 1 | fixed=1 | 1 |
| 6 | NoDynFilt | 1 | 1 | 1 | learned | 0 |

## Dataset configs

| Dataset | enc_in | batch | freq | phase | cycle | slots | hidden | gate_init |
|---|---|---|---|---|---|---|---|---|
| ETTh1 | 7 | 1024 | h | hour | 24 | 1 | 192 | 2.0 |
| ETTm2 | 7 | 1024 | t | day_slot | 96 | 4 | 256 | 2.0 |
| Traffic | 862 | 32 | h | hour_week | 168 | 1 | 256 | 2.0 |
| Weather | 21 | 1024 | t | day_slot | 144 | 6 | 256 | 2.0 |

## Execution order

Each script runs 96-step first (6 variants), then 720-step (commented out by default):
```bash
run_ablation_set 96
# run_ablation_set 720   # uncomment after 96 verified
```

## Known issues fixed

1. **Bug**: Initial backup model was original `SC_FTMixer_SDE_Unified.py` without `use_global_sde`/`use_calendar_sde` switches. Ablation scripts passed these args but model ignored them → Full/GlobalOnly/CalendarOnly produced identical results on ETTh1. **Fixed** 2026-06-22 by adding the two switches.

2. **Invalid ETTh1 results**: The first ETTh1 ablation run (before the fix) is invalid — Full = GlobalOnly = CalendarOnly because the switches weren't connected. Those checkpoints and logs should not be used.

## Next steps

1. Run `ablation_ETTh1.sh` (96 only) → verify GlobalOnly ≠ CalendarOnly ≠ Full
2. If 96 results valid, uncomment 720 and re-run all 4 datasets
3. ETTh1 expected to show SDE overall effective, but may not strongly distinguish global vs calendar
4. ETTm2 / Traffic / Weather are the key datasets for proving global-calendar complementarity

## Related files

- Model: `models/SC_FTMixer_SDE_Unified.py`
- Entry: `main_sde.py`
- Experiment: `experiments/exp_long_term_forecasting_sde.py`
- Registry: `experiments/exp_basic.py`
- Scripts: `script/ablation/ablation_{ETTh1,ETTm2,Traffic,Weather}.sh`

## Dataset scope update - 2026-06-23

PEMS is no longer used in the current SC-FSD paper experiment plan.

Reason: PEMS03/PEMS04 require stronger cross-sensor/spatial traffic propagation than the current SC-FSD temporal/frequency design provides, while PEMS07/PEMS08 gains are not enough to justify keeping a separate PEMS protocol in the main paper. To keep the paper story focused and avoid unnecessary reruns, all follow-up experiment monitoring, ablation, hyperparameter search, and result tables should exclude PEMS unless explicitly re-enabled later.

Current effective dataset scope:

- Main long-term forecasting: ETTh1, ETTh2, ETTm1, ETTm2, Weather, ECL, Traffic
- Ablation focus: ETTm2, Weather, Traffic, and optionally ECL if needed
- Supplementary candidates: ILI or other non-PEMS datasets only after confirming loader/calendar support

Do not spend additional compute on PEMS runs; existing PEMS logs are archival only and should not be used for new paper tables.

