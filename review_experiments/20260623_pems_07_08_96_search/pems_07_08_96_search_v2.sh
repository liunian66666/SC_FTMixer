#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/DM24/workspace/Time_Series_Forecasting/SC_FTMixer_SDE_Unified}"
OUT="${OUT:-${ROOT}/review_experiments/20260623_pems_07_08_96_search}"
DATASET="${1:?PEMS07 or PEMS08 required}"
cd "${ROOT}"
mkdir -p "${OUT}/logs" "${OUT}/checkpoints" "${OUT}/commands"

case "${DATASET}" in
  PEMS07) ENC_IN=883; BASE_BATCH=16 ;;
  PEMS08) ENC_IN=170; BASE_BATCH=64 ;;
  *) echo "Unsupported dataset: ${DATASET}" >&2; exit 2 ;;
esac

run_one() {
  local tag="$1"
  local rec="$2"
  local spectral="$3"
  local gate="$4"
  local hidden="$5"
  local lr="$6"
  local batch="$7"

  local model_id="${DATASET}_96_96_PemsSearchV2_${tag}"
  local log="${OUT}/logs/${model_id}.log"
  local cmd_file="${OUT}/commands/${model_id}.sh"

  local cmd=(
    python3 -u main_sde.py
    --task_name long_term_forecast
    --is_training 1
    --root_path ./dataset/PEMS/
    --data_path "${DATASET}.npz"
    --model_id "${model_id}"
    --model SC_FTMixer_SDE_Unified
    --data PEMS
    --features M
    --seq_len 96
    --label_len 48
    --pred_len 96
    --freq t
    --enc_in "${ENC_IN}"
    --d_model 32
    --d_ff 64
    --e_layers 1
    --n_heads 1
    --dropout 0
    --batch_size "${batch}"
    --learning_rate "${lr}"
    --train_epochs 100
    --patience 10
    --lradj cosine_warmup
    --des "PemsSearchV2_${tag}"
    --itr 1
    --num_workers 0
    --checkpoints "${OUT}/checkpoints/"
    --use_sde 1
    --sde_phase_mode position
    --sde_cycle_len 288
    --sde_slots_per_hour 12
    --sde_calendar_gate_init "${gate}"
    --sde_hidden "${hidden}"
    --sde_rec_weight "${rec}"
    --sde_spectral_weight "${spectral}"
    --use_global_sde 1
    --use_calendar_sde 1
    --use_dynamic_filter 1
    --fix_calendar_gate 0
    --visualize 0
  )

  printf '%q ' "${cmd[@]}" > "${cmd_file}"
  printf '\n' >> "${cmd_file}"
  echo "START ${model_id} $(date --iso-8601=seconds)" | tee -a "${OUT}/${DATASET}_runner.log"
  "${cmd[@]}" > "${log}" 2>&1
  grep -E 'mse:|mae:|REVIEW_PEAK_MEMORY' "${log}" | tail -4 \
    | tee -a "${OUT}/${DATASET}_runner.log" || true
  echo "DONE ${model_id} $(date --iso-8601=seconds)" | tee -a "${OUT}/${DATASET}_runner.log"
}

# B0: exact existing default, included as a reproducibility control.
run_one B0_Default 0.25 0.75 2.0 256 0.005 "${BASE_BATCH}"

# Reduce spectral regularization or calendar strength.
run_one B1_R50S50_G2 0.50 0.50 2.0 256 0.005 "${BASE_BATCH}"
run_one B2_R25S75_G1 0.25 0.75 1.0 256 0.005 "${BASE_BATCH}"
run_one B3_R50S50_G1 0.50 0.50 1.0 256 0.005 "${BASE_BATCH}"

# Stabilize long-horizon training or increase temporal decoder capacity.
run_one B4_LR3 0.25 0.75 2.0 256 0.003 "${BASE_BATCH}"
run_one B5_H320 0.25 0.75 2.0 320 0.005 "${BASE_BATCH}"

# Dataset-specific batch probe; still the same architecture.
if [[ "${DATASET}" == "PEMS07" ]]; then
  run_one B6_B32 0.25 0.75 2.0 256 0.005 32
else
  run_one B6_B32 0.25 0.75 2.0 256 0.005 32
fi
