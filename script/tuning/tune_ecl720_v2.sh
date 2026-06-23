#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/DM24/workspace/Time_Series_Forecasting/SC_FTMixer_SDE_Unified"
cd "${ROOT}"

WORKER_ID="${1:?usage: $0 WORKER_ID [NUM_TUNE_WORKERS]}"
NUM_TUNE_WORKERS="${2:-2}"
RUN_TAG="TuneECL2"
LOG_DIR="results/tuning/${RUN_TAG}/logs"
STATUS_DIR="results/tuning/${RUN_TAG}/status"
mkdir -p "${LOG_DIR}" "${STATUS_DIR}"

# tag rec spectral gate lr hidden batch
jobs=(
  "A 0.50 0.50 3.0 0.005 256 64"
  "B 0.35 0.65 3.0 0.005 256 64"
  "C 0.25 0.75 3.0 0.005 320 64"
  "D 0.35 0.65 3.0 0.005 320 64"
  "E 0.25 0.75 3.0 0.004 256 64"
  "F 0.25 0.75 3.0 0.005 256 32"
)

ratio_tag() {
  case "$1/$2" in
    0.25/0.75) echo "R25S75" ;;
    0.35/0.65) echo "R35S65" ;;
    0.50/0.50) echo "R50S50" ;;
  esac
}

lr_tag() {
  case "$1" in
    0.004) echo "LR004" ;;
    0.005) echo "LR005" ;;
  esac
}

run_job() {
  local tag="$1" rec="$2" spectral="$3" gate="$4"
  local lr="$5" hidden="$6" batch="$7"
  local ratio lr_name model_id des setting metrics log_file
  ratio="$(ratio_tag "${rec}" "${spectral}")"
  lr_name="$(lr_tag "${lr}")"
  model_id="ECL_96_720_${RUN_TAG}_${tag}_${ratio}_${lr_name}_G3_H${hidden}_B${batch}"
  des="${RUN_TAG}_${tag}_${ratio}_${lr_name}_G3_H${hidden}_B${batch}"
  setting="long_term_forecast_${model_id}_SC_FTMixer_SDE_Unified_sl96_ll48_pl720_dm32_nh1_el1_${des}_0_dp0.0"
  metrics="results/log_mse_mae/npy_results/${setting}/metrics.npy"
  log_file="${LOG_DIR}/${model_id}.log"

  if [[ -f "${metrics}" ]]; then
    echo "[$(date '+%F %T')] SKIP completed ${model_id}" | tee -a "${log_file}"
    return
  fi

  echo "[$(date '+%F %T')] START worker=${WORKER_ID} ${model_id}" | tee "${log_file}"
  python3 -u main_sde.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/electricity/ \
    --data_path electricity.csv \
    --model_id "${model_id}" \
    --model SC_FTMixer_SDE_Unified \
    --data custom \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 720 \
    --freq h \
    --enc_in 321 \
    --d_model 32 \
    --d_ff 64 \
    --e_layers 1 \
    --n_heads 1 \
    --dropout 0 \
    --batch_size "${batch}" \
    --learning_rate "${lr}" \
    --train_epochs 100 \
    --patience 10 \
    --lradj cosine_warmup \
    --des "${des}" \
    --itr 1 \
    --num_workers 2 \
    --use_sde 1 \
    --sde_phase_mode hour_week \
    --sde_cycle_len 168 \
    --sde_slots_per_hour 1 \
    --sde_calendar_gate_init "${gate}" \
    --sde_hidden "${hidden}" \
    --sde_rec_weight "${rec}" \
    --sde_spectral_weight "${spectral}" \
    --use_global_sde 1 \
    --use_calendar_sde 1 \
    --use_dynamic_filter 1 \
    2>&1 | tee -a "${log_file}"
  echo "[$(date '+%F %T')] DONE worker=${WORKER_ID} ${model_id}" | tee -a "${log_file}"
}

for i in "${!jobs[@]}"; do
  (( i % NUM_TUNE_WORKERS == WORKER_ID )) || continue
  read -r tag rec spectral gate lr hidden batch <<< "${jobs[$i]}"
  echo "ECL 720 ${tag}" > "${STATUS_DIR}/worker${WORKER_ID}.current"
  run_job "${tag}" "${rec}" "${spectral}" "${gate}" "${lr}" "${hidden}" "${batch}"
done

echo "complete" > "${STATUS_DIR}/worker${WORKER_ID}.current"
echo "[$(date '+%F %T')] WORKER ${WORKER_ID} COMPLETE"
