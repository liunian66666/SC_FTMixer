#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/DM24/workspace/Time_Series_Forecasting/SC_FTMixer_SDE_Unified"
REVIEW_ROOT="${REVIEW_ROOT:?REVIEW_ROOT is required}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
JOB_ID="${1:?job id required}"
DATASET="${2:?dataset required}"
HORIZON="${3:?horizon required}"
VARIANT="${4:?variant required}"
SEED="${5:?seed required}"

JOB_DIR="${REVIEW_ROOT}/jobs/${JOB_ID}"
mkdir -p "${JOB_DIR}" "${REVIEW_ROOT}/checkpoints"

case "${DATASET}" in
  ETTm2)
    DATA="ETTm2"; ROOT_PATH="./dataset/ETT-small/"; DATA_PATH="ETTm2.csv"
    ENC_IN=7; BATCH=1024; FREQ="t"; PHASE="day_slot"; CYCLE=96; SLOTS=4
    HIDDEN=256; GATE=2.0; NUM_WORKERS=2
    ;;
  ECL)
    DATA="custom"; ROOT_PATH="./dataset/electricity/"; DATA_PATH="electricity.csv"
    ENC_IN=321; FREQ="h"; PHASE="hour_week"; CYCLE=168; SLOTS=1
    NUM_WORKERS=2
    case "${HORIZON}" in
      96) HIDDEN=192; GATE=2.0; BATCH=128 ;;
      336) HIDDEN=256; GATE=3.0; BATCH=128 ;;
      720) HIDDEN=320; GATE=3.0; BATCH=64 ;;
    esac
    ;;
  Weather)
    DATA="custom"; ROOT_PATH="./dataset/weather/"; DATA_PATH="weather.csv"
    ENC_IN=21; BATCH=1024; FREQ="t"; PHASE="day_slot"; CYCLE=144; SLOTS=6
    HIDDEN=256; GATE=2.0; NUM_WORKERS=2
    ;;
  Traffic)
    DATA="custom"; ROOT_PATH="./dataset/traffic/"; DATA_PATH="traffic.csv"
    ENC_IN=862; BATCH=32; FREQ="h"; PHASE="hour_week"; CYCLE=168; SLOTS=1
    HIDDEN=256; GATE=2.0; NUM_WORKERS=0
    ;;
  *)
    echo "Unsupported dataset: ${DATASET}" >&2
    exit 2
    ;;
esac

REC_WEIGHT=0.25
SPECTRAL_WEIGHT=0.75
if [[ "${DATASET}" == "Weather" && "${HORIZON}" == "96" ]]; then
  REC_WEIGHT=0.50
  SPECTRAL_WEIGHT=0.50
fi

USE_SDE=1
USE_GLOBAL=1
USE_CALENDAR=1
USE_DYNAMIC=1
FIX_GATE=0
RUN_CYCLE="${CYCLE}"

case "${VARIANT}" in
  Full) ;;
  NoSDE)
    USE_SDE=0; USE_DYNAMIC=0
    ;;
  GlobalOnly)
    USE_CALENDAR=0
    ;;
  FixedGate)
    FIX_GATE=1
    ;;
  NoDynFilt)
    USE_DYNAMIC=0
    ;;
  WrongPhase)
    RUN_CYCLE=24
    ;;
  DynamicOnly)
    USE_SDE=0; USE_DYNAMIC=1
    ;;
  *)
    echo "Unsupported variant: ${VARIANT}" >&2
    exit 2
    ;;
esac

MODEL_ID="ReviewP0_${DATASET}_96_${HORIZON}_${VARIANT}_seed${SEED}_${JOB_ID}"
DES="ReviewP0_${VARIANT}_seed${SEED}_${JOB_ID}"
SETTING="long_term_forecast_${MODEL_ID}_SC_FTMixer_SDE_Unified_sl96_ll48_pl${HORIZON}_dm32_nh1_el1_${DES}_0_dp0.0"

CMD=(
  python3 -u "${REVIEW_ROOT}/main_sde_review.py"
  --task_name long_term_forecast
  --is_training 1
  --seed "${SEED}"
  --root_path "${ROOT_PATH}"
  --data_path "${DATA_PATH}"
  --model_id "${MODEL_ID}"
  --model SC_FTMixer_SDE_Unified
  --data "${DATA}"
  --features M
  --seq_len 96
  --label_len 48
  --pred_len "${HORIZON}"
  --freq "${FREQ}"
  --enc_in "${ENC_IN}"
  --d_model 32
  --d_ff 64
  --e_layers 1
  --n_heads 1
  --dropout 0
  --batch_size "${BATCH}"
  --learning_rate 0.005
  --train_epochs 100
  --patience 10
  --lradj cosine_warmup
  --des "${DES}"
  --itr 1
  --num_workers "${NUM_WORKERS}"
  --checkpoints "${REVIEW_ROOT}/checkpoints/"
  --use_sde "${USE_SDE}"
  --sde_phase_mode "${PHASE}"
  --sde_cycle_len "${RUN_CYCLE}"
  --sde_slots_per_hour "${SLOTS}"
  --sde_calendar_gate_init "${GATE}"
  --sde_hidden "${HIDDEN}"
  --sde_rec_weight "${REC_WEIGHT}"
  --sde_spectral_weight "${SPECTRAL_WEIGHT}"
  --use_global_sde "${USE_GLOBAL}"
  --use_calendar_sde "${USE_CALENDAR}"
  --use_dynamic_filter "${USE_DYNAMIC}"
  --fix_calendar_gate "${FIX_GATE}"
  --visualize 0
)

printf '%q ' "${CMD[@]}" > "${JOB_DIR}/command.sh"
printf '\n' >> "${JOB_DIR}/command.sh"
chmod +x "${JOB_DIR}/command.sh"

python3 - "${JOB_DIR}/model_meta.json" "${ENC_IN}" "${HORIZON}" "${HIDDEN}" "${RUN_CYCLE}" \
  "${USE_SDE}" "${USE_GLOBAL}" "${USE_CALENDAR}" "${USE_DYNAMIC}" "${FIX_GATE}" <<'PY'
import json, sys
from types import SimpleNamespace
from models.SC_FTMixer_SDE_Unified import Model

out, enc_in, pred_len, hidden, cycle, use_sde, use_global, use_calendar, use_dynamic, fix_gate = sys.argv[1:]
args = SimpleNamespace(
    seq_len=96, pred_len=int(pred_len), enc_in=int(enc_in),
    sde_hidden=int(hidden), sde_cycle_len=int(cycle), freq="h",
    sde_phase_mode="auto", sde_slots_per_hour=1,
    use_sde=int(use_sde), use_global_sde=int(use_global),
    use_calendar_sde=int(use_calendar),
    use_dynamic_filter=int(use_dynamic), fix_calendar_gate=int(fix_gate),
    output_attention=False, sde_calendar_gate_init=2.0,
)
model = Model(args)
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
with open(out, "w", encoding="utf-8") as f:
    json.dump({"parameters_total": total, "parameters_trainable": trainable}, f, indent=2)
PY

START_EPOCH="$(date +%s)"
START_ISO="$(date --iso-8601=seconds)"
GPU_LOG="${JOB_DIR}/gpu_memory.csv"
echo "timestamp,pid,used_memory_mib,gpu_util_percent,total_memory_mib" > "${GPU_LOG}"

cd "${ROOT}"
"${CMD[@]}" > "${JOB_DIR}/stdout.log" 2>&1 &
TRAIN_PID=$!

cleanup_training_process() {
  if kill -0 "${TRAIN_PID}" 2>/dev/null; then
    kill "${TRAIN_PID}" 2>/dev/null || true
    wait "${TRAIN_PID}" 2>/dev/null || true
  fi
}
trap cleanup_training_process EXIT INT TERM

while kill -0 "${TRAIN_PID}" 2>/dev/null; do
  TS="$(date --iso-8601=seconds)"
  UTIL="$(nvidia-smi --query-gpu=utilization.gpu,memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')"
  GPU_UTIL="${UTIL%%,*}"
  GPU_TOTAL="${UTIL##*,}"
  USED="$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null \
    | awk -F',' -v pid="${TRAIN_PID}" '$1+0==pid {gsub(/ /,"",$2); print $2; found=1} END{if(!found) print 0}')"
  echo "${TS},${TRAIN_PID},${USED},${GPU_UTIL},${GPU_TOTAL}" >> "${GPU_LOG}"
  sleep 1
done

set +e
wait "${TRAIN_PID}"
EXIT_CODE=$?
set -e
trap - EXIT INT TERM
END_EPOCH="$(date +%s)"
END_ISO="$(date --iso-8601=seconds)"
DURATION=$((END_EPOCH - START_EPOCH))
PEAK="$(awk -F',' 'NR>1 && $3>m {m=$3} END{print m+0}' "${GPU_LOG}")"
TORCH_PEAK="$(grep -o 'REVIEW_PEAK_MEMORY_MIB=[0-9.]*' "${JOB_DIR}/stdout.log" \
  | tail -1 | cut -d= -f2 || true)"
if [[ -n "${TORCH_PEAK}" ]]; then
  PEAK="${TORCH_PEAK}"
fi

METRICS_PATH="${ROOT}/results/log_mse_mae/npy_results/${SETTING}/metrics.npy"
CHECKPOINT_PATH="${REVIEW_ROOT}/checkpoints/${SETTING}/checkpoint.pth"

python3 - "${JOB_DIR}/meta.json" <<PY
import json
meta = {
  "job_id": "${JOB_ID}",
  "dataset": "${DATASET}",
  "input_len": 96,
  "pred_len": ${HORIZON},
  "variant": "${VARIANT}",
  "seed": ${SEED},
  "gpu": "Tesla T4",
  "gpu_uuid": "GPU-83450b41-6731-8b22-f0ed-c75330983fc6",
  "start_time": "${START_ISO}",
  "end_time": "${END_ISO}",
  "duration_seconds": ${DURATION},
  "peak_gpu_memory_mib": ${PEAK},
  "exit_code": ${EXIT_CODE},
  "setting": "${SETTING}",
  "stdout_log": "${JOB_DIR}/stdout.log",
  "gpu_log": "${GPU_LOG}",
  "checkpoint_path": "${CHECKPOINT_PATH}",
  "metrics_path": "${METRICS_PATH}",
  "config": {
    "batch_size": ${BATCH}, "hidden": ${HIDDEN}, "cycle_len": ${RUN_CYCLE},
    "phase_mode": "${PHASE}", "gate_init": ${GATE},
    "use_sde": ${USE_SDE}, "use_global_sde": ${USE_GLOBAL},
    "use_calendar_sde": ${USE_CALENDAR},
    "use_dynamic_filter": ${USE_DYNAMIC}, "fix_calendar_gate": ${FIX_GATE},
    "rec_weight": ${REC_WEIGHT}, "spectral_weight": ${SPECTRAL_WEIGHT},
    "learning_rate": 0.005
  }
}
with open("${JOB_DIR}/model_meta.json", encoding="utf-8") as f:
  meta.update(json.load(f))
with open("${JOB_DIR}/meta.json", "w", encoding="utf-8") as f:
  json.dump(meta, f, indent=2)
PY

if [[ -f "${METRICS_PATH}" ]]; then
  cp "${METRICS_PATH}" "${JOB_DIR}/metrics.npy"
  python3 - "${JOB_DIR}/metrics.npy" "${JOB_DIR}/metrics.json" <<'PY'
import json, numpy as np, sys, math
a = np.load(sys.argv[1])
mae, mse, rmse, mape, mspe = [float(x) for x in a]
def trunc3(x):
    return math.trunc(x * 1000) / 1000
with open(sys.argv[2], "w", encoding="utf-8") as f:
    json.dump({
        "mse": mse, "mae": mae, "rmse": rmse, "mape": mape, "mspe": mspe,
        "mse_trunc3": trunc3(mse), "mae_trunc3": trunc3(mae)
    }, f, indent=2)
PY
fi

exit "${EXIT_CODE}"
