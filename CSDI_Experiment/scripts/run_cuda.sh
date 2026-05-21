#!/usr/bin/env bash
set -euo pipefail

# Runs CSDI walk-forward folds across CUDA GPUs, one worker process per GPU.
#
# Common overrides:
#   GPUS="0 1" N_FOLDS=12 EPOCHS=50 ITR_PER_EPOCH=100 NSAMPLE=50 bash CSDI_Experiment/scripts/run_cuda.sh
#   TARGET_COLUMNS="Agric Food Oil" bash CSDI_Experiment/scripts/run_cuda.sh
#   RUN_ANALYSIS=0 bash CSDI_Experiment/scripts/run_cuda.sh
#
# Slurm example:
#   ml course/cme213/nvhpc/24.1
#   srun --partition=gpu-turing --gres=gpu:4 bash CSDI_Experiment/scripts/run_cuda.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

PYTHON="${PYTHON:-$PROJECT_ROOT/../venv/bin/python}"
PIP="${PIP:-$PROJECT_ROOT/../venv/bin/pip3}"

DATA_CSV="${DATA_CSV:-$PROJECT_ROOT/data/processed/french49_daily_returns.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/CSDI_Experiment/outputs/cuda_walk_forward_$(date +%Y%m%d_%H%M%S)}"

EPOCHS="${EPOCHS:-50}"
ITR_PER_EPOCH="${ITR_PER_EPOCH:-100}"
NSAMPLE="${NSAMPLE:-50}"
N_FOLDS="${N_FOLDS:-0}"
VALID_WINDOWS="${VALID_WINDOWS:-8}"
TRAIN_STRIDE="${TRAIN_STRIDE:-1}"
HISTORY_LENGTH="${HISTORY_LENGTH:-231}"
PRED_LENGTH="${PRED_LENGTH:-21}"
RUN_ANALYSIS="${RUN_ANALYSIS:-1}"
COPY_FOLDS="${COPY_FOLDS:-1}"
STREAM_LOGS="${STREAM_LOGS:-1}"
CSDI_PROGRESS_INTERVAL="${CSDI_PROGRESS_INTERVAL:-10}"

if [[ ! -x "$PYTHON" ]]; then
  echo "Python executable not found: $PYTHON" >&2
  echo "Set PYTHON=/path/to/python or create the project venv first." >&2
  exit 1
fi

if ! "$PYTHON" -c "import torch, yaml, pandas, numpy, tqdm, linear_attention_transformer" >/dev/null 2>&1; then
  if [[ ! -x "$PIP" ]]; then
    echo "Missing CSDI dependencies and pip executable was not found: $PIP" >&2
    exit 1
  fi
  echo "Installing CSDI Python requirements into: $PYTHON"
  "$PIP" install -r "$PROJECT_ROOT/CSDI/requirements.txt"
fi

if [[ ! -f "$DATA_CSV" ]]; then
  echo "Input data not found at $DATA_CSV"
  echo "Building French daily returns dataset..."
  "$PYTHON" "$PROJECT_ROOT/scripts/build_french_returns.py"
fi

if [[ -z "${GPUS:-}" ]]; then
  GPU_COUNT="$("$PYTHON" -c "import torch; print(torch.cuda.device_count())")"
  if [[ "$GPU_COUNT" == "0" ]]; then
    echo "No CUDA GPUs found by PyTorch. Set GPUS=\"0\" only if CUDA visibility is configured externally." >&2
    exit 2
  fi
  GPUS=""
  for ((GPU_ID = 0; GPU_ID < GPU_COUNT; GPU_ID++)); do
    GPUS+="$GPU_ID "
  done
fi

read -r -a GPU_LIST <<< "$GPUS"
WORKER_COUNT="${#GPU_LIST[@]}"
if [[ "$WORKER_COUNT" -eq 0 ]]; then
  echo "No GPUs selected. Set GPUS=\"0 1\" or similar." >&2
  exit 2
fi

PARENT_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
PARENT_DEVICE_LIST=()
if [[ -n "$PARENT_CUDA_VISIBLE_DEVICES" ]]; then
  IFS=',' read -r -a PARENT_DEVICE_LIST <<< "$PARENT_CUDA_VISIBLE_DEVICES"
fi

TOTAL_FOLDS="$("$PYTHON" - <<PY
from pathlib import Path
import sys
sys.path.insert(0, str(Path("$PROJECT_ROOT") / "CSDI_Experiment" / "src"))
from walk_forward_csdi import (
    default_initial_train_size,
    load_timeseries,
    make_fold_specs,
)
dates, features, values, mask = load_timeseries(Path("$DATA_CSV"), "date", None)
initial_train_size = ${INITIAL_TRAIN_SIZE:-None}
step_size = ${STEP_SIZE:-None}
specs = make_fold_specs(
    total_rows=len(values),
    history_length=int("$HISTORY_LENGTH"),
    pred_length=int("$PRED_LENGTH"),
    initial_train_size=initial_train_size,
    step_size=step_size,
    n_folds=int("$N_FOLDS"),
    train_stride=int("$TRAIN_STRIDE"),
    valid_windows=int("$VALID_WINDOWS"),
)
print(len(specs))
PY
)"

if [[ "$TOTAL_FOLDS" -le 0 ]]; then
  echo "No folds to run." >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR/workers"

echo "Output directory: $OUTPUT_DIR"
echo "CUDA GPUs: ${GPU_LIST[*]}"
if [[ -n "$PARENT_CUDA_VISIBLE_DEVICES" ]]; then
  echo "Parent CUDA_VISIBLE_DEVICES: $PARENT_CUDA_VISIBLE_DEVICES"
fi
echo "Total selected folds: $TOTAL_FOLDS"

COMMON_ARGS=(
  "$PROJECT_ROOT/CSDI_Experiment/src/walk_forward_csdi.py"
  --data "$DATA_CSV"
  --epochs "$EPOCHS"
  --itr-per-epoch "$ITR_PER_EPOCH"
  --nsample "$NSAMPLE"
  --n-folds "$N_FOLDS"
  --valid-windows "$VALID_WINDOWS"
  --train-stride "$TRAIN_STRIDE"
  --history-length "$HISTORY_LENGTH"
  --pred-length "$PRED_LENGTH"
)

if [[ -n "${BATCH_SIZE:-}" ]]; then
  COMMON_ARGS+=(--batch-size "$BATCH_SIZE")
fi

if [[ -n "${INITIAL_TRAIN_SIZE:-}" ]]; then
  COMMON_ARGS+=(--initial-train-size "$INITIAL_TRAIN_SIZE")
fi

if [[ -n "${STEP_SIZE:-}" ]]; then
  COMMON_ARGS+=(--step-size "$STEP_SIZE")
fi

if [[ -n "${LR:-}" ]]; then
  COMMON_ARGS+=(--lr "$LR")
fi

if [[ -n "${TARGET_COLUMNS:-}" ]]; then
  read -r -a COLUMNS <<< "$TARGET_COLUMNS"
  COMMON_ARGS+=(--target-columns "${COLUMNS[@]}")
fi

PIDS=()
TAIL_PIDS=()
WORKER_DIRS=()
BASE=$((TOTAL_FOLDS / WORKER_COUNT))
REM=$((TOTAL_FOLDS % WORKER_COUNT))
START=0

for IDX in "${!GPU_LIST[@]}"; do
  COUNT="$BASE"
  if [[ "$IDX" -lt "$REM" ]]; then
    COUNT=$((COUNT + 1))
  fi
  if [[ "$COUNT" -eq 0 ]]; then
    continue
  fi

  GPU="${GPU_LIST[$IDX]}"
  WORKER_DIR="$OUTPUT_DIR/workers/gpu_${GPU}"
  WORKER_DIRS+=("$WORKER_DIR")
  LOG="$OUTPUT_DIR/worker_gpu_${GPU}.log"
  CHILD_VISIBLE_DEVICES="$GPU"
  if [[ -n "$PARENT_CUDA_VISIBLE_DEVICES" && "$GPU" =~ ^[0-9]+$ && "$GPU" -lt "${#PARENT_DEVICE_LIST[@]}" ]]; then
    CHILD_VISIBLE_DEVICES="${PARENT_DEVICE_LIST[$GPU]}"
  fi

  echo "Starting GPU $GPU worker: visible-device=$CHILD_VISIBLE_DEVICES fold-start=$START fold-count=$COUNT log=$LOG"
  CUDA_VISIBLE_DEVICES="$CHILD_VISIBLE_DEVICES" \
    PYTHONUNBUFFERED=1 \
    CSDI_PROGRESS_INTERVAL="$CSDI_PROGRESS_INTERVAL" \
    "$PYTHON" "${COMMON_ARGS[@]}" \
    --device cuda:0 \
    --fold-start "$START" \
    --fold-count "$COUNT" \
    --output-dir "$WORKER_DIR" \
    > "$LOG" 2>&1 &
  PIDS+=("$!")

  if [[ "$STREAM_LOGS" == "1" ]]; then
    tail -n 0 -f "$LOG" | sed -u "s/^/[gpu $GPU] /" &
    TAIL_PIDS+=("$!")
  fi
  START=$((START + COUNT))
done

FAILED=0
for PID in "${PIDS[@]}"; do
  if ! wait "$PID"; then
    FAILED=1
  fi
done

if [[ "${#TAIL_PIDS[@]}" -gt 0 ]]; then
  for TAIL_PID in "${TAIL_PIDS[@]}"; do
    kill "$TAIL_PID" >/dev/null 2>&1 || true
  done
fi

if [[ "$FAILED" -ne 0 ]]; then
  echo "At least one CUDA worker failed. Check logs in $OUTPUT_DIR." >&2
  exit 1
fi

COMBINE_ARGS=(
  "$PROJECT_ROOT/CSDI_Experiment/src/combine_walk_forward_outputs.py"
  --output-dir "$OUTPUT_DIR"
)
if [[ "$COPY_FOLDS" == "1" ]]; then
  COMBINE_ARGS+=(--copy-folds)
fi
COMBINE_ARGS+=("${WORKER_DIRS[@]}")

echo "Combining worker outputs..."
"$PYTHON" "${COMBINE_ARGS[@]}"

if [[ "$RUN_ANALYSIS" == "1" ]]; then
  echo "Generating analysis plots..."
  "$PYTHON" "$PROJECT_ROOT/CSDI_Experiment/src/analyze_paths.py" "$OUTPUT_DIR"
fi

echo "CUDA walk-forward run complete: $OUTPUT_DIR"
