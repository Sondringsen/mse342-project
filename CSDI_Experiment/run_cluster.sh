#!/usr/bin/env bash
set -euo pipefail

# One-command cluster run for the project experiment.
#
# From ~/mse342-project on the cluster:
#   bash CSDI_Experiment/run_cluster.sh
#
# Defaults run matched vanilla vs topology-loss CSDI for 1, 5, and 10-year
# scenarios. Override settings inline if needed, for example:
#   EPOCHS=10 NSAMPLE=10 bash CSDI_Experiment/run_cluster.sh
#   GPUS=4 bash CSDI_Experiment/run_cluster.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

export TRAIN_END_DATE="${TRAIN_END_DATE:-2015-12-31}"
export HORIZON_YEARS="${HORIZON_YEARS:-1 5 10}"
export TARGET_COLUMNS="${TARGET_COLUMNS:-Agric Food Oil Banks Softw Util}"
export RETURN_TRANSFORM="${RETURN_TRANSFORM:-log}"
export HISTORY_LENGTH="${HISTORY_LENGTH:-756}"
export EPOCHS="${EPOCHS:-50}"
export ITR_PER_EPOCH="${ITR_PER_EPOCH:-100}"
export NSAMPLE="${NSAMPLE:-50}"
export TRAIN_STRIDE="${TRAIN_STRIDE:-5}"
export VALID_WINDOWS="${VALID_WINDOWS:-4}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
export TOPOLOGY_LOSS_WEIGHT="${TOPOLOGY_LOSS_WEIGHT:-0.05}"
export CONSTRAINT_LOSS_WEIGHT="${CONSTRAINT_LOSS_WEIGHT:-0.02}"
export CONSTRAINT_VOLATILITY_WEIGHT="${CONSTRAINT_VOLATILITY_WEIGHT:-0.25}"
export CONSTRAINT_SAMPLE_CLAMP="${CONSTRAINT_SAMPLE_CLAMP:-1}"
export CONSTRAINT_LOWER_QUANTILE="${CONSTRAINT_LOWER_QUANTILE:-0.001}"
export CONSTRAINT_UPPER_QUANTILE="${CONSTRAINT_UPPER_QUANTILE:-0.999}"
export CONSTRAINT_MARGIN_Z="${CONSTRAINT_MARGIN_Z:-0.5}"
export CSDI_PROGRESS_INTERVAL="${CSDI_PROGRESS_INTERVAL:-10}"
export DEVICE="${DEVICE:-cuda:0}"
export GPUS="${GPUS:-1}"
export SLURM_PARTITION="${SLURM_PARTITION:-gpu-turing}"
export RUN_MODE="${RUN_MODE:-auto}"

PYTHON="${PYTHON:-$PROJECT_ROOT/../venv/bin/python}"

if command -v ml >/dev/null 2>&1; then
  ml course/cme213/nvhpc/24.1 || true
fi

if [[ -f "$PROJECT_ROOT/../venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$PROJECT_ROOT/../venv/bin/activate"
fi

echo "Running CSDI topology comparison experiment"
echo "Project root: $PROJECT_ROOT"
echo "Train end date: $TRAIN_END_DATE"
echo "Horizons: $HORIZON_YEARS year(s)"
echo "Target columns: $TARGET_COLUMNS"
echo "Return transform: $RETURN_TRANSFORM"
echo "Epochs: $EPOCHS"
echo "Iterations per epoch: $ITR_PER_EPOCH"
echo "Samples: $NSAMPLE"
echo "Topology loss weight: $TOPOLOGY_LOSS_WEIGHT"
echo "Constraint loss weight: $CONSTRAINT_LOSS_WEIGHT"
echo "Constraint sample clamp: $CONSTRAINT_SAMPLE_CLAMP"
echo "Requested GPUs: $GPUS"

if [[ "$RUN_MODE" == "auto" ]]; then
  if [[ "$GPUS" -gt 1 ]]; then
    RUN_MODE="parallel"
  else
    RUN_MODE="sequential"
  fi
fi

if [[ "$RUN_MODE" == "sequential" ]]; then
  srun --partition="$SLURM_PARTITION" --gres="gpu:1" \
    bash "$PROJECT_ROOT/CSDI_Experiment/scripts/run_topology_comparison_cuda.sh"
  exit 0
fi

if [[ "$RUN_MODE" != "parallel" ]]; then
  echo "Unknown RUN_MODE=$RUN_MODE. Use auto, sequential, or parallel." >&2
  exit 2
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
COMPARISON_ROOT="${COMPARISON_ROOT:-$PROJECT_ROOT/CSDI_Experiment/outputs/topology_comparison_parallel_$STAMP}"
mkdir -p "$COMPARISON_ROOT/logs"

read -r -a YEARS <<< "$HORIZON_YEARS"
RUN_DIRS=()
LABELS=()
PIDS=()

launch_job() {
  local variant="$1"
  local weight="$2"
  local year="$3"
  local out_dir="$COMPARISON_ROOT/${variant}_${year}y"
  local log_path="$COMPARISON_ROOT/logs/${variant}_${year}y.log"

  RUN_DIRS+=("$out_dir")
  LABELS+=("$variant")
  echo "Launching $variant horizon=${year}y -> $out_dir"
  env \
    TRAIN_END_DATE="$TRAIN_END_DATE" \
    HORIZON_YEARS="$year" \
    TARGET_COLUMNS="$TARGET_COLUMNS" \
    RETURN_TRANSFORM="$RETURN_TRANSFORM" \
    HISTORY_LENGTH="$HISTORY_LENGTH" \
    EPOCHS="$EPOCHS" \
    ITR_PER_EPOCH="$ITR_PER_EPOCH" \
    NSAMPLE="$NSAMPLE" \
    TRAIN_STRIDE="$TRAIN_STRIDE" \
    VALID_WINDOWS="$VALID_WINDOWS" \
    BATCH_SIZE="$BATCH_SIZE" \
    CSDI_PROGRESS_INTERVAL="$CSDI_PROGRESS_INTERVAL" \
    CONSTRAINT_LOSS_WEIGHT="$CONSTRAINT_LOSS_WEIGHT" \
    CONSTRAINT_VOLATILITY_WEIGHT="$CONSTRAINT_VOLATILITY_WEIGHT" \
    CONSTRAINT_SAMPLE_CLAMP="$CONSTRAINT_SAMPLE_CLAMP" \
    CONSTRAINT_LOWER_QUANTILE="$CONSTRAINT_LOWER_QUANTILE" \
    CONSTRAINT_UPPER_QUANTILE="$CONSTRAINT_UPPER_QUANTILE" \
    CONSTRAINT_MARGIN_Z="$CONSTRAINT_MARGIN_Z" \
    DEVICE="$DEVICE" \
    VARIANT_NAME="$variant" \
    TOPOLOGY_LOSS_WEIGHT="$weight" \
    OUTPUT_DIR="$out_dir" \
    srun --partition="$SLURM_PARTITION" --gres="gpu:1" \
      bash "$PROJECT_ROOT/CSDI_Experiment/scripts/run_fixed_split_cuda.sh" \
      > "$log_path" 2>&1 &
  PIDS+=("$!")
}

wait_batch() {
  local pid
  local failed=0
  for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  PIDS=()
  if [[ "$failed" -ne 0 ]]; then
    echo "At least one GPU job failed. Check logs in $COMPARISON_ROOT/logs" >&2
    exit 1
  fi
}

echo "Parallel comparison root: $COMPARISON_ROOT"
echo "Launching up to $GPUS one-GPU jobs at a time"

for year in "${YEARS[@]}"; do
  launch_job "vanilla" "0" "$year"
  if [[ "${#PIDS[@]}" -ge "$GPUS" ]]; then
    wait_batch
  fi

  launch_job "topoloss" "$TOPOLOGY_LOSS_WEIGHT" "$year"
  if [[ "${#PIDS[@]}" -ge "$GPUS" ]]; then
    wait_batch
  fi
done

if [[ "${#PIDS[@]}" -gt 0 ]]; then
  wait_batch
fi

echo "All GPU jobs finished. Running analytics..."
for run_dir in "${RUN_DIRS[@]}"; do
  "$PYTHON" "$PROJECT_ROOT/CSDI_Experiment/src/analyze_paths.py" \
    "$run_dir" \
    --topology-window "${TOPOLOGY_ANALYSIS_WINDOW:-32}" \
    --topology-stride "${TOPOLOGY_ANALYSIS_STRIDE:-4}" \
    --topology-max-points "${TOPOLOGY_ANALYSIS_MAX_POINTS:-80}" \
    --topology-samples "${TOPOLOGY_ANALYSIS_SAMPLES:-50}"
done

"$PYTHON" "$PROJECT_ROOT/CSDI_Experiment/src/compare_runs.py" \
  "${RUN_DIRS[@]}" \
  --labels "${LABELS[@]}" \
  --output-dir "$COMPARISON_ROOT/comparison"

echo "Investigating raw data and generated paths..."
"$PYTHON" "$PROJECT_ROOT/CSDI_Experiment/src/investigate_results.py" \
  "$COMPARISON_ROOT"

if [[ "${PACKAGE_RESULTS:-1}" == "1" ]]; then
  echo "Packaging shareable results..."
  "$PYTHON" "$PROJECT_ROOT/CSDI_Experiment/src/package_results.py" \
    "$COMPARISON_ROOT"
fi

echo "Parallel topology comparison complete: $COMPARISON_ROOT"
