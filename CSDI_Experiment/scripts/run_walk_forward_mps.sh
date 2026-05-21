#!/usr/bin/env bash
set -euo pipefail

# Runs the CSDI walk-forward experiment through the MPS device path.
#
# Common overrides:
#   EPOCHS=50 ITR_PER_EPOCH=100 NSAMPLE=50 N_FOLDS=3 bash CSDI_Experiment/scripts/run_walk_forward_mps.sh
#   TARGET_COLUMNS="Agric Food Oil" bash CSDI_Experiment/scripts/run_walk_forward_mps.sh
#   REQUIRE_MPS=0 bash CSDI_Experiment/scripts/run_walk_forward_mps.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

PYTHON="${PYTHON:-$PROJECT_ROOT/../venv/bin/python}"
PIP="${PIP:-$PROJECT_ROOT/../venv/bin/pip3}"

DATA_CSV="${DATA_CSV:-$PROJECT_ROOT/data/processed/french49_daily_returns.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/CSDI_Experiment/outputs/mps_walk_forward_$(date +%Y%m%d_%H%M%S)}"

EPOCHS="${EPOCHS:-50}"
ITR_PER_EPOCH="${ITR_PER_EPOCH:-100}"
NSAMPLE="${NSAMPLE:-50}"
N_FOLDS="${N_FOLDS:-3}"
VALID_WINDOWS="${VALID_WINDOWS:-8}"
TRAIN_STRIDE="${TRAIN_STRIDE:-1}"
HISTORY_LENGTH="${HISTORY_LENGTH:-231}"
PRED_LENGTH="${PRED_LENGTH:-21}"
REQUIRE_MPS="${REQUIRE_MPS:-1}"

if [[ ! -x "$PYTHON" ]]; then
  echo "Python executable not found: $PYTHON" >&2
  echo "Set PYTHON=/path/to/python or create the project venv first." >&2
  exit 1
fi

if ! "$PYTHON" -c "import torch, yaml, pandas, numpy, tqdm, linear_attention_transformer" >/dev/null 2>&1; then
  if [[ ! -x "$PIP" ]]; then
    echo "Missing CSDI dependencies and pip executable was not found: $PIP" >&2
    echo "Set PIP=/path/to/pip or install CSDI/requirements.txt manually." >&2
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

read -r MPS_BUILT MPS_AVAILABLE < <(
  "$PYTHON" -c "import torch; print(int(torch.backends.mps.is_built()), int(torch.backends.mps.is_available()))"
)

echo "PyTorch MPS status: built=$MPS_BUILT available=$MPS_AVAILABLE"
if [[ "$MPS_AVAILABLE" != "1" && "$REQUIRE_MPS" == "1" ]]; then
  echo "MPS is not available to this PyTorch build/runtime, so the MPS run was not started." >&2
  echo "Set REQUIRE_MPS=0 to let the pipeline fall back to CPU." >&2
  exit 2
fi

export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"

ARGS=(
  "$PROJECT_ROOT/CSDI_Experiment/src/walk_forward_csdi.py"
  --data "$DATA_CSV"
  --device mps
  --epochs "$EPOCHS"
  --itr-per-epoch "$ITR_PER_EPOCH"
  --nsample "$NSAMPLE"
  --n-folds "$N_FOLDS"
  --valid-windows "$VALID_WINDOWS"
  --train-stride "$TRAIN_STRIDE"
  --history-length "$HISTORY_LENGTH"
  --pred-length "$PRED_LENGTH"
  --output-dir "$OUTPUT_DIR"
)

if [[ -n "${BATCH_SIZE:-}" ]]; then
  ARGS+=(--batch-size "$BATCH_SIZE")
fi

if [[ -n "${INITIAL_TRAIN_SIZE:-}" ]]; then
  ARGS+=(--initial-train-size "$INITIAL_TRAIN_SIZE")
fi

if [[ -n "${STEP_SIZE:-}" ]]; then
  ARGS+=(--step-size "$STEP_SIZE")
fi

if [[ -n "${LR:-}" ]]; then
  ARGS+=(--lr "$LR")
fi

if [[ -n "${TARGET_COLUMNS:-}" ]]; then
  read -r -a COLUMNS <<< "$TARGET_COLUMNS"
  ARGS+=(--target-columns "${COLUMNS[@]}")
fi

echo "Output directory: $OUTPUT_DIR"
echo "Starting walk-forward CSDI run..."
"$PYTHON" "${ARGS[@]}"
