#!/usr/bin/env bash
set -euo pipefail

# Fixed-split scenario generation. Trains once per horizon on data up to
# TRAIN_END_DATE, then generates 1/5/10-year paths from the final history window.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

PYTHON="${PYTHON:-$PROJECT_ROOT/../venv/bin/python}"
PIP="${PIP:-$PROJECT_ROOT/../venv/bin/pip3}"

DATA_CSV="${DATA_CSV:-$PROJECT_ROOT/data/processed/french49_daily_returns.csv}"
VARIANT_NAME="${VARIANT_NAME:-vanilla}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/CSDI_Experiment/outputs/fixed_split_${VARIANT_NAME}_$(date +%Y%m%d_%H%M%S)}"

TRAIN_END_DATE="${TRAIN_END_DATE:-2015-12-31}"
TARGET_COLUMNS="${TARGET_COLUMNS:-Agric Food Oil Banks Softw Util}"
RETURN_TRANSFORM="${RETURN_TRANSFORM:-log}"
HISTORY_LENGTH="${HISTORY_LENGTH:-756}"
HORIZON_YEARS="${HORIZON_YEARS:-1 5 10}"
EPOCHS="${EPOCHS:-50}"
ITR_PER_EPOCH="${ITR_PER_EPOCH:-100}"
NSAMPLE="${NSAMPLE:-50}"
TRAIN_STRIDE="${TRAIN_STRIDE:-5}"
VALID_WINDOWS="${VALID_WINDOWS:-4}"
BATCH_SIZE="${BATCH_SIZE:-1}"
TOPOLOGY_LOSS_WEIGHT="${TOPOLOGY_LOSS_WEIGHT:-0}"
TOPOLOGY_WINDOW="${TOPOLOGY_WINDOW:-32}"
TOPOLOGY_STRIDE="${TOPOLOGY_STRIDE:-4}"
TOPOLOGY_MAX_POINTS="${TOPOLOGY_MAX_POINTS:-64}"
TOPOLOGY_THRESHOLDS="${TOPOLOGY_THRESHOLDS:-12}"
TOPOLOGY_TEMPERATURE="${TOPOLOGY_TEMPERATURE:-0.1}"
TOPOLOGY_RECURRENCE_WEIGHT="${TOPOLOGY_RECURRENCE_WEIGHT:-1.0}"
TOPOLOGY_DISTANCE_WEIGHT="${TOPOLOGY_DISTANCE_WEIGHT:-0.25}"
TOPOLOGY_SPECTRUM_WEIGHT="${TOPOLOGY_SPECTRUM_WEIGHT:-0.25}"
CONSTRAINT_LOSS_WEIGHT="${CONSTRAINT_LOSS_WEIGHT:-0.02}"
CONSTRAINT_VOLATILITY_WEIGHT="${CONSTRAINT_VOLATILITY_WEIGHT:-0.25}"
CONSTRAINT_SAMPLE_CLAMP="${CONSTRAINT_SAMPLE_CLAMP:-1}"
CONSTRAINT_LOWER_QUANTILE="${CONSTRAINT_LOWER_QUANTILE:-0.001}"
CONSTRAINT_UPPER_QUANTILE="${CONSTRAINT_UPPER_QUANTILE:-0.999}"
CONSTRAINT_MARGIN_Z="${CONSTRAINT_MARGIN_Z:-0.5}"
CSDI_PROGRESS_INTERVAL="${CSDI_PROGRESS_INTERVAL:-10}"

if [[ ! -x "$PYTHON" ]]; then
  echo "Python executable not found: $PYTHON" >&2
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

DEVICE="${DEVICE:-cuda:0}"
if [[ "$DEVICE" == cuda* ]]; then
  if ! "$PYTHON" -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" >/dev/null 2>&1; then
    echo "CUDA is not available to this Python environment." >&2
    exit 2
  fi
fi

read -r -a COLUMNS <<< "$TARGET_COLUMNS"
read -r -a YEARS <<< "$HORIZON_YEARS"

echo "Output directory: $OUTPUT_DIR"
echo "Variant: $VARIANT_NAME"
echo "Train end date: $TRAIN_END_DATE"
echo "Horizon years: ${YEARS[*]}"
echo "Target columns: ${COLUMNS[*]}"
echo "Return transform: $RETURN_TRANSFORM"
echo "Constraint loss weight: $CONSTRAINT_LOSS_WEIGHT"
echo "Constraint sample clamp: $CONSTRAINT_SAMPLE_CLAMP"

CLAMP_ARG="--constraint-sample-clamp"
if [[ "$CONSTRAINT_SAMPLE_CLAMP" == "0" || "$CONSTRAINT_SAMPLE_CLAMP" == "false" || "$CONSTRAINT_SAMPLE_CLAMP" == "False" ]]; then
  CLAMP_ARG="--no-constraint-sample-clamp"
fi

PYTHONUNBUFFERED=1 \
CSDI_PROGRESS_INTERVAL="$CSDI_PROGRESS_INTERVAL" \
"$PYTHON" "$PROJECT_ROOT/CSDI_Experiment/src/fixed_split_scenarios.py" \
  --data "$DATA_CSV" \
  --device "$DEVICE" \
  --train-end-date "$TRAIN_END_DATE" \
  --return-transform "$RETURN_TRANSFORM" \
  --history-length "$HISTORY_LENGTH" \
  --horizon-years "${YEARS[@]}" \
  --epochs "$EPOCHS" \
  --itr-per-epoch "$ITR_PER_EPOCH" \
  --nsample "$NSAMPLE" \
  --train-stride "$TRAIN_STRIDE" \
  --valid-windows "$VALID_WINDOWS" \
  --batch-size "$BATCH_SIZE" \
  --variant-name "$VARIANT_NAME" \
  --topology-loss-weight "$TOPOLOGY_LOSS_WEIGHT" \
  --topology-window "$TOPOLOGY_WINDOW" \
  --topology-stride "$TOPOLOGY_STRIDE" \
  --topology-max-points "$TOPOLOGY_MAX_POINTS" \
  --topology-thresholds "$TOPOLOGY_THRESHOLDS" \
  --topology-temperature "$TOPOLOGY_TEMPERATURE" \
  --topology-recurrence-weight "$TOPOLOGY_RECURRENCE_WEIGHT" \
  --topology-distance-weight "$TOPOLOGY_DISTANCE_WEIGHT" \
  --topology-spectrum-weight "$TOPOLOGY_SPECTRUM_WEIGHT" \
  --constraint-loss-weight "$CONSTRAINT_LOSS_WEIGHT" \
  --constraint-volatility-weight "$CONSTRAINT_VOLATILITY_WEIGHT" \
  "$CLAMP_ARG" \
  --constraint-lower-quantile "$CONSTRAINT_LOWER_QUANTILE" \
  --constraint-upper-quantile "$CONSTRAINT_UPPER_QUANTILE" \
  --constraint-margin-z "$CONSTRAINT_MARGIN_Z" \
  --target-columns "${COLUMNS[@]}" \
  --output-dir "$OUTPUT_DIR"

echo "Fixed-split scenario run complete: $OUTPUT_DIR"
