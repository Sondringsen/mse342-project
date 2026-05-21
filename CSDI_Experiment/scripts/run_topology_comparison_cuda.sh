#!/usr/bin/env bash
set -euo pipefail

# Runs matched vanilla and topology-regularized fixed-split CSDI experiments,
# analyzes both, and writes comparison tables.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

PYTHON="${PYTHON:-$PROJECT_ROOT/../venv/bin/python}"
STAMP="$(date +%Y%m%d_%H%M%S)"
COMPARISON_ROOT="${COMPARISON_ROOT:-$PROJECT_ROOT/CSDI_Experiment/outputs/topology_comparison_$STAMP}"
TOPOLOGY_LOSS_WEIGHT="${TOPOLOGY_LOSS_WEIGHT:-0.05}"

COMMON_ENV=(
  "TRAIN_END_DATE=${TRAIN_END_DATE:-2015-12-31}"
  "HORIZON_YEARS=${HORIZON_YEARS:-1 5 10}"
  "TARGET_COLUMNS=${TARGET_COLUMNS:-Agric Food Oil Banks Softw Util}"
  "RETURN_TRANSFORM=${RETURN_TRANSFORM:-log}"
  "HISTORY_LENGTH=${HISTORY_LENGTH:-756}"
  "EPOCHS=${EPOCHS:-50}"
  "ITR_PER_EPOCH=${ITR_PER_EPOCH:-100}"
  "NSAMPLE=${NSAMPLE:-50}"
  "TRAIN_STRIDE=${TRAIN_STRIDE:-5}"
  "VALID_WINDOWS=${VALID_WINDOWS:-4}"
  "BATCH_SIZE=${BATCH_SIZE:-1}"
  "DEVICE=${DEVICE:-cuda:0}"
  "CSDI_PROGRESS_INTERVAL=${CSDI_PROGRESS_INTERVAL:-10}"
  "TOPOLOGY_WINDOW=${TOPOLOGY_WINDOW:-32}"
  "TOPOLOGY_STRIDE=${TOPOLOGY_STRIDE:-4}"
  "TOPOLOGY_MAX_POINTS=${TOPOLOGY_MAX_POINTS:-64}"
  "CONSTRAINT_LOSS_WEIGHT=${CONSTRAINT_LOSS_WEIGHT:-0.02}"
  "CONSTRAINT_VOLATILITY_WEIGHT=${CONSTRAINT_VOLATILITY_WEIGHT:-0.25}"
  "CONSTRAINT_SAMPLE_CLAMP=${CONSTRAINT_SAMPLE_CLAMP:-1}"
  "CONSTRAINT_LOWER_QUANTILE=${CONSTRAINT_LOWER_QUANTILE:-0.001}"
  "CONSTRAINT_UPPER_QUANTILE=${CONSTRAINT_UPPER_QUANTILE:-0.999}"
  "CONSTRAINT_MARGIN_Z=${CONSTRAINT_MARGIN_Z:-0.5}"
)

mkdir -p "$COMPARISON_ROOT"

echo "Comparison root: $COMPARISON_ROOT"
echo "Running vanilla baseline..."
env "${COMMON_ENV[@]}" \
  VARIANT_NAME=vanilla \
  TOPOLOGY_LOSS_WEIGHT=0 \
  OUTPUT_DIR="$COMPARISON_ROOT/vanilla" \
  bash "$SCRIPT_DIR/run_fixed_split_cuda.sh"

echo "Running topology-regularized variant..."
env "${COMMON_ENV[@]}" \
  VARIANT_NAME=topoloss \
  TOPOLOGY_LOSS_WEIGHT="$TOPOLOGY_LOSS_WEIGHT" \
  OUTPUT_DIR="$COMPARISON_ROOT/topoloss" \
  bash "$SCRIPT_DIR/run_fixed_split_cuda.sh"

echo "Analyzing vanilla baseline..."
"$PYTHON" "$PROJECT_ROOT/CSDI_Experiment/src/analyze_paths.py" \
  "$COMPARISON_ROOT/vanilla" \
  --topology-window "${TOPOLOGY_ANALYSIS_WINDOW:-32}" \
  --topology-stride "${TOPOLOGY_ANALYSIS_STRIDE:-4}" \
  --topology-max-points "${TOPOLOGY_ANALYSIS_MAX_POINTS:-80}" \
  --topology-samples "${TOPOLOGY_ANALYSIS_SAMPLES:-50}"

echo "Analyzing topology-regularized variant..."
"$PYTHON" "$PROJECT_ROOT/CSDI_Experiment/src/analyze_paths.py" \
  "$COMPARISON_ROOT/topoloss" \
  --topology-window "${TOPOLOGY_ANALYSIS_WINDOW:-32}" \
  --topology-stride "${TOPOLOGY_ANALYSIS_STRIDE:-4}" \
  --topology-max-points "${TOPOLOGY_ANALYSIS_MAX_POINTS:-80}" \
  --topology-samples "${TOPOLOGY_ANALYSIS_SAMPLES:-50}"

"$PYTHON" "$PROJECT_ROOT/CSDI_Experiment/src/compare_runs.py" \
  "$COMPARISON_ROOT/vanilla" \
  "$COMPARISON_ROOT/topoloss" \
  --labels vanilla topoloss \
  --output-dir "$COMPARISON_ROOT/comparison"

echo "Investigating raw data and generated paths..."
"$PYTHON" "$PROJECT_ROOT/CSDI_Experiment/src/investigate_results.py" \
  "$COMPARISON_ROOT"

if [[ "${PACKAGE_RESULTS:-1}" == "1" ]]; then
  echo "Packaging shareable results..."
  "$PYTHON" "$PROJECT_ROOT/CSDI_Experiment/src/package_results.py" \
    "$COMPARISON_ROOT"
fi

echo "Topology comparison complete: $COMPARISON_ROOT"
