#!/usr/bin/env bash
set -euo pipefail

# Long-horizon CSDI run for business-cycle/regime-aware financial path generation.
#
# Defaults:
#   - condition on 3 years of daily returns: HISTORY_LENGTH=756
#   - generate/evaluate 1 year forward: PRED_LENGTH=252
#   - advance one year per walk-forward fold: STEP_SIZE=252
#   - use a tractable sector subset while iterating
#
# Override any setting inline, for example:
#   EPOCHS=50 NSAMPLE=50 N_FOLDS=5 TARGET_COLUMNS="Agric Food Oil Banks Softw Util" ./CSDI_Experiment/run.sh
#
# This delegates to scripts/run_walk_forward_mps.sh, which requires MPS by
# default. Set REQUIRE_MPS=0 to allow CPU fallback.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export TARGET_COLUMNS="${TARGET_COLUMNS:-Agric Food Oil Banks Softw Util}"
export HISTORY_LENGTH="${HISTORY_LENGTH:-756}"
export PRED_LENGTH="${PRED_LENGTH:-252}"
export STEP_SIZE="${STEP_SIZE:-252}"
export N_FOLDS="${N_FOLDS:-3}"
export EPOCHS="${EPOCHS:-20}"
export ITR_PER_EPOCH="${ITR_PER_EPOCH:-50}"
export NSAMPLE="${NSAMPLE:-20}"
export TRAIN_STRIDE="${TRAIN_STRIDE:-5}"
export VALID_WINDOWS="${VALID_WINDOWS:-4}"

exec bash "$SCRIPT_DIR/scripts/run_walk_forward_mps.sh"
