# CSDI Walk-Forward Experiment

This folder contains a walk-forward pipeline for the CSDI forecasting model.
Each fold:

1. fits feature-wise normalizers using only rows before the forecast origin,
2. trains CSDI on expanding historical windows whose targets also end before the origin,
3. predicts the next `pred-length` rows from the most recent `history-length` rows,
4. writes fold-level and aggregate prediction/metric files.

Run from `mse342-project`:

```bash
../venv/bin/pip3 install -r CSDI/requirements.txt
```

```bash
../venv/bin/python CSDI_Experiment/src/walk_forward_csdi.py \
  --device cpu \
  --epochs 5 \
  --itr-per-epoch 50 \
  --nsample 10 \
  --n-folds 3
```

To run through PyTorch's MPS path on Apple Silicon:

```bash
bash CSDI_Experiment/scripts/run_walk_forward_mps.sh
```

For the long-horizon business-cycle/path-generation experiment, use:

```bash
bash CSDI_Experiment/run.sh
```

This defaults to conditioning on `756` trading days and generating/evaluating
the next `252` trading days for a small sector subset. Override settings inline:

```bash
EPOCHS=50 NSAMPLE=50 N_FOLDS=5 \
  TARGET_COLUMNS="Agric Food Oil Banks Softw Util" \
  bash CSDI_Experiment/run.sh
```

The MPS runner defaults to `N_FOLDS=3`, `EPOCHS=50`, `ITR_PER_EPOCH=100`,
and `NSAMPLE=50`. Override them as environment variables:

```bash
N_FOLDS=0 EPOCHS=100 ITR_PER_EPOCH=200 NSAMPLE=100 \
  bash CSDI_Experiment/scripts/run_walk_forward_mps.sh
```

Use `TARGET_COLUMNS="Agric Food Oil"` to train on a smaller subset. The runner
requires MPS by default; set `REQUIRE_MPS=0` only if you want CPU fallback.

To run folds across CUDA GPUs:

```bash
bash CSDI_Experiment/scripts/run_cuda.sh
```

By default, the CUDA runner uses every CUDA GPU visible to PyTorch, runs all
available folds (`N_FOLDS=0`), combines worker outputs, and creates analysis
plots. You can choose GPUs and shorten the run:

```bash
GPUS="0 1" N_FOLDS=12 EPOCHS=50 ITR_PER_EPOCH=100 NSAMPLE=50 \
  bash CSDI_Experiment/scripts/run_cuda.sh
```

Useful CUDA runner overrides:

- `GPUS="0 1 2 3"`: CUDA device ids to use.
- `OUTPUT_DIR=...`: destination run directory.
- `RUN_ANALYSIS=0`: skip plot generation after training.
- `COPY_FOLDS=0`: keep fold artifacts only in `workers/gpu_*` subdirectories.
- `TARGET_COLUMNS="Agric Food Oil"`: train on a smaller feature subset.

The default input is `data/processed/french49_daily_returns.csv`, forecasting all
non-date columns with `history-length=231` and `pred-length=21`.

Useful options:

- `--initial-train-size`: first forecast origin as an exclusive row index.
- `--step-size`: rows to advance between origins; defaults to `pred-length`.
- `--n-folds`: number of walk-forward folds; use `0` to run every possible fold.
- `--target-columns`: subset of CSV columns to forecast.
- `--output-dir`: explicit output folder, useful with `--skip-training`.

Outputs:

- `run_config.json`: full run setup and fold definitions.
- `fold_000/model.pth`: trained CSDI weights for the fold.
- `fold_000/train_history.csv`: epoch-level training loss and validation loss when validation runs.
- `fold_000/predictions.csv`: long-form actuals, sample median/mean, and quantiles.
- `fold_000/generated_outputs_nsample10.pk`: raw generated sample paths, targets, masks, and scalers.
- `fold_000/metrics.json`: fold MAE/RMSE over the forecast horizon.
- `predictions.csv`: all folds concatenated.
- `metrics_by_fold.csv` and `summary.json`: aggregate metrics.

Generate analysis plots after a run:

```bash
../venv/bin/python CSDI_Experiment/src/analyze_paths.py \
  CSDI_Experiment/outputs/walk_forward_YYYYMMDD_HHMMSS
```

This writes `plots/` inside the run directory, including:

- training loss curves from `train_history.csv`,
- MAE/RMSE/bias by forecast horizon,
- 50% and 90% interval coverage and interval width,
- feature error rankings,
- actual-vs-predicted median scatter,
- quantile calibration,
- forecast fan charts by fold and feature,
- cumulative return plots of generated sample paths versus realized paths.

Path evaluation uses two lenses. First, point accuracy checks whether the
generated path center is close to the realized path, using median-path MAE,
RMSE, bias, and feature/horizon breakdowns. Second, distribution quality checks
whether the realized path falls inside the generated predictive distribution at
the right frequency, using interval coverage, interval width, quantile
calibration, and cumulative path plots.
