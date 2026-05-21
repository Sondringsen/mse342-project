# CSDI Fixed-Split Scenario Generation

This experiment trains CSDI on returns up to a chosen cutoff date, then
generates future scenario paths for selected horizons. This is the main project
setup because the goal is financial path generation, not rolling point
forecasting.

The default horizons are 1, 5, and 10 trading years:

- 1 year: 252 trading days
- 5 years: 1260 trading days
- 10 years: 2520 trading days

## Run

One-command full cluster run:

```bash
bash CSDI_Experiment/run_cluster.sh
```

This runs matched vanilla vs topology-loss CSDI for 1, 5, and 10-year scenarios
using the default project settings. Override inline when needed:

```bash
EPOCHS=10 NSAMPLE=10 HORIZON_YEARS="1" bash CSDI_Experiment/run_cluster.sh
```

The default experiment now trains in log-return space and reports simple returns
for metrics and plots:

- `RETURN_TRANSFORM=log`
- `CONSTRAINT_LOSS_WEIGHT=0.02`
- `CONSTRAINT_VOLATILITY_WEIGHT=0.25`
- `CONSTRAINT_SAMPLE_CLAMP=1`
- `CONSTRAINT_LOWER_QUANTILE=0.001`
- `CONSTRAINT_UPPER_QUANTILE=0.999`
- `CONSTRAINT_MARGIN_Z=0.5`

This prevents impossible compounded paths from simple returns below `-100%` and
keeps generated standardized log returns within empirical training bounds.

To use multiple GPUs, set `GPUS`. This parallelizes across independent
variant/horizon jobs, with one GPU per job:

```bash
GPUS=4 bash CSDI_Experiment/run_cluster.sh
```

This is cluster-level parallelism, not distributed training inside one model.
For the default `HORIZON_YEARS="1 5 10"`, the script launches six one-GPU jobs:
vanilla/topoloss for each horizon.

Smoke test:

```bash
TRAIN_END_DATE=2015-12-31 \
HORIZON_YEARS="1" \
TARGET_COLUMNS="Agric Food Oil" \
EPOCHS=2 ITR_PER_EPOCH=5 NSAMPLE=5 \
bash CSDI_Experiment/scripts/run_fixed_split_cuda.sh
```

Cluster run:

```bash
TRAIN_END_DATE=2015-12-31 \
HORIZON_YEARS="1 5 10" \
TARGET_COLUMNS="Agric Food Oil Banks Softw Util" \
EPOCHS=50 ITR_PER_EPOCH=100 NSAMPLE=50 \
srun --partition=gpu-turing --gres=gpu:1 \
bash CSDI_Experiment/scripts/run_fixed_split_cuda.sh
```

The script writes to:

```text
CSDI_Experiment/outputs/fixed_split_vanilla_YYYYMMDD_HHMMSS
```

Each horizon gets its own model and artifacts:

- `horizon_0252/model.pth`
- `horizon_0252/train_history.csv`
- `horizon_0252/generated_outputs_nsample50.pk`
- `horizon_0252/predictions.csv`
- `horizon_0252/metrics.json`

The run directory also contains combined `predictions.csv`,
`metrics_by_horizon.csv`, `run_config.json`, and `summary.json`.

`predictions.csv` reports simple-return quantities in `actual`, `pred_median`,
and quantile columns. When `RETURN_TRANSFORM=log`, the model-space values are
also saved as `actual_model`, `pred_median_model`, and matching model quantiles.

## Plots

```bash
../venv/bin/python CSDI_Experiment/src/analyze_paths.py \
  CSDI_Experiment/outputs/fixed_split_vanilla_YYYYMMDD_HHMMSS
```

This writes `plots/` with training curves, calibration, interval coverage,
feature error rankings, cumulative generated path plots, and topology
diagnostics.

## Topological Variant

`run_fixed_split_cuda.sh` exposes `VARIANT_NAME` and `TOPOLOGY_LOSS_WEIGHT` so
the vanilla and topology-regularized runs can share the same interface.

The implemented topology term is a differentiable sliding-window proxy inspired
by persistent-homology workflows:

- build delay/sliding-window point clouds from the generated forecast segment,
- match soft recurrence curves over distance thresholds,
- match sorted pairwise-distance structure,
- match low-frequency spectral power of the market-average path.

This is not a full persistence-diagram backpropagation layer. Full persistent
homology is computed in analytics instead, where the generated samples are
compared to the real holdout path using H0 persistence summaries, Betti-1 graph
cycle proxies, recurrence curves, and low-frequency power.

Run matched vanilla and topology-regularized experiments:

```bash
TRAIN_END_DATE=2015-12-31 \
HORIZON_YEARS="1 5 10" \
TARGET_COLUMNS="Agric Food Oil Banks Softw Util" \
EPOCHS=50 ITR_PER_EPOCH=100 NSAMPLE=50 \
TOPOLOGY_LOSS_WEIGHT=0.05 \
srun --partition=gpu-turing --gres=gpu:1 \
bash CSDI_Experiment/scripts/run_topology_comparison_cuda.sh
```

This writes:

- `topology_comparison_*/vanilla`
- `topology_comparison_*/topoloss`
- `topology_comparison_*/comparison/comparison_by_horizon.csv`
- `topology_comparison_*/comparison/comparison_aggregate.csv`

The topology plots and CSVs are under each run's `plots/` directory:

- `topology_metrics.csv`
- `topology_curves.csv`
- `topology_summary.json`
- `topology_distance_by_horizon.png`
- `topology_feature_distributions.png`
- `topology_curves/horizon_*.png`

## Papers

Reference PDFs are stored in `papers/` with source links in
`papers/README.md`.
