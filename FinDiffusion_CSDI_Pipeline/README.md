# FinDiffusion vs CSDI One-Step Pipeline

This folder is a clean comparison pipeline for the project models using the same
market universe and result analysis style as `FinDiffusion/`.

The main change from `CSDI_Experiment/` is that this is a one-step forecasting
setup:

- data defaults match `FinDiffusion/configs/default.yaml` ticker/date choices,
- each asset is treated as a univariate return series,
- each sample observes a 252-day history and predicts the next trading day,
- both model variants train by predicting diffusion noise epsilon,
- evaluation uses the FinDiffusion metrics and stylized-facts analysis for both
  real test returns and generated return paths.

The `csdi` model in this folder is a local CSDI-style masked diffusion forecaster:
it keeps the observed history fixed, masks the future day, and computes diffusion
loss only on the masked target. It does not depend on the older `CSDI_Experiment`
code path.

## Run

From the repo root:

```bash
python3.11 -m venv FinDiffusion_CSDI_Pipeline/.venv
FinDiffusion_CSDI_Pipeline/.venv/bin/pip install --upgrade pip setuptools wheel
FinDiffusion_CSDI_Pipeline/.venv/bin/pip install -r FinDiffusion_CSDI_Pipeline/requirements.txt
FinDiffusion_CSDI_Pipeline/.venv/bin/python FinDiffusion_CSDI_Pipeline/run_pipeline.py --model both
```

Quick smoke run:

```bash
FinDiffusion_CSDI_Pipeline/.venv/bin/python FinDiffusion_CSDI_Pipeline/run_pipeline.py --model both --debug
```

Use cached price data only:

```bash
FinDiffusion_CSDI_Pipeline/.venv/bin/python FinDiffusion_CSDI_Pipeline/run_pipeline.py --model both --no-download
```

## Cluster

The GPU environment used for this project is loaded inside `run_one_gpu.sh`:

```bash
ml course/cme213/nvhpc/24.1
```

Single GPU, directly matching the course command style:

```bash
srun --partition=gpu-turing --gres=gpu:1 ./FinDiffusion_CSDI_Pipeline/scripts/run_one_gpu.sh --debug
```

Parallel comparison run. This launches `findiffusion` and `csdi` as separate
one-GPU `srun` tasks and writes the combined FinDiffusion-style analysis after
both finish. A five-GPU node has enough room for both jobs at once; this
comparison only needs two GPUs because there are two model variants.

```bash
RUN_NAME=one_step_compare bash FinDiffusion_CSDI_Pipeline/scripts/run_cluster.sh --debug
```

For a tuned cluster run that uses multiple GPUs and avoids the slow 1000-step
evaluation sampler:

```bash
RUN_NAME=one_step_compare GPUS_PER_MODEL=2 \
  bash FinDiffusion_CSDI_Pipeline/scripts/run_cluster.sh \
  --no-download --batch-size 512 --eval-batch-size 16 --num-workers 4 --ddim
```

`GPUS_PER_MODEL=2` wraps each model in PyTorch `DataParallel`. For small batches
this can be slower than one GPU per model, so increase `--batch-size` when using
multiple GPUs inside a single model job. `--ddim` keeps the same evaluation
metrics and plots but uses the faster DDIM sampler for generated forecasts.
Keep `--eval-batch-size` lower than the training batch because each evaluation
window expands to `n_samples` generated paths. The default evaluation volume is
25 generated samples over up to 128 test windows per asset; increase
`--n-samples` or `--max-eval-windows-per-asset` for a heavier final analysis.
On 16-CPU nodes, `--num-workers 4` per model is a reasonable starting point when
both models run on the same node.

## Outputs

Results are written under:

```text
FinDiffusion_CSDI_Pipeline/outputs/<run_name>/
```

Start with:

- `outputs/README.md`: generated index of complete, partial, and incomplete runs.
- `outputs/index.csv`: machine-readable run index.
- `outputs/latest_complete/`: symlink to the newest complete two-model comparison.
- `outputs/latest_run/`: symlink to the most recently modified run of any status.
- `outputs/LATEST_COMPLETE_RUN.txt`: plain-text pointer to the newest complete run.
- `outputs/LATEST_RUN.txt`: plain-text pointer to the newest run of any status.
- `outputs/<run_name>/README.md`: short entry point for one completed run.

Each model directory contains:

- `checkpoints/final.pt`
- `predictions.csv`
- `evaluation_results.json`
- `metrics_report.txt`
- `stylized_facts_report.txt`
- `plots/`

The top-level run directory also contains:

- `README.md`
- `comparison_summary.csv`
- `comparison_summary.json`
- `comparison_report.md`
- `comparison_metrics_report.txt`
- `comparison_forecast_metrics.csv`
- `comparison_distribution_metrics.csv`
- `comparison_temporal_metrics.csv`
- `comparison_diversity_metrics.csv`
- `comparison_score_metrics.csv`
- `comparison_stylized_facts.csv`
- `comparison_metric_rankings.csv`
- `plots/`
- `run_config.yaml`

The comparison files include forecast accuracy/calibration, Wasserstein, KS,
Jensen-Shannon divergence, moment differences including skewness and kurtosis,
raw and squared-return autocorrelation errors, diversity, summary scores, and
stylized-facts diagnostics such as fat tails, volatility clustering, leverage
effect, and raw-return autocorrelation. Volume-volatility correlation is not
included because this pipeline currently uses return data only.

The plot set includes explicit generated daily return time-series plots:

- `<run>/<model>/plots/generated_return_timeseries.png`
- `<run>/plots/comparison_generated_timeseries.png`

## Configuration

Edit `config.yaml` to change history length, stride, tickers, training epochs,
model size, or sample count. Defaults intentionally mirror the FinDiffusion data
setup while changing the task from 252-day unconditional generation to one-day
conditional forecasting.
