# mse342-project

## Model variants

All scripts accept `--model {ddpm,ddpm_topo}` (or `real` where applicable (used for training baseline hedging model)). Outputs are
stored under `outputs/{model}/` so each variant has its own isolated artefacts:

```
outputs/
  ddpm/
    checkpoints/final.pt      ← trained diffusion model
    synthetic.csv             ← generated training data for the hedger
    hedging/hedging_model.pt  ← trained deep hedging model
  ddpm_topo/
    (same layout)
  real/
    hedging/hedging_model.pt  ← hedging model trained on real r_train (baseline)
```
---
## Download data


```bash
cd FinDiffusion
python scripts/download_data.py --config configs/default.yaml
```

---

## Running the Full Training Pipeline
Runs all three phases (train DDPM model, generate data, train hedging model on synthetic data) in sequence for a given variant. After this we have everything we need for evaluation. We can either run locally takes about 30 minutes or so (mayeb more) or on Sherlock compute cluster.

### Locally

```bash
cd FinDiffusion
python scripts/pipeline.py --model ddpm --ddim
```

For debugging you can use

```bash
cd FinDiffusion
python scripts/pipeline.py --model ddpm --debug --ddim
```

Optional overrides:

```bash
python scripts/pipeline.py \
  --model ddpm \
  --config configs/default.yaml \
  --n_generate 20000 \
  --n_epochs_hedging 150 \
  --ddim
```

`--model real` is not supported by the pipeline — train the real baseline directly
with `train_hedging.py` (see below).

### Running on Stanford Sherlock (SLURM)

Pre-download data before transferring to Sherlock (compute nodes have no internet). We use `scp` to transfer the data.

Submit the pipeline as a SLURM job:

```bash
sbatch scripts/sherlock_train.sh                     # defaults to --model ddpm
sbatch scripts/sherlock_train.sh --model ddpm
sbatch scripts/sherlock_train.sh --model ddpm_topo
```

W&B runs in offline mode on the cluster. Sync results from the login node after the job completes:

```bash
wandb sync wandb/run-*
```

---

## Individual scripts

We can also run the scripts individually. As a logical overview, this is the order we need to do things in:
1. Download data
2. Train the diffusion model
3. Generate synthetic data
4. Train the hedging model
5. Evaluate the diffusion model and the hedging model

### 1. Download data

Downloads historical price data from Yahoo Finance and saves it to `data/`:

```bash
cd FinDiffusion
python scripts/download_data.py --config configs/default.yaml
```

Tickers, start date, and end date are read from the config. Individual flags override the config:
- `--tickers AAPL,MSFT,...` — comma-separated tickers
- `--n_tickers N` — cap the number of tickers from config
- `--start YYYY-MM-DD` — override start date
- `--end YYYY-MM-DD` — override end date
- `--output PATH` — output CSV path (default: `data/prices.csv`)

### 2. Train diffusion model

```bash
cd FinDiffusion
python scripts/train.py --model ddpm --config configs/default.yaml
```

Optional flags:
- `--checkpoint PATH` — resume from a saved checkpoint
- `--wandb` — enable Weights & Biases logging
- `--debug` — quick run with small dataset and few epochs
- `--seed INT` — set random seed for reproducibility

Saves weights to `outputs/{model}/checkpoints/final.pt`.

### 3. Generate synthetic data

```bash
cd FinDiffusion
python scripts/generate.py --model ddpm --config configs/default.yaml --n_samples 10000 --ddim
```

Add `--ddim` / `--ddim_steps INT` for faster sampling. Saves to
`outputs/{model}/synthetic.csv`.

### 4. Train deep hedging model

On synthetic data (ddpm or ddpm_topo):

```bash
cd FinDiffusion
python scripts/train_hedging.py \
  --model ddpm \
  --n_epochs 1000
```

`--data PATH` overrides the default CSV path (`outputs/{model}/synthetic.csv`).

On real r_train data (baseline — no synthetic data, uses all training windows):

```bash
python scripts/train_hedging.py \
  --model real \
  --config configs/default.yaml \
  --n_epochs 1000
```

Saves weights to `outputs/{model}/hedging/hedging_model.pt`.

### 5.1 Evaluate diffusion model quality (single condition)

Load the pre-generated CSV from the pipeline. Real test data is loaded automatically from `data/` via the config:

```bash
cd FinDiffusion
python scripts/evaluate_single.py \
  --data outputs/ddpm/synthetic.csv \
  --output_dir outputs/ddpm/evaluation_single
```

To generate a separate batch for evaluation first:

```bash
python scripts/generate.py \
  --model ddpm \
  --n_samples 1000 --ddim \
  --output outputs/ddpm/evaluation_single/synthetic.csv

python scripts/evaluate_single.py \
  --data outputs/ddpm/evaluation_single/synthetic.csv \
  --output_dir outputs/ddpm/evaluation_single
```

### 5.2 Evaluate hedging model on real test data

```bash
cd FinDiffusion
python scripts/evaluate_hedging.py \
  --model ddpm \
  --config configs/default.yaml
```

`--hedger PATH` overrides the default model path
(`outputs/{model}/hedging/hedging_model.pt`). Results saved to
`outputs/{model}/hedging_eval/`.

Evaluate the real baseline:

```bash
python scripts/evaluate_hedging.py --model real --config configs/default.yaml
```