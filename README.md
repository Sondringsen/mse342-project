# mse342-project

## Usage

### Training

```bash
cd FinDiffusion
python scripts/train.py --config configs/default.yaml
```

Optional flags:
- `--checkpoint PATH` — resume from a saved checkpoint
- `--wandb` — enable Weights & Biases logging
- `--debug` — quick run with small dataset and few epochs
- `--seed INT` — set random seed for reproducibility

### Evaluation (single condition)

```bash
cd FinDiffusion
python scripts/evaluate_single.py \
  --checkpoint outputs/checkpoints/best_model.pt \
  --config configs/default.yaml \
  --trend 0.1 \
  --vol 0.2 \
  --regime sideways \
  --n_samples 1000 \
  --output_dir outputs/evaluation_single
```

To skip generation and load a previously saved `synthetic.csv`:

```bash
python scripts/evaluate_single.py --data outputs/evaluation_single/synthetic.csv
```

Add `--ddim` to either command for faster sampling.

### Generate synthetic data

```bash
cd FinDiffusion
python scripts/generate.py \
  --checkpoint outputs/checkpoints/best_model.pt \
  --config configs/default.yaml \
  --n_samples 1000 \
  --output outputs/synthetic_returns.csv
```

Optionally condition on market regime:

```bash
python scripts/generate.py \
  --checkpoint outputs/checkpoints/best_model.pt \
  --trend 0.1 \
  --volatility 0.2 \
  --regime bull \
  --output outputs/synthetic_bull.csv
```

Add `--ddim` for faster sampling, or `--ddim_steps INT` to control the number of DDIM steps (default 50).

### Deep hedging

Train on synthetic data (European call, 5% OTM, 30-day TTL):

```bash
cd FinDiffusion
python scripts/train_hedging.py \
  --checkpoint checkpoints/final.pt \
  --config configs/default.yaml \
  --n_samples 10000 \
  --n_epochs 100 \
  --output_dir outputs/hedging
```

Evaluate on real held-out data (reports CVaR and drawdown):

```bash
python scripts/evaluate_hedging.py \
  --model outputs/hedging/hedging_model.pt \
  --config configs/default.yaml \
  --output_dir outputs/hedging_eval
```