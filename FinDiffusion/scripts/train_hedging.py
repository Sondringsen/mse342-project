#!/usr/bin/env python
"""Train a deep hedging model.

Two data sources:
  --model ddpm / ddpm_topo  Load synthetic log returns from a CSV file
                            (default: outputs/{model}/synthetic.csv).
  --model real              Load r_train windows from the real data module
                            (requires --config).

The hedger learns to hedge a European call option (5% OTM, 30-day TTL)
by minimising the CVaR of hedging errors.
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import FinancialDataModule
from src.hedging import DeepHedger, HedgingTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

S0 = 100.0
OTM_FACTOR = 1.05
TTL = 30


def load_synthetic_csv(path: str) -> np.ndarray:
    """Load log-return paths from a synthetic CSV.

    Returns: (N, seq_len) float32 array.
    """
    df = pd.read_csv(path)
    data = df.values.astype(np.float32)
    logger.info(f"Loaded {len(data)} synthetic paths from {path}")
    return data


def load_real_train(config_path: str) -> np.ndarray:
    """Load all log-return windows from the real training split.

    Returns: (N, seq_len) float32 array.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    dm = FinancialDataModule(
        tickers=config["data"]["tickers"],
        start_date=config["data"]["start_date"],
        end_date=config["data"]["end_date"],
        seq_len=config["data"]["seq_len"],
        stride=config["data"]["stride"],
        train_ratio=config["data"]["train_ratio"],
        val_ratio=config["data"]["val_ratio"],
        batch_size=config["evaluation"]["batch_size"],
        data_dir=config["paths"]["data_dir"],
    )
    dm.setup()

    ds = dm.train_dataset
    raw = np.array([ds[i].numpy() for i in range(len(ds))])  # (N, seq_len, 1)
    raw = dm.denormalize(raw)
    if raw.ndim == 3:
        raw = raw[:, :, 0]
    logger.info(f"Loaded {len(raw)} real training paths (r_train)")
    return raw.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="Train deep hedging model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["ddpm", "ddpm_topo", "real"],
        help=(
            "Model variant. 'ddpm'/'ddpm_topo' loads synthetic data from CSV; "
            "'real' loads r_train windows from the data module."
        ),
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to synthetic CSV (ddpm/ddpm_topo only; default: outputs/{model}/synthetic.csv)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Config file (required for --model real)",
    )
    parser.add_argument("--n_samples",  type=int,   default=10000, help="Max training paths to use")
    parser.add_argument("--n_epochs",   type=int,   default=100)
    parser.add_argument("--batch_size", type=int,   default=256)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int,   default=64)
    parser.add_argument("--cvar_alpha", type=float, default=0.95)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: outputs/{model}/hedging/)",
    )
    args = parser.parse_args()

    model_dir = Path("outputs") / args.model
    output_dir = Path(args.output_dir or model_dir / "hedging")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    strike = S0 * OTM_FACTOR
    logger.info(f"Option: European call  S0={S0}  K={strike:.1f} ({(OTM_FACTOR-1):.0%} OTM)  T={TTL} days")

    # --- Load training data ---
    if args.model == "real":
        log_returns = load_real_train(args.config)
    else:
        data_path = args.data or str(model_dir / "synthetic.csv")
        log_returns = load_synthetic_csv(data_path)
        if len(log_returns) > args.n_samples:
            idx = np.random.choice(len(log_returns), args.n_samples, replace=False)
            log_returns = log_returns[idx]

    logger.info(f"Training data shape: {log_returns.shape}")

    # --- Build and train hedging model ---
    hedger = DeepHedger(input_dim=2, hidden_dim=args.hidden_dim)
    logger.info(f"DeepHedger parameters: {sum(p.numel() for p in hedger.parameters()):,}")

    trainer = HedgingTrainer(
        model=hedger,
        strike=strike,
        seq_len=TTL,
        lr=args.lr,
        cvar_alpha=args.cvar_alpha,
        device=device,
    )

    logger.info("Training ...")
    losses = trainer.train(
        synthetic_log_returns=log_returns,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        checkpoint_path=output_dir / "hedging_model.pt",
        s0=S0,
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(losses)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(f"CVaR({args.cvar_alpha:.0%}) Loss")
    ax.set_title(f"Deep Hedging — Training Loss ({args.model})")
    plt.tight_layout()
    plt.savefig(output_dir / "training_loss.png", dpi=150)
    plt.close()

    logger.info(f"Best model saved to {output_dir}/hedging_model.pt")
    logger.info("Done.")


if __name__ == "__main__":
    main()
