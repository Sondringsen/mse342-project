#!/usr/bin/env python
"""Generate synthetic financial data using a trained FinDiffusion model."""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import FinancialDiffusion
from src.data import FinancialDataModule

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic financial data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ddpm",
        choices=["ddpm", "ddpm_topo"],
        help="Model variant — used to resolve default checkpoint and output paths",
    )
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (default: outputs/{model}/checkpoints/final.pt)")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--n_samples", type=int, default=10000)
    parser.add_argument("--trend", type=float, default=None, help="Target annualized return")
    parser.add_argument("--volatility", type=float, default=None, help="Target annualized volatility")
    parser.add_argument("--regime", type=str, default=None, choices=["bear", "sideways", "bull"])
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV path (default: outputs/{model}/synthetic.csv)")
    parser.add_argument("--ddim", action="store_true", help="Use DDIM for faster sampling")
    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    model_dir = Path("outputs") / args.model
    checkpoint_path = args.checkpoint or str(model_dir / "checkpoints" / "final.pt")
    output_path = Path(args.output or model_dir / "synthetic.csv")

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Load data module for denormalization
    data_module = FinancialDataModule(
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
    data_module.setup()

    logger.info(f"Loading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = FinancialDiffusion(
        seq_len=config["data"]["seq_len"],
        input_dim=config["model"]["input_dim"],
        d_model=config["model"]["d_model"],
        n_layers=config["model"]["n_layers"],
        n_heads=config["model"]["n_heads"],
        d_ff=config["model"]["d_ff"],
        d_cond=config["model"]["condition_dim"],
        n_regimes=config["model"]["n_regimes"],
        timesteps=config["model"]["timesteps"],
        beta_schedule=config["model"]["beta_schedule"],
        beta_start=config["model"]["beta_start"],
        beta_end=config["model"]["beta_end"],
        prediction_type=config["model"]["prediction_type"],
        dropout=0.0,
    )
    # Strip topo buffers — not needed for inference, only present in ddpm_topo checkpoints
    state_dict = {
        k: v for k, v in checkpoint["model_state_dict"].items()
        if not k.startswith("topo_loss_fn.")
    }
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    conditions = {}
    if args.trend is not None:
        conditions["trend"] = args.trend
    if args.volatility is not None:
        conditions["volatility"] = args.volatility
    if args.regime is not None:
        conditions["regime"] = args.regime
    if not conditions:
        conditions = None
        logger.info("Generating unconditional samples")
    else:
        logger.info(f"Generating with conditions: {conditions}")

    logger.info(f"Generating {args.n_samples} samples...")
    with torch.no_grad():
        samples = model.generate(
            n_samples=args.n_samples,
            conditions=conditions,
            use_ddim=args.ddim,
            ddim_steps=args.ddim_steps if args.ddim else None,
            device=device,
            progress=True,
        )

    samples = samples.cpu().numpy()                  # (N, seq_len, 1) or (N, seq_len)
    samples = data_module.denormalize(samples)
    if samples.ndim == 3:
        samples = samples[:, :, 0]                   # (N, seq_len)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cols = [f"r_{i}" for i in range(samples.shape[1])]
    pd.DataFrame(samples.astype(np.float32), columns=cols).to_csv(output_path, index=False)

    logger.info(f"Saved {args.n_samples} samples to {output_path}")
    logger.info(f"  Mean daily return:      {samples.mean():.6f}")
    logger.info(f"  Std  daily return:      {samples.std():.6f}")
    logger.info(f"  Annualized return:      {samples.mean() * 252:.2%}")
    logger.info(f"  Annualized volatility:  {samples.std() * np.sqrt(252):.2%}")


if __name__ == "__main__":
    main()
