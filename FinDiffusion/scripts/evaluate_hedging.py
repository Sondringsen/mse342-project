#!/usr/bin/env python
"""Evaluate deep hedging model on real held-out data.

Metrics reported:
  - CVaR (Conditional Value at Risk) of terminal P&L
  - Mean and worst-case max drawdown of the intra-path hedging gain

The model was trained on synthetic data from FinDiffusion; evaluation is
intentionally run on real test-set paths to measure out-of-distribution
generalisation.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import FinancialDataModule
from src.hedging import DeepHedger, log_returns_to_prices, compute_pnl

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

S0 = 100.0
OTM_FACTOR = 1.05
TTL = 30


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def cvar(pnl: np.ndarray, alpha: float) -> float:
    """CVaR of losses (−P&L) at confidence level alpha."""
    losses = -pnl
    var = np.quantile(losses, alpha)
    tail = losses[losses >= var]
    return float(tail.mean())


def max_drawdown_per_path(cum_gains: np.ndarray) -> np.ndarray:
    """Maximum peak-to-trough drawdown for each path.

    Args:
        cum_gains: (N, T) cumulative hedging gains over time
    Returns:
        (N,) max drawdown per path
    """
    running_peak = np.maximum.accumulate(cum_gains, axis=1)
    return (running_peak - cum_gains).max(axis=1)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_real_test_paths(config_path: str, n_samples: int) -> np.ndarray:
    """Load real log-return windows from the test split."""
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

    ds = dm.test_dataset
    n = min(n_samples, len(ds))
    idx = np.random.choice(len(ds), n, replace=False)
    raw = np.array([ds[i].numpy() for i in idx])   # (N, seq_len, 1) or (N, seq_len)
    raw = dm.denormalize(raw)
    if raw.ndim == 3:
        raw = raw[:, :, 0]
    return raw.astype(np.float32)                  # (N, seq_len)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate deep hedging model on real data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["ddpm", "ddpm_topo", "real"],
        help="Model variant — used to resolve default hedger and output paths",
    )
    parser.add_argument(
        "--hedger",
        type=str,
        default=None,
        help="Path to hedging_model.pt (default: outputs/{model}/hedging/hedging_model.pt)",
    )
    parser.add_argument("--config",     type=str, default="configs/default.yaml")
    parser.add_argument("--n_samples",  type=int, default=500,  help="Real test paths to evaluate on")
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--cvar_alpha", type=float, default=0.95)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: outputs/{model}/hedging_eval/)",
    )
    args = parser.parse_args()

    hedger_path = args.hedger or f"outputs/{args.model}/hedging/hedging_model.pt"
    device = torch.device("cpu")
    output_dir = Path(args.output_dir or f"outputs/{args.model}/hedging_eval")
    output_dir.mkdir(parents=True, exist_ok=True)

    strike = S0 * OTM_FACTOR
    logger.info(f"Option: European call  S0={S0}  K={strike:.1f} ({(OTM_FACTOR-1):.0%} OTM)  T={TTL} days")

    # --- Load hedging model ---
    hedger = DeepHedger(input_dim=2, hidden_dim=args.hidden_dim)
    ckpt = torch.load(hedger_path, map_location=device)
    hedger.load_state_dict(ckpt["model_state_dict"])
    hedger.eval()
    logger.info(f"Loaded hedging model from {hedger_path}")

    # --- Load real test data ---
    logger.info("Loading real test data ...")
    log_returns = load_real_test_paths(args.config, args.n_samples)
    log_returns = log_returns[:, :TTL]             # first 30 days of each window
    logger.info(f"Real test paths: {log_returns.shape}")

    prices_np = log_returns_to_prices(log_returns, s0=S0)
    prices = torch.tensor(prices_np)

    # --- Run hedging model ---
    with torch.no_grad():
        pnl_t, delta_t = compute_pnl(prices, hedger, strike, TTL)

    pnl = pnl_t.numpy()
    deltas = delta_t.numpy()

    # Cumulative hedging gains over time (for drawdown)
    price_changes = prices_np[:, 1:] - prices_np[:, :-1]   # (N, TTL)
    cum_gains = np.cumsum(deltas * price_changes, axis=1)   # (N, TTL)

    # --- Compute metrics ---
    cvar_val = cvar(pnl, args.cvar_alpha)
    dd = max_drawdown_per_path(cum_gains)
    mean_dd = float(dd.mean())
    worst_dd = float(dd.max())
    mean_pnl = float(pnl.mean())
    std_pnl = float(pnl.std())
    pct_profitable = float((pnl > 0).mean())

    report_lines = [
        "=" * 54,
        "  DEEP HEDGING EVALUATION — Real Data",
        "=" * 54,
        f"  Option          European call",
        f"  Strike          {strike:.2f}  ({(OTM_FACTOR-1):.0%} OTM)",
        f"  TTL             {TTL} trading days",
        f"  Test paths      {len(pnl)}",
        "-" * 54,
        f"  Mean P&L        {mean_pnl:+.4f}",
        f"  Std  P&L        {std_pnl:.4f}",
        f"  % Profitable    {pct_profitable:.1%}",
        f"  CVaR({args.cvar_alpha:.0%})        {cvar_val:.4f}",
        f"  Mean Max DD     {mean_dd:.4f}",
        f"  Worst Max DD    {worst_dd:.4f}",
        "=" * 54,
    ]
    logger.info("\n" + "\n".join(report_lines))

    # --- Plots ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 1. P&L distribution
    axes[0].hist(pnl, bins=50, density=True, alpha=0.8, color="steelblue", edgecolor="none")
    var_val = np.quantile(-pnl, args.cvar_alpha)
    axes[0].axvline(-var_val,  color="red",     linestyle="--", linewidth=1.2,
                    label=f"VaR({args.cvar_alpha:.0%})")
    axes[0].axvline(-cvar_val, color="darkred", linestyle="--", linewidth=1.2,
                    label=f"CVaR({args.cvar_alpha:.0%}) = {cvar_val:.3f}")
    axes[0].set_xlabel("Terminal P&L")
    axes[0].set_ylabel("Density")
    axes[0].set_title("P&L Distribution (Real Data)")
    axes[0].legend(fontsize=8)

    # 2. Cumulative hedging gain paths (sample)
    n_show = min(60, len(cum_gains))
    for i in range(n_show):
        axes[1].plot(cum_gains[i], alpha=0.15, color="steelblue", linewidth=0.8)
    axes[1].axhline(0, color="k", linewidth=0.8, linestyle="--")
    axes[1].set_xlabel("Trading Day")
    axes[1].set_ylabel("Cumulative Hedging Gain")
    axes[1].set_title(f"Intra-Path Hedging Gains (n={n_show})")

    # 3. Max drawdown distribution
    axes[2].hist(dd, bins=40, density=True, alpha=0.8, color="coral", edgecolor="none")
    axes[2].axvline(mean_dd,  color="red",     linestyle="--", linewidth=1.2,
                    label=f"Mean  = {mean_dd:.3f}")
    axes[2].axvline(worst_dd, color="darkred", linestyle="--", linewidth=1.2,
                    label=f"Worst = {worst_dd:.3f}")
    axes[2].set_xlabel("Max Drawdown")
    axes[2].set_ylabel("Density")
    axes[2].set_title("Max Drawdown Distribution")
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / "hedging_evaluation.png", dpi=150)
    plt.close()
    logger.info(f"Plot saved to {output_dir}/hedging_evaluation.png")

    # --- Save JSON ---
    results = {
        "option": {
            "type": "European call",
            "s0": S0,
            "strike": strike,
            "otm_pct": OTM_FACTOR - 1,
            "ttl_days": TTL,
        },
        "metrics": {
            "mean_pnl": mean_pnl,
            "std_pnl": std_pnl,
            "pct_profitable": pct_profitable,
            f"cvar_{int(args.cvar_alpha * 100)}": cvar_val,
            "mean_max_drawdown": mean_dd,
            "worst_max_drawdown": worst_dd,
        },
        "n_paths": int(len(pnl)),
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {output_dir}/results.json")
    logger.info("Done.")


if __name__ == "__main__":
    main()
