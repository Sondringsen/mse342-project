#!/usr/bin/env python
"""Single-condition evaluation script for FinDiffusion.

Two modes:
  --checkpoint PATH  Generate data for a single (trend, vol) condition, save to CSV, evaluate.
  --data PATH        Load a previously saved synthetic.csv, skip generation, evaluate.

In both modes the script evaluates stylized facts, computes metrics, and produces plots
including a multi-path chart of the generated trajectories.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import FinancialDiffusion
from src.data import FinancialDataModule
from src.evaluation import compute_all_metrics, print_metrics_report, print_stylized_facts_table
from src.evaluation.stylized_facts import compare_distributions, validate_stylized_facts_per_sequence

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model / data loading
# ---------------------------------------------------------------------------

def load_model_and_data(checkpoint_path: str, config_path: str, device: torch.device):
    with open(config_path) as f:
        config = yaml.safe_load(f)

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
        dropout=config["model"]["dropout"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

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
    _restore_norm_stats(data_module, Path(checkpoint_path).parent)

    return model, data_module, config


@torch.no_grad()
def generate_samples(
    model: FinancialDiffusion,
    n_samples: int,
    trend: float | None,
    vol: float | None,
    regime: str | None,
    device: torch.device,
    use_ddim: bool,
) -> np.ndarray:
    if trend is None and vol is None and regime is None:
        logger.info(f"Generating {n_samples} samples unconditionally…")
        conditions = None
    else:
        logger.info(f"Generating {n_samples} samples (trend={trend:.2%}, vol={vol:.2%}, regime={regime})…")
        conditions = {"trend": trend, "volatility": vol, "regime": regime}
    samples = model.generate(
        n_samples=n_samples,
        conditions=conditions,
        use_ddim=use_ddim,
        ddim_steps=50 if use_ddim else None,
        device=device,
        progress=True,
    )
    return samples.cpu().numpy()


def get_real_samples(data_module: FinancialDataModule, n_samples: int) -> np.ndarray:
    test_dataset = data_module.test_dataset
    indices = np.random.choice(len(test_dataset), min(n_samples, len(test_dataset)), replace=False)
    samples = np.array([test_dataset[i].numpy() for i in indices])
    return data_module.denormalize(samples)


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------

def _squeeze(arr: np.ndarray) -> np.ndarray:
    """(N, T, 1) → (N, T) or (N, T) passthrough."""
    if arr.ndim == 3 and arr.shape[2] == 1:
        return arr[:, :, 0]
    return arr.reshape(arr.shape[0], -1)


def _restore_norm_stats(data_module: FinancialDataModule, checkpoint_dir: Path):
    data_state_path = checkpoint_dir / "data_state.pt"
    if data_state_path.exists():
        state = torch.load(data_state_path, map_location="cpu")
        data_module.load_state_dict(state["data_module"])
        logger.info(
            f"Restored normalization stats from {data_state_path} "
            f"(mean={data_module.mean:.6f}, std={data_module.std:.6f})"
        )
    else:
        logger.warning(
            f"data_state.pt not found in {checkpoint_dir} — "
            "using freshly computed normalization stats; scale may differ from training"
        )


def setup_data_module(config_path: str, checkpoint_dir: Path = None) -> FinancialDataModule:
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
    if checkpoint_dir is not None:
        _restore_norm_stats(dm, checkpoint_dir)
    return dm


def save_data(synthetic: np.ndarray, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    syn_2d = _squeeze(synthetic)
    cols = [f"r_{i}" for i in range(syn_2d.shape[1])]
    pd.DataFrame(syn_2d, columns=cols).to_csv(output_dir / "synthetic.csv", index=False)
    logger.info(f"Saved synthetic.csv ({len(syn_2d)} rows) to {output_dir}")


def load_synthetic(data_path: str) -> np.ndarray:
    syn_2d = pd.read_csv(data_path).values.astype(np.float32)
    synthetic = syn_2d[:, :, np.newaxis]
    logger.info(f"Loaded {len(synthetic)} synthetic samples from {data_path}")
    return synthetic


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_stylized_facts(real: np.ndarray, synthetic: np.ndarray) -> dict:
    real_results = validate_stylized_facts_per_sequence(real)
    syn_results = validate_stylized_facts_per_sequence(synthetic)

    logger.info("Real stylized facts:")
    for k, v in real_results.items():
        if k != "summary":
            logger.info(f"  {k}: {'PASS' if v['passed'] else 'FAIL'}")

    logger.info("Synthetic stylized facts:")
    for k, v in syn_results.items():
        if k != "summary":
            logger.info(f"  {k}: {'PASS' if v['passed'] else 'FAIL'}")

    return {
        "real": real_results,
        "synthetic": syn_results,
        "comparison": compare_distributions(real, synthetic),
    }


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

def create_visualizations(
    real: np.ndarray,
    synthetic: np.ndarray,
    output_dir: Path,
    n_paths: int = 7,
    trend: float = None,
    vol: float = None,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    R = _squeeze(real)
    S = _squeeze(synthetic)

    cond_label = f" (trend={trend:.1%}, vol={vol:.0%})" if trend is not None and vol is not None else ""

    # 1. Return distribution + Q-Q
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(R.flatten(), bins=100, alpha=0.7, label="Real", density=True)
    axes[0].hist(S.flatten(), bins=100, alpha=0.7, label="Synthetic", density=True)
    axes[0].set_xlabel("Returns")
    axes[0].set_ylabel("Density")
    axes[0].set_title(f"Return Distribution{cond_label}")
    axes[0].legend()
    clip = max(abs(np.percentile(R.flatten(), 1)), abs(np.percentile(R.flatten(), 99)),
               abs(np.percentile(S.flatten(), 1)), abs(np.percentile(S.flatten(), 99))) * 1.5
    axes[0].set_xlim(-clip, clip)

    real_sorted = np.sort(R.flatten())
    syn_sorted = np.sort(S.flatten())
    n = min(len(real_sorted), len(syn_sorted))
    step = max(1, n // 1000)
    axes[1].scatter(real_sorted[:n:step], syn_sorted[:n:step], alpha=0.5, s=5)
    lims = [min(real_sorted.min(), syn_sorted.min()), max(real_sorted.max(), syn_sorted.max())]
    axes[1].plot(lims, lims, "r--", alpha=0.8)
    axes[1].set_xlabel("Real Quantiles")
    axes[1].set_ylabel("Synthetic Quantiles")
    axes[1].set_title("Q-Q Plot")

    plt.tight_layout()
    plt.savefig(output_dir / "distribution_comparison.png", dpi=150)
    plt.close()

    # 2. Two sample paths: cumulative return + rolling vol
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for i in range(2):
        r1d, s1d = R[i], S[i]
        axes[i, 0].plot(np.cumprod(1 + r1d), label="Real", alpha=0.8)
        axes[i, 0].plot(np.cumprod(1 + s1d), label="Synthetic", alpha=0.8)
        axes[i, 0].set_title(f"Sample Path {i + 1}" + (cond_label if i == 0 else ""))
        axes[i, 0].set_xlabel("Time")
        axes[i, 0].set_ylabel("Cumulative Return")
        axes[i, 0].legend()

        window = 21
        ker = np.ones(window) / window
        axes[i, 1].plot(np.convolve(np.abs(r1d), ker, mode="valid") * np.sqrt(252), label="Real", alpha=0.8)
        axes[i, 1].plot(np.convolve(np.abs(s1d), ker, mode="valid") * np.sqrt(252), label="Synthetic", alpha=0.8)
        axes[i, 1].set_title(f"Rolling Volatility {i + 1}")
        axes[i, 1].set_xlabel("Time")
        axes[i, 1].set_ylabel("Annualized Vol")
        axes[i, 1].legend()

    plt.tight_layout()
    plt.savefig(output_dir / "sample_paths.png", dpi=150)
    plt.close()

    # 3. Autocorrelation of returns + squared returns
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    max_lag = 20
    lags = np.arange(1, max_lag + 1)

    r_flat, s_flat = R.flatten(), S.flatten()
    acf_r = [np.corrcoef(r_flat[:-lag], r_flat[lag:])[0, 1] for lag in lags]
    acf_s = [np.corrcoef(s_flat[:-lag], s_flat[lag:])[0, 1] for lag in lags]

    axes[0].bar(lags - 0.2, acf_r, width=0.4, label="Real", alpha=0.7)
    axes[0].bar(lags + 0.2, acf_s, width=0.4, label="Synthetic", alpha=0.7)
    axes[0].axhline(0, color="k", linewidth=0.5)
    axes[0].set_xlabel("Lag")
    axes[0].set_ylabel("ACF")
    axes[0].set_title("Autocorrelation of Returns")
    axes[0].legend()

    acf_r_sq = [np.corrcoef(r_flat[:-lag] ** 2, r_flat[lag:] ** 2)[0, 1] for lag in lags]
    acf_s_sq = [np.corrcoef(s_flat[:-lag] ** 2, s_flat[lag:] ** 2)[0, 1] for lag in lags]

    axes[1].bar(lags - 0.2, acf_r_sq, width=0.4, label="Real", alpha=0.7)
    axes[1].bar(lags + 0.2, acf_s_sq, width=0.4, label="Synthetic", alpha=0.7)
    axes[1].axhline(0, color="k", linewidth=0.5)
    axes[1].set_xlabel("Lag")
    axes[1].set_ylabel("ACF")
    axes[1].set_title("Autocorrelation of Squared Returns (Vol Clustering)")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_dir / "autocorrelation.png", dpi=150)
    plt.close()

    # 4. Multiple generated paths
    n_plot = min(n_paths, len(S))
    fig, ax = plt.subplots(figsize=(12, 5))

    for i in range(n_plot):
        ax.plot(np.cumprod(1 + S[i]), alpha=0.65, linewidth=1.0)

    ax.axhline(1.0, color="k", linestyle="--", linewidth=0.8, alpha=0.4)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Cumulative Return (start = 1)")
    ax.set_title(f"{n_plot} Generated Paths{cond_label}")

    plt.tight_layout()
    plt.savefig(output_dir / "generated_paths.png", dpi=150)
    plt.close()

    logger.info(f"Saved 4 plots to {output_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate FinDiffusion for a single (trend, vol) condition",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--checkpoint", type=str, metavar="PATH",
                      help="Checkpoint to load and generate from")
    mode.add_argument("--data", type=str, metavar="PATH",
                      help="Path to synthetic.csv (skips generation)")

    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Config file — used to load real test data in both modes")
    parser.add_argument("--trend", type=float, default=None,
                        help="Target annualized trend (e.g. 0.1 = 10%%). Omit for unconditional generation.")
    parser.add_argument("--vol", type=float, default=None,
                        help="Target annualized volatility (e.g. 0.2 = 20%%). Omit for unconditional generation.")
    parser.add_argument("--regime", type=str, default=None,
                        choices=["bull", "bear", "sideways"],
                        help="Market regime. Omit for unconditional generation.")
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--n_paths", type=int, default=7,
                        help="How many paths to show in generated_paths.png")
    parser.add_argument("--output_dir", type=str, default="outputs/evaluation_single")
    parser.add_argument("--ddim", action="store_true", help="Use DDIM for faster sampling")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Generation or CSV load ----
    if args.checkpoint:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Device: {device}")

        model, data_module, _ = load_model_and_data(args.checkpoint, args.config, device)

        synthetic = generate_samples(
            model, args.n_samples, args.trend, args.vol, args.regime, device, args.ddim
        )
        synthetic = data_module.denormalize(synthetic)
        logger.info(f"Synthetic samples: {len(synthetic)}")

        save_data(synthetic, output_dir)

    else:
        synthetic = load_synthetic(args.data)
        data_dir = Path(args.data).parent
        data_module = setup_data_module(args.config, checkpoint_dir=data_dir)

    real = get_real_samples(data_module, len(synthetic))
    logger.info(f"Real samples: {len(real)}")
    r_flat = _squeeze(real).flatten()
    logger.info(f"  Mean daily return (real): {r_flat.mean():.6f}")
    logger.info(f"  Std  daily return (real): {r_flat.std():.6f}")
    logger.info(f"  Annualized return  (real): {r_flat.mean() * 252:.2%}")
    logger.info(f"  Annualized vol     (real): {r_flat.std() * np.sqrt(252):.2%}")
    s_flat = _squeeze(synthetic).flatten()
    logger.info(f"  Mean daily return (synthetic): {s_flat.mean():.6f}")
    logger.info(f"  Std  daily return (synthetic): {s_flat.std():.6f}")
    logger.info(f"  Annualized return  (synthetic): {s_flat.mean() * 252:.2%}")
    logger.info(f"  Annualized vol     (synthetic): {s_flat.std() * np.sqrt(252):.2%}")

    # ---- Stylized facts ----
    sf_results = evaluate_stylized_facts(real, synthetic)

    # ---- Stylized facts table ----
    logger.info("\n" + print_stylized_facts_table(sf_results))

    # ---- Metrics ----
    metrics = compute_all_metrics(real, synthetic)
    logger.info("\n" + print_metrics_report(metrics))

    # ---- Plots ----
    trend_for_label = args.trend
    vol_for_label = args.vol
    create_visualizations(real, synthetic, output_dir, args.n_paths, trend_for_label, vol_for_label)

    # ---- Save JSON ----
    all_results = {
        "stylized_facts": sf_results,
        "metrics": metrics,
        "config": {
            "trend": args.trend,
            "vol": args.vol,
            "regime": args.regime,
            "n_samples": args.n_samples,
            "source": args.checkpoint or args.data,
        },
    }

    def _to_python(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, dict):
            return {k: _to_python(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_to_python(v) for v in obj]
        return obj

    with open(output_dir / "evaluation_results.json", "w") as f:
        json.dump(_to_python(all_results), f, indent=2)

    logger.info(f"Results saved to {output_dir}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
