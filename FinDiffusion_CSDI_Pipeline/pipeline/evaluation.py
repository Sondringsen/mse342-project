"""FinDiffusion-style evaluation for one-step forecasts."""

import json
import os
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mse342_matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FINDIFFUSION_ROOT = PROJECT_ROOT / "FinDiffusion"
if str(FINDIFFUSION_ROOT) not in sys.path:
    sys.path.insert(0, str(FINDIFFUSION_ROOT))

from src.evaluation import (  # noqa: E402
    compute_all_metrics,
    print_metrics_report,
    print_stylized_facts_table,
    validate_stylized_facts_per_sequence,
)
from src.evaluation.stylized_facts import compare_distributions  # noqa: E402

from .data import OneStepReturnDataset, TRADING_DAYS_PER_YEAR


def to_python(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_python(v) for v in obj]
    return obj


@torch.no_grad()
def generate_prediction_frame(
    model: torch.nn.Module,
    dataset: OneStepReturnDataset,
    batch_size: int,
    n_samples: int,
    device: torch.device,
    use_ddim: bool,
    ddim_steps: int,
    max_windows_per_asset: Optional[int],
) -> pd.DataFrame:
    model.eval()
    indices = select_eval_indices(dataset, max_windows_per_asset)
    rows = []

    for offset in range(0, len(indices), batch_size):
        batch_indices = indices[offset : offset + batch_size]
        examples = [dataset[i] for i in batch_indices]
        history = torch.stack([ex["history"] for ex in examples]).to(device)
        target_raw = torch.stack([ex["target_raw"] for ex in examples]).cpu().numpy()
        samples = model.sample(
            history,
            n_samples=n_samples,
            use_ddim=use_ddim,
            ddim_steps=ddim_steps,
            progress=False,
        )
        samples_np = samples.detach().cpu().numpy()
        samples_np = dataset.denormalize(samples_np)

        for local_idx, ex in enumerate(examples):
            asset_index = int(ex["asset_index"])
            target_index = int(ex["target_index"])
            ticker, date = dataset.metadata(asset_index, target_index)
            sample_values = samples_np[local_idx, :, 0, 0]
            row = {
                "ticker": ticker,
                "target_date": date,
                "target_index": target_index,
                "actual": float(target_raw[local_idx, 0, 0]),
                "pred_mean": float(sample_values.mean()),
                "pred_median": float(np.median(sample_values)),
                "pred_q05": float(np.quantile(sample_values, 0.05)),
                "pred_q25": float(np.quantile(sample_values, 0.25)),
                "pred_q75": float(np.quantile(sample_values, 0.75)),
                "pred_q95": float(np.quantile(sample_values, 0.95)),
            }
            for sample_idx, value in enumerate(sample_values):
                row[f"sample_{sample_idx:03d}"] = float(value)
            rows.append(row)

    return pd.DataFrame(rows)


def select_eval_indices(dataset: OneStepReturnDataset, max_windows_per_asset: Optional[int]) -> List[int]:
    by_asset = {}  # type: Dict[int, List[int]]
    for idx, (asset_idx, _start) in enumerate(dataset.samples):
        by_asset.setdefault(asset_idx, []).append(idx)

    selected = []  # type: List[int]
    for asset_indices in by_asset.values():
        asset_indices = sorted(asset_indices, key=lambda i: int(dataset[i]["target_index"]))
        if max_windows_per_asset is not None and len(asset_indices) > max_windows_per_asset:
            asset_indices = asset_indices[-max_windows_per_asset:]
        selected.extend(asset_indices)
    return selected


def paths_from_predictions(predictions: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    sample_cols = [c for c in predictions.columns if c.startswith("sample_")]
    real_paths = []
    synthetic_paths = []
    for ticker, ticker_df in predictions.groupby("ticker", sort=True):
        ticker_df = ticker_df.sort_values("target_index")
        real_paths.append(ticker_df["actual"].to_numpy(np.float32))
        for col in sample_cols:
            synthetic_paths.append(ticker_df[col].to_numpy(np.float32))
    return np.asarray(real_paths, dtype=np.float32), np.asarray(synthetic_paths, dtype=np.float32)


def forecast_metrics(predictions: pd.DataFrame) -> Dict[str, float]:
    actual = predictions["actual"].to_numpy(float)
    median = predictions["pred_median"].to_numpy(float)
    mean = predictions["pred_mean"].to_numpy(float)
    err_median = median - actual
    err_mean = mean - actual
    inside_50 = (actual >= predictions["pred_q25"]) & (actual <= predictions["pred_q75"])
    inside_90 = (actual >= predictions["pred_q05"]) & (actual <= predictions["pred_q95"])
    return {
        "median_mae": float(np.mean(np.abs(err_median))),
        "median_rmse": float(np.sqrt(np.mean(err_median**2))),
        "mean_mae": float(np.mean(np.abs(err_mean))),
        "mean_rmse": float(np.sqrt(np.mean(err_mean**2))),
        "bias": float(np.mean(err_median)),
        "coverage_50": float(np.mean(inside_50)),
        "coverage_90": float(np.mean(inside_90)),
        "avg_width_50": float(np.mean(predictions["pred_q75"] - predictions["pred_q25"])),
        "avg_width_90": float(np.mean(predictions["pred_q95"] - predictions["pred_q05"])),
    }


def evaluate_predictions(
    predictions: pd.DataFrame,
    output_dir: Path,
    model_name: str,
) -> Dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "predictions.csv", index=False)

    real_paths, synthetic_paths = paths_from_predictions(predictions)
    stylized = {
        "real": validate_stylized_facts_per_sequence(real_paths),
        "synthetic": validate_stylized_facts_per_sequence(synthetic_paths),
        "comparison": compare_distributions(real_paths, synthetic_paths),
    }
    metrics = compute_all_metrics(real_paths, synthetic_paths)
    forecasts = forecast_metrics(predictions)

    results = {
        "model": model_name,
        "forecast": forecasts,
        "metrics": metrics,
        "stylized_facts": stylized,
        "path_shapes": {
            "real": list(real_paths.shape),
            "synthetic": list(synthetic_paths.shape),
        },
    }
    (output_dir / "evaluation_results.json").write_text(json.dumps(to_python(results), indent=2) + "\n")

    metrics_report = print_metrics_report(metrics)
    sf_report = print_stylized_facts_table(stylized)
    (output_dir / "metrics_report.txt").write_text(metrics_report + "\n")
    (output_dir / "stylized_facts_report.txt").write_text(sf_report + "\n")

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    create_visualizations(real_paths, synthetic_paths, predictions, plot_dir, model_name)
    return results


def create_visualizations(
    real: np.ndarray,
    synthetic: np.ndarray,
    predictions: pd.DataFrame,
    output_dir: Path,
    model_name: str,
) -> None:
    create_distribution_plot(real, synthetic, output_dir / "distribution_comparison.png", model_name)
    create_path_plot(real, synthetic, output_dir / "sample_paths.png", model_name)
    create_generated_timeseries_plot(
        predictions, output_dir / "generated_return_timeseries.png", model_name
    )
    create_acf_plot(real, synthetic, output_dir / "autocorrelation.png", model_name)
    create_calibration_plot(predictions, output_dir / "forecast_calibration.png", model_name)


def create_distribution_plot(real: np.ndarray, synthetic: np.ndarray, path: Path, model_name: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    r_flat = real.flatten()
    s_flat = synthetic.flatten()
    axes[0].hist(r_flat, bins=100, alpha=0.7, label="Real", density=True)
    axes[0].hist(s_flat, bins=100, alpha=0.7, label=model_name, density=True)
    clip = max(
        abs(np.percentile(r_flat, 1)),
        abs(np.percentile(r_flat, 99)),
        abs(np.percentile(s_flat, 1)),
        abs(np.percentile(s_flat, 99)),
    )
    axes[0].set_xlim(-1.5 * clip, 1.5 * clip)
    axes[0].set_xlabel("Daily log return")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Return Distribution")
    axes[0].legend()

    r_sorted = np.sort(r_flat)
    s_sorted = np.sort(s_flat)
    n = min(len(r_sorted), len(s_sorted))
    step = max(1, n // 1000)
    axes[1].scatter(r_sorted[:n:step], s_sorted[:n:step], alpha=0.5, s=5)
    lims = [min(r_sorted.min(), s_sorted.min()), max(r_sorted.max(), s_sorted.max())]
    axes[1].plot(lims, lims, "r--", alpha=0.8)
    axes[1].set_xlabel("Real Quantiles")
    axes[1].set_ylabel("Synthetic Quantiles")
    axes[1].set_title("Q-Q Plot")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def create_path_plot(real: np.ndarray, synthetic: np.ndarray, path: Path, model_name: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    n_pairs = min(2, len(real), len(synthetic))
    for i in range(n_pairs):
        axes[i, 0].plot(np.exp(np.cumsum(real[i])), label="Real", alpha=0.8)
        axes[i, 0].plot(np.exp(np.cumsum(synthetic[i])), label=model_name, alpha=0.8)
        axes[i, 0].set_title(f"One-Step Rolled Path {i + 1}")
        axes[i, 0].set_xlabel("Forecast step")
        axes[i, 0].set_ylabel("Cumulative growth")
        axes[i, 0].legend()

        window = min(21, max(2, real.shape[1] // 4))
        kernel = np.ones(window) / window
        axes[i, 1].plot(
            np.convolve(np.abs(real[i]), kernel, mode="valid") * np.sqrt(TRADING_DAYS_PER_YEAR),
            label="Real",
            alpha=0.8,
        )
        axes[i, 1].plot(
            np.convolve(np.abs(synthetic[i]), kernel, mode="valid") * np.sqrt(TRADING_DAYS_PER_YEAR),
            label=model_name,
            alpha=0.8,
        )
        axes[i, 1].set_title(f"Rolling Volatility {i + 1}")
        axes[i, 1].set_xlabel("Forecast step")
        axes[i, 1].set_ylabel("Annualized vol")
        axes[i, 1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def create_generated_timeseries_plot(
    predictions: pd.DataFrame,
    path: Path,
    model_name: str,
) -> None:
    sample_cols = [col for col in predictions.columns if col.startswith("sample_")]
    if not sample_cols:
        return

    tickers = sorted(predictions["ticker"].unique())
    n_tickers = min(3, len(tickers))
    fig, axes = plt.subplots(n_tickers, 1, figsize=(12, max(3.5, 3.2 * n_tickers)), sharex=False)
    if n_tickers == 1:
        axes = [axes]

    for ax, ticker in zip(axes, tickers[:n_tickers]):
        ticker_df = predictions[predictions["ticker"] == ticker].sort_values("target_index")
        x = np.arange(len(ticker_df))
        ax.plot(x, ticker_df["actual"].to_numpy(float), color="black", linewidth=1.4, label="Real")
        for sample_col in sample_cols[: min(6, len(sample_cols))]:
            ax.plot(
                x,
                ticker_df[sample_col].to_numpy(float),
                linewidth=0.8,
                alpha=0.55,
                label=model_name if sample_col == sample_cols[0] else None,
            )
        ax.axhline(0, color="0.35", linewidth=0.6)
        ax.set_title(f"{ticker} Generated Daily Return Paths")
        ax.set_xlabel("Forecast date index")
        ax.set_ylabel("Daily log return")
        ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def create_acf_plot(real: np.ndarray, synthetic: np.ndarray, path: Path, model_name: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    max_lag = min(20, real.shape[1] - 1, synthetic.shape[1] - 1)
    lags = np.arange(1, max_lag + 1)
    r_flat = real.flatten()
    s_flat = synthetic.flatten()
    acf_real = [_corr_at_lag(r_flat, lag) for lag in lags]
    acf_syn = [_corr_at_lag(s_flat, lag) for lag in lags]
    axes[0].bar(lags - 0.2, acf_real, width=0.4, label="Real", alpha=0.7)
    axes[0].bar(lags + 0.2, acf_syn, width=0.4, label=model_name, alpha=0.7)
    axes[0].axhline(0, color="k", linewidth=0.5)
    axes[0].set_xlabel("Lag")
    axes[0].set_ylabel("ACF")
    axes[0].set_title("Autocorrelation of Returns")
    axes[0].legend()

    acf_real_sq = [_corr_at_lag(r_flat**2, lag) for lag in lags]
    acf_syn_sq = [_corr_at_lag(s_flat**2, lag) for lag in lags]
    axes[1].bar(lags - 0.2, acf_real_sq, width=0.4, label="Real", alpha=0.7)
    axes[1].bar(lags + 0.2, acf_syn_sq, width=0.4, label=model_name, alpha=0.7)
    axes[1].axhline(0, color="k", linewidth=0.5)
    axes[1].set_xlabel("Lag")
    axes[1].set_ylabel("ACF")
    axes[1].set_title("Autocorrelation of Squared Returns")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def create_calibration_plot(predictions: pd.DataFrame, path: Path, model_name: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    actual = predictions["actual"]
    median = predictions["pred_median"]
    axes[0].scatter(actual, median, alpha=0.35, s=8)
    lims = [min(actual.min(), median.min()), max(actual.max(), median.max())]
    axes[0].plot(lims, lims, "r--", alpha=0.8)
    axes[0].set_xlabel("Actual next return")
    axes[0].set_ylabel("Predicted median")
    axes[0].set_title(f"{model_name} Median Calibration")

    errors = median - actual
    axes[1].hist(errors, bins=80, density=True, alpha=0.75)
    axes[1].axvline(0, color="k", linewidth=0.8)
    axes[1].set_xlabel("Median forecast error")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Forecast Error")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _corr_at_lag(values: np.ndarray, lag: int) -> float:
    if lag >= len(values):
        return 0.0
    x = values[:-lag]
    y = values[lag:]
    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])
