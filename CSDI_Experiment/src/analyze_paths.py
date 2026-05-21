#!/usr/bin/env python3
"""Create analysis plots for CSDI walk-forward forecast paths."""

import argparse
import json
import math
import os
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mse342_matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot walk-forward CSDI point, interval, and path diagnostics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("run_dir", type=Path, help="CSDI_Experiment/outputs/... run directory.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Plot directory. Defaults to RUN_DIR/plots.",
    )
    parser.add_argument("--max-features", type=int, default=12)
    parser.add_argument("--fan-folds", type=int, default=2)
    parser.add_argument("--fan-features", type=int, default=6)
    parser.add_argument("--dpi", type=int, default=160)
    return parser.parse_args()


def savefig(path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()


def load_run(run_dir: Path) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Dict]:
    predictions_path = run_dir / "predictions.csv"
    if predictions_path.exists():
        predictions = pd.read_csv(predictions_path)
    else:
        fold_paths = sorted(run_dir.glob("fold_*/predictions.csv"))
        if not fold_paths:
            raise FileNotFoundError(f"Missing {predictions_path} and fold_*/predictions.csv")
        predictions = pd.concat((pd.read_csv(path) for path in fold_paths), ignore_index=True)
    metrics_path = run_dir / "metrics_by_fold.csv"
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
    else:
        fold_metric_paths = sorted(run_dir.glob("fold_*/metrics.json"))
        rows = [json.loads(path.read_text()) for path in fold_metric_paths]
        metrics = pd.DataFrame(rows) if rows else None
    config_path = run_dir / "run_config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    return predictions, metrics, config


def add_diagnostics(predictions: pd.DataFrame) -> pd.DataFrame:
    df = predictions.copy()
    df["error"] = df["pred_median"] - df["actual"]
    df["abs_error"] = df["error"].abs()
    df["sq_error"] = df["error"] ** 2
    df["covered_50"] = df["actual"].between(df["pred_p25"], df["pred_p75"])
    df["covered_90"] = df["actual"].between(df["pred_p05"], df["pred_p95"])
    df["width_50"] = df["pred_p75"] - df["pred_p25"]
    df["width_90"] = df["pred_p95"] - df["pred_p05"]
    df["miss_90_low"] = df["actual"] < df["pred_p05"]
    df["miss_90_high"] = df["actual"] > df["pred_p95"]
    return df


def plot_metrics_by_fold(metrics: Optional[pd.DataFrame], out_dir: Path, dpi: int) -> None:
    if metrics is None or metrics.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(metrics["fold"], metrics["mae"], marker="o", label="MAE")
    ax.plot(metrics["fold"], metrics["rmse"], marker="o", label="RMSE")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Return error")
    ax.set_title("Point Forecast Error by Fold")
    ax.grid(alpha=0.25)
    ax.legend()
    savefig(out_dir / "metrics_by_fold.png", dpi)


def plot_training_history(run_dir: Path, out_dir: Path, dpi: int) -> None:
    paths = sorted(run_dir.glob("fold_*/train_history.csv"))
    if not paths:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for path in paths:
        fold = path.parent.name.replace("fold_", "")
        history = pd.read_csv(path)
        ax.plot(history["epoch"], history["train_loss"], label=f"fold {fold} train", alpha=0.85)
        valid = history.dropna(subset=["valid_loss"])
        if not valid.empty:
            ax.scatter(valid["epoch"], valid["valid_loss"], s=24, label=f"fold {fold} valid")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Diffusion loss")
    ax.set_title("Training History")
    ax.grid(alpha=0.25)
    if len(paths) <= 6:
        ax.legend(fontsize=8)
    savefig(out_dir / "training_history.png", dpi)


def plot_error_by_horizon(df: pd.DataFrame, out_dir: Path, dpi: int) -> None:
    by_h = df.groupby("horizon").agg(
        mae=("abs_error", "mean"),
        rmse=("sq_error", lambda x: math.sqrt(float(np.mean(x)))),
        bias=("error", "mean"),
    )
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(by_h.index, by_h["mae"], marker="o", label="MAE")
    ax.plot(by_h.index, by_h["rmse"], marker="o", label="RMSE")
    ax.plot(by_h.index, by_h["bias"], marker="o", label="Bias")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel("Return error")
    ax.set_title("Error by Horizon")
    ax.grid(alpha=0.25)
    ax.legend()
    savefig(out_dir / "error_by_horizon.png", dpi)


def plot_coverage_by_horizon(df: pd.DataFrame, out_dir: Path, dpi: int) -> None:
    by_h = df.groupby("horizon").agg(
        coverage_50=("covered_50", "mean"),
        coverage_90=("covered_90", "mean"),
        width_50=("width_50", "mean"),
        width_90=("width_90", "mean"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(by_h.index, by_h["coverage_50"], marker="o", label="50% interval")
    axes[0].plot(by_h.index, by_h["coverage_90"], marker="o", label="90% interval")
    axes[0].axhline(0.50, color="gray", linestyle="--", linewidth=1)
    axes[0].axhline(0.90, color="gray", linestyle="--", linewidth=1)
    axes[0].set_ylim(0, 1)
    axes[0].set_xlabel("Forecast horizon")
    axes[0].set_ylabel("Empirical coverage")
    axes[0].set_title("Interval Coverage")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(by_h.index, by_h["width_50"], marker="o", label="50% interval")
    axes[1].plot(by_h.index, by_h["width_90"], marker="o", label="90% interval")
    axes[1].set_xlabel("Forecast horizon")
    axes[1].set_ylabel("Average interval width")
    axes[1].set_title("Interval Width")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    savefig(out_dir / "coverage_width_by_horizon.png", dpi)


def plot_feature_errors(df: pd.DataFrame, out_dir: Path, dpi: int, max_features: int) -> None:
    feature = df.groupby("feature").agg(
        mae=("abs_error", "mean"),
        rmse=("sq_error", lambda x: math.sqrt(float(np.mean(x)))),
        coverage_90=("covered_90", "mean"),
    )
    feature = feature.sort_values("mae", ascending=False).head(max_features).sort_values("mae")
    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(feature))))
    ax.barh(feature.index, feature["mae"], label="MAE")
    ax.plot(feature["rmse"], feature.index, color="#b23a48", marker="o", label="RMSE")
    ax.set_xlabel("Return error")
    ax.set_title(f"Worst {len(feature)} Features by MAE")
    ax.grid(axis="x", alpha=0.25)
    ax.legend()
    savefig(out_dir / "feature_error_ranking.png", dpi)


def plot_actual_vs_predicted(df: pd.DataFrame, out_dir: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.scatter(df["actual"], df["pred_median"], s=10, alpha=0.35)
    lo = float(min(df["actual"].min(), df["pred_median"].min()))
    hi = float(max(df["actual"].max(), df["pred_median"].max()))
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1)
    ax.set_xlabel("Actual return")
    ax.set_ylabel("Predicted median return")
    ax.set_title("Actual vs Predicted Median")
    ax.grid(alpha=0.25)
    savefig(out_dir / "actual_vs_predicted.png", dpi)


def plot_calibration(df: pd.DataFrame, out_dir: Path, dpi: int) -> None:
    quantiles = [
        ("p05", 0.05, "pred_p05"),
        ("p25", 0.25, "pred_p25"),
        ("median", 0.50, "pred_median"),
        ("p75", 0.75, "pred_p75"),
        ("p95", 0.95, "pred_p95"),
    ]
    observed = [float((df["actual"] <= df[col]).mean()) for _, _, col in quantiles]
    expected = [q for _, q, _ in quantiles]

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.plot(expected, observed, marker="o")
    ax.plot([0, 1], [0, 1], color="black", linewidth=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Nominal quantile")
    ax.set_ylabel("Empirical fraction actual <= quantile")
    ax.set_title("Predictive Quantile Calibration")
    ax.grid(alpha=0.25)
    savefig(out_dir / "quantile_calibration.png", dpi)


def plot_fan_charts(
    df: pd.DataFrame,
    out_dir: Path,
    dpi: int,
    fan_folds: int,
    fan_features: int,
) -> None:
    ranked = (
        df.groupby("feature")["abs_error"]
        .mean()
        .sort_values(ascending=False)
        .head(fan_features)
        .index
        .tolist()
    )
    folds = sorted(df["fold"].unique())[:fan_folds]
    for fold in folds:
        fold_df = df[df["fold"] == fold]
        for feature in ranked:
            item = fold_df[fold_df["feature"] == feature].sort_values("horizon")
            if item.empty:
                continue
            x = item["horizon"].to_numpy()
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.fill_between(x, item["pred_p05"], item["pred_p95"], alpha=0.18, label="90% interval")
            ax.fill_between(x, item["pred_p25"], item["pred_p75"], alpha=0.28, label="50% interval")
            ax.plot(x, item["actual"], marker="o", color="black", label="Actual")
            ax.plot(x, item["pred_median"], marker="o", label="Median path")
            ax.axhline(0, color="gray", linewidth=0.8)
            ax.set_xlabel("Forecast horizon")
            ax.set_ylabel("Daily return")
            ax.set_title(f"Fold {fold} Forecast Fan: {feature}")
            ax.grid(alpha=0.25)
            ax.legend()
            savefig(out_dir / "fan_charts" / f"fold_{int(fold):03d}_{feature}.png", dpi)


def plot_sample_cumulative_paths(run_dir: Path, out_dir: Path, dpi: int, max_features: int) -> None:
    config_path = run_dir / "run_config.json"
    if not config_path.exists():
        return
    config = json.loads(config_path.read_text())
    features = config.get("features", [])
    nsample = config.get("nsample")
    if not features or nsample is None:
        return

    for fold_dir in sorted(run_dir.glob("fold_*"))[:2]:
        path = fold_dir / f"generated_outputs_nsample{nsample}.pk"
        if not path.exists():
            continue
        with path.open("rb") as f:
            samples, target, eval_points, _observed, _time, scaler, mean_scaler = pickle.load(f)
        samples_np = samples.numpy()
        target_np = target.numpy()
        scaler_np = scaler.numpy().reshape(1, 1, 1, -1)
        mean_np = mean_scaler.numpy().reshape(1, 1, 1, -1)
        samples_np = samples_np * scaler_np + mean_np
        target_np = target_np * scaler.numpy().reshape(1, 1, -1) + mean_scaler.numpy().reshape(1, 1, -1)
        eval_np = eval_points.numpy()
        forecast_positions = np.where(eval_np[0].sum(axis=1) > 0)[0]
        if len(forecast_positions) == 0:
            continue
        fold_id = fold_dir.name.replace("fold_", "")
        for feature_index, feature in enumerate(features[:max_features]):
            paths = samples_np[0, :, :, feature_index][:, forecast_positions]
            actual = target_np[0, forecast_positions, feature_index]
            cumulative_paths = np.cumprod(1.0 + paths, axis=1) - 1.0
            cumulative_actual = np.cumprod(1.0 + actual) - 1.0
            x = np.arange(1, len(forecast_positions) + 1)
            fig, ax = plt.subplots(figsize=(8, 4.5))
            for sample_path in cumulative_paths[: min(len(cumulative_paths), 50)]:
                ax.plot(x, sample_path, color="#4c78a8", alpha=0.12, linewidth=0.8)
            ax.plot(x, np.median(cumulative_paths, axis=0), color="#f58518", linewidth=2, label="Median generated path")
            ax.plot(x, cumulative_actual, color="black", linewidth=2, label="Actual cumulative return")
            ax.axhline(0, color="gray", linewidth=0.8)
            ax.set_xlabel("Forecast horizon")
            ax.set_ylabel("Cumulative return")
            ax.set_title(f"Fold {fold_id} Generated Cumulative Paths: {feature}")
            ax.grid(alpha=0.25)
            ax.legend()
            savefig(out_dir / "cumulative_paths" / f"fold_{fold_id}_{feature}.png", dpi)


def write_summary(df: pd.DataFrame, out_dir: Path) -> None:
    summary = {
        "rows": int(len(df)),
        "folds": int(df["fold"].nunique()),
        "features": int(df["feature"].nunique()),
        "mae": float(df["abs_error"].mean()),
        "rmse": float(math.sqrt(float(df["sq_error"].mean()))),
        "bias": float(df["error"].mean()),
        "coverage_50": float(df["covered_50"].mean()),
        "coverage_90": float(df["covered_90"].mean()),
        "avg_width_50": float(df["width_50"].mean()),
        "avg_width_90": float(df["width_90"].mean()),
        "miss_90_low": float(df["miss_90_low"].mean()),
        "miss_90_high": float(df["miss_90_high"].mean()),
    }
    (out_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    out_dir = (args.output_dir or run_dir / "plots").resolve()
    predictions, metrics, _config = load_run(run_dir)
    df = add_diagnostics(predictions)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_summary(df, out_dir)
    plot_metrics_by_fold(metrics, out_dir, args.dpi)
    plot_training_history(run_dir, out_dir, args.dpi)
    plot_error_by_horizon(df, out_dir, args.dpi)
    plot_coverage_by_horizon(df, out_dir, args.dpi)
    plot_feature_errors(df, out_dir, args.dpi, args.max_features)
    plot_actual_vs_predicted(df, out_dir, args.dpi)
    plot_calibration(df, out_dir, args.dpi)
    plot_fan_charts(df, out_dir, args.dpi, args.fan_folds, args.fan_features)
    plot_sample_cumulative_paths(run_dir, out_dir, args.dpi, min(args.max_features, args.fan_features))

    print(f"Wrote analysis plots to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
