#!/usr/bin/env python3
"""Compare fixed-split vanilla and topology-regularized CSDI runs."""

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mse342_matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine forecasting and topology diagnostics across CSDI runs."
    )
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--labels", nargs="+", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--max-path-samples",
        type=int,
        default=50,
        help="Maximum generated samples per run to include in path CSVs/plots.",
    )
    return parser.parse_args()


def load_json(path: Path, default):
    if path.exists():
        return json.loads(path.read_text())
    return default


def run_label(run_dir: Path, fallback: str) -> str:
    config = load_json(run_dir / "run_config.json", {})
    return str(config.get("variant_name") or fallback)


def read_run(run_dir: Path, label: str) -> pd.DataFrame:
    metrics_path = run_dir / "metrics_by_horizon.csv"
    if not metrics_path.exists():
        raise FileNotFoundError("Missing %s" % metrics_path)
    metrics = pd.read_csv(metrics_path)
    metrics["variant"] = label
    metrics["run_dir"] = str(run_dir)

    topology_summary = load_json(run_dir / "plots" / "topology_summary.json", [])
    if topology_summary:
        topology = pd.DataFrame(topology_summary)
        metrics = metrics.merge(topology, on="horizon_days", how="left")

    analysis = load_json(run_dir / "plots" / "analysis_summary.json", {})
    for key in ["coverage_50", "coverage_90", "avg_width_50", "avg_width_90", "bias"]:
        if key in analysis:
            metrics["overall_" + key] = analysis[key]
    return metrics


def read_curves(run_dir: Path, label: str) -> pd.DataFrame:
    curves_path = run_dir / "plots" / "topology_curves.csv"
    if not curves_path.exists():
        return pd.DataFrame()
    curves = pd.read_csv(curves_path)
    curves["variant"] = label
    curves["run_dir"] = str(run_dir)
    return curves


def first_generated_output(horizon_dir: Path) -> Optional[Path]:
    paths = sorted(horizon_dir.glob("generated_outputs_nsample*.pk"))
    return paths[0] if paths else None


def horizon_days_from_dir(horizon_dir: Path) -> int:
    return int(horizon_dir.name.replace("horizon_", ""))


def tensor_to_numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def resolve_data_path(config: dict) -> Optional[Path]:
    raw_path = config.get("data")
    if not raw_path:
        return None
    data_path = Path(raw_path)
    if data_path.exists():
        return data_path

    parts = data_path.parts
    if "data" in parts:
        data_index = parts.index("data")
        candidate = PROJECT_ROOT.joinpath(*parts[data_index:])
        if candidate.exists():
            return candidate

    candidate = PROJECT_ROOT / "data" / "processed" / data_path.name
    if candidate.exists():
        return candidate
    return None


def load_index_dates(config: dict, horizon_length: int) -> List[str]:
    """Return day-0 origin date plus one date per forecast step when available."""
    train_end_date = str(config.get("train_end_date", ""))
    dates = [train_end_date]
    data_path = resolve_data_path(config)
    train_end_index = config.get("train_end_index")
    if data_path is None or train_end_index is None:
        dates.extend([""] * horizon_length)
        return dates

    date_column = str(config.get("date_column", "date"))
    raw = pd.read_csv(data_path, usecols=[date_column])
    raw_dates = pd.to_datetime(raw[date_column]).dt.date.astype(str).tolist()
    start = int(train_end_index)
    for offset in range(horizon_length):
        index = start + offset
        dates.append(raw_dates[index] if index < len(raw_dates) else "")
    return dates


def model_to_simple_returns(values: np.ndarray, config: dict) -> np.ndarray:
    transform = str(config.get("return_transform", "simple"))
    if transform == "simple":
        return values
    if transform == "log":
        return np.expm1(np.clip(values, -50.0, 50.0))
    raise ValueError("Unknown return transform for %s: %s" % (config.get("data", "run"), transform))


def load_generated_index_paths(
    run_dir: Path,
    label: str,
    max_samples: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return market-level sample paths and feature-level median paths.

    Values are index levels starting at 100. The market index is an equal-weighted
    average of the selected feature returns.
    """
    config = load_json(run_dir / "run_config.json", {})
    features = config.get("features", [])
    market_rows = []
    feature_rows = []

    for horizon_dir in sorted(run_dir.glob("horizon_*")):
        output_path = first_generated_output(horizon_dir)
        if output_path is None:
            continue
        horizon_days = horizon_days_from_dir(horizon_dir)
        horizon_years = horizon_days / 252.0
        with output_path.open("rb") as f:
            samples, target, eval_points, _observed, _time, scaler, mean_scaler = pickle.load(f)

        samples_np = tensor_to_numpy(samples)
        target_np = tensor_to_numpy(target)
        scaler_np = tensor_to_numpy(scaler)
        mean_np = tensor_to_numpy(mean_scaler)
        samples_np = samples_np * scaler_np.reshape(1, 1, 1, -1) + mean_np.reshape(1, 1, 1, -1)
        target_np = target_np * scaler_np.reshape(1, 1, -1) + mean_np.reshape(1, 1, -1)
        eval_np = tensor_to_numpy(eval_points)
        forecast_positions = np.where(eval_np[0].sum(axis=1) > 0)[0]
        if len(forecast_positions) == 0:
            continue

        sample_count = min(int(max_samples), samples_np.shape[1])
        selected_samples = samples_np[0, :sample_count, :, :]
        generated_returns = model_to_simple_returns(
            selected_samples[:, forecast_positions, :],
            config,
        )
        actual_returns = model_to_simple_returns(target_np[0, forecast_positions, :], config)
        if generated_returns.shape != (sample_count, len(forecast_positions), samples_np.shape[-1]):
            raise ValueError(
                "Unexpected generated path shape for %s: got %s, expected (%d, %d, %d)"
                % (
                    output_path,
                    generated_returns.shape,
                    sample_count,
                    len(forecast_positions),
                    samples_np.shape[-1],
                )
            )
        if actual_returns.shape != (len(forecast_positions), samples_np.shape[-1]):
            raise ValueError(
                "Unexpected actual path shape for %s: got %s, expected (%d, %d)"
                % (
                    output_path,
                    actual_returns.shape,
                    len(forecast_positions),
                    samples_np.shape[-1],
                )
            )

        generated_market_returns = generated_returns.mean(axis=2)
        actual_market_returns = actual_returns.mean(axis=1)
        generated_market_index = 100.0 * np.cumprod(1.0 + generated_market_returns, axis=1)
        actual_market_index = 100.0 * np.cumprod(1.0 + actual_market_returns)

        days = np.arange(0, len(forecast_positions) + 1)
        dates = load_index_dates(config, len(forecast_positions))
        for sample_id, sample_index in enumerate(generated_market_index):
            values = np.concatenate([[100.0], sample_index])
            for day, date, value in zip(days, dates, values):
                market_rows.append(
                    {
                        "variant": label,
                        "horizon_days": horizon_days,
                        "horizon_years": horizon_years,
                        "path_type": "generated",
                        "sample_id": sample_id,
                        "day": int(day),
                        "date": date,
                        "index_level": float(value),
                    }
                )
        actual_values = np.concatenate([[100.0], actual_market_index])
        for day, date, value in zip(days, dates, actual_values):
            market_rows.append(
                {
                    "variant": label,
                    "horizon_days": horizon_days,
                    "horizon_years": horizon_years,
                    "path_type": "actual",
                    "sample_id": -1,
                    "day": int(day),
                    "date": date,
                    "index_level": float(value),
                }
            )

        if features:
            generated_feature_index = 100.0 * np.cumprod(1.0 + generated_returns, axis=1)
            generated_feature_median = np.median(generated_feature_index, axis=0)
            actual_feature_index = 100.0 * np.cumprod(1.0 + actual_returns, axis=0)
            for feature_index, feature in enumerate(features):
                gen_values = np.concatenate([[100.0], generated_feature_median[:, feature_index]])
                act_values = np.concatenate([[100.0], actual_feature_index[:, feature_index]])
                for day, date, generated_value, actual_value in zip(days, dates, gen_values, act_values):
                    feature_rows.append(
                        {
                            "variant": label,
                            "horizon_days": horizon_days,
                            "horizon_years": horizon_years,
                            "feature": feature,
                            "day": int(day),
                            "date": date,
                            "generated_median_index": float(generated_value),
                            "actual_index": float(actual_value),
                        }
                    )

    return pd.DataFrame(market_rows), pd.DataFrame(feature_rows)


def make_aggregate(comparison: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        "mae",
        "rmse",
        "median_topology_distance_to_real",
        "mean_topology_distance_to_real",
        "median_h0_total_persistence",
        "median_beta1_proxy_area",
        "median_recurrence_area",
        "median_lowfreq_power_ratio",
    ]
    available = [col for col in numeric_cols if col in comparison.columns]
    return comparison.groupby("variant", as_index=False)[available].mean()


def savefig(path: Path, dpi: int = 160) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()


def grouped_by_variant_horizon(comparison: pd.DataFrame) -> pd.DataFrame:
    numeric = comparison.select_dtypes(include=[np.number]).columns.tolist()
    group_cols = ["variant", "horizon_days"]
    keep = group_cols + [col for col in numeric if col not in group_cols]
    return comparison[keep].groupby(group_cols, as_index=False).mean()


def make_delta_table(grouped: pd.DataFrame) -> pd.DataFrame:
    if "vanilla" not in set(grouped["variant"]) or "topoloss" not in set(grouped["variant"]):
        return pd.DataFrame()
    metrics = [
        "mae",
        "rmse",
        "overall_coverage_50",
        "overall_coverage_90",
        "overall_avg_width_50",
        "overall_avg_width_90",
        "median_topology_distance_to_real",
        "mean_topology_distance_to_real",
        "median_h0_total_persistence",
        "median_beta1_proxy_area",
        "median_recurrence_area",
        "median_lowfreq_power_ratio",
    ]
    available = [col for col in metrics if col in grouped.columns]
    rows = []
    for horizon, item in grouped.groupby("horizon_days"):
        vanilla = item[item["variant"] == "vanilla"]
        topoloss = item[item["variant"] == "topoloss"]
        if vanilla.empty or topoloss.empty:
            continue
        row = {"horizon_days": int(horizon), "horizon_years": float(horizon) / 252.0}
        for metric in available:
            vanilla_value = float(vanilla.iloc[0][metric])
            topoloss_value = float(topoloss.iloc[0][metric])
            row["vanilla_" + metric] = vanilla_value
            row["topoloss_" + metric] = topoloss_value
            row["delta_" + metric] = topoloss_value - vanilla_value
            denom = abs(vanilla_value) if abs(vanilla_value) > 1e-12 else 1.0
            row["pct_delta_" + metric] = 100.0 * (topoloss_value - vanilla_value) / denom
        rows.append(row)
    return pd.DataFrame(rows)


def plot_metric_panel(
    ax,
    grouped: pd.DataFrame,
    column: str,
    title: str,
    ylabel: str,
    target: Optional[float] = None,
) -> None:
    if column not in grouped.columns:
        ax.set_visible(False)
        return
    for variant, item in grouped.groupby("variant"):
        item = item.sort_values("horizon_days")
        ax.plot(
            item["horizon_days"] / 252.0,
            item[column],
            marker="o",
            linewidth=2,
            label=variant,
        )
    if target is not None:
        ax.axhline(target, color="gray", linestyle="--", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Scenario horizon (years)")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)


def plot_dashboard(grouped: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(13, 12))
    panels = [
        ("mae", "MAE", "Return error", None),
        ("rmse", "RMSE", "Return error", None),
        ("overall_coverage_90", "90% Interval Coverage", "Coverage", 0.90),
        ("overall_avg_width_90", "90% Interval Width", "Width", None),
        (
            "median_topology_distance_to_real",
            "Topology Distance to Real Holdout",
            "Normalized distance",
            None,
        ),
        ("median_beta1_proxy_area", "Betti-1 Proxy Area", "Area", None),
    ]
    for ax, (column, title, ylabel, target) in zip(axes.flat, panels):
        plot_metric_panel(ax, grouped, column, title, ylabel, target)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=max(1, len(labels)))
    fig.suptitle("Vanilla vs Topology-Regularized CSDI", y=1.02, fontsize=14)
    savefig(output_dir / "comparison_dashboard.png")


def plot_topology_dashboard(grouped: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    panels = [
        ("median_h0_total_persistence", "H0 Total Persistence", "Persistence"),
        ("median_recurrence_area", "Recurrence Area", "Area"),
        ("median_lowfreq_power_ratio", "Low-Frequency Power Ratio", "Ratio"),
        ("mean_topology_distance_to_real", "Mean Topology Distance", "Normalized distance"),
    ]
    for ax, (column, title, ylabel) in zip(axes.flat, panels):
        plot_metric_panel(ax, grouped, column, title, ylabel)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=max(1, len(labels)))
    fig.suptitle("Topology Diagnostics by Horizon", y=1.02, fontsize=14)
    savefig(output_dir / "topology_dashboard.png")


def plot_delta_bars(delta: pd.DataFrame, output_dir: Path) -> None:
    if delta.empty:
        return
    metrics = [
        ("pct_delta_mae", "MAE % Change"),
        ("pct_delta_rmse", "RMSE % Change"),
        ("pct_delta_overall_coverage_90", "90% Coverage % Change"),
        ("pct_delta_median_topology_distance_to_real", "Topology Distance % Change"),
    ]
    available = [(col, title) for col, title in metrics if col in delta.columns]
    if not available:
        return
    fig, axes = plt.subplots(len(available), 1, figsize=(10, 2.8 * len(available)))
    if len(available) == 1:
        axes = [axes]
    x = delta["horizon_days"] / 252.0
    for ax, (column, title) in zip(axes, available):
        ax.bar(x.astype(str), delta[column])
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title("Topoloss minus Vanilla: " + title)
        ax.set_xlabel("Scenario horizon (years)")
        ax.set_ylabel("% change")
        ax.grid(axis="y", alpha=0.25)
    savefig(output_dir / "topoloss_minus_vanilla_deltas.png")


def summarize_generated_curves(curves: pd.DataFrame) -> pd.DataFrame:
    generated = curves[curves["path_type"] == "generated"]
    if generated.empty:
        return pd.DataFrame()
    grouped = generated.groupby(
        ["variant", "horizon_days", "normalized_threshold"], as_index=False
    ).agg(
        beta1_proxy_median=("beta1_proxy", "median"),
        beta1_proxy_q10=("beta1_proxy", lambda x: float(x.quantile(0.10))),
        beta1_proxy_q90=("beta1_proxy", lambda x: float(x.quantile(0.90))),
        recurrence_rate_median=("recurrence_rate", "median"),
        recurrence_rate_q10=("recurrence_rate", lambda x: float(x.quantile(0.10))),
        recurrence_rate_q90=("recurrence_rate", lambda x: float(x.quantile(0.90))),
    )
    return grouped


def plot_curve_overlay(curves: pd.DataFrame, output_dir: Path) -> None:
    if curves.empty:
        return
    summary = summarize_generated_curves(curves)
    if summary.empty:
        return
    horizons = sorted(summary["horizon_days"].unique())
    fig, axes = plt.subplots(len(horizons), 2, figsize=(13, 4 * len(horizons)))
    if len(horizons) == 1:
        axes = np.asarray([axes])
    colors = {"vanilla": "#4c78a8", "topoloss": "#f58518"}
    for row, horizon in enumerate(horizons):
        for col, metric in enumerate(["beta1_proxy", "recurrence_rate"]):
            ax = axes[row, col]
            for variant, item in summary[summary["horizon_days"] == horizon].groupby("variant"):
                item = item.sort_values("normalized_threshold")
                color = colors.get(variant, None)
                x = item["normalized_threshold"].values
                median = item[metric + "_median"].values
                q10 = item[metric + "_q10"].values
                q90 = item[metric + "_q90"].values
                ax.plot(x, median, marker=None, linewidth=2, label=variant, color=color)
                ax.fill_between(x, q10, q90, alpha=0.14, color=color)

            real = curves[
                (curves["horizon_days"] == horizon)
                & (curves["path_type"] == "real")
            ]
            if not real.empty:
                # The real curve is identical across variants for a horizon, so draw one.
                real = real.sort_values("normalized_threshold").drop_duplicates(
                    subset=["normalized_threshold"]
                )
                ax.plot(
                    real["normalized_threshold"],
                    real[metric],
                    color="black",
                    linewidth=2,
                    linestyle="--",
                    label="real",
                )
            ax.set_title("%dy %s" % (int(horizon / 252), metric.replace("_", " ")))
            ax.set_xlabel("Distance threshold / median pairwise distance")
            ax.set_ylabel(metric.replace("_", " "))
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
    savefig(output_dir / "topology_curve_overlay.png")


def ordered_variants(values: List[str]) -> List[str]:
    order = {"vanilla": 0, "topoloss": 1}
    return sorted(values, key=lambda value: (order.get(value, 99), value))


def plot_generated_market_paths(paths: pd.DataFrame, output_dir: Path) -> None:
    if paths.empty:
        return
    horizons = sorted(paths["horizon_days"].unique())
    variants = ordered_variants(paths["variant"].unique().tolist())
    fig, axes = plt.subplots(
        len(horizons),
        len(variants),
        figsize=(6.5 * len(variants), 3.8 * len(horizons)),
        squeeze=False,
    )
    for row, horizon in enumerate(horizons):
        for col, variant in enumerate(variants):
            ax = axes[row, col]
            item = paths[
                (paths["horizon_days"] == horizon)
                & (paths["variant"] == variant)
            ]
            generated = item[item["path_type"] == "generated"]
            actual = item[item["path_type"] == "actual"]
            if generated.empty:
                ax.set_visible(False)
                continue
            for _sample_id, sample in generated.groupby("sample_id"):
                ax.plot(
                    sample["day"] / 252.0,
                    sample["index_level"],
                    color="#4c78a8",
                    alpha=0.12,
                    linewidth=0.8,
                )
            median = generated.groupby("day", as_index=False)["index_level"].median()
            ax.plot(
                median["day"] / 252.0,
                median["index_level"],
                color="#f58518",
                linewidth=2.2,
                label="generated median",
            )
            if not actual.empty:
                actual = actual.sort_values("day").drop_duplicates(subset=["day"])
                ax.plot(
                    actual["day"] / 252.0,
                    actual["index_level"],
                    color="black",
                    linewidth=2.0,
                    label="actual holdout",
                )
            ax.axhline(100.0, color="gray", linewidth=0.8)
            ax.set_title("%s, %d-year generated market index" % (variant, int(horizon / 252)))
            ax.set_xlabel("Years into generated scenario")
            ax.set_ylabel("Index level, start = 100")
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
    fig.suptitle("Generated Equal-Weighted Market Paths Starting at 100", y=1.01, fontsize=14)
    savefig(output_dir / "generated_market_index_paths.png")


def plot_generated_market_medians(paths: pd.DataFrame, output_dir: Path) -> None:
    if paths.empty:
        return
    horizons = sorted(paths["horizon_days"].unique())
    variants = ordered_variants(paths["variant"].unique().tolist())
    colors = {"vanilla": "#4c78a8", "topoloss": "#f58518"}
    fig, axes = plt.subplots(len(horizons), 1, figsize=(10, 3.6 * len(horizons)), squeeze=False)
    for row, horizon in enumerate(horizons):
        ax = axes[row, 0]
        horizon_item = paths[paths["horizon_days"] == horizon]
        for variant in variants:
            generated = horizon_item[
                (horizon_item["variant"] == variant)
                & (horizon_item["path_type"] == "generated")
            ]
            if generated.empty:
                continue
            median = generated.groupby("day", as_index=False)["index_level"].median()
            ax.plot(
                median["day"] / 252.0,
                median["index_level"],
                linewidth=2.2,
                color=colors.get(variant),
                label="%s generated median" % variant,
            )
        actual = horizon_item[horizon_item["path_type"] == "actual"]
        if not actual.empty:
            actual = actual.sort_values("day").drop_duplicates(subset=["day"])
            ax.plot(
                actual["day"] / 252.0,
                actual["index_level"],
                color="black",
                linewidth=2.0,
                linestyle="--",
                label="actual holdout",
            )
        ax.axhline(100.0, color="gray", linewidth=0.8)
        ax.set_title("%d-year market index medians" % int(horizon / 252))
        ax.set_xlabel("Years into generated scenario")
        ax.set_ylabel("Index level, start = 100")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("Median Generated Market Index vs Actual Holdout", y=1.01, fontsize=14)
    savefig(output_dir / "generated_market_index_medians.png")


def plot_feature_index_medians(feature_paths: pd.DataFrame, output_dir: Path) -> None:
    if feature_paths.empty:
        return
    horizons = sorted(feature_paths["horizon_days"].unique())
    variants = ordered_variants(feature_paths["variant"].unique().tolist())
    fig, axes = plt.subplots(
        len(horizons),
        len(variants),
        figsize=(6.5 * len(variants), 3.8 * len(horizons)),
        squeeze=False,
    )
    for row, horizon in enumerate(horizons):
        for col, variant in enumerate(variants):
            ax = axes[row, col]
            item = feature_paths[
                (feature_paths["horizon_days"] == horizon)
                & (feature_paths["variant"] == variant)
            ]
            if item.empty:
                ax.set_visible(False)
                continue
            for feature, feature_item in item.groupby("feature"):
                feature_item = feature_item.sort_values("day")
                ax.plot(
                    feature_item["day"] / 252.0,
                    feature_item["generated_median_index"],
                    linewidth=1.4,
                    alpha=0.85,
                    label=feature,
                )
            ax.axhline(100.0, color="gray", linewidth=0.8)
            ax.set_title("%s, %d-year feature medians" % (variant, int(horizon / 252)))
            ax.set_xlabel("Years into generated scenario")
            ax.set_ylabel("Index level, start = 100")
            ax.grid(alpha=0.25)
            if row == 0 and col == len(variants) - 1:
                ax.legend(fontsize=7, ncol=2, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.suptitle("Generated Median Index Paths by Feature", y=1.01, fontsize=14)
    savefig(output_dir / "generated_feature_index_medians.png")


def write_report(
    output_dir: Path,
    aggregate: pd.DataFrame,
    grouped: pd.DataFrame,
    delta: pd.DataFrame,
) -> None:
    lines = [
        "# CSDI Comparison Summary",
        "",
        "This folder aggregates vanilla and topology-regularized CSDI outputs across horizons.",
        "",
        "## Files",
        "",
        "- `comparison_dashboard.png`: main metrics in one figure.",
        "- `topology_dashboard.png`: topology-specific summary metrics.",
        "- `topology_curve_overlay.png`: generated topology curves vs real holdout.",
        "- `generated_market_index_paths.png`: all generated equal-weighted market paths starting at 100.",
        "- `generated_market_index_medians.png`: generated median market paths vs actual holdout.",
        "- `generated_feature_index_medians.png`: generated median paths by selected industry.",
        "- `topoloss_minus_vanilla_deltas.png`: relative changes by horizon.",
        "- `comparison_by_horizon.csv`: raw merged metrics.",
        "- `comparison_aggregate.csv`: variant-level averages.",
        "- `comparison_delta_by_horizon.csv`: topoloss minus vanilla by horizon.",
        "- `generated_market_index_paths.csv`: generated and actual equal-weighted index time series.",
        "- `generated_feature_index_medians.csv`: feature-level generated median and actual index time series.",
        "",
        "## Aggregate",
        "",
        "```text",
        aggregate.to_string(index=False),
        "```",
    ]
    if not delta.empty:
        display_cols = [
            col
            for col in [
                "horizon_years",
                "delta_mae",
                "delta_rmse",
                "delta_overall_coverage_90",
                "delta_median_topology_distance_to_real",
                "pct_delta_mae",
                "pct_delta_rmse",
                "pct_delta_median_topology_distance_to_real",
            ]
            if col in delta.columns
        ]
        lines.extend(
            [
                "",
                "## Topoloss Minus Vanilla",
                "",
                "```text",
                delta[display_cols].to_string(index=False),
                "```",
            ]
        )
    (output_dir / "comparison_report.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    labels: List[str]
    if args.labels is None:
        labels = [run_label(path, path.name) for path in args.run_dirs]
    else:
        if len(args.labels) != len(args.run_dirs):
            raise ValueError("--labels must have the same length as run_dirs")
        labels = args.labels

    output_dir = args.output_dir or args.run_dirs[0].parent / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison = pd.concat(
        [read_run(path, label) for path, label in zip(args.run_dirs, labels)],
        ignore_index=True,
    )
    comparison.to_csv(output_dir / "comparison_by_horizon.csv", index=False)
    grouped = grouped_by_variant_horizon(comparison)
    grouped.to_csv(output_dir / "comparison_by_horizon_grouped.csv", index=False)
    aggregate = make_aggregate(comparison)
    aggregate.to_csv(output_dir / "comparison_aggregate.csv", index=False)
    delta = make_delta_table(grouped)
    if not delta.empty:
        delta.to_csv(output_dir / "comparison_delta_by_horizon.csv", index=False)

    curves = pd.concat(
        [read_curves(path, label) for path, label in zip(args.run_dirs, labels)],
        ignore_index=True,
    )
    if not curves.empty:
        curves.to_csv(output_dir / "topology_curves_combined.csv", index=False)

    market_paths = []
    feature_paths = []
    for path, label in zip(args.run_dirs, labels):
        market_df, feature_df = load_generated_index_paths(
            path,
            label,
            max_samples=args.max_path_samples,
        )
        if not market_df.empty:
            market_paths.append(market_df)
        if not feature_df.empty:
            feature_paths.append(feature_df)
    market_paths_df = pd.concat(market_paths, ignore_index=True) if market_paths else pd.DataFrame()
    feature_paths_df = pd.concat(feature_paths, ignore_index=True) if feature_paths else pd.DataFrame()
    if not market_paths_df.empty:
        market_paths_df.to_csv(output_dir / "generated_market_index_paths.csv", index=False)
    if not feature_paths_df.empty:
        feature_paths_df.to_csv(output_dir / "generated_feature_index_medians.csv", index=False)

    plot_dashboard(grouped, output_dir)
    plot_topology_dashboard(grouped, output_dir)
    plot_delta_bars(delta, output_dir)
    plot_curve_overlay(curves, output_dir)
    plot_generated_market_paths(market_paths_df, output_dir)
    plot_generated_market_medians(market_paths_df, output_dir)
    plot_feature_index_medians(feature_paths_df, output_dir)
    write_report(output_dir, aggregate, grouped, delta)

    print("Wrote comparison tables and plots to %s" % output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
