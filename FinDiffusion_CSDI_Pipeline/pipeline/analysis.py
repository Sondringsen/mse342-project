"""Cross-model comparison analysis for completed pipeline runs."""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mse342_matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CORE_COLUMNS = [
    "model",
    "forecast_median_mae",
    "forecast_median_rmse",
    "forecast_mean_mae",
    "forecast_mean_rmse",
    "analysis_abs_bias",
    "forecast_coverage_50",
    "analysis_coverage_50_error",
    "forecast_coverage_90",
    "analysis_coverage_90_error",
    "forecast_avg_width_50",
    "forecast_avg_width_90",
    "metric_distribution_wasserstein",
    "metric_distribution_ks_statistic",
    "metric_distribution_js_divergence",
    "metric_distribution_mean_diff",
    "metric_distribution_std_diff",
    "metric_distribution_skew_diff",
    "metric_distribution_kurtosis_diff",
    "metric_temporal_acf_mae",
    "metric_temporal_acf_squared_mae",
    "metric_diversity_mean_pairwise_corr",
    "metric_diversity_std_pairwise_corr",
    "metric_diversity_mean_sample_std",
    "metric_diversity_std_sample_std",
    "metric_diversity_unique_ratio",
    "metric_summary_distribution_score",
    "metric_summary_temporal_score",
    "metric_summary_diversity_score",
    "metric_summary_overall_score",
    "stylized_synthetic_summary_pass_rate",
    "stylized_real_summary_pass_rate",
    "dist_comparison_ks_pvalue",
    "dist_comparison_wasserstein_distance",
]


RANKING_SPECS = [
    ("Median MAE", "forecast_median_mae", "min"),
    ("Median RMSE", "forecast_median_rmse", "min"),
    ("Mean MAE", "forecast_mean_mae", "min"),
    ("Mean RMSE", "forecast_mean_rmse", "min"),
    ("Absolute bias", "analysis_abs_bias", "min"),
    ("50% coverage error", "analysis_coverage_50_error", "min"),
    ("90% coverage error", "analysis_coverage_90_error", "min"),
    ("Wasserstein", "metric_distribution_wasserstein", "min"),
    ("KS statistic", "metric_distribution_ks_statistic", "min"),
    ("JS divergence", "metric_distribution_js_divergence", "min"),
    ("Mean difference", "metric_distribution_mean_diff", "min"),
    ("Std difference", "metric_distribution_std_diff", "min"),
    ("Skew difference", "metric_distribution_skew_diff", "min"),
    ("Kurtosis difference", "metric_distribution_kurtosis_diff", "min"),
    ("Return ACF MAE", "metric_temporal_acf_mae", "min"),
    ("Squared-return ACF MAE", "metric_temporal_acf_squared_mae", "min"),
    ("Distribution score", "metric_summary_distribution_score", "max"),
    ("Temporal score", "metric_summary_temporal_score", "max"),
    ("Diversity score", "metric_summary_diversity_score", "max"),
    ("Overall score", "metric_summary_overall_score", "max"),
    ("Unique ratio", "metric_diversity_unique_ratio", "max"),
    ("Stylized pass rate", "stylized_synthetic_summary_pass_rate", "max"),
]


def load_results(run_dir: Path) -> List[Dict[str, Any]]:
    """Load per-model evaluation JSON files from a run directory."""
    results = []
    for result_path in sorted(run_dir.glob("*/evaluation_results.json")):
        with result_path.open("r") as f:
            results.append(json.load(f))
    if not results:
        raise FileNotFoundError(f"No evaluation_results.json files found under {run_dir}")
    return results


def write_comparison_analysis(
    results: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> pd.DataFrame:
    """Write all top-level comparison artifacts and return the full summary table."""
    output_dir.mkdir(parents=True, exist_ok=True)
    results = list(results)
    summary = build_summary_frame(results)
    summary.to_csv(output_dir / "comparison_summary.csv", index=False)
    (output_dir / "comparison_summary.json").write_text(
        json.dumps(_records_for_json(summary), indent=2) + "\n"
    )

    write_grouped_metric_tables(summary, output_dir)
    stylized = build_stylized_facts_table(results)
    if not stylized.empty:
        stylized.to_csv(output_dir / "comparison_stylized_facts.csv", index=False)

    rankings = build_rankings(summary)
    rankings.to_csv(output_dir / "comparison_metric_rankings.csv", index=False)

    write_report(summary, rankings, output_dir)
    write_run_readme(summary, rankings, output_dir)
    write_text_metric_report(summary, rankings, output_dir)
    create_comparison_plots(summary, output_dir)
    create_prediction_overlay_plots(output_dir)
    return summary


def build_summary_frame(results: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows = [flatten_result(result) for result in results]
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame = add_derived_columns(frame)
    if "model" in frame.columns:
        frame = frame.sort_values("model").reset_index(drop=True)
    ordered = [col for col in CORE_COLUMNS if col in frame.columns]
    ordered.extend(sorted(col for col in frame.columns if col not in ordered))
    return frame.reindex(columns=ordered)


def flatten_result(result: Mapping[str, Any]) -> Dict[str, Any]:
    """Flatten one evaluation result while preserving every scalar metric."""
    row: Dict[str, Any] = {"model": str(result.get("model", "unknown"))}
    _flatten_mapping(row, "forecast", result.get("forecast", {}))
    _flatten_mapping(row, "metric", result.get("metrics", {}))

    stylized = result.get("stylized_facts", {})
    if isinstance(stylized, Mapping):
        _flatten_mapping(row, "stylized_real", stylized.get("real", {}))
        _flatten_mapping(row, "stylized_synthetic", stylized.get("synthetic", {}))
        _flatten_mapping(row, "dist_comparison", stylized.get("comparison", {}))

    path_shapes = result.get("path_shapes", {})
    if isinstance(path_shapes, Mapping):
        for name, shape in path_shapes.items():
            if isinstance(shape, Sequence) and not isinstance(shape, (str, bytes)):
                clean = _clean_key(str(name))
                row[f"path_shape_{clean}_rank"] = len(shape)
                if len(shape) >= 1:
                    row[f"path_shape_{clean}_n_sequences"] = shape[0]
                if len(shape) >= 2:
                    row[f"path_shape_{clean}_length"] = shape[1]
    return row


def add_derived_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if "forecast_bias" in frame:
        frame["analysis_abs_bias"] = frame["forecast_bias"].abs()
    if "forecast_coverage_50" in frame:
        frame["analysis_coverage_50_error"] = (frame["forecast_coverage_50"] - 0.50).abs()
    if "forecast_coverage_90" in frame:
        frame["analysis_coverage_90_error"] = (frame["forecast_coverage_90"] - 0.90).abs()
    if {"stylized_synthetic_summary_pass_rate", "stylized_real_summary_pass_rate"}.issubset(
        frame.columns
    ):
        frame["analysis_stylized_pass_rate_gap_to_real"] = (
            frame["stylized_synthetic_summary_pass_rate"] - frame["stylized_real_summary_pass_rate"]
        )
    return frame


def build_stylized_facts_table(results: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows = []
    for result in results:
        model = str(result.get("model", "unknown"))
        stylized = result.get("stylized_facts", {})
        if not isinstance(stylized, Mapping):
            continue
        for source in ["real", "synthetic"]:
            source_results = stylized.get(source, {})
            if not isinstance(source_results, Mapping):
                continue
            for fact_name, values in source_results.items():
                if not isinstance(values, Mapping):
                    continue
                row = {
                    "model": model,
                    "source": source,
                    "fact": fact_name,
                }
                for key, value in values.items():
                    scalar = _coerce_scalar(value)
                    if scalar is not None:
                        row[_clean_key(key)] = scalar
                rows.append(row)
    return pd.DataFrame(rows)


def write_grouped_metric_tables(summary: pd.DataFrame, output_dir: Path) -> None:
    groups = {
        "comparison_forecast_metrics.csv": ("forecast_", "analysis_coverage_", "analysis_abs_bias"),
        "comparison_distribution_metrics.csv": (
            "metric_distribution_",
            "dist_comparison_",
        ),
        "comparison_temporal_metrics.csv": ("metric_temporal_",),
        "comparison_diversity_metrics.csv": ("metric_diversity_",),
        "comparison_score_metrics.csv": ("metric_summary_",),
    }
    for filename, prefixes in groups.items():
        cols = ["model"] + [
            col for col in summary.columns if col != "model" and col.startswith(prefixes)
        ]
        if len(cols) > 1:
            summary.loc[:, cols].to_csv(output_dir / filename, index=False)


def build_rankings(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, column, direction in RANKING_SPECS:
        if column not in summary.columns:
            continue
        values = summary[["model", column]].dropna()
        values = values[np.isfinite(values[column].astype(float))]
        if values.empty:
            continue
        ranked = values.sort_values(column, ascending=(direction == "min"))
        best_model = str(ranked.iloc[0]["model"])
        row = {
            "metric": label,
            "column": column,
            "direction": "lower_is_better" if direction == "min" else "higher_is_better",
            "best_model": best_model,
            "best_value": float(ranked.iloc[0][column]),
        }
        for _, value_row in values.iterrows():
            row[str(value_row["model"])] = float(value_row[column])
        rows.append(row)
    return pd.DataFrame(rows)


def write_report(summary: pd.DataFrame, rankings: pd.DataFrame, output_dir: Path) -> None:
    lines = [
        "# FinDiffusion vs CSDI Comparison",
        "",
        "This report compares both models using the full FinDiffusion evaluation surface: "
        "forecast accuracy, distribution similarity, temporal structure, diversity, "
        "summary scores, and stylized-facts validation.",
        "",
        "## Models",
        "",
    ]
    for model in summary.get("model", pd.Series(dtype=str)).tolist():
        lines.append(f"- {model}")

    lines.extend(
        [
            "",
            "## Core Metrics",
            "",
            _markdown_table(
                summary,
                [
                    ("model", "Model"),
                    ("forecast_median_mae", "Median MAE"),
                    ("forecast_median_rmse", "Median RMSE"),
                    ("analysis_abs_bias", "Abs Bias"),
                    ("forecast_coverage_50", "50% Coverage"),
                    ("forecast_coverage_90", "90% Coverage"),
                    ("metric_distribution_wasserstein", "Wasserstein"),
                    ("metric_distribution_ks_statistic", "KS"),
                    ("metric_temporal_acf_squared_mae", "Sq ACF MAE"),
                    ("metric_diversity_unique_ratio", "Unique Ratio"),
                    ("metric_summary_overall_score", "Overall Score"),
                    ("stylized_synthetic_summary_pass_rate", "Stylized Pass"),
                ],
            ),
            "",
            "## Metric Winners",
            "",
            _markdown_table(
                rankings,
                [
                    ("metric", "Metric"),
                    ("direction", "Direction"),
                    ("best_model", "Best Model"),
                    ("best_value", "Best Value"),
                ],
            ),
            "",
            "## Stylized Facts",
            "",
            _markdown_table(
                _stylized_report_frame(summary),
                [
                    ("model", "Model"),
                    ("source", "Source"),
                    ("excess_kurtosis", "Excess Kurtosis"),
                    ("skewness", "Skewness"),
                    ("tail_index", "Tail Index"),
                    ("vol_cluster_acf1", "Sq Return ACF1"),
                    ("abs_return_acf1", "Abs Return ACF1"),
                    ("leverage_corr", "Leverage Corr"),
                    ("return_acf1", "Return ACF1"),
                    ("pass_rate", "Pass Rate"),
                ],
            ),
            "",
            "## Artifact Index",
            "",
            "- `comparison_summary.csv`: every scalar metric from each model result.",
            "- `comparison_forecast_metrics.csv`: point forecast and interval metrics.",
            "- `comparison_distribution_metrics.csv`: FinDiffusion distribution metrics and two-sample statistics.",
            "- `comparison_temporal_metrics.csv`: raw and squared-return autocorrelation metrics.",
            "- `comparison_diversity_metrics.csv`: synthetic sample diversity metrics.",
            "- `comparison_stylized_facts.csv`: real and synthetic stylized-facts test details.",
            "- `comparison_metric_rankings.csv`: best model for each comparison metric.",
            "- `plots/`: side-by-side comparison plots and prediction overlays when predictions are available.",
        ]
    )
    (output_dir / "comparison_report.md").write_text("\n".join(lines) + "\n")


def write_run_readme(summary: pd.DataFrame, rankings: pd.DataFrame, output_dir: Path) -> None:
    lines = [
        f"# {output_dir.name}",
        "",
        "Use this file as the entry point for this run.",
        "",
        "## Quick Links",
        "",
        "- [Full comparison report](comparison_report.md)",
        "- [All scalar metrics](comparison_summary.csv)",
        "- [Metric winners](comparison_metric_rankings.csv)",
        "- [Stylized facts](comparison_stylized_facts.csv)",
        "- [Comparison plots](plots/)",
        "- [Run config](run_config.yaml)",
        "",
        "## Models",
        "",
        _markdown_table(
            summary,
            [
                ("model", "Model"),
                ("forecast_median_mae", "Median MAE"),
                ("forecast_median_rmse", "Median RMSE"),
                ("metric_summary_overall_score", "Overall Score"),
                ("stylized_synthetic_summary_pass_rate", "Stylized Pass"),
            ],
        ),
        "",
        "## Most Useful Plots",
        "",
        "- [Forecast accuracy](plots/comparison_forecast_accuracy.png)",
        "- [Distribution and temporal errors](plots/comparison_distribution_temporal.png)",
        "- [Coverage](plots/comparison_coverage.png)",
        "- [Stylized-facts pass rate](plots/comparison_stylized_pass_rate.png)",
        "- [Generated return time series](plots/comparison_generated_timeseries.png)",
        "- [Return distribution overlay](plots/comparison_return_distribution_overlay.png)",
        "",
        "## Metric Winners",
        "",
        _markdown_table(
            rankings,
            [
                ("metric", "Metric"),
                ("best_model", "Best Model"),
                ("best_value", "Best Value"),
                ("direction", "Direction"),
            ],
        ),
    ]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n")


def _stylized_report_frame(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    fields = {
        "excess_kurtosis": "fat_tails_excess_kurtosis",
        "skewness": "fat_tails_skewness",
        "tail_index": "fat_tails_tail_index",
        "vol_cluster_acf1": "volatility_clustering_acf_squared_lag1",
        "abs_return_acf1": "volatility_clustering_acf_absolute_lag1",
        "leverage_corr": "leverage_effect_leverage_correlation",
        "return_acf1": "no_autocorrelation_acf_lag1",
        "pass_rate": "summary_pass_rate",
    }
    for _, row in summary.iterrows():
        for source in ["real", "synthetic"]:
            out = {"model": row["model"], "source": source}
            for output_name, suffix in fields.items():
                col = f"stylized_{source}_{suffix}"
                out[output_name] = row[col] if col in summary.columns else np.nan
            rows.append(out)
    return pd.DataFrame(rows)


def write_text_metric_report(summary: pd.DataFrame, rankings: pd.DataFrame, output_dir: Path) -> None:
    lines = ["COMPARISON METRICS", "=" * 60, ""]
    for _, row in summary.iterrows():
        lines.append(str(row["model"]))
        lines.append("-" * 40)
        for col in summary.columns:
            if col == "model":
                continue
            value = row[col]
            if _is_missing(value):
                continue
            lines.append(f"{col}: {_format_value(value)}")
        lines.append("")
    lines.extend(["METRIC WINNERS", "=" * 60])
    for _, row in rankings.iterrows():
        lines.append(
            f"{row['metric']}: {row['best_model']} "
            f"({_format_value(row['best_value'])}, {row['direction']})"
        )
    (output_dir / "comparison_metrics_report.txt").write_text("\n".join(lines) + "\n")


def create_comparison_plots(summary: pd.DataFrame, output_dir: Path) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    _bar_plot(
        summary,
        [
            ("metric_summary_distribution_score", "Distribution"),
            ("metric_summary_temporal_score", "Temporal"),
            ("metric_summary_diversity_score", "Diversity"),
            ("metric_summary_overall_score", "Overall"),
        ],
        plot_dir / "comparison_scores.png",
        "FinDiffusion Summary Scores",
        "Score",
    )
    _bar_plot(
        summary,
        [
            ("forecast_median_mae", "Median MAE"),
            ("forecast_mean_mae", "Mean MAE"),
            ("forecast_median_rmse", "Median RMSE"),
            ("forecast_mean_rmse", "Mean RMSE"),
            ("analysis_abs_bias", "Abs Bias"),
        ],
        plot_dir / "comparison_forecast_accuracy.png",
        "Forecast Accuracy",
        "Return error",
    )
    _bar_plot(
        summary,
        [
            ("metric_distribution_wasserstein", "Wasserstein"),
            ("metric_distribution_ks_statistic", "KS"),
            ("metric_distribution_js_divergence", "JS"),
            ("metric_temporal_acf_mae", "ACF MAE"),
            ("metric_temporal_acf_squared_mae", "Sq ACF MAE"),
        ],
        plot_dir / "comparison_distribution_temporal.png",
        "Distribution and Temporal Errors",
        "Error",
    )
    _coverage_plot(summary, plot_dir / "comparison_coverage.png")
    _bar_plot(
        summary,
        [
            ("stylized_synthetic_summary_pass_rate", "Synthetic"),
            ("stylized_real_summary_pass_rate", "Real"),
        ],
        plot_dir / "comparison_stylized_pass_rate.png",
        "Stylized-Facts Pass Rate",
        "Pass rate",
    )


def create_prediction_overlay_plots(output_dir: Path) -> None:
    prediction_frames = []
    for pred_path in sorted(output_dir.glob("*/predictions.csv")):
        model = pred_path.parent.name
        predictions = pd.read_csv(pred_path)
        predictions["model"] = model
        prediction_frames.append(predictions)
    if not prediction_frames:
        return

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    sample_cols = [col for col in predictions.columns if col.startswith("sample_")]

    if sample_cols:
        fig, ax = plt.subplots(figsize=(9, 5))
        real = predictions.drop_duplicates(["ticker", "target_index"])["actual"].to_numpy(float)
        ax.hist(real, bins=100, density=True, alpha=0.35, label="Real")
        for model, model_df in predictions.groupby("model", sort=True):
            values = model_df[sample_cols].to_numpy(float).reshape(-1)
            ax.hist(values, bins=100, density=True, histtype="step", linewidth=1.8, label=model)
        ax.set_xlabel("Daily log return")
        ax.set_ylabel("Density")
        ax.set_title("Generated Return Distribution Overlay")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / "comparison_return_distribution_overlay.png", dpi=150)
        plt.close(fig)
        _generated_timeseries_overlay(predictions, sample_cols, plot_dir)

    fig, ax = plt.subplots(figsize=(9, 5))
    for model, model_df in predictions.groupby("model", sort=True):
        errors = model_df["pred_median"].to_numpy(float) - model_df["actual"].to_numpy(float)
        ax.hist(errors, bins=80, density=True, alpha=0.45, label=model)
    ax.axvline(0, color="k", linewidth=0.8)
    ax.set_xlabel("Median forecast error")
    ax.set_ylabel("Density")
    ax.set_title("Forecast Error Overlay")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "comparison_forecast_error_overlay.png", dpi=150)
    plt.close(fig)


def _generated_timeseries_overlay(
    predictions: pd.DataFrame,
    sample_cols: Sequence[str],
    plot_dir: Path,
) -> None:
    common_tickers = sorted(predictions["ticker"].dropna().unique())
    if not common_tickers:
        return
    ticker = common_tickers[0]
    ticker_df = predictions[predictions["ticker"] == ticker].sort_values(["model", "target_index"])
    models = sorted(ticker_df["model"].unique())
    if not models:
        return

    fig, axes = plt.subplots(len(models), 1, figsize=(12, max(3.5, 3.2 * len(models))), sharex=True)
    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        model_df = ticker_df[ticker_df["model"] == model].sort_values("target_index")
        x = np.arange(len(model_df))
        ax.plot(x, model_df["actual"].to_numpy(float), color="black", linewidth=1.4, label="Real")
        for sample_col in sample_cols[: min(6, len(sample_cols))]:
            if sample_col not in model_df.columns:
                continue
            ax.plot(
                x,
                model_df[sample_col].to_numpy(float),
                linewidth=0.8,
                alpha=0.55,
                label="Generated" if sample_col == sample_cols[0] else None,
            )
        ax.axhline(0, color="0.35", linewidth=0.6)
        ax.set_title(f"{model} Generated Returns for {ticker}")
        ax.set_ylabel("Daily log return")
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("Forecast date index")
    fig.tight_layout()
    fig.savefig(plot_dir / "comparison_generated_timeseries.png", dpi=150)
    plt.close(fig)


def _bar_plot(
    summary: pd.DataFrame,
    columns: Sequence[Tuple[str, str]],
    path: Path,
    title: str,
    ylabel: str,
) -> None:
    available = [(col, label) for col, label in columns if col in summary.columns]
    if not available or summary.empty:
        return
    models = summary["model"].astype(str).tolist()
    x = np.arange(len(available))
    width = min(0.8 / max(1, len(models)), 0.35)
    fig, ax = plt.subplots(figsize=(max(8, len(available) * 1.4), 5))
    for model_idx, model in enumerate(models):
        values = [
            float(summary.loc[summary["model"] == model, col].iloc[0]) for col, _label in available
        ]
        offset = (model_idx - (len(models) - 1) / 2.0) * width
        ax.bar(x + offset, values, width=width, label=model)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _col, label in available], rotation=30, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _coverage_plot(summary: pd.DataFrame, path: Path) -> None:
    cols = [
        ("forecast_coverage_50", "50% interval", 0.50),
        ("forecast_coverage_90", "90% interval", 0.90),
    ]
    available = [(col, label, target) for col, label, target in cols if col in summary.columns]
    if not available or summary.empty:
        return
    models = summary["model"].astype(str).tolist()
    x = np.arange(len(available))
    width = min(0.8 / max(1, len(models)), 0.35)
    fig, ax = plt.subplots(figsize=(7, 5))
    for model_idx, model in enumerate(models):
        values = [
            float(summary.loc[summary["model"] == model, col].iloc[0])
            for col, _label, _target in available
        ]
        offset = (model_idx - (len(models) - 1) / 2.0) * width
        ax.bar(x + offset, values, width=width, label=model)
    for idx, (_col, _label, target) in enumerate(available):
        ax.hlines(target, idx - 0.45, idx + 0.45, colors="k", linestyles="--", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _col, label, _target in available])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Empirical coverage")
    ax.set_title("Forecast Interval Coverage")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _flatten_mapping(row: Dict[str, Any], prefix: str, values: Any) -> None:
    if not isinstance(values, Mapping):
        return
    for key, value in values.items():
        child_prefix = f"{prefix}_{_clean_key(str(key))}"
        if isinstance(value, Mapping):
            _flatten_mapping(row, child_prefix, value)
            continue
        scalar = _coerce_scalar(value)
        if scalar is not None:
            row[child_prefix] = scalar


def _coerce_scalar(value: Any) -> Optional[Any]:
    if isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return np.nan
        return value
    if isinstance(value, np.generic):
        return _coerce_scalar(value.item())
    return None


def _clean_key(key: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "_", key).strip("_").lower()


def _records_for_json(frame: pd.DataFrame) -> List[Dict[str, Any]]:
    records = []
    for record in frame.to_dict(orient="records"):
        clean = {}
        for key, value in record.items():
            clean[key] = None if _is_missing(value) else _coerce_json_value(value)
        records.append(clean)
    return records


def _coerce_json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _markdown_table(frame: pd.DataFrame, columns: Sequence[Tuple[str, str]]) -> str:
    available = [(col, label) for col, label in columns if col in frame.columns]
    if not available or frame.empty:
        return "_No data available._"
    header = "| " + " | ".join(label for _col, label in available) + " |"
    divider = "| " + " | ".join("---" for _col, _label in available) + " |"
    rows = [header, divider]
    for _, row in frame.iterrows():
        rows.append("| " + " | ".join(_format_value(row[col]) for col, _label in available) + " |")
    return "\n".join(rows)


def _format_value(value: Any) -> str:
    if _is_missing(value):
        return ""
    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6g}"
    return str(value)


def _is_missing(value: Any) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False
