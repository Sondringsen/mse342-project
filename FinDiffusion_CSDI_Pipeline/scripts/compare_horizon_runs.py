#!/usr/bin/env python
"""Aggregate completed horizon-sweep runs into cross-horizon comparison tables."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
import yaml


CORE_COLUMNS = [
    "run_name",
    "horizon",
    "model",
    "forecast_median_mae",
    "forecast_median_rmse",
    "forecast_coverage_50",
    "forecast_coverage_90",
    "analysis_coverage_50_error",
    "analysis_coverage_90_error",
    "metric_distribution_wasserstein",
    "metric_distribution_ks_statistic",
    "metric_distribution_js_divergence",
    "metric_temporal_acf_mae",
    "metric_temporal_acf_squared_mae",
    "metric_temporal_vol_cluster_real",
    "metric_temporal_vol_cluster_syn",
    "analysis_vol_cluster_abs_gap",
    "stylized_real_volatility_clustering_acf_squared_lag1",
    "stylized_synthetic_volatility_clustering_acf_squared_lag1",
    "analysis_stylized_vol_cluster_lag1_gap",
    "stylized_real_volatility_clustering_acf_absolute_lag1",
    "stylized_synthetic_volatility_clustering_acf_absolute_lag1",
    "analysis_stylized_abs_return_lag1_gap",
    "stylized_real_fat_tails_excess_kurtosis",
    "stylized_synthetic_fat_tails_excess_kurtosis",
    "analysis_excess_kurtosis_gap",
    "stylized_real_fat_tails_tail_index",
    "stylized_synthetic_fat_tails_tail_index",
    "analysis_tail_index_gap",
    "stylized_synthetic_summary_pass_rate",
    "metric_summary_distribution_score",
    "metric_summary_temporal_score",
    "metric_summary_overall_score",
    "path_shape_real_length",
    "path_shape_synthetic_length",
]


RANKING_METRICS = [
    ("forecast_median_mae", "min"),
    ("analysis_coverage_90_error", "min"),
    ("metric_distribution_wasserstein", "min"),
    ("metric_temporal_acf_squared_mae", "min"),
    ("analysis_vol_cluster_abs_gap", "min"),
    ("analysis_stylized_vol_cluster_lag1_gap", "min"),
    ("analysis_excess_kurtosis_gap", "min"),
    ("analysis_tail_index_gap", "min"),
    ("stylized_synthetic_summary_pass_rate", "max"),
    ("metric_summary_overall_score", "max"),
]


STEP_COLUMN_RE = re.compile(r"^forecast_by_horizon_step_(\d+)_(.+)$")
HORIZON_RE = re.compile(r"horizon_(\d+)d")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare completed FinDiffusion/CSDI horizon runs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("run_dirs", nargs="+", type=Path, help="Completed run directories")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for cross-horizon comparison artifacts",
    )
    args = parser.parse_args()

    run_dirs = [path.resolve() for path in args.run_dirs]
    output_dir = args.output_dir.resolve() if args.output_dir else default_output_dir(run_dirs)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = build_horizon_summary(run_dirs)
    summary.to_csv(output_dir / "horizon_summary.csv", index=False)

    core = summary.loc[:, [col for col in CORE_COLUMNS if col in summary.columns]]
    core.to_csv(output_dir / "horizon_core_metrics.csv", index=False)

    steps = build_step_metrics(summary)
    if not steps.empty:
        steps.to_csv(output_dir / "horizon_forecast_by_step.csv", index=False)

    stylized = load_stylized_tables(run_dirs)
    if not stylized.empty:
        stylized.to_csv(output_dir / "horizon_stylized_facts.csv", index=False)

    rankings = build_rankings(summary)
    if not rankings.empty:
        rankings.to_csv(output_dir / "horizon_metric_rankings.csv", index=False)

    write_report(core, steps, rankings, output_dir)
    print(f"Wrote {output_dir}")


def default_output_dir(run_dirs: List[Path]) -> Path:
    if len(run_dirs) == 1:
        return run_dirs[0].parent / f"{run_dirs[0].name}_horizon_comparison"
    parents = {path.parent for path in run_dirs}
    parent = parents.pop() if len(parents) == 1 else Path.cwd()
    return parent / "horizon_comparison"


def build_horizon_summary(run_dirs: Iterable[Path]) -> pd.DataFrame:
    frames = []
    for run_dir in run_dirs:
        summary_path = run_dir / "comparison_summary.csv"
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing completed summary: {summary_path}")
        frame = pd.read_csv(summary_path)
        metadata = pd.DataFrame(
            {
                "run_name": [run_dir.name] * len(frame),
                "horizon": [infer_horizon(run_dir, frame)] * len(frame),
            }
        )
        frame = pd.concat([metadata, frame.reset_index(drop=True)], axis=1)
        frames.append(add_gap_columns(frame))

    summary = pd.concat(frames, ignore_index=True)
    if {"horizon", "model"}.issubset(summary.columns):
        summary = summary.sort_values(["horizon", "model"]).reset_index(drop=True)
    return summary


def infer_horizon(run_dir: Path, summary: pd.DataFrame) -> int:
    config_path = run_dir / "run_config.yaml"
    if config_path.exists():
        config = yaml.safe_load(config_path.read_text()) or {}
        horizon = config.get("data", {}).get("prediction_length")
        if horizon is not None:
            return int(horizon)

    match = HORIZON_RE.search(run_dir.name)
    if match:
        return int(match.group(1))

    if "forecast_max_horizon" in summary.columns and summary["forecast_max_horizon"].notna().any():
        return int(summary["forecast_max_horizon"].dropna().iloc[0])

    raise ValueError(f"Could not infer horizon for {run_dir}")


def add_gap_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    add_abs_gap(
        frame,
        "metric_temporal_vol_cluster_real",
        "metric_temporal_vol_cluster_syn",
        "analysis_vol_cluster_abs_gap",
    )
    add_abs_gap(
        frame,
        "stylized_real_volatility_clustering_acf_squared_lag1",
        "stylized_synthetic_volatility_clustering_acf_squared_lag1",
        "analysis_stylized_vol_cluster_lag1_gap",
    )
    add_abs_gap(
        frame,
        "stylized_real_volatility_clustering_acf_absolute_lag1",
        "stylized_synthetic_volatility_clustering_acf_absolute_lag1",
        "analysis_stylized_abs_return_lag1_gap",
    )
    add_abs_gap(
        frame,
        "stylized_real_fat_tails_excess_kurtosis",
        "stylized_synthetic_fat_tails_excess_kurtosis",
        "analysis_excess_kurtosis_gap",
    )
    add_abs_gap(
        frame,
        "stylized_real_fat_tails_tail_index",
        "stylized_synthetic_fat_tails_tail_index",
        "analysis_tail_index_gap",
    )
    return frame


def add_abs_gap(frame: pd.DataFrame, real_col: str, synthetic_col: str, output_col: str) -> None:
    if {real_col, synthetic_col}.issubset(frame.columns):
        frame[output_col] = (frame[synthetic_col] - frame[real_col]).abs()


def build_step_metrics(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    id_cols = ["run_name", "horizon", "model"]
    for _, row in summary.iterrows():
        for col in summary.columns:
            match = STEP_COLUMN_RE.match(col)
            if not match:
                continue
            value = row[col]
            if pd.isna(value):
                continue
            rows.append(
                {
                    **{id_col: row[id_col] for id_col in id_cols},
                    "horizon_step": int(match.group(1)),
                    "metric": match.group(2),
                    "value": value,
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["horizon", "model", "horizon_step", "metric"])


def load_stylized_tables(run_dirs: Iterable[Path]) -> pd.DataFrame:
    frames = []
    for run_dir in run_dirs:
        path = run_dir / "comparison_stylized_facts.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        metadata = pd.DataFrame(
            {
                "run_name": [run_dir.name] * len(frame),
                "horizon": [infer_horizon(run_dir, pd.DataFrame())] * len(frame),
            }
        )
        frame = pd.concat([metadata, frame.reset_index(drop=True)], axis=1)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(
        ["horizon", "model", "source", "fact"]
    )


def build_rankings(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for horizon, group in summary.groupby("horizon", sort=True):
        for metric, direction in RANKING_METRICS:
            if metric not in group.columns:
                continue
            values = group[["model", metric]].dropna()
            if values.empty:
                continue
            values = values[np.isfinite(values[metric].astype(float))]
            if values.empty:
                continue
            ranked = values.sort_values(metric, ascending=(direction == "min"))
            rows.append(
                {
                    "horizon": int(horizon),
                    "metric": metric,
                    "direction": "lower_is_better" if direction == "min" else "higher_is_better",
                    "best_model": ranked.iloc[0]["model"],
                    "best_value": float(ranked.iloc[0][metric]),
                }
            )
    return pd.DataFrame(rows)


def write_report(
    core: pd.DataFrame,
    steps: pd.DataFrame,
    rankings: pd.DataFrame,
    output_dir: Path,
) -> None:
    lines = [
        "# Horizon Sweep Comparison",
        "",
        "This report compares completed FinDiffusion/CSDI runs across prediction horizons.",
        "",
        "## Core Metrics",
        "",
        markdown_table(
            core,
            [
                "horizon",
                "model",
                "forecast_median_mae",
                "forecast_coverage_90",
                "analysis_coverage_90_error",
                "metric_temporal_acf_squared_mae",
                "metric_temporal_vol_cluster_real",
                "metric_temporal_vol_cluster_syn",
                "analysis_vol_cluster_abs_gap",
                "analysis_stylized_vol_cluster_lag1_gap",
                "analysis_excess_kurtosis_gap",
                "analysis_tail_index_gap",
                "stylized_synthetic_summary_pass_rate",
                "metric_summary_overall_score",
            ],
        ),
        "",
        "## Best Model By Horizon",
        "",
        markdown_table(rankings, ["horizon", "metric", "direction", "best_model", "best_value"]),
        "",
        "## Files",
        "",
        "- `horizon_summary.csv`: all scalar metrics from each completed run.",
        "- `horizon_core_metrics.csv`: selected forecast, temporal, tail, and score metrics.",
        "- `horizon_forecast_by_step.csv`: per-horizon-step forecast metrics in long form.",
        "- `horizon_stylized_facts.csv`: stylized-fact tables with run and horizon columns.",
        "- `horizon_metric_rankings.csv`: best model per metric within each horizon.",
    ]
    if not steps.empty:
        coverage = steps[steps["metric"].isin(["coverage_90", "median_mae"])]
        lines.extend(["", "## Forecast Step Detail", "", markdown_table(coverage.head(80))])
    (output_dir / "horizon_report.md").write_text("\n".join(lines) + "\n")


def markdown_table(frame: pd.DataFrame, columns: Optional[List[str]] = None) -> str:
    if frame.empty:
        return "_No rows._"
    display = frame.loc[:, [col for col in (columns or list(frame.columns)) if col in frame.columns]]
    header = "| " + " | ".join(display.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    body = [
        "| " + " | ".join(format_value(row[col]) for col in display.columns) + " |"
        for _, row in display.iterrows()
    ]
    return "\n".join([header, separator] + body)


def format_value(value) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6g}"
    return str(value)


if __name__ == "__main__":
    main()
