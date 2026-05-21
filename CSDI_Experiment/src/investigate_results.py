#!/usr/bin/env python3
"""Investigate raw data and generated CSDI scenario outputs for common issues."""

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mse342_matplotlib")

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create data and generated-path diagnostics for a CSDI comparison run."
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "french49_daily_returns.csv",
    )
    parser.add_argument("--date-column", default="date")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def load_json(path: Path, default):
    if path.exists():
        return json.loads(path.read_text())
    return default


def tensor_to_numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def model_to_simple_returns(values: np.ndarray, config: dict) -> np.ndarray:
    transform = str(config.get("return_transform", "simple"))
    if transform == "simple":
        return values
    if transform == "log":
        return np.expm1(np.clip(values, -50.0, 50.0))
    raise ValueError("Unknown return transform for %s: %s" % (config.get("data", "run"), transform))


def variant_dirs(run_dir: Path) -> List[Path]:
    if (run_dir / "run_config.json").exists():
        return [run_dir]
    return sorted([path for path in run_dir.iterdir() if path.is_dir() and (path / "run_config.json").exists()])


def first_generated_output(horizon_dir: Path) -> Optional[Path]:
    matches = sorted(horizon_dir.glob("generated_outputs_nsample*.pk"))
    return matches[0] if matches else None


def horizon_days_from_dir(horizon_dir: Path) -> int:
    return int(horizon_dir.name.replace("horizon_", ""))


def raw_data_diagnostics(data_path: Path, date_column: str, output_dir: Path) -> Dict[str, object]:
    df = pd.read_csv(data_path)
    if date_column not in df.columns:
        raise ValueError("Missing date column %s in %s" % (date_column, data_path))
    dates = pd.to_datetime(df[date_column])
    values = df.drop(columns=[date_column]).apply(pd.to_numeric, errors="coerce")

    stacked_abs = values.abs().stack().sort_values(ascending=False)
    extreme_rows = []
    for (row_index, feature), abs_value in stacked_abs.head(30).items():
        extreme_rows.append(
            {
                "date": str(pd.Timestamp(dates.iloc[row_index]).date()),
                "feature": feature,
                "return": float(values.iloc[row_index][feature]),
                "absolute_return": float(abs_value),
            }
        )
    extremes = pd.DataFrame(extreme_rows)
    extremes.to_csv(output_dir / "raw_data_extreme_returns.csv", index=False)

    summary = {
        "data_path": str(data_path),
        "rows": int(len(df)),
        "features": int(values.shape[1]),
        "date_start": str(dates.iloc[0].date()),
        "date_end": str(dates.iloc[-1].date()),
        "missing_values": int(values.isna().sum().sum()),
        "duplicate_dates": int(df[date_column].duplicated().sum()),
        "dates_monotonic_increasing": bool(dates.is_monotonic_increasing),
        "min_return": float(values.min().min()),
        "max_return": float(values.max().max()),
        "mean_abs_return": float(values.abs().stack().mean()),
        "count_abs_gt_20pct": int((values.abs() > 0.20).sum().sum()),
        "count_abs_gt_50pct": int((values.abs() > 0.50).sum().sum()),
        "count_return_le_minus_100pct": int((values <= -1.0).sum().sum()),
    }
    return summary


def generated_diagnostics(run_dir: Path, output_dir: Path) -> pd.DataFrame:
    rows = []
    for child in variant_dirs(run_dir):
        config = load_json(child / "run_config.json", {})
        variant = str(config.get("variant_name", child.name))
        nsample = int(config.get("nsample", 0))
        features = list(config.get("features", []))

        for horizon_dir in sorted(child.glob("horizon_*")):
            output_path = first_generated_output(horizon_dir)
            if output_path is None:
                continue
            horizon_days = horizon_days_from_dir(horizon_dir)
            with output_path.open("rb") as f:
                samples, target, eval_points, _observed, _time, scaler, mean_scaler = pickle.load(f)

            samples_np = tensor_to_numpy(samples)
            target_np = tensor_to_numpy(target)
            eval_np = tensor_to_numpy(eval_points)
            scaler_np = tensor_to_numpy(scaler)
            mean_np = tensor_to_numpy(mean_scaler)
            samples_np = samples_np * scaler_np.reshape(1, 1, 1, -1) + mean_np.reshape(1, 1, 1, -1)
            target_np = target_np * scaler_np.reshape(1, 1, -1) + mean_np.reshape(1, 1, -1)
            samples_np = model_to_simple_returns(samples_np, config)
            target_np = model_to_simple_returns(target_np, config)

            forecast_positions = np.where(eval_np[0].sum(axis=1) > 0)[0]
            sample_count = min(nsample if nsample else samples_np.shape[1], samples_np.shape[1])
            generated_returns = samples_np[0, :sample_count, :, :][:, forecast_positions, :]
            actual_returns = target_np[0, forecast_positions, :]
            if generated_returns.shape[1] != len(forecast_positions):
                raise ValueError("Generated path extraction failed for %s" % output_path)

            generated_market_returns = generated_returns.mean(axis=2)
            actual_market_returns = actual_returns.mean(axis=1)
            generated_market_index = 100.0 * np.cumprod(1.0 + generated_market_returns, axis=1)
            actual_market_index = 100.0 * np.cumprod(1.0 + actual_market_returns)
            generated_feature_index = 100.0 * np.cumprod(1.0 + generated_returns, axis=1)

            final_index = generated_market_index[:, -1]
            bad_feature_returns = generated_returns <= -1.0
            very_large_feature_returns = np.abs(generated_returns) > 0.50

            rows.append(
                {
                    "run": child.name,
                    "variant": variant,
                    "horizon_days": int(horizon_days),
                    "horizon_years": float(horizon_days) / 252.0,
                    "features": int(len(features) or samples_np.shape[-1]),
                    "samples": int(sample_count),
                    "forecast_days": int(len(forecast_positions)),
                    "raw_sample_shape": "x".join(str(x) for x in samples_np.shape),
                    "generated_mean_daily_market_return": float(generated_market_returns.mean()),
                    "actual_mean_daily_market_return": float(actual_market_returns.mean()),
                    "generated_daily_market_vol": float(generated_market_returns.std()),
                    "actual_daily_market_vol": float(actual_market_returns.std()),
                    "generated_daily_feature_vol": float(generated_returns.std()),
                    "actual_daily_feature_vol": float(actual_returns.std()),
                    "generated_min_daily_feature_return": float(generated_returns.min()),
                    "actual_min_daily_feature_return": float(actual_returns.min()),
                    "generated_max_daily_feature_return": float(generated_returns.max()),
                    "actual_max_daily_feature_return": float(actual_returns.max()),
                    "generated_feature_returns_le_minus_100pct": int(bad_feature_returns.sum()),
                    "generated_feature_returns_abs_gt_50pct": int(very_large_feature_returns.sum()),
                    "generated_final_market_index_p05": float(np.quantile(final_index, 0.05)),
                    "generated_final_market_index_median": float(np.quantile(final_index, 0.50)),
                    "generated_final_market_index_p95": float(np.quantile(final_index, 0.95)),
                    "actual_final_market_index": float(actual_market_index[-1]),
                    "actual_final_inside_generated_5_95": bool(
                        np.quantile(final_index, 0.05)
                        <= actual_market_index[-1]
                        <= np.quantile(final_index, 0.95)
                    ),
                    "generated_market_index_min": float(generated_market_index.min()),
                    "generated_market_index_max": float(generated_market_index.max()),
                    "generated_feature_index_min": float(generated_feature_index.min()),
                    "generated_feature_index_max": float(generated_feature_index.max()),
                    "generated_market_index_nonpositive_count": int((generated_market_index <= 0.0).sum()),
                    "generated_feature_index_nonpositive_count": int((generated_feature_index <= 0.0).sum()),
                }
            )

    diagnostics = pd.DataFrame(rows).sort_values(["horizon_days", "variant"])
    diagnostics.to_csv(output_dir / "generated_return_diagnostics.csv", index=False)
    return diagnostics


def load_metric_tables(run_dir: Path) -> Dict[str, pd.DataFrame]:
    comparison_dir = run_dir / "comparison"
    tables = {}
    for name in [
        "comparison_by_horizon.csv",
        "comparison_delta_by_horizon.csv",
        "comparison_aggregate.csv",
    ]:
        path = comparison_dir / name
        if path.exists():
            tables[name] = pd.read_csv(path)
    return tables


def flag_issues(
    raw_summary: Dict[str, object],
    generated: pd.DataFrame,
    tables: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    issues = []

    def add(severity: str, issue: str, evidence: str, recommendation: str) -> None:
        issues.append(
            {
                "severity": severity,
                "issue": issue,
                "evidence": evidence,
                "recommendation": recommendation,
            }
        )

    if raw_summary["missing_values"] != 0:
        add("high", "Raw data contains missing values", "%s missing values" % raw_summary["missing_values"], "Decide whether to impute or drop before training.")
    if raw_summary["duplicate_dates"] != 0:
        add("high", "Raw data contains duplicate dates", "%s duplicate dates" % raw_summary["duplicate_dates"], "Deduplicate before running time-series splits.")
    if not raw_summary["dates_monotonic_increasing"]:
        add("high", "Raw dates are not monotonic", "Date column is not increasing", "Sort by date before fitting the scaler and windows.")
    if raw_summary["count_return_le_minus_100pct"] != 0:
        add("high", "Raw returns contain impossible values", "Return <= -100% exists in raw data", "Inspect source parsing and units.")

    limited_features = generated["features"].min() if not generated.empty else 0
    if limited_features and limited_features < 10:
        add(
            "medium",
            "Run uses a small subset of industries",
            "The generated runs use %d features, not all French 49 industries." % int(limited_features),
            "Be explicit that this is a six-industry experiment, or train the final version on a broader universe.",
        )

    for _, row in generated.iterrows():
        prefix = "%s %.0fy" % (row["variant"], row["horizon_years"])
        if int(row["forecast_days"]) != int(row["horizon_days"]):
            add(
                "high",
                "Forecast length mismatch",
                "%s has %d forecast days but horizon is %d." % (prefix, row["forecast_days"], row["horizon_days"]),
                "Check eval masks and generated-output extraction.",
            )
        if int(row["generated_feature_returns_le_minus_100pct"]) > 0:
            add(
                "high",
                "Generated returns can be economically impossible",
                "%s has %d generated feature returns <= -100%%; min return %.3f."
                % (
                    prefix,
                    row["generated_feature_returns_le_minus_100pct"],
                    row["generated_min_daily_feature_return"],
                ),
                "Constrain generated returns, model log returns, clip only for visualization with disclosure, or penalize extreme daily returns.",
            )
        if int(row["generated_feature_returns_abs_gt_50pct"]) > 0:
            add(
                "medium",
                "Generated daily returns have very large outliers",
                "%s has %d generated feature returns with absolute daily return > 50%%; raw data has %s."
                % (
                    prefix,
                    row["generated_feature_returns_abs_gt_50pct"],
                    raw_summary["count_abs_gt_50pct"],
                ),
                "Add tail diagnostics and consider robust scaling or a bounded/log-return target.",
            )
        if int(row["generated_market_index_nonpositive_count"]) > 0:
            add(
                "high",
                "Generated market index becomes nonpositive",
                "%s market path has %d nonpositive index values."
                % (prefix, row["generated_market_index_nonpositive_count"]),
                "Do not present these paths as valid price paths until the return transformation is fixed.",
            )
        if not bool(row["actual_final_inside_generated_5_95"]):
            add(
                "medium",
                "Generated final-index interval misses actual holdout",
                "%s final actual index %.2f is outside generated 5-95%% range [%.2f, %.2f]."
                % (
                    prefix,
                    row["actual_final_market_index"],
                    row["generated_final_market_index_p05"],
                    row["generated_final_market_index_p95"],
                ),
                "Use this as evidence about calibration, not just point-error metrics.",
            )

    delta = tables.get("comparison_delta_by_horizon.csv", pd.DataFrame())
    if not delta.empty:
        for _, row in delta.iterrows():
            horizon = "%.0fy" % row["horizon_years"]
            if "delta_median_topology_distance_to_real" in row and row["delta_median_topology_distance_to_real"] > 0:
                add(
                    "high",
                    "Topology loss did not improve topology distance",
                    "%s topoloss median topology distance is worse by %.4f (%.1f%%)."
                    % (
                        horizon,
                        row["delta_median_topology_distance_to_real"],
                        row.get("pct_delta_median_topology_distance_to_real", np.nan),
                    ),
                    "Tune or redesign the topology regularizer before claiming topology improvement.",
                )
            if "delta_mae" in row and row["delta_mae"] > 0:
                add(
                    "medium",
                    "Topology loss worsens point error",
                    "%s topoloss MAE is worse by %.6f (%.1f%%)."
                    % (horizon, row["delta_mae"], row.get("pct_delta_mae", np.nan)),
                    "Report the tradeoff explicitly and avoid selecting topoloss only by coverage.",
                )
            if "topoloss_overall_avg_width_90" in row and "vanilla_overall_avg_width_90" in row:
                ratio = row["topoloss_overall_avg_width_90"] / max(row["vanilla_overall_avg_width_90"], 1e-12)
                if ratio > 2.0:
                    add(
                        "medium",
                        "Topology loss improves coverage mostly by widening intervals",
                        "%s topoloss 90%% interval width is %.2fx vanilla." % (horizon, ratio),
                        "Compare calibration together with interval width and CRPS-like metrics.",
                    )

    report = pd.DataFrame(issues)
    if report.empty:
        report = pd.DataFrame(
            [
                {
                    "severity": "info",
                    "issue": "No automatic issues flagged",
                    "evidence": "Basic raw-data and generated-path checks passed.",
                    "recommendation": "Still review plots and assumptions manually.",
                }
            ]
        )
    return report


def markdown_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "_No rows._"
    shown = df.head(max_rows)
    columns = list(shown.columns)
    rows = []
    rows.append("| " + " | ".join(columns) + " |")
    rows.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for _, item in shown.iterrows():
        values = []
        for column in columns:
            value = item[column]
            if isinstance(value, float):
                values.append("%.6g" % value)
            else:
                values.append(str(value).replace("|", "\\|"))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def write_report(
    output_dir: Path,
    raw_summary: Dict[str, object],
    generated: pd.DataFrame,
    issues: pd.DataFrame,
    tables: Dict[str, pd.DataFrame],
) -> None:
    metric_cols = [
        "variant",
        "horizon_years",
        "mae",
        "rmse",
        "overall_coverage_90",
        "overall_avg_width_90",
        "median_topology_distance_to_real",
    ]
    comparison = tables.get("comparison_by_horizon.csv", pd.DataFrame())
    if not comparison.empty:
        comparison = comparison[[col for col in metric_cols if col in comparison.columns]]

    generated_cols = [
        "variant",
        "horizon_years",
        "samples",
        "forecast_days",
        "generated_min_daily_feature_return",
        "generated_max_daily_feature_return",
        "generated_feature_returns_le_minus_100pct",
        "generated_feature_returns_abs_gt_50pct",
        "generated_final_market_index_p05",
        "generated_final_market_index_median",
        "generated_final_market_index_p95",
        "actual_final_market_index",
        "actual_final_inside_generated_5_95",
    ]
    generated_display = generated[[col for col in generated_cols if col in generated.columns]]

    lines = [
        "# Data and Generated-Path Investigation",
        "",
        "## Raw Data",
        "",
        "- Rows: `%s`" % raw_summary["rows"],
        "- Features: `%s`" % raw_summary["features"],
        "- Date range: `%s` to `%s`" % (raw_summary["date_start"], raw_summary["date_end"]),
        "- Missing values: `%s`" % raw_summary["missing_values"],
        "- Duplicate dates: `%s`" % raw_summary["duplicate_dates"],
        "- Min/max daily return: `%.4f` / `%.4f`" % (raw_summary["min_return"], raw_summary["max_return"]),
        "- Raw returns with `abs(return) > 50%%`: `%s`" % raw_summary["count_abs_gt_50pct"],
        "",
        "## Flagged Issues",
        "",
        markdown_table(issues),
        "",
        "## Generated Path Diagnostics",
        "",
        markdown_table(generated_display),
        "",
        "## Forecast Metrics",
        "",
        markdown_table(comparison),
        "",
        "## Output Files",
        "",
        "- `generated_return_diagnostics.csv`: generated-return and index diagnostics.",
        "- `raw_data_extreme_returns.csv`: largest raw daily returns.",
        "- `flagged_issues.csv`: machine-readable issue list.",
    ]
    (output_dir / "data_investigation_report.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    output_dir = (args.output_dir or run_dir / "comparison").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_summary = raw_data_diagnostics(args.data.resolve(), args.date_column, output_dir)
    generated = generated_diagnostics(run_dir, output_dir)
    tables = load_metric_tables(run_dir)
    issues = flag_issues(raw_summary, generated, tables)
    issues.to_csv(output_dir / "flagged_issues.csv", index=False)
    write_report(output_dir, raw_summary, generated, issues, tables)
    print("Wrote investigation report to %s" % (output_dir / "data_investigation_report.md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
