#!/usr/bin/env python3
"""Create plots for fixed-split CSDI scenario paths."""

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


def boxplot_with_labels(ax, data, labels, **kwargs) -> None:
    try:
        ax.boxplot(data, tick_labels=labels, **kwargs)
    except TypeError:
        ax.boxplot(data, labels=labels, **kwargs)


def integrate_curve(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def tensor_to_numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def model_to_simple_returns(values: np.ndarray, config: Dict) -> np.ndarray:
    transform = str(config.get("return_transform", "simple"))
    if transform == "simple":
        return values
    if transform == "log":
        return np.expm1(np.clip(values, -50.0, 50.0))
    raise ValueError("Unknown return transform: %s" % transform)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot fixed-split CSDI scenario diagnostics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-features", type=int, default=12)
    parser.add_argument("--sample-paths", type=int, default=50)
    parser.add_argument("--topology-window", type=int, default=32)
    parser.add_argument("--topology-stride", type=int, default=4)
    parser.add_argument("--topology-max-points", type=int, default=80)
    parser.add_argument("--topology-samples", type=int, default=50)
    parser.add_argument("--dpi", type=int, default=160)
    return parser.parse_args()


def savefig(path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()


def load_run(run_dir: Path) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Dict]:
    predictions_path = run_dir / "predictions.csv"
    if not predictions_path.exists():
        horizon_paths = sorted(run_dir.glob("horizon_*/predictions.csv"))
        if not horizon_paths:
            raise FileNotFoundError("Missing predictions.csv and horizon_*/predictions.csv")
        predictions = pd.concat((pd.read_csv(path) for path in horizon_paths), ignore_index=True)
    else:
        predictions = pd.read_csv(predictions_path)

    metrics_path = run_dir / "metrics_by_horizon.csv"
    metrics = pd.read_csv(metrics_path) if metrics_path.exists() else None
    config_path = run_dir / "run_config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    return predictions, metrics, config


def sliding_point_cloud_np(
    path: np.ndarray,
    window: int,
    stride: int,
    max_points: int,
) -> Optional[np.ndarray]:
    if path.ndim != 2 or len(path) < window or window <= 1:
        return None
    starts = np.arange(0, len(path) - window + 1, max(1, stride))
    if len(starts) == 0:
        return None
    if len(starts) > max_points:
        starts = starts[np.linspace(0, len(starts) - 1, max_points).astype(int)]
    cloud = np.stack([path[start : start + window].reshape(-1) for start in starts])
    cloud = cloud - cloud.mean(axis=0, keepdims=True)
    scale = cloud.std()
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = 1.0
    return cloud / scale


def pairwise_distances_np(cloud: np.ndarray) -> np.ndarray:
    diff = cloud[:, None, :] - cloud[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=-1))


def mst_edge_lengths(distances: np.ndarray) -> np.ndarray:
    n = distances.shape[0]
    if n <= 1:
        return np.asarray([], dtype=float)
    selected = np.zeros(n, dtype=bool)
    selected[0] = True
    best = distances[0].copy()
    edges = []
    for _ in range(n - 1):
        masked = np.where(selected, np.inf, best)
        j = int(np.argmin(masked))
        if not np.isfinite(masked[j]):
            break
        edges.append(float(masked[j]))
        selected[j] = True
        best = np.minimum(best, distances[j])
    return np.asarray(edges, dtype=float)


def component_count_from_edges(n: int, edges: np.ndarray) -> int:
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in edges:
        ri = find(int(i))
        rj = find(int(j))
        if ri != rj:
            parent[rj] = ri
    return len({find(i) for i in range(n)})


def curve_metrics(distances: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = distances.shape[0]
    upper = distances[np.triu_indices(n, k=1)]
    median = float(np.median(upper)) if len(upper) else 1.0
    if not np.isfinite(median) or median <= 1e-12:
        median = 1.0
    thresholds = np.linspace(0.25, 1.50, 24) * median
    beta1 = []
    recurrence = []
    max_edges = max(1, n * (n - 1) // 2)
    triu = np.triu_indices(n, k=1)
    for eps in thresholds:
        edge_mask = distances[triu] <= eps
        edge_count = int(edge_mask.sum())
        edges = np.column_stack((triu[0][edge_mask], triu[1][edge_mask]))
        components = component_count_from_edges(n, edges)
        beta1.append(max(0, edge_count - n + components) / float(max_edges))
        recurrence.append(edge_count / float(max_edges))
    return thresholds / median, np.asarray(beta1), np.asarray(recurrence)


def path_topology_summary(
    path: np.ndarray,
    window: int,
    stride: int,
    max_points: int,
) -> Optional[Dict[str, object]]:
    cloud = sliding_point_cloud_np(path, window, stride, max_points)
    if cloud is None or len(cloud) < 3:
        return None
    distances = pairwise_distances_np(cloud)
    mst = mst_edge_lengths(distances)
    normalized_thresholds, beta1_curve, recurrence_curve = curve_metrics(distances)
    weights = mst / max(float(mst.sum()), 1e-12)
    h0_entropy = -float(np.sum(weights * np.log(weights + 1e-12))) if len(weights) else 0.0
    market = path.mean(axis=1)
    market = market - market.mean()
    power = np.abs(np.fft.rfft(market)[1:]) ** 2
    lowfreq_ratio = 0.0
    if len(power) > 0 and float(power.sum()) > 1e-12:
        lowfreq_ratio = float(power[: min(10, len(power))].sum() / power.sum())
    return {
        "n_points": int(len(cloud)),
        "h0_total_persistence": float(mst.sum()) if len(mst) else 0.0,
        "h0_max_persistence": float(mst.max()) if len(mst) else 0.0,
        "h0_entropy": h0_entropy,
        "beta1_proxy_area": integrate_curve(beta1_curve, normalized_thresholds),
        "beta1_proxy_max": float(beta1_curve.max()) if len(beta1_curve) else 0.0,
        "recurrence_area": integrate_curve(recurrence_curve, normalized_thresholds),
        "recurrence_entropy": float(
            -np.sum(
                (recurrence_curve / max(float(recurrence_curve.sum()), 1e-12))
                * np.log(recurrence_curve / max(float(recurrence_curve.sum()), 1e-12) + 1e-12)
            )
        ),
        "lowfreq_power_ratio": lowfreq_ratio,
        "thresholds": normalized_thresholds,
        "beta1_curve": beta1_curve,
        "recurrence_curve": recurrence_curve,
    }


def topology_distance(record: Dict[str, object], real_record: Dict[str, object]) -> float:
    keys = [
        "h0_total_persistence",
        "h0_max_persistence",
        "h0_entropy",
        "beta1_proxy_area",
        "beta1_proxy_max",
        "recurrence_area",
        "recurrence_entropy",
        "lowfreq_power_ratio",
    ]
    diffs = []
    for key in keys:
        real = float(real_record[key])
        generated = float(record[key])
        diffs.append((generated - real) / (abs(real) + 1e-6))
    return float(np.sqrt(np.mean(np.asarray(diffs) ** 2)))


def add_diagnostics(predictions: pd.DataFrame) -> pd.DataFrame:
    df = predictions.copy()
    df["error"] = df["pred_median"] - df["actual"]
    df["abs_error"] = df["error"].abs()
    df["sq_error"] = df["error"] ** 2
    df["covered_50"] = df["actual"].between(df["pred_p25"], df["pred_p75"])
    df["covered_90"] = df["actual"].between(df["pred_p05"], df["pred_p95"])
    df["width_50"] = df["pred_p75"] - df["pred_p25"]
    df["width_90"] = df["pred_p95"] - df["pred_p05"]
    if "horizon_days" not in df.columns:
        df["horizon_days"] = df["horizon"].max()
    return df


def plot_metrics_by_scenario_horizon(metrics: Optional[pd.DataFrame], out_dir: Path, dpi: int) -> None:
    if metrics is None or metrics.empty:
        return
    x = metrics["horizon_years"] if "horizon_years" in metrics.columns else metrics["horizon_days"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, metrics["mae"], marker="o", label="MAE")
    ax.plot(x, metrics["rmse"], marker="o", label="RMSE")
    ax.set_xlabel("Scenario horizon")
    ax.set_ylabel("Return error")
    ax.set_title("Point Error by Scenario Horizon")
    ax.grid(alpha=0.25)
    ax.legend()
    savefig(out_dir / "metrics_by_scenario_horizon.png", dpi)


def plot_training_history(run_dir: Path, out_dir: Path, dpi: int) -> None:
    paths = sorted(run_dir.glob("horizon_*/train_history.csv"))
    if not paths:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for path in paths:
        horizon = path.parent.name.replace("horizon_", "")
        history = pd.read_csv(path)
        ax.plot(history["epoch"], history["train_loss"], label="%s days" % int(horizon), alpha=0.85)
        valid = history.dropna(subset=["valid_loss"])
        if not valid.empty:
            ax.scatter(valid["epoch"], valid["valid_loss"], s=20, alpha=0.7)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Diffusion loss")
    ax.set_title("Training History")
    ax.grid(alpha=0.25)
    if len(paths) <= 6:
        ax.legend(fontsize=8)
    savefig(out_dir / "training_history.png", dpi)


def plot_error_by_step(df: pd.DataFrame, out_dir: Path, dpi: int) -> None:
    for horizon_days, item in df.groupby("horizon_days"):
        by_step = item.groupby("horizon").agg(
            mae=("abs_error", "mean"),
            rmse=("sq_error", lambda x: math.sqrt(float(np.mean(x)))),
            bias=("error", "mean"),
        )
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(by_step.index, by_step["mae"], label="MAE")
        ax.plot(by_step.index, by_step["rmse"], label="RMSE")
        ax.plot(by_step.index, by_step["bias"], label="Bias")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Days into generated scenario")
        ax.set_ylabel("Return error")
        ax.set_title("Error Over %d-Day Scenario" % int(horizon_days))
        ax.grid(alpha=0.25)
        ax.legend()
        savefig(out_dir / "error_by_step" / ("horizon_%04d.png" % int(horizon_days)), dpi)


def plot_coverage_by_step(df: pd.DataFrame, out_dir: Path, dpi: int) -> None:
    for horizon_days, item in df.groupby("horizon_days"):
        by_step = item.groupby("horizon").agg(
            coverage_50=("covered_50", "mean"),
            coverage_90=("covered_90", "mean"),
            width_50=("width_50", "mean"),
            width_90=("width_90", "mean"),
        )
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        axes[0].plot(by_step.index, by_step["coverage_50"], label="50% interval")
        axes[0].plot(by_step.index, by_step["coverage_90"], label="90% interval")
        axes[0].axhline(0.50, color="gray", linestyle="--", linewidth=1)
        axes[0].axhline(0.90, color="gray", linestyle="--", linewidth=1)
        axes[0].set_ylim(0, 1)
        axes[0].set_xlabel("Days into generated scenario")
        axes[0].set_ylabel("Empirical coverage")
        axes[0].set_title("Interval Coverage")
        axes[0].grid(alpha=0.25)
        axes[0].legend()
        axes[1].plot(by_step.index, by_step["width_50"], label="50% interval")
        axes[1].plot(by_step.index, by_step["width_90"], label="90% interval")
        axes[1].set_xlabel("Days into generated scenario")
        axes[1].set_ylabel("Average interval width")
        axes[1].set_title("Interval Width")
        axes[1].grid(alpha=0.25)
        axes[1].legend()
        savefig(out_dir / "coverage_by_step" / ("horizon_%04d.png" % int(horizon_days)), dpi)


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
    ax.set_title("Worst Features by MAE")
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
    ax.set_ylabel("Generated median return")
    ax.set_title("Actual vs Generated Median")
    ax.grid(alpha=0.25)
    savefig(out_dir / "actual_vs_generated_median.png", dpi)


def plot_calibration(df: pd.DataFrame, out_dir: Path, dpi: int) -> None:
    quantiles = [
        (0.05, "pred_p05"),
        (0.25, "pred_p25"),
        (0.50, "pred_median"),
        (0.75, "pred_p75"),
        (0.95, "pred_p95"),
    ]
    expected = [q for q, _ in quantiles]
    observed = [float((df["actual"] <= df[col]).mean()) for _, col in quantiles]
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


def plot_cumulative_paths(
    run_dir: Path,
    out_dir: Path,
    dpi: int,
    max_features: int,
    sample_paths: int,
) -> None:
    config_path = run_dir / "run_config.json"
    if not config_path.exists():
        return
    config = json.loads(config_path.read_text())
    features = config.get("features", [])
    nsample = config.get("nsample")
    if not features or nsample is None:
        return

    for horizon_dir in sorted(run_dir.glob("horizon_*")):
        path = horizon_dir / ("generated_outputs_nsample%d.pk" % int(nsample))
        if not path.exists():
            continue
        with path.open("rb") as f:
            samples, target, eval_points, _observed, _time, scaler, mean_scaler = pickle.load(f)
        samples_np = tensor_to_numpy(samples)
        target_np = tensor_to_numpy(target)
        scaler_np = tensor_to_numpy(scaler).reshape(1, 1, 1, -1)
        mean_np = tensor_to_numpy(mean_scaler).reshape(1, 1, 1, -1)
        samples_np = samples_np * scaler_np + mean_np
        target_np = target_np * tensor_to_numpy(scaler).reshape(1, 1, -1) + tensor_to_numpy(mean_scaler).reshape(1, 1, -1)
        samples_np = model_to_simple_returns(samples_np, config)
        target_np = model_to_simple_returns(target_np, config)
        eval_np = tensor_to_numpy(eval_points)
        forecast_positions = np.where(eval_np[0].sum(axis=1) > 0)[0]
        if len(forecast_positions) == 0:
            continue
        horizon_label = horizon_dir.name.replace("horizon_", "")
        for feature_index, feature in enumerate(features[:max_features]):
            paths = samples_np[0, :, :, feature_index][:, forecast_positions]
            actual = target_np[0, forecast_positions, feature_index]
            cumulative_paths = np.cumprod(1.0 + paths, axis=1) - 1.0
            cumulative_actual = np.cumprod(1.0 + actual) - 1.0
            x = np.arange(1, len(forecast_positions) + 1)
            fig, ax = plt.subplots(figsize=(8, 4.5))
            for sample_path in cumulative_paths[: min(len(cumulative_paths), sample_paths)]:
                ax.plot(x, sample_path, color="#4c78a8", alpha=0.12, linewidth=0.8)
            ax.plot(x, np.median(cumulative_paths, axis=0), color="#f58518", linewidth=2, label="Median generated path")
            ax.plot(x, cumulative_actual, color="black", linewidth=2, label="Actual cumulative return")
            ax.axhline(0, color="gray", linewidth=0.8)
            ax.set_xlabel("Days into generated scenario")
            ax.set_ylabel("Cumulative return")
            ax.set_title("%s-Day Generated Cumulative Paths: %s" % (int(horizon_label), feature))
            ax.grid(alpha=0.25)
            ax.legend()
            savefig(
                out_dir / "cumulative_paths" / ("horizon_%s_%s.png" % (horizon_label, feature)),
                dpi,
            )


def compute_topology_diagnostics(
    run_dir: Path,
    out_dir: Path,
    config: Dict,
    window: int,
    stride: int,
    max_points: int,
    max_samples: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    features = config.get("features", [])
    nsample = config.get("nsample")
    if not features or nsample is None:
        return pd.DataFrame(), pd.DataFrame()

    records = []
    curves = []
    for horizon_dir in sorted(run_dir.glob("horizon_*")):
        path = horizon_dir / ("generated_outputs_nsample%d.pk" % int(nsample))
        if not path.exists():
            continue
        horizon_days = int(horizon_dir.name.replace("horizon_", ""))
        with path.open("rb") as f:
            samples, target, eval_points, _observed, _time, scaler, mean_scaler = pickle.load(f)

        samples_np = tensor_to_numpy(samples)
        target_np = tensor_to_numpy(target)
        scaler_np = tensor_to_numpy(scaler).reshape(1, 1, 1, -1)
        mean_np = tensor_to_numpy(mean_scaler).reshape(1, 1, 1, -1)
        samples_np = samples_np * scaler_np + mean_np
        target_np = target_np * tensor_to_numpy(scaler).reshape(1, 1, -1) + tensor_to_numpy(mean_scaler).reshape(1, 1, -1)
        samples_np = model_to_simple_returns(samples_np, config)
        target_np = model_to_simple_returns(target_np, config)
        eval_np = tensor_to_numpy(eval_points)
        forecast_positions = np.where(eval_np[0].sum(axis=1) > 0)[0]
        if len(forecast_positions) == 0:
            continue

        real_path = target_np[0, forecast_positions, :]
        real_summary = path_topology_summary(real_path, window, stride, max_points)
        if real_summary is None:
            continue

        real_record = {
            "horizon_days": horizon_days,
            "path_type": "real",
            "sample_id": -1,
            "topology_distance_to_real": 0.0,
        }
        for key, value in real_summary.items():
            if key not in {"thresholds", "beta1_curve", "recurrence_curve"}:
                real_record[key] = value
        records.append(real_record)
        for threshold, beta1_value, recurrence_value in zip(
            real_summary["thresholds"],
            real_summary["beta1_curve"],
            real_summary["recurrence_curve"],
        ):
            curves.append(
                {
                    "horizon_days": horizon_days,
                    "path_type": "real",
                    "sample_id": -1,
                    "normalized_threshold": float(threshold),
                    "beta1_proxy": float(beta1_value),
                    "recurrence_rate": float(recurrence_value),
                }
            )

        sample_count = min(samples_np.shape[1], max_samples)
        for sample_id in range(sample_count):
            generated_path = samples_np[0, sample_id, forecast_positions, :]
            generated_summary = path_topology_summary(
                generated_path,
                window,
                stride,
                max_points,
            )
            if generated_summary is None:
                continue
            generated_record = {
                "horizon_days": horizon_days,
                "path_type": "generated",
                "sample_id": sample_id,
            }
            for key, value in generated_summary.items():
                if key not in {"thresholds", "beta1_curve", "recurrence_curve"}:
                    generated_record[key] = value
            generated_record["topology_distance_to_real"] = topology_distance(
                generated_record,
                real_record,
            )
            records.append(generated_record)
            for threshold, beta1_value, recurrence_value in zip(
                generated_summary["thresholds"],
                generated_summary["beta1_curve"],
                generated_summary["recurrence_curve"],
            ):
                curves.append(
                    {
                        "horizon_days": horizon_days,
                        "path_type": "generated",
                        "sample_id": sample_id,
                        "normalized_threshold": float(threshold),
                        "beta1_proxy": float(beta1_value),
                        "recurrence_rate": float(recurrence_value),
                    }
                )

    topology_df = pd.DataFrame(records)
    curves_df = pd.DataFrame(curves)
    if not topology_df.empty:
        topology_df.to_csv(out_dir / "topology_metrics.csv", index=False)
    if not curves_df.empty:
        curves_df.to_csv(out_dir / "topology_curves.csv", index=False)
    return topology_df, curves_df


def plot_topology_distances(topology_df: pd.DataFrame, out_dir: Path, dpi: int) -> None:
    generated = topology_df[topology_df["path_type"] == "generated"]
    if generated.empty:
        return
    horizons = sorted(generated["horizon_days"].unique())
    data = [
        generated[generated["horizon_days"] == horizon]["topology_distance_to_real"].values
        for horizon in horizons
    ]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    boxplot_with_labels(ax, data, [str(int(h)) for h in horizons], showfliers=False)
    ax.set_xlabel("Scenario horizon in trading days")
    ax.set_ylabel("Normalized topology distance to real path")
    ax.set_title("Generated Path Topology Distance")
    ax.grid(axis="y", alpha=0.25)
    savefig(out_dir / "topology_distance_by_horizon.png", dpi)


def plot_topology_feature_distributions(topology_df: pd.DataFrame, out_dir: Path, dpi: int) -> None:
    generated = topology_df[topology_df["path_type"] == "generated"]
    real = topology_df[topology_df["path_type"] == "real"]
    if generated.empty or real.empty:
        return
    metrics = [
        ("h0_total_persistence", "H0 Total Persistence"),
        ("beta1_proxy_area", "Betti-1 Proxy Area"),
        ("recurrence_area", "Recurrence Area"),
        ("lowfreq_power_ratio", "Low-Frequency Power Ratio"),
    ]
    horizons = sorted(generated["horizon_days"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (column, title) in zip(axes.flat, metrics):
        data = [generated[generated["horizon_days"] == h][column].values for h in horizons]
        boxplot_with_labels(ax, data, [str(int(h)) for h in horizons], showfliers=False)
        for idx, horizon in enumerate(horizons, start=1):
            real_values = real[real["horizon_days"] == horizon][column]
            if not real_values.empty:
                ax.scatter(idx, real_values.iloc[0], color="black", s=32, zorder=3, label="Real" if idx == 1 else None)
        ax.set_title(title)
        ax.set_xlabel("Horizon days")
        ax.grid(axis="y", alpha=0.25)
    axes.flat[0].legend()
    savefig(out_dir / "topology_feature_distributions.png", dpi)


def plot_topology_curves(curves_df: pd.DataFrame, out_dir: Path, dpi: int) -> None:
    if curves_df.empty:
        return
    for horizon, horizon_curves in curves_df.groupby("horizon_days"):
        real = horizon_curves[horizon_curves["path_type"] == "real"]
        generated = horizon_curves[horizon_curves["path_type"] == "generated"]
        if real.empty or generated.empty:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for ax, column, ylabel in [
            (axes[0], "beta1_proxy", "Betti-1 graph-cycle proxy"),
            (axes[1], "recurrence_rate", "Recurrence rate"),
        ]:
            pivot = generated.pivot_table(
                index="normalized_threshold",
                columns="sample_id",
                values=column,
            ).sort_index()
            x = pivot.index.values
            median = pivot.median(axis=1).values
            lo = pivot.quantile(0.10, axis=1).values
            hi = pivot.quantile(0.90, axis=1).values
            ax.fill_between(x, lo, hi, color="#4c78a8", alpha=0.20, label="Generated 10-90%")
            ax.plot(x, median, color="#4c78a8", linewidth=2, label="Generated median")
            ax.plot(
                real["normalized_threshold"],
                real[column],
                color="black",
                linewidth=2,
                label="Real holdout",
            )
            ax.set_xlabel("Distance threshold / median pairwise distance")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
        fig.suptitle("Topology Curves for %d-Day Scenario" % int(horizon))
        savefig(out_dir / "topology_curves" / ("horizon_%04d.png" % int(horizon)), dpi)


def write_topology_summary(topology_df: pd.DataFrame, out_dir: Path) -> None:
    if topology_df.empty:
        return
    generated = topology_df[topology_df["path_type"] == "generated"]
    summary_rows = []
    for horizon, item in generated.groupby("horizon_days"):
        summary_rows.append(
            {
                "horizon_days": int(horizon),
                "generated_samples": int(len(item)),
                "median_topology_distance_to_real": float(item["topology_distance_to_real"].median()),
                "mean_topology_distance_to_real": float(item["topology_distance_to_real"].mean()),
                "median_h0_total_persistence": float(item["h0_total_persistence"].median()),
                "median_beta1_proxy_area": float(item["beta1_proxy_area"].median()),
                "median_recurrence_area": float(item["recurrence_area"].median()),
                "median_lowfreq_power_ratio": float(item["lowfreq_power_ratio"].median()),
            }
        )
    (out_dir / "topology_summary.json").write_text(json.dumps(summary_rows, indent=2) + "\n")


def write_summary(df: pd.DataFrame, out_dir: Path) -> None:
    summary = {
        "rows": int(len(df)),
        "scenario_horizons": [int(x) for x in sorted(df["horizon_days"].unique())],
        "features": int(df["feature"].nunique()),
        "mae": float(df["abs_error"].mean()),
        "rmse": float(math.sqrt(float(df["sq_error"].mean()))),
        "bias": float(df["error"].mean()),
        "coverage_50": float(df["covered_50"].mean()),
        "coverage_90": float(df["covered_90"].mean()),
        "avg_width_50": float(df["width_50"].mean()),
        "avg_width_90": float(df["width_90"].mean()),
    }
    (out_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    out_dir = (args.output_dir or run_dir / "plots").resolve()
    predictions, metrics, config = load_run(run_dir)
    df = add_diagnostics(predictions)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_summary(df, out_dir)
    plot_metrics_by_scenario_horizon(metrics, out_dir, args.dpi)
    plot_training_history(run_dir, out_dir, args.dpi)
    plot_error_by_step(df, out_dir, args.dpi)
    plot_coverage_by_step(df, out_dir, args.dpi)
    plot_feature_errors(df, out_dir, args.dpi, args.max_features)
    plot_actual_vs_predicted(df, out_dir, args.dpi)
    plot_calibration(df, out_dir, args.dpi)
    plot_cumulative_paths(run_dir, out_dir, args.dpi, args.max_features, args.sample_paths)
    topology_df, curves_df = compute_topology_diagnostics(
        run_dir,
        out_dir,
        config,
        args.topology_window,
        args.topology_stride,
        args.topology_max_points,
        args.topology_samples,
    )
    write_topology_summary(topology_df, out_dir)
    plot_topology_distances(topology_df, out_dir, args.dpi)
    plot_topology_feature_distributions(topology_df, out_dir, args.dpi)
    plot_topology_curves(curves_df, out_dir, args.dpi)

    print("Wrote analysis plots to %s" % out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
