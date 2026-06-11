#!/usr/bin/env python
"""Evaluate rolling multivariate topology in pipeline prediction outputs."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mse342_matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FINDIFFUSION_ROOT = PROJECT_ROOT / "FinDiffusion"
if str(FINDIFFUSION_ROOT) not in sys.path:
    sys.path.insert(0, str(FINDIFFUSION_ROOT))


DISTANCE_KEYS = [
    "h0_total_persistence",
    "h0_max_persistence",
    "h0_entropy",
    "beta1_proxy_area",
    "beta1_proxy_max",
    "recurrence_area",
    "recurrence_entropy",
    "lowfreq_power_ratio",
]

CURVE_METRICS = [
    "topology_distance_to_real",
    "beta1_proxy_area",
    "recurrence_area",
    "lowfreq_power_ratio",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Gidea/Katz-style rolling multivariate topology for real and "
            "synthetic FinDiffusion prediction panels."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("run_dirs", nargs="+", type=Path, help="Pipeline run directories")
    parser.add_argument("--labels", nargs="+", default=None, help="Optional labels for run dirs")
    parser.add_argument("--model-name", default="findiffusion", help="Model subdir to read")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rolling-window", type=int, default=252)
    parser.add_argument("--rolling-stride", type=int, default=21)
    parser.add_argument("--max-cloud-points", type=int, default=80)
    parser.add_argument("--max-samples", type=int, default=25)
    parser.add_argument("--min-common-days", type=int, default=252)
    parser.add_argument(
        "--exact-h1",
        choices=["auto", "on", "off"],
        default="auto",
        help="Compute exact H1 persistence-landscape diagnostics when feasible",
    )
    parser.add_argument(
        "--exact-h1-max-tasks",
        type=int,
        default=300,
        help="Maximum exact H1 point clouds in auto mode",
    )
    parser.add_argument("--h1-landscapes", type=int, default=3)
    parser.add_argument("--h1-grid-points", type=int, default=50)
    parser.add_argument("--dpi", type=int, default=160)
    return parser.parse_args()


def resolve_predictions_path(run_dir: Path, model_name: str) -> Path:
    candidates = [
        run_dir / model_name / "predictions.csv",
        run_dir / "predictions.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No predictions.csv found for {run_dir}")


def run_label(path: Path, labels: Optional[Sequence[str]], index: int) -> str:
    if labels is not None:
        return str(labels[index])
    return path.name


def read_predictions(path: Path) -> pd.DataFrame:
    sample_cols = None
    header = pd.read_csv(path, nrows=0)
    sample_cols = [col for col in header.columns if col.startswith("sample_")]
    required = ["ticker", "target_index", "actual", *sample_cols]
    if "target_date" in header.columns:
        required.append("target_date")
    missing = [col for col in ["ticker", "target_index", "actual"] if col not in header.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    if not sample_cols:
        raise ValueError(f"{path} has no sample_* columns")
    return pd.read_csv(path, usecols=required)


def pivot_panel(frame: pd.DataFrame, value_col: str, tickers: Sequence[str], indices: Sequence[int]) -> pd.DataFrame:
    panel = frame.pivot_table(
        index="target_index",
        columns="ticker",
        values=value_col,
        aggfunc="first",
    )
    panel = panel.reindex(index=list(indices), columns=list(tickers))
    if panel.isna().any().any():
        missing = int(panel.isna().sum().sum())
        raise ValueError(f"Panel for {value_col} has {missing} missing aligned values")
    return panel.astype(float)


def common_tickers_and_indices(frames: Sequence[pd.DataFrame]) -> Tuple[List[str], List[int]]:
    ticker_sets = [set(frame["ticker"].astype(str).unique()) for frame in frames]
    index_sets = [set(frame["target_index"].astype(int).unique()) for frame in frames]
    tickers = sorted(set.intersection(*ticker_sets))
    indices = sorted(set.intersection(*index_sets))
    return tickers, indices


def target_date_map(frames: Sequence[pd.DataFrame], indices: Sequence[int]) -> Dict[int, str]:
    for frame in frames:
        if "target_date" not in frame.columns:
            continue
        pairs = frame.drop_duplicates("target_index").set_index("target_index")["target_date"]
        return {int(idx): str(pairs.get(idx, "")) for idx in indices}
    return {int(idx): "" for idx in indices}


def sample_columns(frame: pd.DataFrame, max_samples: int) -> List[str]:
    cols = sorted(col for col in frame.columns if col.startswith("sample_"))
    return cols[: max(0, min(int(max_samples), len(cols)))]


def integrate_curve(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def standardize_cloud(panel_window: np.ndarray, max_points: int) -> np.ndarray:
    arr = np.asarray(panel_window, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 3:
        return np.empty((0, 0), dtype=np.float64)
    arr = arr - np.nanmean(arr, axis=0, keepdims=True)
    scale = np.nanstd(arr, axis=0, keepdims=True)
    scale[~np.isfinite(scale) | (scale <= 1e-12)] = 1.0
    arr = arr / scale
    arr = np.nan_to_num(arr, copy=False)
    if max_points > 0 and arr.shape[0] > max_points:
        selected = np.linspace(0, arr.shape[0] - 1, max_points, dtype=np.int64)
        arr = arr[selected]
    return arr


def pairwise_distances(cloud: np.ndarray) -> np.ndarray:
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


def component_count(n: int, edges: np.ndarray) -> int:
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


def graph_curves(distances: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = distances.shape[0]
    upper = distances[np.triu_indices(n, k=1)]
    median = float(np.median(upper)) if len(upper) else 1.0
    if not np.isfinite(median) or median <= 1e-12:
        median = 1.0
    thresholds = np.linspace(0.25, 1.50, 24) * median
    triu = np.triu_indices(n, k=1)
    beta1 = []
    recurrence = []
    max_edges = max(1, n * (n - 1) // 2)
    for eps in thresholds:
        edge_mask = distances[triu] <= eps
        edge_count = int(edge_mask.sum())
        edges = np.column_stack((triu[0][edge_mask], triu[1][edge_mask]))
        components = component_count(n, edges)
        beta1.append(max(0, edge_count - n + components) / float(max_edges))
        recurrence.append(edge_count / float(max_edges))
    return thresholds / median, np.asarray(beta1), np.asarray(recurrence)


def path_topology_summary(panel_window: np.ndarray, max_cloud_points: int) -> Dict[str, object]:
    cloud = standardize_cloud(panel_window, max_cloud_points)
    if cloud.shape[0] < 3:
        raise ValueError(f"Need at least 3 point-cloud points, got {cloud.shape[0]}")
    distances = pairwise_distances(cloud)
    mst = mst_edge_lengths(distances)
    thresholds, beta1_curve, recurrence_curve = graph_curves(distances)
    weights = mst / max(float(mst.sum()), 1e-12)
    market = np.asarray(panel_window, dtype=np.float64).mean(axis=1)
    market = market - market.mean()
    power = np.abs(np.fft.rfft(market)[1:]) ** 2
    lowfreq_ratio = 0.0
    if len(power) and float(power.sum()) > 1e-12:
        lowfreq_ratio = float(power[: min(10, len(power))].sum() / power.sum())
    return {
        "n_points": int(cloud.shape[0]),
        "h0_total_persistence": float(mst.sum()) if len(mst) else 0.0,
        "h0_max_persistence": float(mst.max()) if len(mst) else 0.0,
        "h0_entropy": -float(np.sum(weights * np.log(weights + 1e-12))) if len(weights) else 0.0,
        "beta1_proxy_area": integrate_curve(beta1_curve, thresholds),
        "beta1_proxy_max": float(beta1_curve.max()) if len(beta1_curve) else 0.0,
        "recurrence_area": integrate_curve(recurrence_curve, thresholds),
        "recurrence_entropy": recurrence_entropy(recurrence_curve),
        "lowfreq_power_ratio": lowfreq_ratio,
        "thresholds": thresholds,
        "beta1_curve": beta1_curve,
        "recurrence_curve": recurrence_curve,
    }


def recurrence_entropy(curve: np.ndarray) -> float:
    denom = max(float(curve.sum()), 1e-12)
    weights = curve / denom
    return float(-np.sum(weights * np.log(weights + 1e-12)))


def topology_distance(record: Mapping[str, float], real_record: Mapping[str, float]) -> float:
    diffs = []
    for key in DISTANCE_KEYS:
        real = float(real_record[key])
        generated = float(record[key])
        diffs.append((generated - real) / (abs(real) + 1e-6))
    return float(np.sqrt(np.mean(np.asarray(diffs) ** 2)))


class ExactH1:
    def __init__(self, enabled: bool, n_landscapes: int, n_grid_points: int) -> None:
        self.enabled = enabled
        self.n_landscapes = int(n_landscapes)
        self.n_grid_points = int(n_grid_points)
        self._gudhi = None
        self._landscape_fn = None
        if enabled:
            import gudhi
            from src.models.topo_loss import persistence_landscape

            self._gudhi = gudhi
            self._landscape_fn = persistence_landscape

    def landscape(self, cloud: np.ndarray, max_edge_length: float) -> np.ndarray:
        if not self.enabled:
            return np.full((self.n_landscapes, self.n_grid_points), np.nan, dtype=np.float32)
        if cloud.shape[0] < 3:
            return np.zeros((self.n_landscapes, self.n_grid_points), dtype=np.float32)
        t_grid = np.linspace(0.0, max_edge_length, self.n_grid_points, dtype=np.float32)
        rips = self._gudhi.RipsComplex(
            points=cloud.astype(np.float64),
            max_edge_length=float(max_edge_length),
        )
        simplex_tree = rips.create_simplex_tree(max_dimension=2)
        simplex_tree.compute_persistence()
        pairs = [
            (birth, death)
            for dim, (birth, death) in simplex_tree.persistence()
            if dim == 1 and np.isfinite(death)
        ]
        if not pairs:
            return np.zeros((self.n_landscapes, self.n_grid_points), dtype=np.float32)
        import torch

        births = torch.tensor([birth for birth, _death in pairs], dtype=torch.float32)
        deaths = torch.tensor([death for _birth, death in pairs], dtype=torch.float32)
        grid = torch.tensor(t_grid, dtype=torch.float32)
        return self._landscape_fn(births, deaths, grid, self.n_landscapes).cpu().numpy()


def decide_exact_h1(mode: str, task_count: int, max_tasks: int) -> Tuple[bool, str]:
    if mode == "off":
        return False, "disabled by --exact-h1 off"
    try:
        import gudhi  # noqa: F401
        from src.models.topo_loss import persistence_landscape  # noqa: F401
    except ImportError as exc:
        if mode == "on":
            raise RuntimeError(f"--exact-h1 on requested but dependencies are missing: {exc}") from exc
        return False, f"disabled because exact H1 dependencies are missing: {exc}"
    if mode == "auto" and task_count > max_tasks:
        return False, f"disabled in auto mode because {task_count} tasks exceed {max_tasks}"
    return True, "enabled"


def rolling_slices(n_days: int, window: int, stride: int) -> List[Tuple[int, int]]:
    if window <= 0 or stride <= 0:
        raise ValueError("--rolling-window and --rolling-stride must be positive")
    if n_days < window:
        return []
    return [(start, start + window) for start in range(0, n_days - window + 1, stride)]


def compute_run_diagnostics(
    label: str,
    synthetic_panels: Sequence[pd.DataFrame],
    real_panel: pd.DataFrame,
    indices: Sequence[int],
    date_lookup: Mapping[int, str],
    slices: Sequence[Tuple[int, int]],
    max_cloud_points: int,
    exact_h1: ExactH1,
    real_cache: Dict[int, Dict[str, object]],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    records: List[Dict[str, object]] = []
    curve_records: List[Dict[str, object]] = []
    real_values = real_panel.to_numpy(float)

    for window_ordinal, (start, end) in enumerate(slices):
        start_index = int(indices[start])
        end_index = int(indices[end - 1])
        if window_ordinal not in real_cache:
            real_summary = path_topology_summary(real_values[start:end], max_cloud_points)
            real_cloud = standardize_cloud(real_values[start:end], max_cloud_points)
            real_max_edge = estimate_max_edge_length(real_cloud)
            real_landscape = exact_h1.landscape(real_cloud, real_max_edge)
            real_record = base_window_record(
                "real",
                "real",
                -1,
                window_ordinal,
                start_index,
                end_index,
                date_lookup.get(start_index, ""),
                date_lookup.get(end_index, ""),
                real_summary,
            )
            real_record["topology_distance_to_real"] = 0.0
            real_record["h1_landscape_norm"] = vector_norm(real_landscape)
            real_record["h1_landscape_l2_to_real"] = 0.0 if exact_h1.enabled else np.nan
            real_record["h1_landscape_mse_to_real"] = 0.0 if exact_h1.enabled else np.nan
            real_record["_h1_landscape"] = real_landscape
            real_record["_h1_max_edge_length"] = real_max_edge
            real_cache[window_ordinal] = real_record
        real_record = real_cache[window_ordinal]

        for sample_id, panel in enumerate(synthetic_panels):
            synthetic_values = panel.to_numpy(float)
            synthetic_summary = path_topology_summary(synthetic_values[start:end], max_cloud_points)
            record = base_window_record(
                label,
                "synthetic",
                sample_id,
                window_ordinal,
                start_index,
                end_index,
                date_lookup.get(start_index, ""),
                date_lookup.get(end_index, ""),
                synthetic_summary,
            )
            record["topology_distance_to_real"] = topology_distance(record, real_record)
            synthetic_cloud = standardize_cloud(synthetic_values[start:end], max_cloud_points)
            synthetic_landscape = exact_h1.landscape(
                synthetic_cloud,
                float(real_record["_h1_max_edge_length"]),
            )
            record["h1_landscape_norm"] = vector_norm(synthetic_landscape)
            record["h1_landscape_l2_to_real"] = landscape_l2(
                synthetic_landscape,
                real_record["_h1_landscape"],
                exact_h1.enabled,
            )
            record["h1_landscape_mse_to_real"] = landscape_mse(
                synthetic_landscape,
                real_record["_h1_landscape"],
                exact_h1.enabled,
            )
            records.append(record)
            curve_records.extend(curve_rows(record, real_record, label))

    return records, curve_records


def base_window_record(
    run_label: str,
    path_type: str,
    sample_id: int,
    window_ordinal: int,
    start_index: int,
    end_index: int,
    start_date: str,
    end_date: str,
    summary: Mapping[str, object],
) -> Dict[str, object]:
    record: Dict[str, object] = {
        "run_label": run_label,
        "path_type": path_type,
        "sample_id": int(sample_id),
        "window_ordinal": int(window_ordinal),
        "window_start_index": int(start_index),
        "window_end_index": int(end_index),
        "window_start_date": start_date,
        "window_end_date": end_date,
    }
    for key, value in summary.items():
        if key not in {"thresholds", "beta1_curve", "recurrence_curve"}:
            record[key] = value
    return record


def curve_rows(record: Mapping[str, object], real_record: Mapping[str, object], label: str) -> List[Dict[str, object]]:
    rows = []
    for metric in CURVE_METRICS + ["h1_landscape_l2_to_real"]:
        if metric not in record:
            continue
        rows.append(
            {
                "run_label": label,
                "sample_id": int(record["sample_id"]),
                "window_ordinal": int(record["window_ordinal"]),
                "window_start_index": int(record["window_start_index"]),
                "window_end_index": int(record["window_end_index"]),
                "window_start_date": record.get("window_start_date", ""),
                "window_end_date": record.get("window_end_date", ""),
                "metric": metric,
                "real_value": float(real_record.get(metric, 0.0)),
                "synthetic_value": float(record[metric]),
            }
        )
    return rows


def estimate_max_edge_length(cloud: np.ndarray) -> float:
    if cloud.shape[0] < 2:
        return 1.0
    distances = pairwise_distances(cloud)
    upper = distances[np.triu_indices(distances.shape[0], k=1)]
    if not len(upper):
        return 1.0
    value = float(np.percentile(upper, 90))
    return value if np.isfinite(value) and value > 1e-8 else 1.0


def vector_norm(value: np.ndarray) -> float:
    if not np.isfinite(value).all():
        return float("nan")
    return float(np.linalg.norm(value))


def landscape_l2(value: np.ndarray, real: np.ndarray, enabled: bool) -> float:
    if not enabled:
        return float("nan")
    return float(np.linalg.norm(value - real))


def landscape_mse(value: np.ndarray, real: np.ndarray, enabled: bool) -> float:
    if not enabled:
        return float("nan")
    diff = value - real
    return float(np.mean(diff * diff))


def build_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    generated = metrics[metrics["path_type"] == "synthetic"].copy()
    if generated.empty:
        return pd.DataFrame()
    for key in DISTANCE_KEYS:
        generated[f"{key}_abs_gap"] = (generated[key] - generated[f"real_{key}"]).abs()
    aggregations = {
        "topology_distance_to_real": ["median", "mean", "std", "min", "max"],
        "beta1_proxy_area_abs_gap": ["median", "mean"],
        "recurrence_area_abs_gap": ["median", "mean"],
        "lowfreq_power_ratio_abs_gap": ["median", "mean"],
        "h1_landscape_l2_to_real": ["median", "mean"],
        "sample_id": pd.Series.nunique,
        "window_ordinal": pd.Series.nunique,
    }
    summary = generated.groupby("run_label", sort=True).agg(aggregations)
    summary.columns = [
        "n_samples" if col[0] == "sample_id" else "n_windows" if col[0] == "window_ordinal" else "_".join(col).rstrip("_")
        for col in summary.columns.to_flat_index()
    ]
    summary = summary.reset_index()
    return summary.sort_values("topology_distance_to_real_median").reset_index(drop=True)


def add_real_reference_columns(metrics: pd.DataFrame, real_rows: pd.DataFrame) -> pd.DataFrame:
    real_lookup = real_rows.set_index("window_ordinal")
    output = metrics.copy()
    for key in DISTANCE_KEYS:
        output[f"real_{key}"] = output["window_ordinal"].map(real_lookup[key])
    return output


def build_curve_summary(curves: pd.DataFrame) -> pd.DataFrame:
    if curves.empty:
        return curves
    grouped = curves.groupby(
        [
            "run_label",
            "window_ordinal",
            "window_start_index",
            "window_end_index",
            "window_start_date",
            "window_end_date",
            "metric",
        ],
        dropna=False,
    )
    return grouped.agg(
        real_value=("real_value", "first"),
        synthetic_median=("synthetic_value", "median"),
        synthetic_q10=("synthetic_value", lambda x: float(x.quantile(0.10))),
        synthetic_q90=("synthetic_value", lambda x: float(x.quantile(0.90))),
    ).reset_index()


def write_report(summary: pd.DataFrame, output_dir: Path, metadata: Mapping[str, object]) -> None:
    lines = [
        "# Business-Cycle Topology Diagnostic",
        "",
        "This diagnostic aligns ticker panels from FinDiffusion prediction outputs and compares rolling multivariate topology between real returns and pseudo-joint synthetic sample panels.",
        "",
        "## Configuration",
        "",
        f"- Rolling window: `{metadata['rolling_window']}` trading days",
        f"- Rolling stride: `{metadata['rolling_stride']}` trading days",
        f"- Common tickers: `{metadata['n_tickers']}`",
        f"- Common target days: `{metadata['n_common_days']}`",
        f"- Synthetic samples per run: `{metadata['max_samples']}`",
        f"- Exact H1: `{metadata['exact_h1_status']}`",
        "",
        "## Ranking",
        "",
    ]
    if summary.empty:
        lines.append("No synthetic topology rows were generated.")
    else:
        display_cols = [
            "run_label",
            "topology_distance_to_real_median",
            "beta1_proxy_area_abs_gap_median",
            "lowfreq_power_ratio_abs_gap_median",
            "h1_landscape_l2_to_real_median",
        ]
        display_cols = [col for col in display_cols if col in summary.columns]
        lines.append(markdown_table(summary[display_cols]))
        best = summary.iloc[0]
        lines.extend(
            [
                "",
                "Lower topology distance means the synthetic rolling-window topology is closer to the real multivariate market panel under the selected proxy metrics.",
                f"The best run by median topology distance is `{best['run_label']}`.",
            ]
        )
    lines.extend(
        [
            "",
            "## Caveat",
            "",
            "Current FinDiffusion outputs are per-ticker forecast samples. Aligning `sample_000` across tickers creates pseudo-joint synthetic market panels, not native joint 30-asset samples.",
        ]
    )
    (output_dir / "business_cycle_topology_report.md").write_text("\n".join(lines) + "\n")


def markdown_table(frame: pd.DataFrame) -> str:
    headers = list(frame.columns)
    rows = []
    for _, row in frame.iterrows():
        values = []
        for col in headers:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.4g}")
            else:
                values.append(str(value))
        rows.append(values)
    widths = [
        max(len(header), *(len(row[i]) for row in rows)) if rows else len(header)
        for i, header in enumerate(headers)
    ]
    header = "| " + " | ".join(name.ljust(widths[i]) for i, name in enumerate(headers)) + " |"
    sep = "| " + " | ".join("-" * width for width in widths) + " |"
    body = [
        "| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |"
        for row in rows
    ]
    return "\n".join([header, sep, *body])


def plot_outputs(metrics: pd.DataFrame, curves: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    generated = metrics[metrics["path_type"] == "synthetic"].copy()
    if not generated.empty:
        labels = list(generated["run_label"].drop_duplicates())
        values = [
            generated.loc[generated["run_label"] == label, "topology_distance_to_real"].dropna().to_numpy()
            for label in labels
        ]
        fig, ax = plt.subplots(figsize=(max(7, 1.2 * len(labels)), 4.5))
        try:
            ax.boxplot(values, tick_labels=labels, showfliers=False)
        except TypeError:
            ax.boxplot(values, labels=labels, showfliers=False)
        ax.set_ylabel("Distance to real")
        ax.set_title("Rolling Business-Cycle Topology Distance")
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=0.25)
        savefig(plot_dir / "topology_distance_boxplot.png", dpi)

    if not curves.empty:
        for metric in ["topology_distance_to_real", "beta1_proxy_area", "lowfreq_power_ratio"]:
            item = curves[curves["metric"] == metric]
            if item.empty:
                continue
            fig, ax = plt.subplots(figsize=(9, 4.8))
            for label, group in item.groupby("run_label", sort=True):
                x = group["window_end_index"].to_numpy()
                ax.plot(x, group["synthetic_median"], label=label, linewidth=1.8)
                ax.fill_between(
                    x,
                    group["synthetic_q10"].to_numpy(),
                    group["synthetic_q90"].to_numpy(),
                    alpha=0.12,
                )
            if metric != "topology_distance_to_real":
                first = item.drop_duplicates("window_end_index").sort_values("window_end_index")
                ax.plot(
                    first["window_end_index"],
                    first["real_value"],
                    color="black",
                    linewidth=2.2,
                    label="real",
                )
            ax.set_xlabel("Target index")
            ax.set_ylabel(metric)
            ax.set_title(metric.replace("_", " ").title())
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
            savefig(plot_dir / f"{metric}_trajectory.png", dpi)


def savefig(path: Path, dpi: int) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()


def main() -> None:
    args = parse_args()
    if args.labels is not None and len(args.labels) != len(args.run_dirs):
        raise SystemExit("--labels must have the same length as run_dirs")
    frames = []
    prediction_paths = []
    for run_dir in args.run_dirs:
        path = resolve_predictions_path(run_dir, args.model_name)
        prediction_paths.append(path)
        frames.append(read_predictions(path))

    tickers, indices = common_tickers_and_indices(frames)
    if len(indices) < int(args.min_common_days):
        raise SystemExit(
            f"Only {len(indices)} common target days remain; need at least {args.min_common_days}"
        )
    slices = rolling_slices(len(indices), int(args.rolling_window), int(args.rolling_stride))
    if not slices:
        raise SystemExit(
            f"No rolling windows for {len(indices)} days with window={args.rolling_window}"
        )
    date_lookup = target_date_map(frames, indices)
    real_panel = pivot_panel(frames[0], "actual", tickers, indices)
    total_exact_tasks = len(slices) * (1 + len(args.run_dirs) * max(0, min(args.max_samples, 25)))
    exact_enabled, exact_status = decide_exact_h1(
        args.exact_h1,
        total_exact_tasks,
        int(args.exact_h1_max_tasks),
    )
    exact_h1 = ExactH1(exact_enabled, args.h1_landscapes, args.h1_grid_points)

    all_records: List[Dict[str, object]] = []
    all_curves: List[Dict[str, object]] = []
    real_cache: Dict[int, Dict[str, object]] = {}
    for i, (run_dir, frame) in enumerate(zip(args.run_dirs, frames)):
        label = run_label(run_dir, args.labels, i)
        panels = [
            pivot_panel(frame, col, tickers, indices)
            for col in sample_columns(frame, int(args.max_samples))
        ]
        records, curves = compute_run_diagnostics(
            label=label,
            synthetic_panels=panels,
            real_panel=real_panel,
            indices=indices,
            date_lookup=date_lookup,
            slices=slices,
            max_cloud_points=int(args.max_cloud_points),
            exact_h1=exact_h1,
            real_cache=real_cache,
        )
        all_records.extend(records)
        all_curves.extend(curves)

    real_rows = [
        {key: value for key, value in row.items() if not key.startswith("_")}
        for row in real_cache.values()
    ]
    metrics = pd.DataFrame(real_rows + all_records)
    metrics = metrics.drop(columns=[col for col in metrics.columns if col.startswith("_")], errors="ignore")
    real_metrics = metrics[metrics["path_type"] == "real"]
    metrics = add_real_reference_columns(metrics, real_metrics)
    curves = build_curve_summary(pd.DataFrame(all_curves))
    summary = build_summary(metrics)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output_dir / "business_cycle_topology_metrics.csv", index=False)
    curves.to_csv(args.output_dir / "business_cycle_topology_curves.csv", index=False)
    summary.to_csv(args.output_dir / "business_cycle_topology_summary.csv", index=False)
    metadata = {
        "run_dirs": [str(path) for path in args.run_dirs],
        "prediction_paths": [str(path) for path in prediction_paths],
        "labels": [run_label(path, args.labels, i) for i, path in enumerate(args.run_dirs)],
        "n_tickers": len(tickers),
        "tickers": tickers,
        "n_common_days": len(indices),
        "rolling_window": int(args.rolling_window),
        "rolling_stride": int(args.rolling_stride),
        "n_windows": len(slices),
        "max_cloud_points": int(args.max_cloud_points),
        "max_samples": int(args.max_samples),
        "exact_h1_status": exact_status,
    }
    (args.output_dir / "run_config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    write_report(summary, args.output_dir, metadata)
    plot_outputs(metrics, curves, args.output_dir, int(args.dpi))
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
