#!/usr/bin/env python3
"""Walk-forward training and prediction pipeline for CSDI forecasting.

Each fold fits normalizers on the expanding training window, trains a fresh
CSDI forecasting model using only windows that end before the fold origin, and
predicts the next horizon from the latest available history.
"""

import argparse
import datetime as dt
import json
import pickle
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CSDI_ROOT = PROJECT_ROOT / "CSDI"

RUNTIME_IMPORT_ERROR = None  # type: Optional[ImportError]
try:
    import numpy as np
    import pandas as pd
    import torch
    import yaml
    from torch.utils.data import DataLoader, Dataset

    if str(CSDI_ROOT) not in sys.path:
        sys.path.insert(0, str(CSDI_ROOT))
    from main_model import CSDI_Forecasting  # noqa: E402
    from utils import train as train_csdi  # noqa: E402
except ImportError as exc:
    RUNTIME_IMPORT_ERROR = exc
    np = None  # type: ignore[assignment]
    pd = None  # type: ignore[assignment]
    torch = None  # type: ignore[assignment]
    yaml = None  # type: ignore[assignment]
    DataLoader = object  # type: ignore[assignment,misc]
    Dataset = object  # type: ignore[assignment,misc]
    CSDI_Forecasting = object  # type: ignore[assignment,misc]
    train_csdi = None  # type: ignore[assignment]


@dataclass(frozen=True)
class FoldSpec:
    fold: int
    train_end: int
    forecast_start: int
    forecast_end: int
    train_windows: int
    valid_windows: int


class WindowedForecastDataset(Dataset):
    """CSDI-compatible forecasting windows for a single walk-forward fold."""

    def __init__(
        self,
        values: np.ndarray,
        mask: np.ndarray,
        starts: Iterable[int],
        history_length: int,
        pred_length: int,
    ) -> None:
        self.values = values.astype(np.float32, copy=False)
        self.mask = mask.astype(np.float32, copy=False)
        self.starts = np.asarray(list(starts), dtype=np.int64)
        self.history_length = int(history_length)
        self.pred_length = int(pred_length)
        self.seq_length = self.history_length + self.pred_length

        if self.values.shape != self.mask.shape:
            raise ValueError("values and mask must have the same shape")
        if self.values.ndim != 2:
            raise ValueError("values must be a 2D array shaped (time, features)")
        if np.any(self.starts < 0):
            raise ValueError("window starts must be nonnegative")
        if np.any(self.starts + self.seq_length > len(self.values)):
            raise ValueError("window starts exceed the available data length")

    def __len__(self) -> int:
        return int(len(self.starts))

    def __getitem__(self, item: int) -> Dict[str, np.ndarray]:
        start = int(self.starts[item])
        stop = start + self.seq_length

        observed_data = self.values[start:stop]
        observed_mask = self.mask[start:stop]
        gt_mask = observed_mask.copy()
        gt_mask[-self.pred_length :] = 0.0

        return {
            "observed_data": observed_data,
            "observed_mask": observed_mask,
            "gt_mask": gt_mask,
            "timepoints": np.arange(self.seq_length, dtype=np.float32),
            "feature_id": np.arange(self.values.shape[1], dtype=np.float32),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and predict CSDI with an expanding walk-forward split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_ROOT / "data/processed/french49_daily_returns.csv",
        help="CSV containing a date column plus one or more numeric series.",
    )
    parser.add_argument("--date-column", default="date")
    parser.add_argument(
        "--target-columns",
        nargs="+",
        default=None,
        help="Columns to forecast. By default all non-date columns are used.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CSDI_ROOT / "config/base_forecasting.yaml",
        help="Base CSDI YAML config.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for run artifacts. Defaults to CSDI_Experiment/outputs/walk_forward_TIMESTAMP.",
    )
    parser.add_argument("--history-length", type=int, default=231)
    parser.add_argument("--pred-length", type=int, default=21)
    parser.add_argument(
        "--initial-train-size",
        type=int,
        default=None,
        help="Rows available before the first forecast origin. Defaults to 70%% of the data.",
    )
    parser.add_argument(
        "--step-size",
        type=int,
        default=None,
        help="Rows to move the forecast origin after each fold. Defaults to pred-length.",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=3,
        help="Number of folds to run. Use 0 or a negative value to run all possible folds.",
    )
    parser.add_argument(
        "--fold-start",
        type=int,
        default=0,
        help="First fold index to run after fold definitions are created.",
    )
    parser.add_argument(
        "--fold-count",
        type=int,
        default=None,
        help="Number of folds to run from fold-start. Defaults to all selected folds.",
    )
    parser.add_argument("--train-stride", type=int, default=1)
    parser.add_argument("--valid-windows", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--itr-per-epoch", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--valid-epoch-interval", type=int, default=20)
    parser.add_argument("--nsample", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Load fold model.pth files from output-dir and only run prediction.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> str:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print(f"Requested {requested}, but CUDA is not available. Falling back to CPU.")
        return "cpu"
    if requested == "mps" and not torch.backends.mps.is_available():
        print("Requested mps, but MPS is not available. Falling back to CPU.")
        return "cpu"
    return requested


def require_runtime_dependencies() -> None:
    if RUNTIME_IMPORT_ERROR is None:
        return
    raise SystemExit(
        "Missing CSDI runtime dependency: "
        f"{RUNTIME_IMPORT_ERROR}\n"
        "Install the CSDI requirements, for example:\n"
        "  ../venv/bin/pip3 install -r CSDI/requirements.txt"
    )


def as_int(value: object) -> int:
    return int(float(value))


def load_config(path: Path, args: argparse.Namespace) -> dict:
    with path.open("r") as f:
        config = yaml.safe_load(f)

    config["model"]["is_unconditional"] = int(config["model"].get("is_unconditional", 0))
    config["model"]["target_strategy"] = "test"

    if args.batch_size is not None:
        config["train"]["batch_size"] = int(args.batch_size)
    if args.epochs is not None:
        config["train"]["epochs"] = int(args.epochs)
    if args.itr_per_epoch is not None:
        config["train"]["itr_per_epoch"] = int(args.itr_per_epoch)
    if args.lr is not None:
        config["train"]["lr"] = float(args.lr)

    config["train"]["epochs"] = as_int(config["train"]["epochs"])
    config["train"]["batch_size"] = as_int(config["train"]["batch_size"])
    config["train"]["itr_per_epoch"] = as_int(config["train"]["itr_per_epoch"])
    config["train"]["lr"] = float(config["train"]["lr"])
    return config


def load_timeseries(
    csv_path: Path,
    date_column: str,
    target_columns: Optional[List[str]],
) -> Tuple[Optional[pd.Series], List[str], np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    if date_column in df.columns:
        dates = pd.to_datetime(df[date_column])
    else:
        dates = None

    if target_columns is None:
        excluded = {date_column} if date_column in df.columns else set()
        target_columns = [col for col in df.columns if col not in excluded]

    missing = [col for col in target_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Target columns are missing from {csv_path}: {missing}")

    values = df[target_columns].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    mask = np.isfinite(values).astype(np.float32)
    if values.shape[1] == 0:
        raise ValueError("No target columns selected")
    if len(values) == 0:
        raise ValueError("Input data is empty")
    return dates, target_columns, values, mask


def fit_scaler(values: np.ndarray, mask: np.ndarray, train_end: int) -> Tuple[np.ndarray, np.ndarray]:
    train_values = values[:train_end]
    train_mask = mask[:train_end].astype(bool)
    masked = np.where(train_mask, train_values, np.nan)

    mean = np.nanmean(masked, axis=0)
    std = np.nanstd(masked, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0).astype(np.float32)
    std = np.where(np.isfinite(std) & (std > 0), std, 1.0).astype(np.float32)
    return mean, std


def standardize(values: np.ndarray, mask: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    scaled = (values - mean) / std
    scaled = np.where(mask.astype(bool), scaled, 0.0)
    return scaled.astype(np.float32)


def default_initial_train_size(total_rows: int, history_length: int, pred_length: int) -> int:
    minimum = history_length + pred_length
    return max(minimum, int(total_rows * 0.70))


def make_fold_specs(
    total_rows: int,
    history_length: int,
    pred_length: int,
    initial_train_size: Optional[int],
    step_size: Optional[int],
    n_folds: int,
    train_stride: int,
    valid_windows: int,
) -> List[FoldSpec]:
    if history_length <= 0 or pred_length <= 0:
        raise ValueError("history-length and pred-length must be positive")
    if train_stride <= 0:
        raise ValueError("train-stride must be positive")

    initial_train_size = initial_train_size or default_initial_train_size(
        total_rows, history_length, pred_length
    )
    step_size = step_size or pred_length
    if step_size <= 0:
        raise ValueError("step-size must be positive")

    seq_length = history_length + pred_length
    min_train_end = seq_length
    max_train_end = total_rows - pred_length
    if initial_train_size < min_train_end:
        raise ValueError(
            f"initial-train-size must be at least history+pred ({seq_length})"
        )
    if initial_train_size > max_train_end:
        raise ValueError(
            "initial-train-size leaves no complete forecast horizon. "
            f"Need <= {max_train_end}, got {initial_train_size}."
        )

    specs = []  # type: List[FoldSpec]
    train_end = initial_train_size
    fold = 0
    while train_end <= max_train_end:
        all_train_starts = np.arange(0, train_end - seq_length + 1, train_stride)
        if len(all_train_starts) == 0:
            raise ValueError(f"No training windows available for train_end={train_end}")

        fold_valid_windows = min(max(valid_windows, 0), max(0, len(all_train_starts) - 1))
        fold_train_windows = len(all_train_starts) - fold_valid_windows
        specs.append(
            FoldSpec(
                fold=fold,
                train_end=train_end,
                forecast_start=train_end,
                forecast_end=train_end + pred_length,
                train_windows=int(fold_train_windows),
                valid_windows=int(fold_valid_windows),
            )
        )

        fold += 1
        if n_folds > 0 and fold >= n_folds:
            break
        train_end += step_size

    return specs


def make_train_valid_starts(
    train_end: int,
    history_length: int,
    pred_length: int,
    train_stride: int,
    valid_windows: int,
) -> Tuple[np.ndarray, np.ndarray]:
    seq_length = history_length + pred_length
    all_starts = np.arange(0, train_end - seq_length + 1, train_stride)
    fold_valid_windows = min(max(valid_windows, 0), max(0, len(all_starts) - 1))
    if fold_valid_windows == 0:
        return all_starts, np.asarray([], dtype=np.int64)
    return all_starts[:-fold_valid_windows], all_starts[-fold_valid_windows:]


def make_loader(
    dataset: WindowedForecastDataset,
    batch_size: int,
    shuffle: bool,
    device: str,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=device.startswith("cuda"),
    )


def format_date(dates: Optional[pd.Series], index: int) -> str:
    if dates is None:
        return str(index)
    return pd.Timestamp(dates.iloc[index]).date().isoformat()


def predict_fold(
    model: CSDI_Forecasting,
    loader: DataLoader,
    nsample: int,
    mean: np.ndarray,
    std: np.ndarray,
    dates: Optional[pd.Series],
    features: List[str],
    fold: int,
    origin_index: int,
    pred_length: int,
    fold_dir: Path,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    model.eval()
    prediction_rows = []  # type: List[Dict[str, Any]]
    abs_errors = []  # type: List[float]
    sq_errors = []  # type: List[float]

    all_generated_samples = []
    all_target = []
    all_evalpoint = []
    all_observed_point = []
    all_observed_time = []

    scaler = torch.from_numpy(std).to(model.device).float()
    mean_scaler = torch.from_numpy(mean).to(model.device).float()

    with torch.no_grad():
        for batch in loader:
            samples, target, eval_points, observed_points, observed_time = model.evaluate(
                batch, nsample
            )
            samples = samples.permute(0, 1, 3, 2)
            target = target.permute(0, 2, 1)
            eval_points = eval_points.permute(0, 2, 1)
            observed_points = observed_points.permute(0, 2, 1)

            all_generated_samples.append(samples.detach().cpu())
            all_target.append(target.detach().cpu())
            all_evalpoint.append(eval_points.detach().cpu())
            all_observed_point.append(observed_points.detach().cpu())
            all_observed_time.append(observed_time.detach().cpu())

            samples_np = samples.detach().cpu().numpy()
            target_np = target.detach().cpu().numpy()
            eval_np = eval_points.detach().cpu().numpy()

            samples_np = samples_np * std.reshape(1, 1, 1, -1) + mean.reshape(1, 1, 1, -1)
            target_np = target_np * std.reshape(1, 1, -1) + mean.reshape(1, 1, -1)

            for batch_item in range(samples_np.shape[0]):
                for horizon in range(1, pred_length + 1):
                    seq_pos = samples_np.shape[2] - pred_length + horizon - 1
                    target_index = origin_index + horizon - 1
                    for feature_index, feature in enumerate(features):
                        if eval_np[batch_item, seq_pos, feature_index] <= 0:
                            continue
                        draws = samples_np[batch_item, :, seq_pos, feature_index]
                        actual = float(target_np[batch_item, seq_pos, feature_index])
                        median = float(np.median(draws))
                        mean_pred = float(np.mean(draws))
                        error = median - actual
                        abs_errors.append(abs(error))
                        sq_errors.append(error * error)

                        prediction_rows.append(
                            {
                                "fold": fold,
                                "origin_index": origin_index,
                                "origin_date": format_date(dates, origin_index - 1),
                                "target_index": target_index,
                                "target_date": format_date(dates, target_index),
                                "horizon": horizon,
                                "feature": feature,
                                "actual": actual,
                                "pred_median": median,
                                "pred_mean": mean_pred,
                                "pred_p05": float(np.quantile(draws, 0.05)),
                                "pred_p25": float(np.quantile(draws, 0.25)),
                                "pred_p75": float(np.quantile(draws, 0.75)),
                                "pred_p95": float(np.quantile(draws, 0.95)),
                                "n_samples": nsample,
                            }
                        )

    predictions = pd.DataFrame(prediction_rows)
    metrics = {
        "fold": fold,
        "origin_index": origin_index,
        "eval_points": len(abs_errors),
        "mae": float(np.mean(abs_errors)) if abs_errors else float("nan"),
        "rmse": float(np.sqrt(np.mean(sq_errors))) if sq_errors else float("nan"),
    }

    with (fold_dir / f"generated_outputs_nsample{nsample}.pk").open("wb") as f:
        pickle.dump(
            [
                torch.cat(all_generated_samples, dim=0),
                torch.cat(all_target, dim=0),
                torch.cat(all_evalpoint, dim=0),
                torch.cat(all_observed_point, dim=0),
                torch.cat(all_observed_time, dim=0),
                scaler.detach().cpu(),
                mean_scaler.detach().cpu(),
            ],
            f,
        )

    return predictions, metrics


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def run_fold(
    spec: FoldSpec,
    args: argparse.Namespace,
    base_config: Dict[str, Any],
    values: np.ndarray,
    mask: np.ndarray,
    dates: Optional[pd.Series],
    features: List[str],
    output_dir: Path,
    device: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    fold_dir = output_dir / f"fold_{spec.fold:03d}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed + spec.fold)
    mean, std = fit_scaler(values, mask, spec.train_end)
    scaled_values = standardize(values, mask, mean, std)

    train_starts, valid_starts = make_train_valid_starts(
        spec.train_end,
        args.history_length,
        args.pred_length,
        args.train_stride,
        args.valid_windows,
    )
    train_dataset = WindowedForecastDataset(
        scaled_values,
        mask,
        train_starts,
        args.history_length,
        args.pred_length,
    )
    valid_dataset = (
        WindowedForecastDataset(
            scaled_values,
            mask,
            valid_starts,
            args.history_length,
            args.pred_length,
        )
        if len(valid_starts) > 0
        else None
    )

    test_start = spec.train_end - args.history_length
    test_dataset = WindowedForecastDataset(
        scaled_values,
        mask,
        [test_start],
        args.history_length,
        args.pred_length,
    )

    train_loader = make_loader(
        train_dataset,
        base_config["train"]["batch_size"],
        shuffle=True,
        device=device,
    )
    valid_loader = (
        make_loader(valid_dataset, base_config["train"]["batch_size"], shuffle=False, device=device)
        if valid_dataset is not None
        else None
    )
    test_loader = make_loader(test_dataset, batch_size=1, shuffle=False, device=device)

    config = json.loads(json.dumps(base_config))
    target_dim = len(features)
    model = CSDI_Forecasting(config, device, target_dim).to(device)

    write_json(
        fold_dir / "fold_config.json",
        {
            "fold": asdict(spec),
            "features": features,
            "history_length": args.history_length,
            "pred_length": args.pred_length,
            "train_starts": {
                "first": int(train_starts[0]),
                "last": int(train_starts[-1]),
                "count": int(len(train_starts)),
            },
            "valid_starts": {
                "first": int(valid_starts[0]) if len(valid_starts) else None,
                "last": int(valid_starts[-1]) if len(valid_starts) else None,
                "count": int(len(valid_starts)),
            },
            "csdi_config": config,
        },
    )
    np.savez(fold_dir / "scaler.npz", mean=mean, std=std, features=np.asarray(features))

    model_path = fold_dir / "model.pth"
    if args.skip_training:
        if not model_path.exists():
            raise FileNotFoundError(f"--skip-training requested but {model_path} does not exist")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        train_csdi(
            model,
            config["train"],
            train_loader,
            valid_loader=valid_loader,
            valid_epoch_interval=args.valid_epoch_interval,
            foldername=str(fold_dir),
        )

    model.target_dim = target_dim
    predictions, metrics = predict_fold(
        model,
        test_loader,
        args.nsample,
        mean,
        std,
        dates,
        features,
        spec.fold,
        spec.train_end,
        args.pred_length,
        fold_dir,
    )

    predictions.to_csv(fold_dir / "predictions.csv", index=False)
    write_json(fold_dir / "metrics.json", metrics)
    return predictions, metrics


def make_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return PROJECT_ROOT / "CSDI_Experiment" / "outputs" / f"walk_forward_{stamp}"


def select_fold_specs(
    specs: List[FoldSpec],
    fold_start: int,
    fold_count: Optional[int],
) -> List[FoldSpec]:
    if fold_start < 0:
        raise ValueError("fold-start must be nonnegative")
    if fold_count is not None and fold_count < 0:
        raise ValueError("fold-count must be nonnegative")
    if fold_start >= len(specs):
        return []
    stop = None if fold_count is None else fold_start + fold_count
    return specs[fold_start:stop]


def main() -> int:
    args = parse_args()
    require_runtime_dependencies()
    device = resolve_device(args.device)
    output_dir = make_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    dates, features, values, mask = load_timeseries(
        args.data,
        args.date_column,
        args.target_columns,
    )
    config = load_config(args.config, args)
    all_specs = make_fold_specs(
        total_rows=len(values),
        history_length=args.history_length,
        pred_length=args.pred_length,
        initial_train_size=args.initial_train_size,
        step_size=args.step_size,
        n_folds=args.n_folds,
        train_stride=args.train_stride,
        valid_windows=args.valid_windows,
    )
    specs = select_fold_specs(all_specs, args.fold_start, args.fold_count)
    if not specs:
        raise SystemExit(
            f"No folds selected. Available folds: {len(all_specs)}, "
            f"fold-start={args.fold_start}, fold-count={args.fold_count}."
        )

    run_config = {
        "data": str(args.data),
        "date_column": args.date_column,
        "features": features,
        "rows": int(len(values)),
        "history_length": args.history_length,
        "pred_length": args.pred_length,
        "device": device,
        "seed": args.seed,
        "nsample": args.nsample,
        "total_available_folds": len(all_specs),
        "fold_start": args.fold_start,
        "fold_count": args.fold_count,
        "folds": [asdict(spec) for spec in specs],
        "csdi_config": config,
    }
    write_json(output_dir / "run_config.json", run_config)

    print(f"Writing walk-forward CSDI outputs to {output_dir}")
    print(f"Running {len(specs)} fold(s) on {len(values)} rows x {len(features)} features")

    all_predictions = []  # type: List[pd.DataFrame]
    all_metrics = []  # type: List[Dict[str, Any]]
    for spec in specs:
        print(
            f"\nFold {spec.fold}: train rows [0, {spec.train_end}), "
            f"predict rows [{spec.forecast_start}, {spec.forecast_end})"
        )
        predictions, metrics = run_fold(
            spec,
            args,
            config,
            values,
            mask,
            dates,
            features,
            output_dir,
            device,
        )
        all_predictions.append(predictions)
        all_metrics.append(metrics)
        print(
            f"Fold {spec.fold} metrics: "
            f"MAE={metrics['mae']:.6g}, RMSE={metrics['rmse']:.6g}, "
            f"eval_points={int(metrics['eval_points'])}"
        )

        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    predictions_df = pd.concat(all_predictions, ignore_index=True)
    metrics_df = pd.DataFrame(all_metrics)
    predictions_df.to_csv(output_dir / "predictions.csv", index=False)
    metrics_df.to_csv(output_dir / "metrics_by_fold.csv", index=False)

    summary = {
        "folds": int(len(metrics_df)),
        "eval_points": int(metrics_df["eval_points"].sum()),
        "mean_fold_mae": float(metrics_df["mae"].mean()),
        "mean_fold_rmse": float(metrics_df["rmse"].mean()),
        "predictions_csv": str(output_dir / "predictions.csv"),
        "metrics_by_fold_csv": str(output_dir / "metrics_by_fold.csv"),
    }
    write_json(output_dir / "summary.json", summary)
    print("\nSummary:")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
