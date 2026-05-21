#!/usr/bin/env python3
"""Fixed-split CSDI scenario generation for financial path experiments."""

import argparse
import datetime as dt
import json
import pickle
import random
import sys
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


TRADING_DAYS_PER_YEAR = 252


class WindowedForecastDataset(Dataset):
    """CSDI-compatible windows with the final pred_length rows masked."""

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
        observed_mask = self.mask[start:stop]
        gt_mask = observed_mask.copy()
        gt_mask[-self.pred_length :] = 0.0

        return {
            "observed_data": self.values[start:stop],
            "observed_mask": observed_mask,
            "gt_mask": gt_mask,
            "timepoints": np.arange(self.seq_length, dtype=np.float32),
            "feature_id": np.arange(self.values.shape[1], dtype=np.float32),
        }


def require_runtime_dependencies() -> None:
    if RUNTIME_IMPORT_ERROR is None:
        return
    raise SystemExit(
        "Missing CSDI runtime dependency: %s\n"
        "Install the CSDI requirements, for example:\n"
        "  ../venv/bin/pip3 install -r CSDI/requirements.txt" % RUNTIME_IMPORT_ERROR
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> str:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("Requested %s, but CUDA is not available. Falling back to CPU." % requested)
        return "cpu"
    if requested == "mps" and not torch.backends.mps.is_available():
        print("Requested mps, but MPS is not available. Falling back to CPU.")
        return "cpu"
    return requested


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def as_int(value: object) -> int:
    return int(float(value))


def load_config(path: Path, args: argparse.Namespace) -> Dict[str, Any]:
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
    config["topology"] = {
        "weight": float(args.topology_loss_weight),
        "window": int(args.topology_window),
        "stride": int(args.topology_stride),
        "max_points": int(args.topology_max_points),
        "thresholds": int(args.topology_thresholds),
        "temperature": float(args.topology_temperature),
        "recurrence_weight": float(args.topology_recurrence_weight),
        "distance_weight": float(args.topology_distance_weight),
        "spectrum_weight": float(args.topology_spectrum_weight),
    }
    config["constraints"] = {
        "loss_weight": float(args.constraint_loss_weight),
        "volatility_weight": float(args.constraint_volatility_weight),
        "sample_clamp": bool(args.constraint_sample_clamp),
        "lower_quantile": float(args.constraint_lower_quantile),
        "upper_quantile": float(args.constraint_upper_quantile),
        "margin_z": float(args.constraint_margin_z),
        "lower_z": None,
        "upper_z": None,
    }
    return config


def load_timeseries(
    csv_path: Path,
    date_column: str,
    target_columns: Optional[List[str]],
) -> Tuple[Optional[pd.Series], List[str], np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    dates = pd.to_datetime(df[date_column]) if date_column in df.columns else None

    if target_columns is None:
        excluded = {date_column} if date_column in df.columns else set()
        target_columns = [col for col in df.columns if col not in excluded]

    missing = [col for col in target_columns if col not in df.columns]
    if missing:
        raise ValueError("Target columns are missing from %s: %s" % (csv_path, missing))

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


def transform_returns(values: np.ndarray, mask: np.ndarray, transform: str) -> np.ndarray:
    if transform == "simple":
        return values.astype(np.float32, copy=True)
    if transform != "log":
        raise ValueError("Unknown return transform: %s" % transform)
    observed = mask.astype(bool)
    if np.any(values[observed] <= -1.0):
        raise ValueError("Log-return transform requires simple returns greater than -100%")
    transformed = values.astype(np.float32, copy=True)
    transformed[observed] = np.log1p(transformed[observed])
    transformed[~observed] = 0.0
    return transformed.astype(np.float32)


def model_to_simple_returns(values: np.ndarray, transform: str) -> np.ndarray:
    if transform == "simple":
        return values
    if transform == "log":
        return np.expm1(np.clip(values, -50.0, 50.0))
    raise ValueError("Unknown return transform: %s" % transform)


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


def predict_paths(
    model: CSDI_Forecasting,
    loader: DataLoader,
    nsample: int,
    mean: np.ndarray,
    std: np.ndarray,
    dates: Optional[pd.Series],
    features: List[str],
    origin_index: int,
    pred_length: int,
    output_dir: Path,
    return_transform: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    model.eval()
    prediction_rows = []  # type: List[Dict[str, Any]]
    abs_errors = []  # type: List[float]
    sq_errors = []  # type: List[float]
    model_abs_errors = []  # type: List[float]
    model_sq_errors = []  # type: List[float]
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
            samples_simple_np = model_to_simple_returns(samples_np, return_transform)
            target_simple_np = model_to_simple_returns(target_np, return_transform)

            for batch_item in range(samples_np.shape[0]):
                for horizon in range(1, pred_length + 1):
                    seq_pos = samples_np.shape[2] - pred_length + horizon - 1
                    target_index = origin_index + horizon - 1
                    for feature_index, feature in enumerate(features):
                        if eval_np[batch_item, seq_pos, feature_index] <= 0:
                            continue
                        draws_model = samples_np[batch_item, :, seq_pos, feature_index]
                        actual_model = float(target_np[batch_item, seq_pos, feature_index])
                        median_model = float(np.median(draws_model))
                        mean_model = float(np.mean(draws_model))
                        model_error = median_model - actual_model
                        model_abs_errors.append(abs(model_error))
                        model_sq_errors.append(model_error * model_error)

                        draws = samples_simple_np[batch_item, :, seq_pos, feature_index]
                        actual = float(target_simple_np[batch_item, seq_pos, feature_index])
                        median = float(np.median(draws))
                        mean_pred = float(np.mean(draws))
                        error = median - actual
                        abs_errors.append(abs(error))
                        sq_errors.append(error * error)
                        prediction_rows.append(
                            {
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
                                "actual_model": actual_model,
                                "pred_median_model": median_model,
                                "pred_mean_model": mean_model,
                                "pred_p05_model": float(np.quantile(draws_model, 0.05)),
                                "pred_p25_model": float(np.quantile(draws_model, 0.25)),
                                "pred_p75_model": float(np.quantile(draws_model, 0.75)),
                                "pred_p95_model": float(np.quantile(draws_model, 0.95)),
                                "n_samples": nsample,
                            }
                        )

    with (output_dir / ("generated_outputs_nsample%d.pk" % nsample)).open("wb") as f:
        pickle_payload = [
            torch.cat(all_generated_samples, dim=0),
            torch.cat(all_target, dim=0),
            torch.cat(all_evalpoint, dim=0),
            torch.cat(all_observed_point, dim=0),
            torch.cat(all_observed_time, dim=0),
            scaler.detach().cpu(),
            mean_scaler.detach().cpu(),
        ]
        pickle.dump(pickle_payload, f)

    metrics = {
        "eval_points": len(abs_errors),
        "mae": float(np.mean(abs_errors)) if abs_errors else float("nan"),
        "rmse": float(np.sqrt(np.mean(sq_errors))) if sq_errors else float("nan"),
        "model_mae": float(np.mean(model_abs_errors)) if model_abs_errors else float("nan"),
        "model_rmse": float(np.sqrt(np.mean(model_sq_errors))) if model_sq_errors else float("nan"),
        "return_transform": return_transform,
    }
    return pd.DataFrame(prediction_rows), metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train CSDI once on a pre-cutoff sample and generate scenario paths.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_ROOT / "data/processed/french49_daily_returns.csv",
    )
    parser.add_argument("--date-column", default="date")
    parser.add_argument("--target-columns", nargs="+", default=None)
    parser.add_argument("--config", type=Path, default=CSDI_ROOT / "config/base_forecasting.yaml")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--train-end-date", default="2015-12-31")
    parser.add_argument("--history-length", type=int, default=756)
    parser.add_argument(
        "--horizon-years",
        nargs="+",
        type=int,
        default=[1, 5, 10],
        help="Forecast/scenario horizons in trading years.",
    )
    parser.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        default=None,
        help="Override horizon-years with exact trading-day horizons.",
    )
    parser.add_argument(
        "--return-transform",
        choices=["log", "simple"],
        default="log",
        help="Model simple returns directly or transform input simple returns to log returns.",
    )
    parser.add_argument("--train-stride", type=int, default=5)
    parser.add_argument("--valid-windows", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--itr-per-epoch", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--valid-epoch-interval", type=int, default=20)
    parser.add_argument("--nsample", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--variant-name", default="vanilla")
    parser.add_argument(
        "--topology-loss-weight",
        type=float,
        default=0.0,
        help="Weight for the differentiable sliding-window topology proxy loss.",
    )
    parser.add_argument("--topology-window", type=int, default=32)
    parser.add_argument("--topology-stride", type=int, default=4)
    parser.add_argument("--topology-max-points", type=int, default=64)
    parser.add_argument("--topology-thresholds", type=int, default=12)
    parser.add_argument("--topology-temperature", type=float, default=0.1)
    parser.add_argument("--topology-recurrence-weight", type=float, default=1.0)
    parser.add_argument("--topology-distance-weight", type=float, default=0.25)
    parser.add_argument("--topology-spectrum-weight", type=float, default=0.25)
    parser.add_argument(
        "--constraint-loss-weight",
        type=float,
        default=0.02,
        help="Weight for financial validity penalties on denoised standardized returns.",
    )
    parser.add_argument(
        "--constraint-volatility-weight",
        type=float,
        default=0.25,
        help="Relative weight inside the constraint loss for matching target-window volatility.",
    )
    parser.add_argument(
        "--constraint-sample-clamp",
        dest="constraint_sample_clamp",
        action="store_true",
        help="Clamp reverse-diffusion samples to empirical standardized-return bounds.",
    )
    parser.add_argument(
        "--no-constraint-sample-clamp",
        dest="constraint_sample_clamp",
        action="store_false",
    )
    parser.set_defaults(constraint_sample_clamp=True)
    parser.add_argument("--constraint-lower-quantile", type=float, default=0.001)
    parser.add_argument("--constraint-upper-quantile", type=float, default=0.999)
    parser.add_argument("--constraint-margin-z", type=float, default=0.5)
    parser.add_argument("--skip-training", action="store_true")
    return parser.parse_args()


def make_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        PROJECT_ROOT
        / "CSDI_Experiment"
        / "outputs"
        / ("fixed_split_%s_%s" % (args.variant_name, stamp))
    )


def resolve_train_end_index(dates: Optional[pd.Series], train_end_date: str, rows: int) -> int:
    if dates is None:
        return int(train_end_date)
    cutoff = pd.Timestamp(train_end_date)
    keep = dates <= cutoff
    train_end = int(keep.sum())
    if train_end <= 0:
        raise ValueError("train-end-date is before the first data row")
    if train_end >= rows:
        raise ValueError("train-end-date leaves no holdout rows")
    return train_end


def make_train_valid_starts_for_horizon(
    train_end: int,
    history_length: int,
    pred_length: int,
    train_stride: int,
    valid_windows: int,
) -> Tuple[np.ndarray, np.ndarray]:
    seq_length = history_length + pred_length
    last_start = train_end - seq_length
    if last_start < 0:
        raise ValueError(
            "Not enough pre-cutoff data for history_length=%d and pred_length=%d. "
            "Need at least %d training rows, got %d."
            % (history_length, pred_length, seq_length, train_end)
        )
    all_starts = np.arange(0, last_start + 1, train_stride)
    valid_count = min(max(valid_windows, 0), max(0, len(all_starts) - 1))
    if valid_count == 0:
        return all_starts, np.asarray([], dtype=np.int64)
    return all_starts[:-valid_count], all_starts[-valid_count:]


def write_horizon_metadata(
    path: Path,
    args: argparse.Namespace,
    horizon: int,
    train_end: int,
    dates: Optional[pd.Series],
    features: List[str],
    train_starts: np.ndarray,
    valid_starts: np.ndarray,
    config: Dict[str, Any],
) -> None:
    write_json(
        path,
        {
            "variant_name": args.variant_name,
            "topology_loss_weight": args.topology_loss_weight,
            "return_transform": args.return_transform,
            "reported_return_unit": "simple_return",
            "model_return_unit": "log_return" if args.return_transform == "log" else "simple_return",
            "train_end_index": train_end,
            "train_end_date": format_date(dates, train_end - 1),
            "holdout_start_date": format_date(dates, train_end),
            "history_length": args.history_length,
            "pred_length": horizon,
            "features": features,
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


def run_horizon(
    horizon: int,
    args: argparse.Namespace,
    base_config: Dict[str, Any],
    values: np.ndarray,
    mask: np.ndarray,
    dates: Optional[pd.Series],
    features: List[str],
    train_end: int,
    output_dir: Path,
    device: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    horizon_dir = output_dir / ("horizon_%04d" % horizon)
    horizon_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed + horizon)
    mean, std = fit_scaler(values, mask, train_end)
    scaled_values = standardize(values, mask, mean, std)
    train_starts, valid_starts = make_train_valid_starts_for_horizon(
        train_end,
        args.history_length,
        horizon,
        args.train_stride,
        args.valid_windows,
    )

    available_future = len(values) - train_end
    eval_horizon = min(horizon, available_future)
    if eval_horizon < horizon:
        print(
            "Horizon %d extends past available real data; generated paths are kept, "
            "metrics use first %d available holdout rows." % (horizon, eval_horizon),
            flush=True,
        )

    train_dataset = WindowedForecastDataset(
        scaled_values,
        mask,
        train_starts,
        args.history_length,
        horizon,
    )
    valid_dataset = (
        WindowedForecastDataset(scaled_values, mask, valid_starts, args.history_length, horizon)
        if len(valid_starts) > 0
        else None
    )

    test_start = train_end - args.history_length
    test_dataset = WindowedForecastDataset(
        scaled_values,
        mask,
        [test_start],
        args.history_length,
        horizon,
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
    constraint_values = scaled_values[:train_end][mask[:train_end].astype(bool)]
    if constraint_values.size > 0:
        lower_q = float(config["constraints"].get("lower_quantile", 0.001))
        upper_q = float(config["constraints"].get("upper_quantile", 0.999))
        margin = float(config["constraints"].get("margin_z", 0.5))
        config["constraints"]["lower_z"] = float(np.quantile(constraint_values, lower_q) - margin)
        config["constraints"]["upper_z"] = float(np.quantile(constraint_values, upper_q) + margin)
    model = CSDI_Forecasting(config, device, len(features)).to(device)

    write_horizon_metadata(
        horizon_dir / "horizon_config.json",
        args,
        horizon,
        train_end,
        dates,
        features,
        train_starts,
        valid_starts,
        config,
    )
    np.savez(horizon_dir / "scaler.npz", mean=mean, std=std, features=np.asarray(features))

    model_path = horizon_dir / "model.pth"
    if args.skip_training:
        if not model_path.exists():
            raise FileNotFoundError("--skip-training requested but %s does not exist" % model_path)
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        train_csdi(
            model,
            config["train"],
            train_loader,
            valid_loader=valid_loader,
            valid_epoch_interval=args.valid_epoch_interval,
            foldername=str(horizon_dir),
        )

    model.target_dim = len(features)
    predictions, metrics = predict_paths(
        model,
        test_loader,
        args.nsample,
        mean,
        std,
        dates,
        features,
        origin_index=train_end,
        pred_length=horizon,
        output_dir=horizon_dir,
        return_transform=args.return_transform,
    )

    predictions["horizon_days"] = horizon
    predictions["horizon_years"] = horizon / float(TRADING_DAYS_PER_YEAR)
    if eval_horizon < horizon:
        predictions = predictions[predictions["horizon"] <= eval_horizon].copy()

    metrics["horizon_days"] = horizon
    metrics["horizon_years"] = horizon / float(TRADING_DAYS_PER_YEAR)
    metrics["variant_name"] = args.variant_name
    metrics["train_end_date"] = format_date(dates, train_end - 1)
    metrics["eval_horizon_days"] = eval_horizon
    metrics["constraint_lower_z"] = config["constraints"].get("lower_z")
    metrics["constraint_upper_z"] = config["constraints"].get("upper_z")
    metrics["constraint_sample_clamp"] = bool(config["constraints"].get("sample_clamp", False))

    predictions.to_csv(horizon_dir / "predictions.csv", index=False)
    write_json(horizon_dir / "metrics.json", metrics)
    return predictions, metrics


def main() -> int:
    args = parse_args()
    require_runtime_dependencies()
    device = resolve_device(args.device)
    output_dir = make_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    dates, features, simple_values, mask = load_timeseries(args.data, args.date_column, args.target_columns)
    values = transform_returns(simple_values, mask, args.return_transform)
    train_end = resolve_train_end_index(dates, args.train_end_date, len(values))
    horizons = args.horizons if args.horizons is not None else [
        years * TRADING_DAYS_PER_YEAR for years in args.horizon_years
    ]
    horizons = [int(horizon) for horizon in horizons]

    config = load_config(args.config, args)
    run_config = {
        "experiment": "fixed_split_scenarios",
        "variant_name": args.variant_name,
        "topology_loss_weight": args.topology_loss_weight,
        "data": str(args.data),
        "date_column": args.date_column,
        "return_transform": args.return_transform,
        "reported_return_unit": "simple_return",
        "model_return_unit": "log_return" if args.return_transform == "log" else "simple_return",
        "features": features,
        "rows": int(len(values)),
        "train_end_index": train_end,
        "train_end_date": format_date(dates, train_end - 1),
        "holdout_start_date": format_date(dates, train_end),
        "history_length": args.history_length,
        "horizons": horizons,
        "device": device,
        "seed": args.seed,
        "nsample": args.nsample,
        "csdi_config": config,
    }
    write_json(output_dir / "run_config.json", run_config)

    print("Writing fixed-split scenario outputs to %s" % output_dir)
    print(
        "Training rows [0, %d] ending %s; holdout starts %s"
        % (train_end - 1, run_config["train_end_date"], run_config["holdout_start_date"]),
        flush=True,
    )
    print("Running horizons: %s" % ", ".join(str(h) for h in horizons), flush=True)

    all_predictions = []  # type: List[pd.DataFrame]
    all_metrics = []  # type: List[Dict[str, Any]]
    for horizon in horizons:
        print("\nHorizon %d trading days" % horizon, flush=True)
        predictions, metrics = run_horizon(
            horizon,
            args,
            config,
            values,
            mask,
            dates,
            features,
            train_end,
            output_dir,
            device,
        )
        all_predictions.append(predictions)
        all_metrics.append(metrics)
        print(
            "Horizon %d metrics: MAE=%.6g RMSE=%.6g eval_points=%d"
            % (horizon, metrics["mae"], metrics["rmse"], int(metrics["eval_points"])),
            flush=True,
        )
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    predictions_df = pd.concat(all_predictions, ignore_index=True)
    metrics_df = pd.DataFrame(all_metrics)
    predictions_df.to_csv(output_dir / "predictions.csv", index=False)
    metrics_df.to_csv(output_dir / "metrics_by_horizon.csv", index=False)

    summary = {
        "variant_name": args.variant_name,
        "train_end_date": format_date(dates, train_end - 1),
        "horizons": horizons,
        "mean_horizon_mae": float(metrics_df["mae"].mean()),
        "mean_horizon_rmse": float(metrics_df["rmse"].mean()),
        "predictions_csv": str(output_dir / "predictions.csv"),
        "metrics_by_horizon_csv": str(output_dir / "metrics_by_horizon.csv"),
    }
    write_json(output_dir / "summary.json", summary)
    print("\nSummary:")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
