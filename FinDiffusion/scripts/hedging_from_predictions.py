#!/usr/bin/env python
"""Train and evaluate deep hedgers from pipeline prediction CSVs.

The FinDiffusion hedging scripts expect a synthetic.csv where each row is a
return path. The comparison pipeline stores conditional forecast samples in
predictions.csv. This adapter converts those samples into fixed-length return
windows, trains one hedger per run, and evaluates every hedger on the same real
held-out windows.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.hedging import DeepHedger, HedgingTrainer, compute_pnl, log_returns_to_prices  # noqa: E402


LOGGER = logging.getLogger(__name__)

S0 = 100.0
OTM_FACTOR = 1.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train/evaluate deep hedgers from pipeline predictions.csv files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("run_dirs", nargs="+", type=Path, help="Pipeline run dirs or predictions.csv files")
    parser.add_argument(
        "--model-name",
        default="findiffusion",
        help="Model subdirectory to read when a run directory is provided",
    )
    parser.add_argument(
        "--eval-run-dir",
        type=Path,
        default=None,
        help="Run dir or predictions.csv used for common real evaluation paths",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for hedger checkpoints and metrics",
    )
    parser.add_argument("--seq-len", type=int, default=30, help="Option TTL / hedge path length")
    parser.add_argument("--window-stride", type=int, default=30, help="Stride for path windowing")
    parser.add_argument("--n-train-windows", type=int, default=10000, help="Max synthetic windows per run")
    parser.add_argument("--n-eval-windows", type=int, default=1000, help="Max real eval windows")
    parser.add_argument("--epochs", type=int, default=250, help="Hedger training epochs")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--cvar-alpha", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Training/evaluation device",
    )
    return parser.parse_args()


def resolve_predictions_path(path: Path, model_name: str) -> Path:
    if path.is_file():
        return path
    candidates = [
        path / model_name / "predictions.csv",
        path / "predictions.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No predictions.csv found for {path}")


def paths_from_predictions(predictions: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    sample_cols = [col for col in predictions.columns if col.startswith("sample_")]
    if not sample_cols:
        raise ValueError("predictions.csv has no sample_ columns")

    if "forecast_start_index" in predictions.columns and "horizon_step" in predictions.columns:
        return horizon_paths_from_predictions(predictions, sample_cols)

    real_paths = []
    synthetic_paths = []
    for _ticker, ticker_df in predictions.groupby("ticker", sort=True):
        ticker_df = ticker_df.sort_values("target_index")
        real_paths.append(ticker_df["actual"].to_numpy(np.float32))
        for col in sample_cols:
            synthetic_paths.append(ticker_df[col].to_numpy(np.float32))
    return stack_paths(real_paths), stack_paths(synthetic_paths)


def horizon_paths_from_predictions(
    predictions: pd.DataFrame,
    sample_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    real_paths = []
    synthetic_paths = []
    for _ticker, ticker_df in predictions.groupby("ticker", sort=True):
        real_values = []
        synthetic_values = {col: [] for col in sample_cols}
        for _forecast_index, block in ticker_df.groupby("forecast_start_index", sort=True):
            block = block.sort_values("horizon_step")
            real_values.extend(block["actual"].to_numpy(np.float32).tolist())
            for col in sample_cols:
                synthetic_values[col].extend(block[col].to_numpy(np.float32).tolist())
        if real_values:
            real_paths.append(np.asarray(real_values, dtype=np.float32))
        for col in sample_cols:
            if synthetic_values[col]:
                synthetic_paths.append(np.asarray(synthetic_values[col], dtype=np.float32))
    return stack_paths(real_paths), stack_paths(synthetic_paths)


def stack_paths(paths: List[np.ndarray]) -> np.ndarray:
    if not paths:
        return np.empty((0, 0), dtype=np.float32)
    min_length = min(len(path) for path in paths)
    if min_length <= 0:
        return np.empty((len(paths), 0), dtype=np.float32)
    return np.asarray([path[-min_length:] for path in paths], dtype=np.float32)


def window_paths(
    paths: np.ndarray,
    seq_len: int,
    stride: int,
    max_windows: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if paths.ndim != 2:
        raise ValueError(f"Expected 2D paths, got shape {paths.shape}")
    if seq_len <= 0:
        raise ValueError("--seq-len must be positive")
    if stride <= 0:
        raise ValueError("--window-stride must be positive")

    windows = []
    for path in paths:
        if len(path) < seq_len:
            continue
        for start in range(0, len(path) - seq_len + 1, stride):
            windows.append(path[start : start + seq_len])
    if not windows:
        raise ValueError(f"No windows of length {seq_len} could be built from paths {paths.shape}")

    arr = np.asarray(windows, dtype=np.float32)
    if max_windows > 0 and len(arr) > max_windows:
        idx = rng.choice(len(arr), size=max_windows, replace=False)
        arr = arr[idx]
    return arr


def choose_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cvar(pnl: np.ndarray, alpha: float) -> float:
    losses = -pnl
    var = np.quantile(losses, alpha)
    tail = losses[losses >= var]
    return float(tail.mean())


def max_drawdown_per_path(cum_gains: np.ndarray) -> np.ndarray:
    running_peak = np.maximum.accumulate(cum_gains, axis=1)
    return (running_peak - cum_gains).max(axis=1)


def evaluate_hedger(
    hedger: DeepHedger,
    eval_returns: np.ndarray,
    seq_len: int,
    strike: float,
    alpha: float,
    device: torch.device,
) -> Dict[str, float]:
    prices_np = log_returns_to_prices(eval_returns[:, :seq_len], s0=S0)
    prices = torch.tensor(prices_np, device=device)
    hedger.eval()
    with torch.no_grad():
        pnl_t, delta_t = compute_pnl(prices, hedger, strike, seq_len)
    pnl = pnl_t.detach().cpu().numpy()
    deltas = delta_t.detach().cpu().numpy()
    price_changes = prices_np[:, 1:] - prices_np[:, :-1]
    cum_gains = np.cumsum(deltas * price_changes, axis=1)
    drawdown = max_drawdown_per_path(cum_gains)
    return {
        "mean_pnl": float(pnl.mean()),
        "std_pnl": float(pnl.std()),
        "pct_profitable": float((pnl > 0).mean()),
        f"cvar_{int(alpha * 100)}": cvar(pnl, alpha),
        "mean_max_drawdown": float(drawdown.mean()),
        "worst_max_drawdown": float(drawdown.max()),
    }


def evaluate_no_hedge(
    eval_returns: np.ndarray,
    seq_len: int,
    strike: float,
    alpha: float,
) -> Dict[str, float]:
    prices_np = log_returns_to_prices(eval_returns[:, :seq_len], s0=S0)
    payoff = np.maximum(prices_np[:, -1] - strike, 0.0)
    pnl = -payoff
    drawdown = np.zeros(len(pnl), dtype=np.float32)
    return {
        "mean_pnl": float(pnl.mean()),
        "std_pnl": float(pnl.std()),
        "pct_profitable": float((pnl > 0).mean()),
        f"cvar_{int(alpha * 100)}": cvar(pnl, alpha),
        "mean_max_drawdown": float(drawdown.mean()),
        "worst_max_drawdown": float(drawdown.max()),
    }


def safe_label(path: Path) -> str:
    label = path.stem if path.is_file() else path.name
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in label)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    strike = S0 * OTM_FACTOR
    LOGGER.info("Device: %s", device)
    LOGGER.info("Option: European call S0=%.1f K=%.1f TTL=%d", S0, strike, args.seq_len)

    eval_source = args.eval_run_dir or args.run_dirs[0]
    eval_predictions = pd.read_csv(resolve_predictions_path(eval_source, args.model_name))
    real_paths, _synthetic_unused = paths_from_predictions(eval_predictions)
    eval_windows = window_paths(
        real_paths,
        seq_len=args.seq_len,
        stride=args.window_stride,
        max_windows=args.n_eval_windows,
        rng=rng,
    )
    LOGGER.info("Common real eval windows: %s from %s", eval_windows.shape, eval_source)

    rows = []
    baseline = evaluate_no_hedge(eval_windows, args.seq_len, strike, args.cvar_alpha)
    rows.append(
        {
            "label": "no_hedge",
            "source": "baseline",
            "n_train_windows": 0,
            "n_eval_windows": len(eval_windows),
            **baseline,
        }
    )

    for run_dir in args.run_dirs:
        label = safe_label(run_dir)
        predictions_path = resolve_predictions_path(run_dir, args.model_name)
        LOGGER.info("Preparing %s from %s", label, predictions_path)
        predictions = pd.read_csv(predictions_path)
        _real_paths, synthetic_paths = paths_from_predictions(predictions)
        train_windows = window_paths(
            synthetic_paths,
            seq_len=args.seq_len,
            stride=args.window_stride,
            max_windows=args.n_train_windows,
            rng=rng,
        )
        LOGGER.info("%s synthetic train windows: %s", label, train_windows.shape)

        run_output = args.output_dir / label
        run_output.mkdir(parents=True, exist_ok=True)
        hedger = DeepHedger(input_dim=2, hidden_dim=args.hidden_dim)
        trainer = HedgingTrainer(
            model=hedger,
            strike=strike,
            seq_len=args.seq_len,
            lr=args.lr,
            cvar_alpha=args.cvar_alpha,
            device=device,
        )
        losses = trainer.train(
            synthetic_log_returns=train_windows,
            n_epochs=args.epochs,
            batch_size=args.batch_size,
            checkpoint_path=run_output / "hedging_model.pt",
            s0=S0,
        )
        pd.DataFrame({"epoch": np.arange(1, len(losses) + 1), "loss": losses}).to_csv(
            run_output / "training_loss.csv",
            index=False,
        )

        checkpoint = torch.load(run_output / "hedging_model.pt", map_location=device)
        hedger.load_state_dict(checkpoint["model_state_dict"])
        metrics = evaluate_hedger(hedger, eval_windows, args.seq_len, strike, args.cvar_alpha, device)
        result = {
            "label": label,
            "source": str(run_dir),
            "predictions_path": str(predictions_path),
            "n_train_windows": int(len(train_windows)),
            "n_eval_windows": int(len(eval_windows)),
            "best_train_epoch": int(checkpoint.get("epoch", -1)) + 1,
            "best_train_loss": float(checkpoint.get("loss", np.nan)),
            **metrics,
        }
        rows.append(result)
        (run_output / "results.json").write_text(json.dumps(result, indent=2) + "\n")

    summary = pd.DataFrame(rows)
    summary.to_csv(args.output_dir / "hedging_summary.csv", index=False)
    (args.output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "run_dirs": [str(path) for path in args.run_dirs],
                "eval_run_dir": str(eval_source),
                "seq_len": args.seq_len,
                "window_stride": args.window_stride,
                "n_train_windows": args.n_train_windows,
                "n_eval_windows": args.n_eval_windows,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "hidden_dim": args.hidden_dim,
                "cvar_alpha": args.cvar_alpha,
                "seed": args.seed,
            },
            indent=2,
        )
        + "\n"
    )
    LOGGER.info("Wrote %s", args.output_dir / "hedging_summary.csv")


if __name__ == "__main__":
    main()
