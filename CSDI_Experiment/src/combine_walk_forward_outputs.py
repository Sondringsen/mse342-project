#!/usr/bin/env python3
"""Combine parallel walk-forward worker outputs into one run directory."""

import argparse
import json
import math
import shutil
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge CSDI walk-forward worker directories.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("worker_dirs", nargs="+", type=Path)
    parser.add_argument(
        "--copy-folds",
        action="store_true",
        help="Copy fold_*/ artifacts from workers into output-dir.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    prediction_frames = []
    metric_frames = []
    run_configs = []
    for worker_dir in args.worker_dirs:
        worker_dir = worker_dir.resolve()
        predictions_path = worker_dir / "predictions.csv"
        metrics_path = worker_dir / "metrics_by_fold.csv"
        run_config_path = worker_dir / "run_config.json"
        if not predictions_path.exists() or not metrics_path.exists() or not run_config_path.exists():
            raise FileNotFoundError(f"Worker output is incomplete: {worker_dir}")

        prediction_frames.append(pd.read_csv(predictions_path))
        metric_frames.append(pd.read_csv(metrics_path))
        run_configs.append(read_json(run_config_path))

        if args.copy_folds:
            for fold_dir in sorted(worker_dir.glob("fold_*")):
                target_dir = args.output_dir / fold_dir.name
                if target_dir.exists():
                    shutil.rmtree(target_dir)
                shutil.copytree(fold_dir, target_dir)

    predictions = pd.concat(prediction_frames, ignore_index=True).sort_values(
        ["fold", "target_index", "feature"]
    )
    metrics = pd.concat(metric_frames, ignore_index=True).sort_values("fold")

    predictions.to_csv(args.output_dir / "predictions.csv", index=False)
    metrics.to_csv(args.output_dir / "metrics_by_fold.csv", index=False)

    summary = {
        "folds": int(metrics["fold"].nunique()),
        "eval_points": int(metrics["eval_points"].sum()),
        "mean_fold_mae": float(metrics["mae"].mean()),
        "mean_fold_rmse": float(metrics["rmse"].mean()),
        "pooled_mae": float((predictions["pred_median"] - predictions["actual"]).abs().mean()),
        "pooled_rmse": float(
            math.sqrt(float(((predictions["pred_median"] - predictions["actual"]) ** 2).mean()))
        ),
        "worker_dirs": [str(path.resolve()) for path in args.worker_dirs],
        "predictions_csv": str(args.output_dir / "predictions.csv"),
        "metrics_by_fold_csv": str(args.output_dir / "metrics_by_fold.csv"),
    }
    write_json(args.output_dir / "summary.json", summary)

    combined_config = dict(run_configs[0])
    combined_config["parallel_workers"] = [str(path.resolve()) for path in args.worker_dirs]
    combined_config["folds"] = sorted(
        [fold for config in run_configs for fold in config.get("folds", [])],
        key=lambda item: item["fold"],
    )
    combined_config["total_available_folds"] = max(
        config.get("total_available_folds", len(combined_config["folds"]))
        for config in run_configs
    )
    write_json(args.output_dir / "run_config.json", combined_config)

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
