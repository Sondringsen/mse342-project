#!/usr/bin/env python
"""Aggregate repeated deep-hedging comparison runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate hedging_summary.csv files across repeated runs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("hedging_dirs", nargs="+", type=Path, help="Directories with hedging_summary.csv")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for aggregate CSV files")
    return parser.parse_args()


def load_seed(path: Path) -> int | None:
    config_path = path / "run_config.json"
    if not config_path.exists():
        return None
    try:
        return int(json.loads(config_path.read_text()).get("seed"))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def main() -> None:
    args = parse_args()
    frames: List[pd.DataFrame] = []
    for hedging_dir in args.hedging_dirs:
        summary_path = hedging_dir / "hedging_summary.csv"
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing {summary_path}")
        frame = pd.read_csv(summary_path)
        frame.insert(0, "hedging_dir", str(hedging_dir))
        frame.insert(1, "seed", load_seed(hedging_dir))
        frames.append(frame)

    rows = pd.concat(frames, ignore_index=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(args.output_dir / "hedging_seed_rows.csv", index=False)

    numeric_metrics = [
        col
        for col in rows.columns
        if col
        not in {
            "hedging_dir",
            "seed",
            "label",
            "source",
            "predictions_path",
        }
        and pd.api.types.is_numeric_dtype(rows[col])
    ]
    aggregations: Dict[str, List[str]] = {col: ["mean", "std", "min", "max"] for col in numeric_metrics}
    summary = rows.groupby("label", sort=True).agg(aggregations)
    summary.columns = ["_".join(col).rstrip("_") for col in summary.columns.to_flat_index()]
    summary = summary.reset_index()
    summary.to_csv(args.output_dir / "hedging_seed_summary.csv", index=False)


if __name__ == "__main__":
    main()
