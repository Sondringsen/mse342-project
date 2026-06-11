#!/usr/bin/env python
"""Recompute evaluation artifacts from saved prediction CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from pipeline.analysis import write_comparison_analysis
from pipeline.evaluation import evaluate_predictions
from pipeline.output_index import write_outputs_index


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recompute evaluation_results.json and comparison summaries from predictions.csv"
    )
    parser.add_argument("run_dirs", nargs="+", type=Path, help="Run directories to re-evaluate")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["findiffusion", "csdi"],
        help="Model subdirectories to re-evaluate",
    )
    args = parser.parse_args()

    for run_dir in args.run_dirs:
        run_dir = run_dir.resolve()
        results = []
        for model_name in args.models:
            model_dir = run_dir / model_name
            predictions_path = model_dir / "predictions.csv"
            if not predictions_path.exists():
                continue
            predictions = pd.read_csv(predictions_path)
            results.append(evaluate_predictions(predictions, model_dir, model_name))
            print(f"Re-evaluated {predictions_path}")

        if len(results) > 1:
            write_comparison_analysis(results, run_dir)
            print(f"Wrote comparison summary for {run_dir}")
        write_outputs_index(run_dir.parent)


if __name__ == "__main__":
    main()
