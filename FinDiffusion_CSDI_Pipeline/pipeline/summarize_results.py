#!/usr/bin/env python
"""Rebuild comparison_summary files for a completed parallel cluster run."""

import argparse
from pathlib import Path

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.analysis import load_results, write_comparison_analysis
from pipeline.output_index import write_outputs_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize comparison model outputs")
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()

    results = load_results(args.run_dir)
    write_comparison_analysis(results, args.run_dir)
    write_outputs_index(args.run_dir.parent)
    print("Wrote %s" % (args.run_dir / "comparison_summary.csv"))


if __name__ == "__main__":
    main()
