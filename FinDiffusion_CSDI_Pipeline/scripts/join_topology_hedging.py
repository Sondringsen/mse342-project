#!/usr/bin/env python
"""Join business-cycle topology diagnostics with hedging outcomes."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mse342_matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Join topology summary metrics with hedging aggregate metrics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--topology-summary", type=Path, required=True)
    parser.add_argument("--hedging-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=160)
    return parser.parse_args()


def normalized_hedging_summary(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "label" not in frame.columns:
        raise ValueError(f"{path} must contain a label column")
    rename = {}
    for base in ["cvar_95", "mean_pnl", "std_pnl", "pct_profitable", "mean_max_drawdown", "worst_max_drawdown"]:
        if f"{base}_mean" in frame.columns:
            rename[f"{base}_mean"] = base
    return frame.rename(columns=rename)


def main() -> None:
    args = parse_args()
    topology = pd.read_csv(args.topology_summary)
    hedging = normalized_hedging_summary(args.hedging_summary)
    if "run_label" not in topology.columns:
        raise ValueError(f"{args.topology_summary} must contain run_label")

    joined = topology.merge(hedging, left_on="run_label", right_on="label", how="inner")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    joined.to_csv(args.output_dir / "topology_hedging_summary.csv", index=False)
    write_report(joined, args.output_dir)
    plot(joined, args.output_dir, args.dpi)
    print(f"Wrote {args.output_dir}")


def write_report(joined: pd.DataFrame, output_dir: Path) -> None:
    lines = ["# Topology And Hedging Join", ""]
    if joined.empty:
        lines.append("No matching run labels were found between topology and hedging summaries.")
    else:
        cols = [
            "run_label",
            "topology_distance_to_real_median",
            "cvar_95",
            "mean_pnl",
            "pct_profitable",
        ]
        cols = [col for col in cols if col in joined.columns]
        lines.append(to_markdown(joined[cols].sort_values(cols[1] if len(cols) > 1 else cols[0])))
        if {"topology_distance_to_real_median", "cvar_95"}.issubset(joined.columns) and len(joined) > 1:
            corr = joined["topology_distance_to_real_median"].corr(joined["cvar_95"])
            lines.extend(["", f"Correlation between topology distance and CVaR95: `{corr:.4f}`."])
    (output_dir / "topology_hedging_report.md").write_text("\n".join(lines) + "\n")


def to_markdown(frame: pd.DataFrame) -> str:
    headers = list(frame.columns)
    rows = []
    for _, row in frame.iterrows():
        values = []
        for col in headers:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        rows.append(values)
    widths = [
        max(len(header), *(len(row[i]) for row in rows)) if rows else len(header)
        for i, header in enumerate(headers)
    ]
    header_line = "| " + " | ".join(header.ljust(widths[i]) for i, header in enumerate(headers)) + " |"
    sep_line = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    body = [
        "| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |"
        for row in rows
    ]
    return "\n".join([header_line, sep_line, *body])


def plot(joined: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    needed = {"topology_distance_to_real_median", "cvar_95", "run_label"}
    if joined.empty or not needed.issubset(joined.columns):
        return
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.scatter(joined["topology_distance_to_real_median"], joined["cvar_95"], s=55)
    for _, row in joined.iterrows():
        ax.annotate(
            str(row["run_label"]),
            (row["topology_distance_to_real_median"], row["cvar_95"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
        )
    ax.set_xlabel("Median topology distance to real")
    ax.set_ylabel("Hedging CVaR95")
    ax.set_title("Business-Cycle Topology vs Hedging Tail Risk")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "topology_vs_hedging_cvar.png", dpi=dpi)
    plt.close()


if __name__ == "__main__":
    main()
