#!/usr/bin/env python3
"""Create a compact shareable package from a CSDI comparison run."""

import argparse
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package CSDI comparison results for sharing."
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Where to create the package folder. Defaults to run_dir parent.",
    )
    parser.add_argument("--no-zip", action="store_true")
    return parser.parse_args()


def copy_file(src: Path, dst: Path, manifest: list) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(src), str(dst))
    manifest.append(str(dst))


def copy_comparison(comparison_dir: Path, package_dir: Path, manifest: list) -> None:
    wanted = [
        "comparison_report.md",
        "comparison_dashboard.png",
        "topology_dashboard.png",
        "topology_curve_overlay.png",
        "generated_market_index_paths.png",
        "generated_market_index_medians.png",
        "generated_feature_index_medians.png",
        "topoloss_minus_vanilla_deltas.png",
        "comparison_aggregate.csv",
        "comparison_by_horizon.csv",
        "comparison_by_horizon_grouped.csv",
        "comparison_delta_by_horizon.csv",
        "generated_market_index_paths.csv",
        "generated_feature_index_medians.csv",
        "topology_curves_combined.csv",
        "data_investigation_report.md",
        "generated_return_diagnostics.csv",
        "raw_data_extreme_returns.csv",
        "flagged_issues.csv",
    ]
    for name in wanted:
        target = "README.md" if name == "comparison_report.md" else name
        copy_file(comparison_dir / name, package_dir / target, manifest)


def copy_run_summaries(run_dir: Path, package_dir: Path, manifest: list) -> None:
    summaries_dir = package_dir / "per_run_summaries"
    for child in sorted(run_dir.iterdir()):
        if not child.is_dir() or child.name in {"comparison", "logs"}:
            continue
        for rel in [
            "run_config.json",
            "summary.json",
            "metrics_by_horizon.csv",
            "plots/analysis_summary.json",
            "plots/topology_summary.json",
        ]:
            src = child / rel
            safe_name = child.name + "_" + rel.replace("/", "_")
            copy_file(src, summaries_dir / safe_name, manifest)


def write_manifest(package_dir: Path, run_dir: Path, manifest: list) -> None:
    payload = {
        "source_run_dir": str(run_dir),
        "package_dir": str(package_dir),
        "files": [str(Path(path).relative_to(package_dir)) for path in manifest],
    }
    (package_dir / "MANIFEST.json").write_text(json.dumps(payload, indent=2) + "\n")


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    comparison_dir = run_dir / "comparison"
    if not comparison_dir.exists():
        raise FileNotFoundError("Missing comparison directory: %s" % comparison_dir)

    output_root = (args.output_root or run_dir.parent).resolve()
    package_dir = output_root / ("%s_shareable" % run_dir.name)
    if package_dir.exists():
        shutil.rmtree(str(package_dir))
    package_dir.mkdir(parents=True)

    manifest = []
    copy_comparison(comparison_dir, package_dir, manifest)
    copy_run_summaries(run_dir, package_dir, manifest)
    write_manifest(package_dir, run_dir, manifest)

    if not args.no_zip:
        archive_base = str(package_dir)
        archive_path = shutil.make_archive(archive_base, "zip", str(package_dir))
        print("Wrote package folder: %s" % package_dir)
        print("Wrote zip archive: %s" % archive_path)
    else:
        print("Wrote package folder: %s" % package_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
