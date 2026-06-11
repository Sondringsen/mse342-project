"""Tests for business-cycle topology diagnostics."""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PIPELINE_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_ROOT))

import business_cycle_topology as bct  # noqa: E402


def toy_predictions(offset: float = 0.0, n_days: int = 8) -> pd.DataFrame:
    rows = []
    for ticker_index, ticker in enumerate(["AAA", "BBB", "CCC"]):
        for target_index in range(n_days):
            actual = offset + 0.01 * np.sin(target_index / 2.0 + ticker_index)
            rows.append(
                {
                    "ticker": ticker,
                    "target_index": target_index,
                    "target_date": f"2020-01-{target_index + 1:02d}",
                    "actual": actual,
                    "sample_000": actual + 0.001 * (ticker_index + 1),
                    "sample_001": actual - 0.001 * (ticker_index + 1),
                }
            )
    return pd.DataFrame(rows)


class BusinessCycleTopologyTest(unittest.TestCase):
    def test_common_alignment_and_sample_panels(self) -> None:
        first = toy_predictions(n_days=6)
        second = toy_predictions(offset=0.1, n_days=5)

        tickers, indices = bct.common_tickers_and_indices([first, second])
        panel = bct.pivot_panel(first, "sample_000", tickers, indices)

        self.assertEqual(tickers, ["AAA", "BBB", "CCC"])
        self.assertEqual(indices, [0, 1, 2, 3, 4])
        self.assertEqual(panel.shape, (5, 3))
        self.assertAlmostEqual(
            panel.loc[0, "BBB"],
            first[(first["target_index"] == 0) & (first["ticker"] == "BBB")]["sample_000"].iloc[0],
        )
        self.assertEqual(bct.sample_columns(first, max_samples=1), ["sample_000"])

    def test_topology_summary_and_distance_are_finite(self) -> None:
        t = np.linspace(0.0, 2.0 * np.pi, 12)
        real = np.column_stack([np.sin(t), np.cos(t), np.sin(2.0 * t)])
        synthetic = real + 0.05

        real_summary = bct.path_topology_summary(real, max_cloud_points=8)
        synthetic_summary = bct.path_topology_summary(synthetic, max_cloud_points=8)
        distance = bct.topology_distance(synthetic_summary, real_summary)

        self.assertTrue(np.isfinite(real_summary["beta1_proxy_area"]))
        self.assertTrue(np.isfinite(real_summary["lowfreq_power_ratio"]))
        self.assertTrue(np.isfinite(distance))
        self.assertGreaterEqual(distance, 0.0)

    def test_cli_writes_business_cycle_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_a = root / "run_a" / "findiffusion"
            run_b = root / "run_b" / "findiffusion"
            run_a.mkdir(parents=True)
            run_b.mkdir(parents=True)
            toy_predictions(n_days=8).to_csv(run_a / "predictions.csv", index=False)
            toy_predictions(offset=0.02, n_days=8).to_csv(run_b / "predictions.csv", index=False)
            output_dir = root / "topology"

            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS_ROOT / "business_cycle_topology.py"),
                    str(root / "run_a"),
                    str(root / "run_b"),
                    "--labels",
                    "plain",
                    "topo",
                    "--output-dir",
                    str(output_dir),
                    "--rolling-window",
                    "4",
                    "--rolling-stride",
                    "2",
                    "--max-cloud-points",
                    "4",
                    "--max-samples",
                    "2",
                    "--min-common-days",
                    "4",
                    "--exact-h1",
                    "off",
                ],
                check=True,
            )

            summary = pd.read_csv(output_dir / "business_cycle_topology_summary.csv")
            metrics = pd.read_csv(output_dir / "business_cycle_topology_metrics.csv")

            self.assertEqual(set(summary["run_label"]), {"plain", "topo"})
            self.assertIn("topology_distance_to_real_median", summary.columns)
            self.assertEqual(set(metrics["path_type"]), {"real", "synthetic"})
            self.assertTrue((output_dir / "business_cycle_topology_report.md").exists())


if __name__ == "__main__":
    unittest.main()
