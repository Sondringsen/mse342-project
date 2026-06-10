"""Tests for comparison analysis artifact generation."""

from pathlib import Path
import json
import shutil
import tempfile
import unittest

import pandas as pd

import sys

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE_ROOT))

from pipeline.analysis import build_summary_frame, write_comparison_analysis
from pipeline.output_index import write_outputs_index


def sample_result(model: str, offset: float = 0.0) -> dict:
    return {
        "model": model,
        "forecast": {
            "median_mae": 0.010 + offset,
            "median_rmse": 0.015 + offset,
            "mean_mae": 0.011 + offset,
            "mean_rmse": 0.016 + offset,
            "bias": -0.001 + offset,
            "coverage_50": 0.52,
            "coverage_90": 0.88,
            "avg_width_50": 0.02,
            "avg_width_90": 0.05,
        },
        "metrics": {
            "distribution": {
                "wasserstein": 0.006 + offset,
                "ks_statistic": 0.14 + offset,
                "js_divergence": 0.09 + offset,
                "mean_diff": 0.001,
                "std_diff": 0.005,
                "skew_diff": 1.7,
                "kurtosis_diff": 12.6,
            },
            "temporal": {
                "acf_mae": 0.12,
                "acf_squared_mae": 0.16 + offset,
                "vol_cluster_real": 0.08,
                "vol_cluster_syn": 0.03,
            },
            "diversity": {
                "mean_pairwise_corr": 0.01,
                "std_pairwise_corr": 0.13,
                "mean_sample_std": 0.019,
                "std_sample_std": 0.002,
                "unique_ratio": 1.0,
            },
            "summary": {
                "distribution_score": 0.99,
                "temporal_score": 0.86,
                "diversity_score": 1.0,
                "overall_score": 0.95 - offset,
            },
        },
        "stylized_facts": {
            "real": stylized_block(pass_rate=0.5, vol_passed=True),
            "synthetic": stylized_block(pass_rate=0.5 - offset, vol_passed=False),
            "comparison": {
                "statistics": {
                    "mean": {"real": 0.001, "synthetic": 0.0002, "difference": -0.0008},
                    "std": {"real": 0.014, "synthetic": 0.019, "difference": 0.005},
                    "skewness": {"real": -1.8, "synthetic": -0.1, "difference": 1.7},
                    "kurtosis": {"real": 13.1, "synthetic": 0.5, "difference": -12.6},
                },
                "ks_statistic": 0.14 + offset,
                "ks_pvalue": 0.002,
                "wasserstein_distance": 0.006 + offset,
            },
        },
        "path_shapes": {"real": [3, 64], "synthetic": [24, 64]},
    }


def stylized_block(pass_rate: float, vol_passed: bool) -> dict:
    tests_passed = int(round(pass_rate * 4))
    return {
        "fat_tails": {
            "test": "fat_tails",
            "passed": False,
            "n_sequences": 3,
            "excess_kurtosis": 4.1,
            "skewness": -0.9,
            "jarque_bera_stat": 144.0,
            "jarque_bera_pvalue": 0.35,
            "tail_index": 3.3,
        },
        "volatility_clustering": {
            "test": "volatility_clustering",
            "passed": vol_passed,
            "n_sequences": 3,
            "acf_squared_lag1": 0.075,
            "acf_squared_lag5": 0.001,
            "acf_squared_lag10": -0.14,
            "acf_absolute_lag1": 0.06,
            "ljung_box_pvalue": 0.65,
        },
        "leverage_effect": {
            "test": "leverage_effect",
            "passed": True,
            "n_sequences": 3,
            "leverage_correlation": -0.023,
            "vol_after_negative": 0.0105,
            "vol_after_positive": 0.0101,
            "asymmetry_ratio": 1.04,
        },
        "no_autocorrelation": {
            "test": "no_autocorrelation",
            "passed": True,
            "n_sequences": 3,
            "acf_lag1": -0.05,
            "acf_lag5": -0.02,
            "ljung_box_pvalue": 0.61,
        },
        "summary": {
            "tests_passed": tests_passed,
            "tests_total": 4,
            "pass_rate": pass_rate,
            "overall_quality": "Fair",
        },
    }


class AnalysisTest(unittest.TestCase):
    def test_summary_frame_keeps_findiffusion_metrics(self) -> None:
        summary = build_summary_frame([sample_result("findiffusion"), sample_result("csdi", 0.001)])
        self.assertEqual(set(summary["model"]), {"findiffusion", "csdi"})
        for column in [
            "metric_distribution_skew_diff",
            "metric_distribution_kurtosis_diff",
            "metric_temporal_acf_squared_mae",
            "stylized_synthetic_fat_tails_skewness",
            "stylized_synthetic_volatility_clustering_acf_squared_lag1",
            "stylized_synthetic_leverage_effect_leverage_correlation",
            "stylized_synthetic_no_autocorrelation_acf_lag1",
            "dist_comparison_statistics_skewness_difference",
        ]:
            self.assertIn(column, summary.columns)

    def test_write_comparison_analysis_outputs_report_surface(self) -> None:
        tmpdir = Path(tempfile.mkdtemp())
        try:
            write_comparison_analysis(
                [sample_result("findiffusion"), sample_result("csdi", 0.001)],
                tmpdir,
            )
            expected = [
                "comparison_summary.csv",
                "comparison_forecast_metrics.csv",
                "comparison_distribution_metrics.csv",
                "comparison_temporal_metrics.csv",
                "comparison_diversity_metrics.csv",
                "comparison_score_metrics.csv",
                "comparison_stylized_facts.csv",
                "comparison_metric_rankings.csv",
                "comparison_report.md",
                "README.md",
                "comparison_metrics_report.txt",
            ]
            for name in expected:
                self.assertTrue((tmpdir / name).exists(), name)

            summary = pd.read_csv(tmpdir / "comparison_summary.csv")
            self.assertIn("metric_distribution_js_divergence", summary.columns)
            self.assertIn("stylized_synthetic_summary_pass_rate", summary.columns)

            report = (tmpdir / "comparison_report.md").read_text()
            self.assertIn("## Stylized Facts", report)
            self.assertIn("Sq Return ACF1", report)

            run_readme = (tmpdir / "README.md").read_text()
            self.assertIn("Full comparison report", run_readme)
            self.assertIn("Generated return time series", run_readme)
        finally:
            shutil.rmtree(tmpdir)

    def test_outputs_index_marks_complete_and_partial_runs(self) -> None:
        tmpdir = Path(tempfile.mkdtemp())
        try:
            complete = tmpdir / "complete_run"
            partial = tmpdir / "partial_run"
            write_comparison_analysis(
                [sample_result("findiffusion"), sample_result("csdi", 0.001)],
                complete,
            )
            for model_name in ["findiffusion", "csdi"]:
                (complete / model_name).mkdir(parents=True)
                (complete / model_name / "evaluation_results.json").write_text(
                    json.dumps(sample_result(model_name)) + "\n"
                )
            (partial / "findiffusion").mkdir(parents=True)
            (partial / "findiffusion" / "evaluation_results.json").write_text(
                json.dumps(sample_result("findiffusion")) + "\n"
            )

            index = write_outputs_index(tmpdir)
            self.assertTrue((tmpdir / "README.md").exists())
            self.assertTrue((tmpdir / "index.csv").exists())
            self.assertTrue((tmpdir / "LATEST_COMPLETE_RUN.txt").exists())
            self.assertTrue((tmpdir / "LATEST_RUN.txt").exists())
            statuses = dict(zip(index["run_name"], index["status"]))
            self.assertEqual(statuses["complete_run"], "complete")
            self.assertEqual(statuses["partial_run"], "partial")

            outputs_readme = (tmpdir / "README.md").read_text()
            self.assertIn("Latest complete run", outputs_readme)
            self.assertIn("complete_run", outputs_readme)
        finally:
            shutil.rmtree(tmpdir)


if __name__ == "__main__":
    unittest.main()
