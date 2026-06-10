"""Tests for horizon-aware forecast evaluation helpers."""

from pathlib import Path
import sys
import unittest

import pandas as pd

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE_ROOT))

from pipeline.evaluation import forecast_metrics, paths_from_predictions, select_eval_indices


class FakeDataset:
    def __init__(self) -> None:
        self.history_length = 10
        self.prediction_length = 3
        self.samples = [(0, start) for start in range(10)]


class EvaluationTest(unittest.TestCase):
    def test_horizon_paths_concatenate_blocks_by_ticker_and_sample(self) -> None:
        predictions = pd.DataFrame(
            [
                row
                for forecast_start, actuals, sample0, sample1 in [
                    (10, [1.0, 2.0], [10.0, 20.0], [100.0, 200.0]),
                    (12, [3.0, 4.0], [30.0, 40.0], [300.0, 400.0]),
                ]
                for row in [
                    {
                        "ticker": "AAA",
                        "forecast_start_index": forecast_start,
                        "target_index": forecast_start + horizon_offset,
                        "horizon_step": horizon_offset + 1,
                        "actual": actuals[horizon_offset],
                        "sample_000": sample0[horizon_offset],
                        "sample_001": sample1[horizon_offset],
                    }
                    for horizon_offset in range(2)
                ]
            ]
        )

        real, synthetic = paths_from_predictions(predictions)

        self.assertEqual(real.tolist(), [[1.0, 2.0, 3.0, 4.0]])
        self.assertEqual(
            synthetic.tolist(),
            [[10.0, 20.0, 30.0, 40.0], [100.0, 200.0, 300.0, 400.0]],
        )

    def test_forecast_metrics_include_per_horizon_blocks(self) -> None:
        predictions = pd.DataFrame(
            {
                "horizon_step": [1, 1, 2, 2],
                "actual": [0.0, 1.0, 2.0, 3.0],
                "pred_mean": [0.0, 1.0, 1.0, 5.0],
                "pred_median": [0.0, 1.0, 1.0, 5.0],
                "pred_q05": [-1.0, 0.0, 0.0, 4.0],
                "pred_q25": [-0.5, 0.5, 0.5, 4.5],
                "pred_q75": [0.5, 1.5, 1.5, 5.5],
                "pred_q95": [1.0, 2.0, 2.0, 6.0],
            }
        )

        metrics = forecast_metrics(predictions)

        self.assertEqual(metrics["horizon_count"], 2)
        self.assertEqual(metrics["max_horizon"], 2)
        self.assertEqual(metrics["by_horizon"]["step_01"]["median_mae"], 0.0)
        self.assertEqual(metrics["by_horizon"]["step_02"]["median_mae"], 1.5)

    def test_select_eval_indices_uses_non_overlapping_horizon_blocks(self) -> None:
        selected = select_eval_indices(FakeDataset(), max_windows_per_asset=2)
        target_indices = [FakeDataset().samples[idx][1] + FakeDataset().history_length for idx in selected]

        self.assertEqual(target_indices, [16, 19])
        self.assertGreaterEqual(target_indices[1] - target_indices[0], FakeDataset().prediction_length)


if __name__ == "__main__":
    unittest.main()
