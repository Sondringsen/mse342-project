"""Tests for hedging evaluation adapter."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "hedging_from_predictions.py"
spec = importlib.util.spec_from_file_location("hedging_from_predictions", SCRIPT_PATH)
hedging_from_predictions = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(hedging_from_predictions)


def test_horizon_predictions_are_reconstructed_into_paths():
    rows = []
    for ticker in ["AAA", "BBB"]:
        for forecast_start_index in [10, 12]:
            for horizon_step in [1, 2]:
                target_index = forecast_start_index + horizon_step - 1
                value = float(target_index)
                rows.append(
                    {
                        "ticker": ticker,
                        "forecast_start_index": forecast_start_index,
                        "target_index": target_index,
                        "horizon_step": horizon_step,
                        "actual": value,
                        "sample_000": value + 100.0,
                        "sample_001": value + 200.0,
                    }
                )
    predictions = pd.DataFrame(rows)

    real, synthetic = hedging_from_predictions.paths_from_predictions(predictions)

    assert real.shape == (2, 4)
    assert synthetic.shape == (4, 4)
    np.testing.assert_allclose(real[0], [10.0, 11.0, 12.0, 13.0])
    np.testing.assert_allclose(synthetic[0], [110.0, 111.0, 112.0, 113.0])
    np.testing.assert_allclose(synthetic[1], [210.0, 211.0, 212.0, 213.0])


def test_window_paths_respects_stride_and_limit():
    paths = np.asarray(
        [
            np.arange(6, dtype=np.float32),
            np.arange(10, 16, dtype=np.float32),
        ]
    )
    rng = np.random.default_rng(123)

    windows = hedging_from_predictions.window_paths(
        paths,
        seq_len=3,
        stride=2,
        max_windows=0,
        rng=rng,
    )

    assert windows.shape == (4, 3)
    np.testing.assert_allclose(windows[0], [0.0, 1.0, 2.0])
    np.testing.assert_allclose(windows[1], [2.0, 3.0, 4.0])
    np.testing.assert_allclose(windows[2], [10.0, 11.0, 12.0])
    np.testing.assert_allclose(windows[3], [12.0, 13.0, 14.0])

    limited = hedging_from_predictions.window_paths(
        paths,
        seq_len=3,
        stride=2,
        max_windows=2,
        rng=np.random.default_rng(123),
    )
    assert limited.shape == (2, 3)
