"""Tests for the current FinDiffusion data module."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import FinancialDataModule  # noqa: E402


@pytest.fixture
def cached_price_dir(tmp_path):
    n_days = 80
    dates = pd.date_range("2020-01-01", periods=n_days, freq="B")
    returns = np.column_stack(
        [
            np.full(n_days, 0.001),
            np.linspace(-0.002, 0.003, n_days),
        ]
    )
    prices = pd.DataFrame(
        100.0 * np.exp(np.cumsum(returns, axis=0)),
        index=dates,
        columns=["AAA", "BBB"],
    )
    prices.to_csv(tmp_path / "prices.csv")
    return tmp_path


def make_module(data_dir: Path) -> FinancialDataModule:
    return FinancialDataModule(
        tickers=["AAA", "BBB"],
        start_date="2020-01-01",
        end_date="2020-04-30",
        seq_len=10,
        stride=5,
        train_ratio=0.6,
        val_ratio=0.2,
        batch_size=4,
        data_dir=str(data_dir),
    )


def test_setup_uses_cached_prices_and_builds_splits(cached_price_dir):
    dm = make_module(cached_price_dir)

    dm.setup()

    assert dm.train_dataset is not None
    assert dm.val_dataset is not None
    assert dm.test_dataset is not None
    assert len(dm.train_dataset) > 0
    assert len(dm.val_dataset) > 0
    assert len(dm.test_dataset) > 0
    sample = dm.train_dataset[0]
    assert isinstance(sample, torch.Tensor)
    assert sample.shape == (10, 1)
    assert sample.dtype == torch.float32
    assert dm.mean is not None
    assert dm.std is not None


def test_denormalize_recovers_raw_train_returns(cached_price_dir):
    prices = pd.read_csv(cached_price_dir / "prices.csv", index_col=0, parse_dates=True)
    raw_returns = np.log(prices / prices.shift(1)).dropna().values
    train_end = int(len(raw_returns) * 0.6)
    train_raw = raw_returns[:train_end]
    dm = make_module(cached_price_dir)

    dm.setup()
    sample = dm.train_dataset[0].numpy()
    recovered = dm.denormalize(sample)[:, 0]

    np.testing.assert_allclose(recovered, train_raw[:10, 0], rtol=1e-5, atol=1e-7)


def test_state_dict_round_trip(cached_price_dir):
    dm = make_module(cached_price_dir)
    dm.setup()

    state = dm.state_dict()
    restored = make_module(cached_price_dir)
    restored.load_state_dict(state)

    assert restored.mean == dm.mean
    assert restored.std == dm.std
    assert restored.tickers == dm.tickers
    assert restored.seq_len == dm.seq_len
    assert restored.stride == dm.stride
