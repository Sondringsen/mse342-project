"""Data utilities for rolling return-horizon forecasting."""

from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


TRADING_DAYS_PER_YEAR = 252


class SplitIndices:
    def __init__(self, train_end: int, val_end: int, n_rows: int) -> None:
        self.train_end = train_end
        self.val_end = val_end
        self.n_rows = n_rows


class NormalizerState:
    def __init__(self, mean: float, std: float) -> None:
        self.mean = mean
        self.std = std


class OneStepReturnDataset(Dataset):
    """Univariate rolling windows: observed history plus future target."""

    def __init__(
        self,
        returns: pd.DataFrame,
        starts: Iterable[int],
        tickers: List[str],
        history_length: int,
        prediction_length: int,
        normalizer: NormalizerState,
    ) -> None:
        self.returns = returns[tickers].astype(np.float32)
        self.values = self.returns.to_numpy(np.float32)
        self.dates = pd.to_datetime(self.returns.index).date.astype(str).tolist()
        self.starts = list(starts)
        self.tickers = list(tickers)
        self.history_length = int(history_length)
        self.prediction_length = int(prediction_length)
        self.normalizer = normalizer

        samples = []  # type: List[Tuple[int, int]]
        total_length = self.history_length + self.prediction_length
        for asset_idx, _ticker in enumerate(self.tickers):
            asset_values = self.values[:, asset_idx]
            for start in self.starts:
                stop = start + total_length
                if stop <= len(asset_values) and np.isfinite(asset_values[start:stop]).all():
                    samples.append((asset_idx, start))
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        asset_idx, start = self.samples[index]
        hist_stop = start + self.history_length
        target_stop = hist_stop + self.prediction_length

        window = self.values[start:target_stop, asset_idx]
        history_raw = window[: self.history_length]
        target_raw = window[self.history_length :]

        history = self._normalize(history_raw)[:, None]
        target = self._normalize(target_raw)[:, None]

        return {
            "history": torch.from_numpy(history),
            "target": torch.from_numpy(target),
            "target_raw": torch.from_numpy(target_raw.astype(np.float32)[:, None]),
            "asset_index": torch.tensor(asset_idx, dtype=torch.long),
            "start_index": torch.tensor(start, dtype=torch.long),
            "target_index": torch.tensor(hist_stop, dtype=torch.long),
        }

    def metadata(self, asset_index: int, target_index: int) -> Tuple[str, str]:
        return self.tickers[int(asset_index)], self.dates[int(target_index)]

    def denormalize(self, values: np.ndarray) -> np.ndarray:
        return values * self.normalizer.std + self.normalizer.mean

    def _normalize(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.normalizer.mean) / self.normalizer.std).astype(np.float32)


def load_returns(config: Dict[str, Any], allow_download: bool = True) -> pd.DataFrame:
    """Load cached returns or download Yahoo prices and compute log returns."""

    data_cfg = config["data"]
    tickers = list(data_cfg["tickers"])
    data_dir = Path(data_cfg["data_dir"])
    data_dir.mkdir(parents=True, exist_ok=True)
    returns_path = data_dir / data_cfg.get("returns_cache", "log_returns.csv")
    price_path = data_dir / data_cfg.get("price_cache", "yahoo_prices.csv")

    if returns_path.exists():
        returns = pd.read_csv(returns_path, index_col=0, parse_dates=True)
        if has_columns(returns, tickers):
            return returns[tickers].dropna(how="all")
        if not allow_download and not price_path.exists():
            raise_missing_cache_error(returns_path, tickers, returns.columns)

    if price_path.exists():
        prices = pd.read_csv(price_path, index_col=0, parse_dates=True)
        if not has_columns(prices, tickers):
            if not allow_download:
                raise_missing_cache_error(price_path, tickers, prices.columns)
            prices = download_yahoo_prices(
                tickers=tickers,
                start_date=data_cfg["start_date"],
                end_date=data_cfg["end_date"],
            )
            prices.to_csv(price_path)
    else:
        if not allow_download:
            raise FileNotFoundError(
                f"No cached returns at {returns_path} and no cached prices at {price_path}"
            )
        prices = download_yahoo_prices(
            tickers=tickers,
            start_date=data_cfg["start_date"],
            end_date=data_cfg["end_date"],
        )
        prices.to_csv(price_path)

    returns = compute_log_returns(prices)
    returns.to_csv(returns_path)
    if not has_columns(returns, tickers):
        raise_missing_cache_error(returns_path, tickers, returns.columns)
    return returns[tickers].dropna(how="all")


def has_columns(frame: pd.DataFrame, columns: List[str]) -> bool:
    return set(columns).issubset(set(frame.columns))


def raise_missing_cache_error(path: Path, tickers: List[str], columns) -> None:
    missing = [ticker for ticker in tickers if ticker not in set(columns)]
    raise FileNotFoundError(
        "Cached data at %s is missing requested tickers: %s. "
        "Rerun without --no-download to refresh the cache."
        % (path, ", ".join(missing))
    )


def download_yahoo_prices(tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    """Download adjusted close prices from Yahoo Finance."""

    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("Install yfinance to download prices, or provide cached CSVs") from exc

    raw = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
        group_by="column",
    )
    if raw.empty:
        raise RuntimeError("Yahoo Finance returned no rows")

    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            prices = raw["Close"]
        elif "Adj Close" in raw.columns.get_level_values(0):
            prices = raw["Adj Close"]
        else:
            raise RuntimeError("Could not find Close or Adj Close columns in Yahoo data")
    else:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

    prices = prices.reindex(columns=tickers)
    prices.index.name = "date"
    return prices.dropna(how="all")


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    prices = prices.sort_index().replace([np.inf, -np.inf], np.nan)
    returns = np.log(prices).diff()
    returns.index.name = "date"
    return returns.replace([np.inf, -np.inf], np.nan).dropna(how="all")


def split_indices(n_rows: int, train_ratio: float, val_ratio: float) -> SplitIndices:
    train_end = int(n_rows * train_ratio)
    val_end = train_end + int(n_rows * val_ratio)
    return SplitIndices(train_end=train_end, val_end=val_end, n_rows=n_rows)


def fit_normalizer(returns: pd.DataFrame, train_end: int) -> NormalizerState:
    train_values = returns.iloc[:train_end].to_numpy(np.float32)
    observed = train_values[np.isfinite(train_values)]
    if observed.size == 0:
        raise ValueError("No finite training returns available for normalization")
    mean = float(observed.mean())
    std = float(observed.std())
    if not np.isfinite(std) or std <= 0:
        std = 1.0
    return NormalizerState(mean=mean, std=std)


def make_starts(
    split_start: int,
    split_end: int,
    history_length: int,
    prediction_length: int,
    stride: int,
) -> List[int]:
    total_length = history_length + prediction_length
    last_start = split_end - total_length
    if last_start < split_start:
        return []
    return list(range(split_start, last_start + 1, stride))


def build_datasets(
    config: Dict[str, Any],
    returns: pd.DataFrame,
) -> Tuple[Dict[str, OneStepReturnDataset], SplitIndices]:
    data_cfg = config["data"]
    tickers = list(data_cfg["tickers"])
    history_length = int(data_cfg["history_length"])
    prediction_length = int(data_cfg["prediction_length"])
    stride = int(data_cfg["stride"])

    splits = split_indices(len(returns), float(data_cfg["train_ratio"]), float(data_cfg["val_ratio"]))
    normalizer = fit_normalizer(returns[tickers], splits.train_end)

    train_starts = make_starts(0, splits.train_end, history_length, prediction_length, stride)
    val_starts = make_starts(splits.train_end, splits.val_end, history_length, prediction_length, stride)
    test_starts = make_starts(splits.val_end, splits.n_rows, history_length, prediction_length, 1)

    datasets = {
        "train": OneStepReturnDataset(
            returns, train_starts, tickers, history_length, prediction_length, normalizer
        ),
        "val": OneStepReturnDataset(
            returns, val_starts, tickers, history_length, prediction_length, normalizer
        ),
        "test": OneStepReturnDataset(
            returns, test_starts, tickers, history_length, prediction_length, normalizer
        ),
    }
    return datasets, splits


def make_loader(
    dataset: OneStepReturnDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
