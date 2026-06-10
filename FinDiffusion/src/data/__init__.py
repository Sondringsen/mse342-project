"""Data loading and preprocessing for FinDiffusion."""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


def get_sp500_tickers() -> List[str]:
    """Fetch current S&P 500 tickers from Wikipedia."""
    try:
        table = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
        return table["Symbol"].str.replace(".", "-", regex=False).tolist()
    except Exception as e:
        logger.warning(f"Could not fetch S&P 500 tickers from Wikipedia: {e}. Using fallback list.")
        return [
            "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM", "V", "JNJ",
            "WMT", "PG", "MA", "HD", "DIS", "ADBE", "NFLX", "INTC", "CSCO", "PFE",
            "KO", "PEP", "MRK", "ABT", "TMO", "COST", "AVGO", "NKE", "MCD", "UNH",
        ]


def download_stock_data(
    tickers: List[str],
    start_date: str,
    end_date: str,
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """Download adjusted close prices via yfinance and return as a DataFrame."""
    import yfinance as yf

    logger.info(f"Downloading {len(tickers)} tickers from {start_date} to {end_date}...")
    raw = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True, progress=True)

    prices = raw["Close"] if len(tickers) > 1 else raw[["Close"]].rename(columns={"Close": tickers[0]})

    # Drop columns with more than 5% missing values, then fill forward/back
    thresh = int(len(prices) * 0.95)
    prices = prices.dropna(axis=1, thresh=thresh).ffill().bfill()

    logger.info(f"Result: {len(prices)} trading days, {len(prices.columns)} tickers")

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        prices.to_csv(save_path)
        logger.info(f"Saved prices to {save_path}")

    return prices


class _ReturnWindowDataset(Dataset):
    """Sliding-window dataset of normalised log returns (univariate, input_dim=1)."""

    def __init__(self, returns: np.ndarray, seq_len: int, stride: int):
        windows = []
        T = len(returns)
        n_assets = returns.shape[1] if returns.ndim == 2 else 1
        if returns.ndim == 1:
            returns = returns[:, np.newaxis]

        for asset in range(n_assets):
            col = returns[:, asset]
            for start in range(0, T - seq_len + 1, stride):
                w = col[start : start + seq_len, np.newaxis]  # (seq_len, 1)
                windows.append(w)

        self.data = torch.tensor(np.array(windows, dtype=np.float32))  # (N, seq_len, 1)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]


class FinancialDataModule:
    """Downloads, preprocesses, and serves financial return data for training."""

    def __init__(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        seq_len: int = 252,
        stride: int = 21,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        batch_size: int = 64,
        data_dir: str = "data",
        num_workers: int = 0,
    ):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.seq_len = seq_len
        self.stride = stride
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.batch_size = batch_size
        self.data_dir = Path(data_dir)
        self.num_workers = num_workers

        self.mean: Optional[float] = None
        self.std: Optional[float] = None

        self.train_dataset: Optional[_ReturnWindowDataset] = None
        self.val_dataset: Optional[_ReturnWindowDataset] = None
        self.test_dataset: Optional[_ReturnWindowDataset] = None

    def _load_or_download(self) -> pd.DataFrame:
        prices_path = self.data_dir / "prices.csv"
        if prices_path.exists():
            logger.info(f"Loading cached prices from {prices_path}")
            return pd.read_csv(prices_path, index_col=0, parse_dates=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        return download_stock_data(
            tickers=self.tickers,
            start_date=self.start_date,
            end_date=self.end_date,
            save_path=str(prices_path),
        )

    def setup(self):
        """Download data, compute log returns, normalize, and create train/val/test datasets."""
        prices = self._load_or_download()

        log_returns = np.log(prices / prices.shift(1)).dropna().values  # (T, N)

        T = len(log_returns)
        train_end = int(T * self.train_ratio)
        val_end = int(T * (self.train_ratio + self.val_ratio))

        train_raw = log_returns[:train_end]
        val_raw = log_returns[train_end:val_end]
        test_raw = log_returns[val_end:]

        # Fit normalization on training data only
        self.mean = float(train_raw.mean())
        self.std = float(train_raw.std())

        def _norm(x: np.ndarray) -> np.ndarray:
            return (x - self.mean) / self.std

        self.train_dataset = _ReturnWindowDataset(_norm(train_raw), self.seq_len, self.stride)
        self.val_dataset = _ReturnWindowDataset(_norm(val_raw), self.seq_len, self.stride)
        self.test_dataset = _ReturnWindowDataset(_norm(test_raw), self.seq_len, self.stride)

        logger.info(
            f"Dataset split — train: {len(self.train_dataset)}, "
            f"val: {len(self.val_dataset)}, test: {len(self.test_dataset)}"
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def denormalize(self, x: np.ndarray) -> np.ndarray:
        """Reverse z-score normalization to recover raw log returns."""
        if self.mean is None or self.std is None:
            raise RuntimeError("Call setup() before denormalize()")
        return x * self.std + self.mean

    def state_dict(self) -> Dict:
        return {
            "mean": self.mean,
            "std": self.std,
            "tickers": self.tickers,
            "seq_len": self.seq_len,
            "stride": self.stride,
            "train_ratio": self.train_ratio,
            "val_ratio": self.val_ratio,
        }

    def load_state_dict(self, state: Dict):
        self.mean = state["mean"]
        self.std = state["std"]
