#!/usr/bin/env python
"""Download financial data for training."""

import argparse
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import download_stock_data, get_sp500_tickers

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Download stock data")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config YAML")
    parser.add_argument("--tickers", type=str, help="Comma-separated tickers (overrides config)")
    parser.add_argument("--start", type=str, help="Start date YYYY-MM-DD (overrides config)")
    parser.add_argument("--end", type=str, help="End date YYYY-MM-DD (overrides config)")
    parser.add_argument("--output", type=str, default="data/prices.csv", help="Output path")
    parser.add_argument("--n_tickers", type=int, help="Number of tickers if using SP500 (overrides config)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    data_cfg = cfg.get("data", {})

    start = args.start or data_cfg.get("start_date", "2010-01-01")
    end = args.end or data_cfg.get("end_date", "2024-01-01")

    if args.tickers:
        tickers = [t.strip() for t in args.tickers.split(",")]
    else:
        cfg_tickers = data_cfg.get("tickers")
        if cfg_tickers:
            n = args.n_tickers or len(cfg_tickers)
            tickers = cfg_tickers[:n]
        else:
            n = args.n_tickers or 30
            tickers = get_sp500_tickers()[:n]

    logger.info(f"Downloading {len(tickers)} tickers from {start} to {end}")

    prices = download_stock_data(
        tickers=tickers,
        start_date=start,
        end_date=end,
        save_path=args.output,
    )

    logger.info(f"Downloaded {len(prices)} trading days, {len(prices.columns)} tickers")
    logger.info(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
