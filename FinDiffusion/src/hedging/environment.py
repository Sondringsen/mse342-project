"""Hedging environment: price path construction and P&L computation."""

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn


def log_returns_to_prices(log_returns: np.ndarray, s0: float = 100.0) -> np.ndarray:
    """Convert log-return sequences to price paths.

    Args:
        log_returns: (N, T) array of daily log returns
        s0: initial stock price
    Returns:
        (N, T+1) price paths with prices[:, 0] == s0
    """
    N, T = log_returns.shape
    prices = np.empty((N, T + 1), dtype=np.float32)
    prices[:, 0] = s0
    prices[:, 1:] = s0 * np.exp(np.cumsum(log_returns, axis=1))
    return prices


def compute_pnl(
    prices: torch.Tensor,
    model: nn.Module,
    strike: float,
    seq_len: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute hedging P&L for all paths.

    We are the seller of a European call: we collect (implicitly) the premium,
    delta-hedge throughout the option's life, and pay the payoff at maturity.
    P&L = hedging gains − option payoff.

    Args:
        prices: (N, seq_len+1) price tensor
        model: DeepHedger
        strike: option strike price
        seq_len: number of hedging steps
    Returns:
        pnl: (N,) terminal P&L
        delta_paths: (N, seq_len) hedge ratios (detached, for inspection)
    """
    N = prices.shape[0]
    device = prices.device
    cumulative_gains = torch.zeros(N, device=device)
    delta_list = []

    for t in range(seq_len):
        log_moneyness = torch.log(prices[:, t] / strike)
        ttm = torch.full((N,), (seq_len - t) / seq_len, device=device)
        state = torch.stack([log_moneyness, ttm], dim=1)

        delta = model(state)                           # (N,) — grads flow through here
        delta_list.append(delta.detach())
        price_change = prices[:, t + 1] - prices[:, t]
        cumulative_gains = cumulative_gains + delta * price_change

    payoff = torch.clamp(prices[:, -1] - strike, min=0.0)
    pnl = cumulative_gains - payoff
    delta_paths = torch.stack(delta_list, dim=1)       # (N, seq_len)
    return pnl, delta_paths


def cvar_loss(pnl: torch.Tensor, alpha: float = 0.95) -> torch.Tensor:
    """CVaR training loss: minimise the expected loss in the worst (1−alpha) tail."""
    losses = -pnl
    var = torch.quantile(losses, alpha)
    tail = losses[losses >= var]
    return tail.mean()
