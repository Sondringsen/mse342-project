"""Training loop for the deep hedging model."""

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from .model import DeepHedger
from .environment import log_returns_to_prices, compute_pnl, cvar_loss

logger = logging.getLogger(__name__)


class HedgingTrainer:
    def __init__(
        self,
        model: DeepHedger,
        strike: float,
        seq_len: int,
        lr: float = 1e-3,
        cvar_alpha: float = 0.95,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.strike = strike
        self.seq_len = seq_len
        self.alpha = cvar_alpha
        self.device = device or torch.device("cpu")
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.model.to(self.device)

    def train(
        self,
        synthetic_log_returns: np.ndarray,
        n_epochs: int = 100,
        batch_size: int = 256,
        checkpoint_path: Optional[Path] = None,
        s0: float = 100.0,
    ) -> List[float]:
        """Train on synthetic log-return paths.

        Args:
            synthetic_log_returns: (N, T) array, T >= seq_len
            n_epochs: training epochs
            batch_size: paths per gradient step
            checkpoint_path: save best model here
            s0: initial stock price for price path construction
        Returns:
            list of per-epoch CVaR losses
        """
        returns = synthetic_log_returns[:, : self.seq_len]
        prices_np = log_returns_to_prices(returns, s0=s0)
        prices_all = torch.tensor(prices_np, device=self.device)

        N = len(prices_all)
        epoch_losses: List[float] = []
        best_loss = float("inf")

        for epoch in range(n_epochs):
            self.model.train()
            perm = torch.randperm(N)
            batch_losses = []

            for start in range(0, N, batch_size):
                idx = perm[start : start + batch_size]
                prices = prices_all[idx]

                self.optimizer.zero_grad()
                pnl, _ = compute_pnl(prices, self.model, self.strike, self.seq_len)
                loss = cvar_loss(pnl, self.alpha)
                loss.backward()
                self.optimizer.step()

                batch_losses.append(loss.item())

            epoch_loss = float(np.mean(batch_losses))
            epoch_losses.append(epoch_loss)

            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch:4d}/{n_epochs}  CVaR({self.alpha:.0%}) loss: {epoch_loss:.6f}")

            if checkpoint_path is not None and epoch_loss < best_loss:
                best_loss = epoch_loss
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {"model_state_dict": self.model.state_dict(), "epoch": epoch, "loss": best_loss},
                    checkpoint_path,
                )

        return epoch_losses
