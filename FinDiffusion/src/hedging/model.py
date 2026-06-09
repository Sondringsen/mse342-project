"""Deep hedging network: 5 linear layers with LayerNorm and ReLU."""

import torch
import torch.nn as nn


class DeepHedger(nn.Module):
    """
    MLP hedger: maps (log-moneyness, time-to-maturity) to a hedge ratio in [0, 1].

    Architecture: 4 hidden blocks of (Linear → LayerNorm → ReLU) plus a final
    Linear → Sigmoid output layer — 5 linear layers in total.
    """

    def __init__(self, input_dim: int = 2, hidden_dim: int = 64):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(4):
            layers += [nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU()]
            in_dim = hidden_dim
        layers += [nn.Linear(hidden_dim, 1), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, input_dim) state tensor — [log(S/K), ttm]
        Returns:
            (N,) hedge ratios in [0, 1]
        """
        return self.net(x).squeeze(-1)
