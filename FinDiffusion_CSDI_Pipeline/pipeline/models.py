"""One-step diffusion forecasters."""

from pathlib import Path
import sys
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FINDIFFUSION_ROOT = PROJECT_ROOT / "FinDiffusion"
if str(FINDIFFUSION_ROOT) not in sys.path:
    sys.path.insert(0, str(FINDIFFUSION_ROOT))

from src.models import GaussianDiffusion, TransformerDenoiser  # noqa: E402


class HistoryConditionEncoder(nn.Module):
    """Encode an observed return history into a condition vector."""

    def __init__(self, history_length: int, d_cond: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(history_length, d_cond),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_cond),
            nn.Linear(d_cond, d_cond),
        )

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        if history.dim() == 2:
            history = history.unsqueeze(-1)
        return self.net(history.squeeze(-1))


class FinDiffusionOneStepForecaster(nn.Module):
    """FinDiffusion-style conditional DDPM for the next return only."""

    def __init__(
        self,
        history_length: int,
        prediction_length: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        d_ff: int,
        d_cond: int,
        timesteps: int,
        beta_schedule: str,
        beta_start: float,
        beta_end: float,
        prediction_type: str,
        dropout: float,
    ) -> None:
        super().__init__()
        self.history_length = int(history_length)
        self.prediction_length = int(prediction_length)
        self.condition_encoder = HistoryConditionEncoder(history_length, d_cond, dropout)
        denoiser = TransformerDenoiser(
            input_dim=1,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            d_cond=d_cond,
            max_seq_len=prediction_length,
            dropout=dropout,
        )
        self.diffusion = GaussianDiffusion(
            denoiser=denoiser,
            timesteps=timesteps,
            beta_schedule=beta_schedule,
            beta_start=beta_start,
            beta_end=beta_end,
            prediction_type=prediction_type,
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        history = batch["history"]
        target = batch["target"]
        cond = self.condition_encoder(history)
        return self.diffusion.training_loss(target, cond=cond)

    @torch.no_grad()
    def sample(
        self,
        history: torch.Tensor,
        n_samples: int,
        use_ddim: bool = False,
        ddim_steps: int = 50,
        progress: bool = False,
    ) -> torch.Tensor:
        device = next(self.parameters()).device
        history = history.to(device)
        batch_size = history.shape[0]
        cond = self.condition_encoder(history)
        cond = cond.repeat_interleave(n_samples, dim=0)
        shape = (batch_size * n_samples, self.prediction_length, 1)
        if use_ddim:
            samples = self.diffusion.ddim_sample(
                shape, cond=cond, n_steps=ddim_steps, device=device, progress=progress
            )
        else:
            samples = self.diffusion.sample(shape, cond=cond, device=device, progress=progress)
        return samples.view(batch_size, n_samples, self.prediction_length, 1)


class CSDIStyleOneStepForecaster(nn.Module):
    """CSDI-style masked diffusion forecaster.

    The model receives the full history+future canvas plus a condition mask.
    History values are always fixed. The future values are diffused, and loss is
    computed only on the masked future positions.
    """

    def __init__(
        self,
        history_length: int,
        prediction_length: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        d_ff: int,
        timesteps: int,
        beta_schedule: str,
        beta_start: float,
        beta_end: float,
        prediction_type: str,
        dropout: float,
    ) -> None:
        super().__init__()
        if prediction_type != "epsilon":
            raise ValueError("CSDI-style forecaster currently expects prediction_type='epsilon'")
        self.history_length = int(history_length)
        self.prediction_length = int(prediction_length)
        self.seq_len = self.history_length + self.prediction_length
        self.prediction_type = prediction_type

        self.denoiser = TransformerDenoiser(
            input_dim=2,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            d_cond=None,
            max_seq_len=self.seq_len,
            dropout=dropout,
        )
        self.diffusion = GaussianDiffusion(
            denoiser=self.denoiser,
            timesteps=timesteps,
            beta_schedule=beta_schedule,
            beta_start=beta_start,
            beta_end=beta_end,
            prediction_type=prediction_type,
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        history = batch["history"]
        target = batch["target"]
        clean = torch.cat([history, target], dim=1)
        cond_mask = self._cond_mask(clean)
        target_mask = 1.0 - cond_mask

        batch_size = clean.shape[0]
        device = clean.device
        t = torch.randint(0, self.diffusion.timesteps, (batch_size,), device=device, dtype=torch.long)
        noise = torch.randn_like(clean)
        x_t, _ = self.diffusion.q_sample(clean, t, noise)

        observed_or_noisy = cond_mask * clean + target_mask * x_t
        model_input = torch.cat([observed_or_noisy, cond_mask], dim=-1)
        predicted = self.denoiser(model_input, t, cond=None)[..., :1]

        residual = (predicted - noise) * target_mask
        denom = target_mask.sum().clamp_min(1.0)
        loss = (residual.square().sum() / denom)
        return {"loss": loss}

    @torch.no_grad()
    def sample(
        self,
        history: torch.Tensor,
        n_samples: int,
        use_ddim: bool = False,
        ddim_steps: int = 50,
        progress: bool = False,
    ) -> torch.Tensor:
        if use_ddim:
            # The masked sampler keeps history fixed, so use the same update
            # loop over a reduced timestep grid for DDIM-like speed.
            return self._sample_masked(history, n_samples, ddim_steps, eta=0.0, progress=progress)
        return self._sample_masked(history, n_samples, self.diffusion.timesteps, eta=1.0, progress=progress)

    def _cond_mask(self, clean: torch.Tensor) -> torch.Tensor:
        mask = torch.zeros_like(clean)
        mask[:, : self.history_length] = 1.0
        return mask

    @torch.no_grad()
    def _sample_masked(
        self,
        history: torch.Tensor,
        n_samples: int,
        n_steps: int,
        eta: float,
        progress: bool,
    ) -> torch.Tensor:
        device = next(self.parameters()).device
        history = history.to(device)
        batch_size = history.shape[0]
        history = history.repeat_interleave(n_samples, dim=0)
        future = torch.randn(batch_size * n_samples, self.prediction_length, 1, device=device)
        cond_mask = torch.cat(
            [
                torch.ones_like(history),
                torch.zeros_like(future),
            ],
            dim=1,
        )

        if n_steps >= self.diffusion.timesteps:
            timesteps = list(reversed(range(self.diffusion.timesteps)))
        else:
            step_size = max(1, self.diffusion.timesteps // n_steps)
            timesteps = list(reversed(range(0, self.diffusion.timesteps, step_size)))

        iterator = timesteps
        if progress:
            from tqdm import tqdm

            iterator = tqdm(timesteps, desc="Masked sampling", leave=False)

        for i, step in enumerate(iterator):
            t = torch.full((history.shape[0],), int(step), device=device, dtype=torch.long)
            clean_canvas = torch.cat([history, future], dim=1)
            model_input = torch.cat([clean_canvas, cond_mask], dim=-1)
            predicted_noise = self.denoiser(model_input, t, cond=None)[..., :1]

            pred_x0 = self.diffusion.predict_x0_from_eps(clean_canvas, t, predicted_noise)
            pred_x0 = pred_x0.clamp(*self.diffusion.clip_range)

            if eta == 0.0 and n_steps < self.diffusion.timesteps:
                future = self._ddim_step(clean_canvas, pred_x0, predicted_noise, timesteps, i, t)
            else:
                mean, _variance, log_variance = self.diffusion.q_posterior_mean_variance(
                    pred_x0, clean_canvas, t
                )
                noise = torch.randn_like(clean_canvas)
                nonzero = (t != 0).float().view(-1, 1, 1)
                updated = mean + nonzero * torch.exp(0.5 * log_variance) * noise
                future = updated[:, self.history_length :]

        return future.view(batch_size, n_samples, self.prediction_length, 1)

    def _ddim_step(
        self,
        clean_canvas: torch.Tensor,
        pred_x0: torch.Tensor,
        predicted_noise: torch.Tensor,
        timesteps: List[int],
        index: int,
        t: torch.Tensor,
    ) -> torch.Tensor:
        device = clean_canvas.device
        alpha_t = self.diffusion._extract(self.diffusion.alphas_cumprod, t, clean_canvas.shape)
        if index < len(timesteps) - 1:
            alpha_prev = self.diffusion.alphas_cumprod[timesteps[index + 1]].to(device)
        else:
            alpha_prev = torch.tensor(1.0, device=device)
        updated = torch.sqrt(alpha_prev) * pred_x0 + torch.sqrt(1.0 - alpha_prev) * predicted_noise
        return updated[:, self.history_length :]


def build_model(name: str, config: Dict) -> nn.Module:
    data_cfg = config["data"]
    model_cfg = config["model"]
    common = {
        "history_length": int(data_cfg["history_length"]),
        "prediction_length": int(data_cfg["prediction_length"]),
        "d_model": int(model_cfg["d_model"]),
        "n_layers": int(model_cfg["n_layers"]),
        "n_heads": int(model_cfg["n_heads"]),
        "d_ff": int(model_cfg["d_ff"]),
        "timesteps": int(model_cfg["timesteps"]),
        "beta_schedule": str(model_cfg["beta_schedule"]),
        "beta_start": float(model_cfg["beta_start"]),
        "beta_end": float(model_cfg["beta_end"]),
        "prediction_type": str(model_cfg["prediction_type"]),
        "dropout": float(model_cfg["dropout"]),
    }
    if name == "findiffusion":
        return FinDiffusionOneStepForecaster(
            **common,
            d_cond=int(model_cfg["condition_dim"]),
        )
    if name == "csdi":
        return CSDIStyleOneStepForecaster(**common)
    raise ValueError(f"Unknown model name: {name}")
