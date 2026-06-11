"""Auxiliary training losses for market-realistic return paths."""

from __future__ import annotations

import logging
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np
import torch
from torch import Tensor


LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FINDIFFUSION_ROOT = PROJECT_ROOT / "FinDiffusion"
if str(FINDIFFUSION_ROOT) not in sys.path:
    sys.path.insert(0, str(FINDIFFUSION_ROOT))


DEFAULT_REALISM_WEIGHTS = {
    "squared_acf": 0.10,
    "absolute_acf": 0.05,
    "kurtosis": 0.05,
    "leverage": 0.05,
}


LOSS_PROFILES = {
    "none": {
        "realism": {"enabled": False},
        "topology": {"enabled": False},
    },
    "vol_only": {
        "realism": {
            "enabled": True,
            "weights": {
                "squared_acf": DEFAULT_REALISM_WEIGHTS["squared_acf"],
                "absolute_acf": DEFAULT_REALISM_WEIGHTS["absolute_acf"],
                "kurtosis": 0.0,
                "leverage": 0.0,
            },
        },
        "topology": {"enabled": False},
    },
    "vol_tail": {
        "realism": {
            "enabled": True,
            "weights": {
                "squared_acf": DEFAULT_REALISM_WEIGHTS["squared_acf"],
                "absolute_acf": DEFAULT_REALISM_WEIGHTS["absolute_acf"],
                "kurtosis": DEFAULT_REALISM_WEIGHTS["kurtosis"],
                "leverage": 0.0,
            },
        },
        "topology": {"enabled": False},
    },
    "vol_tail_leverage": {
        "realism": {
            "enabled": True,
            "weights": dict(DEFAULT_REALISM_WEIGHTS),
        },
        "topology": {"enabled": False},
    },
    "realism_topology": {
        "realism": {
            "enabled": True,
            "weights": dict(DEFAULT_REALISM_WEIGHTS),
        },
        "topology": {"enabled": True, "evaluate": True},
    },
}


def apply_loss_profile(config: Dict[str, Any], profile: str) -> None:
    """Apply a named loss profile to the run configuration."""

    if profile not in LOSS_PROFILES:
        raise ValueError(f"Unknown loss profile: {profile}")

    losses_cfg = config.setdefault("losses", {})
    profile_cfg = LOSS_PROFILES[profile]
    for section in ("realism", "topology"):
        section_cfg = losses_cfg.setdefault(section, {})
        section_cfg.update(profile_cfg.get(section, {}))
        if section == "realism" and "weights" in profile_cfg.get(section, {}):
            weights_cfg = section_cfg.setdefault("weights", {})
            weights_cfg.update(profile_cfg[section]["weights"])


class AdaptiveLossNormalizer:
    """Normalize auxiliary losses by detached EMA magnitudes."""

    def __init__(self, decay: float = 0.98, eps: float = 1e-8) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"loss normalizer decay must be in [0, 1), got {decay}")
        if eps <= 0.0:
            raise ValueError(f"loss normalizer eps must be positive, got {eps}")
        self.decay = float(decay)
        self.eps = float(eps)
        self.scales: Dict[str, float] = {}

    def normalize(self, name: str, loss: Tensor, update: bool) -> Tensor:
        value = float(loss.detach().abs().mean().cpu())
        value = max(value, self.eps)
        if update or name not in self.scales:
            old = self.scales.get(name)
            self.scales[name] = value if old is None else self.decay * old + (1.0 - self.decay) * value
        scale = max(self.scales.get(name, value), self.eps)
        return loss / scale

    def scale(self, name: str) -> float:
        return self.scales.get(name, self.eps)


class AuxiliaryLossComposer:
    """Compose base diffusion loss with optional realism and topology terms."""

    def __init__(
        self,
        config: Mapping[str, Any],
        train_loader: Optional[Iterable[Mapping[str, Tensor]]] = None,
        checkpoint_dir: Optional[Path] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        losses_cfg = config.get("losses", {}) if isinstance(config, Mapping) else {}
        normalizer_cfg = losses_cfg.get("normalizer", {}) or {}
        self.normalizer = AdaptiveLossNormalizer(
            decay=float(normalizer_cfg.get("decay", 0.98)),
            eps=float(normalizer_cfg.get("eps", 1e-8)),
        )

        self.realism_cfg = losses_cfg.get("realism", {}) or {}
        self.realism_enabled = bool(self.realism_cfg.get("enabled", False))
        self.realism_context_length = int(self.realism_cfg.get("context_length", 64))
        self.realism_lags = [int(lag) for lag in self.realism_cfg.get("lags", [1, 5, 10])]
        self.realism_weights = {
            name: float(value)
            for name, value in (self.realism_cfg.get("weights", {}) or {}).items()
        }

        self.topology_cfg = losses_cfg.get("topology", {}) or {}
        self.topology_enabled = bool(self.topology_cfg.get("enabled", False))
        self.topology_weight = float(self.topology_cfg.get("weight", 0.02))
        self.topology_context_length = int(self.topology_cfg.get("context_length", 252))
        self.topology_apply_on_validation = bool(
            self.topology_cfg.get("apply_on_validation", False)
        )
        self.topology_loss_fn = None
        if self.topology_enabled:
            if train_loader is None or checkpoint_dir is None or device is None:
                raise ValueError("Topology loss requires train_loader, checkpoint_dir, and device")
            self.topology_loss_fn = self._build_topology_loss(train_loader, checkpoint_dir, device)

    def compose(
        self,
        outputs: Mapping[str, Tensor],
        batch: Mapping[str, Tensor],
        update_normalizer: bool,
        include_topology: bool,
    ) -> tuple[Tensor, Dict[str, Tensor]]:
        base_loss = scalar_loss(outputs.get("base_loss", outputs["loss"]))
        total = base_loss
        metrics: Dict[str, Tensor] = {"loss": total, "base_loss": base_loss}

        x_hat_future = outputs.get("x_hat_future")
        if x_hat_future is None:
            return total, metrics

        if self.realism_enabled:
            pred_context = build_loss_context(
                batch["history"], x_hat_future, self.realism_context_length
            )
            real_context = build_loss_context(
                batch["history"], batch["target"], self.realism_context_length
            )
            for name, raw_loss in realism_loss_terms(
                pred_context, real_context, self.realism_lags
            ).items():
                weight = float(self.realism_weights.get(name, 0.0))
                metrics[f"realism_{name}_raw"] = raw_loss.detach()
                if weight == 0.0:
                    continue
                normalized = self.normalizer.normalize(
                    f"realism_{name}", raw_loss, update=update_normalizer
                )
                weighted = weight * normalized
                total = total + weighted
                metrics[f"realism_{name}_norm"] = normalized.detach()
                metrics[f"realism_{name}_weighted"] = weighted.detach()
                metrics[f"realism_{name}_scale"] = raw_loss.new_tensor(
                    self.normalizer.scale(f"realism_{name}")
                )

        if (
            self.topology_enabled
            and self.topology_loss_fn is not None
            and self.topology_weight != 0.0
            and (include_topology or self.topology_apply_on_validation)
        ):
            pred_context = build_loss_context(
                batch["history"], x_hat_future, self.topology_context_length
            ).squeeze(-1)
            raw_topology = self.topology_loss_fn.compute(pred_context)
            metrics["topology_raw"] = raw_topology.detach()
            normalized = self.normalizer.normalize(
                "topology", raw_topology, update=update_normalizer
            )
            weighted = self.topology_weight * normalized
            total = total + weighted
            metrics["topology_norm"] = normalized.detach()
            metrics["topology_weighted"] = weighted.detach()
            metrics["topology_scale"] = raw_topology.new_tensor(self.normalizer.scale("topology"))

        metrics["loss"] = total
        return total, metrics

    def _build_topology_loss(self, train_loader, checkpoint_dir: Path, device: torch.device):
        try:
            from src.models import TopologicalLoss
        except ImportError as exc:
            raise RuntimeError(
                "Topology loss requires FinDiffusion dependencies, including gudhi. "
                "Install them into the active environment before enabling losses.topology."
            ) from exc

        topo = TopologicalLoss(
            window_dim=int(self.topology_cfg.get("window_dim", 3)),
            n_landscapes=int(self.topology_cfg.get("n_landscapes", 3)),
            n_grid_points=int(self.topology_cfg.get("n_grid_points", 50)),
            topo_weight=1.0,
            apply_every_n_steps=int(self.topology_cfg.get("apply_every_n_steps", 10)),
            topo_batch_size=int(self.topology_cfg.get("topo_batch_size", 4)),
            n_ref_samples=int(self.topology_cfg.get("n_ref_samples", 200)),
        )
        cache_path = checkpoint_dir / "topology_reference.pt"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        context_loader = ContextReturnsLoader(train_loader, self.topology_context_length)
        if not topo.load_reference(str(cache_path), device=torch.device("cpu")):
            topo.compute_reference(context_loader, device=torch.device("cpu"))
            topo.save_reference(str(cache_path))
        topo.to(device)
        LOGGER.info("Topology loss enabled with reference cache %s", cache_path)
        return topo


class ContextReturnsLoader:
    """Adapter that exposes history+target contexts as `returns` batches."""

    def __init__(self, loader, context_length: int) -> None:
        self.loader = loader
        self.context_length = int(context_length)

    def __iter__(self):
        for batch in self.loader:
            yield {"returns": build_loss_context(batch["history"], batch["target"], self.context_length)}

    def __len__(self) -> int:
        return len(self.loader)


def build_loss_context(history: Tensor, future: Tensor, context_length: int) -> Tensor:
    """Join the needed history tail with a true or predicted future path."""

    history = ensure_3d(history)
    future = ensure_3d(future)
    context_length = int(context_length)
    if context_length <= 0:
        raise ValueError(f"context_length must be positive, got {context_length}")
    future_len = future.shape[1]
    history_len = max(context_length - future_len, 0)
    pieces = []
    if history_len > 0:
        pieces.append(history[:, -history_len:])
    pieces.append(future)
    context = torch.cat(pieces, dim=1)
    return context[:, -context_length:]


def realism_loss_terms(pred_context: Tensor, real_context: Tensor, lags: list[int]) -> Dict[str, Tensor]:
    """Compute differentiable stylized-fact losses on context paths."""

    pred = ensure_2d(pred_context).float()
    real = ensure_2d(real_context).float().detach()
    return {
        "squared_acf": acf_profile_loss(pred.square(), real.square(), lags),
        "absolute_acf": acf_profile_loss(pred.abs(), real.abs(), lags),
        "kurtosis": (excess_kurtosis(pred) - excess_kurtosis(real)).square().mean(),
        "leverage": (leverage_correlation(pred) - leverage_correlation(real)).square().mean(),
    }


def acf_profile_loss(pred: Tensor, real: Tensor, lags: list[int]) -> Tensor:
    losses = []
    for lag in lags:
        if lag <= 0 or pred.shape[1] <= lag:
            continue
        losses.append((autocorr_lag(pred, lag) - autocorr_lag(real, lag)).square())
    if not losses:
        return pred.new_zeros(())
    return torch.stack(losses).mean()


def autocorr_lag(x: Tensor, lag: int, eps: float = 1e-8) -> Tensor:
    x = x - x.mean(dim=1, keepdim=True)
    left = x[:, :-lag]
    right = x[:, lag:]
    numerator = (left * right).mean(dim=1)
    denom = x.square().mean(dim=1).clamp_min(eps)
    return numerator / denom


def excess_kurtosis(x: Tensor, eps: float = 1e-8) -> Tensor:
    centered = x - x.mean(dim=1, keepdim=True)
    std = centered.square().mean(dim=1, keepdim=True).clamp_min(eps).sqrt()
    z = centered / std
    return z.pow(4).mean(dim=1) - 3.0


def leverage_correlation(x: Tensor, eps: float = 1e-8) -> Tensor:
    if x.shape[1] < 2:
        return x.new_zeros(x.shape[0])
    past_returns = x[:, :-1]
    future_vol = x[:, 1:].abs()
    return rowwise_corr(past_returns, future_vol, eps=eps)


def rowwise_corr(a: Tensor, b: Tensor, eps: float = 1e-8) -> Tensor:
    a = a - a.mean(dim=1, keepdim=True)
    b = b - b.mean(dim=1, keepdim=True)
    numerator = (a * b).mean(dim=1)
    denom = (a.square().mean(dim=1) * b.square().mean(dim=1)).sqrt().clamp_min(eps)
    return numerator / denom


def ensure_3d(x: Tensor) -> Tensor:
    if x.dim() == 2:
        return x.unsqueeze(-1)
    return x


def ensure_2d(x: Tensor) -> Tensor:
    if x.dim() == 3:
        return x.squeeze(-1)
    return x


def scalar_loss(loss: Tensor) -> Tensor:
    return loss.mean() if loss.ndim > 0 else loss


def topology_diagnostics(
    real_paths: np.ndarray,
    synthetic_paths: np.ndarray,
    config: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Compare mean H1 persistence landscapes for real and synthetic paths."""

    topology_cfg = ((config or {}).get("losses", {}) or {}).get("topology", {}) or {}
    should_evaluate = bool(
        topology_cfg.get("evaluate", topology_cfg.get("enabled", False))
    )
    if not should_evaluate:
        return {}

    diagnostics: Dict[str, Any] = {
        "enabled": bool(topology_cfg.get("enabled", False)),
        "evaluated": False,
    }
    try:
        import gudhi  # noqa: F401
        from src.models.topo_loss import persistence_landscape, sliding_window_embedding
    except ImportError as exc:
        diagnostics["error"] = (
            "Topology diagnostics require FinDiffusion topology dependencies, "
            f"including gudhi: {exc}"
        )
        return diagnostics

    window_dim = int(topology_cfg.get("window_dim", 3))
    n_landscapes = int(topology_cfg.get("n_landscapes", 3))
    n_grid_points = int(topology_cfg.get("n_grid_points", 50))
    context_length = int(topology_cfg.get("context_length", 252))
    eval_n_real = int(topology_cfg.get("eval_n_real", 30))
    eval_n_synthetic = int(topology_cfg.get("eval_n_synthetic", 50))

    if min(window_dim, n_landscapes, n_grid_points, context_length) <= 0:
        diagnostics["error"] = "Topology diagnostic settings must be positive"
        return diagnostics

    real = _prepare_topology_paths(real_paths, context_length, window_dim)
    synthetic = _prepare_topology_paths(synthetic_paths, context_length, window_dim)
    real = _sample_rows_evenly(real, eval_n_real)
    synthetic = _sample_rows_evenly(synthetic, eval_n_synthetic)

    diagnostics.update(
        {
            "n_real": int(real.shape[0]),
            "n_synthetic": int(synthetic.shape[0]),
            "context_length": int(context_length),
            "window_dim": int(window_dim),
            "n_landscapes": int(n_landscapes),
            "n_grid_points": int(n_grid_points),
        }
    )
    if real.size == 0 or synthetic.size == 0:
        diagnostics["error"] = "Not enough paths for topology diagnostics"
        return diagnostics

    max_edge_length = _estimate_topology_scale(real, window_dim)
    t_grid = torch.linspace(0.0, max_edge_length, n_grid_points, dtype=torch.float32)
    real_landscape = _mean_topology_landscape(
        real,
        window_dim=window_dim,
        n_landscapes=n_landscapes,
        t_grid=t_grid,
        max_edge_length=max_edge_length,
        persistence_landscape_fn=persistence_landscape,
        sliding_window_embedding_fn=sliding_window_embedding,
    )
    synthetic_landscape = _mean_topology_landscape(
        synthetic,
        window_dim=window_dim,
        n_landscapes=n_landscapes,
        t_grid=t_grid,
        max_edge_length=max_edge_length,
        persistence_landscape_fn=persistence_landscape,
        sliding_window_embedding_fn=sliding_window_embedding,
    )
    diff = synthetic_landscape - real_landscape

    diagnostics.update(
        {
            "evaluated": True,
            "max_edge_length": float(max_edge_length),
            "landscape_l2": float(torch.linalg.vector_norm(diff).item()),
            "landscape_mse": float(diff.square().mean().item()),
            "real_landscape_norm": float(torch.linalg.vector_norm(real_landscape).item()),
            "synthetic_landscape_norm": float(torch.linalg.vector_norm(synthetic_landscape).item()),
        }
    )
    return diagnostics


def _prepare_topology_paths(
    paths: np.ndarray,
    context_length: int,
    window_dim: int,
) -> np.ndarray:
    arr = np.asarray(paths, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 2 or arr.shape[1] < window_dim:
        return np.empty((0, 0), dtype=np.float32)
    arr = arr[:, -min(context_length, arr.shape[1]) :]
    valid = np.isfinite(arr).all(axis=1)
    arr = arr[valid]
    if arr.shape[1] < window_dim:
        return np.empty((0, 0), dtype=np.float32)
    return arr


def _sample_rows_evenly(paths: np.ndarray, n: int) -> np.ndarray:
    if paths.size == 0 or n <= 0:
        return np.empty((0, 0), dtype=np.float32)
    if paths.shape[0] <= n:
        return paths
    indices = np.linspace(0, paths.shape[0] - 1, n, dtype=np.int64)
    return paths[indices]


def _estimate_topology_scale(paths: np.ndarray, window_dim: int) -> float:
    distances = []
    for seq in paths:
        pc = _delay_embedding_numpy(seq, window_dim)
        if pc.shape[0] < 2:
            continue
        n_points = min(pc.shape[0], 30)
        indices = np.linspace(0, pc.shape[0] - 1, n_points, dtype=np.int64)
        sub = pc[indices]
        for i in range(n_points):
            for j in range(i + 1, n_points):
                distances.append(float(np.linalg.norm(sub[i] - sub[j])))
    if not distances:
        return 2.0
    return max(float(np.percentile(distances, 90)), 1e-8)


def _mean_topology_landscape(
    paths: np.ndarray,
    window_dim: int,
    n_landscapes: int,
    t_grid: Tensor,
    max_edge_length: float,
    persistence_landscape_fn,
    sliding_window_embedding_fn,
) -> Tensor:
    import gudhi

    landscapes = []
    for seq in paths:
        seq_tensor = torch.as_tensor(seq, dtype=torch.float32)
        point_cloud = (
            sliding_window_embedding_fn(seq_tensor.unsqueeze(0), window_dim)
            .squeeze(0)
            .cpu()
        )
        rips = gudhi.RipsComplex(
            points=point_cloud.numpy().astype(np.float64),
            max_edge_length=max_edge_length,
        )
        simplex_tree = rips.create_simplex_tree(max_dimension=2)
        simplex_tree.compute_persistence()
        pairs = [
            (birth, death)
            for dim, (birth, death) in simplex_tree.persistence()
            if dim == 1 and np.isfinite(death)
        ]
        if pairs:
            births = torch.tensor([birth for birth, _death in pairs], dtype=torch.float32)
            deaths = torch.tensor([death for _birth, death in pairs], dtype=torch.float32)
            landscape = persistence_landscape_fn(
                births,
                deaths,
                t_grid,
                n_landscapes,
            )
        else:
            landscape = torch.zeros(n_landscapes, t_grid.shape[0], dtype=torch.float32)
        landscapes.append(landscape)

    if not landscapes:
        return torch.zeros(n_landscapes, t_grid.shape[0], dtype=torch.float32)
    return torch.stack(landscapes).mean(dim=0)


def _delay_embedding_numpy(seq: np.ndarray, window_dim: int) -> np.ndarray:
    n_windows = len(seq) - window_dim + 1
    if n_windows <= 0:
        return np.empty((0, window_dim), dtype=np.float32)
    return np.stack([seq[i : i + n_windows] for i in range(window_dim)], axis=-1)
