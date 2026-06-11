"""
Topological regularization loss for financial time series diffusion models.

Mathematical background (full derivation in topo_plan.md):

Given a predicted clean sequence x̂₀ ∈ ℝᵀ (log returns):

  1. Delay embedding   : P_d(x) = {(x_t,...,x_{t+d-1}) : t=1,...,T-d+1} ⊂ ℝ^d
  2. Vietoris-Rips H₁  : persistence diagram Dgm₁ = {(bᵢ, dᵢ)} of 1-cycles
  3. Persistence landscape (Bubenik 2015):
       Λᵢ(t) = max(0, min(t−bᵢ, dᵢ−t))   (tent function)
       λₙ(t) = n-th largest Λᵢ(t) over all i
     → landscape matrix λ ∈ ℝ^{N_λ × N_g}
  4. Reference mean landscape λ̄^real computed once from training data.
  5. Loss: L_topo = (1/B') Σ_b ||λ(x̂₀^b) − λ̄^real||²_F

Differentiability (Section 3 of topo_plan.md):
  Each bᵢ/dᵢ equals the Euclidean length of a specific "critical" edge
  identified in the forward pass by gudhi. The gradient ∂bᵢ/∂p = unit vector
  along that edge (standard subgradient of the norm). The landscape uses
  torch.topk, which is natively differentiable.

References:
  Gidea et al. (2018) "Topological Data Analysis of Financial Time Series"
  Bubenik (2015) "Statistical Topological Data Analysis using Persistence Landscapes"
  Carrière et al. (2021) "Differentiating through the Persistent Homology of a Point Cloud"
"""

import logging
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ── Delay embedding ─────────────────────────────────────────────────────────

def sliding_window_embedding(x: Tensor, d: int) -> Tensor:
    """
    Takens delay embedding: (..., T) → (..., T−d+1, d).

    P_d(x)[..., t, k] = x[..., t+k]  for k = 0, ..., d−1.

    This is a linear operation so gradients flow through it unchanged.
    """
    N = x.shape[-1] - d + 1
    return torch.stack([x[..., i : i + N] for i in range(d)], dim=-1)


# ── Differentiable H₁ persistence ───────────────────────────────────────────

class _RipsPersistenceH1Function(torch.autograd.Function):
    """
    Wraps gudhi's Vietoris-Rips H₁ computation in a differentiable
    torch.autograd.Function.

    Forward  : calls gudhi to get birth/death pairs and stores the
               critical simplex pairing (σ_birth, σ_death) in ctx.
    Backward : for each H₁ feature i with critical edge (vₐ, v_b):

               bᵢ = ‖p_vₐ − p_vb‖  →  ∂bᵢ/∂p_vₐ = (p_vₐ−p_vb)/‖…‖

               dᵢ = length of the longest edge in the death triangle
                    →  gradient flows through that single edge only.

    The critical assignment is treated as fixed from the forward pass
    (standard subgradient / envelope theorem argument).
    """

    @staticmethod
    def forward(
        ctx,
        point_cloud: Tensor,
        max_edge_length: float,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            point_cloud   : (N, d) — coordinates on any device, any float dtype.
            max_edge_length: filtration cut-off; controls which simplices are
                            included (estimated from data in compute_reference).
        Returns:
            births : (n_h1,) birth filtration values
            deaths : (n_h1,) death filtration values  (finite pairs only)
        """
        import gudhi

        pts_np = point_cloud.detach().cpu().numpy().astype(np.float64)

        rips = gudhi.RipsComplex(points=pts_np, max_edge_length=max_edge_length)
        st = rips.create_simplex_tree(max_dimension=2)
        st.compute_persistence()

        birth_vals, death_vals = [], []
        birth_edges, death_tris = [], []

        for birth_simplex, death_simplex in st.persistence_pairs():
            # H₁ feature: born as edge (len 2), killed by triangle (len 3)
            if len(birth_simplex) == 2 and len(death_simplex) == 3:
                b = st.filtration(birth_simplex)
                d = st.filtration(death_simplex)
                if np.isfinite(d) and d > b:
                    birth_vals.append(b)
                    death_vals.append(d)
                    birth_edges.append(tuple(birth_simplex))
                    death_tris.append(tuple(death_simplex))

        ctx.save_for_backward(point_cloud)
        ctx.birth_edges = birth_edges
        ctx.death_tris = death_tris

        if not birth_vals:
            empty = point_cloud.new_zeros(0)
            return empty, empty

        return (
            point_cloud.new_tensor(birth_vals),
            point_cloud.new_tensor(death_vals),
        )

    @staticmethod
    def backward(
        ctx,
        grad_births: Tensor,
        grad_deaths: Tensor,
    ) -> Tuple[Tensor, None]:
        (point_cloud,) = ctx.saved_tensors
        grad_pc = torch.zeros_like(point_cloud)

        for i, (birth_edge, death_tri) in enumerate(
            zip(ctx.birth_edges, ctx.death_tris)
        ):
            # ── birth: bᵢ = ‖p_vₐ − p_vb‖ ──────────────────────────────
            va, vb = birth_edge
            diff_b = point_cloud[va] - point_cloud[vb]
            dist_b = diff_b.norm()
            if dist_b > 1e-8:
                unit_b = diff_b / dist_b
                grad_pc[va] = grad_pc[va] + grad_births[i] * unit_b
                grad_pc[vb] = grad_pc[vb] - grad_births[i] * unit_b

            # ── death: dᵢ = length of longest edge in death triangle ─────
            v1, v2, v3 = death_tri
            edges = [(v1, v2), (v1, v3), (v2, v3)]
            diffs = [point_cloud[u] - point_cloud[v] for u, v in edges]
            dists = [df.norm() for df in diffs]
            k = max(range(3), key=lambda j: dists[j].item())
            dist_d = dists[k]
            if dist_d > 1e-8:
                unit_d = diffs[k] / dist_d
                u, v = edges[k]
                grad_pc[u] = grad_pc[u] + grad_deaths[i] * unit_d
                grad_pc[v] = grad_pc[v] - grad_deaths[i] * unit_d

        return grad_pc, None


# ── Persistence landscape ────────────────────────────────────────────────────

def persistence_landscape(
    births: Tensor,
    deaths: Tensor,
    t_grid: Tensor,
    n_landscapes: int,
) -> Tensor:
    """
    Compute persistence landscape matrix from H₁ birth/death pairs.

    Tent function for pair i:  Λᵢ(t) = max(0, min(t−bᵢ, dᵢ−t))
    n-th landscape:            λₙ(t)  = n-th largest Λᵢ(t)

    Evaluated on t_grid this gives a (n_landscapes, n_grid) matrix.
    torch.topk carries gradients to the selected tent values.

    Args:
        births      : (n_pairs,)
        deaths      : (n_pairs,)
        t_grid      : (n_grid,)  evaluation points in [0, max_edge_length]
        n_landscapes: number of landscape functions N_λ

    Returns:
        (n_landscapes, n_grid) landscape matrix
    """
    n_grid = t_grid.shape[0]

    if births.numel() == 0:
        return torch.zeros(
            n_landscapes, n_grid, device=t_grid.device, dtype=t_grid.dtype
        )

    # Broadcast births/deaths onto the t_grid
    t = t_grid.unsqueeze(0)                         # (1, n_grid)
    b = births.unsqueeze(1).to(t_grid)              # (n_pairs, 1)
    d = deaths.unsqueeze(1).to(t_grid)              # (n_pairs, 1)

    tents = torch.clamp(torch.minimum(t - b, d - t), min=0.0)  # (n_pairs, n_grid)

    k = min(n_landscapes, tents.shape[0])
    topk = torch.topk(tents, k, dim=0).values       # (k, n_grid)

    if k < n_landscapes:
        pad = torch.zeros(
            n_landscapes - k, n_grid, device=t_grid.device, dtype=t_grid.dtype
        )
        topk = torch.cat([topk, pad], dim=0)

    return topk  # (n_landscapes, n_grid)


# ── TopologicalLoss module ───────────────────────────────────────────────────

class TopologicalLoss(nn.Module):
    """
    H₁ persistence landscape regularization for the DDPM training loss.

    Penalises the MSE between the landscape of predicted sequences x̂₀ and
    the mean landscape of real training data (λ̄^real), pre-computed once.

    The combined DDPM + topo loss is:
        L_total = L_DDPM + topo_weight · L_topo

    Usage
    -----
    Before training::

        topo = TopologicalLoss(...)
        topo.compute_reference(train_loader, device=torch.device("cpu"))
        # model.to(device) will move the reference buffer to the right device

    During training (called from GaussianDiffusion.training_loss)::

        loss_topo = topo.compute(x_hat_0)   # (B, T) → scalar
    """

    def __init__(
        self,
        window_dim: int = 3,
        n_landscapes: int = 3,
        n_grid_points: int = 50,
        topo_weight: float = 0.1,
        apply_every_n_steps: int = 5,
        topo_batch_size: int = 16,
        n_ref_samples: int = 500,
    ):
        """
        Args:
            window_dim          : delay embedding dimension d (Gidea uses d=3)
            n_landscapes        : number of landscape functions N_λ
            n_grid_points       : grid resolution N_g for landscape evaluation
            topo_weight         : α_topo weight in the combined loss
            apply_every_n_steps : compute topo loss every N training steps
            topo_batch_size     : random sub-batch size for persistence (cost control)
            n_ref_samples       : max sequences used to compute λ̄^real
        """
        super().__init__()
        self.window_dim = window_dim
        self.n_landscapes = n_landscapes
        self.n_grid_points = n_grid_points
        self.topo_weight = topo_weight
        self.apply_every_n_steps = apply_every_n_steps
        self.topo_batch_size = topo_batch_size
        self.n_ref_samples = n_ref_samples

        # t_grid and max_edge_length are finalised in compute_reference()
        self.register_buffer("t_grid", torch.linspace(0.0, 2.0, n_grid_points))
        self.register_buffer(
            "reference_landscape", torch.zeros(n_landscapes, n_grid_points)
        )
        self.max_edge_length: float = 2.0
        self._reference_computed: bool = False
        self._step: int = 0

    # ── Reference computation ────────────────────────────────────────────────

    def save_reference(self, path: str) -> None:
        torch.save(
            {
                "reference_landscape": self.reference_landscape.cpu(),
                "t_grid": self.t_grid.cpu(),
                "max_edge_length": self.max_edge_length,
            },
            path,
        )
        logger.info(f"Reference landscape saved to {path}")

    def load_reference(self, path: str, device: torch.device) -> bool:
        """Returns True if loaded successfully, False if file not found."""
        import os
        if not os.path.exists(path):
            return False
        data = torch.load(path, map_location=device)
        self.reference_landscape.copy_(data["reference_landscape"].to(device))
        self.t_grid.copy_(data["t_grid"].to(device))
        self.max_edge_length = data["max_edge_length"]
        self._reference_computed = True
        logger.info(f"Reference landscape loaded from {path}")
        return True

    @torch.no_grad()
    def compute_reference(
        self,
        train_loader,
        device: torch.device,
    ) -> None:
        """
        Pre-compute λ̄^real = mean H₁ persistence landscape over training data.

        Also estimates max_edge_length as the 90th percentile of pairwise
        distances in a small sample of training point clouds, which sets the
        filtration range and the evaluation grid [0, max_edge_length].

        After this call the reference_landscape buffer is filled and
        _reference_computed is set to True. When model.to(device) is called
        later, the buffer is moved to the training device automatically.

        Args:
            train_loader : DataLoader yielding tensors of shape (B, T) or (B, T, 1)
            device       : device where the reference tensor is stored
                           (pass cpu here; the Trainer will move it to GPU)
        """
        import gudhi

        # ── Step 1: estimate max_edge_length ─────────────────────────────
        logger.info("Estimating filtration scale from training data...")
        sample_dists: list[float] = []
        n_seen = 0
        for batch in train_loader:
            x = _extract_returns(batch).cpu()
            for seq in x:
                pc = sliding_window_embedding(seq.unsqueeze(0), self.window_dim).squeeze(0)
                pts = pc.numpy().astype(np.float64)
                # Sample a small subset of rows to estimate pairwise distance scale
                n = min(pts.shape[0], 30)
                idx = np.random.choice(pts.shape[0], n, replace=False)
                sub = pts[idx]
                for i in range(n):
                    for j in range(i + 1, n):
                        sample_dists.append(float(np.linalg.norm(sub[i] - sub[j])))
                n_seen += 1
                if n_seen >= 50:
                    break
            if n_seen >= 50:
                break

        max_edge_length = (
            float(np.percentile(sample_dists, 90)) if sample_dists else 2.0
        )
        self.max_edge_length = max_edge_length
        self.t_grid.copy_(
            torch.linspace(0.0, max_edge_length, self.n_grid_points)
        )
        logger.info(f"Filtration scale (90th pct): {max_edge_length:.4f}")

        # ── Step 2: compute mean landscape ───────────────────────────────
        logger.info(
            f"Computing reference H₁ landscape "
            f"(up to {self.n_ref_samples} sequences)..."
        )
        landscapes: list[Tensor] = []
        n_processed = 0

        for batch in tqdm(train_loader, desc="Reference landscape"):
            x = _extract_returns(batch).cpu()
            for seq in x:
                if n_processed >= self.n_ref_samples:
                    break

                pc = sliding_window_embedding(seq.unsqueeze(0), self.window_dim).squeeze(0)
                pts = pc.numpy().astype(np.float64)

                rips = gudhi.RipsComplex(points=pts, max_edge_length=max_edge_length)
                st = rips.create_simplex_tree(max_dimension=2)
                st.compute_persistence()

                diag = [
                    (b, d)
                    for dim, (b, d) in st.persistence()
                    if dim == 1 and np.isfinite(d)
                ]

                if diag:
                    births = torch.tensor([b for b, _ in diag], dtype=torch.float32)
                    deaths = torch.tensor([d for _, d in diag], dtype=torch.float32)
                    land = persistence_landscape(
                        births, deaths, self.t_grid.cpu(), self.n_landscapes
                    )
                else:
                    land = torch.zeros(self.n_landscapes, self.n_grid_points)

                landscapes.append(land)
                n_processed += 1

            if n_processed >= self.n_ref_samples:
                break

        if not landscapes:
            logger.warning("No training sequences found; reference landscape is zero.")
            return

        mean_land = torch.stack(landscapes).mean(0)
        self.reference_landscape.copy_(mean_land.to(device))
        self._reference_computed = True
        logger.info(
            f"Reference landscape ready ({n_processed} sequences). "
            f"Mean absolute value: {mean_land.abs().mean():.5f}"
        )

    # ── Per-sequence differentiable landscape ────────────────────────────────

    def _sequence_landscape(self, x: Tensor) -> Tensor:
        """
        Differentiable H₁ landscape for a single sequence x of shape (T,).

        Chain: x (T,) → point cloud (N,d) → (births, deaths) → landscape (N_λ, N_g)
        All steps carry gradients back to x.

        Returns:
            (n_landscapes, n_grid_points)
        """
        # (T,) → (1, T) → (1, N, d) → (N, d)
        pc = sliding_window_embedding(x.unsqueeze(0), self.window_dim).squeeze(0)

        # Cast to float32 for gudhi (handles AMP float16 inputs safely).
        # .float() is differentiable so the grad chain is not broken.
        births, deaths = _RipsPersistenceH1Function.apply(
            pc.float(), self.max_edge_length
        )
        return persistence_landscape(births, deaths, self.t_grid, self.n_landscapes)

    # ── Training-time compute ─────────────────────────────────────────────────

    def compute(self, x_hat_0: Tensor) -> Tensor:
        """
        Topological regularization loss for a batch of predicted clean sequences.

        Skips on most steps (controlled by apply_every_n_steps) to amortise
        the cost of persistence computation. On a skipped step returns a zero
        scalar that contributes nothing to the backward pass.

        Args:
            x_hat_0 : (B, T) predicted clean log-return sequences

        Returns:
            Scalar L_topo  (or 0 on skipped steps)
        """
        self._step += 1

        if not self._reference_computed:
            return x_hat_0.new_zeros(())

        if self._step % self.apply_every_n_steps != 0:
            return x_hat_0.new_zeros(())

        B = x_hat_0.shape[0]
        n = min(self.topo_batch_size, B)
        idx = torch.randperm(B, device=x_hat_0.device)[:n]
        sub = x_hat_0[idx]  # (n, T)

        losses = []
        for seq in sub:
            land = self._sequence_landscape(seq)
            diff = land - self.reference_landscape
            losses.append((diff * diff).sum())

        return torch.stack(losses).mean()


# ── Helpers ──────────────────────────────────────────────────────────────────

def _extract_returns(batch) -> Tensor:
    """Extract (B, T) return tensor from a DataLoader batch."""
    x = batch if isinstance(batch, Tensor) else batch["returns"]
    if x.dim() == 3:
        x = x.squeeze(-1)   # (B, T, 1) → (B, T)
    return x
