"""Tests for auxiliary realism loss helpers."""

from pathlib import Path
import sys
import unittest

import numpy as np
import torch


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE_ROOT))

from pipeline.losses import (  # noqa: E402
    AdaptiveLossNormalizer,
    build_loss_context,
    realism_loss_terms,
    topology_diagnostics,
)
from pipeline.models import build_model  # noqa: E402


def small_config(prediction_length: int = 5) -> dict:
    return {
        "data": {
            "history_length": 16,
            "prediction_length": prediction_length,
        },
        "model": {
            "d_model": 32,
            "n_layers": 1,
            "n_heads": 4,
            "d_ff": 64,
            "condition_dim": 16,
            "timesteps": 8,
            "beta_schedule": "linear",
            "beta_start": 0.0001,
            "beta_end": 0.02,
            "prediction_type": "epsilon",
            "dropout": 0.0,
        },
    }


class LossesTest(unittest.TestCase):
    def test_build_loss_context_uses_history_tail_and_future(self) -> None:
        history = torch.arange(20, dtype=torch.float32).view(1, 20, 1)
        future = torch.tensor([[[100.0], [101.0], [102.0]]])

        context = build_loss_context(history, future, context_length=8)

        self.assertEqual(context.shape, (1, 8, 1))
        self.assertEqual(context.squeeze(-1).tolist(), [[15.0, 16.0, 17.0, 18.0, 19.0, 100.0, 101.0, 102.0]])

    def test_realism_terms_are_finite_and_differentiable(self) -> None:
        torch.manual_seed(1)
        pred = torch.randn(4, 32, 1, requires_grad=True)
        real = torch.randn(4, 32, 1)

        terms = realism_loss_terms(pred, real, lags=[1, 5, 10])
        loss = sum(terms.values())
        loss.backward()

        self.assertTrue(torch.isfinite(loss).item())
        self.assertIsNotNone(pred.grad)
        self.assertTrue(torch.isfinite(pred.grad).all().item())

    def test_adaptive_normalizer_preserves_gradient(self) -> None:
        normalizer = AdaptiveLossNormalizer(decay=0.5)
        x = torch.tensor([2.0, -3.0], requires_grad=True)
        raw = x.square().mean()

        normalized = normalizer.normalize("demo", raw, update=True)
        normalized.backward()

        self.assertGreater(normalizer.scale("demo"), 0.0)
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all().item())

    def test_forecasters_return_clean_future_for_auxiliary_losses(self) -> None:
        torch.manual_seed(2)
        batch = {
            "history": torch.randn(2, 16, 1),
            "target": torch.randn(2, 5, 1),
        }

        for model_name in ["findiffusion", "csdi"]:
            model = build_model(model_name, small_config())
            outputs = model(batch)
            self.assertIn("base_loss", outputs)
            self.assertIn("x_hat_future", outputs)
            self.assertEqual(outputs["x_hat_future"].shape, batch["target"].shape)
            outputs["loss"].backward()
            self.assertTrue(any(p.grad is not None for p in model.parameters()))

    def test_topology_diagnostics_are_finite_when_enabled(self) -> None:
        try:
            import gudhi  # noqa: F401
        except ImportError:
            self.skipTest("gudhi is not installed")

        grid = np.linspace(0.0, 4.0 * np.pi, 24, dtype=np.float32)
        real = np.stack(
            [
                0.01 * np.sin(grid),
                0.01 * np.sin(grid + 0.3),
            ]
        )
        synthetic = np.stack(
            [
                0.01 * np.sin(grid + 0.6),
                0.012 * np.sin(grid + 0.9),
            ]
        )
        config = {
            "losses": {
                "topology": {
                    "enabled": True,
                    "evaluate": True,
                    "context_length": 24,
                    "window_dim": 3,
                    "n_landscapes": 2,
                    "n_grid_points": 8,
                    "eval_n_real": 2,
                    "eval_n_synthetic": 2,
                }
            }
        }

        diagnostics = topology_diagnostics(real, synthetic, config)

        self.assertTrue(diagnostics["evaluated"])
        self.assertEqual(diagnostics["n_real"], 2)
        self.assertEqual(diagnostics["n_synthetic"], 2)
        self.assertTrue(np.isfinite(diagnostics["landscape_l2"]))
        self.assertTrue(np.isfinite(diagnostics["landscape_mse"]))


if __name__ == "__main__":
    unittest.main()
