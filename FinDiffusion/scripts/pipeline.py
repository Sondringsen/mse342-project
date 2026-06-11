#!/usr/bin/env python
"""End-to-end training pipeline: train DDPM → generate data → train deep hedger.

Usage:
    python scripts/pipeline.py --model ddpm
    python scripts/pipeline.py --model ddpm_topo --n_generate 20000 --n_epochs_diffusion 200

All outputs land under outputs/{model}/:
    checkpoints/final.pt      — trained data generation model
    synthetic.csv             — generated training data for the hedger
    hedging/hedging_model.pt  — trained deep hedging model
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run(cmd: list[str]):
    logger.info("Running: " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end pipeline: train DDPM → generate → train hedger",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["ddpm", "ddpm_topo"],
        help="Model variant",
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml")

    # Diffusion training
    parser.add_argument("--n_epochs_diffusion", type=int, default=None,
                        help="Epochs for the diffusion model (default: from config)")
    parser.add_argument("--debug", action="store_true",
                        help="Quick smoke-test: 2 epochs, 5 tickers, tiny generate/hedge")
    parser.add_argument("--seed",  type=int, default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--ddim",  action="store_true", help="Use DDIM for generation")
    parser.add_argument("--ddim_steps", type=int, default=50)

    # Generation
    parser.add_argument("--n_generate", type=int, default=10000,
                        help="Number of synthetic paths to generate for hedger training")

    # Hedger training
    parser.add_argument("--n_epochs_hedging", type=int, default=1000)
    parser.add_argument("--batch_size_hedging", type=int, default=256)
    parser.add_argument("--lr_hedging", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--cvar_alpha", type=float, default=0.95)

    args = parser.parse_args()
    py = sys.executable
    scripts = Path(__file__).parent

    # ── Phase 1: train diffusion model ────────────────────────────────────────
    logger.info("=" * 60)
    logger.info(f"Phase 1 — Training {args.model} diffusion model")
    logger.info("=" * 60)

    train_cmd = [py, str(scripts / "train.py"), "--model", args.model, "--config", args.config]
    if args.debug:
        train_cmd.append("--debug")
    if args.n_epochs_diffusion is not None:
        logger.warning(
            "--n_epochs_diffusion is not yet supported; set training.epochs in the config file."
        )
    if args.seed is not None:
        train_cmd += ["--seed", str(args.seed)]
    if args.wandb:
        train_cmd.append("--wandb")
    run(train_cmd)

    if args.debug:
        args.n_generate = min(args.n_generate, 100)
        args.n_epochs_hedging = min(args.n_epochs_hedging, 2)

    # ── Phase 2: generate synthetic data ──────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Phase 2 — Generating synthetic data")
    logger.info("=" * 60)

    gen_cmd = [
        py, str(scripts / "generate.py"),
        "--model", args.model,
        "--config", args.config,
        "--n_samples", str(args.n_generate),
    ]
    if args.ddim:
        gen_cmd += ["--ddim", "--ddim_steps", str(args.ddim_steps)]
    if args.seed is not None:
        gen_cmd += ["--seed", str(args.seed)]
    run(gen_cmd)

    # ── Phase 3: train deep hedger ─────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Phase 3 — Training deep hedging model")
    logger.info("=" * 60)

    hedge_cmd = [
        py, str(scripts / "train_hedging.py"),
        "--model", args.model,
        "--n_samples",  str(args.n_generate),
        "--n_epochs",   str(args.n_epochs_hedging),
        "--batch_size", str(args.batch_size_hedging),
        "--lr",         str(args.lr_hedging),
        "--hidden_dim", str(args.hidden_dim),
        "--cvar_alpha", str(args.cvar_alpha),
    ]
    run(hedge_cmd)

    logger.info("=" * 60)
    logger.info(f"Pipeline complete. Outputs in outputs/{args.model}/")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
