"""Training and evaluation entry point for the comparison pipeline."""

import argparse
import copy
import datetime as dt
import logging
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
import yaml
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from .analysis import write_comparison_analysis
from .data import build_datasets, load_returns, make_loader
from .models import build_model
from .output_index import write_outputs_index


LOGGER = logging.getLogger("comparison_pipeline")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare FinDiffusion-style and CSDI-style one-step diffusion forecasters",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=Path("FinDiffusion_CSDI_Pipeline/config.yaml"))
    parser.add_argument("--model", choices=["findiffusion", "csdi", "both"], default="both")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--debug", action="store_true", help="Small CPU/GPU smoke-test settings")
    parser.add_argument("--no-download", action="store_true", help="Require cached CSV data")
    parser.add_argument("--eval-only", action="store_true", help="Skip training and load final checkpoints")
    parser.add_argument("--epochs", type=int, default=None, help="Override training.epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override training.batch_size")
    parser.add_argument("--num-workers", type=int, default=None, help="Override training.num_workers")
    parser.add_argument("--n-samples", type=int, default=None, help="Override sampling.n_samples")
    parser.add_argument("--eval-batch-size", type=int, default=None, help="Override sampling.batch_size")
    parser.add_argument(
        "--max-eval-windows-per-asset",
        type=int,
        default=None,
        help="Override sampling.max_eval_windows_per_asset",
    )
    sampling_group = parser.add_mutually_exclusive_group()
    sampling_group.add_argument(
        "--ddim",
        dest="use_ddim",
        action="store_true",
        default=None,
        help="Use DDIM sampling for evaluation",
    )
    sampling_group.add_argument(
        "--full-sampling",
        dest="use_ddim",
        action="store_false",
        help="Use full DDPM sampling for evaluation",
    )
    parser.add_argument("--ddim-steps", type=int, default=None, help="Override sampling.ddim_steps")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    config = load_config(args.config)
    if args.debug:
        apply_debug_overrides(config)
    apply_runtime_overrides(config, args)
    check_runtime_dependencies()

    set_seed(int(config["training"].get("seed", 42)))
    device = resolve_device()
    LOGGER.info("Using device: %s", device)

    run_name = args.run_name or dt.datetime.now().strftime("one_step_%Y%m%d_%H%M%S")
    output_root = Path(config["paths"]["output_dir"]) / run_name
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "run_config.yaml").open("w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    returns = load_returns(config, allow_download=not args.no_download)
    datasets, _splits = build_datasets(config, returns)
    LOGGER.info(
        "Dataset sizes: train=%d val=%d test=%d",
        len(datasets["train"]),
        len(datasets["val"]),
        len(datasets["test"]),
    )

    model_names = ["findiffusion", "csdi"] if args.model == "both" else [args.model]
    results = []
    for model_name in model_names:
        model_dir = output_root / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        result = run_model(model_name, config, datasets, model_dir, device, eval_only=args.eval_only)
        results.append(result)

    if len(results) > 1:
        write_comparison_analysis(results, output_root)
        write_outputs_index(output_root.parent)
        LOGGER.info("Wrote comparison summary to %s", output_root / "comparison_summary.csv")
    else:
        write_outputs_index(output_root.parent)
        LOGGER.info("Single-model run complete; skipping top-level comparison summary")


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        return yaml.safe_load(f)


def apply_debug_overrides(config: dict) -> None:
    config["data"]["tickers"] = config["data"]["tickers"][:3]
    config["data"]["history_length"] = min(int(config["data"]["history_length"]), 64)
    config["model"]["d_model"] = 64
    config["model"]["n_layers"] = 2
    config["model"]["n_heads"] = 4
    config["model"]["d_ff"] = 128
    config["model"]["condition_dim"] = 64
    config["model"]["timesteps"] = 20
    config["training"]["epochs"] = 1
    config["training"]["batch_size"] = 16
    config["sampling"]["n_samples"] = 8
    config["sampling"]["max_eval_windows_per_asset"] = 64


def apply_runtime_overrides(config: dict, args: argparse.Namespace) -> None:
    overrides = {
        ("training", "epochs"): args.epochs,
        ("training", "batch_size"): args.batch_size,
        ("training", "num_workers"): args.num_workers,
        ("sampling", "n_samples"): args.n_samples,
        ("sampling", "batch_size"): args.eval_batch_size,
        ("sampling", "max_eval_windows_per_asset"): args.max_eval_windows_per_asset,
        ("sampling", "ddim_steps"): args.ddim_steps,
    }
    for (section, key), value in overrides.items():
        if value is None:
            continue
        if int(value) <= 0:
            raise ValueError(f"{section}.{key} must be positive, got {value}")
        config[section][key] = int(value)
    if args.use_ddim is not None:
        config["sampling"]["use_ddim"] = bool(args.use_ddim)


def check_runtime_dependencies() -> None:
    missing = []
    for module_name in ["scipy"]:
        try:
            __import__(module_name)
        except ImportError:
            missing.append(module_name)
    if missing:
        raise SystemExit(
            "Missing required analysis dependencies: %s\n"
            "Install the FinDiffusion requirements in your environment, for example:\n"
            "  ../venv/bin/pip install -r FinDiffusion/requirements.txt"
            % ", ".join(missing)
        )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_model(
    model_name: str,
    config: Dict[str, Any],
    datasets: Dict[str, Any],
    output_dir: Path,
    device: torch.device,
    eval_only: bool,
) -> Dict:
    from .evaluation import evaluate_predictions, generate_prediction_frame

    LOGGER.info("Running model: %s", model_name)
    model = build_model(model_name, config).to(device)
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        LOGGER.info("Using DataParallel across %d GPUs for %s", torch.cuda.device_count(), model_name)
        model = torch.nn.DataParallel(model)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    final_checkpoint = checkpoint_dir / "final.pt"

    if eval_only:
        checkpoint_path = final_checkpoint if final_checkpoint.exists() else checkpoint_dir / "best.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                "Missing checkpoint for --eval-only: %s or %s"
                % (final_checkpoint, checkpoint_dir / "best.pt")
            )
        LOGGER.info("Loading checkpoint for evaluation: %s", checkpoint_path)
        load_checkpoint(model, checkpoint_path, device)
    else:
        train_model(model, config, datasets, checkpoint_dir, device)

    test_cfg = config["sampling"]
    predictions = generate_prediction_frame(
        model=unwrap_model(model),
        dataset=datasets["test"],
        batch_size=int(test_cfg.get("batch_size", config["training"]["batch_size"])),
        n_samples=int(test_cfg["n_samples"]),
        device=device,
        use_ddim=bool(test_cfg["use_ddim"]),
        ddim_steps=int(test_cfg["ddim_steps"]),
        max_windows_per_asset=int(test_cfg["max_eval_windows_per_asset"])
        if test_cfg.get("max_eval_windows_per_asset") is not None
        else None,
    )
    return evaluate_predictions(predictions, output_dir, model_name)


def train_model(
    model: torch.nn.Module,
    config: Dict[str, Any],
    datasets: Dict[str, Any],
    checkpoint_dir: Path,
    device: torch.device,
) -> None:
    train_cfg = config["training"]
    train_loader = make_loader(
        datasets["train"],
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 0)),
    )
    val_loader = make_loader(
        datasets["val"],
        batch_size=int(train_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(train_cfg.get("num_workers", 0)),
    )

    optimizer = AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
        betas=tuple(float(x) for x in train_cfg["betas"]),
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(train_cfg["epochs"]) * max(1, len(train_loader))),
        eta_min=float(train_cfg["min_lr"]),
    )
    use_amp = bool(train_cfg.get("use_amp", True)) and device.type == "cuda"
    scaler = GradScaler() if use_amp else None
    best_val = float("inf")
    history = []

    for epoch in range(int(train_cfg["epochs"])):
        train_loss = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            train=True,
            clip_grad_norm=float(train_cfg["clip_grad_norm"]),
        )
        val_loss = run_epoch(
            model=model,
            loader=val_loader,
            optimizer=None,
            scheduler=None,
            scaler=None,
            device=device,
            train=False,
            clip_grad_norm=0.0,
        )
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
        LOGGER.info(
            "Epoch %d/%d - train_loss=%.6f val_loss=%.6f",
            epoch + 1,
            int(train_cfg["epochs"]),
            train_loss,
            val_loss,
        )
        if bool(train_cfg.get("save_best", True)) and val_loss < best_val:
            best_val = val_loss
            save_checkpoint(model, checkpoint_dir / "best.pt", history, config)

    save_checkpoint(model, checkpoint_dir / "final.pt", history, config)
    pd.DataFrame(history).to_csv(checkpoint_dir / "train_history.csv", index=False)


def run_epoch(
    model: torch.nn.Module,
    loader,
    optimizer,
    scheduler,
    scaler,
    device: torch.device,
    train: bool,
    clip_grad_norm: float,
) -> float:
    model.train(mode=train)
    total = 0.0
    count = 0
    iterator = tqdm(loader, desc="Train" if train else "Val", leave=False)

    for batch in iterator:
        tensor_batch = {
            key: value.to(device) for key, value in batch.items() if isinstance(value, torch.Tensor)
        }
        if train:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                with autocast():
                    loss = scalar_loss(model(tensor_batch)["loss"])
                scaler.scale(loss).backward()
                if clip_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = scalar_loss(model(tensor_batch)["loss"])
                loss.backward()
                if clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                optimizer.step()
            if scheduler is not None:
                scheduler.step()
        else:
            with torch.no_grad():
                loss = scalar_loss(model(tensor_batch)["loss"])

        total += float(loss.detach().cpu())
        count += 1
        iterator.set_postfix(loss=total / count)

    return total / max(1, count)


def save_checkpoint(
    model: torch.nn.Module,
    path: Path,
    history: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> None:
    model_to_save = unwrap_model(model)
    torch.save(
        {
            "model_state_dict": model_to_save.state_dict(),
            "history": copy.deepcopy(history),
            "config": copy.deepcopy(config),
        },
        path,
    )


def load_checkpoint(model: torch.nn.Module, path: Path, device: torch.device) -> None:
    checkpoint = torch.load(path, map_location=device)
    unwrap_model(model).load_state_dict(checkpoint["model_state_dict"])


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def scalar_loss(loss: torch.Tensor) -> torch.Tensor:
    return loss.mean() if loss.ndim > 0 else loss
