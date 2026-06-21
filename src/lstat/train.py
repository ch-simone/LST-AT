from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .collate import pad_collate
from .config import load_config, resolve_output_dir
from .dataset import LstatDataset, Normalization, input_channel_count
from .index import build_examples
from .losses import masked_huber, masked_mae
from .model import ResUNet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.toml")
    args = parser.parse_args()
    config = load_config(args.config)
    train(config, args.config)


def train(config: dict, config_path: str | Path) -> None:
    seed = int(config.get("seed", 42))
    _seed_everything(seed)

    device = _resolve_device(config["training"].get("device", "auto"))
    output_dir = resolve_output_dir(config, config_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_root = config["data_root"]
    data_cfg = config["data"]
    train_examples = build_examples(data_root, years=config["split"]["train_years"])
    val_examples = build_examples(data_root, years=config["split"]["val_years"])
    print(f"train examples: {len(train_examples)}")
    print(f"val examples:   {len(val_examples)}")

    normalization = Normalization(
        lst_mean=float(data_cfg["lst_mean"]),
        lst_std=float(data_cfg["lst_std"]),
        tair_mean=float(data_cfg["tair_mean"]),
        tair_std=float(data_cfg["tair_std"]),
        apply_modis_correction=bool(data_cfg["apply_modis_correction"]),
    )
    train_ds = LstatDataset(
        train_examples,
        normalization=normalization,
        include_mask_channel=bool(data_cfg["include_mask_channel"]),
        include_time_channels=bool(data_cfg["include_time_channels"]),
    )
    val_ds = LstatDataset(
        val_examples,
        normalization=normalization,
        include_mask_channel=bool(data_cfg["include_mask_channel"]),
        include_time_channels=bool(data_cfg["include_time_channels"]),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["training"]["num_workers"]),
        collate_fn=partial(
            pad_collate,
            min_size=int(config["training"]["min_pad_size"]),
            multiple=int(config["training"]["pad_multiple"]),
        ),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["training"]["num_workers"]),
        collate_fn=partial(
            pad_collate,
            min_size=int(config["training"]["min_pad_size"]),
            multiple=int(config["training"]["pad_multiple"]),
        ),
    )

    in_channels = input_channel_count(
        include_mask_channel=bool(data_cfg["include_mask_channel"]),
        include_time_channels=bool(data_cfg["include_time_channels"]),
    )
    model = ResUNet(
        in_channels=in_channels,
        out_channels=1,
        base_channels=int(config["model"]["base_channels"]),
        depth=int(config["model"]["depth"]),
        dropout=float(config["model"]["dropout"]),
    ).to(device)
    wandb_run = _init_wandb(
        config=config,
        train_examples=len(train_examples),
        val_examples=len(val_examples),
        device=str(device),
        model=model,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    best_val = float("inf")
    loss_name = config["training"].get("loss", "huber")

    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        start_time = time.time()
        train_loss = _run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            loss_name,
            normalization.tair_std,
        )
        val_metrics = _evaluate(model, val_loader, device, loss_name, normalization.tair_std)
        epoch_seconds = time.time() - start_time
        print(
            f"epoch {epoch:03d} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_mae_c={val_metrics['mae_c']:.3f} "
            f"val_rmse_c={val_metrics['rmse_c']:.3f} "
            f"seconds={epoch_seconds:.1f}"
        )
        _wandb_log(
            wandb_run,
            {
                "epoch": epoch,
                "train/loss": train_loss,
                "val/loss": val_metrics["loss"],
                "val/mae_c": val_metrics["mae_c"],
                "val/rmse_c": val_metrics["rmse_c"],
                "train/epoch_seconds": epoch_seconds,
                "train/learning_rate": optimizer.param_groups[0]["lr"],
            },
        )
        if val_metrics["mae_c"] < best_val:
            best_val = val_metrics["mae_c"]
            checkpoint_path = output_dir / "best.pt"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "val_mae_c": best_val,
                },
                checkpoint_path,
            )
            _wandb_log(
                wandb_run,
                {"best/epoch": epoch, "best/val_mae_c": best_val},
            )
            _wandb_save(wandb_run, checkpoint_path)

    _wandb_finish(wandb_run)


def _run_epoch(model, loader, optimizer, device, loss_name: str, target_std: float) -> float:
    model.train()
    total_loss = 0.0
    total_pixels = 0.0
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        mask = batch["mask"].to(device)
        pred = model(x)
        loss = _loss(pred, y, mask, loss_name)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        pixels = mask.sum().item()
        total_loss += loss.item() * pixels
        total_pixels += pixels
    return total_loss / max(total_pixels, 1.0)


@torch.no_grad()
def _evaluate(model, loader, device, loss_name: str, target_std: float) -> dict[str, float]:
    model.eval()
    totals = {"loss": 0.0, "abs_c": 0.0, "sq_c": 0.0, "pixels": 0.0}
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        mask = batch["mask"].to(device)
        pred = model(x)
        pixels = mask.sum().item()
        diff_c = (pred - y) * target_std
        totals["loss"] += _loss(pred, y, mask, loss_name).item() * pixels
        totals["abs_c"] += (diff_c.abs() * mask).sum().item()
        totals["sq_c"] += ((diff_c**2) * mask).sum().item()
        totals["pixels"] += pixels
    pixels = max(totals["pixels"], 1.0)
    return {
        "loss": totals["loss"] / pixels,
        "mae_c": totals["abs_c"] / pixels,
        "rmse_c": (totals["sq_c"] / pixels) ** 0.5,
    }


def _loss(pred, target, mask, loss_name: str):
    if loss_name == "mae":
        return masked_mae(pred, target, mask)
    if loss_name == "huber":
        return masked_huber(pred, target, mask)
    raise ValueError(f"Unknown loss: {loss_name}")


def _resolve_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _init_wandb(
    config: dict,
    train_examples: int,
    val_examples: int,
    device: str,
    model: torch.nn.Module,
):
    wandb_cfg = config.get("wandb", {})
    if not bool(wandb_cfg.get("enabled", False)):
        return None

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "Weights & Biases is enabled but not installed. "
            "Install it with: pip install -e '.[tracking]'"
        ) from exc

    init_kwargs = {
        "project": wandb_cfg.get("project", "lst-at"),
        "config": {
            **config,
            "runtime": {
                "train_examples": train_examples,
                "val_examples": val_examples,
                "device": device,
                "torch_version": torch.__version__,
            },
        },
        "mode": wandb_cfg.get("mode", "online"),
    }
    if wandb_cfg.get("entity"):
        init_kwargs["entity"] = wandb_cfg["entity"]
    if wandb_cfg.get("run_name"):
        init_kwargs["name"] = wandb_cfg["run_name"]

    run = wandb.init(**init_kwargs)
    if bool(wandb_cfg.get("watch_model", False)):
        wandb.watch(model, log="gradients", log_freq=100)
    return run


def _wandb_log(run, metrics: dict) -> None:
    if run is not None:
        run.log(metrics)


def _wandb_save(run, path: Path) -> None:
    if run is not None:
        run.save(str(path))


def _wandb_finish(run) -> None:
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
