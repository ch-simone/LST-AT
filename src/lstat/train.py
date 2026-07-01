from __future__ import annotations

import argparse
from functools import partial
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .cache import CachedLstatDataset
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

    data_cfg = config["data"]
    normalization = Normalization(
        lst_mean=float(data_cfg["lst_mean"]),
        lst_std=float(data_cfg["lst_std"]),
        tair_mean=float(data_cfg["tair_mean"]),
        tair_std=float(data_cfg["tair_std"]),
        apply_modis_correction=bool(data_cfg["apply_modis_correction"]),
    )
    train_ds = _build_dataset(
        config=config,
        split="train",
        normalization=normalization,
        seed=seed,
    )
    val_ds = _build_dataset(
        config=config,
        split="val",
        normalization=normalization,
        seed=seed + 1,
    )
    print(f"train examples: {len(train_ds)}")
    print(f"val examples:   {len(val_ds)}")

    num_workers = int(config["training"]["num_workers"])
    loader_worker_kwargs = _dataloader_worker_kwargs(config, device, num_workers)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        num_workers=num_workers,
        **loader_worker_kwargs,
        collate_fn=partial(
            pad_collate,
            min_size=int(config["training"]["min_pad_size"]),
            multiple=int(config["training"]["pad_multiple"]),
            crop_size=int(config["training"].get("train_crop_size", 0)),
            random_crop=True,
        ),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=num_workers,
        **loader_worker_kwargs,
        collate_fn=partial(
            pad_collate,
            min_size=int(config["training"]["min_pad_size"]),
            multiple=int(config["training"]["pad_multiple"]),
            crop_size=int(config["training"].get("eval_crop_size", 0)),
            random_crop=False,
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
        train_examples=len(train_ds),
        val_examples=len(val_ds),
        device=str(device),
        model=model,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    best_val = float("inf")
    best_epoch = 0
    loss_name = config["training"].get("loss", "huber")
    total_epochs = int(config["training"]["epochs"])
    scheduler = _build_scheduler(optimizer, config, total_epochs=total_epochs)
    final_val_metrics = {}
    completed_epochs = 0
    epochs_without_improvement = 0
    early_stopping_patience = int(
        config["training"].get("early_stopping_patience", 0) or 0
    )
    early_stopping_min_delta = float(
        config["training"].get("early_stopping_min_delta", 0.0)
    )

    for epoch in range(1, total_epochs + 1):
        completed_epochs = epoch
        start_time = time.time()
        train_loss = _run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            loss_name,
            normalization.tair_std,
            epoch=epoch,
            total_batches=len(train_loader),
            log_every_batches=int(config["training"].get("log_every_batches", 0)),
            wandb_run=wandb_run,
        )
        val_metrics = _evaluate(model, val_loader, device, loss_name, normalization.tair_std)
        epoch_seconds = time.time() - start_time
        print(
            f"epoch {epoch:03d} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_mae_c={val_metrics['mae_c']:.3f} "
            f"val_rmse_c={val_metrics['rmse_c']:.3f} "
            f"val_r2={val_metrics['r2']:.4f} "
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
                "val/r2": val_metrics["r2"],
                "train/epoch_seconds": epoch_seconds,
                "train/learning_rate": optimizer.param_groups[0]["lr"],
            },
        )
        if _should_log_eval_images(config, epoch, total_epochs):
            _log_eval_images(
                wandb_run=wandb_run,
                model=model,
                loader=val_loader,
                device=device,
                normalization=normalization,
                config=config,
                epoch=epoch,
            )
        significant_improvement = (
            val_metrics["mae_c"] < best_val - early_stopping_min_delta
        )
        improved = val_metrics["mae_c"] < best_val
        if improved:
            best_val = val_metrics["mae_c"]
            best_epoch = epoch
            if significant_improvement:
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
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
        else:
            epochs_without_improvement += 1

        final_val_metrics = val_metrics
        if scheduler is not None:
            scheduler.step()
        _wandb_log(
            wandb_run,
            {"train/next_learning_rate": optimizer.param_groups[0]["lr"]},
        )
        if (
            early_stopping_patience > 0
            and epochs_without_improvement >= early_stopping_patience
        ):
            print(
                f"early stopping at epoch {epoch:03d}; "
                f"best_epoch={best_epoch:03d} best_val_mae_c={best_val:.3f}"
            )
            break

    final_metrics = {
        "best_val_mae_c": best_val,
        "best_epoch": best_epoch,
        "completed_epochs": completed_epochs,
        "final_val_loss": final_val_metrics.get("loss"),
        "final_val_mae_c": final_val_metrics.get("mae_c"),
        "final_val_rmse_c": final_val_metrics.get("rmse_c"),
        "final_val_r2": final_val_metrics.get("r2"),
    }
    (output_dir / "final_metrics.json").write_text(
        json.dumps(final_metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    _wandb_log(
        wandb_run,
        {
            "final/best_val_mae_c": final_metrics["best_val_mae_c"],
            "final/best_epoch": final_metrics["best_epoch"],
            "final/completed_epochs": final_metrics["completed_epochs"],
            "final/val_loss": final_metrics["final_val_loss"],
            "final/val_mae_c": final_metrics["final_val_mae_c"],
            "final/val_rmse_c": final_metrics["final_val_rmse_c"],
            "final/val_r2": final_metrics["final_val_r2"],
        },
    )
    _wandb_finish(wandb_run)


def _run_epoch(
    model,
    loader,
    optimizer,
    device,
    loss_name: str,
    target_std: float,
    epoch: int = 0,
    total_batches: int | None = None,
    log_every_batches: int = 0,
    wandb_run=None,
) -> float:
    model.train()
    total_loss = 0.0
    total_pixels = 0.0
    interval_loss = 0.0
    interval_pixels = 0.0
    epoch_start = time.time()
    total_batches = total_batches or len(loader)
    for batch_index, batch in enumerate(loader, start=1):
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
        interval_loss += loss.item() * pixels
        interval_pixels += pixels
        if log_every_batches > 0 and (
            batch_index % log_every_batches == 0 or batch_index == total_batches
        ):
            elapsed = time.time() - epoch_start
            running_loss = total_loss / max(total_pixels, 1.0)
            recent_loss = interval_loss / max(interval_pixels, 1.0)
            batches_per_second = batch_index / max(elapsed, 1e-6)
            print(
                f"epoch {epoch:03d} batch {batch_index:05d}/{total_batches:05d} "
                f"recent_loss={recent_loss:.4f} "
                f"running_loss={running_loss:.4f} "
                f"shape={tuple(x.shape)} "
                f"batches_per_s={batches_per_second:.2f} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )
            _wandb_log(
                wandb_run,
                {
                    "epoch": epoch,
                    "train/batch": batch_index,
                    "train/recent_loss": recent_loss,
                    "train/running_loss": running_loss,
                    "train/batches_per_second": batches_per_second,
                    "train/epoch_elapsed_seconds": elapsed,
                },
            )
            interval_loss = 0.0
            interval_pixels = 0.0
    return total_loss / max(total_pixels, 1.0)


@torch.no_grad()
def _evaluate(model, loader, device, loss_name: str, target_std: float) -> dict[str, float]:
    model.eval()
    totals = {
        "loss": 0.0,
        "abs_c": 0.0,
        "sq_c": 0.0,
        "target_sum_c": 0.0,
        "target_sq_sum_c": 0.0,
        "pixels": 0.0,
    }
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        mask = batch["mask"].to(device)
        pred = model(x)
        pixels = mask.sum().item()
        diff_c = (pred - y) * target_std
        y_c = y * target_std
        totals["loss"] += _loss(pred, y, mask, loss_name).item() * pixels
        totals["abs_c"] += (diff_c.abs() * mask).sum().item()
        totals["sq_c"] += ((diff_c**2) * mask).sum().item()
        totals["target_sum_c"] += (y_c * mask).sum().item()
        totals["target_sq_sum_c"] += ((y_c**2) * mask).sum().item()
        totals["pixels"] += pixels
    pixels = max(totals["pixels"], 1.0)
    target_mean_c = totals["target_sum_c"] / pixels
    target_ss_tot_c = totals["target_sq_sum_c"] - pixels * target_mean_c**2
    r2 = 1.0 - totals["sq_c"] / target_ss_tot_c if target_ss_tot_c > 0 else float("nan")
    return {
        "loss": totals["loss"] / pixels,
        "mae_c": totals["abs_c"] / pixels,
        "rmse_c": (totals["sq_c"] / pixels) ** 0.5,
        "r2": r2,
    }


def _loss(pred, target, mask, loss_name: str):
    if loss_name == "mae":
        return masked_mae(pred, target, mask)
    if loss_name == "huber":
        return masked_huber(pred, target, mask)
    raise ValueError(f"Unknown loss: {loss_name}")


def _build_scheduler(optimizer, config: dict, total_epochs: int):
    scheduler_name = config["training"].get("lr_scheduler", "none").lower()
    if scheduler_name in ("", "none"):
        return None
    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_epochs,
            eta_min=float(config["training"].get("min_learning_rate", 1e-5)),
        )
    raise ValueError(f"Unknown lr_scheduler: {scheduler_name}")


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


def _limit_examples(examples: list, limit: int, seed: int) -> list:
    limit = int(limit or 0)
    if limit <= 0 or limit >= len(examples):
        return examples
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(examples)), limit))
    return [examples[index] for index in indices]


def _build_dataset(
    *,
    config: dict,
    split: str,
    normalization: Normalization,
    seed: int,
):
    cache_cfg = config.get("cache", {})
    if bool(cache_cfg.get("enabled", False)):
        cache_root = Path(cache_cfg["root"])
        dataset = CachedLstatDataset(cache_root / split)
        limit = _split_limit(config, split)
        if limit > 0 and limit < len(dataset):
            raise ValueError(
                "max_*_examples limits are not supported when training from "
                "a prebuilt cache. Build a smaller cache instead."
            )
        return dataset

    split_years = {
        "train": config["split"]["train_years"],
        "val": config["split"]["val_years"],
        "test": config["split"].get("test_years", []),
    }[split]
    examples = build_examples(config["data_root"], years=split_years)
    examples = _limit_examples(examples, _split_limit(config, split), seed)
    data_cfg = config["data"]
    return LstatDataset(
        examples,
        normalization=normalization,
        include_mask_channel=bool(data_cfg["include_mask_channel"]),
        include_time_channels=bool(data_cfg["include_time_channels"]),
    )


def _split_limit(config: dict, split: str) -> int:
    limit_key = {
        "train": "max_train_examples",
        "val": "max_val_examples",
        "test": "max_test_examples",
    }[split]
    return int(config["training"].get(limit_key, 0) or 0)


def _dataloader_worker_kwargs(
    config: dict,
    device: torch.device,
    num_workers: int,
) -> dict:
    training_cfg = config["training"]
    kwargs = {
        "pin_memory": bool(training_cfg.get("pin_memory", device.type == "cuda")),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(
            training_cfg.get("persistent_workers", False)
        )
        kwargs["prefetch_factor"] = int(training_cfg.get("prefetch_factor", 2))
    return kwargs


def _should_log_eval_images(config: dict, epoch: int, total_epochs: int) -> bool:
    interval = int(config.get("evaluation", {}).get("log_image_interval", 0))
    if interval <= 0:
        return False
    return epoch == 1 or epoch == total_epochs or epoch % interval == 0


@torch.no_grad()
def _log_eval_images(
    wandb_run,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    normalization: Normalization,
    config: dict,
    epoch: int,
) -> None:
    if wandb_run is None:
        return

    try:
        import wandb
    except ImportError:
        return

    eval_cfg = config.get("evaluation", {})
    max_examples = int(eval_cfg.get("num_image_examples", 4))
    residual_limit = float(eval_cfg.get("residual_heatmap_limit_c", 10.0))
    image_selection = eval_cfg.get("image_selection", "most_valid")
    max_batches = int(eval_cfg.get("max_image_batches", 0))
    model.eval()
    records = []
    for batch_index, batch in enumerate(loader):
        if max_batches > 0 and batch_index >= max_batches:
            break
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        mask = batch["mask"].to(device)
        pred = model(x)

        x_lst_c = x[:, 0:1] * normalization.lst_std + normalization.lst_mean
        y_c = y * normalization.tair_std + normalization.tair_mean
        pred_c = pred * normalization.tair_std + normalization.tair_mean
        residual_c = y_c - pred_c

        count = x.shape[0]
        for index in range(count):
            height, width = [int(v) for v in batch["shape"][index].tolist()]
            valid_mask = mask[index, 0, :height, :width].detach().cpu().numpy() > 0
            coverage = float(valid_mask.mean()) if valid_mask.size else 0.0
            if coverage <= 0:
                continue
            city = batch["city"][index]
            year = int(batch["year"][index])
            month = int(batch["month"][index])
            phase = batch["phase"][index]
            caption_prefix = f"{city} {year}-{month:02d} {phase}"
            record = {
                "coverage": coverage,
                "caption_prefix": caption_prefix,
                "lst": x_lst_c[index, 0, :height, :width].detach().cpu().numpy(),
                "target": y_c[index, 0, :height, :width].detach().cpu().numpy(),
                "pred": pred_c[index, 0, :height, :width].detach().cpu().numpy(),
                "residual": residual_c[index, 0, :height, :width].detach().cpu().numpy(),
                "mask": valid_mask,
            }
            records.append(record)
            if image_selection != "most_valid" and len(records) >= max_examples:
                break
        if image_selection != "most_valid" and len(records) >= max_examples:
            break

    if image_selection == "most_valid":
        records = sorted(records, key=lambda item: item["coverage"], reverse=True)
    records = records[:max_examples]

    images = []
    for record in records:
        valid_mask = record["mask"]
        coverage_pct = record["coverage"] * 100.0
        caption_prefix = f"{record['caption_prefix']} | valid={coverage_pct:.1f}%"
        lst_img = _scalar_map_to_rgb(record["lst"], valid_mask)
        target_img = _scalar_map_to_rgb(record["target"], valid_mask)
        pred_img = _scalar_map_to_rgb(record["pred"], valid_mask)
        residual_img = _residual_map_to_rgb(
            record["residual"],
            valid_mask,
            limit=residual_limit,
        )
        images.extend(
            [
                wandb.Image(lst_img, caption=f"{caption_prefix} | input LST C"),
                wandb.Image(target_img, caption=f"{caption_prefix} | target AT C"),
                wandb.Image(pred_img, caption=f"{caption_prefix} | predicted AT C"),
                wandb.Image(
                    residual_img,
                    caption=(
                        f"{caption_prefix} | residual target-pred C "
                        f"[-{residual_limit:g}, {residual_limit:g}]"
                    ),
                ),
            ]
        )

    if images:
        wandb_run.log({"eval/maps": images, "epoch": epoch})


def _scalar_map_to_rgb(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    scaled = _robust_scale(values, mask)
    gray = (scaled * 255).astype("uint8")
    rgb = np.stack([gray, gray, gray], axis=-1)
    rgb[~mask] = np.array([40, 40, 40], dtype="uint8")
    return rgb


def _residual_map_to_rgb(values: np.ndarray, mask: np.ndarray, limit: float) -> np.ndarray:
    limit = max(float(limit), 1e-6)
    scaled = np.clip(values / limit, -1.0, 1.0)
    red = np.full(values.shape, 255.0, dtype="float32")
    green = np.full(values.shape, 255.0, dtype="float32")
    blue = np.full(values.shape, 255.0, dtype="float32")

    positive = np.clip(scaled, 0.0, 1.0)
    negative = np.clip(-scaled, 0.0, 1.0)
    green -= 255.0 * positive
    blue -= 255.0 * positive
    red -= 255.0 * negative
    green -= 255.0 * negative

    rgb = np.stack([red, green, blue], axis=-1)
    rgb[~mask] = np.array([40, 40, 40], dtype="float32")
    return rgb.astype("uint8")


def _robust_scale(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    valid = values[mask & np.isfinite(values)]
    if valid.size == 0:
        return np.zeros_like(values, dtype="float32")
    lo, hi = np.percentile(valid, [2, 98])
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0).astype("float32")


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
