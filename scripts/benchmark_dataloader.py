from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from torch.utils.data import DataLoader

from lstat.collate import pad_collate
from lstat.config import load_config
from lstat.dataset import Normalization
from lstat.train import (
    _build_dataset,
    _dataloader_worker_kwargs,
    _resolve_device,
    _seed_everything,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/full_gpu_daily_pilot.toml")
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--batches", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--copy-to-device", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    seed = int(config.get("seed", 42))
    _seed_everything(seed)

    data_cfg = config["data"]
    train_cfg = config["training"]
    batch_size = args.batch_size or int(train_cfg["batch_size"])
    num_workers = (
        args.num_workers
        if args.num_workers is not None
        else int(train_cfg["num_workers"])
    )
    device = _resolve_device(train_cfg.get("device", "auto"))
    normalization = Normalization(
        lst_mean=float(data_cfg["lst_mean"]),
        lst_std=float(data_cfg["lst_std"]),
        tair_mean=float(data_cfg["tair_mean"]),
        tair_std=float(data_cfg["tair_std"]),
        apply_modis_correction=bool(data_cfg["apply_modis_correction"]),
    )
    dataset = _build_dataset(
        config=config,
        split=args.split,
        normalization=normalization,
        seed=seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=args.split == "train",
        num_workers=num_workers,
        **_dataloader_worker_kwargs(config, device, num_workers),
        collate_fn=partial(
            pad_collate,
            min_size=int(train_cfg["min_pad_size"]),
            multiple=int(train_cfg["pad_multiple"]),
            crop_size=int(
                train_cfg.get(
                    "train_crop_size" if args.split == "train" else "eval_crop_size",
                    0,
                )
            ),
            random_crop=args.split == "train",
        ),
    )

    print(f"examples: {len(dataset)}")
    print(f"batch_size: {batch_size}")
    print(f"num_workers: {num_workers}")
    print(f"copy_to_device: {args.copy_to_device}")
    print(f"device: {device}")
    print(
        "crop_size:",
        int(
            train_cfg.get(
                "train_crop_size" if args.split == "train" else "eval_crop_size",
                0,
            )
        ),
    )

    start = time.time()
    total_samples = 0
    total_pixels = 0
    total_batches = min(args.batches, len(loader))
    for batch_index, batch in enumerate(loader, start=1):
        if args.copy_to_device:
            batch["x"].to(device, non_blocking=True)
            batch["y"].to(device, non_blocking=True)
            batch["mask"].to(device, non_blocking=True)
        if batch_index == 1:
            print(f"batch_shape: {tuple(batch['x'].shape)}")
        total_samples += int(batch["x"].shape[0])
        total_pixels += int(batch["mask"].sum().item())
        if batch_index >= total_batches:
            break

    elapsed = time.time() - start
    print(f"batches: {total_batches}")
    print(f"samples: {total_samples}")
    print(f"valid_pixels: {total_pixels}")
    print(f"seconds: {elapsed:.2f}")
    print(f"batches_per_s: {total_batches / max(elapsed, 1e-6):.3f}")
    print(f"samples_per_s: {total_samples / max(elapsed, 1e-6):.3f}")


if __name__ == "__main__":
    main()
