from __future__ import annotations

import argparse
from functools import partial
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from torch.utils.data import DataLoader

from lstat.collate import pad_collate
from lstat.config import load_config, resolve_output_dir
from lstat.dataset import LstatDataset, Normalization, input_channel_count
from lstat.index import build_examples
from lstat.model import ResUNet
from lstat.train import _evaluate, _limit_examples, _resolve_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/full_gpu.toml")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--output", default="")
    parser.add_argument("--device", default="")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = resolve_output_dir(config, args.config)
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else output_dir / "best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = _resolve_device(args.device or config["training"].get("device", "auto"))
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = checkpoint.get("config", config)
    data_config = config["data"]
    split_years = {
        "train": config["split"]["train_years"],
        "val": config["split"]["val_years"],
        "test": config["split"]["test_years"],
    }[args.split]

    examples = build_examples(config["data_root"], years=split_years)
    limit_key = {
        "train": "max_train_examples",
        "val": "max_val_examples",
        "test": "max_test_examples",
    }[args.split]
    examples = _limit_examples(
        examples,
        config["training"].get(limit_key, 0),
        int(config.get("seed", 42)) + 10,
    )

    normalization = Normalization(
        lst_mean=float(data_config["lst_mean"]),
        lst_std=float(data_config["lst_std"]),
        tair_mean=float(data_config["tair_mean"]),
        tair_std=float(data_config["tair_std"]),
        apply_modis_correction=bool(data_config["apply_modis_correction"]),
    )
    dataset = LstatDataset(
        examples,
        normalization=normalization,
        include_mask_channel=bool(data_config["include_mask_channel"]),
        include_time_channels=bool(data_config["include_time_channels"]),
    )
    loader = DataLoader(
        dataset,
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
        include_mask_channel=bool(data_config["include_mask_channel"]),
        include_time_channels=bool(data_config["include_time_channels"]),
    )
    model = ResUNet(
        in_channels=in_channels,
        out_channels=1,
        base_channels=int(model_config["model"]["base_channels"]),
        depth=int(model_config["model"]["depth"]),
        dropout=float(model_config["model"]["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])

    metrics = _evaluate(
        model=model,
        loader=loader,
        device=device,
        loss_name=config["training"].get("loss", "huber"),
        target_std=normalization.tair_std,
    )
    result = {
        "split": args.split,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_val_mae_c": checkpoint.get("val_mae_c"),
        "examples": len(examples),
        **metrics,
    }

    output_path = Path(args.output) if args.output else output_dir / f"{args.split}_metrics.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"wrote: {output_path}")


if __name__ == "__main__":
    main()
