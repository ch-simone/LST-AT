from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lstat.cache import write_cache_split
from lstat.config import load_config
from lstat.dataset import LstatDataset, Normalization
from lstat.index import build_examples
from lstat.train import _limit_examples, _split_limit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/full_gpu_daily_pilot_cached.toml")
    parser.add_argument("--output", default="")
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "val", "test"],
        default=["train", "val"],
    )
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--ignore-limits", action="store_true")
    parser.add_argument("--progress-interval", type=int, default=500)
    args = parser.parse_args()

    config = load_config(args.config)
    cache_root = Path(args.output or config.get("cache", {}).get("root", "data/cache/lstat"))
    cache_root.mkdir(parents=True, exist_ok=True)

    normalization = Normalization(
        lst_mean=float(config["data"]["lst_mean"]),
        lst_std=float(config["data"]["lst_std"]),
        tair_mean=float(config["data"]["tair_mean"]),
        tair_std=float(config["data"]["tair_std"]),
        apply_modis_correction=bool(config["data"]["apply_modis_correction"]),
    )
    root_metadata = {
        "config": str(args.config),
        "data_root": str(config["data_root"]),
        "dtype": args.dtype,
        "splits": args.splits,
    }
    (cache_root / "cache.json").write_text(
        _json_dump(root_metadata),
        encoding="utf-8",
    )

    for split in args.splits:
        start = time.time()
        examples = _examples_for_split(
            config=config,
            split=split,
            ignore_limits=args.ignore_limits,
        )
        dataset = LstatDataset(
            examples,
            normalization=normalization,
            include_mask_channel=bool(config["data"]["include_mask_channel"]),
            include_time_channels=bool(config["data"]["include_time_channels"]),
        )
        print(
            f"{split}: building cache at {cache_root / split} "
            f"for {len(examples)} examples",
            flush=True,
        )
        metadata = write_cache_split(
            cache_dir=cache_root / split,
            split=split,
            examples=examples,
            dataset=dataset,
            dtype=args.dtype,
            overwrite=args.overwrite,
            progress_interval=args.progress_interval,
        )
        seconds = time.time() - start
        print(
            f"{split}: done in {seconds:.1f}s "
            f"shape=({metadata['count']}, {metadata['channels']}, "
            f"{metadata['height']}, {metadata['width']})",
            flush=True,
        )


def _examples_for_split(config: dict, split: str, ignore_limits: bool):
    split_years = {
        "train": config["split"]["train_years"],
        "val": config["split"]["val_years"],
        "test": config["split"].get("test_years", []),
    }[split]
    examples = build_examples(config["data_root"], years=split_years)
    if ignore_limits:
        return examples
    seed_offset = {"train": 0, "val": 1, "test": 10}[split]
    return _limit_examples(
        examples,
        _split_limit(config, split),
        int(config.get("seed", 42)) + seed_offset,
    )


def _json_dump(value: dict) -> str:
    import json

    return json.dumps(value, indent=2) + "\n"


if __name__ == "__main__":
    main()
