from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import shutil
from typing import Iterable

import numpy as np

from .dataset import LstatDataset
from .geotiff import read_geotiff_info
from .index import SingleMapExample


CACHE_VERSION = 1
SAMPLE_DIR_NAME = "samples"


class CachedLstatDataset:
    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        metadata_path = self.cache_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Cache metadata not found: {metadata_path}")

        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.records = self.metadata["records"]
        self.storage = self.metadata.get("storage", "memmap")
        self.x = None
        self.y = None
        self.mask = None

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        height = int(record["height"])
        width = int(record["width"])
        x, y, mask = self._load_item_arrays(index, record, height, width)
        return {
            "x": x,
            "y": y,
            "mask": mask,
            "city": record["city"],
            "year": int(record["year"]),
            "month": int(record["month"]),
            "day": int(record["day"]),
            "temporal_resolution": record["temporal_resolution"],
            "phase": record["phase"],
        }

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["x"] = None
        state["y"] = None
        state["mask"] = None
        return state

    def _load_item_arrays(
        self,
        index: int,
        record: dict,
        height: int,
        width: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.storage == "samples":
            sample_path = self.cache_dir / record["file"]
            with np.load(sample_path) as data:
                return (
                    np.array(data["x"], dtype="float32"),
                    np.array(data["y"], dtype="float32"),
                    np.array(data["mask"], dtype="float32"),
                )
        if self.storage == "memmap":
            self._ensure_memmaps_open()
            return (
                np.array(self.x[index, :, :height, :width], dtype="float32"),
                np.array(self.y[index, :, :height, :width], dtype="float32"),
                np.array(self.mask[index, :, :height, :width], dtype="float32"),
            )
        raise ValueError(f"Unknown cache storage: {self.storage}")

    def _ensure_memmaps_open(self) -> None:
        if self.x is None:
            self.x = np.load(self.cache_dir / "x.npy", mmap_mode="r")
            self.y = np.load(self.cache_dir / "y.npy", mmap_mode="r")
            self.mask = np.load(self.cache_dir / "mask.npy", mmap_mode="r")


def write_cache_split(
    *,
    cache_dir: str | Path,
    split: str,
    examples: list[SingleMapExample],
    dataset: LstatDataset,
    dtype: str = "float16",
    overwrite: bool = False,
    progress_interval: int = 500,
) -> dict:
    cache_dir = Path(cache_dir)
    if cache_dir.exists() and any(cache_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Cache split already exists: {cache_dir}. "
            "Use --overwrite to replace it."
        )
    if cache_dir.exists() and overwrite:
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = cache_dir / SAMPLE_DIR_NAME
    sample_dir.mkdir()

    shapes = _scan_shapes(examples, progress_interval=progress_interval)
    max_height = max(height for height, _ in shapes) if shapes else 0
    max_width = max(width for _, width in shapes) if shapes else 0
    channels = dataset[0]["x"].shape[0] if examples else 0
    count = len(examples)

    records = []
    for index in range(count):
        item = dataset[index]
        height, width = item["x"].shape[-2:]
        filename = f"{index:08d}.npz"
        np.savez(
            sample_dir / filename,
            x=item["x"].astype(dtype, copy=False),
            y=item["y"].astype(dtype, copy=False),
            mask=item["mask"].astype("uint8", copy=False),
        )
        records.append(
            {
                "city": item["city"],
                "year": int(item["year"]),
                "month": int(item["month"]),
                "day": int(item["day"]),
                "temporal_resolution": item["temporal_resolution"],
                "phase": item["phase"],
                "height": int(height),
                "width": int(width),
                "file": f"{SAMPLE_DIR_NAME}/{filename}",
                "source": _example_to_jsonable(examples[index]),
            }
        )
        if progress_interval > 0 and (
            (index + 1) % progress_interval == 0 or index + 1 == count
        ):
            print(
                f"{split}: cached {index + 1}/{count} examples "
                f"({height}x{width})",
                flush=True,
            )

    metadata = {
        "version": CACHE_VERSION,
        "storage": "samples",
        "split": split,
        "count": count,
        "dtype": dtype,
        "channels": channels,
        "height": max_height,
        "width": max_width,
        "records": records,
    }
    (cache_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata


def _scan_shapes(
    examples: Iterable[SingleMapExample],
    progress_interval: int,
) -> list[tuple[int, int]]:
    shapes = []
    seen = {}
    examples = list(examples)
    for index, example in enumerate(examples):
        key = example.modis_path
        if key not in seen:
            info = read_geotiff_info(example.modis_path)
            seen[key] = (info.height, info.width)
        shapes.append(seen[key])
        if progress_interval > 0 and (
            (index + 1) % progress_interval == 0 or index + 1 == len(examples)
        ):
            print(
                f"scan: {index + 1}/{len(examples)} examples",
                flush=True,
            )
    return shapes


def _example_to_jsonable(example: SingleMapExample) -> dict:
    data = asdict(example)
    data["modis_path"] = str(example.modis_path)
    data["era5_path"] = str(example.era5_path)
    return data
