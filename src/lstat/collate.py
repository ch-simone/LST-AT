from __future__ import annotations

import math
import random

import torch
import torch.nn.functional as F


def pad_collate(
    batch: list[dict],
    min_size: int = 32,
    multiple: int = 8,
    crop_size: int = 0,
    random_crop: bool = False,
) -> dict:
    if crop_size > 0:
        batch = [
            _crop_item(item, crop_size=crop_size, random_crop=random_crop)
            for item in batch
        ]
    max_h = max(item["x"].shape[-2] for item in batch)
    max_w = max(item["x"].shape[-1] for item in batch)
    out_h = _round_up(max(max_h, min_size), multiple)
    out_w = _round_up(max(max_w, min_size), multiple)

    xs, ys, masks = [], [], []
    shapes = []
    for item in batch:
        x = torch.from_numpy(item["x"])
        y = torch.from_numpy(item["y"])
        mask = torch.from_numpy(item["mask"])
        shapes.append((x.shape[-2], x.shape[-1]))
        pad = (0, out_w - x.shape[-1], 0, out_h - x.shape[-2])
        xs.append(F.pad(x, pad))
        ys.append(F.pad(y, pad))
        masks.append(F.pad(mask, pad))

    return {
        "x": torch.stack(xs),
        "y": torch.stack(ys),
        "mask": torch.stack(masks),
        "shape": torch.tensor(shapes, dtype=torch.long),
        "city": [item["city"] for item in batch],
        "year": torch.tensor([item["year"] for item in batch]),
        "month": torch.tensor([item["month"] for item in batch]),
        "day": torch.tensor([item["day"] for item in batch]),
        "temporal_resolution": [item["temporal_resolution"] for item in batch],
        "phase": [item["phase"] for item in batch],
    }


def _round_up(value: int, multiple: int) -> int:
    return int(math.ceil(value / multiple) * multiple)


def _crop_item(item: dict, crop_size: int, random_crop: bool) -> dict:
    h, w = item["x"].shape[-2:]
    crop_h = min(h, crop_size)
    crop_w = min(w, crop_size)
    if crop_h == h and crop_w == w:
        return item

    if random_crop:
        y0 = random.randint(0, h - crop_h) if h > crop_h else 0
        x0 = random.randint(0, w - crop_w) if w > crop_w else 0
    else:
        y0 = max((h - crop_h) // 2, 0)
        x0 = max((w - crop_w) // 2, 0)
    y1 = y0 + crop_h
    x1 = x0 + crop_w

    cropped = dict(item)
    cropped["x"] = item["x"][:, y0:y1, x0:x1]
    cropped["y"] = item["y"][:, y0:y1, x0:x1]
    cropped["mask"] = item["mask"][:, y0:y1, x0:x1]
    return cropped
