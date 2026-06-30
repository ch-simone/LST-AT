from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def pad_collate(batch: list[dict], min_size: int = 32, multiple: int = 8) -> dict:
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
