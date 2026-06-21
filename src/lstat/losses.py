from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_mae(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    diff = (pred - target).abs() * mask
    return diff.sum() / mask.sum().clamp_min(1.0)


def masked_huber(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    beta: float = 1.0,
) -> torch.Tensor:
    loss = F.smooth_l1_loss(pred, target, reduction="none", beta=beta)
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


def masked_rmse_celsius(
    pred_norm: torch.Tensor,
    target_norm: torch.Tensor,
    mask: torch.Tensor,
    target_std: float,
) -> torch.Tensor:
    diff_c = (pred_norm - target_norm) * target_std
    mse = ((diff_c**2) * mask).sum() / mask.sum().clamp_min(1.0)
    return torch.sqrt(mse)


def masked_mae_celsius(
    pred_norm: torch.Tensor,
    target_norm: torch.Tensor,
    mask: torch.Tensor,
    target_std: float,
) -> torch.Tensor:
    diff_c = (pred_norm - target_norm).abs() * target_std
    return (diff_c * mask).sum() / mask.sum().clamp_min(1.0)
