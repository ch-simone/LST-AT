from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        groups = _group_count(out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = F.silu(self.norm1(self.conv1(x)))
        x = self.dropout(x)
        x = self.norm2(self.conv2(x))
        return F.silu(x + residual)


class ResUNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        base_channels: int = 32,
        depth: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        channels = [base_channels * (2**i) for i in range(depth + 1)]
        self.stem = ResidualBlock(in_channels, channels[0], dropout)
        self.down_blocks = nn.ModuleList(
            ResidualBlock(channels[i], channels[i + 1], dropout) for i in range(depth)
        )
        self.up_convs = nn.ModuleList(
            nn.ConvTranspose2d(channels[i + 1], channels[i], kernel_size=2, stride=2)
            for i in reversed(range(depth))
        )
        self.up_blocks = nn.ModuleList(
            ResidualBlock(channels[i] * 2, channels[i], dropout)
            for i in reversed(range(depth))
        )
        self.head = nn.Conv2d(channels[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        x = self.stem(x)
        skips.append(x)
        for block in self.down_blocks:
            x = F.max_pool2d(x, kernel_size=2)
            x = block(x)
            skips.append(x)

        skips = skips[:-1][::-1]
        for up_conv, block, skip in zip(self.up_convs, self.up_blocks, skips):
            x = up_conv(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = block(x)

        return self.head(x)


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1
