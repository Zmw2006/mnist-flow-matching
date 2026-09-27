"""Small time-conditioned U-Net predicting an image-space velocity field."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        frequencies = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device, dtype=t.dtype) / (half - 1)
        )
        angles = t[:, None] * frequencies[None] * (2 * math.pi)
        return self.proj(torch.cat((angles.sin(), angles.cos()), dim=1))


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.time_proj = nn.Linear(time_dim, out_channels)
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(F.silu(self.norm1(x)))
        hidden = hidden + self.time_proj(F.silu(time))[:, :, None, None]
        hidden = self.conv2(F.silu(self.norm2(hidden)))
        return hidden + self.skip(x)


class VelocityUNet(nn.Module):
    def __init__(self, channels: int = 32, time_dim: int = 128):
        super().__init__()
        if channels < 8 or channels % 8 or time_dim < 4 or time_dim % 2:
            raise ValueError("channels must be a positive multiple of 8; time_dim must be even and >= 4")
        self.time = TimeEmbedding(time_dim)
        self.stem = nn.Conv2d(1, channels, 3, padding=1)
        self.enc1 = ResBlock(channels, channels, time_dim)
        self.down1 = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
        self.enc2 = ResBlock(channels, channels * 2, time_dim)
        self.down2 = nn.Conv2d(channels * 2, channels * 2, 3, stride=2, padding=1)
        self.mid = ResBlock(channels * 2, channels * 4, time_dim)
        self.dec2 = ResBlock(channels * 6, channels * 2, time_dim)
        self.dec1 = ResBlock(channels * 3, channels, time_dim)
        self.head = nn.Sequential(nn.GroupNorm(8, channels), nn.SiLU(), nn.Conv2d(channels, 1, 3, padding=1))

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1:] != (1, 28, 28) or t.shape != (x.shape[0],):
            raise ValueError("expected x [batch, 1, 28, 28] and t [batch]")
        time = self.time(t)
        skip1 = self.enc1(self.stem(x), time)
        skip2 = self.enc2(self.down1(skip1), time)
        hidden = self.mid(self.down2(skip2), time)
        hidden = F.interpolate(hidden, size=skip2.shape[-2:], mode="nearest")
        hidden = self.dec2(torch.cat((hidden, skip2), dim=1), time)
        hidden = F.interpolate(hidden, size=skip1.shape[-2:], mode="nearest")
        return self.head(self.dec1(torch.cat((hidden, skip1), dim=1), time))
