from math import gcd
from typing import cast

import torch
from torch import nn
from torch.nn import functional as F


class Resize(nn.Module):
    def forward(self, value, size):
        return F.interpolate(
            value,
            size=size,
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )


class ConvBlock(nn.Sequential):
    def __init__(self, input_channels, output_channels, stride=1):
        super().__init__(
            nn.Conv2d(
                input_channels,
                output_channels,
                3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(gcd(8, output_channels), output_channels),
            nn.SiLU(),
        )


class BlurPool(nn.Module):
    def __init__(self, channels, stride=2):
        super().__init__()

        kernel_1d = torch.tensor([1.0, 2.0, 1.0])
        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
        kernel_2d = kernel_2d / kernel_2d.sum()

        kernel = kernel_2d[None, None, :, :]
        kernel = kernel.repeat(channels, 1, 1, 1)

        self.register_buffer("kernel", kernel)
        self.channels = channels
        self.stride = stride

    def forward(self, x):
        x = F.pad(x, (1, 1, 1, 1), mode="reflect")
        kernel = cast(torch.Tensor, self.kernel)
        return F.conv2d(
            x,
            kernel,
            stride=self.stride,
            padding=0,
            groups=self.channels,
        )


class ResidualBlock(nn.Module):
    def __init__(self, input_channels, output_channels, stride=1):
        super().__init__()
        if stride == 2:
            self.downsample = BlurPool(
                input_channels,
                stride=2,
            )
        else:
            self.downsample = nn.Identity()
        self.body = nn.Sequential(
            ConvBlock(input_channels, output_channels),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(gcd(8, output_channels), output_channels),
        )
        self.skip = (
            nn.Identity()
            if input_channels == output_channels and stride == 1
            else nn.Sequential(
                nn.Conv2d(
                    input_channels,
                    output_channels,
                    1,
                    bias=False,
                ),
                nn.GroupNorm(gcd(8, output_channels), output_channels),
            )
        )

    def forward(self, x):
        x = self.downsample(x)
        return F.silu(self.body(x) + self.skip(x))


class ModulatedBlock(nn.Module):
    """AdaLN 调制块：由外部条件给出 (γ, β, g)，控制归一化后的缩放偏移与分支强度。

    调制注入在 pre-norm 之后；只有输入输出通道一致且不下采样时才有外层残差
    `X + g·h`，因此这类块在 g=0 时完全旁路，训练初期等价于纯 backbone。
    下采样块的形状改变，只能退化成普通的 pre-activation 残差块。

    dropout 丢整条通道（Dropout2d）且作用在 body 输出上、加残差之前，shortcut 不受影响。
    """

    def __init__(self, input_channels, output_channels, stride=1, dropout=0.3):
        super().__init__()
        self.downsample = (
            BlurPool(input_channels, stride=2) if stride == 2 else nn.Identity()
        )
        self.norm = nn.GroupNorm(gcd(8, input_channels), input_channels)
        self.body = nn.Sequential(
            ConvBlock(input_channels, output_channels),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(gcd(8, output_channels), output_channels),
        )
        self.skip = (
            nn.Identity()
            if input_channels == output_channels and stride == 1
            else nn.Sequential(
                nn.Conv2d(input_channels, output_channels, 1, bias=False),
                nn.GroupNorm(gcd(8, output_channels), output_channels),
            )
        )
        self.gated = input_channels == output_channels and stride == 1
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x, gamma, beta, gate, gamma_spatial=None, beta_spatial=None):
        residual = x if self.gated else None
        features = self.downsample(x)
        normalized = self.norm(features)
        if gamma_spatial is not None:
            gamma = gamma + gamma_spatial
            beta = beta + beta_spatial
        # GroupNorm 在 autocast 下保持 fp32，γ/β 也由 fp32 的调制头给出，这里的
        # 1+γ 必须留在 fp32：bf16 下 1 与接近 0 的 γ 相加会直接吞掉 γ
        modulated = (1.0 + gamma) * normalized + beta
        features = F.silu(
            self.dropout(self.body(modulated.to(features.dtype)))
            + self.skip(features)
        )
        if residual is None:
            return features
        return residual + gate * features


class ProjectionHead(nn.Sequential):
    def __init__(self, input_features, output_features):
        super().__init__(
            nn.Linear(input_features, output_features),
            nn.LayerNorm(output_features),
            nn.GELU(),
            nn.Linear(output_features, output_features),
        )
