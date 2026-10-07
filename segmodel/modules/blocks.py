# 基础层: 通道归一化、特征缩放、spot 投影头。
import torch
from torch import nn
from torch.nn import functional as F

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x_hat = (x - u) / torch.sqrt(s + self.eps)

        x_hat = self.weight[:, None, None] * x_hat + self.bias[:, None, None]

        return x_hat


class Resize(nn.Module):
    def forward(self, value, size):
        return F.interpolate(
            value,
            size=size,
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )


class ProjectionHead(nn.Sequential):
    def __init__(self, input_features, output_features):
        super().__init__(
            nn.Linear(input_features, output_features),
            nn.LayerNorm(output_features),
            nn.GELU(),
            nn.Linear(output_features, output_features),
        )
