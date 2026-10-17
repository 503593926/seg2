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
        # 参数仅在前向转换；原生 LayerNorm 内部用 FP32 累积。
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = F.layer_norm(
                x.permute(0, 2, 3, 1), self.normalized_shape,
                self.weight.to(dtype=x.dtype), self.bias.to(dtype=x.dtype), self.eps,
            )
        return x.permute(0, 3, 1, 2)


class Resize(nn.Module):
    def forward(self, value, size):
        # 抗锯齿插值使用 FP32，结果恢复为条件特征的精度。
        return F.interpolate(
            value.float(),
            size=size,
            mode="bilinear",
            align_corners=False,
            antialias=True,
        ).to(dtype=value.dtype)
