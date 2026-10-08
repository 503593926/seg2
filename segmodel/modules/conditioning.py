from functools import partial

import torch
from torch import nn
from torch.nn import functional as F

from segmodel.modules.blocks import Resize

class PFMDecomposer(nn.Module):
    def __init__(self, input_dims, feature_dim, spatial):
        super().__init__()

        project = partial(nn.Conv2d, kernel_size=1) if spatial else nn.Linear

        self.projects = nn.ModuleList(
            project(int(input_dim), feature_dim * 2) for input_dim in input_dims
        )
        self.shared_task_proj = project(feature_dim * 2, feature_dim)
        self.shared_identity_proj = project(feature_dim * 2, feature_dim)

    def forward(self, values):
        task_f, identity_f = [], []
        for project, value in zip(self.projects, values):
            value = F.normalize(value.float(), p=2, dim=1, eps=1e-6)
            projected = project(value)
            task_f.append(self.shared_task_proj(projected))
            identity_f.append(self.shared_identity_proj(projected))
        return task_f, identity_f


class JointRouter(nn.Module):
    def __init__(self, channels, feature_dim, pfm_count, top_k):
        super().__init__()
        self.state_norm = nn.LayerNorm(channels)
        self.feature_norm = nn.LayerNorm(feature_dim)
        self.top_k = int(top_k)

        input_dim = channels + pfm_count * feature_dim
        self.router = nn.Sequential(
            nn.Linear(input_dim, input_dim // 4),
            nn.SiLU(),
            nn.Linear(input_dim // 4, pfm_count),
        )

    def forward(self, state, features, available=None):
        features = torch.stack(features, dim=1).float()  # [B,P,D]
        features = self.feature_norm(features)

        if available is not None:
            features = features * available[:, :, None]

        features = features.flatten(1)

        state = self.state_norm(state.float())
        logits = self.router(torch.cat((state, features), dim=1))

        if available is not None:
            logits = logits.masked_fill(~available, float("-inf"))

        k = min(self.top_k, logits.shape[-1])
        values, indices = logits.topk(k, dim=-1)

        topk_logits = torch.full_like(logits, float("-inf"))
        topk_logits.scatter_(-1, indices, values)

        return torch.softmax(topk_logits, dim=-1)


class SpatialRouter(nn.Module):
    def __init__(self, channels, feature_dim, pfm_count, top_k):
        super().__init__()
        self.top_k = top_k
        in_channels = channels + pfm_count * feature_dim
        self.router = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 1),
            nn.SiLU(),
            nn.Conv2d(in_channels // 4, pfm_count, 1),
        )

    def forward(self, state, features, available=None):
        features = torch.stack(features, dim=1)  # [B, P, D, H, W]

        if available is not None:
            features = features * available[:, :, None, None, None]

        features = features.flatten(1, 2)  # [B, P*D, H, W]
        joint = torch.cat((state, features), dim=1)

        logits = self.router(joint).float()
        if available is not None:
            logits = logits.masked_fill(
                ~available[:, :, None, None],
                float("-inf"),
            )

        k = min(self.top_k, logits.shape[1])
        values, indices = logits.topk(k, dim=1)

        topk_logits = torch.full_like(logits, float("-inf"))
        topk_logits.scatter_(1, indices, values)
        return torch.softmax(topk_logits, dim=1)


class ConditionalNorm(nn.Module):
    def __init__(self, channels, global_dim, spatial_dim):
        super().__init__()
        self.eps = 1e-6

        self.global_head = _zero_modulation(nn.Linear(global_dim, 2 * channels))
        self.spatial_head = _zero_modulation(nn.Conv2d(spatial_dim, 2, 1))

        self.resize = Resize()

    def forward(self, x, task_global, task_spatial, available, routing, global_router, spatial_router):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x_hat = (x - u) / torch.sqrt(s + self.eps)

        weight = global_router(x.mean(dim=(2, 3)), task_global, available)
        routing.append(weight.detach())
        c_g = sum(weight[:, i : i + 1] * f for i, f in enumerate(task_global))
        gamma_g, beta_g = self.global_head(c_g).chunk(2, dim=-1)

        grid = x.shape[-2:]
        f1_spatial = [
            f if f.shape[-2:] == grid else self.resize(f, grid)
            for f in task_spatial
        ]
        alpha = spatial_router(x, f1_spatial, available)
        c_s = sum(alpha[:, i : i + 1] * f for i, f in enumerate(f1_spatial))
        gamma_s, beta_s = self.spatial_head(c_s).chunk(2, dim=1)
        gamma = gamma_s + gamma_g[:, :, None, None]
        beta = beta_s + beta_g[:, :, None, None]

        return (1.0 + gamma) * x_hat + beta


def _zero_modulation(module):
    """AdaLN-Zero：调制头输出层零初始化，训练初期整个条件分支不生效。"""
    nn.init.zeros_(module.weight)
    nn.init.zeros_(module.bias)
    return module

