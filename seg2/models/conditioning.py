from functools import partial

import torch
from torch import nn
from torch.nn import functional as F

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-06):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = F.layer_norm(
                x.permute(0, 2, 3, 1),
                self.normalized_shape,
                self.weight.to(dtype=x.dtype),
                self.bias.to(dtype=x.dtype),
                self.eps,
            )
        return x.permute(0, 3, 1, 2)

# class PFMProjector(nn.Module):
#     def __init__(self, input_dims, feature_dim, spatial):
#         super().__init__()
#         project = partial(nn.Conv2d, kernel_size=1) if spatial else nn.Linear
#         self.projects = nn.ModuleList(
#             project(int(input_dim), feature_dim) for input_dim in input_dims
#         )
#         self.shared_proj = project(feature_dim, feature_dim)
#
#     def forward(self, values):
#         values = [F.normalize(value, p=2, dim=1, eps=1e-06) for value in values]
#         return [
#             self.shared_proj(project(value))
#             for project, value in zip(self.projects, values)
#         ]

class PFMProjector(nn.Module):
    def __init__(self, input_dims, feature_dim, spatial):
        super().__init__()
        project = partial(nn.Conv2d, kernel_size=1) if spatial else nn.Linear
        self.projects = nn.ModuleList(
            project(int(input_dim), feature_dim) for input_dim in input_dims
        )

    def forward(self, values):
        values = [F.normalize(value, p=2, dim=1, eps=1e-06) for value in values]
        return [project(value) for project, value in zip(self.projects, values)]


class GlobalRouter(nn.Module):
    def __init__(self, channels, feature_dim, pfm_count, top_k):
        super().__init__()
        self.state_norm = nn.LayerNorm(channels)
        self.feature_norm = nn.LayerNorm(feature_dim)
        self.top_k = int(top_k)
        input_dim = channels + pfm_count * feature_dim
        self.router = nn.Sequential(
            nn.Linear(input_dim, input_dim // 4),
            nn.GELU(),
            nn.Linear(input_dim // 4, pfm_count),
        )

    def forward(self, state, features):
        features = torch.stack(features, dim=1).float()
        features = self.feature_norm(features)
        features = features.flatten(1)
        state = self.state_norm(state.float())
        logits = self.router(torch.cat((state, features), dim=1))
        return _topk_softmax(logits, self.top_k, dim=-1)


class SpatialRouter(nn.Module):
    def __init__(self, channels, feature_dim, pfm_count, top_k):
        super().__init__()
        self.state_norm = LayerNorm(channels)
        self.feature_norm = LayerNorm(feature_dim)
        self.top_k = top_k
        in_channels = channels + pfm_count * feature_dim
        self.router = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 1),
            nn.GELU(),
            nn.Conv2d(in_channels // 4, pfm_count, 1),
        )

    def forward(self, state, features):
        state = self.state_norm(state.float())
        features = torch.stack(
            [self.feature_norm(f.float()) for f in features], dim=1
        ).flatten(1, 2)
        joint = torch.cat((state, features), dim=1)
        logits = self.router(joint).float()
        return _topk_softmax(logits, self.top_k, dim=1)


class ConditionalNorm(nn.Module):
    def __init__(self, channels, global_dim, spatial_dim, use_spatial=False):
        super().__init__()
        self.eps = 1e-06
        self.use_spatial = bool(use_spatial)
        self.global_head = _zero_modulation(nn.Linear(global_dim, 2 * channels))
        self.spatial_head = _zero_modulation(nn.Conv2d(spatial_dim, 2 * channels, 1))

    def forward(self, x, task_global, task_spatial, global_router, spatial_router):
        with torch.autocast(device_type=x.device.type, enabled=False):
            x_hat = F.layer_norm(
                x.float().permute(0, 2, 3, 1), (x.shape[1],), eps=self.eps
            )
        x_hat = x_hat.permute(0, 3, 1, 2).to(dtype=x.dtype)
        weight = global_router(x.mean(dim=(2, 3)), task_global)
        c_g = sum((weight[:, i : i + 1] * f for i, f in enumerate(task_global)))
        gamma_g, beta_g = self.global_head(c_g).chunk(2, dim=-1)
        gamma = gamma_g[:, :, None, None]
        beta = beta_g[:, :, None, None]
        if self.use_spatial:
            grid = x.shape[-2:]
            spatial_features = [
                f
                if f.shape[-2:] == grid
                else F.interpolate(
                    f.float(), size=grid, mode="bilinear", align_corners=False, antialias=True
                ).to(dtype=f.dtype)
                for f in task_spatial
            ]
            alpha = spatial_router(x, spatial_features)
            c_s = sum((alpha[:, i : i + 1] * f for i, f in enumerate(spatial_features)))
            gamma_s, beta_s = self.spatial_head(c_s).chunk(2, dim=1)
            gamma = gamma + gamma_s
            beta = beta + beta_s
        return (1.0 + gamma.to(dtype=x.dtype)) * x_hat + beta.to(dtype=x.dtype)


def _zero_modulation(module):
    nn.init.zeros_(module.weight)
    nn.init.zeros_(module.bias)
    return module


def _topk_softmax(logits, top_k, dim):
    values, indices = logits.topk(min(top_k, logits.shape[dim]), dim=dim)
    selected = torch.full_like(logits, float("-inf"))
    selected.scatter_(dim, indices, values)
    return torch.softmax(selected, dim=dim)
