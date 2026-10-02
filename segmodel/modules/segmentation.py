"""多 frozen PFM 条件调制所需的模块：解耦投影、路由器与逐层条件生成。"""

from functools import partial
from math import gcd

import torch
from timm.layers import DropPath
from torch import nn
from torch.nn import functional as F
import pandas as pd
from .blocks import ResidualBlock, Resize

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

class ConditionalNorm(nn.Module):
  def __init__(self, channels, global_dim, spatial_dim, pfm_count, g_top_k, s_top_k):
      super().__init__()
      self.eps = 1e-6

      self.global_head = _zero_modulation(nn.Linear(global_dim, 2*channels))
      self.spatial_head = _zero_modulation(nn.Conv2d(spatial_dim, 2, 1))

      self.resize = Resize()

  def forward(self, x, task_global, task_spatial, available, routing, global_router, spatial_router):
      u = x.mean(1, keepdim=True)
      s = (x - u).pow(2).mean(1, keepdim=True)
      x_hat = (x - u) / torch.sqrt(s + self.eps)

      weight = global_router(x.mean(dim=(2, 3)), task_global, available)
      routing.append(weight.detach())
      c_g = sum(weight[:, i:i+1] * f for i, f in enumerate(task_global))
      gamma_g, beta_g = self.global_head(c_g).chunk(2, dim=-1)

      grid = x.shape[-2:]
      f1_spatial = [
          f if f.shape[-2:] == grid else self.resize(f, grid)
          for f in task_spatial
      ]
      alpha = spatial_router(x, f1_spatial, available)
      c_s = sum(alpha[:, i:i+1] * f for i, f in enumerate(f1_spatial))
      gamma_s, beta_s = self.spatial_head(c_s).chunk(2, dim=1)
      gamma = gamma_s + gamma_g[:, :, None, None]
      beta = beta_s + beta_g[:, :, None, None]

      return (1.0 + gamma) * x_hat + beta

class Stem(nn.Module):
    def __init__(self, in_chans=3, out_chans=64, down_factor=7):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=down_factor, stride=down_factor),
            LayerNorm(out_chans, eps=1e-6)
        )

    def forward(self, image):
        fm1 = self.block1(image)
        return fm1

class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, drop_path, cnorm_kwargs, layer_scale_init_value=1e-2):
        super().__init__()

        self.dwconv = nn.Conv2d(
            dim, dim,
            kernel_size=7,
            padding=3,
            groups=dim
        )

        self.cnorm = ConditionalNorm(dim, **cnorm_kwargs)
        self.norm = LayerNorm(dim, eps=1e-6)

        self.pwconv1 = nn.Conv2d(dim, dim * 4, kernel_size=1)

        self.act = nn.GELU()

        self.pwconv2 = nn.Conv2d(dim * 4, dim, kernel_size=1)

        self.drop_path = DropPath(drop_path)

        self.gamma = nn.Parameter(
            layer_scale_init_value * torch.ones(dim)
        )

    def forward(self, x, task_global, task_spatial, available, routing, global_router, spatial_router):
        shortcut = x

        x = self.dwconv(x)

        x = self.cnorm(x, task_global, task_spatial, available, routing, global_router, spatial_router)
        # x = self.norm(x)

        x = self.pwconv1(x)

        x = self.act(x)

        x = self.pwconv2(x)

        gamma = self.gamma[None, :, None, None]
        x = shortcut + self.drop_path(gamma * x)

        return x


class ConvNeXtStage(nn.Module):
    def __init__(self, channels, drop_path, blocks, cnorm_kwargs):
        super().__init__()
        self.global_router = JointRouter(
            channels,
            cnorm_kwargs["global_dim"],
            cnorm_kwargs["pfm_count"],
            cnorm_kwargs["g_top_k"],
        )
        self.spatial_router = SpatialRouter(
            channels,
            cnorm_kwargs["spatial_dim"],
            cnorm_kwargs["pfm_count"],
            cnorm_kwargs["s_top_k"],
        )
        self.blocks = nn.ModuleList(
            ConvNeXtBlock(channels, drop_path, cnorm_kwargs) for _ in range(blocks)
        )

    def forward(self, x, task_global, task_spatial, available, routing):
        for block in self.blocks:
            x = block(
                x, task_global, task_spatial, available, routing,
                self.global_router, self.spatial_router,
            )
        return x


class Downsample(nn.Module):
    def __init__(self, in_chans, out_chans):
        super().__init__()

        self.down = nn.Sequential(
            LayerNorm(in_chans, eps=1e-6),
            nn.Conv2d(in_chans, out_chans, kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.down(x)


class Upsample(nn.Module):
    def __init__(self, in_chans, out_chans):
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_chans,
            out_chans,
            kernel_size=2,
            stride=2,
        )

    def forward(self, x):
        return self.up(x)


class DecoderStage(nn.Module):
    def __init__(
        self,
        in_chans,
        skip_chans,
        out_chans,
        drop_path,
        blocks,
        cnorm_kwargs,
        skip_drop_path=0.0,
    ):
        super().__init__()

        self.up = Upsample( in_chans, out_chans)
        self.skip_drop_path = DropPath(skip_drop_path)

        self.fuse = nn.Sequential(
            LayerNorm(out_chans + skip_chans, eps=1e-6),
            nn.Conv2d(out_chans + skip_chans, out_chans, kernel_size=1),
        )

        self.blocks = ConvNeXtStage(out_chans, drop_path, blocks, cnorm_kwargs)

    def forward(self, x, skip, task_global, task_spatial, available, routing):
        x = self.up(x)

        # U-Net skip connection
        x = torch.cat([x, self.skip_drop_path(skip)], dim=1)

        # channel fusion
        x = self.fuse(x)

        # ConvNeXt feature extraction
        x = self.blocks(x, task_global, task_spatial, available, routing)

        return x

class ConvNeXtUnet(nn.Module):
    def __init__(
        self,
        encoder_chans=(64, 96, 128),
        decoder_chans=(96, 64),
        drop_path=(0.1, 0.3, 0.5),
        blocks=2,
        global_dim=256,
        spatial_dim=64,
        pfm_count=5,
        g_top_k=2,
        s_top_k=2,
        skip_drop_path=0.1,
    ):
        super().__init__()

        cnorm_kwargs = dict(
            global_dim=global_dim,
            spatial_dim=spatial_dim,
            pfm_count=pfm_count,
            g_top_k=g_top_k,
            s_top_k=s_top_k,
        )

        # Encoder stages
        self.encoder_stages = nn.ModuleList()

        for channels, rate in zip(encoder_chans, drop_path):
            self.encoder_stages.append(
                ConvNeXtStage(channels, rate, blocks, cnorm_kwargs)
            )

        # Downsampling
        self.downsamples = nn.ModuleList()

        for i in range(len(encoder_chans) - 1):
            self.downsamples.append(
                Downsample(encoder_chans[i], encoder_chans[i + 1])
            )

        # Decoder
        self.decoder_stages = nn.ModuleList()

        current_chans = encoder_chans[-1]

        # skip channel:
        skip_chans_list = list(
            reversed(encoder_chans[:-1])
        )

        for out_chans, skip_chans, rate in zip(
            decoder_chans,
            skip_chans_list,
            reversed(drop_path[:-1]),
        ):
            self.decoder_stages.append(
                DecoderStage(
                    in_chans=current_chans,
                    skip_chans=skip_chans,
                    out_chans=out_chans,
                    drop_path=rate,
                    blocks=blocks,
                    cnorm_kwargs=cnorm_kwargs,
                    skip_drop_path=skip_drop_path,
                )
            )

            current_chans = out_chans

        self.out_channels = current_chans

    def forward(self, x, task_global, task_spatial, available, routing):

        skips = []
        # Encoder
        for i, stage in enumerate(self.encoder_stages):
            x = stage(x, task_global, task_spatial, available, routing)

            if i < len(self.downsamples):
                skips.append(x)
                x = self.downsamples[i](x)

        # Decoder
        for stage, skip in zip(self.decoder_stages, reversed(skips)):
            x = stage(x, skip, task_global, task_spatial, available, routing)

        return x


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


def _zero_modulation(module):
    """AdaLN-Zero：调制头输出层零初始化，训练初期整个条件分支不生效。"""
    nn.init.zeros_(module.weight)
    nn.init.zeros_(module.bias)
    return module
