import torch
from timm.layers.drop import DropPath
from torch import nn

from segmodel.modules.blocks import LayerNorm
from segmodel.modules.conditioning import ConditionalNorm, JointRouter, SpatialRouter


def init_convnext_backbone(module):
    """只重置图像骨干, 保留条件调制头和路由器的初始化。"""
    if isinstance(module, (ConditionalNorm, JointRouter, SpatialRouter)):
        return
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    for child in module.children():
        init_convnext_backbone(child)


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

        self.encoder_stages = nn.ModuleList()

        for channels, rate in zip(encoder_chans, drop_path):
            self.encoder_stages.append(
                ConvNeXtStage(channels, rate, blocks, cnorm_kwargs)
            )

        self.downsamples = nn.ModuleList()

        for i in range(len(encoder_chans) - 1):
            self.downsamples.append(
                Downsample(encoder_chans[i], encoder_chans[i + 1])
            )

        self.decoder_stages = nn.ModuleList()

        current_chans = encoder_chans[-1]

        skip_chans_list = list(reversed(encoder_chans[:-1]))

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

        for i, stage in enumerate(self.encoder_stages):
            x = stage(x, task_global, task_spatial, available, routing)

            if i < len(self.downsamples):
                skips.append(x)
                x = self.downsamples[i](x)

        for stage, skip in zip(self.decoder_stages, reversed(skips)):
            x = stage(x, skip, task_global, task_spatial, available, routing)

        return x


class Stem(nn.Module):
    def __init__(self, in_chans=3, out_chans=64, down_factor=7):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=down_factor, stride=down_factor),
            LayerNorm(out_chans, eps=1e-6)
        )

    def forward(self, image):
        return self.block1(image)


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


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, drop_path, cnorm_kwargs, layer_scale_init_value=1e-2):
        super().__init__()

        self.dwconv = nn.Conv2d(
            dim, dim,
            kernel_size=7,
            padding=3,
            groups=dim
        )

        self.cnorm = ConditionalNorm(
            dim,
            cnorm_kwargs["global_dim"],
            cnorm_kwargs["spatial_dim"],
        )
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

        x = self.pwconv1(x)

        x = self.act(x)

        x = self.pwconv2(x)

        gamma = self.gamma[None, :, None, None]
        x = shortcut + self.drop_path(gamma * x)

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

        self.up = Upsample(in_chans, out_chans)
        self.skip_drop_path = DropPath(skip_drop_path)

        self.fuse = nn.Sequential(
            LayerNorm(out_chans + skip_chans, eps=1e-6),
            nn.Conv2d(out_chans + skip_chans, out_chans, kernel_size=1),
        )

        self.blocks = ConvNeXtStage(out_chans, drop_path, blocks, cnorm_kwargs)

    def forward(self, x, skip, task_global, task_spatial, available, routing):
        x = self.up(x)

        # skip 单独应用 DropPath, 再拼接和融合通道。
        x = torch.cat([x, self.skip_drop_path(skip)], dim=1)

        x = self.fuse(x)

        x = self.blocks(x, task_global, task_spatial, available, routing)

        return x


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
