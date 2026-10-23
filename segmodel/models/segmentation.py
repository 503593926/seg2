from numbers import Integral

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from segmodel.modules.backbone import ConvNeXtUnet, Stem, init_convnext_backbone
from segmodel.modules.conditioning import PFMProjector
from segmodel.modules.tissue_states import TissueStateDecoder

def _mask_pfm_features(mask, *feature_groups):
    """把被屏蔽 PFM 的条件特征清零，返回与输入同结构的元组。

    必须清零而不是只压低路由 logits：Joint Router 的输入包含每个 PFM 的特征，若这一路
    仍带有效数值，联合 MLP 可以让它影响其他 expert 的 logits——等于模型仍间接用到了被
    声明为不可用的信息。清零后它的加权贡献恒为 0，投影层在这一步也拿不到梯度。
    """
    groups = []
    for features in feature_groups:
        zeroed = []
        for index, feature in enumerate(features):
            keep = mask[:, index].reshape(-1, *([1] * (feature.dim() - 1)))
            zeroed.append(feature * keep.to(feature.dtype))
        groups.append(tuple(zeroed))
    return groups


class SegmentationModel(nn.Module):
    gene_names: list[str]  # 由 pipeline 按当前训练 fold 的基因顺序写入。
    model_config: dict  # 由 build_model 保存完整、已解析的构造参数。

    def __init__(
        self,
        pfm_patch_dims,
        pfm_cls_dims,
        gene_count,
        *,
        top_k,
        encoder_chans,
        decoder_chans,
        global_dim,
        spatial_dim,
        skip_drop_path,
        drop_path,
        backbone_init,
        tissue_state_count,
        tissue_state_dim,
        output_scale,
        gradient_checkpointing,
        downsample_factor,
        encoder_spatial,
        decoder_spatial,
    ):
        super().__init__()
        if backbone_init not in ("pytorch", "convnext"):
            raise ValueError(f"Unknown backbone_init: {backbone_init!r}; use 'pytorch' or 'convnext'")
        if not isinstance(tissue_state_count, Integral) or tissue_state_count < 2:
            raise ValueError("tissue_state_count must be an integer of at least 2")
        self.pfm_patch_dims = tuple(pfm_patch_dims)
        self.pfm_cls_dims = tuple(pfm_cls_dims)
        self.pfm_count = len(self.pfm_patch_dims)
        self.tissue_state_count = int(tissue_state_count)
        self.output_scale = int(output_scale)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.downsample_factor = int(downsample_factor)
        if self.output_scale not in (1, 2, 7) or self.output_scale != output_scale:
            raise ValueError("output_scale must be 1, 2 or 7 input pixels per prediction")

        self.stem = Stem(
            in_chans=3,
            out_chans=encoder_chans[0],
            down_factor=self.output_scale,
        )
        self.unet = ConvNeXtUnet(
            encoder_chans=encoder_chans,
            decoder_chans=decoder_chans,
            drop_path=drop_path,
            blocks=2,
            global_dim=global_dim,
            spatial_dim=spatial_dim,
            pfm_count=self.pfm_count,
            g_top_k=top_k,
            s_top_k=top_k,
            skip_drop_path=skip_drop_path,
            gradient_checkpointing=self.gradient_checkpointing,
            downsample_factor=self.downsample_factor,
            encoder_spatial=encoder_spatial,
            decoder_spatial=decoder_spatial,
        )

        self.feature_channels = int(decoder_chans[-1])
        self.global_projector = PFMProjector(
            pfm_cls_dims, global_dim, spatial=False
        )
        self.spatial_projector = PFMProjector(
            pfm_patch_dims, spatial_dim, spatial=True
        )

        if backbone_init == "convnext":
            with torch.random.fork_rng(devices=[]):
                init_convnext_backbone(self.stem)
                init_convnext_backbone(self.unet)

        self.gene_count = int(gene_count)
        with torch.random.fork_rng(devices=[]):
            self.tissue_state_decoder = TissueStateDecoder(
                self.feature_channels, self.gene_count, self.tissue_state_count, tissue_state_dim,
            )

    def forward(
        self,
        image,
        patch,
        cls,
        available=None,
        compute_expression=True,
        state_usage_mask=None,
        state_usage_grad=True
    ):
        # PFM 投影为 MoMu 条件特征，屏蔽不可用的输入。
        task_global = self.global_projector(
            [value.float().mean(dim=(-2, -1)) for value in cls]
        )
        task_spatial = (
            self.spatial_projector(
                patch if torch.is_autocast_enabled(image.device.type)
                else [value.float() for value in patch]
            )
            if self.unet.use_spatial else []
        )
        if available is not None:
            task_global, task_spatial = _mask_pfm_features(
                available, task_global, task_spatial
            )

        if image.dtype == torch.uint8:
            image = image.float().mul_(2.0 / 255.0).sub_(1.0)

        # Stem 在 1 mpp 模式只变换通道，MoMu U-Net 直接解码到原生预测网格。
        decoded_fm = self.unet(self.stem(image), task_global, task_spatial, available)

        # 分割、表达与整片动态传播共用 MoMu 特征。
        output = {"features": decoded_fm}
        # 状态头整体重算，避免保存多张全分辨率 FP32 中间状态图。
        decoder_args = (decoded_fm, compute_expression,
                        state_usage_mask, state_usage_grad)
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            states = checkpoint(self.tissue_state_decoder, *decoder_args, use_reentrant=False)
        else:
            states = self.tissue_state_decoder(*decoder_args)
        output.update(states)
        return output
