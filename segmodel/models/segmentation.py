import torch
from torch import nn
from torch.nn import functional as F

from ..modules import (
    PFMDecomposer,
    Stem,
)
from ..modules.segmentation import ConvNeXtUnet

def _mask_pfm_features(mask, *feature_groups):
    """把被屏蔽 PFM 的 f1/f2 清零，返回与输入同结构的元组。

    必须清零而不是只压低路由 logits：Joint Router 的输入包含每个 PFM 的 f1，若这一路
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
    def __init__(
        self,
        pfm_patch_dims,
        pfm_cls_dims,
        patch_scale,
        label_scale,
        propagation_channels=4,
        dim=128,
        feature_dim=128,
        spatial_pool=4,
        router_hidden_dims=(128, 64),
        top_k=2,
        stem_chans=128,
        encoder_chans=(128, 160, 192,),
        decoder_chans=(160, 128),
        global_dim=256,
        spatial_dim=32,
        pfm_count=5,
    ):
        super().__init__()
        self.patch_scale = int(patch_scale)
        self.label_scale = int(label_scale)
        self.pfm_count = len(pfm_patch_dims)
        self.pfm_patch_dims = tuple(int(value) for value in pfm_patch_dims)
        self.pfm_cls_dims = tuple(int(value) for value in pfm_cls_dims)
        self.spatial_pool = int(spatial_pool)
        self.router_hidden_dims = tuple(int(value) for value in router_hidden_dims)
        self.top_k = int(top_k)

        self.stem_chans = stem_chans

        self.stem = Stem(
            in_chans=3,
            out_chans=stem_chans,
        )
        self.global_dim = global_dim
        self.spatial_dim = spatial_dim
        self.unet = ConvNeXtUnet(
            encoder_chans=encoder_chans,
            decoder_chans=decoder_chans,
            drop_path=(0.1, 0.2, 0.3),
            blocks=2,
            global_dim=global_dim,
            spatial_dim=spatial_dim,
            pfm_count=pfm_count,
            g_top_k=2,
            s_top_k=2,
        )


        self.feature_channels = int(decoder_chans[-1])
        self.propagation = nn.Sequential(
            nn.Conv2d(self.feature_channels, propagation_channels, 3, padding=1),
            nn.GELU(),
        )
        self.segmentation_head = nn.Conv2d(propagation_channels, 1, 1)
        self.affinity_head = nn.Conv2d(self.feature_channels, 2, 1)

        self.global_decomposer = PFMDecomposer(
            pfm_cls_dims, global_dim, spatial=False
        )
        self.spatial_decomposer = PFMDecomposer(
            pfm_patch_dims, spatial_dim, spatial=True
        )
        self.global_classifier = nn.Linear(global_dim, self.pfm_count)
        self.spatial_classifier = nn.Linear(spatial_dim, self.pfm_count)

    def forward(self, image, patch, cls, available=None):

        task_global, identity_global = self.global_decomposer(
            [value.float().mean(dim=(-2, -1)) for value in cls]
        )
        task_spatial, identity_spatial = self.spatial_decomposer(
            [value.float() for value in patch]
        )
        if available is not None:
            task_global, identity_global, task_spatial, identity_spatial = _mask_pfm_features(
                available, task_global, identity_global, task_spatial,identity_spatial
            )

        if image.dtype == torch.uint8:
            image = image.float().mul_(2.0 / 255.0).sub_(1.0)

        routing = []
        decoded_fm = self.unet(self.stem(image), task_global, task_spatial, available, routing)


        propagation = self.propagation(decoded_fm)
        return {
            "logits": self.segmentation_head(propagation),
            "affinity_logits": self.affinity_head(decoded_fm),
            "features": decoded_fm,
            "propagation_features": propagation,
            **self._decomposition(task_global, identity_global, task_spatial, identity_spatial),
        }


    def _decomposition(self, task_global, identity_global, task_spatial, identity_spatial):
        """导出解耦损失所需的两路统计量：f2 的分类 logits 与参与去相关的 f1/f2。

        空间维池化到 spatial_pool² 而不是整图均值：保留粗空间结构，统计量级
        与 global 侧可比（每个 (样本, PFM) 贡献 16 个向量）。
        """
        count = self.pfm_count
        pooled_task_spatial = [
            F.adaptive_avg_pool2d(value.float(), self.spatial_pool)
            .flatten(2)
            .transpose(1, 2)
            for value in task_spatial
        ]
        pooled_identity_spatial = [
            F.adaptive_avg_pool2d(value.float(), self.spatial_pool)
            .flatten(2)
            .transpose(1, 2)
            for value in identity_spatial
        ]
        return {
            "pfm_logits_global": torch.stack(
                [self.global_classifier(value.float()) for value in identity_global]
            ).reshape(-1, count),
            "pfm_logits_spatial": torch.stack(
                [
                    self.spatial_classifier(value.mean(dim=1))
                    for value in pooled_identity_spatial
                ]
            ).reshape(-1, count),
            "pfm_f1_global": torch.stack(task_global).reshape(-1, self.global_dim),
            "pfm_f2_global": torch.stack(identity_global).reshape(-1, self.global_dim),
            "pfm_f1_spatial": torch.stack(pooled_task_spatial).reshape(-1, self.spatial_dim),
            "pfm_f2_spatial": torch.stack(pooled_identity_spatial).reshape(-1, self.spatial_dim),
        }
