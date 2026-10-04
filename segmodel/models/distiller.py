"""训练期分子特征正则：冻结无监督基因 VAE，单向约束图像 spot 特征。"""

import math

import torch
from torch import nn

from ..modules import ProjectionHead
from ..training.losses import distillation_loss


def interpolate_posterior(
    mean: torch.Tensor,
    logvar: torch.Tensor,
    neighbors: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.autocast(device_type=mean.device.type, enabled=False):
        weights = weights.float()
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        weights = weights[..., None]
        parent_mean = mean.float()[neighbors]
        parent_variance = logvar.float().exp()[neighbors]
        query_mean = (weights * parent_mean).sum(dim=1)
        # 高斯混合的对角二阶矩: 后验方差 + 父节点均值分歧, 用中心形式避免相减消差.
        query_variance = (
            weights * (parent_variance + (parent_mean - query_mean[:, None]).square())
        ).sum(dim=1)
        return query_mean, query_variance.clamp_min(1e-6)


class Distiller(nn.Module):
    def __init__(self, model, gene_vae, spot_radius_px, label_scale=1):
        super().__init__()
        self.spot_radius_px = float(spot_radius_px)
        self.label_scale = int(label_scale)
        self.gene_teacher = gene_vae.requires_grad_(False)
        self.gene_teacher.eval()
        self.image_projector = ProjectionHead(
            model.feature_channels, gene_vae.latent_dim
        )

    @property
    def image_parameters(self):
        """跟随图像学习率训练的投影头参数。"""
        return list(self.image_projector.parameters())

    def train(self, mode=True):
        super().train(mode)
        self.gene_teacher.eval()
        return self

    def losses(self, output, batch, interpolation_enabled: bool = False):
        interpolation = output['features'].float().sum() * 0
        interpolation_count = 0
        interpolation_std = interpolation.detach()
        interpolation_weight = interpolation.detach()
        if interpolation_enabled:
            interpolation, interpolation_count, interpolation_std, interpolation_weight = (
                self._interpolation_loss(output, batch)
            )
        image_features = []
        expression = []
        for index, spots in enumerate(batch["spots"]):
            if not len(spots):
                continue
            image_features.append(self._pool_spots(output["features"][index], spots))
            expression.append(batch["gene_expression"][index])

        if not expression:
            zero = output["features"].sum() * 0
            return {"molecular": zero, 'interpolation': interpolation}, {
                "spots": 0.0,
                "distillation/gene_to_image": zero.detach(),
                "distillation/distillation_coverage": 0.0,
                "distillation/gene_embedding_std": zero.detach(),
                "distillation/image_embedding_std": zero.detach(),
                'distillation/interpolation_spots': float(interpolation_count),
                'distillation/interpolation_std': interpolation_std.detach(),
                'distillation/interpolation_weight': interpolation_weight.detach(),
            }

        image_embedding = self.image_projector(torch.cat(image_features))
        with torch.no_grad():
            gene_embedding = self.gene_teacher.embedding(torch.cat(expression))

        molecular = distillation_loss(
            image_embedding.float(),
            gene_embedding.float(),
            torch.ones(gene_embedding.shape[0], device=gene_embedding.device),
        )
        diagnostics = {
            "spots": float(len(gene_embedding)),
            "distillation/gene_to_image": molecular.detach(),
            "distillation/distillation_coverage": 1.0,
            "distillation/gene_embedding_std": gene_embedding.float().std(
                dim=0, unbiased=False
            ).mean(),
            "distillation/image_embedding_std": image_embedding.detach().float().std(
                dim=0, unbiased=False
            ).mean(),
            'distillation/interpolation_spots': float(interpolation_count),
            'distillation/interpolation_std': interpolation_std.detach(),
            'distillation/interpolation_weight': interpolation_weight.detach(),
        }
        return {"molecular": molecular, 'interpolation': interpolation}, diagnostics

    def _interpolation_loss(self, output, batch) -> tuple[torch.Tensor, int, torch.Tensor, torch.Tensor]:
        image_features: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        variances: list[torch.Tensor] = []
        for index, coordinates in enumerate(batch['interpolation_spots']):
            if not len(coordinates):
                continue
            image_features.append(self._pool_spots(output['features'][index], coordinates))
            with torch.no_grad():
                mean, logvar = self.gene_teacher.encode(batch['interpolation_expression'][index])
                neighbors = batch['interpolation_neighbors'][index]
                weights = batch['interpolation_weights'][index]
                query_mean, query_variance = interpolate_posterior(mean, logvar, neighbors, weights)
                targets.append(query_mean)
                variances.append(query_variance)
        if not targets:
            zero = output['features'].float().sum() * 0
            return zero, 0, zero.detach(), zero.detach()
        image_embedding = self.image_projector(torch.cat(image_features)).float()
        gene_embedding = torch.cat(targets)
        variance = torch.cat(variances)
        # 固定教师给出方差代理, 不归一化权重和, 使整体不确定时监督也减弱.
        weights = 1.0 / (1.0 + variance.mean(dim=1))
        loss = distillation_loss(image_embedding, gene_embedding, weights)
        return loss, len(gene_embedding), variance.sqrt().mean(), weights.mean()

    def _pool_spots(self, features, coordinates):
        """按 spot 圆盘均值池化，坐标与半径按 label_scale 换算到特征网格。"""
        channels = features.shape[0]
        radius = self.spot_radius_px / self.label_scale
        pooled = []
        for center in coordinates:
            center_x = float(center[0]) / self.label_scale
            center_y = float(center[1]) / self.label_scale
            left = math.ceil(center_x - radius)
            right = math.floor(center_x + radius) + 1
            top = math.ceil(center_y - radius)
            bottom = math.floor(center_y + radius) + 1
            rows = torch.arange(top, bottom, device=features.device)[:, None]
            columns = torch.arange(left, right, device=features.device)[None, :]
            disk = (columns - center_x).square() + (rows - center_y).square() <= radius**2
            values = features[:, top:bottom, left:right].float()[:, disk]
            pooled.append(values.mean(dim=1))
        if not pooled:
            return features.new_empty((0, channels))
        return torch.stack(pooled)
