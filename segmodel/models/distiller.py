"""训练期 MIL 分子正则：逐位置解码表达，在 spot 内聚合后监督，基因 VAE 冻结。"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from ..modules import ProjectionHead
from ..training.losses import relation_distillation_loss


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

    def losses(self, output, batch, relation_enabled: bool = False):
        image_features = []
        predicted_expression = []
        expression = []
        slide_ids = []
        spot_ids = []
        for index, spots in enumerate(batch["spots"]):
            if not len(spots):
                continue
            instances = self._spot_instances(output["features"][index], spots)
            image_features.extend(values.mean(dim=0) for values in instances)
            # 只解码 spot 覆盖的位置，同一 crop 内合并前向，避免逐 spot 调用网络。
            local_embedding = self.image_projector(torch.cat(instances))
            # decoder 参数冻结，但保留对输入的梯度；先解码，再聚合表达。
            local_expression = self.gene_teacher.decoder(local_embedding).float()
            predicted_expression.extend(
                values.mean(dim=0)
                for values in local_expression.split([len(values) for values in instances])
            )
            expression.append(batch["gene_expression"][index])
            if relation_enabled:
                ids = batch['spot_ids'][index]
                spot_ids.append(ids)
                slide_ids.append(batch['slide_id'][index].expand_as(ids))

        if not expression:
            zero = output["features"].sum() * 0
            return {"molecular": zero, 'relation': zero}, {
                "spots": 0.0,
                "distillation/expression_mse": zero.detach(),
                "distillation/distillation_coverage": 0.0,
                "distillation/gene_embedding_std": zero.detach(),
                "distillation/image_embedding_std": zero.detach(),
                'distillation/relation_pairs': 0.0,
                'distillation/relation_slides': 0.0,
            }

        # 保留 relation 原来的定义：先池化图像特征，再投影到基因潜空间。
        image_embedding = self.image_projector(torch.stack(image_features))
        target_expression = torch.cat(expression).float()
        with torch.no_grad():
            gene_embedding = self.gene_teacher.embedding(target_expression)

        molecular = F.mse_loss(
            torch.stack(predicted_expression), target_expression,
        )
        relation = image_embedding.sum() * 0
        pair_count, slide_count = 0, 0
        if relation_enabled:
            relation, pair_count, slide_count = relation_distillation_loss(
                image_embedding,
                gene_embedding,
                torch.cat(slide_ids),
                torch.cat(spot_ids),
            )
        diagnostics = {
            "spots": float(len(gene_embedding)),
            "distillation/expression_mse": molecular.detach(),
            "distillation/distillation_coverage": 1.0,
            "distillation/gene_embedding_std": gene_embedding.float().std(
                dim=0, unbiased=False
            ).mean(),
            "distillation/image_embedding_std": image_embedding.detach().float().std(
                dim=0, unbiased=False
            ).mean(),
            'distillation/relation_pairs': float(pair_count),
            'distillation/relation_slides': float(slide_count),
        }
        return {"molecular": molecular, 'relation': relation}, diagnostics

    def _spot_instances(self, features, coordinates):
        """提取各 spot 圆盘内的 (位置数, 通道数) 特征，几何量换算到特征网格。"""
        radius = self.spot_radius_px / self.label_scale
        instances = []
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
            if not values.shape[1]:
                raise ValueError("Spot has no feature-grid positions inside its disk")
            instances.append(values.transpose(0, 1))
        return instances
