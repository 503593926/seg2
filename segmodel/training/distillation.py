"""训练期分子特征正则：冻结无监督基因 VAE，单向约束图像 spot 特征。"""

import math

import torch
from torch import nn

from segmodel.modules.blocks import ProjectionHead
from segmodel.training.losses import distillation_loss, relation_distillation_loss


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

    def train(self, mode=True):
        super().train(mode)
        self.gene_teacher.eval()
        return self

    def losses(self, output, batch, relation_enabled: bool = False):
        image_features = []
        expression = []
        slide_ids = []
        crop_ids: list[torch.Tensor] = []
        spot_ids = []
        for index, spots in enumerate(batch["spots"]):
            if not len(spots):
                continue
            image_features.append(self._pool_spots(output["features"][index], spots))
            expression.append(batch["gene_expression"][index])
            if relation_enabled:
                ids = batch['spot_ids'][index]
                spot_ids.append(ids)
                crop_ids.append(torch.full_like(ids, index))
                if torch.unique(ids).numel() >= 2:
                    slide_ids.append(batch['slide_id'][index])

        if not expression:
            zero = output["features"].sum() * 0
            return {"molecular": zero, 'relation': zero}, {
                "spots": 0.0,
                "distillation/gene_to_image": zero.detach(),
                "distillation/distillation_coverage": 0.0,
                "distillation/gene_embedding_std": zero.detach(),
                "distillation/image_embedding_std": zero.detach(),
                'distillation/relation_pairs': 0.0,
                'distillation/relation_crops': 0.0,
                'distillation/relation_slides': 0.0,
            }

        image_embedding = self.image_projector(torch.cat(image_features))
        with torch.no_grad():
            gene_embedding = self.gene_teacher.embedding(torch.cat(expression))

        molecular = distillation_loss(
            image_embedding.float(),
            gene_embedding.float(),
            torch.ones(gene_embedding.shape[0], device=gene_embedding.device),
        )

        relation = image_embedding.sum() * 0
        pair_count, crop_count, slide_count = 0, 0, 0
        if relation_enabled:
            relation, pair_count, crop_count = relation_distillation_loss(
                image_embedding,
                gene_embedding,
                torch.cat(crop_ids),
                torch.cat(spot_ids),
            )
            if slide_ids:
                slide_count = torch.unique(torch.stack(slide_ids)).numel()
        diagnostics = {
            "spots": float(len(gene_embedding)),
            "distillation/gene_to_image": molecular.detach(),
            "distillation/gene_embedding_std": gene_embedding.float().std(
                dim=0, unbiased=False
            ).mean(),
            "distillation/image_embedding_std": image_embedding.detach().float().std(
                dim=0, unbiased=False
            ).mean(),
            'distillation/relation_pairs': float(pair_count),
            'distillation/relation_crops': float(crop_count),
            'distillation/relation_slides': float(slide_count),
        }
        return {"molecular": molecular, 'relation': relation}, diagnostics

    def _pool_spots(self, features, coordinates):
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
        return torch.stack(pooled)
