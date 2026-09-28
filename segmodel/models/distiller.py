"""训练期蒸馏：在线基因 VAE 微调、图像/基因 EMA 教师与双向蒸馏损失。"""

import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

from ..modules import ProjectionHead
from ..training.losses import distillation_loss, get_gene_losses


def _batch_spots(batch):
    """收集 batch 内所有完整 spot：crop 下标、坐标、表达、标签（无 spot 时后两者为空）。"""
    crop_indices = []
    coordinates = []
    expression = []
    labels = []
    for index, spots in enumerate(batch["spots"]):
        if not len(spots):
            continue
        crop_indices.append(index)
        coordinates.append(spots)
        expression.append(batch["gene_expression"][index])
        labels.append(batch["spot_labels"][index])
    if not crop_indices:
        return [], [], batch["gene_expression"][0], batch["spot_labels"][0]
    return crop_indices, coordinates, torch.cat(expression), torch.cat(labels)


def _binary_entropy_confidence(probability):
    probability = probability.float().clamp(1e-6, 1 - 1e-6)
    entropy = -(
        probability * probability.log() + (1 - probability) * (1 - probability).log()
    )
    return (1 - entropy / math.log(2)).clamp(0, 1)


def _geometric_confidence(*values):
    product = torch.ones_like(values[0])
    for value in values:
        product = product * value.clamp(1e-6, 1)
    return product.pow(1 / len(values))


class Distiller(nn.Module):
    def __init__(self, model, gene_vae, distillation_cfg, vae_cfg, spot_radius, label_scale=1):
        super().__init__()
        self.cfg = distillation_cfg
        self.vae_cfg = vae_cfg
        self.spot_radius = float(spot_radius)
        self.label_scale = int(label_scale)
        self.gene_vae = gene_vae

        self.image_projector = ProjectionHead(
            model.feature_channels, gene_vae.classifier.in_features
        )
        self.image_teacher = copy.deepcopy(model).requires_grad_(False)
        self.image_teacher_projector = copy.deepcopy(
            self.image_projector
        ).requires_grad_(False)
        self.gene_teacher = copy.deepcopy(gene_vae).requires_grad_(False)
        self.image_teacher.eval()
        self.image_teacher_projector.eval()
        self.gene_teacher.eval()

    @property
    def image_parameters(self):
        """跟随图像学习率的参数：投影头。"""
        return list(self.image_projector.parameters())

    @property
    def gene_parameters(self):
        """在线基因 VAE 的参数（encoder / decoder / projector / classifier 全部参与微调）。"""
        return [
            parameter
            for parameter in self.gene_vae.parameters()
            if parameter.requires_grad
        ]

    def train(self, mode=True):
        super().train(mode)
        for teacher in (
            self.image_teacher,
            self.image_teacher_projector,
            self.gene_teacher,
        ):
            teacher.eval()
        return self

    def ema_update(self, image_model, step):
        """更新三个教师；图像教师追的是外部传入的在线分割模型。"""
        base_decay = float(self.cfg.get("ema_decay", 0.99))
        decay = min(base_decay, 1 - 1 / (max(int(step), 1) + 1))
        self._ema_update(self.image_teacher, image_model, decay)
        self._ema_update(
            self.image_teacher_projector,
            self.image_projector,
            decay,
        )
        self._ema_update(self.gene_teacher, self.gene_vae, decay)

    @staticmethod
    @torch.no_grad()
    def _ema_update(teacher, online, decay):
        for teacher_parameter, online_parameter in zip(
            teacher.parameters(), online.parameters()
        ):
            teacher_parameter.mul_(decay).add_(
                online_parameter.detach(), alpha=1 - decay
            )
        for teacher_buffer, online_buffer in zip(teacher.buffers(), online.buffers()):
            teacher_buffer.copy_(online_buffer)

    def losses(self, output, batch, epoch):
        spot_batch = _batch_spots(batch)
        crop_indices, _, expression, labels = spot_batch

        if not expression.shape[0]:
            zero = next(self.gene_vae.parameters()).sum() * 0
            losses = {"molecular": zero, "task": zero, "vae": zero}
            diagnostics = {"reconstruction": zero, "kl": zero, "spots": 0.0}
            return losses, diagnostics

        gene_output = self.gene_vae(expression)
        gene_losses = get_gene_losses(
            gene_output["reconstruction"],
            expression,
            gene_output["mean"],
            gene_output["logvar"],
            gene_output["spot_logits"],
            labels,
        )

        with torch.no_grad():
            # 教师必须看到与在线分支相同的 PFM 可用性，否则蒸馏目标与学生的条件不一致
            image_teacher_output = self.image_teacher(
                batch["image"][crop_indices],
                [value[crop_indices] for value in batch["patch"]],
                [value[crop_indices] for value in batch["cls"]],
                batch["pfm_mask"][crop_indices],
            )

        bidirectional = epoch >= int(self.cfg.bidirectional_start_epoch)
        molecular, statistics = self._distillation_forward(
            output,
            image_teacher_output,
            spot_batch,
            gene_output["embedding"],
            bidirectional,
        )

        losses = {
            "molecular": molecular,
            "task": gene_losses["task"],
            "vae": (
                float(self.vae_cfg.rec_loss) * gene_losses["reconstruction"]
                + float(self.vae_cfg.kl_weight) * gene_losses["kl"]
            ),
        }

        diagnostics = {
            "reconstruction": gene_losses["reconstruction"],
            "kl": gene_losses["kl"],
            "spots": float(len(labels)),
            "distillation/bidirectional_active": bool(bidirectional),
            **{
                f"distillation/{name}": value
                for name, value in statistics.items()
            },
        }
        return losses, diagnostics

    def _distillation_forward(
        self,
        image_output,
        image_teacher_output,
        spot_batch,
        gene_embedding,
        bidirectional,
    ):
        """蒸馏的前向部分：池化 spot 特征、算两侧置信度与方向权重（损失本身在 losses.py）。"""
        crop_indices, coordinates, expression, _ = spot_batch
        image_embedding = torch.cat(
            [
                self._pool_spots(image_output["features"][index], spots)
                for index, spots in zip(crop_indices, coordinates)
            ]
        )
        image_embedding = self.image_projector(image_embedding)


        teacher_image_features = []
        teacher_image_probability = []
        teacher_image_variance = []
        online_image_probability = []
        teacher_probability_map = image_teacher_output["logits"].float().sigmoid()
        online_probability_map = image_output["logits"].detach().float().sigmoid()
        for position, (index, spots) in enumerate(zip(crop_indices, coordinates)):
            teacher_image_features.append(
                self._pool_spots(image_teacher_output["features"][position], spots)
            )
            teacher_mean, variance = self._pool_spot_mean_variance(
                teacher_probability_map[position], spots
            )
            online_mean, _ = self._pool_spot_mean_variance(
                online_probability_map[index], spots
            )
            teacher_image_probability.append(teacher_mean)
            teacher_image_variance.append(variance)
            online_image_probability.append(online_mean)

        with torch.no_grad():
            teacher_image_embedding = self.image_teacher_projector(
                torch.cat(teacher_image_features)
            )
            gene_confidence, teacher_gene_embedding = self._gene_confidence(expression)
            image_confidence = self._image_confidence(
                torch.cat(teacher_image_probability),
                torch.cat(teacher_image_variance),
                torch.cat(online_image_probability),
            )
            gene_to_image_weight, image_to_gene_weight = self._distillation_weights(
                gene_confidence,
                image_confidence,
                bidirectional,
            )
            coverage = ((gene_to_image_weight + image_to_gene_weight) > 0).float()

        # 双向蒸馏：图像嵌入对齐基因教师，基因嵌入对齐图像教师
        gene_to_image = distillation_loss(
            image_embedding, teacher_gene_embedding, gene_to_image_weight
        )
        image_to_gene = distillation_loss(
            gene_embedding,
            teacher_image_embedding,
            image_to_gene_weight,
        )
        statistics = {
            "gene_to_image": gene_to_image.detach(),
            "image_to_gene": image_to_gene.detach(),
            "gene_confidence": gene_confidence.mean(),
            "image_confidence": image_confidence.mean(),
            "gene_to_image_weight": gene_to_image_weight.mean(),
            "image_to_gene_weight": image_to_gene_weight.mean(),
            "distillation_coverage": coverage.mean(),
        }
        return gene_to_image + image_to_gene, statistics

    def _pool_spots(self, features, coordinates):
        """按 spot 圆盘对特征做均值池化；特征在 7mpp 网格，坐标与半径按 label_scale 换算。"""
        channels = features.shape[0]
        radius = self.spot_radius / self.label_scale
        pooled = []
        for center in coordinates:
            x, y = float(center[0]) / self.label_scale, float(center[1]) / self.label_scale
            x0, x1 = math.ceil(x - radius), math.floor(x + radius) + 1
            y0, y1 = math.ceil(y - radius), math.floor(y + radius) + 1
            yy = torch.arange(y0, y1, device=features.device)[:, None]
            xx = torch.arange(x0, x1, device=features.device)[None, :]
            disk = (xx - x).square() + (yy - y).square() <= radius**2
            values = features[:, y0:y1, x0:x1].float()[:, disk]
            pooled.append(values.mean(dim=1))
        if not pooled:
            return features.new_empty((0, channels))
        return torch.stack(pooled)

    def _pool_spot_mean_variance(self, values, coordinates):
        """单个 spot 圆盘内概率图的均值与方差；坐标与半径按 label_scale 换算。"""
        means = []
        variances = []
        radius = self.spot_radius / self.label_scale
        for center in coordinates:
            x, y = float(center[0]) / self.label_scale, float(center[1]) / self.label_scale
            x0, x1 = math.ceil(x - radius), math.floor(x + radius) + 1
            y0, y1 = math.ceil(y - radius), math.floor(y + radius) + 1
            yy = torch.arange(y0, y1, device=values.device)[:, None]
            xx = torch.arange(x0, x1, device=values.device)[None, :]
            disk = (xx - x).square() + (yy - y).square() <= radius**2
            spot_values = values[:, y0:y1, x0:x1].float()[:, disk]
            means.append(spot_values.mean())
            variances.append(spot_values.var(unbiased=False))
        if not means:
            empty = values.new_empty((0,), dtype=torch.float32)
            return empty, empty
        return torch.stack(means), torch.stack(variances)

    def _gene_confidence(self, expression):
        """基因教师在该 spot 上的可信度，以及供基因→图像方向做目标用的 embedding。"""
        mean, logvar = self.gene_teacher.encode(expression)
        reconstruction = self.gene_teacher.decoder(mean)

        posterior_samples = max(
            1, int(self.cfg.get("gene_posterior_samples", 4))
        )
        standard_deviation = (0.5 * logvar).exp()
        probabilities = []
        for _ in range(posterior_samples):
            sampled = mean + torch.randn_like(mean) * standard_deviation
            probabilities.append(
                self.gene_teacher.classifier(self.gene_teacher.projector(sampled))
                .softmax(dim=1)[:, 1]
            )
        probabilities = torch.stack(probabilities)

        entropy_confidence = _binary_entropy_confidence(probabilities.mean(dim=0))
        posterior_temperature = max(
            float(self.cfg.get("gene_posterior_temperature", 0.02)),
            1e-6,
        )
        posterior_confidence = torch.exp(
            -probabilities.var(dim=0, unbiased=False) / posterior_temperature
        )
        reconstruction_error = F.mse_loss(
            reconstruction.float(),
            expression.float(),
            reduction="none",
        ).mean(dim=1)
        # 用这批 spot 的方差当标尺：未解释方差达到整体方差时，置信度为 1/e
        reconstruction_scale = (
            expression.var(dim=0, unbiased=False).mean().clamp_min(1e-6)
        )
        reconstruction_confidence = torch.exp(
            -reconstruction_error / reconstruction_scale
        )
        return (
            _geometric_confidence(
                entropy_confidence,
                posterior_confidence,
                reconstruction_confidence,
            ),
            self.gene_teacher.projector(mean),
        )

    def _image_confidence(
        self,
        teacher_probability,
        teacher_variance,
        online_probability,
    ):
        """图像教师在该 spot 上的可信度：预测果断、与在线模型一致、圆盘内空间均匀。"""
        consistency_temperature = max(
            float(self.cfg.get("image_consistency_temperature", 0.1)),
            1e-6,
        )
        consistency_confidence = torch.exp(
            -(teacher_probability - online_probability).abs() / consistency_temperature
        )
        spatial_temperature = max(
            float(self.cfg.get("image_spatial_temperature", 0.05)),
            1e-6,
        )
        spatial_confidence = torch.exp(-teacher_variance / spatial_temperature)
        return _geometric_confidence(
            _binary_entropy_confidence(teacher_probability),
            consistency_confidence,
            spatial_confidence,
        )

    def _distillation_weights(
        self,
        gene_confidence,
        image_confidence,
        bidirectional,
    ):
        """按置信度决定两个方向的权重；单向时只保留基因→图像。"""
        confidence_threshold = float(self.cfg.get("confidence_threshold", 0.3))
        if not bidirectional:
            gene_to_image = gene_confidence * (
                gene_confidence >= confidence_threshold
            )
            return gene_to_image, torch.zeros_like(gene_to_image)

        confidence_gamma = float(self.cfg.get("confidence_gamma", 2.0))
        maximum_confidence = torch.maximum(gene_confidence, image_confidence)
        gene_score = gene_confidence.pow(confidence_gamma) * (
            gene_confidence >= confidence_threshold
        )
        image_score = image_confidence.pow(confidence_gamma) * (
            image_confidence >= confidence_threshold
        )
        score_sum = (gene_score + image_score).clamp_min(1e-6)
        return (
            maximum_confidence * gene_score / score_sum,
            maximum_confidence * image_score / score_sum,
        )
