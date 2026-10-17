from copy import deepcopy
from math import ceil
from pathlib import Path
from typing import cast

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import OptimizerLRSchedulerConfig
from torchmetrics.classification import BinaryF1Score, BinaryJaccardIndex

from segmodel.inference.validation import load_expert_masks, validate_slide
from segmodel.inference.metrics import summarize_expression_metrics
from segmodel.inference.visualization import save_prediction_result, save_top_gene_predictions
from segmodel.training.augment import cutmix_batch, mixup_batch
from segmodel.training.losses import (
    image_losses,
    gene_prediction_losses,
    segmentation_loss,
)


class SegmentationModule(pl.LightningModule):
    def __init__(
        self,
        model,
        image_cfg,
        augment_cfg,
        validation_roots=None,
        tiling=None,
        mpp=1.0,
        visualization_dir=None,
        label_scale=1,
        spot_radius_px=50.0,
    ):
        super().__init__()
        self.model = model
        template_cfg = image_cfg.get("template_init", {})
        model.tissue_state_decoder.template_logits.requires_grad_(
            bool(template_cfg.get("trainable", True))
        )
        self.ema_model = deepcopy(model).requires_grad_(False)
        self.ema_model.eval()
        self._last_ema_step = 0
        self.image_cfg = image_cfg
        self.augment_cfg = augment_cfg
        self.validation_roots = validation_roots
        self.tiling = tiling
        self.mpp = mpp
        self.visualization_dir = (
            Path(visualization_dir) if visualization_dir is not None else None
        )
        self.label_scale = int(label_scale)
        self.spot_radius_px = float(spot_radius_px)
        self.expression_transform = str(image_cfg.expression_transform)
        self._validation_metrics = {}
        self._validation_expression = {}

        self.train_dice = BinaryF1Score(threshold=0.5)
        self.train_iou = BinaryJaccardIndex(threshold=0.5)

    def forward(self, image, patch, cls, pfm_mask=None, *, dense_expression=True):
        return self.model(image, patch, cls, pfm_mask, dense_expression=dense_expression)

    def _update_expression_loss(self, losses, output, batch):
        """全部有效训练 spot 监督表达。"""
        weights = self.image_cfg.loss_weights
        if any(float(weights.get(name, 0)) for name in ("gene_prediction", "gene_correlation")):
            expression_losses, count = gene_prediction_losses(
                output, batch, self.spot_radius_px, self.label_scale, self.expression_transform,
                weights=weights,
                correlation_min_spots=int(self.image_cfg.gene_correlation_min_spots),
            )
            losses.update(expression_losses)
            self.log("train/gene_spots", float(count), on_step=True, on_epoch=True,
                     prog_bar=True, batch_size=len(batch["image"]))

    def training_step(self, batch) -> torch.Tensor:
        output = self(
            batch["image"], batch["patch"], batch["cls"], batch["pfm_mask"],
            dense_expression=False,
        )
        weights = self.image_cfg.loss_weights
        losses = image_losses(output, batch, weights)
        self._update_expression_loss(losses, output, batch)

        loss = cast(
            torch.Tensor,
            sum(
                (float(weight) * losses.get(name, 0.0)
                 for name, weight in weights.items() if float(weight) != 0),
                output["logits"].sum() * 0,
            ),
        )

        mixup_loss = self._mixup_segmentation_loss(batch)
        if mixup_loss is not None:
            loss = loss + float(self.augment_cfg.mixup.weight) * mixup_loss

        cutmix_loss = self._cutmix_segmentation_loss(batch)
        if cutmix_loss is not None:
            loss = loss + float(self.augment_cfg.cutmix.weight) * cutmix_loss

        logged_losses = {f"train/{name}": value for name, value in losses.items()}
        if "tissue_states" in output:
            states = output["tissue_states"].detach()
            # 分块复用约 16 MiB 的 FP32 临时区，只统计日志，不修改状态图。
            chunk_rows = max(1, 4_194_304 // (states.shape[0] * states.shape[1] * states.shape[3]))
            entropy = states.new_zeros((), dtype=torch.float32)
            for chunk in states.split(chunk_rows, dim=2):
                terms = chunk.clamp_min(1e-8).log_().mul_(chunk)
                entropy.sub_(terms.sum())
                del terms
            usage = states.mean(dim=(0, 2, 3))
            logged_losses["train/tissue_state_entropy"] = entropy / (states.shape[0] * states.shape[2] * states.shape[3])
            logged_losses["train/tissue_state_usage_entropy"] = -(usage * usage.clamp_min(1e-8).log()).sum()
            logged_losses["train/expression_amplitude_mean"] = output["expression_amplitude"].detach().mean()
        expression_names = {"train/gene_prediction", "train/gene_correlation"}
        self.log_dict(
            {name: value for name, value in logged_losses.items() if name in expression_names},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(batch["image"]),
        )
        self.log_dict(
            {name: value for name, value in logged_losses.items() if name not in expression_names},
            on_step=False,
            on_epoch=True,
            batch_size=len(batch["image"]),
        )
        self._update_segmentation_metrics(output, batch)
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(batch["image"]),
        )
        return loss


    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        image_parameters = list(self.model.named_parameters())
        decay, no_decay = [], []
        for name, parameter in image_parameters:
            if not parameter.requires_grad:
                continue
            if parameter.ndim <= 1 or name.endswith(".bias"):
                no_decay.append(parameter)
            else:
                decay.append(parameter)
        optimizer = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": float(self.image_cfg.weight_decay)},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=float(self.image_cfg.lr),
        )

        # 独立检查优化器参数组时没有 Trainer，不推测训练的总步数。
        if self._trainer is None:
            return cast(OptimizerLRSchedulerConfig, {"optimizer": optimizer})

        # 使用实际优化器更新数，包含梯度累积、batch 限制和 max_steps。
        total_steps = int(self.trainer.estimated_stepping_batches)
        warmup_ratio = float(self.image_cfg.warmup_steps)
        if total_steps < 1 or not 0 <= warmup_ratio <= 1:
            raise ValueError("Training steps must be positive and warmup_steps must be a ratio in [0, 1]")
        warmup_steps = ceil(total_steps * warmup_ratio)

        if warmup_steps:
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps,
            )
            if warmup_steps >= total_steps:
                scheduler = warmup
            else:
                scheduler = torch.optim.lr_scheduler.SequentialLR(
                    optimizer,
                    schedulers=[warmup, torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=total_steps - warmup_steps, eta_min=0,
                    )],
                    milestones=[warmup_steps],
                )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_steps, eta_min=0,
            )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


    def on_validation_epoch_start(self):
        self._validation_metrics = {}
        self._validation_expression = {}

    def validation_step(self, slide, batch_index):
        # 留出切片验证使用 EMA; 最终测试使用的模型由 pipeline 决定。
        if self.validation_roots is None or self.tiling is None:
            return None
        gene_options = {}
        data_module = getattr(self.trainer, "datamodule")
        evaluate_segmentation = slide.has_segmentation
        multi_slide_evaluation = len(data_module.test_names) > 1
        if self.model.gene_count:
            gene_options = {
                "gene_adata": data_module.val_gene_adata,
                "spot_radius_px": self.spot_radius_px,
                "expression_transform": self.expression_transform,
                "expression_results": self._validation_expression,
                "save_dense_expression": (
                    self.visualization_dir is not None
                    and int(self.image_cfg.get("gene_visualization_top_n", 10)) > 0
                    and self.trainer.is_global_zero
                ),
            }
        probability, metrics = validate_slide(
            self.ema_model,
            slide,
            self.validation_roots[slide.name],
            self.device,
            **self.tiling,
            mpp=self.mpp,
            **gene_options,
            evaluate_segmentation=evaluate_segmentation,
        )
        self._validation_metrics[slide.name] = metrics
        if self.visualization_dir is not None and evaluate_segmentation:
            labels, valid = load_expert_masks(self.validation_roots[slide.name], slide)
            save_prediction_result(
                self.visualization_dir / f"epoch_{self.current_epoch + 1:04d}",
                slide.image,
                probability,
                labels,
                valid,
                self.label_scale,
            )
        self.log_dict(
            {
                f"val/{name}": float(metrics[name])
                for name in metrics
                if name in {"loss", "dice", "iou"}
                or (name.startswith("gene_") and not multi_slide_evaluation)
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=1,
        )
        return metrics

    def on_validation_epoch_end(self):
        data_module = getattr(self.trainer, "datamodule")
        summary = summarize_expression_metrics(self._validation_metrics, self._validation_expression)
        if not summary:
            return
        top_n = int(self.image_cfg.get("gene_visualization_top_n", 10))
        if self.visualization_dir is not None and top_n > 0 and self.trainer.is_global_zero:
            save_top_gene_predictions(
                self.visualization_dir / f"epoch_{self.current_epoch + 1:04d}" / "gene_expression",
                self._validation_expression, list(map(str, data_module.val_gene_adata.var_names)),
                data_module.val_slides, top_n=top_n, spot_radius_px=self.spot_radius_px,
                expression_transform=self.expression_transform,
            )
            for record in self._validation_expression.values():
                record.pop("dense_expression", None)
                record.pop("dense_tissue", None)
        self.log("val/gene_prediction_pcc_macro", summary["macro"]["gene_prediction_pcc"],
                 prog_bar=True, batch_size=1)
        if len(data_module.test_names) <= 1:
            return
        self.log_dict(
            {f"val/{key}": value for key, value in summary["micro"].items()},
            prog_bar=True, batch_size=1,
        )
        for name, metrics in summary["slides"].items():
            self.log_dict({f"val/{name}/{key}": value for key, value in metrics.items()}, batch_size=1)


    def on_train_batch_end(self, outputs, batch, batch_idx):
        self._update_ema_model()


    @torch.no_grad()
    def _update_ema_model(self):
        step = int(self.global_step)
        if step <= self._last_ema_step:
            return

        decay = min(float(self.image_cfg.ema_decay), 1.0 - 1.0 / step)
        for ema_param, param in zip(
            self.ema_model.parameters(), self.model.parameters()
        ):
            ema_param.lerp_(param.detach(), 1.0 - decay)
        for ema_buffer, buffer in zip(
            self.ema_model.buffers(), self.model.buffers()
        ):
            ema_buffer.copy_(buffer)

        self._last_ema_step = step
        self.ema_model.eval()


    def _update_segmentation_metrics(self, output, batch):
        valid = batch["valid"].bool()
        # forward 同时计算当前 batch 指标和累积 epoch 统计；保留 batch 维以支持空掩码。
        probability = output["logits"].detach().sigmoid()[valid].reshape(1, -1)
        target = (batch["cancer_probability"] >= 0.5)[valid].reshape(1, -1)
        self.train_dice(probability, target)
        self.train_iou(probability, target)
        self.log(
            "train/dice",
            self.train_dice,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "train/iou",
            self.train_iou,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )


    def _mixup_segmentation_loss(self, batch):
        if not bool(self.augment_cfg.enabled):
            return None
        cfg = self.augment_cfg.get("mixup")
        if cfg is None or not bool(cfg.enabled) or self.current_epoch < int(cfg.start_epoch):
            return None
        if float(cfg.probability) <= 0 or torch.rand(()).item() >= float(cfg.probability):
            return None

        result = mixup_batch(batch, cfg)
        if result is None:
            return None
        mixed, ratio, indices, partners = result
        output = self(mixed["image"], mixed["patch"], mixed["cls"], mixed["pfm_mask"])
        logits = output["logits"]
        return (
            ratio * segmentation_loss(
                logits, batch["cancer_probability"][indices], batch["valid"][indices]
            )
            + (1 - ratio) * segmentation_loss(
                logits, batch["cancer_probability"][partners], batch["valid"][partners]
            )
        )


    def _cutmix_segmentation_loss(self, batch):
        if not bool(self.augment_cfg.enabled):
            return None
        cfg = self.augment_cfg.get("cutmix")
        if cfg is None or not bool(cfg.enabled) or self.current_epoch < int(cfg.start_epoch):
            return None
        if float(cfg.probability) <= 0 or torch.rand(()).item() >= float(cfg.probability):
            return None
        mixed = cutmix_batch(batch, cfg)
        if mixed is None:
            return None
        output = self(mixed["image"], mixed["patch"], mixed["cls"], mixed["pfm_mask"])
        return segmentation_loss(output["logits"], mixed["cancer_probability"], mixed["valid"])
