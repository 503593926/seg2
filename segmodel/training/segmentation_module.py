"""Lightning 训练入口：前向、损失组合、日志、EMA 与生命周期接线。"""

from copy import deepcopy
from pathlib import Path
from typing import cast

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import OptimizerLRSchedulerConfig
from torchmetrics.classification import BinaryF1Score, BinaryJaccardIndex

from segmodel.inference.memory import release_inference_cache
from segmodel.training.diagnostics import state_diagnostics
from segmodel.training.losses import (
    image_losses,
    gene_prediction_losses,
    state_balance_loss,
    template_entropy_loss,
    template_orthogonality_loss,
)
from segmodel.training.optim import configure_optimizer
from segmodel.training.validation import SlideValidation


class SegmentationModule(pl.LightningModule):
    def __init__(
        self,
        model,
        image_cfg,
        validation_roots=None,
        tiling=None,
        mpp=1.0,
        visualization_dir=None,
        label_scale=1,
        spot_radius_px=50.0,
    ):
        super().__init__()
        self.model = model
        self.ema_model = deepcopy(model).requires_grad_(False)
        self.ema_model.eval()
        self._last_ema_step = 0
        self.image_cfg = image_cfg
        for name in ("state_balance", "template_orthogonal"):
            if float(image_cfg.get("loss_weights", {}).get(name, 0)) < 0:
                raise ValueError(f"loss_weights.{name} must be nonnegative")
        self.state_balance_alpha = float(image_cfg.state_balance_alpha)
        if not 0 < self.state_balance_alpha <= 1:
            raise ValueError("state_balance_alpha must be in (0, 1]")
        self.expression_transform = str(image_cfg.expression_transform)
        self.validation = SlideValidation(
            roots=validation_roots,
            tiling=tiling,
            directory=Path(visualization_dir) if visualization_dir is not None else None,
            mpp=mpp,
            label_scale=int(label_scale),
            spot_radius_px=float(spot_radius_px),
            expression_transform=self.expression_transform,
        )

        self.train_dice = BinaryF1Score(threshold=0.5)
        self.train_iou = BinaryJaccardIndex(threshold=0.5)

    def forward(self, image, patch, cls, pfm_mask=None, *,
                state_usage_mask=None, state_usage_grad=True):
        return self.model(image, patch, cls, pfm_mask,
                          state_usage_mask=state_usage_mask, state_usage_grad=state_usage_grad)

    def _update_expression_loss(self, losses, output, batch):
        """全部有效训练 spot 监督表达。"""
        weights = self.image_cfg.loss_weights
        if any(float(weights.get(name, 0)) for name in ("gene_prediction", "gene_correlation")):
            expression_losses, count = gene_prediction_losses(
                output, batch, expression_transform=self.expression_transform,
                weights=weights,
                correlation_min_spots=int(self.image_cfg.gene_correlation_min_spots),
                correlation_std_floor=float(self.image_cfg.gene_correlation_std_floor),
            )
            losses.update(expression_losses)
            self.log("train/gene_spots", float(count), on_step=True, on_epoch=True,
                     prog_bar=True, batch_size=len(batch["image"]))

    def _update_template_losses(self, losses, output):
        """模板几何正则：正交项去冗余，熵项按权重符号调稀疏/弥散。"""
        weights = self.image_cfg.loss_weights
        orthogonal = float(weights.get("template_orthogonal", 0))
        entropy = float(weights.get("template_entropy", 0))
        if not orthogonal and not entropy:
            return
        templates = output["expression_templates"]
        if orthogonal:
            losses["template_orthogonal"] = template_orthogonality_loss(templates)
        if entropy:
            losses["template_entropy"] = template_entropy_loss(templates)

    def training_step(self, batch) -> torch.Tensor:
        weights = self.image_cfg.loss_weights
        balance_weight = float(weights.get("state_balance", 0))
        output = self(
            batch["image"], batch["patch"], batch["cls"], batch["pfm_mask"],
            state_usage_mask=batch["tissue"],
            state_usage_grad=bool(balance_weight),
        )
        losses = image_losses(output, batch, weights)
        self._update_expression_loss(losses, output, batch)
        self._update_template_losses(losses, output)

        losses["state_balance"] = state_balance_loss(
            output["tissue_state_log_usage"], output["tissue_state_pixels"],
            alpha=self.state_balance_alpha,
        )

        loss = cast(
            torch.Tensor,
            sum(
                (float(weight) * losses.get(name, 0.0)
                 for name, weight in weights.items() if float(weight) != 0),
                output["logits"].sum() * 0,
            ),
        )

        self._log_training(loss, losses, output, batch)
        return loss

    def _log_training(self, loss, losses, output, batch):
        logged_losses = {f"train/{name}": value for name, value in losses.items()}
        logged_losses.update(
            {f"train/{name}": value for name, value in state_diagnostics(output, batch).items()}
        )
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

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        return configure_optimizer(self.model, self.image_cfg, self._trainer)

    def on_validation_epoch_start(self):
        self.validation.reset()
        release_inference_cache(self.device, f"validation epoch {self.current_epoch + 1} begin")

    def validation_step(self, slide, batch_index):
        if not self.validation.enabled:
            return None
        data = self.trainer.datamodule
        metrics = self.validation.evaluate(
            self.ema_model, slide, data.val_gene_adata,
            device=self.device, epoch=self.current_epoch + 1,
            save_dense=int(self.image_cfg.gene_visualization_top_n) > 0 and self.trainer.is_global_zero,
        )
        self.log_dict(
            {f"val/{name}": float(value) for name, value in metrics.items()
             if name in {"loss", "dice", "iou"}
             or (name.startswith("gene_") and len(data.test_names) <= 1)},
            on_step=False, on_epoch=True, prog_bar=True, batch_size=1,
        )
        return metrics

    def on_validation_epoch_end(self):
        try:
            data = self.trainer.datamodule
            summary = self.validation.summarize(
                data.val_gene_adata, data.val_slides, epoch=self.current_epoch + 1,
                top_n=int(self.image_cfg.gene_visualization_top_n),
                save_images=self.trainer.is_global_zero,
            )
            if not summary:
                return
            self.log("val/gene_prediction_pcc_macro", summary["macro"]["gene_prediction_pcc"],
                     prog_bar=True, batch_size=1)
            if len(data.test_names) > 1:
                self.log_dict({f"val/{key}": value for key, value in summary["micro"].items()},
                              prog_bar=True, batch_size=1)
                for name, metrics in summary["slides"].items():
                    self.log_dict({f"val/{name}/{key}": value for key, value in metrics.items()}, batch_size=1)
        finally:
            release_inference_cache(self.device, f"validation epoch {self.current_epoch + 1} end")

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
        for name, metric in (("dice", self.train_dice), ("iou", self.train_iou)):
            self.log(f"train/{name}", metric, on_step=True, on_epoch=True, prog_bar=True)
