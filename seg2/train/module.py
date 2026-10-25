from copy import deepcopy

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import OptimizerLRSchedulerConfig
from torchmetrics.classification import BinaryF1Score, BinaryJaccardIndex

from seg2.train.diagnostics import state_diagnostics
from seg2.train.losses import (
    gene_prediction_losses,
    image_losses,
    state_balance_loss,
    template_entropy_loss,
    template_orthogonality_loss,
)
from seg2.train.optim import configure_optimizer


class SegmentationModule(pl.LightningModule):
    def __init__(self, model, image_cfg):
        super().__init__()
        self.model = model
        self.ema_model = deepcopy(model).requires_grad_(False)
        self.ema_model.eval()
        self._last_ema_step = 0
        self.image_cfg = image_cfg
        self.state_balance_alpha = float(image_cfg.state_balance_alpha)
        self.expression_transform = str(image_cfg.expression_transform)
        self.train_dice = BinaryF1Score(threshold=0.5)
        self.train_iou = BinaryJaccardIndex(threshold=0.5)

    def forward(
        self, image, patch, cls, *, state_usage_mask=None, state_usage_grad=True
    ):
        return self.model(
            image,
            patch,
            cls,
            state_usage_mask=state_usage_mask,
            state_usage_grad=state_usage_grad,
        )

    def _update_expression_loss(self, losses, output, batch):
        weights = self.image_cfg.loss_weights
        if any(
            float(weights[name]) for name in ("gene_prediction", "gene_correlation")
        ):
            expression_losses, count = gene_prediction_losses(
                output,
                batch,
                expression_transform=self.expression_transform,
                weights=weights,
                correlation_min_spots=int(self.image_cfg.gene_correlation_min_spots),
                correlation_std_floor=float(self.image_cfg.gene_correlation_std_floor),
            )
            losses.update(expression_losses)
            self.log(
                "train/gene_spots",
                float(count),
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                batch_size=len(batch["image"]),
            )

    def _update_template_losses(self, losses, output):
        weights = self.image_cfg.loss_weights
        orthogonal = float(weights["template_orthogonal"])
        entropy = float(weights["template_entropy"])
        if not orthogonal and (not entropy):
            return
        templates = output["expression_templates"]
        if orthogonal:
            losses["template_orthogonal"] = template_orthogonality_loss(templates)
        if entropy:
            losses["template_entropy"] = template_entropy_loss(templates)

    def training_step(self, batch) -> torch.Tensor:
        weights = self.image_cfg.loss_weights
        balance_weight = float(weights["state_balance"])
        output = self(
            batch["image"],
            batch["patch"],
            batch["cls"],
            state_usage_mask=batch["tissue"],
            state_usage_grad=bool(balance_weight),
        )
        losses = image_losses(output, batch, weights)
        self._update_expression_loss(losses, output, batch)
        self._update_template_losses(losses, output)
        losses["state_balance"] = state_balance_loss(
            output["tissue_state_log_usage"],
            output["tissue_state_pixels"],
            alpha=self.state_balance_alpha,
        )
        loss = sum(
            (
                float(weight) * losses[name]
                for name, weight in weights.items()
                if float(weight) != 0
            ),
            output["logits"].sum() * 0,
        )
        self._log_training(loss, losses, output, batch)
        return loss

    def _log_training(self, loss, losses, output, batch):
        logged_losses = {f"train/{name}": value for name, value in losses.items()}
        logged_losses.update(
            {
                f"train/{name}": value
                for name, value in state_diagnostics(output, batch).items()
            }
        )
        expression_names = {"train/gene_prediction", "train/gene_correlation"}
        self.log_dict(
            {
                name: value
                for name, value in logged_losses.items()
                if name in expression_names
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(batch["image"]),
        )
        self.log_dict(
            {
                name: value
                for name, value in logged_losses.items()
                if name not in expression_names
            },
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
        for ema_buffer, buffer in zip(self.ema_model.buffers(), self.model.buffers()):
            ema_buffer.copy_(buffer)
        self._last_ema_step = step
        self.ema_model.eval()

    def _update_segmentation_metrics(self, output, batch):
        valid = batch["valid"].bool()
        probability = output["logits"].detach().sigmoid()[valid].reshape(1, -1)
        target = (batch["cancer_probability"] >= 0.5)[valid].reshape(1, -1)
        self.train_dice(probability, target)
        self.train_iou(probability, target)
        for name, metric in (("dice", self.train_dice), ("iou", self.train_iou)):
            self.log(
                f"train/{name}", metric, on_step=True, on_epoch=True, prog_bar=True
            )
