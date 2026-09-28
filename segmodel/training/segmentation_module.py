from copy import deepcopy
from pathlib import Path

import pytorch_lightning as pl
import torch
from torchmetrics.classification import BinaryF1Score, BinaryJaccardIndex

from ..inference.validation import load_expert_masks, validate_slide
from ..inference.visualization import save_prediction_result
from .losses import image_losses, pfm_losses


class SegmentationModule(pl.LightningModule):
    """分割训练：只负责图像侧的损失、指标与整片验证，基因分支交给 distiller。"""

    def __init__(
        self,
        model,
        distiller,
        image_cfg,
        validation_roots=None,
        tiling=None,
        mpp=1.0,
        visualization_dir=None,
        label_scale=1,
    ):
        super().__init__()
        self.model = model
        self.ema_model = deepcopy(model).requires_grad_(False)
        self.ema_model.eval()
        self._last_ema_step = 0
        self.distiller = distiller
        self.image_cfg = image_cfg
        self.validation_roots = validation_roots
        self.tiling = tiling
        self.mpp = mpp
        self.visualization_dir = (
            Path(visualization_dir) if visualization_dir is not None else None
        )
        self.label_scale = int(label_scale)

        self.train_dice = BinaryF1Score(threshold=0.5)
        self.train_iou = BinaryJaccardIndex(threshold=0.5)

        # 参数漂移诊断：记录图像模型与只被蒸馏训练的 image_projector 各自相对初始值变了多少
        self._drift_reference = {
            "model": [
                parameter.detach().float().cpu().clone()
                for parameter in self.model.parameters()
            ],
            "projector": [
                parameter.detach().float().cpu().clone()
                for parameter in self.distiller.image_parameters
            ],
        }

    def forward(self, image, patch, cls, pfm_mask=None):
        return self.model(image, patch, cls, pfm_mask)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.distiller.ema_update(self.model, self.global_step)
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

    def on_train_epoch_end(self):
        """记录两组参数相对初始值的 L2 漂移：model 由分割+蒸馏共同驱动，projector 只由蒸馏驱动。"""
        drift = {
            name: self._relative_drift(name, parameters)
            for name, parameters in (
                ("model", self.model.parameters()),
                ("projector", self.distiller.image_parameters),
            )
        }
        self.log_dict(
            {
                "train/drift/model": drift["model"],
                "train/drift/projector": drift["projector"],
                "train/drift/projector_over_model": drift["projector"]
                / max(drift["model"], 1e-12),
            },
            on_step=False,
            on_epoch=True,
        )

    def _relative_drift(self, name, parameters):
        reference = self._drift_reference[name]
        delta = base = 0.0
        for parameter, initial in zip(parameters, reference):
            delta += float((parameter.detach().float().cpu() - initial).square().sum())
            base += float(initial.square().sum())
        return float(delta**0.5) / max(float(base**0.5), 1e-12)

    def _update_segmentation_metrics(self, output, batch):
        valid = batch["valid"].bool()
        if valid.any():
            probability = output["logits"].detach().sigmoid()[valid]
            target = (batch["cancer_probability"] >= 0.5)[valid]
            self.train_dice.update(probability, target)
            self.train_iou.update(probability, target)
        self.log(
            "train/dice",
            self.train_dice,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "train/iou",
            self.train_iou,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def training_step(self, batch, batch_index):
        output = self(batch["image"], batch["patch"], batch["cls"], batch["pfm_mask"])
        losses = image_losses(output, batch)
        losses.update(pfm_losses(output, batch, self.model.pfm_count))
        gene_losses, diagnostics = self.distiller.losses(
            output, batch, self.current_epoch
        )
        losses.update(gene_losses)

        loss = sum(
            float(self.image_cfg.loss_weights[name]) * value
            for name, value in losses.items()
        )
        self.log_dict(
            {
                **{f"train/{name}": value for name, value in losses.items()},
                **{f"train/{name}": value for name, value in diagnostics.items()},
            },
            on_step=False,
            on_epoch=True,
            batch_size=len(batch["image"]),
        )
        self._update_segmentation_metrics(output, batch)
        self.log(
            "train/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(batch["image"]),
        )
        return loss

    def validation_step(self, slide, batch_index):
        if self.validation_roots is None or self.tiling is None:
            return None
        probability, metrics = validate_slide(
            self.ema_model,
            slide,
            self.validation_roots[slide.name],
            self.device,
            **self.tiling,
            mpp=self.mpp,
        )
        if self.visualization_dir is not None:
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
                for name in ("loss", "dice", "iou", "hd95")
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=1,
        )
        return metrics

    def configure_optimizers(self):
        weight_decay = float(self.image_cfg.weight_decay)
        image_parameters = list(self.model.named_parameters()) + list(
            self.distiller.image_projector.named_parameters()
        )
        parameter_groups = []
        for named_parameters, lr in (
            (image_parameters, float(self.image_cfg.lr)),
            (self.distiller.gene_vae.named_parameters(), float(self.image_cfg.gene_lr)),
        ):
            decay, no_decay = [], []
            for name, parameter in named_parameters:
                if not parameter.requires_grad:
                    continue
                if parameter.ndim <= 1 or name.endswith(".bias"):
                    no_decay.append(parameter)
                else:
                    decay.append(parameter)
            parameter_groups.extend(
                (
                    {"params": decay, "lr": lr, "weight_decay": weight_decay},
                    {"params": no_decay, "lr": lr, "weight_decay": 0.0},
                )
            )
        optimizer = torch.optim.AdamW(parameter_groups)

        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=0.1,
                    end_factor=1.0,
                    total_iters=self.image_cfg.warmup_epochs,
                ),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=self.image_cfg.epochs - self.image_cfg.warmup_epochs,
                    eta_min=1e-6,
                ),
            ],
            milestones=[self.image_cfg.warmup_epochs],
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
