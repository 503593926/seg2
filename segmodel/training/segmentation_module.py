from copy import deepcopy
from pathlib import Path
from typing import cast

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import OptimizerLRSchedulerConfig
from torchmetrics.classification import BinaryF1Score, BinaryJaccardIndex

from segmodel.inference.validation import load_expert_masks, validate_slide
from segmodel.inference.visualization import save_prediction_result
from segmodel.training.augment import cutmix_batch, mixup_batch
from segmodel.training.losses import image_losses, pfm_losses, segmentation_loss


class SegmentationModule(pl.LightningModule):
    def __init__(
        self,
        model,
        distiller,
        image_cfg,
        augment_cfg,
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
        self.augment_cfg = augment_cfg
        self.validation_roots = validation_roots
        self.tiling = tiling
        self.mpp = mpp
        self.visualization_dir = (
            Path(visualization_dir) if visualization_dir is not None else None
        )
        self.label_scale = int(label_scale)

        self.train_dice = BinaryF1Score(threshold=0.5)
        self.train_iou = BinaryJaccardIndex(threshold=0.5)

    def forward(self, image, patch, cls, pfm_mask=None):
        return self.model(image, patch, cls, pfm_mask)

    def training_step(self, batch) -> torch.Tensor:
        output = self(batch["image"], batch["patch"], batch["cls"], batch["pfm_mask"])
        losses = image_losses(output, batch)
        losses.update(pfm_losses(output, batch, self.model.pfm_count))

        diagnostics = None
        relation_enabled = float(self.image_cfg.loss_weights.get('relation', 0)) != 0
        if float(self.image_cfg.loss_weights.molecular) != 0 or relation_enabled:
            molecular_losses, diagnostics = self.distiller.losses(
                output, batch, relation_enabled=relation_enabled,
            )
            losses.update(molecular_losses)

        loss = cast(
            torch.Tensor,
            sum(
                float(weight) * losses.get(name, 0.0)
                for name, weight in self.image_cfg.loss_weights.items()
            ),
        )

        mixup_loss = self._mixup_segmentation_loss(batch)
        if mixup_loss is not None:
            loss = loss + float(self.augment_cfg.mixup.weight) * mixup_loss

        cutmix_loss = self._cutmix_segmentation_loss(batch)
        if cutmix_loss is not None:
            loss = loss + float(self.augment_cfg.cutmix.weight) * cutmix_loss

        logged_losses = {f"train/{name}": value for name, value in losses.items()}
        if diagnostics is not None:
            logged_losses.update(
                {f"train/{name}": value for name, value in diagnostics.items()}
            )
        self.log_dict(
            logged_losses,
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


    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        image_parameters = list(self.model.named_parameters()) + list(
            self.distiller.image_projector.named_parameters()
        )
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


    def validation_step(self, slide, batch_index):
        # 留出切片验证使用 EMA; 最终测试使用的模型由 pipeline 决定。
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
