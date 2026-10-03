from copy import deepcopy
from math import lcm
from pathlib import Path

import pytorch_lightning as pl
import torch
from torchmetrics.classification import BinaryF1Score, BinaryJaccardIndex

from ..inference.validation import load_expert_masks, validate_slide
from ..inference.visualization import save_prediction_result
from .losses import image_losses, pfm_losses, segmentation_loss


class SegmentationModule(pl.LightningModule):
    """分割训练：只负责图像侧的损失、指标与整片验证，基因分支交给 distiller。"""

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

    def _mixup_segmentation_loss(self, batch):
        cfg = self.augment_cfg.get("mixup")
        if cfg is None or not bool(cfg.enabled) or self.current_epoch < int(cfg.start_epoch):
            return None
        if float(cfg.probability) <= 0 or torch.rand(()).item() >= float(cfg.probability):
            return None

        # 布尔 pfm_mask 无法表示插值后的可用程度，只选择所有 PFM 均可用的样本。
        indices = batch["pfm_mask"].all(dim=1).nonzero(as_tuple=True)[0]
        if indices.numel() < 2:
            return None
        indices = indices[
            torch.randperm(indices.numel(), device=indices.device)[: int(cfg.max_samples)]
        ]
        if indices.numel() < 2:
            return None
        partners = indices.roll(1)
        ratio = torch.distributions.Beta(float(cfg.alpha), float(cfg.alpha)).sample().item()

        def mix(value):
            return ratio * value[indices].float() + (1 - ratio) * value[partners].float()

        image = batch["image"]
        if image.dtype == torch.uint8:
            image = image.float().mul(2.0 / 255.0).sub(1.0)
        output = self(
            mix(image),
            tuple(mix(value) for value in batch["patch"]),
            tuple(mix(value) for value in batch["cls"]),
            batch["pfm_mask"][indices],
        )
        logits = output["logits"]
        return (
            ratio * segmentation_loss(
                logits, batch["cancer_probability"][indices], batch["valid"][indices]
            )
            + (1 - ratio) * segmentation_loss(
                logits, batch["cancer_probability"][partners], batch["valid"][partners]
            )
        )

    def _cutmix_batch(self, batch, cfg):
        image = batch["image"]
        target = batch["cancer_probability"][:, 0]
        valid = batch["valid"][:, 0]
        available = batch["pfm_mask"].all(dim=1)
        positive = (target >= float(cfg.positive_threshold)) & valid & available[:, None, None]
        negative = (target <= float(cfg.negative_threshold)) & valid & available[:, None, None]
        donors = positive.flatten(1).any(dim=1).nonzero(as_tuple=True)[0]
        receivers = negative.flatten(1).any(dim=1).nonzero(as_tuple=True)[0]
        if not donors.numel() or not receivers.numel():
            return None

        height, width = image.shape[-2:]
        grids = (batch["cancer_probability"], *batch["patch"], *batch["cls"])
        if any(height % value.shape[-2] or width % value.shape[-1] for value in grids):
            raise ValueError("CutMix requires image, label and PFM grids to divide the crop")
        unit_y = lcm(*(height // value.shape[-2] for value in grids))
        unit_x = lcm(*(width // value.shape[-1] for value in grids))
        if height % unit_y or width % unit_x:
            raise ValueError("CutMix grids cannot share aligned crop boundaries")
        cells_y, cells_x = height // unit_y, width // unit_x
        min_cells = int(cfg.min_cells)
        if cells_y < min_cells or cells_x < min_cells:
            return None

        receivers = receivers[
            torch.randperm(receivers.numel(), device=receivers.device)[: int(cfg.max_samples)]
        ]
        # 每个目标 crop 选一个不同的病灶来源；来源和目标都必须有完整 PFM。
        entries = []
        for receiver in receivers.tolist():
            choices = donors[donors != receiver]
            if not choices.numel():
                continue
            donor = int(choices[torch.randint(choices.numel(), (1,), device=choices.device)])
            donor_points = positive[donor].nonzero()
            receiver_points = negative[receiver].nonzero()
            source_y, source_x = donor_points[
                torch.randint(donor_points.shape[0], (1,), device=donor_points.device)
            ][0].tolist()
            dest_y, dest_x = receiver_points[
                torch.randint(receiver_points.shape[0], (1,), device=receiver_points.device)
            ][0].tolist()

            cells_h = int(torch.randint(min_cells, min(int(cfg.max_cells), cells_y) + 1, (1,)))
            cells_w = int(torch.randint(min_cells, min(int(cfg.max_cells), cells_x) + 1, (1,)))
            source_y = source_y * (height // target.shape[-2]) // unit_y
            source_x = source_x * (width // target.shape[-1]) // unit_x
            dest_y = dest_y * (height // target.shape[-2]) // unit_y
            dest_x = dest_x * (width // target.shape[-1]) // unit_x
            source_y = min(max(source_y - cells_h // 2, 0), cells_y - cells_h) * unit_y
            source_x = min(max(source_x - cells_w // 2, 0), cells_x - cells_w) * unit_x
            dest_y = min(max(dest_y - cells_h // 2, 0), cells_y - cells_h) * unit_y
            dest_x = min(max(dest_x - cells_w // 2, 0), cells_x - cells_w) * unit_x
            entries.append((receiver, donor, source_y, source_x, dest_y, dest_x,
                            cells_h * unit_y, cells_w * unit_x))

        if not entries:
            return None
        receiver_indices = [entry[0] for entry in entries]
        mixed = {
            key: batch[key][receiver_indices].clone()
            for key in ("image", "cancer_probability", "valid")
        }
        mixed["patch"] = tuple(value[receiver_indices].clone() for value in batch["patch"])
        mixed["cls"] = tuple(value[receiver_indices].clone() for value in batch["cls"])
        mixed["pfm_mask"] = batch["pfm_mask"][receiver_indices]

        def paste(destination, source, index, entry):
            _, donor, sy, sx, dy, dx, h, w = entry
            scale_y = height // source.shape[-2]
            scale_x = width // source.shape[-1]
            destination[index, :, dy // scale_y:(dy + h) // scale_y,
                        dx // scale_x:(dx + w) // scale_x] = source[
                donor, :, sy // scale_y:(sy + h) // scale_y,
                sx // scale_x:(sx + w) // scale_x
            ]

        for index, entry in enumerate(entries):
            for key in ("image", "cancer_probability", "valid"):
                paste(mixed[key], batch[key], index, entry)
            for key in ("patch", "cls"):
                for destination, source in zip(mixed[key], batch[key]):
                    paste(destination, source, index, entry)
        return mixed

    def _cutmix_segmentation_loss(self, batch):
        cfg = self.augment_cfg.get("cutmix")
        if cfg is None or not bool(cfg.enabled) or self.current_epoch < int(cfg.start_epoch):
            return None
        if float(cfg.probability) <= 0 or torch.rand(()).item() >= float(cfg.probability):
            return None
        mixed = self._cutmix_batch(batch, cfg)
        if mixed is None:
            return None
        output = self(mixed["image"], mixed["patch"], mixed["cls"], mixed["pfm_mask"])
        return segmentation_loss(output["logits"], mixed["cancer_probability"], mixed["valid"])

    def training_step(self, batch, batch_index):
        output = self(batch["image"], batch["patch"], batch["cls"], batch["pfm_mask"])
        losses = image_losses(output, batch)
        losses.update(pfm_losses(output, batch, self.model.pfm_count))
        if float(self.image_cfg.loss_weights.molecular) != 0:
            molecular_losses, diagnostics = self.distiller.losses(output, batch)
            losses.update(molecular_losses)
        else:
            losses["molecular"] = output["features"].sum() * 0
            diagnostics = {}

        loss = sum(
            float(weight) * losses[name]
            for name, weight in self.image_cfg.loss_weights.items()
        )
        mixup_loss = self._mixup_segmentation_loss(batch)
        self.log(
            "train/mixup_applied",
            float(mixup_loss is not None),
            on_step=False,
            on_epoch=True,
            batch_size=len(batch["image"]),
        )
        if mixup_loss is not None:
            loss = loss + float(self.augment_cfg.mixup.weight) * mixup_loss
            self.log(
                "train/mixup_segmentation",
                mixup_loss,
                on_step=False,
                on_epoch=True,
                batch_size=len(batch["image"]),
            )
        cutmix_loss = self._cutmix_segmentation_loss(batch)
        self.log(
            "train/cutmix_applied",
            float(cutmix_loss is not None),
            on_step=False,
            on_epoch=True,
            batch_size=len(batch["image"]),
        )
        if cutmix_loss is not None:
            loss = loss + float(self.augment_cfg.cutmix.weight) * cutmix_loss
            self.log(
                "train/cutmix_segmentation",
                cutmix_loss,
                on_step=False,
                on_epoch=True,
                batch_size=len(batch["image"]),
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
