import pytorch_lightning as pl
import torch
from torchmetrics.classification import BinaryAUROC, BinaryF1Score

from .losses import get_gene_losses


class GeneVAEModule(pl.LightningModule):
    def __init__(self, model, cfg):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.train_auc = BinaryAUROC()
        self.train_f1 = BinaryF1Score(threshold=0.5)
        self.val_auc = BinaryAUROC()
        self.val_f1 = BinaryF1Score(threshold=0.5)

    def _shared_step(self, batch, stage):
        expression, spot_labels = batch
        output = self.model(expression)
        losses = get_gene_losses(
            output["reconstruction"],
            expression,
            output["mean"],
            output["logvar"],
            output["spot_logits"],
            spot_labels,
        )
        loss = (
            self.cfg.rec_loss * losses["reconstruction"]
            + self.cfg.kl_weight * losses["kl"]
            + self.cfg.task_weight * losses["task"]
        )
        prefix = "vae" if stage == "train" else "vae/val"
        self.log_dict(
            {f"{prefix}/{name}": value for name, value in losses.items()},
            on_step=False,
            on_epoch=True,
            batch_size=len(expression),
        )
        self.log(
            f"{prefix}/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(expression),
        )
        known = (spot_labels == 0) | (spot_labels == 1)
        if known.any():
            probability = output["spot_logits"][known].softmax(dim=1)[:, 1]
            labels = spot_labels[known]
            auc = self.train_auc if stage == "train" else self.val_auc
            f1 = self.train_f1 if stage == "train" else self.val_f1
            auc.update(probability, labels)
            f1.update(probability, labels)
            self.log(
                f"{prefix}/auc",
                auc,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
            self.log(
                f"{prefix}/f1",
                f1,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
            self.log_dict(
                {
                    f"{prefix}/positive": (labels == 1).sum().float(),
                    f"{prefix}/negative": (labels == 0).sum().float(),
                },
                on_step=False,
                on_epoch=True,
                reduce_fx="sum",
            )
        return loss

    def training_step(self, batch, batch_index):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_index):
        return self._shared_step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.cfg.pretrain_lr),
            weight_decay=float(self.cfg.weight_decay),
        )
