import pytorch_lightning as pl
import torch

from .losses import get_gene_losses


class GeneVAEModule(pl.LightningModule):
    def __init__(self, model, cfg):
        super().__init__()
        self.model = model
        self.cfg = cfg

    def _shared_step(self, batch, stage):
        expression, _ = batch
        output = self.model(expression)
        losses = get_gene_losses(
            output["reconstruction"],
            expression,
            output["mean"],
            output["logvar"],
        )
        loss = (
            self.cfg.rec_loss * losses["reconstruction"]
            + self.cfg.kl_weight * losses["kl"]
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
