from math import ceil

import torch
from pytorch_lightning.utilities.types import OptimizerLRSchedulerConfig


def configure_optimizer(model, cfg, trainer) -> OptimizerLRSchedulerConfig:
    decay, no_decay = ([], [])
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": float(cfg.weight_decay)},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=float(cfg.lr),
    )
    total_steps = int(trainer.estimated_stepping_batches)
    warmup_ratio = float(cfg.warmup_steps)
    warmup_steps = ceil(total_steps * warmup_ratio)
    if warmup_steps:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps
        )
        if warmup_steps >= total_steps:
            scheduler = warmup
        else:
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[
                    warmup,
                    torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=total_steps - warmup_steps, eta_min=0
                    ),
                ],
                milestones=[warmup_steps],
            )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=0)
    return {
        "optimizer": optimizer,
        "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
    }
