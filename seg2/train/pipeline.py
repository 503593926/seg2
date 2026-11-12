import gc
from pathlib import Path
from typing import Any, cast

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning.loggers import CSVLogger

from seg2.models.model import JointModel
from seg2.train.callbacks import DatasetEpochCallback, MetricsBelowRichProgressBar
from seg2.train.data.datamodule import FoldDataModule
from seg2.train.module import SegmentationModule
from seg2.train.propagation.callback import DynamicPropagationCallback


def run_fold(fold, seed, cfg, output_dir, extra_callbacks=()):
    data_module = FoldDataModule(fold, seed, cfg)
    data_module.setup("fit")

    pl.seed_everything(seed, workers=True)
    module = _train_model(cfg, data_module, output_dir, extra_callbacks)
    data_module.teardown("fit")

    save_model(module.model, output_dir / "final_model.pt")
    save_model(module.ema_model, output_dir / "final_ema_model.pt")

    del module
    gc.collect()


def build_model(model_cfg, slide, gene_names):
    model_options = cast(
        dict[str, Any], OmegaConf.to_container(model_cfg, resolve=True)
    )
    model_options.update(
        pfm_patch_dims=[feature.shape[-1] for feature in slide["patch_maps"]],
        pfm_cls_dims=[feature.shape[-1] for feature in slide["cls_maps"]],
        gene_count=len(gene_names),
    )
    model = JointModel(**model_options)
    model.model_config = model_options
    model.gene_names = list(map(str, gene_names))
    return model


def _train_model(cfg, data_module, output_dir, extra_callbacks=()):
    slide = next(iter(data_module.train_slides.values()))
    train_cfg = cfg.training.image
    model = build_model(cfg.model.image, slide, data_module.train_gene_adata.var_names)
    module = SegmentationModule(model, train_cfg)
    callbacks: list[pl.Callback] = [
        DatasetEpochCallback(),
        MetricsBelowRichProgressBar(
            metrics_every_n_steps=int(train_cfg.progress_every_n_steps),
            refresh_rate=100,
            leave=True,
        ),
    ]
    dynamic_cfg = cfg.dynamic
    if bool(dynamic_cfg.enabled):
        callbacks.append(
            DynamicPropagationCallback(
                dynamic_cfg, cfg.prep.rw, output_dir / "dynamic_pseudo_labels"
            )
        )
    callbacks.extend(extra_callbacks)
    runtime_cfg = cfg.training.runtime
    trainer = pl.Trainer(
        accelerator=str(runtime_cfg.accelerator),
        devices=runtime_cfg.devices,
        precision=runtime_cfg.precision,
        accumulate_grad_batches=int(train_cfg.accumulate_grad_batches),
        max_epochs=int(train_cfg.epochs),
        limit_val_batches=0,
        default_root_dir=output_dir,
        callbacks=callbacks,
        logger=CSVLogger(save_dir=output_dir, name="", version=""),
        enable_checkpointing=False,
        num_sanity_val_steps=0,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
    )
    trainer.fit(module, datamodule=data_module)
    return module


def save_model(model, path):
    path = Path(path)
    torch.save(
        {
            "format_version": 3,
            "state_dict": {
                name: value.detach().cpu() for name, value in model.state_dict().items()
            },
            "model_config": model.model_config,
            "gene_names": list(model.gene_names),
        },
        path,
    )
