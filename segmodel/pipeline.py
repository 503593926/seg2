"""一个 fold/seed 的完整流程：VAE 预训练 → 分割训练 → 保存模型 → 整片测试。

按阶段拆成函数，中间量（几何、模块）用参数与返回值逐级传递，不用编排类持有状态。
"""

import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning.loggers import CSVLogger

from .dataset import FoldDataModule
from .dynamic import DynamicPropagationCallback
from .inference.metrics import summarize_metrics
from .inference.validation import validate_slide
from .models import Distiller, GeneVAE, SegmentationModel
from .training.callbacks import DatasetEpochCallback, MetricsBelowRichProgressBar
from .training.gene_module import GeneVAEModule
from .training.segmentation_module import SegmentationModule


def run_fold(train_names, test_names, seed, cfg, output_dir):
    """跑完一个 fold/seed，返回整片测试的汇总指标。"""
    output_dir = _prepare_output(cfg, output_dir, seed, train_names, test_names)

    pl.seed_everything(seed, workers=True)

    data_module = FoldDataModule(train_names, test_names, seed, cfg)
    data_module.setup("fit")

    tiling = _resolve_tiling(cfg, data_module)

    gene_vae = _pretrain_gene_vae(cfg, data_module, output_dir)

    pl.seed_everything(seed, workers=True)
    module = _train_segmentation(cfg, data_module, gene_vae, tiling, output_dir)
    data_module.teardown("fit")

    _save_final_model(cfg, module.model, tiling, output_dir)
    return _test_final_model(cfg, data_module, module.model, module.device, tiling, output_dir)


def _resolve_tiling(cfg, data_module):
    """滑窗参数：crop_size 训练与推理共用，stride 取一半，起点须与网格对齐（alignment）。"""
    crop_size = int(cfg.experiment.dataset.crop_size)
    stride = crop_size // 2
    alignment = data_module.alignment
    if crop_size % alignment or stride % alignment:
        raise ValueError(
            f"crop_size={crop_size} and stride={stride} must be divisible by "
            f"MPP-derived alignment={alignment}"
        )
    return {
        "crop_size": crop_size,
        "stride": stride,
        "alignment": alignment,
        "batch_size": int(cfg.experiment.dataset.inference_batch_size),
    }


def _prepare_output(cfg, output_dir, seed, train_names, test_names):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "vae").mkdir(exist_ok=True)
    (output_dir / "segmentation").mkdir(exist_ok=True)
    OmegaConf.save(cfg, output_dir / "config.yaml", resolve=True)
    _write_json(
        output_dir / "split.json",
        {"seed": int(seed), "train": list(train_names), "test": list(test_names)},
    )
    return output_dir


def _csv_logger(output_dir, stage):
    return CSVLogger(save_dir=output_dir, name=stage, version="")


def _pretrain_gene_vae(cfg, data_module, output_dir):
    runtime_cfg = cfg.experiment.training.runtime
    gene_cfg = cfg.experiment.model.gene
    gene_vae = GeneVAE(
        input_dim=data_module.train_gene_adata.n_vars,
        latent_dim=int(gene_cfg.latent_dim),
        hidden_dims=tuple(gene_cfg.hidden_dims),
        dropout=float(gene_cfg.dropout),
    )
    module = GeneVAEModule(gene_vae, cfg.experiment.training.vae)
    trainer = pl.Trainer(
        accelerator=str(runtime_cfg.accelerator),
        devices=runtime_cfg.devices,
        precision=runtime_cfg.precision,
        max_epochs=int(cfg.experiment.training.vae.pretrain_epochs),
        default_root_dir=output_dir / "vae",
        callbacks=[
            MetricsBelowRichProgressBar(refresh_rate=100, leave=True),
        ],
        enable_progress_bar=True,
        logger=_csv_logger(output_dir, "vae"),
        enable_checkpointing=False,
    )
    trainer.fit(
        module,
        train_dataloaders=data_module.vae_dataloader(),
        val_dataloaders=data_module.vae_val_dataloader(),
    )
    return gene_vae


def _train_segmentation(cfg, data_module, gene_vae, tiling, output_dir):
    label_scale = data_module.label_scale
    model_cfg = cfg.experiment.model.image
    train_cfg = cfg.experiment.training.image
    slide_data = next(iter(data_module.train_slide_data.values()))
    model = SegmentationModel(
        pfm_patch_dims=slide_data["patch_dims"],
        pfm_cls_dims=slide_data["cls_dims"],
        patch_scale=data_module.patch_scale,
        label_scale=label_scale,
        propagation_channels=int(model_cfg.propagation_channels),
        feature_dim=int(model_cfg.decomposition.feature_dim),
        spatial_pool=int(model_cfg.decomposition.spatial_pool),
        router_hidden_dims=tuple(model_cfg.router.hidden_dims),
        top_k=int(model_cfg.router.top_k),
    )
    distiller = Distiller(
        model,
        gene_vae,
        train_cfg.distillation,
        cfg.experiment.training.vae,
        float(cfg.experiment.dataset.spot_radius),
        label_scale=label_scale,
    )
    module = SegmentationModule(
        model,
        distiller,
        train_cfg,
        validation_roots=data_module.data_roots,
        tiling=tiling,
        mpp=float(cfg.resolution.target_mpp),
        visualization_dir=output_dir / "validation",
        label_scale=label_scale,
    )
    callbacks: list[pl.Callback] = [
        DatasetEpochCallback(),
        MetricsBelowRichProgressBar(refresh_rate=100, leave=True),
    ]
    if cfg.experiment.dynamic.enabled:
        callbacks.append(
            DynamicPropagationCallback(
                cfg.experiment.dynamic,
                tiling,
                output_dir / "dynamic_pseudo_labels",
            )
        )
    runtime_cfg = cfg.experiment.training.runtime
    trainer = pl.Trainer(
        accelerator=str(runtime_cfg.accelerator),
        devices=runtime_cfg.devices,
        precision=runtime_cfg.precision,
        max_epochs=int(train_cfg.epochs),
        check_val_every_n_epoch=int(train_cfg.val_every_n_epochs),
        default_root_dir=output_dir / "segmentation",
        callbacks=callbacks,
        logger=_csv_logger(output_dir, "segmentation"),
        enable_checkpointing=False,
        num_sanity_val_steps=0,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
    )
    trainer.fit(module, datamodule=data_module)
    return module


def _save_final_model(cfg, model, tiling, output_dir):
    """保存权重与重建所需信息，让 checkpoint 自描述（供外部单跑推理）。"""
    model_config = OmegaConf.to_container(cfg.experiment.model.image, resolve=True)
    model_config["patch_scale"] = model.patch_scale
    model_config["label_scale"] = model.label_scale
    model_config["pfm_patch_dims"] = list(model.pfm_patch_dims)
    model_config["pfm_cls_dims"] = list(model.pfm_cls_dims)
    artifact = {
        "state_dict": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
        "model_config": model_config,
        "inference_config": {**tiling, "threshold": 0.5},
        "image_feature_map": list(cfg.experiment.dataset.image_feature_maps),
    }
    torch.save(artifact, output_dir / "final_model.pt")


def _test_final_model(cfg, data_module, model, device, tiling, output_dir):
    """整片测试：逐张预测、落盘概率图与指标，返回汇总指标。"""
    data_module.setup("test")
    slide_metrics = {}
    try:
        for slide in data_module.test_dataloader():
            probability, metrics = validate_slide(
                model,
                slide,
                data_module.data_roots[slide.name],
                device,
                **tiling,
                mpp=float(cfg.resolution.target_mpp),
            )
            slide_dir = output_dir / "test" / slide.name
            slide_dir.mkdir(parents=True, exist_ok=True)
            probability = probability.astype(np.float32)
            mask = probability >= 0.5
            np.save(slide_dir / "probability.npy", probability)
            cv2.imwrite(str(slide_dir / "mask.png"), mask.astype(np.uint8) * 255)
            _write_json(slide_dir / "metrics.json", metrics)
            slide_metrics[slide.name] = metrics
    finally:
        data_module.teardown("test")

    summary = summarize_metrics(slide_metrics)
    _write_json(output_dir / "metrics.json", summary)
    return summary


def _write_json(path, value):
    Path(path).write_text(
        json.dumps(
            _json_value(value),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ),
        encoding="utf-8",
    )


def _json_value(value):
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
