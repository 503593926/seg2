"""一个 fold/seed 的完整流程：VAE 预训练 → 分割训练 → 保存模型 → 整片测试。

按阶段拆成函数，中间量（几何、模块）用参数与返回值逐级传递，不用编排类持有状态。
"""

import json
import math
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning.loggers import CSVLogger

from segmodel.dataset.datamodule import FoldDataModule
from segmodel.dataset.loading import load_gene_adata
from segmodel.dynamic.callback import (
    DynamicPropagationCallback,
    SoftLabelVisualizationCallback,
)
from segmodel.inference.metrics import summarize_metrics
from segmodel.inference.validation import validate_slide
from segmodel.models.gene_vae import GeneVAE
from segmodel.models.segmentation import SegmentationModel
from segmodel.modules.propagation import DifferentiablePropagation
from segmodel.training.callbacks import DatasetEpochCallback, MetricsBelowRichProgressBar
from segmodel.training.distillation import Distiller
from segmodel.training.gene_module import GeneVAEModule
from segmodel.training.propagation_supervision import PropagationSupervision
from segmodel.training.segmentation_module import SegmentationModule


def run_fold(train_names, test_names, seed, cfg, output_dir):
    """跑完一个 fold/seed，返回整片测试的汇总指标。"""
    # 准备当前 fold 的输出目录和配置。
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "vae").mkdir(exist_ok=True)
    (output_dir / "segmentation").mkdir(exist_ok=True)
    OmegaConf.save(cfg, output_dir / "config.yaml", resolve=True)
    _write_json(
        output_dir / "split.json",
        {"seed": int(seed), "train": list(train_names), "test": list(test_names)},
    )

    # 先加载训练与验证数据，再训练基因 VAE。
    pl.seed_everything(seed, workers=True)

    data_module = FoldDataModule(train_names, test_names, seed, cfg)
    data_module.setup("fit")

    tiling = _resolve_tiling(cfg, data_module)

    gene_vae = _pretrain_gene_vae(cfg, data_module, output_dir)

    # 分割阶段重新设种子，保持与 VAE 训练的随机状态独立。
    pl.seed_everything(seed, workers=True)
    module = _train_segmentation(cfg, data_module, gene_vae, tiling, output_dir)
    data_module.teardown("fit")

    # 保存最终图像模型，然后执行整片测试。
    _save_final_model(cfg, module.model, tiling, output_dir)
    return _test_final_model(cfg, data_module, module.model, module.device, tiling, output_dir)


def _resolve_tiling(cfg, data_module):
    """滑窗参数：crop_size 训练与推理共用，stride 取一半，起点须与网格对齐（alignment）。"""
    crop_size = int(cfg.experiment.dataset.crop_size)
    stride = crop_size // 2
    alignment = next(iter(data_module.train_slides.values())).alignment
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


def _pretrain_gene_vae(cfg, data_module, output_dir):
    runtime_cfg = cfg.experiment.training.runtime
    gene_cfg = cfg.experiment.model.gene
    loss_weights = cfg.experiment.training.image.loss_weights
    if (
        any(float(loss_weights.get(name, 0)) != 0 for name in ('molecular', 'relation'))
        and int(cfg.experiment.training.vae.pretrain_epochs) <= 0
    ):
        raise ValueError("Molecular regularization requires gene VAE pretraining")
    gene_vae = GeneVAE(
        input_dim=data_module.train_gene_adata.n_vars,
        latent_dim=int(gene_cfg.latent_dim),
        hidden_dims=tuple(gene_cfg.hidden_dims),
        dropout=float(gene_cfg.dropout),
    )
    if int(cfg.experiment.training.vae.pretrain_epochs) <= 0:
        return gene_vae
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
        logger=CSVLogger(save_dir=output_dir, name="vae", version=""),
        enable_checkpointing=False,
    )
    trainer.fit(
        module,
        train_dataloaders=data_module.vae_dataloader(),
        val_dataloaders=data_module.vae_val_dataloader(),
    )
    return gene_vae


def _train_segmentation(cfg, data_module, gene_vae, tiling, output_dir):
    slide = next(iter(data_module.train_slides.values()))
    label_scale = slide.label_scale
    model_cfg = cfg.experiment.model.image
    train_cfg = cfg.experiment.training.image
    model_options = cast(dict[str, Any], OmegaConf.to_container(model_cfg, resolve=True))
    supervision_cfg = cfg.experiment.get("propagation_supervision", {})
    if supervision_cfg.get("enabled", False) and supervision_cfg.get("gene_mode") == "prediction":
        model_options["gene_count"] = data_module.train_gene_adata.n_vars
    model = SegmentationModel(
        pfm_patch_dims=slide.patch_dims,
        pfm_cls_dims=slide.cls_dims,
        **model_options,
    )
    if model.gene_count:
        model.gene_names = list(map(str, data_module.train_gene_adata.var_names))
    distiller = Distiller(
        model,
        gene_vae,
        spot_radius_px=slide.spot_radius_px,
        label_scale=label_scale,
    )
    propagation = None
    if bool(cfg.experiment.propagation.enabled):
        propagation_losses = ("propagation", "propagation_query", "gene_propagation", "gene_auxiliary", "gene_prediction")
        if not any(float(train_cfg.loss_weights.get(name, 0)) for name in propagation_losses):
            raise ValueError(
                "experiment.propagation.enabled=true 但 "
                "所有传播相关 loss_weights 均为 0, 传播不会进入总损失"
            )
        propagation_cfg = cast(
            dict[str, Any],
            OmegaConf.to_container(cfg.experiment.propagation, resolve=True),
        )
        propagation_cfg.pop("enabled", None)
        propagation = DifferentiablePropagation(
            label_scale=label_scale,
            **propagation_cfg,
        )
    supervision = _build_propagation_supervision(cfg, data_module, model, propagation, slide)
    module = SegmentationModule(
        model,
        distiller,
        train_cfg,
        cfg.experiment.augment,
        validation_roots=data_module.data_roots,
        tiling=tiling,
        mpp=float(cfg.resolution.target_mpp),
        visualization_dir=output_dir / "validation",
        label_scale=label_scale,
        propagation=propagation,
        propagation_supervision=supervision,
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
    if bool(cfg.experiment.soft_label.enabled):
        if propagation is None:
            # 关掉 propagation 跑基线是常见操作, 可视化直接跳过而不是报错
            print(
                "experiment.soft_label.enabled=true 但 propagation 未启用, "
                "跳过软标签可视化",
                flush=True,
            )
        else:
            callbacks.append(
                SoftLabelVisualizationCallback(
                    cfg.experiment.soft_label,
                    tiling,
                    output_dir / "soft_labels",
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
        logger=CSVLogger(save_dir=output_dir, name="segmentation", version=""),
        enable_checkpointing=False,
        num_sanity_val_steps=0,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
    )
    trainer.fit(module, datamodule=data_module)
    return module


def _build_propagation_supervision(cfg, data_module, model, propagation, slide):
    supervision_cfg = cfg.experiment.get("propagation_supervision")
    weights = cfg.experiment.training.image.loss_weights
    names = ("propagation_query", "gene_auxiliary", "gene_propagation", "gene_prediction")
    active = supervision_cfg is not None and bool(supervision_cfg.enabled)
    if not active:
        if any(float(weights.get(name, 0)) for name in names):
            raise ValueError("Propagation supervision losses require propagation_supervision.enabled=true")
        return None
    if propagation is None:
        raise ValueError("Propagation supervision requires propagation.enabled=true")
    if cfg.experiment.dynamic.enabled or any(
        float(weights.get(name, 0)) for name in ("segmentation", "affinity", "superpixel")
    ) or any(cfg.experiment.augment[name].enabled for name in ("mixup", "cutmix")):
        raise ValueError(
            "Hidden-spot supervision requires dynamic=false, segmentation/affinity/superpixel=0 "
            "and mixup/cutmix=false: existing full-seed pseudo labels would reveal queries"
        )
    mode = str(supervision_cfg.gene_mode)
    for name, expected in (("gene_auxiliary", "auxiliary"), ("gene_propagation", "propagation"), ("gene_prediction", "prediction")):
        if bool(float(weights.get(name, 0))) != (mode == expected):
            raise ValueError(f"{name} weight must be nonzero exactly when gene_mode={expected}")
    if mode == "prediction" and model.gene_count != data_module.train_gene_adata.n_vars:
        raise ValueError("gene_mode=prediction requires a dense model gene head matching the selected genes")
    if any(float(weights.get(name, 0)) for name in ("propagation_query", "gene_propagation")):
        molecular_gradients = (
            mode == "prediction" and float(supervision_cfg.gene_edge_strength) > 0
            and not supervision_cfg.detach_gene_edges
        )
        if not propagation.feature_gradients and propagation.gate.gate_range == 0 and not molecular_gradients:
            raise ValueError("Query losses require feature_gradients=true or a learnable gate")
    if str(supervision_cfg.expression_transform) == "log1p" and "log1p" not in data_module.train_gene_adata.uns:
        raise ValueError("expression_transform=log1p requires log1p expression metadata")
    options = cast(dict[str, Any], OmegaConf.to_container(supervision_cfg, resolve=True))
    options.pop("enabled")
    return PropagationSupervision(
        feature_channels=model.feature_channels, gene_count=data_module.train_gene_adata.n_vars,
        spot_radius_px=slide.spot_radius_px, label_scale=slide.label_scale,
        seed=data_module.seed, **options,
    )


def _save_final_model(cfg, model, tiling, output_dir):
    """保存权重与重建所需信息，让 checkpoint 自描述（供外部单跑推理）。"""
    model_config = cast(
        dict[str, Any],
        OmegaConf.to_container(cfg.experiment.model.image, resolve=True),
    )
    model_config["pfm_patch_dims"] = list(model.pfm_patch_dims)
    model_config["pfm_cls_dims"] = list(model.pfm_cls_dims)
    model_config["gene_count"] = model.gene_count
    artifact = {
        "state_dict": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
        "model_config": model_config,
        "inference_config": {**tiling, "threshold": 0.5},
        "image_feature_map": list(cfg.experiment.dataset.image_feature_maps),
    }
    if model.gene_count:
        artifact["gene_names"] = model.gene_names
        artifact["gene_expression_space"] = "linear_relative_expression"
    torch.save(artifact, output_dir / "final_model.pt")


def _test_final_model(cfg, data_module, model, device, tiling, output_dir):
    """整片测试：逐张预测、落盘概率图与指标，返回汇总指标。"""
    data_module.setup("test")
    gene_options = {}
    if model.gene_count:
        gene_options = {
            "gene_adata": load_gene_adata(data_module.gene_path, data_module.test_names),
            "spot_radius_px": float(cfg.experiment.dataset.spot_diameter_um) / (2 * float(cfg.resolution.target_mpp)),
            "expression_transform": str(cfg.experiment.propagation_supervision.expression_transform),
        }
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
                **gene_options,
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
