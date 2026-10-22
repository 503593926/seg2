"""一个 fold/seed 的完整流程：准备数据 → 联合训练 → 保存模型 → 整片测试。"""

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
from segmodel.inference.metrics import summarize_metrics
from segmodel.inference.validation import validate_slide
from segmodel.models.segmentation import SegmentationModel
from segmodel.dynamic.callback import DynamicPropagationCallback
from segmodel.training.callbacks import DatasetEpochCallback, MetricsBelowRichProgressBar
from segmodel.training.segmentation_module import SegmentationModule


def run_fold(fold, seed, cfg, output_dir):
    OmegaConf.save(cfg, output_dir / "config.yaml", resolve=True)
    split_info = {key: fold[key] for key in ("train_names", "test_names")}
    split_info["seed"] = int(seed)
    if "test_patient" in fold:
        split_info["test_patient"] = fold["test_patient"]
    _write_json(output_dir / "split.json", split_info)
    data_module = FoldDataModule(fold, seed, cfg)
    data_module.setup("fit")
    tiling = _resolve_tiling(cfg, data_module)

    pl.seed_everything(seed, workers=True)
    module = _train_segmentation(cfg, data_module, tiling, output_dir)
    data_module.teardown("fit")

    # 保存最终图像模型，然后执行整片测试。
    _save_final_model(cfg, module.model, tiling, output_dir)
    # fit 清理时模型已移回 CPU；测试恢复训练设备与精度上下文。
    trainer = module.trainer
    device = trainer.strategy.root_device
    module.model.to(device)
    print(f"final test: device={device}, precision={trainer.precision}", flush=True)
    with trainer.precision_plugin.forward_context():
        return _test_final_model(cfg, data_module, module.model, device, tiling, output_dir)


def _resolve_tiling(cfg, data_module):
    """训练与推理共用 crop；推理步长取半窗附近的网格对齐值。"""
    crop_size = int(cfg.experiment.dataset.crop_size)
    alignment = next(iter(data_module.train_slides.values())).alignment
    stride = max(alignment, crop_size // 2 // alignment * alignment)
    if crop_size % alignment or stride % alignment:
        raise ValueError(
            f"crop_size={crop_size} and stride={stride} must be divisible by "
            f"MPP-derived alignment={alignment}"
        )
    model_cfg = cfg.experiment.model.image
    model_scale = int(model_cfg.output_scale) * int(model_cfg.downsample_factor) ** (len(model_cfg.encoder_chans) - 1)
    if crop_size % model_scale:
        raise ValueError(f"crop_size={crop_size} must be divisible by model total downsampling={model_scale}")
    return {
        "crop_size": crop_size,
        "stride": stride,
        "alignment": alignment,
        "batch_size": int(cfg.experiment.dataset.inference_batch_size),
    }


def _train_segmentation(cfg, data_module, tiling, output_dir):
    slide = next(iter(data_module.train_slides.values()))
    label_scale = slide.label_scale
    model_cfg = cfg.experiment.model.image
    train_cfg = cfg.experiment.training.image
    model_options = cast(dict[str, Any], OmegaConf.to_container(model_cfg, resolve=True))
    if int(model_options["output_scale"]) != label_scale:
        raise ValueError("model.image.output_scale must match resolution.label_mpp / target_mpp")
    model_options["gene_count"] = data_module.train_gene_adata.n_vars
    model_options["tissue_state_count"] = train_cfg.template_init.random_count
    model = SegmentationModel(
        pfm_patch_dims=slide.patch_dims,
        pfm_cls_dims=slide.cls_dims,
        **model_options,
    )
    model.gene_names = list(map(str, data_module.train_gene_adata.var_names))
    module = SegmentationModule(
        model,
        train_cfg,
        cfg.experiment.augment,
        validation_roots=data_module.data_roots,
        tiling=tiling,
        mpp=float(cfg.resolution.target_mpp),
        visualization_dir=output_dir / "validation",
        label_scale=label_scale,
        spot_radius_px=slide.spot_radius_px,
    )
    callbacks: list[pl.Callback] = [
        DatasetEpochCallback(),
        MetricsBelowRichProgressBar(
            metrics_every_n_steps=int(train_cfg.progress_every_n_steps),
            refresh_rate=100, leave=True,
        ),
    ]
    dynamic_cfg = cfg.experiment.dynamic
    if bool(dynamic_cfg.enabled):
        callbacks.append(DynamicPropagationCallback(
            dynamic_cfg, tiling, output_dir / "dynamic_pseudo_labels",
        ))
    runtime_cfg = cfg.experiment.training.runtime
    trainer = pl.Trainer(
        accelerator=str(runtime_cfg.accelerator),
        devices=runtime_cfg.devices,
        precision=runtime_cfg.precision,
        accumulate_grad_batches=int(train_cfg.accumulate_grad_batches),
        max_epochs=int(train_cfg.epochs),
        check_val_every_n_epoch=int(train_cfg.val_every_n_epochs),
        limit_val_batches=1.0 if cfg.experiment.dataset.split_mode == "patient_loocv" else 0.0,
        default_root_dir=output_dir,
        callbacks=callbacks,
        logger=CSVLogger(save_dir=output_dir, name="", version=""),
        enable_checkpointing=False,
        num_sanity_val_steps=0,
        # gradient_clip_val=1.0,
        # gradient_clip_algorithm="norm",
    )
    trainer.fit(module, datamodule=data_module)
    return module


def _save_final_model(cfg, model, tiling, output_dir):
    model_config = cast(
        dict[str, Any],
        OmegaConf.to_container(cfg.experiment.model.image, resolve=True),
    )
    model_config["pfm_patch_dims"] = list(model.pfm_patch_dims)
    model_config["pfm_cls_dims"] = list(model.pfm_cls_dims)
    model_config["gene_count"] = model.gene_count
    model_config["tissue_state_count"] = model.tissue_state_count
    model_config["output_scale"] = model.output_scale
    model_config["gradient_checkpointing"] = model.gradient_checkpointing
    model_config["downsample_factor"] = model.downsample_factor
    artifact = {
        "state_dict": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
        "model_config": model_config,
        "inference_config": {**tiling, "threshold": 0.5},
        "image_feature_map": list(cfg.experiment.dataset.image_feature_maps),
        "resolution": {
            "target_mpp": float(cfg.resolution.target_mpp),
            "prediction_mpp": float(cfg.resolution.target_mpp) * model.output_scale,
        },
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
            "expression_transform": str(cfg.experiment.training.image.expression_transform),
        }
    slide_metrics = {}
    expression_results = {}
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
                evaluate_segmentation=slide.has_segmentation,
                expression_results=expression_results,
            )
            slide_dir = output_dir / "test" / slide.name
            slide_dir.mkdir(parents=True, exist_ok=True)
            if slide.has_segmentation:
                probability = probability.astype(np.float32)
                mask = probability >= 0.5
                np.save(slide_dir / "probability.npy", probability)
                cv2.imwrite(str(slide_dir / "mask.png"), mask.astype(np.uint8) * 255)
            if slide.name in expression_results:
                np.savez_compressed(
                    slide_dir / "gene_predictions.npz",
                    **expression_results[slide.name],
                    gene_names=np.asarray(model.gene_names, dtype=str),
                    expression_transform=gene_options["expression_transform"],
                )
            _write_json(slide_dir / "metrics.json", metrics)
            slide_metrics[slide.name] = metrics
    finally:
        data_module.teardown("test")

    summary = summarize_metrics(slide_metrics, expression_results)
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
