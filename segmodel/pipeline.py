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
from segmodel.modules.propagation import DifferentiablePropagation
from segmodel.training.callbacks import DatasetEpochCallback, MetricsBelowRichProgressBar
from segmodel.training.dynamic_snic import DynamicSNICCallback
from segmodel.training.segmentation_module import SegmentationModule
from segmodel.training.template_initialization import initialize_expression_templates
from segmodel.dataset.splits import patient_id


def run_fold(train_names, test_names, seed, cfg, output_dir):
    """跑完一个 fold/seed，返回整片测试的汇总指标。"""
    # 准备当前 fold 的输出目录和配置。
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "segmentation").mkdir(exist_ok=True)
    OmegaConf.save(cfg, output_dir / "config.yaml", resolve=True)
    _write_json(
        output_dir / "split.json",
        {"seed": int(seed), "train": list(train_names), "test": list(test_names)},
    )

    # 加载训练数据和留出患者的评估数据。
    pl.seed_everything(seed, workers=True)

    data_module = FoldDataModule(train_names, test_names, seed, cfg)
    if cfg.experiment.dataset.split_mode == "patient_loocv":
        _write_json(output_dir / "split.json", {
            "seed": int(seed), "train": list(train_names), "test": list(test_names),
            "train_labeled": data_module.train_labeled_names,
            "train_expression_only": [name for name in train_names if name not in data_module.train_labeled_names],
            "test_expression": data_module.expression_test_names,
            "test_patient": patient_id(test_names[0]),
        })
    data_module.setup("fit")

    tiling = _resolve_tiling(cfg, data_module)

    # 模型初始化和训练使用独立于数据准备的随机状态。
    pl.seed_everything(seed, workers=True)
    module = _train_segmentation(cfg, data_module, tiling, output_dir)
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


def _train_segmentation(cfg, data_module, tiling, output_dir):
    slide = next(iter(data_module.train_slides.values()))
    label_scale = slide.label_scale
    model_cfg = cfg.experiment.model.image
    train_cfg = cfg.experiment.training.image
    model_options = cast(dict[str, Any], OmegaConf.to_container(model_cfg, resolve=True))
    if model_options.get("joint_prediction", False) or model_options.get("tissue_state_count", 0) or any(
        float(train_cfg.loss_weights.get(name, 0)) for name in ("gene_prediction", "gene_correlation")
    ):
        model_options["gene_count"] = data_module.train_gene_adata.n_vars
    model = SegmentationModel(
        pfm_patch_dims=slide.patch_dims,
        pfm_cls_dims=slide.cls_dims,
        **model_options,
    )
    if model.gene_count:
        model.gene_names = list(map(str, data_module.train_gene_adata.var_names))
    template_cfg = train_cfg.get("template_init", {})
    if str(template_cfg.get("method", "random")).lower() != "random":
        initialize_expression_templates(
            model, data_module.train_gene_adata, data_module.train_names, template_cfg,
            expression_transform=str(train_cfg.expression_transform),
            project_root=Path(str(cfg.project_root)), output_dir=output_dir,
        )
    propagation = None
    if bool(cfg.experiment.propagation.enabled):
        propagation_cfg = cast(
            dict[str, Any],
            OmegaConf.to_container(cfg.experiment.propagation, resolve=True),
        )
        propagation_cfg.pop("enabled", None)
        propagation_cfg.pop("dynamic_snic", None)
        propagation = DifferentiablePropagation(
            label_scale=label_scale,
            **propagation_cfg,
        )
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
        propagation=propagation,
    )
    callbacks: list[pl.Callback] = [
        DatasetEpochCallback(),
        MetricsBelowRichProgressBar(
            metrics_every_n_steps=int(train_cfg.progress_every_n_steps),
            refresh_rate=100, leave=True,
        ),
    ]
    dynamic_cfg = cfg.experiment.propagation.get("dynamic_snic")
    if propagation is not None and dynamic_cfg is not None and bool(dynamic_cfg.enabled) and any(
        float(train_cfg.loss_weights.get(name, 0)) for name in ("propagation", "propagation_initial")
    ):
        callbacks.append(DynamicSNICCallback(
            dynamic_cfg, tiling, output_dir / "soft_labels", float(cfg.resolution.target_mpp),
        ))
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
            "gene_adata": load_gene_adata(data_module.gene_path, data_module.expression_test_names),
            "spot_radius_px": float(cfg.experiment.dataset.spot_diameter_um) / (2 * float(cfg.resolution.target_mpp)),
            "expression_transform": str(cfg.experiment.training.image.expression_transform),
        }
    slide_metrics = {}
    expression_results = {}
    try:
        for slide in data_module.test_dataloader():
            segmentation_results = {}
            probability, metrics = validate_slide(
                model,
                slide,
                data_module.data_roots[slide.name],
                device,
                **tiling,
                mpp=float(cfg.resolution.target_mpp),
                **gene_options,
                evaluate_segmentation=slide.name in data_module.test_names,
                expression_results=expression_results,
                segmentation_results=segmentation_results,
            )
            slide_dir = output_dir / "test" / slide.name
            slide_dir.mkdir(parents=True, exist_ok=True)
            if slide.name in data_module.test_names:
                probability = probability.astype(np.float32)
                mask = probability >= 0.5
                np.save(slide_dir / "probability.npy", probability)
                cv2.imwrite(str(slide_dir / "mask.png"), mask.astype(np.uint8) * 255)
                if segmentation_results:
                    initial = segmentation_results["initial_probability"]
                    np.save(slide_dir / "initial_probability.npy", initial)
                    np.save(slide_dir / "refinement_gate.npy", segmentation_results["refinement_gate"])
                    cv2.imwrite(str(slide_dir / "initial_mask.png"), (initial >= 0.5).astype(np.uint8) * 255)
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
