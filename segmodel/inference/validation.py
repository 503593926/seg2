from pathlib import Path

import cv2
import numpy as np
import torch

from segmodel.dataset.loading import read_image
from segmodel.inference.metrics import segmentation_metrics
from segmodel.inference.predictor import predict_slide
from segmodel.modules.graph_measurements import pool_spot_expression


def gene_prediction_metrics(field, tissue, spots, truth, radius_px, label_scale, expression_transform="log1p"):
    """图像预测场在真实 spot 上的独立评估, 真值只参与指标计算。"""
    prediction, valid = pool_spot_expression(field, tissue, spots, radius_px, label_scale)
    metrics = {"gene_spots": int(valid.sum()), "gene_pcc_genes": 0}
    if not bool(valid.any()):
        return {**metrics, "gene_prediction": float("nan"), "gene_prediction_pcc": float("nan")}
    prediction, target = prediction[valid], truth[valid].float()
    if expression_transform == "log1p":
        prediction = prediction.log1p()
    metrics["gene_prediction"] = float((prediction - target).square().mean())
    centered_prediction = prediction - prediction.mean(0)
    centered_target = target - target.mean(0)
    p_std = prediction.std(0, unbiased=False)
    t_std = target.std(0, unbiased=False)
    usable = (p_std > 1e-6) & (t_std > 1e-6)
    metrics["gene_pcc_genes"] = int(usable.sum())
    metrics["gene_prediction_pcc"] = float(
        ((centered_prediction[:, usable] * centered_target[:, usable]).mean(0)
         / (p_std[usable] * t_std[usable])).clamp(-1, 1).mean()
    ) if bool(usable.any()) else float("nan")
    return metrics


def load_expert_masks(root, slide):
    sample = Path(root) / "samples" / slide.name
    labels = read_image(sample / "masks/training_mask.png", cv2.IMREAD_GRAYSCALE)
    valid = read_image(sample / "masks/valid_mask.png", cv2.IMREAD_GRAYSCALE) > 0
    canvas = read_image(sample / "spatial/canvas_mask.png", cv2.IMREAD_GRAYSCALE) > 0
    if any(
        array.shape != (slide.height, slide.width) for array in (labels, valid, canvas)
    ):
        raise ValueError(
            f"{slide.name}: expert masks must match the target-resolution H&E canvas"
        )
    if not np.isin(labels, (0, 1, 255)).all():
        raise ValueError(f"{slide.name}: training_mask must contain only 0/1/255")
    valid &= canvas & (labels != 255)
    if not valid.any():
        raise ValueError(f"{slide.name}: no valid expert pixels")
    return labels, valid


def validate_slide(
    model,
    slide,
    root,
    device,
    crop_size,
    stride,
    alignment,
    batch_size=1,
    mpp=1.0,
    threshold=0.5,
    gene_adata=None,
    spot_radius_px=None,
    expression_transform="log1p",
):
    probability, expression = predict_slide(
        model,
        slide,
        crop_size=crop_size,
        stride=stride,
        alignment=alignment,
        device=device,
        batch_size=batch_size,
        return_features=gene_adata is not None,
        feature_key="gene_expression" if gene_adata is not None else "propagation_features",
    )
    labels, valid = load_expert_masks(root, slide)
    metrics = segmentation_metrics(
        probability,
        labels,
        valid,
        threshold=threshold,
        mpp=mpp,
    )
    if gene_adata is not None:
        selected = gene_adata.obs["name"].astype(str).to_numpy() == slide.name
        spots = torch.from_numpy(gene_adata.obs.loc[selected, ["pixel_x", "pixel_y"]].to_numpy(dtype=np.float32))
        truth = torch.from_numpy(np.asarray(gene_adata.X[selected], dtype=np.float32))
        tissue = torch.from_numpy(read_image(
            Path(root) / "label_mpp" / slide.name / "spatial/tissue_mask.png", cv2.IMREAD_GRAYSCALE,
        ) > 0)
        metrics.update(gene_prediction_metrics(
            torch.from_numpy(expression).permute(2, 0, 1), tissue, spots, truth,
            spot_radius_px, slide.label_scale, expression_transform,
        ))
    return probability, metrics
