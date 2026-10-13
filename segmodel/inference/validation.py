from pathlib import Path

import cv2
import numpy as np
import torch

from segmodel.dataset.loading import read_image
from segmodel.inference.metrics import expression_metrics, segmentation_metrics
from segmodel.inference.predictor import predict_slide
from segmodel.modules.spot_pooling import pool_spot_expression


def gene_prediction_metrics(field, tissue, spots, truth, radius_px, label_scale, expression_transform="log1p", spot_results=None, gene_observed=None):
    """图像预测场在真实 spot 上的独立评估, 真值只参与指标计算。"""
    prediction, valid = pool_spot_expression(field, tissue, spots, radius_px, label_scale)
    prediction, target = prediction[valid], truth[valid].float()
    observed = torch.ones_like(target, dtype=torch.bool) if gene_observed is None else gene_observed[valid].bool()
    if expression_transform == "log1p":
        prediction = prediction.log1p()
    if spot_results is not None:
        spot_results.update(prediction=prediction.cpu(), target=target.cpu(), observed=observed.cpu(), valid=valid.cpu())
    return expression_metrics(prediction, target, observed)


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
    evaluate_segmentation=True,
    expression_results=None,
    segmentation_results=None,
):
    auxiliary = {} if getattr(model, "joint_prediction", False) and evaluate_segmentation else None
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
        auxiliary_outputs=auxiliary,
    )
    metrics = {}
    if evaluate_segmentation:
        labels, valid = load_expert_masks(root, slide)
        metrics.update(segmentation_metrics(
            probability, labels, valid, threshold=threshold, mpp=mpp,
        ))
        if auxiliary:
            initial_metrics = segmentation_metrics(
                auxiliary["initial_probability"], labels, valid, threshold=threshold, mpp=mpp,
            )
            metrics.update({f"initial_{key}": initial_metrics[key] for key in ("loss", "dice", "iou", "hd95")})
            metrics["refinement_dice_gain"] = metrics["dice"] - initial_metrics["dice"]
            if segmentation_results is not None:
                segmentation_results.update(auxiliary)
    if gene_adata is not None:
        selected = gene_adata.obs["name"].astype(str).to_numpy() == slide.name
        spots = torch.from_numpy(gene_adata.obs.loc[selected, ["pixel_x", "pixel_y"]].to_numpy(dtype=np.float32))
        truth = torch.from_numpy(np.asarray(gene_adata.X[selected], dtype=np.float32))
        observed = torch.ones_like(truth, dtype=torch.bool)
        missing = getattr(gene_adata, "uns", {}).get("missing_selected_genes_by_slide", {}).get(slide.name, [])
        if len(missing):
            observed[:, torch.from_numpy(np.isin(gene_adata.var_names, missing))] = False
        tissue = torch.from_numpy(read_image(
            Path(root) / "label_mpp" / slide.name / "spatial/tissue_mask.png", cv2.IMREAD_GRAYSCALE,
        ) > 0)
        spot_results = {} if expression_results is not None else None
        metrics.update(gene_prediction_metrics(
            torch.from_numpy(expression).permute(2, 0, 1), tissue, spots, truth,
            spot_radius_px, slide.label_scale, expression_transform, spot_results, observed,
        ))
        if expression_results is not None:
            assert spot_results is not None
            valid = spot_results.pop("valid").numpy()
            spot_results["spot_ids"] = gene_adata.obs.index[selected].to_numpy(dtype=str)[valid]
            spot_results["spots"] = spots.numpy()[valid]
            expression_results[slide.name] = spot_results
    return probability, metrics
