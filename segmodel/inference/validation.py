from pathlib import Path
from typing import cast

import cv2
import numpy as np
import torch

from segmodel.inference.metrics import expression_metrics, segmentation_metrics
from segmodel.inference.predictor import predict_slide
from segmodel.modules.spot_pooling import pool_spot_expression, prepare_spot_pool


def load_expert_masks(root, slide):
    sample = Path(root) / "samples" / slide.name
    labels = np.asarray(cv2.imread(str(sample / "masks/training_mask.png"), cv2.IMREAD_GRAYSCALE))
    valid = np.asarray(cv2.imread(str(sample / "masks/valid_mask.png"), cv2.IMREAD_GRAYSCALE)) > 0
    canvas = np.asarray(cv2.imread(str(sample / "spatial/canvas_mask.png"), cv2.IMREAD_GRAYSCALE)) > 0
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
    save_dense_expression=False,
):
    factorized = bool(getattr(model, "tissue_state_count", 0)) and gene_adata is not None
    cache_dense = save_dense_expression and gene_adata is not None and expression_results is not None
    metrics = {}
    factors = {}
    spot_pool = None
    if gene_adata is not None:
        selected = gene_adata.obs["name"].astype(str).to_numpy() == slide.name
        spots = torch.from_numpy(gene_adata.obs.loc[selected, ["pixel_x", "pixel_y"]].to_numpy(dtype=np.float32))
        spot_ids = gene_adata.obs.index[selected].to_numpy(dtype=str)
        truth = torch.from_numpy(np.asarray(gene_adata.X[selected], dtype=np.float32))
        observed = torch.ones_like(truth, dtype=torch.bool)
        missing = getattr(gene_adata, "uns", {}).get("missing_selected_genes_by_slide", {}).get(slide.name, [])
        if len(missing):
            observed[:, torch.from_numpy(np.isin(gene_adata.var_names, missing))] = False
        tissue = torch.from_numpy(np.asarray(cv2.imread(
            str(Path(root) / "samples" / slide.name / "spatial/tissue_mask.png"), cv2.IMREAD_GRAYSCALE,
        )) > 0)
        if factorized:
            # 按整片像素位置排序，窗口内只扫描对应行范围的 spot 像素。
            plan = prepare_spot_pool(tissue, spots, spot_radius_px, slide.label_scale)
            pixels, rows = plan["pixels"].numpy(), plan["rows"].numpy()
            order = np.argsort(pixels, kind="stable")
            spot_pool = {
                "pixels": pixels[order], "rows": rows[order],
                "pixel_counts": np.bincount(rows, minlength=plan["count"]),
                "indices": plan["indices"].numpy(),
            }
            del plan
        probability, expression = predict_slide(
            model,
            slide,
            crop_size=crop_size,
            stride=stride,
            alignment=alignment,
            device=device,
            batch_size=batch_size,
            return_features=not factorized,
            feature_key="gene_expression",
            expression_factors=factors if cache_dense else None,
            spot_pool=spot_pool,
        )
        if spot_pool is not None:
            valid = torch.zeros(len(spots), dtype=torch.bool)
            valid[torch.from_numpy(spot_pool["indices"])] = True
            prediction = torch.from_numpy(spot_pool["coefficients"]) @ torch.from_numpy(spot_pool["templates"])
        else:
            prediction, valid = pool_spot_expression(
                torch.from_numpy(cast(np.ndarray, expression)).permute(2, 0, 1),
                tissue, spots, spot_radius_px, slide.label_scale,
            )
            prediction = prediction[valid]
        target, observed = truth[valid].float(), observed[valid]
        if expression_transform == "log1p":
            prediction = prediction.log1p()
        metrics.update(expression_metrics(prediction, target, observed))
        if expression_results is not None:
            valid = valid.numpy()
            spot_results = {
                "prediction": prediction.cpu(), "target": target.cpu(), "observed": observed.cpu(),
                "spot_ids": spot_ids[valid], "spots": spots.numpy()[valid],
            }
            if cache_dense:
                spot_results["dense_expression"] = factors or {"prediction": expression}
                spot_results["dense_tissue"] = tissue.numpy()
                spot_results["dense_mpp"] = float(mpp) * slide.label_scale
            expression_results[slide.name] = spot_results
    else:
        probability, _ = predict_slide(
            model, slide, crop_size=crop_size, stride=stride, alignment=alignment,
            device=device, batch_size=batch_size, return_features=False,
        )
    if evaluate_segmentation:
        labels, valid = load_expert_masks(root, slide)
        metrics = {**segmentation_metrics(probability, labels, valid, threshold=threshold), **metrics}
    return probability, metrics
