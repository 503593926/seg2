from pathlib import Path

import cv2
import numpy as np
import torch

from seg2.infer.predictor import predict_slide
from seg2.models.spot_pooling import prepare_spot_pool
from seg2.test.data import label_grid_root
from seg2.test.metrics import expression_metrics, segmentation_metrics


def load_expert_masks(root, slide, mpp=1.0):
    sample = (
        Path(root) / "samples"
        if slide["label_scale"] == 1
        else label_grid_root(root, mpp, float(mpp) * slide["label_scale"])
    ) / slide["name"]
    labels = np.asarray(
        cv2.imread(str(sample / "masks/training_mask.png"), cv2.IMREAD_GRAYSCALE)
    )
    valid = (
        np.asarray(
            cv2.imread(str(sample / "masks/valid_mask.png"), cv2.IMREAD_GRAYSCALE)
        )
        > 0
    )
    canvas = (
        np.asarray(
            cv2.imread(str(sample / "spatial/canvas_mask.png"), cv2.IMREAD_GRAYSCALE)
        )
        > 0
    )
    valid &= canvas & (labels != 255)
    return (labels, valid)


def validate_slide(
    model,
    slide,
    root,
    device,
    crop_size,
    stride=None,
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
    cache_dense = (
        save_dense_expression
        and gene_adata is not None
        and (expression_results is not None)
    )
    metrics = {}
    factors = {}
    spot_pool = None
    if gene_adata is not None:
        selected = gene_adata.obs["name"].astype(str).to_numpy() == slide["name"]
        spots = torch.from_numpy(
            gene_adata.obs.loc[selected, ["pixel_x", "pixel_y"]].to_numpy(
                dtype=np.float32
            )
        )
        spot_ids = gene_adata.obs.index[selected].to_numpy(dtype=str)
        truth = torch.from_numpy(np.asarray(gene_adata.X[selected], dtype=np.float32))
        observed = torch.ones_like(truth, dtype=torch.bool)
        missing = gene_adata.uns.get("missing_selected_genes_by_slide", {}).get(
            slide["name"], []
        )
        if len(missing):
            observed[:, torch.from_numpy(np.isin(gene_adata.var_names, missing))] = (
                False
            )
        tissue_root = (
            Path(root) / "samples"
            if slide["label_scale"] == 1
            else label_grid_root(root, mpp, float(mpp) * slide["label_scale"])
        )
        tissue = torch.from_numpy(
            np.asarray(
                cv2.imread(
                    str(tissue_root / slide["name"] / "spatial/tissue_mask.png"),
                    cv2.IMREAD_GRAYSCALE,
                )
            )
            > 0
        )
        plan = prepare_spot_pool(tissue, spots, spot_radius_px, slide["label_scale"])
        pixels, rows = (plan["pixels"].numpy(), plan["rows"].numpy())
        order = np.argsort(pixels, kind="stable")
        spot_pool = {
            "pixels": pixels[order],
            "rows": rows[order],
            "pixel_counts": np.bincount(rows, minlength=plan["count"]),
            "indices": plan["indices"].numpy(),
        }
        del plan
        probability, _ = predict_slide(
            model,
            slide,
            crop_size=crop_size,
            stride=stride,
            device=device,
            batch_size=batch_size,
            return_features=False,
            expression_factors=factors if cache_dense else None,
            spot_pool=spot_pool,
        )
        valid = torch.zeros(len(spots), dtype=torch.bool)
        valid[torch.from_numpy(spot_pool["indices"])] = True
        prediction = torch.from_numpy(spot_pool["coefficients"]) @ torch.from_numpy(
            spot_pool["templates"]
        )
        target, observed = (truth[valid].float(), observed[valid])
        if expression_transform == "log1p":
            prediction = prediction.log1p()
        metrics.update(expression_metrics(prediction, target, observed))
        if expression_results is not None:
            valid = valid.numpy()
            spot_results = {
                "prediction": prediction.cpu(),
                "target": target.cpu(),
                "observed": observed.cpu(),
                "spot_ids": spot_ids[valid],
                "spots": spots.numpy()[valid],
            }
            if cache_dense:
                spot_results["dense_expression"] = factors
                spot_results["dense_tissue"] = tissue.numpy()
                spot_results["dense_mpp"] = float(mpp) * slide["label_scale"]
            expression_results[slide["name"]] = spot_results
    else:
        probability, _ = predict_slide(
            model,
            slide,
            crop_size=crop_size,
            stride=stride,
            device=device,
            batch_size=batch_size,
            return_features=False,
        )
    if evaluate_segmentation:
        labels, valid = load_expert_masks(root, slide, mpp)
        metrics = {
            **segmentation_metrics(probability, labels, valid, threshold=threshold),
            **metrics,
        }
    return (probability, metrics)
