import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from seg2.infer.predictor import predict_slide
from seg2.models.spot_pooling import prepare_spot_pool
from seg2.test.metrics import expression_metrics, segmentation_metrics


def evaluate_spot_probabilities(
    probability, root, name, target_mpp, prediction_mpp, spot_diameter_um, threshold=0.5
):
    """Pool probabilities over physical disks independently of expression QC."""
    from seg2.test.data import label_sample_dir

    spatial = pd.read_csv(
        Path(root) / "samples" / name / "spatial" / f"spatial_{target_mpp:g}mpp.tsv",
        sep="\t",
    )
    canvas_path = (
        label_sample_dir(root, name, prediction_mpp, target_mpp)
        / "spatial/canvas_mask.png"
    )
    canvas = cv2.imread(str(canvas_path), cv2.IMREAD_GRAYSCALE)
    if canvas is None or canvas.shape != probability.shape:
        raise ValueError(f"[{name}] Probability and canvas grids do not match")
    if float(spot_diameter_um) <= 0:
        raise ValueError("Spot classification requires a positive physical diameter")
    points = torch.from_numpy(
        spatial[["pixel_x", "pixel_y"]].to_numpy(dtype=np.float32)
    )
    plan = prepare_spot_pool(
        torch.from_numpy((canvas > 0) & np.isfinite(probability)),
        points,
        float(spot_diameter_um) / (2 * target_mpp),
        prediction_mpp / target_mpp,
    )
    rows, pixels = plan["rows"].numpy(), plan["pixels"].numpy()
    counts = np.bincount(rows, minlength=plan["count"])
    sums = np.bincount(
        rows, weights=probability.ravel()[pixels], minlength=plan["count"]
    )
    indices = plan["indices"].numpy()
    spatial["probability"] = np.nan
    spatial.loc[indices, "probability"] = sums / counts
    spatial["valid_pixel_count"] = 0
    spatial.loc[indices, "valid_pixel_count"] = counts
    radius = spot_diameter_um / (2 * prediction_mpp)
    expected = []
    for center in points / (prediction_mpp / target_mpp):
        x, y = center.tolist()
        xx = torch.arange(math.ceil(x - radius), math.floor(x + radius) + 1)
        yy = torch.arange(math.ceil(y - radius), math.floor(y + radius) + 1)
        expected.append(
            int(
                ((xx[None, :] - x).square() + (yy[:, None] - y).square() <= radius**2)
                .sum()
                .item()
            )
        )
    spatial["coverage"] = spatial.valid_pixel_count / np.maximum(expected, 1)
    spatial["classification_valid"] = (
        spatial.spot_label.isin([0, 1])
        & spatial.in_content
        & spatial.selected.eq(1)
        & spatial.probability.notna()
    )
    spatial["prediction"] = pd.array(
        [int(p >= threshold) if np.isfinite(p) else pd.NA for p in spatial.probability],
        dtype="Int64",
    )
    if "patient_id" not in spatial:
        spatial["patient_id"] = name
    spatial["name"] = name
    return spatial


def load_expert_masks(root, slide, mpp=1.0):
    sample = (
        Path(root) / "samples"
        if slide["label_scale"] == 1
        else Path(root) / "label_mpp" / f"{float(mpp) * slide['label_scale']:g}mpp"
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
            else Path(root) / "label_mpp" / f"{float(mpp) * slide['label_scale']:g}mpp"
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
        outputs = ["probability", "expression"]
        if cache_dense:
            outputs.append("dense_expression")
        result = predict_slide(
            model,
            slide,
            crop_size=crop_size,
            stride=stride,
            device=device,
            batch_size=batch_size,
            outputs=outputs,
            expression_pool=spot_pool,
        )
        expression = result.get("expression")
        assert expression is not None
        valid = torch.zeros(len(spots), dtype=torch.bool)
        valid[torch.from_numpy(spot_pool["indices"])] = True
        prediction = torch.from_numpy(expression["coefficients"]) @ torch.from_numpy(
            expression["templates"]
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
                factors = result.get("dense_expression")
                assert factors is not None
                spot_results["dense_expression"] = factors
                spot_results["dense_tissue"] = tissue.numpy()
                spot_results["dense_mpp"] = float(mpp) * slide["label_scale"]
            expression_results[slide["name"]] = spot_results
    else:
        result = predict_slide(
            model,
            slide,
            crop_size=crop_size,
            stride=stride,
            device=device,
            batch_size=batch_size,
        )
    probability = result.get("probability")
    assert probability is not None
    if evaluate_segmentation:
        labels, valid = load_expert_masks(root, slide, mpp)
        metrics = {
            **segmentation_metrics(probability, labels, valid, threshold=threshold),
            **metrics,
        }
    return (probability, metrics)
