from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from seg2.infer.predictor import predict_slide
from seg2.models.region_pooling import prepare_region_pool
from seg2.test.metrics import expression_metrics, segmentation_metrics


def evaluate_region_probabilities(
    probability, root, name, target_mpp, prediction_mpp,
    spot_diameter_um=None, threshold=0.5, region_map=None,
):
    """Pool canvas-supported probabilities independently of expression QC."""
    from seg2.test.data import label_sample_dir

    spatial = pd.read_csv(
        filepath_or_buffer=Path(root) / "samples" / name / "spatial" / f"spatial_{target_mpp:g}mpp.tsv",
        sep="\t",
    )
    canvas_path = label_sample_dir(root=root, name=name, label_mpp=prediction_mpp, target_mpp=target_mpp) / "spatial/canvas_mask.png"
    canvas = cv2.imread(filename=str(canvas_path), flags=cv2.IMREAD_GRAYSCALE)
    if canvas is None or canvas.shape != probability.shape:
        raise ValueError(f"[{name}] Probability and canvas grids do not match")
    plan = prepare_region_pool(
        valid_mask=(canvas > 0) & np.isfinite(probability),
        coordinates=spatial[["pixel_x", "pixel_y"]].to_numpy(dtype=np.float32),
        radius_px=None if spot_diameter_um is None else float(spot_diameter_um) / (2 * target_mpp),
        label_scale=prediction_mpp / target_mpp,
        region_map=region_map,
        region_ids=spatial.spot_id.to_numpy(dtype=np.int64) if region_map is not None else None,
    )
    rows, pixels = plan["rows"].numpy(), plan["pixels"].numpy()
    counts = np.bincount(rows, minlength=plan["count"])
    sums = np.bincount(rows, weights=probability.ravel()[pixels], minlength=plan["count"])
    indices = plan["indices"].numpy()
    spatial["probability"] = np.nan
    spatial.loc[indices, "probability"] = sums / counts
    spatial["valid_pixel_count"] = 0
    spatial.loc[indices, "valid_pixel_count"] = counts
    spatial["coverage"] = spatial.valid_pixel_count / np.maximum(plan["expected_counts"].numpy(), 1)
    spatial["classification_valid"] = (
        spatial.spot_label.isin([0, 1]) & spatial.in_content
        & spatial.selected.eq(1) & spatial.probability.notna()
    )
    spatial["prediction"] = pd.array(
        [int(p >= threshold) if np.isfinite(p) else pd.NA for p in spatial.probability], dtype="Int64",
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
        region_ids = gene_adata.obs.loc[selected, "spot_id"].astype(str).to_numpy(dtype=str)
        truth = torch.from_numpy(np.asarray(gene_adata.X[selected], dtype=np.float32))
        observed = torch.ones_like(truth, dtype=torch.bool)
        missing = gene_adata.uns.get("missing_selected_genes_by_slide", {}).get(
            slide["name"], []
        )
        if len(missing):
            observed[:, torch.from_numpy(np.isin(gene_adata.var_names, missing))] = (
                False
            )
        mask_root = (
            Path(root) / "samples"
            if slide["label_scale"] == 1
            else Path(root) / "label_mpp" / f"{float(mpp) * slide['label_scale']:g}mpp"
        )
        canvas = cv2.imread(
            str(mask_root / slide["name"] / "spatial/canvas_mask.png"), cv2.IMREAD_GRAYSCALE,
        )
        if canvas is None:
            raise FileNotFoundError(mask_root / slide["name"] / "spatial/canvas_mask.png")
        plan = prepare_region_pool(
            valid_mask=canvas > 0, coordinates=spots, radius_px=spot_radius_px,
            label_scale=slide["label_scale"], region_map=slide["region_map"],
            region_ids=region_ids.astype(np.int64) if slide["region_type"] == "polygon" else None,
        )
        pixels, rows = (plan["pixels"].numpy(), plan["rows"].numpy())
        order = np.argsort(pixels, kind="stable")
        region_pool = {
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
            model=model,
            slide=slide,
            crop_size=crop_size,
            stride=stride,
            device=device,
            batch_size=batch_size,
            outputs=outputs,
            expression_pool=region_pool,
        )
        expression = result.get("expression")
        assert expression is not None
        valid = torch.zeros(len(spots), dtype=torch.bool)
        valid[torch.from_numpy(region_pool["indices"])] = True
        prediction = torch.from_numpy(expression["coefficients"]) @ torch.from_numpy(
            expression["templates"]
        )
        target, observed = (truth[valid].float(), observed[valid])
        if expression_transform == "log1p":
            prediction = prediction.log1p()
        metrics.update(expression_metrics(prediction=prediction, target=target, observed=observed))
        if expression_results is not None:
            valid = valid.numpy()
            region_results = {
                "prediction": prediction.cpu(),
                "target": target.cpu(),
                "observed": observed.cpu(),
                "region_ids": region_ids[valid],
                "coordinates": spots.numpy()[valid],
            }
            if cache_dense:
                factors = result.get("dense_expression")
                assert factors is not None
                region_results["dense_expression"] = factors
                # Keep the existing display masks separate from expression pooling.
                region_results["dense_tissue"] = canvas > 0 if slide["region_type"] == "polygon" else (
                    np.asarray(cv2.imread(
                        str(mask_root / slide["name"] / "spatial/tissue_mask.png"), cv2.IMREAD_GRAYSCALE,
                    )) > 0
                )
                region_results["dense_mpp"] = float(mpp) * slide["label_scale"]
            expression_results[slide["name"]] = region_results
    else:
        result = predict_slide(
            model=model,
            slide=slide,
            crop_size=crop_size,
            stride=stride,
            device=device,
            batch_size=batch_size,
        )
    probability = result.get("probability")
    assert probability is not None
    if evaluate_segmentation:
        labels, valid = load_expert_masks(root=root, slide=slide, mpp=mpp)
        metrics = {
            **segmentation_metrics(probability=probability, labels=labels, valid=valid, threshold=threshold),
            **metrics,
        }
    return (probability, metrics)
