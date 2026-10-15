import math
from typing import cast

import numpy as np
from numpy.typing import NDArray
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt


def probability_loss(probability, target, valid):
    if not valid.any():
        raise ValueError("metrics require at least one valid pixel")
    probability = np.clip(probability[valid], 1e-7, 1 - 1e-7)
    target = target[valid].astype(np.float32)
    bce = -(target * np.log(probability) + (1 - target) * np.log1p(-probability)).mean(
        dtype=np.float64
    )
    dice = 1 - (2 * np.sum(probability * target, dtype=np.float64) + 1) / (
        probability.sum(dtype=np.float64) + target.sum(dtype=np.float64) + 1
    )
    return float(bce + dice)


def hd95(prediction, truth, valid, mpp=1.0):
    prediction = np.asarray(prediction, bool) & valid
    truth = np.asarray(truth, bool) & valid
    if not prediction.any() and not truth.any():
        return 0.0
    if not prediction.any() or not truth.any():
        return float("inf")
    if np.array_equal(prediction, truth):
        return 0.0
    prediction_surface = prediction & ~cast(NDArray[np.bool_], binary_erosion(prediction | ~valid))
    truth_surface = truth & ~cast(NDArray[np.bool_], binary_erosion(truth | ~valid))
    if not prediction_surface.any() or not truth_surface.any():
        return float("nan")
    # sampling=mpp：把距离从"像素数"换算成微米，HD95 因此带物理单位
    forward = cast(NDArray[np.float64], distance_transform_edt(~truth_surface, sampling=mpp))[
        prediction_surface
    ]
    backward = cast(NDArray[np.float64], distance_transform_edt(~prediction_surface, sampling=mpp))[
        truth_surface
    ]
    return float(np.percentile(np.concatenate((forward, backward)), 95))


def scores_from_counts(tp, fp, fn, tn):
    total = tp + fp + fn + tn
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "valid_pixels": total,
        "dice": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0,
        "iou": tp / (tp + fp + fn) if tp + fp + fn else 1.0,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "accuracy": (tp + tn) / total,
        "balanced_accuracy": (
            (recall + specificity) / 2
            if recall is not None and specificity is not None
            else None
        ),
        "mcc": (tp * tn - fp * fn) / denominator if denominator else None,
    }


def segmentation_metrics(probability, labels, valid, threshold=0.5, mpp=1.0):
    if probability.shape != labels.shape or labels.shape != valid.shape:
        raise ValueError("probability, labels and valid must have the same shape")
    valid = valid & (labels != 255)
    if not valid.any():
        raise ValueError("metrics require at least one valid pixel")
    prediction = probability >= threshold
    truth = labels == 1
    scores = scores_from_counts(
        int(np.count_nonzero(prediction & truth & valid)),
        int(np.count_nonzero(prediction & ~truth & valid)),
        int(np.count_nonzero(~prediction & truth & valid)),
        int(np.count_nonzero(~prediction & ~truth & valid)),
    )
    scores["loss"] = probability_loss(probability, truth, valid)
    scores["hd95"] = hd95(prediction, truth, valid, mpp)
    return scores


def _summarize_head(metrics_list):
    counts = [
        sum(scores[key] for scores in metrics_list)
        for key in ("tp", "fp", "fn", "tn")
    ]
    macro_keys = ("loss", "dice", "iou")
    distances = np.asarray([scores["hd95"] for scores in metrics_list])
    defined_distances = distances[~np.isnan(distances)]
    return {
        "macro": {
            **{
                key: float(np.mean([scores[key] for scores in metrics_list]))
                for key in macro_keys
            },
            "hd95": (
                float(defined_distances.mean())
                if defined_distances.size
                else float("nan")
            ),
            "hd95_infinite_slides": int(np.isinf(distances).sum()),
            "hd95_undefined_slides": int(np.isnan(distances).sum()),
        },
        "micro": scores_from_counts(*counts),
    }


def gene_pearson_correlations(prediction, target, observed=None):
    """逐基因跨 spot 的 PCC；观测不足、常量预测或常量真值返回 NaN。"""
    prediction, target = prediction.float(), target.float()
    observed = torch.ones_like(target, dtype=torch.bool) if observed is None else observed.bool()
    counts = observed.sum(0).clamp_min(1)
    centered_prediction = torch.where(observed, prediction - (prediction * observed).sum(0) / counts, 0.)
    centered_target = torch.where(observed, target - (target * observed).sum(0) / counts, 0.)
    p_std = (centered_prediction.square().sum(0) / counts).sqrt()
    t_std = (centered_target.square().sum(0) / counts).sqrt()
    usable = (counts >= 2) & (p_std > 1e-6) & (t_std > 1e-6)
    pcc = prediction.new_full((prediction.shape[1],), float("nan"))
    pcc[usable] = (
        ((centered_prediction[:, usable] * centered_target[:, usable]).sum(0) / counts[usable]
         / (p_std[usable] * t_std[usable])).clamp(-1, 1)
    )
    return pcc


def expression_metrics(prediction, target, observed=None):
    """同一基因跨spot的PCC统计均值和极值; 常量预测/目标不计入PCC。"""
    prediction, target = prediction.float(), target.float()
    observed = torch.ones_like(target, dtype=torch.bool) if observed is None else observed.bool()
    metrics: dict[str, int | float] = {
        "gene_spots": len(prediction), "gene_pcc_genes": 0, "gene_values": int(observed.sum())
    }
    if not bool(observed.any()):
        return {**metrics, "gene_prediction": float("nan"), "gene_prediction_rmse": float("nan"),
                "gene_prediction_pcc": float("nan"), "gene_prediction_pcc_max": float("nan"),
                "gene_prediction_pcc_min": float("nan")}
    mse = float((prediction - target)[observed].square().mean())
    metrics.update(gene_prediction=mse, gene_prediction_rmse=math.sqrt(mse))
    correlations = gene_pearson_correlations(prediction, target, observed)
    pcc = correlations[torch.isfinite(correlations)]
    metrics["gene_pcc_genes"] = pcc.numel()
    metrics["gene_prediction_pcc"] = float(pcc.mean()) if pcc.numel() else float("nan")
    metrics["gene_prediction_pcc_max"] = float(pcc.max()) if pcc.numel() else float("nan")
    metrics["gene_prediction_pcc_min"] = float(pcc.min()) if pcc.numel() else float("nan")
    return metrics


def summarize_expression_metrics(slides, expression_results):
    valid = [record for record in expression_results.values() if len(record["prediction"])]
    if not valid:
        return {}
    micro = expression_metrics(
        torch.cat([record["prediction"] for record in valid]),
        torch.cat([record["target"] for record in valid]),
        torch.cat([record.get("observed", torch.ones_like(record["target"], dtype=torch.bool)) for record in valid]),
    )
    per_slide = {
        name: {key: value for key, value in metrics.items() if key.startswith("gene_")}
        for name, metrics in slides.items() if "gene_prediction" in metrics
    }
    macro = {}
    for key in ("gene_prediction", "gene_prediction_rmse", "gene_prediction_pcc"):
        values = [metrics[key] for metrics in per_slide.values() if np.isfinite(metrics[key])]
        macro[key] = float(np.mean(values)) if values else float("nan")
    return {"micro": micro, "macro": macro, "slides": per_slide}


def summarize_metrics(slides, expression_results=None):
    values = [metrics for metrics in slides.values() if "dice" in metrics]
    summary = _summarize_head(values) if values else {"macro": {}, "micro": {}}
    summary["slides"] = slides
    if expression_results:
        summary["expression"] = summarize_expression_metrics(slides, expression_results)
    return summary
