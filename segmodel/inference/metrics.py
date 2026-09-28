import math

import numpy as np
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
    prediction_surface = prediction & ~binary_erosion(prediction | ~valid)
    truth_surface = truth & ~binary_erosion(truth | ~valid)
    if not prediction_surface.any() or not truth_surface.any():
        return float("nan")
    # sampling=mpp：把距离从"像素数"换算成微米，HD95 因此带物理单位
    forward = distance_transform_edt(~truth_surface, sampling=mpp)[
        prediction_surface
    ]
    backward = distance_transform_edt(~prediction_surface, sampling=mpp)[
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


def summarize_metrics(slides):
    values = list(slides.values())
    summary = _summarize_head(values)
    summary["slides"] = slides
    return summary
