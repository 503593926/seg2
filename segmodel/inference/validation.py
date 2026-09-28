from pathlib import Path

import cv2
import numpy as np

from ..dataset.slide import read_image
from .metrics import segmentation_metrics
from .predictor import predict_slide


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
):
    probability, _ = predict_slide(
        model,
        slide,
        crop_size=crop_size,
        stride=stride,
        alignment=alignment,
        device=device,
        batch_size=batch_size,
        return_features=False,
    )
    labels, valid = load_expert_masks(root, slide)
    metrics = segmentation_metrics(
        probability,
        labels,
        valid,
        threshold=threshold,
        mpp=mpp,
    )
    return probability, metrics
