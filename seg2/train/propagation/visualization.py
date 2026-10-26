import json
from pathlib import Path

import cv2
import numpy as np


def _write_image(path, image, parameters=()):
    cv2.imwrite(str(path), image, list(parameters))


def _blend(image, selected, color, alpha=0.45):
    image[selected] = np.rint(
        image[selected].astype(np.float32) * (1.0 - alpha) + np.asarray(color, dtype=np.float32) * alpha
    ).astype(np.uint8)


def _mask_overlay(image_bgr, mask, region):
    overlay = image_bgr.copy()
    _blend(overlay, region & (mask == 0), (255, 0, 0))
    _blend(overlay, region & (mask == 1), (0, 0, 255))
    return overlay


def _overview_panel(image, title, size=512):
    canvas = np.full((size, size, 3), 245, dtype=np.uint8)
    available_height = size - 36
    height, width = image.shape[:2]
    scale = min(size / width, available_height / height)
    resized = cv2.resize(
        image, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv2.INTER_AREA
    )
    top = 36 + (available_height - resized.shape[0]) // 2
    left = (size - resized.shape[1]) // 2
    canvas[top : top + resized.shape[0], left : left + resized.shape[1]] = resized
    cv2.putText(canvas, title, (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 1, cv2.LINE_AA)
    return canvas


def save_soft_label_result(
    directory,
    image_rgb,
    tissue,
    probability,
    mask,
    spot_labels,
    *,
    non_cancer_threshold,
    cancer_threshold,
    metadata,
):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    pseudo_overlay = _mask_overlay(image_bgr, mask, tissue)
    jpg = (cv2.IMWRITE_JPEG_QUALITY, 90)
    overview = _overview_panel(pseudo_overlay, "RW pseudo label")
    _write_image(directory / "overview.jpg", overview, jpg)
    values = probability[tissue]
    record = {
        **metadata,
        "shape": list(tissue.shape),
        "supervised_fraction": float((mask != 255).sum() / max(int(tissue.sum()), 1)),
        "soft_class_counts": {str(label): int((mask == label).sum()) for label in (0, 1, 255)},
        "seed_counts": {
            "non_cancer": int((spot_labels == 0).sum()),
            "cancer": int((spot_labels == 1).sum()),
            "unknown": int((spot_labels == 255).sum()),
        },
        "soft_probability": {
            "finite_fraction": float(np.isfinite(values).mean()),
            "mean": float(values.mean()),
            **{f"p{percentile}": float(np.percentile(values, percentile)) for percentile in (10, 50, 90)},
        },
        "thresholds": {"non_cancer": float(non_cancer_threshold), "cancer": float(cancer_threshold)},
        "overlay_colors": {"blue": "non_cancer", "red": "cancer"},
        "files": {"overview": "overview.jpg"},
    }
    (directory / "metadata.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return record
