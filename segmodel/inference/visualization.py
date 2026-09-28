from pathlib import Path

import cv2
import numpy as np


def _write_image(path, image, parameters=()):
    if not cv2.imwrite(str(path), image, list(parameters)):
        raise OSError(f"failed to write image: {path}")


def _uint16_probability(probability):
    values = np.nan_to_num(probability, nan=0.5, posinf=1.0, neginf=0.0)
    return np.rint(np.clip(values, 0.0, 1.0) * 65535).astype(np.uint16)


def _blend(image, selected, color, alpha=0.45):
    image[selected] = np.rint(
        image[selected].astype(np.float32) * (1.0 - alpha)
        + np.asarray(color, dtype=np.float32) * alpha
    ).astype(np.uint8)


def _mask_overlay(image_bgr, mask, region):
    """在 region 内按 0/1 涂色：蓝=非癌，红=癌，其余标签（255）保持 H&E。"""
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
        image,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    top = 36 + (available_height - resized.shape[0]) // 2
    left = (size - resized.shape[1]) // 2
    canvas[top : top + resized.shape[0], left : left + resized.shape[1]] = resized
    cv2.putText(
        canvas,
        title,
        (12, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    return canvas


def save_prediction_result(
    directory,
    image_rgb,
    probability,
    labels,
    valid,
    scale,
    threshold=0.5,
):
    """保存一个 epoch 的留出切片预测可视化，全部落在标签网格上。"""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    size = (
        round(probability.shape[1] / scale),
        round(probability.shape[0] / scale),
    )
    down_probability = cv2.resize(probability, size, interpolation=cv2.INTER_AREA)
    down_image = cv2.cvtColor(
        cv2.resize(image_rgb, size, interpolation=cv2.INTER_AREA),
        cv2.COLOR_RGB2BGR,
    )
    down_labels = cv2.resize(labels, size, interpolation=cv2.INTER_NEAREST)
    down_valid = (
        cv2.resize(valid.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
    )

    prediction = ((down_probability >= threshold) & down_valid).astype(np.uint8)
    prediction_overlay = _mask_overlay(down_image, prediction, down_valid)
    expert_overlay = _mask_overlay(down_image, down_labels, down_valid)

    png_parameters = (cv2.IMWRITE_PNG_COMPRESSION, 3)
    jpg_parameters = (cv2.IMWRITE_JPEG_QUALITY, 90)
    _write_image(
        directory / "probability.png",
        _uint16_probability(down_probability),
        png_parameters,
    )
    _write_image(directory / "mask_overlay.jpg", prediction_overlay, jpg_parameters)
    overview = np.hstack(
        (
            _overview_panel(down_image, "H&E"),
            _overview_panel(prediction_overlay, "Prediction"),
            _overview_panel(expert_overlay, "Expert label"),
        )
    )
    _write_image(directory / "overview.jpg", overview, jpg_parameters)
