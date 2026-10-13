import json
from pathlib import Path

import cv2
import numpy as np
from skimage.morphology import thin
from skimage.segmentation import find_boundaries


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
    initial_probability=None,
    refinement_gate=None,
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
    if initial_probability is not None:
        save_prediction_result(directory / "initial", image_rgb, initial_probability, labels, valid, scale, threshold)
    if refinement_gate is not None:
        _write_image(directory / "refinement_gate.png", _uint16_probability(refinement_gate), png_parameters)


def _superpixel_overlay(image_bgr, superpixels, label_mpp):
    """与静态 SNIC QC 一致：红色边界，低分辨率标签网格使用细线。"""
    boundary = find_boundaries(superpixels, connectivity=2, mode="thick")
    boundary &= superpixels > 0
    if label_mpp >= 4.0:
        boundary = thin(boundary)
    overlay = image_bgr.copy()
    overlay[boundary] = (0, 0, 255)
    return overlay


def _probability_heatmap(probability, tissue):
    values = (_uint16_probability(probability) / 257).astype(np.uint8)
    heatmap = cv2.applyColorMap(values, cv2.COLORMAP_TURBO)
    heatmap[~tissue] = 255
    return heatmap


def save_soft_label_result(
    directory, image_rgb, tissue, probability, supervised, superpixels, spots, spot_labels,
    *, model_probability, non_cancer_threshold, cancer_threshold, metadata,
):
    """保存新 SNIC 上的 RW 场及红色细线边界；不读取评估像素标注。"""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    mask = np.full(tissue.shape, 255, dtype=np.uint8)
    mask[tissue & supervised & (probability <= non_cancer_threshold)] = 0
    mask[tissue & supervised & (probability >= cancer_threshold)] = 1
    pseudo_overlay = _mask_overlay(image_bgr, mask, tissue)
    snic_overlay = _superpixel_overlay(image_bgr, superpixels, float(metadata["label_mpp"]))
    heatmap = _probability_heatmap(probability, tissue)
    model_heatmap = _probability_heatmap(model_probability, tissue)
    seed_overlay = image_bgr.copy()
    height, width = tissue.shape
    for (x, y), label in zip(np.rint(spots).astype(np.int64), spot_labels):
        if label in (0, 1) and 0 <= x < width and 0 <= y < height:
            cv2.circle(seed_overlay, (int(x), int(y)), 3, (255, 0, 0) if label == 0 else (0, 0, 255), -1)

    # 固定局部随机源，不改变训练的 NumPy/Torch 随机状态。
    colors = np.random.default_rng(0).integers(32, 224, size=(int(superpixels.max()) + 1, 3), dtype=np.uint8)
    colors[0] = 255
    png = (cv2.IMWRITE_PNG_COMPRESSION, 3)
    jpg = (cv2.IMWRITE_JPEG_QUALITY, 90)
    images = {
        "soft_probability.png": _uint16_probability(probability),
        "soft_probability_heatmap.png": heatmap,
        "soft_pseudo_mask.png": mask,
        "supervised_mask.png": (mask != 255).astype(np.uint8) * 255,
        "superpixel_labels.png": colors[superpixels],
        "superpixel_boundary_overlay.png": snic_overlay,
        "model_probability.png": _uint16_probability(model_probability),
        "model_probability_heatmap.png": model_heatmap,
    }
    for filename, values in images.items():
        _write_image(directory / filename, values, png)
    for filename, values in {
        "soft_pseudo_mask_overlay.jpg": pseudo_overlay,
        "spot_seeds_overlay.jpg": seed_overlay,
    }.items():
        _write_image(directory / filename, values, jpg)
    panels = [
        (seed_overlay, "H&E + spot seeds"), (model_heatmap, "Model probability"),
        (heatmap, "RW probability"), (pseudo_overlay, "RW pseudo label"),
        (snic_overlay, "Dynamic SNIC boundaries"), (image_bgr, "H&E"),
    ]
    overview = np.vstack([
        np.hstack([_overview_panel(values, title) for values, title in panels[start:start + 3]])
        for start in (0, 3)
    ])
    _write_image(directory / "overview.jpg", overview, jpg)
    np.save(directory / "superpixels.npy", superpixels)
    np.save(directory / "soft_probability.npy", probability)
    values = probability[tissue]
    record = {
        **metadata, "shape": list(tissue.shape),
        "supervised_fraction": float((mask != 255).sum() / max(int(tissue.sum()), 1)),
        "soft_class_counts": {str(label): int((mask == label).sum()) for label in (0, 1, 255)},
        "seed_counts": {
            "non_cancer": int((spot_labels == 0).sum()), "cancer": int((spot_labels == 1).sum()),
            "unknown": int((spot_labels == 255).sum()),
        },
        "soft_probability": {
            "finite_fraction": float(np.isfinite(values).mean()), "mean": float(values.mean()),
            **{f"p{percentile}": float(np.percentile(values, percentile)) for percentile in (10, 50, 90)},
        },
        "thresholds": {"non_cancer": float(non_cancer_threshold), "cancer": float(cancer_threshold)},
        "files": {
            "probability_encoding": "uint16_png; probability = value / 65535; npy stores unquantized float32",
            "pseudo_mask_encoding": {"0": "non_cancer", "1": "cancer", "255": "fuzzy_or_outside_tissue"},
            "overlay_colors": {"blue": "non_cancer", "red": "cancer"},
            "snic_boundary_color": "red", "snic_boundary_file": "superpixel_boundary_overlay.png",
            "snic_boundary_thinned": float(metadata["label_mpp"]) >= 4.0,
            "superpixels_encoding": "int32_npy; 0=background, 1..K=nodes",
        },
    }
    (directory / "metadata.json").write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return record
