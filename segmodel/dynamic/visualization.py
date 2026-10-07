import json
from pathlib import Path

import cv2
import numpy as np


def _write_image(path, image, parameters=()):
    if not cv2.imwrite(str(path), image, list(parameters)):
        raise OSError(f"failed to write image: {path}")


def _uint16_probability(probability):
    values = np.nan_to_num(probability, nan=0.5, posinf=1.0, neginf=0.0)
    return np.rint(np.clip(values, 0.0, 1.0) * 65535).astype(np.uint16)


def _probability_heatmap(probability, tissue):
    values = (_uint16_probability(probability) / 257).astype(np.uint8)
    heatmap = cv2.applyColorMap(values, cv2.COLORMAP_TURBO)
    heatmap[~tissue] = 255
    return heatmap


def _blend(image, selected, color, alpha=0.45):
    image[selected] = np.rint(
        image[selected].astype(np.float32) * (1.0 - alpha)
        + np.asarray(color, dtype=np.float32) * alpha
    ).astype(np.uint8)


def _pseudo_mask_overlay(image_bgr, pseudo_mask_uint8):
    overlay = image_bgr.copy()
    _blend(overlay, pseudo_mask_uint8 == 0, (255, 0, 0))
    _blend(overlay, pseudo_mask_uint8 == 1, (0, 0, 255))
    return overlay


def _change_overlay(image_bgr, previous_mask, candidate_mask):
    overlay = image_bgr.copy()
    changes = {
        "added_cancer": (previous_mask != 1) & (candidate_mask == 1),
        "removed_cancer": (previous_mask == 1) & (candidate_mask != 1),
        "added_non_cancer": (previous_mask == 255) & (candidate_mask == 0),
        "removed_non_cancer": (previous_mask == 0) & (candidate_mask == 255),
    }
    colors = {
        "added_cancer": (0, 0, 255),
        "removed_cancer": (255, 0, 0),
        "added_non_cancer": (0, 180, 0),
        "removed_non_cancer": (0, 220, 255),
    }
    for name, selected in changes.items():
        _blend(overlay, selected, colors[name], alpha=0.65)
    return overlay, {name: int(selected.sum()) for name, selected in changes.items()}


def _superpixel_overlay(image_bgr, superpixels):
    boundary = np.zeros(superpixels.shape, dtype=bool)
    boundary[1:] |= superpixels[1:] != superpixels[:-1]
    boundary[:-1] |= superpixels[:-1] != superpixels[1:]
    boundary[:, 1:] |= superpixels[:, 1:] != superpixels[:, :-1]
    boundary[:, :-1] |= superpixels[:, :-1] != superpixels[:, 1:]
    boundary &= superpixels > 0
    overlay = image_bgr.copy()
    overlay[boundary] = (0, 255, 255)
    return overlay


def _learned_feature_norm(learned, tissue):
    values = np.nan_to_num(
        np.asarray(learned, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    reference = values[tissue]
    if not len(reference):
        return np.full((*tissue.shape, 3), 255, dtype=np.uint8)
    mean = reference.mean(axis=0)
    scale = np.maximum(reference.std(axis=0), 1e-6)
    norm = np.sqrt(np.square((values - mean) / scale).mean(axis=-1))
    low, high = np.percentile(norm[tissue], (1, 99))
    normalized = np.clip((norm - low) / max(float(high - low), 1e-6), 0.0, 1.0)
    heatmap = cv2.applyColorMap(
        np.rint(normalized * 255).astype(np.uint8),
        cv2.COLORMAP_VIRIDIS,
    )
    heatmap[~tissue] = 255
    return heatmap


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


def _probability_statistics(probability, tissue):
    finite = np.isfinite(probability) & tissue
    result = {"finite_fraction": float(finite.sum() / max(int(tissue.sum()), 1))}
    if finite.any():
        values = probability[finite]
        result.update(
            mean=float(values.mean()),
            p10=float(np.percentile(values, 10)),
            p50=float(np.percentile(values, 50)),
            p90=float(np.percentile(values, 90)),
        )
    return result


def _class_counts(mask):
    return {str(value): int((mask == value).sum()) for value in (0, 1, 255)}


def save_dynamic_result(
    output_dir,
    image_rgb,
    tissue,
    model_probability,
    propagated_probability,
    candidate_mask,
    previous_mask,
    superpixels,
    learned,
    metadata,
):
    """保存一张切片的一轮动态传播候选结果和诊断可视化。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tissue = np.asarray(tissue, dtype=bool)
    image_bgr = cv2.cvtColor(np.asarray(image_rgb), cv2.COLOR_RGB2BGR)

    model_heatmap = _probability_heatmap(model_probability, tissue)
    propagated_heatmap = _probability_heatmap(propagated_probability, tissue)
    pseudo_overlay = _pseudo_mask_overlay(image_bgr, candidate_mask)
    change_overlay, change_counts = _change_overlay(
        image_bgr,
        previous_mask,
        candidate_mask,
    )
    superpixel_overlay = _superpixel_overlay(image_bgr, superpixels)
    feature_norm = _learned_feature_norm(learned, tissue)

    png_parameters = (cv2.IMWRITE_PNG_COMPRESSION, 3)
    jpg_parameters = (cv2.IMWRITE_JPEG_QUALITY, 90)
    _write_image(
        output_dir / "model_probability.png",
        _uint16_probability(model_probability),
        png_parameters,
    )
    _write_image(
        output_dir / "candidate_probability.png",
        _uint16_probability(propagated_probability),
        png_parameters,
    )
    _write_image(
        output_dir / "candidate_pseudo_mask.png",
        np.asarray(candidate_mask, dtype=np.uint8),
        png_parameters,
    )
    _write_image(
        output_dir / "model_probability_heatmap.png",
        model_heatmap,
        png_parameters,
    )
    _write_image(
        output_dir / "propagated_probability_heatmap.png",
        propagated_heatmap,
        png_parameters,
    )
    _write_image(
        output_dir / "candidate_pseudo_mask_overlay.jpg",
        pseudo_overlay,
        jpg_parameters,
    )
    _write_image(
        output_dir / "change_from_previous_overlay.jpg",
        change_overlay,
        jpg_parameters,
    )
    _write_image(
        output_dir / "superpixel_boundary_overlay.jpg",
        superpixel_overlay,
        jpg_parameters,
    )
    _write_image(
        output_dir / "learned_feature_norm.png",
        feature_norm,
        png_parameters,
    )

    overview = np.vstack(
        (
            np.hstack(
                (
                    _overview_panel(image_bgr, "H&E"),
                    _overview_panel(model_heatmap, "Model probability"),
                    _overview_panel(propagated_heatmap, "Propagated probability"),
                )
            ),
            np.hstack(
                (
                    _overview_panel(pseudo_overlay, "Candidate pseudo label"),
                    _overview_panel(change_overlay, "Change from previous"),
                    _overview_panel(superpixel_overlay, "Superpixel boundaries"),
                )
            ),
        )
    )
    _write_image(output_dir / "overview.jpg", overview, jpg_parameters)

    record = {
        **metadata,
        "shape": list(candidate_mask.shape),
        "candidate_class_counts": _class_counts(candidate_mask),
        "previous_class_counts": _class_counts(previous_mask),
        "change_counts": change_counts,
        "model_probability": _probability_statistics(model_probability, tissue),
        "propagated_probability": _probability_statistics(
            propagated_probability,
            tissue,
        ),
        "files": {
            "probability_encoding": "uint16_png; probability = value / 65535",
            "pseudo_mask_encoding": {
                "0": "non_cancer",
                "1": "cancer",
                "255": "ignored",
            },
            "pseudo_overlay_colors": {"blue": "non_cancer", "red": "cancer"},
            "change_overlay_colors": {
                "red": "added_cancer",
                "blue": "removed_cancer",
                "green": "added_non_cancer",
                "yellow": "removed_non_cancer",
            },
        },
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
