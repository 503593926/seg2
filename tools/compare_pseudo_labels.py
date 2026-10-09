"""用专家 mask 评估伪标签质量, 对齐比较不同来源 (prepare / 动态传播 / 可微传播)。

评估在标签网格上进行: 专家 mask 按 label_scale 分块聚合, 伪标签直接读标签网格,
两者网格天然对齐 (slide 尺寸能被 label_scale 整除)。

指标:
- coverage: 专家有效格子里, 伪标签给出 0/1 (非 255) 的比例
- dice/error: 覆盖格子上的硬标签与专家多数票的一致性
- brier/ece: 连续概率与专家阳性比例的校准误差 (全有效格子)

用法:
  python tools/compare_pseudo_labels.py --root data/her2st/1mpp --slide B1 \
      --source prepare=data/her2st/1mpp/label_mpp/B1/pseudo_label_ratio@100 \
      --source dynamic=outputs/.../dynamic_pseudo_labels \
      --source soft=outputs/.../soft_labels
带 epoch_XXXX 子目录的来源会自动展开 (只评估包含该切片的轮次)。
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def read_gray(path):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    return image


def read_probability(path):
    """概率图编码不定: uint16 时 value/65535, uint8 时 value/255。"""
    image = read_gray(path)
    if image.dtype == np.uint16:
        return image.astype(np.float64) / 65535.0
    if image.dtype == np.uint8:
        return image.astype(np.float64) / 255.0
    return image.astype(np.float64)


def block_sum(values, scale):
    height = values.shape[0] // scale * scale
    width = values.shape[1] // scale * scale
    cropped = values[:height, :width].astype(np.float64)
    return cropped.reshape(height // scale, scale, width // scale, scale).sum(
        axis=(1, 3)
    )


def expert_targets(root, slide, label_scale):
    """返回标签网格上的 (阳性比例, 有效格子, 阳性像素数, 有效像素数)。"""
    sample = Path(root) / "samples" / slide
    labels = read_gray(sample / "masks/training_mask.png")
    valid = read_gray(sample / "masks/valid_mask.png") > 0
    canvas_path = sample / "masks/canvas_mask.png"
    if canvas_path.exists():
        valid &= read_gray(canvas_path) > 0
    valid &= labels != 255
    positive = (labels == 1) & valid
    valid_count = block_sum(valid, label_scale)
    positive_count = block_sum(positive, label_scale)
    fraction = positive_count / np.maximum(valid_count, 1)
    return fraction, valid_count, positive_count


def find_files(directory):
    """在目录里找伪标签 mask 与概率图, 排除对照/热图/QC 产物。"""
    mask = None
    for path in sorted(Path(directory).glob("*pseudo_mask.png")):
        if "overlay" in path.name or "qc" in path.name:
            continue
        mask = path
        break
    probability = None
    for path in sorted(Path(directory).glob("*probability.png")):
        if "model" in path.name or "heatmap" in path.name:
            continue
        probability = path
        break
    return mask, probability


def evaluate(directory, fraction, valid_count, min_pixels, bins=10, band=2):
    mask_path, probability_path = find_files(directory)
    if mask_path is None or probability_path is None:
        raise FileNotFoundError(f"no pseudo label files in {directory}")
    hard = read_gray(mask_path).astype(np.int64)
    probability = read_probability(probability_path)
    if hard.shape != fraction.shape or probability.shape != fraction.shape:
        raise ValueError(
            f"{directory}: shape {hard.shape}/{probability.shape} "
            f"does not match expert grid {fraction.shape}"
        )

    cell_valid = valid_count >= min_pixels
    covered = cell_valid & (hard != 255)
    truth = (fraction >= 0.5) & cell_valid
    record = {
        "directory": str(directory),
        "cells": int(cell_valid.sum()),
        "coverage": float(covered.sum() / max(cell_valid.sum(), 1)),
    }
    if covered.any():
        prediction = hard[covered] == 1
        target = truth[covered]
        intersection = float((prediction & target).sum())
        record["dice"] = float(
            2 * intersection / max(prediction.sum() + target.sum(), 1)
        )
        record["error"] = float((prediction != target).mean())
        record["confident_wrong"] = float(
            ((prediction != target) & ((hard[covered] == 0) | (hard[covered] == 1))).mean()
        )
    else:
        record.update(dice=float("nan"), error=float("nan"), confident_wrong=float("nan"))

    # 边界带: 专家多数票区域的膨胀-腐蚀差值, 是分割误差集中的地方
    kernel = np.ones((2 * band + 1, 2 * band + 1), dtype=np.uint8)
    mask_uint8 = truth.astype(np.uint8)
    band_mask = (
        cv2.dilate(mask_uint8, kernel) - cv2.erode(mask_uint8, kernel)
    ).astype(bool)
    in_band = covered & band_mask
    record["band_cells"] = int(in_band.sum())
    if in_band.any():
        prediction = hard[in_band] == 1
        target = truth[in_band]
        intersection = float((prediction & target).sum())
        record["band_dice"] = float(
            2 * intersection / max(prediction.sum() + target.sum(), 1)
        )
        record["band_error"] = float((prediction != target).mean())
    else:
        record.update(band_dice=float("nan"), band_error=float("nan"))

    if cell_valid.any():
        values = np.clip(probability[cell_valid], 0.0, 1.0)
        target = fraction[cell_valid]
        record["brier"] = float(np.mean((values - target) ** 2))
        edges = np.linspace(0.0, 1.0, bins + 1)
        index = np.clip(np.digitize(values, edges) - 1, 0, bins - 1)
        error = 0.0
        total = values.size
        for bin_index in range(bins):
            selected = index == bin_index
            if not selected.any():
                continue
            error += (
                selected.sum()
                / total
                * abs(values[selected].mean() - target[selected].mean())
            )
        record["ece"] = float(error)
    else:
        record.update(brier=float("nan"), ece=float("nan"))
    return record


def expand_sources(source, slide):
    """name=dir: 目录含 epoch_XXXX 子目录时按轮次展开。"""
    results = []
    for name, directory in source:
        directory = Path(directory)
        epochs = sorted(
            path for path in directory.glob("epoch_*") if (path / slide).is_dir()
        )
        if epochs:
            for path in epochs:
                results.append((f"{name}/{path.name}", path / slide))
        else:
            results.append((name, directory))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/her2st/1mpp")
    parser.add_argument("--slide", default="B1")
    parser.add_argument("--label-scale", type=int, default=7)
    parser.add_argument(
        "--min-pixels",
        type=int,
        default=4,
        help="一个标签格子里至少要有的专家有效像素数",
    )
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        help="name=directory, 可重复",
    )
    parser.add_argument("--json", default=None)
    arguments = parser.parse_args()

    sources = [item.split("=", 1) for item in arguments.source]
    fraction, valid_count, _ = expert_targets(
        arguments.root, arguments.slide, arguments.label_scale
    )
    print(
        f"expert {arguments.slide}: 有效格子 {int((valid_count >= arguments.min_pixels).sum())} "
        f"/ {valid_count.size}, 阳性格子占比 "
        f"{float(((fraction >= 0.5) & (valid_count >= arguments.min_pixels)).sum() / max((valid_count >= arguments.min_pixels).sum(), 1)):.3f}"
    )
    print(f"{'source':<26} {'coverage':>9} {'dice':>7} {'error':>7} {'band_dice':>9} {'band_err':>8} {'brier':>8} {'ece':>7}")
    records = []
    for name, directory in expand_sources(sources, arguments.slide):
        record = evaluate(
            directory, fraction, valid_count, arguments.min_pixels
        )
        record["name"] = name
        records.append(record)
        print(
            f"{name:<26} {record['coverage']:>9.3f} {record['dice']:>7.4f} "
            f"{record['error']:>7.4f} {record['band_dice']:>9.4f} "
            f"{record['band_error']:>8.4f} {record['brier']:>8.5f} {record['ece']:>7.4f}"
        )
    if arguments.json:
        Path(arguments.json).write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
