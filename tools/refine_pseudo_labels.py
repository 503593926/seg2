"""引导滤波精修伪标签场: 精修前后按专家 mask 量化对照。

输入是标签网格上的概率场 (piecewise-constant, 每个超像素一个值) 与同一网格的
H&E 引导图; 引导滤波把块内取值按图像边缘重新分配, 然后与精修前一起用
compare_pseudo_labels 的口径评估 (覆盖率 / Dice / 边界带 Dice)。

用法:
  python tools/refine_pseudo_labels.py --slide B1 \
      --source prepare=data/her2st/1mpp/label_mpp/B1/pseudo_label_ratio@100 \
      --source soft=data/.../soft_labels/epoch_0030/B1
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.compare_pseudo_labels import expert_targets, find_files, read_probability
from tools.snic_features import prepare_snic_lab
from tools.propagate_pixel_vs_superpixel import evaluate


def box_filter(values, radius):
    return cv2.boxFilter(
        values.astype(np.float32),
        -1,
        (2 * radius + 1, 2 * radius + 1),
        normalize=True,
        borderType=cv2.BORDER_REFLECT,
    )


def guided_refine(coarse, guidance, radius, eps):
    """引导滤波 (He et al. 2010): 窗口内 output ≈ a·guidance + b, 最小二乘拟合。"""
    mean_i = box_filter(guidance, radius)
    mean_p = box_filter(coarse, radius)
    var_i = box_filter(guidance * guidance, radius) - mean_i**2
    cov_ip = box_filter(guidance * coarse, radius) - mean_i * mean_p
    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i
    return np.clip(box_filter(a, radius) * guidance + box_filter(b, radius), 0.0, 1.0)


def metrics(values, tissue, fraction, valid_count, thresholds, min_pixels=4):
    """覆盖率 / Dice / 边界带 Dice / 边界带错误率 (与 compare_pseudo_labels 同口径)。"""
    low, high = thresholds
    confident = ((values <= low) | (values >= high)) & tissue
    hard = np.full(values.shape, 255, dtype=np.int64)
    hard[confident] = (values[confident] >= 0.5).astype(np.int64)
    cell_valid = valid_count >= min_pixels
    covered = cell_valid & (hard != 255)
    truth = (fraction >= 0.5) & cell_valid
    kernel = np.ones((5, 5), dtype=np.uint8)
    band = (
        cv2.dilate(truth.astype(np.uint8), kernel)
        - cv2.erode(truth.astype(np.uint8), kernel)
    ).astype(bool)
    result = {"coverage": float(covered.sum() / max(cell_valid.sum(), 1))}
    for tag, selected in (("dice", covered), ("band_dice", covered & band)):
        if selected.any():
            prediction = hard[selected] == 1
            target = truth[selected]
            result[tag] = float(
                2 * (prediction & target).sum() / max(prediction.sum() + target.sum(), 1)
            )
            result[f"{tag}_error"] = float((prediction != target).mean())
        else:
            result[tag] = float("nan")
            result[f"{tag}_error"] = float("nan")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/her2st/1mpp")
    parser.add_argument("--slide", default="B1")
    parser.add_argument("--label-scale", type=int, default=7)
    parser.add_argument("--thresholds", type=float, nargs=2, default=(0.2, 0.8))
    parser.add_argument("--radius", type=int, nargs="+", default=[3, 6, 10])
    parser.add_argument("--eps", type=float, nargs="+", default=[1e-4, 1e-3, 1e-2])
    parser.add_argument("--source", action="append", required=True, help="name=directory")
    arguments = parser.parse_args()

    root = Path(arguments.root)
    slide = arguments.slide

    tissue = (
        cv2.imread(
            str(root / "label_mpp" / slide / "spatial/tissue_mask.png"),
            cv2.IMREAD_GRAYSCALE,
        )
        > 0
    )
    image = cv2.imread(
        str(root / "samples" / slide / "spatial/he_1mpp.png"), cv2.IMREAD_COLOR
    )
    image = cv2.resize(
        image, (tissue.shape[1], tissue.shape[0]), interpolation=cv2.INTER_AREA
    )
    guidance = prepare_snic_lab(image)[..., 0].astype(np.float32)
    guidance = (guidance - guidance.min()) / max(
        float(guidance.max() - guidance.min()), 1e-6
    )

    fraction, valid_count, _ = expert_targets(str(root), slide, arguments.label_scale)
    print(
        f"[{slide}] 引导图 L 通道局部方差 ~ {float(np.var(guidance[:64, :64])):.2e}, "
        f"阈值 {arguments.thresholds}"
    )
    print(
        f"{'source':<20} {'refine':<16} {'coverage':>9} {'dice':>7} "
        f"{'band_dice':>10} {'band_err':>9}"
    )

    for item in arguments.source:
        name, directory = item.split("=", 1)
        _, probability_path = find_files(directory)
        field = read_probability(probability_path)
        if field.shape != tissue.shape:
            raise ValueError(f"{directory}: {field.shape} vs {tissue.shape}")
        rows = [("none", field)]
        for radius in arguments.radius:
            for eps in arguments.eps:
                rows.append(
                    (
                        f"r={radius},eps={eps:g}",
                        guided_refine(field, guidance, radius, eps),
                    )
                )
        for tag, values in rows:
            result = metrics(
                values, tissue, fraction, valid_count, arguments.thresholds
            )
            print(
                f"{name:<20} {tag:<16} {result['coverage']:>9.3f} {result['dice']:>7.4f} "
                f"{result['band_dice']:>10.4f} {result['band_dice_error']:>9.4f}"
            )
        print()


if __name__ == "__main__":
    main()
