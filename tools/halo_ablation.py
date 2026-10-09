"""halo 对照: crop 内独立求解 vs 带上下文(halo)求解 vs 全片求解。

背景: 训练时每个 crop 独立解 Dirichlet 问题, 图只用 crop 内的超像素与种子,
crop 边缘因此缺少外部种子的锚定。本脚本在中心 128x128 格上评估三种求解域,
并在不同点密度(分层抽样)下比较覆盖率与边界带 Dice。

用法:
  python tools/halo_ablation.py --slide B1
"""

import argparse
import sys
from pathlib import Path

import anndata as ad
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.compare_pseudo_labels import expert_targets
from tools.propagate_pixel_vs_superpixel import seeds_on_grid, solve, superpixel_edges
from tools.snic_features import prepare_snic_lab

SIZE = 128
FULL_SLIDE = 4000


def solve_region(arrays, top, left, halo, spots, labels, label_scale):
    """在中心 128x128 外扩 halo 的区域求解, 只返回中心区的场。"""
    tissue, superpixels, lab, feat = arrays
    height, width = tissue.shape
    t0, l0 = max(top - halo, 0), max(left - halo, 0)
    t1, l1 = min(top + SIZE + halo, height), min(left + SIZE + halo, width)
    sp_window = superpixels[t0:t1, l0:l1]
    tissue_window = tissue[t0:t1, l0:l1]
    channels = [lab[t0:t1, l0:l1, c] for c in range(lab.shape[-1])] + [
        feat[t0:t1, l0:l1, c] for c in range(feat.shape[-1])
    ]

    nodes = np.unique(sp_window)
    nodes = nodes[nodes > 0]
    flat = sp_window.reshape(-1).astype(np.int64)
    nmax = int(sp_window.max()) + 1
    counts = np.bincount(flat, minlength=nmax).astype(np.float64)
    means = np.zeros((nmax, len(channels)))
    for index, values in enumerate(channels):
        means[:, index] = np.bincount(
            flat, weights=np.asarray(values, dtype=np.float64).reshape(-1), minlength=nmax
        ) / np.maximum(counts, 1)
    means = means[nodes]
    means = (means - means.mean(0)) / np.maximum(means.std(0), 1e-6)

    lookup = -np.ones(nmax, dtype=np.int64)
    lookup[nodes] = np.arange(nodes.size)
    first, second = superpixel_edges(sp_window, lookup, tissue_window)

    keep = (
        (spots[0] >= l0) & (spots[0] < l1) & (spots[1] >= t0) & (spots[1] < t1)
        & (labels != 255)
    )
    seed_grid = seeds_on_grid(
        spots[0][keep] - l0, spots[1][keep] - t0, labels[keep], tissue_window, 0
    )
    seed_values = np.full(nodes.size, 255.0)
    known = seed_grid != 255
    if not known.any():
        return None
    positive = np.bincount(
        lookup[sp_window[known]], weights=(seed_grid[known] == 1), minlength=nodes.size
    )
    negative = np.bincount(
        lookup[sp_window[known]], weights=(seed_grid[known] == 0), minlength=nodes.size
    )
    seed_values[positive > negative] = 1.0
    seed_values[negative > positive] = 0.0

    values, _, _, _ = solve(
        first, second, means[first] - means[second], seed_values, nodes.size,
        0.5, 400, 1e-8,
    )
    field = np.full(tissue_window.shape, 0.5)
    field[tissue_window] = values[lookup[sp_window[tissue_window]]]
    return field[top - t0 : top - t0 + SIZE, left - l0 : left - l0 + SIZE]


def hard_labels(field, thresholds=(0.2, 0.8)):
    hard = np.full(field.shape, 255, dtype=np.int64)
    confident = (field <= thresholds[0]) | (field >= thresholds[1])
    hard[confident] = (field[confident] >= 0.5).astype(np.int64)
    return hard


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/her2st/1mpp")
    parser.add_argument("--slide", default="B1")
    parser.add_argument("--label-scale", type=int, default=7)
    parser.add_argument("--features", default="feature_ho0")
    parser.add_argument(
        "--patch-map", default="tile_224_mpp_1_resize_1_overlap_1_cls_overlap_1"
    )
    parser.add_argument("--feature-channels", type=int, default=32)
    parser.add_argument("--halos", type=int, nargs="+", default=[0, 16, 32, 64])
    parser.add_argument("--rates", type=float, nargs="+", default=[1.0, 0.5, 0.25, 0.1])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--stride", type=int, default=64)
    arguments = parser.parse_args()

    root = Path(arguments.root)
    slide = arguments.slide
    scale = arguments.label_scale
    tissue = (
        cv2.imread(
            str(root / "label_mpp" / slide / "spatial/tissue_mask.png"),
            cv2.IMREAD_GRAYSCALE,
        )
        > 0
    )
    superpixels = np.load(root / "label_mpp" / slide / "snic/superpixels.npy").astype(
        np.int64
    )
    image = cv2.resize(
        cv2.imread(str(root / "samples" / slide / "spatial/he_1mpp.png"), cv2.IMREAD_COLOR),
        (tissue.shape[1], tissue.shape[0]),
        interpolation=cv2.INTER_AREA,
    )
    lab = prepare_snic_lab(image)
    patch_map = np.load(
        root / arguments.features / arguments.patch_map / f"{slide}/patch_map.npy",
        mmap_mode="r",
    )
    feat = np.asarray(
        cv2.resize(
            np.asarray(patch_map[..., : arguments.feature_channels], dtype=np.float32),
            (tissue.shape[1], tissue.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
    )
    fraction, valid_count, _ = expert_targets(str(root), slide, scale)

    obs = ad.read_h5ad(root / "gene_expression" / "task1_loocv" / "test_C1.h5ad").obs
    obs = obs[obs["name"].astype(str) == slide]
    spots = (
        np.rint(obs["pixel_x"].to_numpy() / scale).astype(np.int64),
        np.rint(obs["pixel_y"].to_numpy() / scale).astype(np.int64),
    )
    labels = obs["spot_label"].to_numpy().astype(np.int64)

    windows = []
    for top in range(0, tissue.shape[0] - SIZE, arguments.stride):
        for left in range(0, tissue.shape[1] - SIZE, arguments.stride):
            if tissue[top : top + SIZE, left : left + SIZE].sum() < 0.8 * SIZE * SIZE:
                continue
            keep = (
                (spots[0] >= left) & (spots[0] < left + SIZE)
                & (spots[1] >= top) & (spots[1] < top + SIZE)
                & (labels != 255)
            )
            if (labels[keep] == 1).sum() >= 3 and (labels[keep] == 0).sum() >= 3:
                windows.append((top, left))
    print(
        f"[{slide}] 窗口 {len(windows)}, spot {spots[0].size} "
        f"(阳性 {(labels == 1).sum()} / 阴性 {(labels == 0).sum()})"
    )

    def evaluate(subset, halo):
        coverages, bands = [], []
        for top, left in windows:
            field = solve_region(
                (tissue, superpixels, lab, feat), top, left, halo,
                (spots[0][subset], spots[1][subset]), labels[subset], scale,
            )
            if field is None:
                continue
            hard = hard_labels(field)
            cell_valid = valid_count[top : top + SIZE, left : left + SIZE] >= 4
            truth = (fraction[top : top + SIZE, left : left + SIZE] >= 0.5) & cell_valid
            covered = cell_valid & (hard != 255)
            coverages.append(float(covered.sum() / max(cell_valid.sum(), 1)))
            kernel = np.ones((5, 5), dtype=np.uint8)
            band = (
                cv2.dilate(truth.astype(np.uint8), kernel)
                - cv2.erode(truth.astype(np.uint8), kernel)
            ).astype(bool)
            selected = covered & band
            if selected.any():
                prediction, target = hard[selected] == 1, truth[selected]
                bands.append(
                    float(
                        2 * (prediction & target).sum()
                        / max(prediction.sum() + target.sum(), 1)
                    )
                )
        return (
            np.mean(coverages) if coverages else float("nan"),
            np.nanmean(bands) if bands else float("nan"),
        )

    rng = np.random.default_rng(7)
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    halos = list(arguments.halos) + [FULL_SLIDE]
    header = " ".join(f"{'全片' if h > 400 else h:>16}" for h in halos)
    print(f"{'保留率':>6} {header}")
    for rate in arguments.rates:
        accumulated = {h: ([], []) for h in halos}
        repeats = 1 if rate >= 1.0 else arguments.repeats
        for _ in range(repeats):
            keep_positive = max(2, int(positive.size * rate))
            keep_negative = max(2, int(negative.size * rate))
            subset = np.concatenate(
                (
                    rng.permutation(positive)[:keep_positive],
                    rng.permutation(negative)[:keep_negative],
                )
            )
            for h in halos:
                coverage, band = evaluate(subset, h)
                accumulated[h][0].append(coverage)
                accumulated[h][1].append(band)
        row = " ".join(
            f"{np.nanmean(accumulated[h][0]):>7.3f}/{np.nanmean(accumulated[h][1]):>8.4f}"
            for h in halos
        )
        print(f"{rate:>6.2f} {row}")


if __name__ == "__main__":
    main()
