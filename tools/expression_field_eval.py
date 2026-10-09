"""表达场评估（三层判据第 1 层）: 重建是否超出简单插值的信息。

固定隐藏划分（按超像素整组隐藏, 两类各保留一个 anchor）, 比较:
  - RW 重建（与训练同一算子: (MᵀM + smoothing·L + ridge·I) f = Mᵀx + ridge·prior）
  - 均值基线（anchor 逐基因均值）
  - 最近 spot
  - kNN 加权（k=3, 距离倒数权重）
指标: 逐基因 PCC（log1p 空间）、相对均值基线的 MSE 改善、正相关基因占比。

用法:
  python tools/expression_field_eval.py --slide B1 --beta 0.0
  python tools/expression_field_eval.py --slide B1 --beta 0.5
"""

import argparse
import sys
from pathlib import Path

import anndata as ad
import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from segmodel.modules.graph_measurements import (
    reconstruct_expression,
    select_measurements,
    spot_measurements,
    spot_node_ids,
)
from tools.propagate_pixel_vs_superpixel import superpixel_edges
from tools.snic_features import prepare_snic_lab


def node_features(lab, feat, superpixels, nodes, learned_weight=1.0):
    """与模块同口径: 逐通道节点均值 + z-score; learned 缩放 sqrt(3w/C)。"""
    flat = superpixels.reshape(-1).astype(np.int64)
    nmax = int(superpixels.max()) + 1
    counts = np.bincount(flat, minlength=nmax).astype(np.float64)

    def aggregate(values):
        stacked = values.reshape(-1, values.shape[-1])
        means = np.zeros((nmax, stacked.shape[1]))
        for channel in range(stacked.shape[1]):
            means[:, channel] = np.bincount(
                flat, weights=stacked[:, channel].astype(np.float64), minlength=nmax
            ) / np.maximum(counts, 1)
        subset = means[nodes]
        return (subset - subset.mean(0)) / np.maximum(subset.std(0), 1e-6)

    color = aggregate(lab)
    learned = aggregate(feat) * np.sqrt(3.0 * learned_weight / feat.shape[-1])
    return np.concatenate((color, learned), axis=-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/her2st/1mpp")
    parser.add_argument("--slide", default="B1")
    parser.add_argument("--fold-reference", default="C1")
    parser.add_argument("--label-scale", type=int, default=7)
    parser.add_argument("--target-mpp", type=float, default=1.0)
    parser.add_argument("--spot-diameter-um", type=float, default=100.0)
    parser.add_argument("--features", default="feature_ho0")
    parser.add_argument(
        "--patch-map", default="tile_224_mpp_1_resize_1_overlap_1_cls_overlap_1"
    )
    parser.add_argument("--feature-channels", type=int, default=32)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--query-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoothing", type=float, default=1.0)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--steps", type=int, default=85)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--genes", type=int, default=512)
    parser.add_argument("--neighbors", type=int, default=3)
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
        cv2.imread(
            str(root / "samples" / slide / "spatial/he_1mpp.png"), cv2.IMREAD_COLOR
        ),
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

    nodes = np.unique(superpixels)
    nodes = nodes[nodes > 0]
    lookup = -np.ones(int(superpixels.max()) + 1, dtype=np.int64)
    lookup[nodes] = np.arange(nodes.size)
    combined = node_features(lab, feat, superpixels, nodes)
    first, second = superpixel_edges(superpixels, lookup, tissue)
    difference = combined[first] - combined[second]
    if arguments.beta > 0:
        conductance = torch.from_numpy(
            np.maximum(np.exp(-arguments.beta * (difference**2).sum(1)), 1e-6)
        ).float()
    else:
        conductance = torch.ones(first.size, dtype=torch.float32)
    edges = torch.from_numpy(np.stack((first, second), 1).astype(np.int64))

    adata = ad.read_h5ad(
        root / "gene_expression" / "task1_loocv" / f"test_{arguments.fold_reference}.h5ad"
    )
    mask = adata.obs["name"].astype(str).to_numpy() == slide
    expression = torch.from_numpy(
        np.asarray(adata.X[mask].todense(), dtype=np.float32)[:, : arguments.genes]
    )
    obs = adata.obs[mask]
    # spot_measurements 期望"图像坐标 + label_scale"; 这里统一换成标签网格坐标与半径
    radius_label = (arguments.spot_diameter_um / (2 * arguments.target_mpp)) / scale
    spots = torch.from_numpy(
        np.stack(
            (
                np.rint(obs["pixel_x"].to_numpy() / scale),
                np.rint(obs["pixel_y"].to_numpy() / scale),
            ),
            1,
        ).astype(np.float32)
    )
    labels = torch.from_numpy(obs["spot_label"].to_numpy().astype(np.int64))
    print(
        f"[{slide}] spot {spots.shape[0]}, 基因 {expression.shape[1]}, 节点 {nodes.size}, "
        f"边 {first.size}, β={arguments.beta}, spot 半径 {radius_label:.1f} 格"
    )

    grid = torch.from_numpy(superpixels)
    node_ids = spot_node_ids(grid, spots, 1)
    matrix = spot_measurements(
        grid, spots, torch.from_numpy(nodes), float(radius_label), 1
    )
    nz = matrix.nonzero(as_tuple=True)
    sparse = torch.sparse_coo_tensor(torch.stack(nz), matrix[nz], matrix.shape).coalesce()

    generator = torch.Generator().manual_seed(arguments.seed)
    eligible = node_ids > 0
    groups = torch.unique(node_ids[eligible]).tolist()
    protected = set()
    for label in (0, 1):
        candidates = set(node_ids[eligible & (labels == label)].tolist())
        chosen = next((int(g) for g in groups if g in candidates), None)
        if chosen is not None:
            protected.add(chosen)
    candidates = [int(g) for g in groups if g not in protected]
    order = torch.randperm(len(candidates), generator=generator).tolist()
    candidates = [candidates[i] for i in order]
    count = min(len(candidates), max(1, round(len(groups) * arguments.query_fraction)))
    query_ids = set(candidates[:count])
    query = torch.tensor([int(v) in query_ids for v in node_ids.tolist()]) & eligible
    anchor = eligible & ~query & (labels != 255)
    print(f"  隐藏 {int(query.sum())} spot / anchor {int(anchor.sum())} spot")

    anchor_matrix = select_measurements(sparse, anchor)
    query_matrix = select_measurements(sparse, query)
    linear = expression[anchor].expm1()
    prior = linear.mean(0, keepdim=True).expand(nodes.size, -1).contiguous()
    field, residual = reconstruct_expression(
        conductance,
        edges,
        anchor_matrix,
        linear,
        prior,
        node_groups=torch.zeros(nodes.size, dtype=torch.long),
        smoothing=arguments.smoothing,
        ridge=arguments.ridge,
        steps=arguments.steps,
        tolerance=arguments.tolerance,
    )
    rw_prediction = torch.sparse.mm(query_matrix, field).log1p()
    print(f"  RW 相对残差 {float(residual.max()):.2e}")

    truth = expression[query]
    mean_prediction = expression[anchor].mean(0, keepdim=True).expand(truth.shape[0], -1)
    distance = torch.cdist(spots[query], spots[anchor])
    nearest_prediction = expression[anchor][distance.argmin(1)]
    k = min(arguments.neighbors, anchor.sum())
    values, index = distance.topk(k, largest=False)
    weight = 1.0 / values.clamp_min(1e-6)
    knn_prediction = (
        (expression[anchor][index] * weight[:, :, None]).sum(1)
        / weight.sum(1, keepdim=True)
    )

    print(f"\n{'方法':<10} {'逐基因PCC均值':>13} {'PCC中位':>9} {'MSE':>9} {'相对均值改善':>12} {'正相关占比':>10}")
    reference_mse = None
    for tag, prediction in (
        ("均值", mean_prediction),
        ("最近 spot", nearest_prediction),
        ("kNN", knn_prediction),
        ("RW", rw_prediction),
    ):
        mse = float(((prediction - truth) ** 2).mean())
        if reference_mse is None:
            reference_mse = mse
        pcc = []
        for gene in range(truth.shape[1]):
            a, b = prediction[:, gene], truth[:, gene]
            if float(a.std()) < 1e-6 or float(b.std()) < 1e-6:
                continue
            pcc.append(float(torch.corrcoef(torch.stack((a, b)))[0, 1]))
        pcc = np.array(pcc) if pcc else np.array([float("nan")])
        print(
            f"{tag:<10} {pcc.mean():>13.4f} {np.median(pcc):>9.4f} {mse:>9.4f} "
            f"{1 - mse / reference_mse:>12.4f} {float((pcc > 0).mean()):>10.3f}"
        )


if __name__ == "__main__":
    main()
