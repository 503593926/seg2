"""离线对照: 同一张切片上, 超像素图传播 vs 像素图传播。

两者共用同一套输入 (Lab + 模型特征差构图, spot 真值做种子, 同一 beta / 阈值 /
PCG 求解器), 只换图的单位。用来回答: 去掉超像素直接按像素传播, 值不值。

评估口径与 tools/compare_pseudo_labels.py 一致: 组织内覆盖率 + 边界带 Dice。

用法:
  python tools/propagate_pixel_vs_superpixel.py --root data/her2st/1mpp --slide B1
  # 训练相关尺度: 128x128 标签格窗口 + 精确解 (直接法) 作参照
  python tools/propagate_pixel_vs_superpixel.py --slide B1 --window 128 --exact

注意: 像素图上 Jacobi 预条件 CG 收敛极慢 (全片 2000 步残差仍 1e-4), 未收敛时
初值 rhs=0 会把解整体拉向 0, 表现为"覆盖率正常但 dice 崩到 0.05"。判断像素图
结果是否可信只看残差, 或者直接用 --exact。
"""

import argparse
import sys
import time
from pathlib import Path

import anndata as ad
import cv2
import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.compare_pseudo_labels import expert_targets
from tools.snic_features import prepare_snic_lab


def grid_edges(mask):
    """组织内 4 邻接格子对, 索引按 mask 展平后的前景顺序。"""
    height, width = mask.shape
    index = -np.ones(mask.shape, dtype=np.int64)
    row, column = np.nonzero(mask)
    index[row, column] = np.arange(row.size)
    first, second = [], []
    keep = mask[:, :-1] & mask[:, 1:]
    first.append(index[:, :-1][keep])
    second.append(index[:, 1:][keep])
    keep = mask[:-1, :] & mask[1:, :]
    first.append(index[:-1, :][keep])
    second.append(index[1:, :][keep])
    return np.concatenate(first), np.concatenate(second)


def superpixel_edges(superpixels, node_lookup, tissue):
    """相邻且都在组织内的超像素对。"""
    first, second = [], []
    for axis in (0, 1):
        if axis == 0:
            a, b = superpixels[:-1, :], superpixels[1:, :]
            ta, tb = tissue[:-1, :], tissue[1:, :]
        else:
            a, b = superpixels[:, :-1], superpixels[:, 1:]
            ta, tb = tissue[:, :-1], tissue[:, 1:]
        keep = (a != b) & (a > 0) & (b > 0) & ta & tb
        first.append(node_lookup[a[keep]])
        second.append(node_lookup[b[keep]])
    return np.concatenate(first), np.concatenate(second)


def node_means(features, superpixels, nodes):
    """按超像素平均并标准化 (与模块同口径, 通道整体标准化)。"""
    size = int(superpixels.max()) + 1
    flat = superpixels.reshape(-1)
    channels = features.reshape(-1, features.shape[-1])
    counts = np.bincount(flat, minlength=size).astype(np.float64)
    means = np.zeros((size, channels.shape[1]), dtype=np.float64)
    for channel in range(channels.shape[1]):
        sums = np.bincount(flat, weights=channels[:, channel], minlength=size)
        means[:, channel] = sums / np.maximum(counts, 1)
    means = means[nodes]
    return (means - means.mean(axis=0)) / np.maximum(means.std(axis=0), 1e-6)


def standardize(features, mask):
    values = features[mask]
    return (features - values.mean(axis=0)) / np.maximum(values.std(axis=0), 1e-6)


def seeds_on_grid(spot_x, spot_y, spot_labels, tissue, radius):
    """spot 真值落到网格: radius=0 只取圆心, 否则铺一个圆盘。"""
    height, width = tissue.shape
    seed = np.full((height, width), 255, dtype=np.int64)
    offsets = np.arange(-radius, radius + 1)
    dy, dx = np.meshgrid(offsets, offsets, indexing="ij")
    disc = (dy**2 + dx**2) <= radius**2
    dy, dx = dy[disc], dx[disc]
    for x, y, label in zip(spot_x, spot_y, spot_labels):
        if label == 255:
            continue
        yy = np.clip(y + dy, 0, height - 1)
        xx = np.clip(x + dx, 0, width - 1)
        valid = tissue[yy, xx]
        seed[yy[valid], xx[valid]] = label
    return seed


def pcg_solve(laplacian, rhs, steps, tolerance):
    """Jacobi 预条件 CG, 与模块同一算法 (内部点全部待解)。

    返回 (解, 实际步数, 相对残差)。相对残差是判断"步数够不够"的唯一可靠信号:
    未收敛时初值 rhs(内部点为 0) 会把解整体拉向 0, 覆盖率看起来正常但标签全错。
    """
    diagonal = np.asarray(laplacian.diagonal()).ravel()
    inverse = 1.0 / np.maximum(diagonal, 1e-30)
    u = rhs.copy()
    residual = rhs - laplacian @ u
    z = inverse * residual
    direction = z.copy()
    rz = float(residual @ z)
    scale = max(float(rhs @ rhs), 1e-30)
    threshold = tolerance * tolerance * scale
    used = 0
    for _ in range(steps):
        weighted = laplacian @ direction
        denominator = float(direction @ weighted)
        if not np.isfinite(denominator) or denominator <= 0:
            break
        alpha = rz / denominator
        u += alpha * direction
        residual -= alpha * weighted
        used += 1
        if float(residual @ residual) <= threshold:
            break
        z = inverse * residual
        rz_new = float(residual @ z)
        direction = z + (rz_new / max(rz, 1e-30)) * direction
        rz = rz_new
    relative = float(np.sqrt(float(residual @ residual) / scale))
    return np.clip(u, 0.0, 1.0), used, relative


def exact_solve(laplacian, rhs):
    """直接法 (稀疏 LU) 求精确解, 只用于 crop 尺度作参照。"""
    from scipy.sparse.linalg import spsolve

    return spsolve(laplacian.tocsc(), rhs)


def solve(first, second, difference, seed_values, node_count, beta, steps, tolerance, exact=False):
    """构图 -> conductance -> Dirichlet 解。返回 (值, 可解掩码, 步数, 相对残差)。"""
    delta_squared = (difference**2).sum(axis=1)
    conductance = np.maximum(np.exp(-beta * delta_squared), 1e-6)
    adjacency = coo_matrix(
        (
            np.concatenate((conductance, conductance)),
            (np.concatenate((first, second)), np.concatenate((second, first))),
        ),
        shape=(node_count, node_count),
    ).tocsr()
    laplacian = diags(np.asarray(adjacency.sum(axis=1)).ravel()) - adjacency

    seeded = seed_values != 255.0
    component = connected_components(adjacency, directed=False)[1]
    if not seeded.any():
        return np.full(node_count, 0.5), np.zeros(node_count, dtype=bool), 0, 0.0
    solvable = np.isin(component, np.unique(component[seeded]))
    interior = solvable & ~seeded
    values = np.where(seeded, np.where(seed_values == 1, 1.0, 0.0), 0.5)
    used, relative = 0, 0.0
    if interior.any():
        boundary = np.flatnonzero(seeded)
        rhs = -laplacian[interior][:, boundary] @ values[boundary]
        system = laplacian[interior][:, interior].tocsr()
        if exact:
            solution = np.clip(exact_solve(system, rhs), 0.0, 1.0)
            used, relative = -1, 0.0
        else:
            solution, used, relative = pcg_solve(system, rhs, steps, tolerance)
        values[interior] = solution
    return values, solvable, used, relative


def evaluate(field, solvable, tissue, fraction, valid_count, thresholds, min_pixels=4):
    """组织内覆盖率与边界带 Dice, 与 compare_pseudo_labels 同口径。"""
    low, high = thresholds
    confident = ((field <= low) | (field >= high)) & solvable & tissue
    hard = np.full(field.shape, 255, dtype=np.int64)
    hard[confident] = (field[confident] >= 0.5).astype(np.int64)

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
        else:
            result[tag] = float("nan")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/her2st/1mpp")
    parser.add_argument("--slide", default="B1")
    parser.add_argument("--label-scale", type=int, default=7)
    parser.add_argument("--features", default="feature_ho0")
    parser.add_argument(
        "--patch-map", default="tile_224_mpp_1_resize_1_overlap_1_cls_overlap_1"
    )
    parser.add_argument("--feature-channels", type=int, default=16)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--thresholds", type=float, nargs=2, default=(0.2, 0.8))
    parser.add_argument("--seed-radius", type=int, default=7)
    parser.add_argument(
        "--window",
        type=int,
        default=0,
        help="只评估一个 window x window 标签格窗口 (0 = 全片)",
    )
    parser.add_argument("--exact", action="store_true", help="用直接法解, 作精度参照")
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
    image = cv2.imread(
        str(root / "samples" / slide / "spatial/he_1mpp.png"), cv2.IMREAD_COLOR
    )
    image = cv2.resize(
        image, (tissue.shape[1], tissue.shape[0]), interpolation=cv2.INTER_AREA
    )
    lab = prepare_snic_lab(image).astype(np.float64)
    patch_map = np.load(
        root / arguments.features / arguments.patch_map / f"{slide}/patch_map.npy",
        mmap_mode="r",
    )
    learned = np.asarray(
        cv2.resize(
            np.asarray(patch_map[..., : arguments.feature_channels], dtype=np.float32),
            (tissue.shape[1], tissue.shape[0]),
            interpolation=cv2.INTER_AREA,
        ),
        dtype=np.float64,
    )
    combined = np.concatenate((lab, learned), axis=-1)

    fraction, valid_count, _ = expert_targets(str(root), slide, scale)

    genes = ad.read_h5ad(root / "gene_expression" / "task1_loocv" / "test_C1.h5ad")
    obs = genes.obs[genes.obs["name"].astype(str) == slide]
    spot_x = np.rint(obs["pixel_x"].to_numpy() / scale).astype(np.int64)
    spot_y = np.rint(obs["pixel_y"].to_numpy() / scale).astype(np.int64)
    spot_labels = obs["spot_label"].to_numpy().astype(np.int64)

    if arguments.window:
        size = arguments.window
        integral = cv2.integral(tissue.astype(np.uint8))
        best, best_area = None, -1
        for top in range(0, tissue.shape[0] - size, 32):
            for left in range(0, tissue.shape[1] - size, 32):
                area = (
                    integral[top + size, left + size]
                    - integral[top, left + size]
                    - integral[top + size, left]
                    + integral[top, left]
                )
                inside = (
                    (spot_x >= left)
                    & (spot_x < left + size)
                    & (spot_y >= top)
                    & (spot_y < top + size)
                    & (spot_labels != 255)
                )
                positive = int((spot_labels[inside] == 1).sum())
                negative = int((spot_labels[inside] == 0).sum())
                if positive >= 4 and negative >= 4 and area > best_area:
                    best_area, best = area, (top, left)
        if best is None:
            raise ValueError("no window with both positive and negative seeds")
        top, left = best
        tissue = tissue[top : top + size, left : left + size]
        superpixels = superpixels[top : top + size, left : left + size]
        combined = combined[top : top + size, left : left + size]
        # fraction/valid_count 已经按 label_scale 分块, 直接切
        fraction = fraction[top : top + size, left : left + size]
        valid_count = valid_count[top : top + size, left : left + size]
        keep = (
            (spot_x >= left)
            & (spot_x < left + size)
            & (spot_y >= top)
            & (spot_y < top + size)
        )
        spot_x, spot_y, spot_labels = (
            spot_x[keep] - left,
            spot_y[keep] - top,
            spot_labels[keep],
        )
        # 窗口是标签格单位, tissue 已裁剪; 超像素/特征直接切窗口
        print(f"窗口 top={top} left={left} (标签格), 种子 {spot_x.size}")

    # --- 超像素图 ---
    nodes = np.unique(superpixels)
    nodes = nodes[nodes > 0]
    lookup = -np.ones(int(superpixels.max()) + 1, dtype=np.int64)
    lookup[nodes] = np.arange(nodes.size)
    means = node_means(combined, superpixels, nodes)
    h_first, h_second = superpixel_edges(superpixels, lookup, tissue)
    difference = means[h_first] - means[h_second]

    seed_grid = seeds_on_grid(spot_x, spot_y, spot_labels, tissue, 0)
    seed_nodes = np.full(nodes.size, 255, dtype=np.float64)
    node_of_spot = superpixels[tissue]
    keep = seed_grid[tissue] != 255
    positive = np.bincount(
        lookup[node_of_spot[keep]],
        weights=(seed_grid[tissue][keep] == 1),
        minlength=nodes.size,
    )
    negative = np.bincount(
        lookup[node_of_spot[keep]],
        weights=(seed_grid[tissue][keep] == 0),
        minlength=nodes.size,
    )
    seed_nodes[positive > negative] = 1.0
    seed_nodes[negative > positive] = 0.0

    start = time.perf_counter()
    values, solvable, used, relative = solve(
        h_first,
        h_second,
        difference,
        seed_nodes,
        nodes.size,
        arguments.beta,
        arguments.steps,
        arguments.tolerance,
        exact=arguments.exact,
    )
    time_sp = time.perf_counter() - start
    field = np.full(tissue.shape, 0.5)
    field[tissue] = values[lookup[superpixels[tissue]]]
    solvable_grid = np.zeros(tissue.shape, dtype=bool)
    solvable_grid[tissue] = solvable[lookup[superpixels[tissue]]]
    result_sp = evaluate(
        field, solvable_grid, tissue, fraction, valid_count, arguments.thresholds
    )
    print(
        f"[{slide}] 超像素图: {nodes.size} 节点 / {h_first.size} 边, "
        f"pcg={used} 步 残差={relative:.1e} {time_sp:.2f}s, coverage={result_sp['coverage']:.3f} "
        f"dice={result_sp['dice']:.4f} band_dice={result_sp['band_dice']:.4f}"
    )

    # --- 像素图 ---
    pixel_features = standardize(combined, tissue).reshape(-1, combined.shape[-1])
    p_first, p_second = grid_edges(tissue)
    p_difference = pixel_features[p_first] - pixel_features[p_second]
    node_count = int(tissue.sum())
    print(f"     像素图: {node_count} 节点 / {p_first.size} 边 ({node_count / nodes.size:.0f}x 节点)")
    for radius in (0, arguments.seed_radius):
        pixels = seeds_on_grid(spot_x, spot_y, spot_labels, tissue, radius)
        seed_values = pixels[tissue].astype(np.float64)
        start = time.perf_counter()
        values, solvable, used, relative = solve(
            p_first,
            p_second,
            p_difference,
            seed_values,
            node_count,
            arguments.beta,
            arguments.steps,
            arguments.tolerance,
            exact=arguments.exact,
        )
        elapsed = time.perf_counter() - start
        field = np.full(tissue.shape, 0.5)
        field[tissue] = values
        solvable_grid = np.zeros(tissue.shape, dtype=bool)
        solvable_grid[tissue] = solvable
        result = evaluate(
            field, solvable_grid, tissue, fraction, valid_count, arguments.thresholds
        )
        print(
            f"     像素图(种子半径={radius}): pcg={used} 步 残差={relative:.1e} {elapsed:.2f}s, "
            f"coverage={result['coverage']:.3f} dice={result['dice']:.4f} "
            f"band_dice={result['band_dice']:.4f}"
        )


if __name__ == "__main__":
    main()
