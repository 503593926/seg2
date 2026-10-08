"""对照 scipy 精确解与可微迭代解，验证传播算子可微化的数值可行性。

用法:
    python tools/validate_jacobi_scipy.py [data_root] [slide]

验证三件事:
    1. 收敛性: Jacobi / CG 相对 scipy 精确解的误差随迭代步数变化
    2. 实用性: 截断解与精确解在最终伪标签(阈值化后)上的不一致比例
    3. 可微性: loss 对边权重的解析梯度与有限差分是否一致

种子来自 h5ad 的真实 spot 标注, 与 pipeline 的 compute_seed_votes 同流程。
"""

import sys
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import spsolve

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.propagate_random_walker import (
    build_random_walker_topology,
    compute_seed_votes,
)


def build_system(superpixels, pixel_features, beta, learned_weight):
    """复刻 propagation.propagate_features 的构图逻辑。"""
    topology = build_random_walker_topology(superpixels)
    labels = topology.foreground_labels

    means = np.zeros(
        (topology.max_label + 1, pixel_features.shape[2]), dtype=np.float64
    )
    flat = superpixels.ravel()
    for channel in range(pixel_features.shape[2]):
        sums = np.bincount(
            flat, weights=pixel_features[..., channel].ravel(), minlength=len(means)
        )
        means[labels, channel] = sums[labels] / topology.pixel_counts[labels]

    lab_channels = 3
    means[labels, :lab_channels] = (
        means[labels, :lab_channels] - means[labels, :lab_channels].mean(axis=0)
    ) / np.maximum(means[labels, :lab_channels].std(axis=0), 1e-6)
    learned = means[labels, lab_channels:]
    learned = (learned - learned.mean(axis=0)) / np.maximum(learned.std(axis=0), 1e-6)
    learned *= np.sqrt(
        lab_channels * learned_weight / (pixel_features.shape[2] - lab_channels)
    )
    means[labels, lab_channels:] = learned

    delta = means[topology.first_labels] - means[topology.second_labels]
    conductance = np.maximum(np.exp(-beta * np.square(delta).sum(axis=1)), 1e-6)

    first, second = topology.first_nodes, topology.second_nodes
    adjacency = coo_matrix(
        (
            np.concatenate((conductance, conductance)),
            (np.concatenate((first, second)), np.concatenate((second, first))),
        ),
        shape=(len(labels), len(labels)),
    ).tocsr()
    laplacian = diags(np.asarray(adjacency.sum(axis=1)).ravel()) - adjacency
    return topology, laplacian, conductance


def setup_dirichlet(topology, laplacian, votes):
    """与 propagate_features 相同的 Dirichlet 系统设置。"""
    labels = topology.foreground_labels
    node_votes = votes[labels]
    seeded = (node_votes == 1) | (node_votes == 2)
    component_labels = topology.component_labels
    seeded_components = np.unique(component_labels[seeded])
    solvable = ~seeded & np.isin(component_labels, seeded_components)

    interior = np.flatnonzero(solvable)
    boundary = np.flatnonzero(seeded)
    laplacian_uu = laplacian[interior][:, interior].tocsr()
    seed_values = np.where(node_votes[boundary] == 1, 0.0, 1.0)
    rhs = -np.asarray(laplacian[interior][:, boundary] @ seed_values).ravel()

    probability = np.full(len(labels), 0.5)
    probability[node_votes == 1] = 0.0
    probability[node_votes == 2] = 1.0
    return interior, boundary, laplacian_uu, rhs, probability, solvable


def jacobi(laplacian_uu, rhs, steps):
    diagonal = laplacian_uu.diagonal()
    off = laplacian_uu - diags(diagonal)
    u = np.full(rhs.shape, 0.5)
    for _ in range(steps):
        u = (rhs - off @ u) / diagonal
    return u


def conjugate_gradient(laplacian_uu, rhs, steps, tolerance=1e-12):
    """返回 (解, 实际迭代步数)。"""
    u = np.zeros_like(rhs)
    residual = rhs - laplacian_uu @ u
    rs = float(residual @ residual)
    if rs == 0.0:
        return u, 0
    direction = residual.copy()
    threshold = tolerance * tolerance * float(rhs @ rhs)
    used = 0
    for _ in range(steps):
        ap = laplacian_uu @ direction
        denominator = float(direction @ ap)
        if denominator <= 0.0 or not np.isfinite(denominator):
            break
        alpha = rs / denominator
        u += alpha * direction
        residual -= alpha * ap
        used += 1
        rs_new = float(residual @ residual)
        if rs_new <= threshold:
            break
        direction = residual + (rs_new / rs) * direction
        rs = rs_new
    return u, used


def preconditioned_cg(laplacian_uu, rhs, steps, tolerance=1e-12):
    """Jacobi 预处理共轭梯度。对角预处理对度分布不均的拉普拉斯系统很关键。"""
    diagonal = laplacian_uu.diagonal()
    inverse_diagonal = 1.0 / np.maximum(diagonal, 1e-30)
    u = np.zeros_like(rhs)
    residual = rhs - laplacian_uu @ u
    z = inverse_diagonal * residual
    direction = z.copy()
    rz = float(residual @ z)
    if rz <= 0.0:
        return u, 0
    threshold = tolerance * tolerance * float(rhs @ rhs)
    used = 0
    for _ in range(steps):
        ap = laplacian_uu @ direction
        denominator = float(direction @ ap)
        if denominator <= 0.0 or not np.isfinite(denominator):
            break
        alpha = rz / denominator
        u += alpha * direction
        residual -= alpha * ap
        used += 1
        if float(residual @ residual) <= threshold:
            break
        z = inverse_diagonal * residual
        rz_new = float(residual @ z)
        direction = z + (rz_new / rz) * direction
        rz = rz_new
    return u, used


def condition_numbers(laplacian_uu):
    """原始尺度与行归一化尺度下的条件数 (后者对应 D^{-1}L 的谱)。"""
    diagonal = laplacian_uu.diagonal()
    row_scale = np.maximum(diagonal, 1e-30)
    normalized = (laplacian_uu / row_scale[:, None]).toarray()
    eigenvalues = np.linalg.eigvals(normalized).real
    positive = eigenvalues[eigenvalues > 1e-10]
    raw = float(np.linalg.cond(laplacian_uu.toarray()))
    return float(positive.max() / positive.min()), raw


def spectral_radius(laplacian_uu):
    diagonal = laplacian_uu.diagonal()
    off = (laplacian_uu - diags(diagonal)).toarray()
    iteration = -off / diagonal[:, None]
    return float(np.abs(np.linalg.eigvals(iteration)).max())


def hard_label_agreement(exact, approximate, interior, non_cancer=0.35, cancer=0.65):
    def hard(values):
        labels = np.full(values.shape, 255, dtype=np.uint8)
        labels[values <= non_cancer] = 0
        labels[values >= cancer] = 1
        return labels

    return float((hard(exact[interior]) != hard(approximate[interior])).mean())


def gradient_check(conductance, first, second, node_count, node_budget=192):
    """在连通的采样子图上检查 d(loss)/d(conductance)，前向用稠密精确解。"""
    import torch
    from scipy.sparse import coo_matrix as scipy_coo
    from scipy.sparse.csgraph import connected_components

    # 取边表前段（空间上连续）作为候选区域，再取最大连通分量
    pool = min(len(first), 600)
    node_index = -np.ones(node_count, dtype=np.int64)
    pool_nodes = np.unique(np.concatenate((first[:pool], second[:pool])))
    node_index[pool_nodes] = np.arange(pool_nodes.size)
    small_first = node_index[first[:pool]]
    small_second = node_index[second[:pool]]
    connectivity = scipy_coo(
        (np.ones(small_first.size), (small_first, small_second)),
        shape=(pool_nodes.size, pool_nodes.size),
    ).tocsr()
    _, component = connected_components(connectivity, directed=False)
    largest = np.bincount(component).argmax()
    members = np.flatnonzero(component == largest)
    rng = np.random.default_rng(1)
    chosen = np.sort(rng.choice(members, size=min(node_budget, members.size), replace=False))
    keep = np.zeros(pool_nodes.size, dtype=bool)
    keep[chosen] = True
    edge_keep = keep[small_first] & keep[small_second]
    small_first = small_first[edge_keep]
    small_second = small_second[edge_keep]
    small_conductance = conductance[:pool][edge_keep]

    # 采样后可能有孤立点: 剔除零度节点并再次重索引
    reindex = -np.ones(pool_nodes.size, dtype=np.int64)
    reindex[chosen] = np.arange(chosen.size)
    small_first = reindex[small_first]
    small_second = reindex[small_second]
    n = chosen.size
    degree = np.zeros(n, dtype=np.int64)
    np.add.at(degree, small_first, 1)
    np.add.at(degree, small_second, 1)
    keep_map = degree > 0
    old_to_new = -np.ones(n, dtype=np.int64)
    old_to_new[keep_map] = np.arange(keep_map.sum())
    edge_keep = keep_map[small_first] & keep_map[small_second]
    small_first = old_to_new[small_first[edge_keep]]
    small_second = old_to_new[small_second[edge_keep]]
    small_conductance = small_conductance[edge_keep]
    n = int(keep_map.sum())
    # 检查用下界: 数值有限差分在 1e-6 量级边上不可分辨, 抬到 1e-3
    small_conductance = np.maximum(small_conductance, 1e-3)

    # 再次取最大连通分量 (剔除零度点后仍可能分裂)
    connectivity = scipy_coo(
        (np.ones(small_first.size), (small_first, small_second)),
        shape=(n, n),
    ).tocsr()
    _, component = connected_components(connectivity, directed=False)
    largest = np.bincount(component).argmax()
    in_component = component == largest
    old_to_new = -np.ones(n, dtype=np.int64)
    old_to_new[in_component] = np.arange(in_component.sum())
    edge_keep = in_component[small_first] & in_component[small_second]
    small_first = old_to_new[small_first[edge_keep]]
    small_second = old_to_new[small_second[edge_keep]]
    small_conductance = small_conductance[edge_keep]
    n = int(in_component.sum())
    print(f"  subgraph: nodes={n}, edges={len(small_conductance)}")

    # 两个相隔的固定种子 (行替换为恒等, rhs 给 0.3 / 0.7)
    midpoint = len(small_conductance) // 2
    seed_a, seed_b = int(small_first[0]), int(small_second[midpoint])
    seed_mask = np.zeros(n, dtype=bool)
    seed_mask[seed_a] = seed_mask[seed_b] = True
    rhs_value = np.zeros(n, dtype=np.float64)
    rhs_value[seed_a] = 0.3
    rhs_value[seed_b] = 0.7
    identity = torch.eye(n, dtype=torch.float64)
    row_mask = torch.tensor(seed_mask)
    rhs_tensor = torch.tensor(rhs_value, dtype=torch.float64)
    edge_index = torch.from_numpy(np.stack((small_first, small_second)))
    weight = torch.linspace(0.1, 1.0, n, dtype=torch.float64)

    def forward(conductance_value):
        adjacency = torch.zeros(n, n, dtype=torch.float64)
        adjacency.index_put_(
            (edge_index[0], edge_index[1]), conductance_value, accumulate=True
        )
        adjacency = adjacency + adjacency.T
        laplacian = torch.diag(adjacency.sum(dim=1)) - adjacency
        # 行归一化使矩阵数值范围一致
        scale = adjacency.sum(dim=1).clamp_min(1e-8)
        normalized = laplacian / scale[:, None]
        normalized = torch.where(row_mask[:, None], identity, normalized)
        return torch.linalg.solve(normalized, rhs_tensor)

    conductance_value = torch.tensor(
        small_conductance, dtype=torch.float64, requires_grad=True
    )
    solution = forward(conductance_value)
    (gradient,) = torch.autograd.grad((solution * weight).sum(), conductance_value)

    indices = rng.choice(len(small_conductance), size=16, replace=False)
    analytic_values = gradient[indices].numpy()
    numeric_values = np.zeros(indices.size, dtype=np.float64)
    print("  analytic vs finite-difference (edge weight, epsilon=1e-5):")
    for position, index in enumerate(indices):
        epsilon = 1e-5 * max(float(small_conductance[index]), 1e-2)
        plus = small_conductance.copy()
        plus[index] += epsilon
        minus = small_conductance.copy()
        minus[index] -= epsilon
        plus_value = forward(torch.tensor(plus, dtype=torch.float64))
        minus_value = forward(torch.tensor(minus, dtype=torch.float64))
        numeric_values[position] = (
            float(((plus_value - minus_value) * weight).sum()) / (2 * epsilon)
        )
        print(
            f"    edge {int(index):>6}: analytic={analytic_values[position]:>14.8f} "
            f"numeric={numeric_values[position]:>14.8f}"
        )
    vector_relative = float(
        np.linalg.norm(analytic_values - numeric_values)
        / max(np.linalg.norm(analytic_values), np.linalg.norm(numeric_values), 1e-30)
    )
    worst = float(np.abs(analytic_values - numeric_values).max())
    print(
        f"  vector relative error (2-norm, {indices.size} 条边) = {vector_relative:.3e}, "
        f"max abs error = {worst:.3e}"
    )
    return worst


def main(data_root, slide):
    import anndata as ad
    import cv2

    root = Path(data_root)
    superpixels = np.load(root / "label_mpp" / slide / "snic/superpixels.npy").astype(
        np.int32
    )
    image = cv2.imread(
        str(root / "samples" / slide / "spatial/he_1mpp.png"), cv2.IMREAD_COLOR
    )
    image = cv2.resize(
        image, (superpixels.shape[1], superpixels.shape[0]), interpolation=cv2.INTER_AREA
    )
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    patch_map = np.load(
        root
        / "feature_ho0"
        / "tile_224_mpp_1_resize_1_overlap_1_cls_overlap_1"
        / slide
        / "patch_map.npy",
        mmap_mode="r",
    )
    learned = np.asarray(
        cv2.resize(
            np.asarray(patch_map[..., :16], dtype=np.float32),
            (superpixels.shape[1], superpixels.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
    )
    pixel_features = np.concatenate((lab, learned), axis=-1)

    topology, laplacian, conductance = build_system(superpixels, pixel_features, 0.5, 1.0)
    print(
        f"graph: nodes={topology.foreground_labels.size} "
        f"edges={topology.edge_keys.size} "
        f"components={np.unique(topology.component_labels).size} "
        f"conductance[min/median/max]="
        f"{conductance.min():.2e}/{np.median(conductance):.2e}/{conductance.max():.2e}"
    )
    print(f"edges at floor (<=1.01e-6): {(conductance <= 1.01e-6).sum()} / {conductance.size}")

    # 真实 spot 种子: 与 pipeline 相同流程
    adata = ad.read_h5ad(
        root / "gene_expression" / "task1_loocv" / f"test_{slide}.h5ad"
    )
    label_scale = 7  # label_mpp / target_mpp
    coordinates = adata.obs[["pixel_x", "pixel_y"]].to_numpy(dtype=np.float32)
    spot_labels = adata.obs["spot_label"].to_numpy(dtype=np.uint8)
    coordinates = np.rint(coordinates / label_scale).astype(np.int64)
    height, width = superpixels.shape
    inside = (
        (coordinates[:, 0] >= 0)
        & (coordinates[:, 0] < width)
        & (coordinates[:, 1] >= 0)
        & (coordinates[:, 1] < height)
    )
    x, y, labels = coordinates[inside, 0], coordinates[inside, 1], spot_labels[inside]
    print(
        f"spots: total={len(x)} inside={inside.sum()} "
        f"tumor={(labels == 1).sum()} normal={(labels == 0).sum()} "
        f"unknown={(labels == 255).sum()}"
    )
    votes = compute_seed_votes(superpixels, x, y, labels, topology)
    node_votes = votes[topology.foreground_labels]
    print(
        f"seed superpixels: normal={(node_votes == 1).sum()} "
        f"tumor={(node_votes == 2).sum()} conflict={(node_votes == 3).sum()}"
    )

    interior, boundary, laplacian_uu, rhs, probability, _ = setup_dirichlet(
        topology, laplacian, votes
    )
    print(f"seeds={boundary.size} interior={interior.size}")
    print(
        f"diagnostics: |rhs|={np.linalg.norm(rhs):.4e} "
        f"diag[min/max]={laplacian_uu.diagonal().min():.2e}/"
        f"{laplacian_uu.diagonal().max():.2e}"
    )

    radius = spectral_radius(laplacian_uu)
    normalized_condition, raw_condition = condition_numbers(laplacian_uu)
    print(f"Jacobi spectral radius = {radius:.8f}  (1 - rho = {1 - radius:.2e})")
    print(
        f"condition number: raw={raw_condition:.3e} "
        f"row-normalized={normalized_condition:.3e}"
    )

    exact_interior = spsolve(laplacian_uu, rhs)
    exact = probability.copy()
    exact[interior] = np.clip(exact_interior, 0.0, 1.0)

    print(f"\n{'steps':>6} | {'jac max':>9} {'jac mean':>10} {'labels':>8} "
          f"| {'cg max':>9} {'cg mean':>10} {'labels':>8} {'used':>6} "
          f"| {'pcg max':>9} {'pcg mean':>10} {'labels':>8} {'used':>6}")
    for steps in (10, 20, 50, 100, 200, 500):
        jac_interior = jacobi(laplacian_uu, rhs, steps)
        cg_interior, cg_used = conjugate_gradient(laplacian_uu, rhs, steps)
        pcg_interior, pcg_used = preconditioned_cg(laplacian_uu, rhs, steps)
        jac_full = probability.copy()
        jac_full[interior] = np.clip(jac_interior, 0.0, 1.0)
        cg_full = probability.copy()
        cg_full[interior] = np.clip(cg_interior, 0.0, 1.0)
        pcg_full = probability.copy()
        pcg_full[interior] = np.clip(pcg_interior, 0.0, 1.0)
        jac_diff = np.abs(jac_interior - exact_interior)
        cg_diff = np.abs(cg_interior - exact_interior)
        pcg_diff = np.abs(pcg_interior - exact_interior)
        print(
            f"{steps:>6} | {jac_diff.max():>9.2e} {jac_diff.mean():>10.2e} "
            f"{hard_label_agreement(exact, jac_full, interior):>8.4f} "
            f"| {cg_diff.max():>9.2e} {cg_diff.mean():>10.2e} "
            f"{hard_label_agreement(exact, cg_full, interior):>8.4f} {cg_used:>6} "
            f"| {pcg_diff.max():>9.2e} {pcg_diff.mean():>10.2e} "
            f"{hard_label_agreement(exact, pcg_full, interior):>8.4f} {pcg_used:>6}"
        )

    print("\ngradient check (dense solve, connected subgraph):")
    worst = gradient_check(
        conductance,
        topology.first_nodes,
        topology.second_nodes,
        topology.foreground_labels.size,
    )
    print(f"  worst |analytic - numeric| = {worst:.3e}")


if __name__ == "__main__":
    data_root = sys.argv[1] if len(sys.argv) > 1 else "data/her2st/1mpp"
    slide = sys.argv[2] if len(sys.argv) > 2 else "B1"
    main(data_root, slide)
