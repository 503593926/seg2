"""当前 SNIC 拓扑上的 RW：颜色/中间特征构造边权，Jacobi-PCG 求解软标签。

内部保留可微计算和视觉门控；当前训练在 no_grad 中生成目标，gate_range=0。
"""

import numpy as np
import torch
from scipy.sparse import coo_matrix as scipy_coo
from scipy.sparse.csgraph import connected_components
from torch import nn
from torch.nn import functional as F


class ConductionGate(nn.Module):
    """每条边一个标量调制, 输入是相邻节点的特征差。初始化接近恒等。

    完全零初始化得到均匀常数并与 base 相乘会退化, 因此末层加小随机扰动:
    训练开始时 g≈1, 传播行为等价于现有的固定核。
    """

    def __init__(self, gate_range=0.3):
        super().__init__()
        self.gate_range = float(gate_range)
        self.network = nn.Sequential(nn.Linear(3, 8), nn.Tanh(), nn.Linear(8, 1))
        nn.init.normal_(self.network[0].get_parameter("weight"), std=0.1)
        nn.init.zeros_(self.network[0].get_parameter("bias"))
        nn.init.normal_(self.network[2].get_parameter("weight"), std=0.01)
        nn.init.zeros_(self.network[2].get_parameter("bias"))

    def forward(self, difference_squared, difference_l1, base):
        features = torch.stack(
            (
                torch.log1p(difference_squared),
                torch.log1p(difference_l1),
                torch.log1p(base.clamp_min(1e-6)),
            ),
            dim=-1,
        )
        return 1.0 + self.gate_range * torch.tanh(self.network(features)).squeeze(-1)


def rgb_to_lab(rgb):
    """RGB (0..255) 转 CIE-Lab, 与 SNIC 的 rgbtolab 同一套公式。返回同形状张量。"""
    values = rgb.to(torch.promote_types(rgb.dtype, torch.float32)) / 255.0
    linear = torch.where(
        values <= 0.04045,
        values / 12.92,
        ((values + 0.055) / 1.055) ** 2.4,
    )
    red, green, blue = linear.unbind(0)
    x = red * 0.4124564 + green * 0.3575761 + blue * 0.1804375
    y = red * 0.2126729 + green * 0.7151522 + blue * 0.0721750
    z = red * 0.0193339 + green * 0.1191920 + blue * 0.9503041
    xr, yr, zr = x / 0.950456, y, z / 1.088754
    epsilon = 0.008856
    fx = torch.where(xr > epsilon, xr.pow(1.0 / 3.0), (903.3 * xr + 16.0) / 116.0)
    fy = torch.where(yr > epsilon, yr.pow(1.0 / 3.0), (903.3 * yr + 16.0) / 116.0)
    fz = torch.where(zr > epsilon, zr.pow(1.0 / 3.0), (903.3 * zr + 16.0) / 116.0)
    return torch.stack((116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)))


def node_means(values, superpixels, node_ids, *, size=None, counts=None):
    """按超像素聚合并对每个通道做前景标准化。返回 (N, C)。

    values: (C, H, W); superpixels: (H, W) 的 0..max_id (0 为背景);
    node_ids: (N,) 升序的唯一前景 id。两者空间网格必须一致: scatter_add_ 在
    index 比 src 短时不报错, 只取 src 前缀, 形状错位会静默算错。
    数据侧分辨率高于超像素网格时调用方先用面积下采样对齐。
    """
    if values.shape[-2:] != superpixels.shape:
        raise ValueError(
            f"node_means: feature grid {tuple(values.shape[-2:])} does not match "
            f"superpixel grid {tuple(superpixels.shape)}"
        )
    flat_ids = superpixels.reshape(-1)
    flat_values = values.reshape(values.shape[0], -1)
    if size is None:
        size = int(node_ids.max()) + 1
    if counts is None:
        counts = torch.bincount(flat_ids, minlength=size).clamp_min(1)
    sums = values.new_zeros(values.shape[0], size)
    sums.scatter_add_(1, flat_ids.unsqueeze(0).expand(values.shape[0], -1), flat_values)
    means = (sums / counts).transpose(0, 1)[node_ids]
    return (means - means.mean(dim=0)) / means.std(dim=0).clamp_min(1e-6)


def build_edges(superpixels):
    """crop 内超像素的相邻对 (a<b)。返回 (E, 2) int64。"""
    pairs = []
    for first, second in (
        (superpixels[:, :-1], superpixels[:, 1:]),
        (superpixels[:-1, :], superpixels[1:, :]),
    ):
        boundary = (first != second) & (first > 0) & (second > 0)
        pairs.append(torch.stack((first[boundary], second[boundary]), dim=1))
    pairs = torch.cat(pairs)
    smaller = pairs.min(dim=1).values
    larger = pairs.max(dim=1).values
    return torch.unique(torch.stack((smaller, larger), dim=1), dim=0)


def component_labels(node_count, first_nodes, second_nodes):
    """crop 子图的连通分量 (无向)。边为空时每个节点各成一个分量。

    训练时在数据加载 worker 中调用，避免 GPU 往返。
    """
    if first_nodes.numel() == 0:
        return np.arange(node_count, dtype=np.int64)
    first = first_nodes.cpu().numpy()
    second = second_nodes.cpu().numpy()
    connectivity = scipy_coo(
        (
            np.ones(first.size * 2, dtype=np.int8),
            (
                np.concatenate((first, second)),
                np.concatenate((second, first)),
            ),
        ),
        shape=(node_count, node_count),
    ).tocsr()
    _, labels = connected_components(connectivity, directed=False)
    return labels.astype(np.int64)


def scatter_seed_values(superpixels, spot_labels, spot_positions, node_ids, label_scale):
    """spot 投票到节点: 多数类别作为种子，平票或无标签为 255。

    spot_positions: (S, 2) 图像像素坐标 (x, y), 与超像素同一 crop 网格。
    """
    if spot_labels.numel() == 0 or node_ids.numel() == 0:
        return torch.full(
            (node_ids.numel(),), 255, dtype=torch.long, device=node_ids.device
        )
    height, width = superpixels.shape
    coordinates = torch.round(spot_positions / label_scale).to(torch.int64)
    coordinates[:, 0] = coordinates[:, 0].clamp(0, width - 1)
    coordinates[:, 1] = coordinates[:, 1].clamp(0, height - 1)
    node_of_spot = superpixels[coordinates[:, 1], coordinates[:, 0]]
    position = torch.searchsorted(node_ids, node_of_spot.contiguous())
    position = position.clamp_max(max(node_ids.numel() - 1, 0))
    known = (
        (node_of_spot > 0) & (node_ids[position] == node_of_spot) & (spot_labels != 255)
    )

    positive = torch.zeros(node_ids.numel(), dtype=torch.long, device=node_ids.device)
    negative = torch.zeros_like(positive)
    positive.index_add_(0, position[known], (spot_labels[known] == 1).long())
    negative.index_add_(0, position[known], (spot_labels[known] == 0).long())
    values = torch.full_like(positive, 255)
    values = torch.where(positive > negative, torch.ones_like(values), values)
    values = torch.where(negative > positive, torch.zeros_like(values), values)
    return values


def prepare_graph(superpixels, spot_labels, spot_positions, label_scale):
    """CPU 上准备 crop 拓扑与种子；边权仍由本次前向的图像/特征生成。"""
    superpixels = superpixels.cpu().long()
    nodes = torch.unique(superpixels)
    nodes = nodes[nodes > 0]
    if nodes.numel() == 0:
        return None
    edges = build_edges(superpixels)
    if edges.numel() == 0:
        return None
    size = int(nodes.max()) + 1
    lookup = torch.full((size,), -1, dtype=torch.long)
    lookup[nodes] = torch.arange(nodes.numel())
    edges = lookup[edges]
    seeds = scatter_seed_values(
        superpixels, spot_labels.cpu(), spot_positions.cpu(), nodes, label_scale,
    )
    seeded = seeds != 255
    if not bool(seeded.any()):
        return None
    labels = component_labels(nodes.numel(), edges[:, 0], edges[:, 1])
    solvable = np.isin(labels, np.unique(labels[seeded.numpy()]))
    return {
        "nodes": nodes, "edges": edges, "seeds": seeds,
        "solvable": torch.from_numpy(solvable), "size": size,
        "counts": torch.bincount(superpixels.reshape(-1), minlength=size).clamp_min(1),
    }


def _apply_conductance(values, conductance, first_nodes, second_nodes):
    """(L u)_i = Σ_j c_ij (u_i - u_j), 对称边表。"""
    difference = values[first_nodes] - values[second_nodes]
    contribution = torch.zeros_like(values)
    contribution.index_add_(0, first_nodes, difference * conductance)
    contribution.index_add_(0, second_nodes, -difference * conductance)
    return contribution


@torch.jit.script
def _pcg_cuda_iterations(
    conductance: torch.Tensor, first_nodes: torch.Tensor, second_nodes: torch.Tensor,
    interior: torch.Tensor, inverse: torch.Tensor, u: torch.Tensor,
    residual: torch.Tensor, direction: torch.Tensor, rz: torch.Tensor,
    threshold: torch.Tensor, steps: int,
):
    """在脚本循环中调度小算子，避免每次迭代往返 Python。"""
    active = torch.ones((), dtype=torch.bool, device=conductance.device)
    used = torch.zeros((), dtype=torch.long, device=conductance.device)
    for iteration in range(steps):
        weighted = _apply_conductance(direction, conductance, first_nodes, second_nodes)
        weighted = torch.where(interior, weighted, 0)
        denominator = (direction * weighted).sum()
        take_step = active & torch.isfinite(denominator) & (denominator > 0)
        alpha = torch.where(take_step, rz / torch.where(take_step, denominator, 1), 0)
        u = torch.where(take_step, u + alpha * direction, u)
        residual = torch.where(take_step, residual - alpha * weighted, residual)
        used = used + take_step.long()
        active = take_step & ~((residual * residual).sum() <= threshold)
        z = inverse * residual
        rz_new = (residual * z).sum()
        ratio = rz_new / torch.where(active, rz.clamp_min(1e-30), 1)
        direction = torch.where(active, z + ratio * direction, 0)
        rz = torch.where(active, rz_new, rz)
        # 每步在 GPU 上冻结结果，每 8 步才读取是否终止。
        if (iteration + 1) % 8 == 0 and not bool(active):
            break
    return u, used


def pcg_solve(conductance, edges, seed_values, seeded, interior, steps, tolerance):
    """解 Dirichlet 系统: 内部节点 (L u)_i = 0, 种子节点 u_i = seed。

    边界与不可解节点用单位行替换, 因此残差只统计内部节点。
    """
    first_nodes, second_nodes = edges[:, 0], edges[:, 1]
    node_count = seed_values.shape[0]

    right_hand_side = torch.where(seeded, seed_values, torch.zeros_like(seed_values))

    def operator(values):
        product = _apply_conductance(values, conductance, first_nodes, second_nodes)
        return torch.where(interior, product, values - right_hand_side)

    diagonal = torch.zeros(
        node_count, dtype=conductance.dtype, device=conductance.device
    )
    diagonal.index_add_(0, first_nodes, conductance)
    diagonal.index_add_(0, second_nodes, conductance)
    inverse = torch.where(
        interior, 1.0 / diagonal.clamp_min(1e-30), torch.ones_like(diagonal)
    )

    u = right_hand_side.clone()
    residual = torch.where(interior, -operator(u), torch.zeros_like(u))
    z = inverse * residual
    direction = z.clone()
    rz = (residual * z).sum()
    scale = (right_hand_side * right_hand_side).sum().clamp_min(1e-30)
    threshold = tolerance * tolerance * scale
    # 零容差用于精确收敛/梯度检查，保留原算子顺序，避免脚本融合的舍入差异。
    if conductance.is_cuda and tolerance > 0:
        u, used = _pcg_cuda_iterations(
            conductance, first_nodes, second_nodes, interior, inverse,
            u, residual, direction, rz, threshold, steps,
        )
        return u, int(used)
    used = 0
    for _ in range(steps):
        weighted = _apply_conductance(direction, conductance, first_nodes, second_nodes)
        weighted = torch.where(interior, weighted, torch.zeros_like(weighted))
        denominator = (direction * weighted).sum()
        if not bool(torch.isfinite(denominator)) or bool(denominator <= 0):
            break
        alpha = rz / denominator
        u = u + alpha * direction
        residual = residual - alpha * weighted
        used += 1
        if bool((residual * residual).sum() <= threshold):
            break
        z = inverse * residual
        rz_new = (residual * z).sum()
        direction = z + (rz_new / rz.clamp_min(1e-30)) * direction
        rz = rz_new
    return u, int(used)


class DifferentiablePropagation(nn.Module):
    """用 crop 内全部类别种子、颜色和模型中间特征生成 RW 软目标。"""

    def __init__(
        self,
        label_scale=1,
        beta=0.5,
        learned_weight=1.0,
        steps=85,
        tolerance=1e-8,
        gate_range=0.3,
        non_cancer_threshold=0.35,
        cancer_threshold=0.65,
        feature_gradients=False,
    ):
        super().__init__()
        self.label_scale = int(label_scale)
        self.beta = float(beta)
        self.learned_weight = float(learned_weight)
        self.steps = int(steps)
        self.tolerance = float(tolerance)
        self.non_cancer_threshold = float(non_cancer_threshold)
        self.cancer_threshold = float(cancer_threshold)
        self.feature_gradients = bool(feature_gradients)
        self.gate = ConductionGate(gate_range)

    def forward(self, superpixels, features, rgb, spot_labels, spot_positions, graphs=None):
        """返回有效样本索引、软目标、监督范围和 PCG 诊断；无可解图时返回 None。"""
        if graphs is None:
            graphs = [prepare_graph(grid[0], labels, spots, self.label_scale)
                      for grid, labels, spots in zip(superpixels, spot_labels, spot_positions)]
            graphs = [None if graph is None else {
                name: value.to(superpixels.device) if isinstance(value, torch.Tensor) else value
                for name, value in graph.items()
            } for graph in graphs]
        prepared = [
            self._sample_graph(
                superpixels[index, 0], features[index], rgb[index],
                graphs[index],
            )
            for index in range(superpixels.shape[0])
        ]
        indices = [index for index, item in enumerate(prepared) if item is not None]
        if not indices:
            return None
        return self._solve_batch(superpixels, indices, [prepared[index] for index in indices])

    def _sample_graph(
        self, superpixels, features, rgb, graph,
    ):
        """单样本: 节点/边/边特征/种子。返回 None 表示该样本无有效图。"""
        if graph is None:
            return None
        nodes, local_edges = graph["nodes"], graph["edges"]

        # 训练图: rgb 是 1mpp 裁剪图, 超像素在标签网格; 先面积下采样再转 Lab,
        # 与预处理使用同一套颜色空间。
        # 计算精度取 rgb/特征/float32 中最宽的那个: bf16 上采样到 fp32,
        # fp64 原样保留 (数值梯度检查要用双精度)。
        compute_dtype = torch.promote_types(
            torch.promote_types(rgb.dtype, features.dtype), torch.float32
        )
        image = rgb.to(compute_dtype)
        if image.shape[-2:] != superpixels.shape:
            image = F.interpolate(
                image[None], size=tuple(superpixels.shape), mode="area"
            )[0]
        color = node_means(rgb_to_lab(image), superpixels, nodes,
                           size=graph["size"], counts=graph["counts"])
        learned = features.to(compute_dtype)
        if not self.feature_gradients:
            learned = learned.detach()
        learned = node_means(learned, superpixels, nodes,
                             size=graph["size"], counts=graph["counts"])
        learned = learned * np.sqrt(
            3.0 * self.learned_weight / max(learned.shape[1], 1)
        )
        combined = torch.cat((color, learned), dim=1)
        difference = combined[local_edges[:, 0]] - combined[local_edges[:, 1]]
        edge_delta = difference.square().sum(dim=-1)
        edge_l1 = difference.abs().sum(dim=-1)

        return graph, edge_delta, edge_l1

    def _solve_batch(self, superpixels, indices, prepared):
        """批级求解: 节点整体偏移后一次 PCG; 不可解节点保持 0.5。

        返回 (sample_indices, soft, valid, diagnostics), 只覆盖可解样本。
        """
        device = superpixels.device
        offsets = []
        total = 0
        for item in prepared:
            offsets.append(total)
            total += item[0]["nodes"].numel()

        global_edges, global_delta, global_l1 = [], [], []
        global_seed = torch.full((total,), 255, dtype=torch.long, device=device)
        for offset, item in zip(offsets, prepared):
            graph, edge_delta, edge_l1 = item
            local_edges, nodes, seeds = graph["edges"], graph["nodes"], graph["seeds"]
            global_edges.append(local_edges + offset)
            global_delta.append(edge_delta)
            global_l1.append(edge_l1)
            global_seed[offset : offset + nodes.numel()] = seeds

        edges = torch.cat(global_edges)
        delta_squared = torch.cat(global_delta)
        delta_l1 = torch.cat(global_l1)

        seeded = global_seed != 255

        solvable = torch.cat([item[0]["solvable"] for item in prepared])
        interior = solvable & ~seeded

        # bf16-mixed 下 gate 的 Linear 会被 autocast 降精度, 而求解器按 fp32 验证过
        with torch.autocast(device_type=device.type, enabled=False):
            base = torch.exp(-self.beta * delta_squared).clamp_min(1e-6)
            gate = self.gate(delta_squared, delta_l1, base.detach())
            # 此开关控制算子内部的梯度；训练调用统一停止目标梯度。
            conductance = (base if self.feature_gradients else base.detach()) * gate
            seed_values = torch.where(
                seeded,
                (global_seed == 1).to(conductance.dtype),
                torch.zeros(total, dtype=conductance.dtype, device=device),
            )
            values, used = pcg_solve(
                conductance,
                edges,
                seed_values,
                seeded,
                interior,
                self.steps,
                self.tolerance,
            )
        values = torch.where(solvable, values, torch.full_like(values, 0.5))

        # 无解样本 (例如种子落在组织外) 整体剔除, 不回退成 0.5 参与损失
        kept_indices = []
        soft, valid, foreground = [], [], []
        for index, offset, item in zip(indices, offsets, prepared):
            graph = item[0]
            nodes, size = graph["nodes"], graph["size"]
            table = torch.zeros(size, dtype=values.dtype, device=device)
            table[nodes] = values[offset : offset + nodes.numel()]
            grid = superpixels[index, 0]
            tissue = grid > 0
            pixel = table[grid]
            kept_indices.append(index)
            soft.append(pixel)
            foreground.append(tissue)
            # 组织外 (超像素 id 0) 与模糊区间
            # 都不参与。table[0] 恒为 0, 不加 tissue 约束会把白底当成确定非癌监督。
            valid.append(
                tissue
                & (
                    (pixel <= self.non_cancer_threshold)
                    | (pixel >= self.cancer_threshold)
                )
            )

        result = None
        if kept_indices:
            valid_tensor = torch.stack(valid)
            gate_mean, gate_std, valid_fraction = torch.stack((
                gate.detach().mean(), gate.detach().std(unbiased=False),
                valid_tensor.sum() / torch.stack(foreground).sum().clamp_min(1),
            )).detach().cpu().tolist()
            diagnostics = {
                "pcg_steps": float(used),
                "gate_mean": gate_mean,
                "gate_std": gate_std,
                "valid_fraction": valid_fraction,
            }
            result = (
                kept_indices,
                torch.stack(soft).unsqueeze(1),
                valid_tensor.unsqueeze(1),
                diagnostics,
            )
        return result
