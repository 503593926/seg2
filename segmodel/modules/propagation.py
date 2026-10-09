"""可微标签传播: 固定拓扑 + 可学边权重, 行归一化 Jacobi 预条件 CG 求解。

设计约束 (与 tools/validate_jacobi_scipy.py 的数值结论一致):
- 固定拓扑: 超像素邻接由 SNIC 决定, 不参与优化, 避免节点生死带来的非平稳
- base conductance 由 Lab + 中间特征差固定, 默认与特征一起 detach, 只学一个
  3 维 gate 调制。梯度只走 gate, 防止模型把传播调成对自己有利的退化解
- feature_gradients=True 时 base 与特征不再 detach, 传播损失回传到模型特征
  (档 2: 固定核 + 开特征, 配合 gate_range=0 使用)。这会让监督信号依赖模型
  自己的表示, 有自指风险, 需要监控软标签是否坍缩成模型预测的副本
- 求解器: 对 Dirichlet 系统跑 Jacobi 预条件 CG。原始度分布不均时裸 CG 500 步
  不收敛, Jacobi 预条件后 ~85 步到机器精度, 不要替换
"""

import numpy as np
import torch
from scipy.sparse import coo_matrix as scipy_coo
from scipy.sparse.csgraph import connected_components
from torch import nn
from torch.nn import functional as F

from segmodel.modules.graph_measurements import node_average


class ConductionGate(nn.Module):
    """每条边一个标量调制, 输入是相邻节点的特征差。初始化接近恒等。

    完全零初始化得到均匀常数并与 base 相乘会退化, 因此末层加小随机扰动:
    训练开始时 g≈1, 传播行为等价于现有的固定核。
    """

    def __init__(self, gate_range=0.3):
        super().__init__()
        self.gate_range = float(gate_range)
        self.network = nn.Sequential(nn.Linear(3, 8), nn.Tanh(), nn.Linear(8, 1))
        nn.init.normal_(self.network[0].weight, std=0.1)
        nn.init.zeros_(self.network[0].bias)
        nn.init.normal_(self.network[2].weight, std=0.01)
        nn.init.zeros_(self.network[2].bias)

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


def node_means(values, superpixels, node_ids):
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
    size = int(node_ids.max()) + 1
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

    scipy 只能吃 CPU 数组, 这里显式搬一次; 每步每样本一次转换, 代价可忽略。
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
    """spot 投票到节点: 只有一致节点成为种子, 冲突/未标注为 255。

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


def _apply_conductance(values, conductance, first_nodes, second_nodes):
    """(L u)_i = Σ_j c_ij (u_i - u_j), 对称边表。"""
    difference = values[first_nodes] - values[second_nodes]
    contribution = torch.zeros_like(values)
    contribution.index_add_(0, first_nodes, difference * conductance)
    contribution.index_add_(0, second_nodes, -difference * conductance)
    return contribution


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
    return u, used


class DifferentiablePropagation(nn.Module):
    """用 crop 内的 spot 真值和模型中间特征生成软监督。

    与 DynamicPropagationCallback 的分工: callback 仍负责按轮次刷新 SNIC 和
    硬伪标签；这里在每个训练步内重算软标签, 让传播算子与分割损失联合优化。
    """

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

    def forward(
        self, superpixels, features, rgb, spot_labels, spot_positions, *, return_graph=False,
        molecular_expression=None, molecular_strength=0.0, molecular_beta=1.0,
        detach_molecular=False,
    ):
        """superpixels (B,1,H,W) int64, features (B,C,H,W), rgb (B,3,H,W) float,
        spot 列表为 batch 字段 (变长)。

        返回 (sample_indices, soft_target, valid, diagnostics), 其中只包含可解样本。
        全部样本都不可解时返回 None。return_graph=True 另返回本次实际使用的
        批图, 供隐藏 spot 的独立监督复用边权, 不额外构造第二套传播关系。
        """
        batch = superpixels.shape[0]
        prepared = [
            self._sample_graph(
                superpixels[index, 0],
                features[index],
                rgb[index],
                spot_labels[index],
                spot_positions[index],
                allow_unseeded=return_graph,
                molecular_expression=None if molecular_expression is None else molecular_expression[index],
                molecular_strength=molecular_strength, molecular_beta=molecular_beta,
                detach_molecular=detach_molecular,
            )
            for index in range(batch)
        ]
        indices = [index for index, item in enumerate(prepared) if item is not None]
        if not indices:
            return (None, None) if return_graph else None
        prepared = [prepared[index] for index in indices]
        return self._solve_batch(superpixels, indices, prepared, return_graph=return_graph)

    def _sample_graph(
        self, superpixels, features, rgb, spot_labels, spot_positions, *, allow_unseeded=False,
        molecular_expression=None, molecular_strength=0.0, molecular_beta=1.0,
        detach_molecular=False,
    ):
        """单样本: 节点/边/边特征/种子。返回 None 表示该样本无有效图。"""
        unique = torch.unique(superpixels)
        nodes = unique[unique > 0]
        if nodes.numel() == 0:
            return None
        edges = build_edges(superpixels)
        if edges.numel() == 0:
            return None

        size = int(nodes.max()) + 1
        lookup = torch.full((size,), -1, dtype=torch.long, device=nodes.device)
        lookup[nodes] = torch.arange(nodes.numel(), device=nodes.device)
        local_edges = lookup[edges]

        # 训练图: rgb 是 1mpp 裁剪图, 超像素在标签网格; 先面积下采样再转 Lab,
        # 与 dynamic 侧的加权特征保持同一套颜色空间。
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
        color = node_means(rgb_to_lab(image), superpixels, nodes)
        learned = features.to(compute_dtype)
        if not self.feature_gradients:
            learned = learned.detach()
        learned = node_means(learned, superpixels, nodes)
        learned = learned * np.sqrt(
            3.0 * self.learned_weight / max(learned.shape[1], 1)
        )
        combined = torch.cat((color, learned), dim=1)
        difference = combined[local_edges[:, 0]] - combined[local_edges[:, 1]]
        edge_delta = difference.square().sum(dim=-1)
        edge_l1 = difference.abs().sum(dim=-1)

        molecular = torch.ones_like(edge_delta)
        if molecular_expression is not None and molecular_strength:
            # 预测表达在线性空间取节点均值, log1p 后比较相邻节点。
            expression = node_average(molecular_expression, superpixels, nodes).log1p()
            if detach_molecular:
                expression = expression.detach()
            distance = (expression[local_edges[:, 0]] - expression[local_edges[:, 1]]).square().mean(1)
            similarity = torch.exp(-molecular_beta * distance)
            molecular = 1 + molecular_strength * (2 * similarity - 1)

        seeds = scatter_seed_values(
            superpixels, spot_labels, spot_positions, nodes, self.label_scale
        )
        if not allow_unseeded and not bool((seeds != 255).any()):
            return None
        return lookup, local_edges, nodes, seeds, edge_delta, edge_l1, molecular

    def _solve_batch(self, superpixels, indices, prepared, *, return_graph=False):
        """批级求解: 节点整体偏移后一次 PCG; 不可解节点保持 0.5。

        返回 (sample_indices, soft, valid, diagnostics), 只覆盖可解样本。
        """
        device = superpixels.device
        offsets = []
        total = 0
        for item in prepared:
            offsets.append(total)
            total += int(item[2].numel())

        global_edges, global_delta, global_l1, global_molecular = [], [], [], []
        global_seed = torch.full((total,), 255, dtype=torch.long, device=device)
        for offset, item in zip(offsets, prepared):
            _, local_edges, nodes, seeds, edge_delta, edge_l1, molecular = item
            global_edges.append(local_edges + offset)
            global_delta.append(edge_delta)
            global_l1.append(edge_l1)
            global_molecular.append(molecular)
            global_seed[offset : offset + nodes.numel()] = seeds

        edges = torch.cat(global_edges)
        delta_squared = torch.cat(global_delta)
        delta_l1 = torch.cat(global_l1)
        molecular = torch.cat(global_molecular)

        seeded = global_seed != 255

        # 连通分量: 只有含种子的连通区域可解, 与现有实现一致
        seeded_numpy = seeded.cpu().numpy()
        solvable_flags = []
        components, component_offset = [], 0
        for offset, item in zip(offsets, prepared):
            _, local_edges, nodes, _, _, _, _ = item
            labels = component_labels(
                int(nodes.numel()), local_edges[:, 0], local_edges[:, 1]
            )
            seeded_labels = np.unique(
                labels[seeded_numpy[offset : offset + nodes.numel()]]
            )
            solvable_flags.append(np.isin(labels, seeded_labels))
            components.append(labels + component_offset)
            component_offset += int(labels.max()) + 1
        solvable = torch.from_numpy(np.concatenate(solvable_flags)).to(device)
        interior = solvable & ~seeded
        if not return_graph and not bool(interior.any()):
            return None

        # bf16-mixed 下 gate 的 Linear 会被 autocast 降精度, 而求解器按 fp32 验证过
        with torch.autocast(device_type=device.type, enabled=False):
            base = torch.exp(-self.beta * delta_squared).clamp_min(1e-6)
            gate = self.gate(delta_squared, delta_l1, base.detach())
            # 默认只让梯度走 gate (base 与特征已 detach); 打开 feature_gradients
            # 后传播损失回传到模型特征, 核仍是同一公式
            visual_conductance = (base if self.feature_gradients else base.detach()) * gate
            conductance = visual_conductance * molecular
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
            _, _, nodes, _, _, _, _ = item
            sample_solvable = solvable[offset : offset + nodes.numel()]
            if not bool(sample_solvable.any()):
                continue
            size = int(nodes.max()) + 1
            table = torch.zeros(size, dtype=values.dtype, device=device)
            table[nodes] = values[offset : offset + nodes.numel()]
            grid = superpixels[index, 0]
            tissue = grid > 0
            pixel = table[grid]
            kept_indices.append(index)
            soft.append(pixel)
            foreground.append(tissue)
            # 监督范围与 build_pseudo_mask 一致: 组织外 (超像素 id 0) 与模糊区间
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
            diagnostics = {
                "pcg_steps": float(used),
                "gate_mean": float(gate.detach().mean()),
                "gate_std": float(gate.detach().std(unbiased=False)),
                "valid_fraction": float(
                    valid_tensor.sum() / torch.stack(foreground).sum().clamp_min(1)
                ),
            }
            result = (
                kept_indices,
                torch.stack(soft).unsqueeze(1),
                valid_tensor.unsqueeze(1),
                diagnostics,
            )
        if not return_graph:
            return result
        return result, {
            "indices": indices,
            "nodes": [item[2] for item in prepared],
            "offsets": offsets,
            "edges": edges,
            "conductance": conductance,
            "visual_conductance": visual_conductance,
            "molecular_modulation": molecular,
            "probability": values,
            "solvable": solvable,
            "seed_labels": global_seed,
            "components": torch.from_numpy(np.concatenate(components)).to(device),
        }
