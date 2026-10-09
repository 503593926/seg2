"""Spot 圆盘观测与图平滑重建。表达在线性空间聚合, 求解梯度用伴随系统。"""

import math

import torch


def spot_node_ids(superpixels, spots, label_scale):
    coordinates = torch.round(spots / label_scale).long()
    height, width = superpixels.shape
    inside = (
        (coordinates[:, 0] >= 0) & (coordinates[:, 0] < width)
        & (coordinates[:, 1] >= 0) & (coordinates[:, 1] < height)
    )
    x = coordinates[:, 0].clamp(0, width - 1)
    y = coordinates[:, 1].clamp(0, height - 1)
    return torch.where(inside, superpixels[y, x], 0)


def _spot_windows(spots, shape, radius_px, label_scale):
    height, width = shape
    radius = radius_px / label_scale
    for center in spots / label_scale:
        x, y = center.tolist()
        left, right = max(0, math.ceil(x - radius)), min(width, math.floor(x + radius) + 1)
        top, bottom = max(0, math.ceil(y - radius)), min(height, math.floor(y + radius) + 1)
        if left >= right or top >= bottom:
            yield top, bottom, left, right, None
            continue
        yy = torch.arange(top, bottom, device=spots.device)[:, None]
        xx = torch.arange(left, right, device=spots.device)[None, :]
        disk = (xx - x).square() + (yy - y).square() <= radius**2
        yield top, bottom, left, right, disk


def spot_measurements(superpixels, spots, nodes, radius_px, label_scale):
    """A[s,n]: spot 圆盘内属于节点 n 的组织像素比例, 非空行和为 1。

    这是等面积加权的相对表达观测, 不把 CP10k 当绝对分子计数。
    不跨白底聚合。坐标与半径由图像网格换算到标签网格。
    """
    rows = []
    for top, bottom, left, right, disk in _spot_windows(spots, superpixels.shape, radius_px, label_scale):
        if disk is None:
            rows.append(torch.zeros(len(nodes), device=nodes.device))
            continue
        ids = superpixels[top:bottom, left:right][disk]
        ids = ids[ids > 0]
        local = torch.searchsorted(nodes, ids.contiguous())
        counts = torch.bincount(local, minlength=len(nodes)).float()
        rows.append(counts / counts.sum().clamp_min(1))
    return torch.stack(rows) if rows else torch.zeros(0, len(nodes), device=nodes.device)


def pool_spot_expression(field, superpixels, spots, radius_px, label_scale):
    """直接聚合稠密预测场, 保留超像素内部的表达变化; 返回预测与有效行。"""
    predictions, valid = [], []
    for top, bottom, left, right, disk in _spot_windows(spots, superpixels.shape, radius_px, label_scale):
        if disk is not None:
            disk = disk & (superpixels[top:bottom, left:right] > 0)
        usable = disk is not None and bool(disk.any())
        valid.append(usable)
        predictions.append(
            field[:, top:bottom, left:right][:, disk].float().mean(1)
            if usable else field.new_zeros(field.shape[0], dtype=torch.float32)
        )
    values = torch.stack(predictions) if predictions else field.new_zeros(0, field.shape[0], dtype=torch.float32)
    return values, torch.tensor(valid, dtype=torch.bool, device=field.device)


def node_average(features, superpixels, nodes):
    """未标准化的节点均值, 用于普通基因辅助预测的公平对照。"""
    ids = superpixels.reshape(-1)
    sums = features.float().new_zeros(features.shape[0], int(nodes.max()) + 1)
    sums.index_add_(1, ids, features.float().flatten(1))
    counts = torch.bincount(ids, minlength=sums.shape[1]).clamp_min(1)
    return (sums / counts).T[nodes]


def select_measurements(matrix, keep):
    """从 COO 观测矩阵取行, 避免构造跨 batch 的稠密矩阵。"""
    row_map = torch.full((matrix.shape[0],), -1, device=keep.device, dtype=torch.long)
    row_map[keep] = torch.arange(int(keep.sum()), device=keep.device)
    rows, columns = matrix.indices()
    selected = row_map[rows] >= 0
    return torch.sparse_coo_tensor(
        torch.stack((row_map[rows[selected]], columns[selected])),
        matrix.values()[selected], (int(keep.sum()), matrix.shape[1]),
    ).coalesce()


def laplacian_product(values, conductance, edges):
    first, second = edges.T
    difference = (values[first] - values[second]) * conductance[:, None]
    product = torch.zeros_like(values)
    product.index_add_(0, first, difference)
    product.index_add_(0, second, -difference)
    return product


def measurement_operator(values, conductance, edges, matrix, smoothing, ridge):
    observed = torch.sparse.mm(matrix, values)
    return (
        torch.sparse.mm(matrix.T, observed)
        + smoothing * laplacian_product(values, conductance, edges)
        + ridge * values
    )


def grouped_sum(values, groups, group_count=None):
    if group_count is None:
        group_count = int(groups.max()) + 1
    result = values.new_zeros(group_count, values.shape[1])
    return result.index_add(0, groups, values)


def measurement_pcg(conductance, edges, matrix, rhs, groups, smoothing, ridge, steps, tolerance):
    """解 (AᵀA + λL + εI)Z = b, 每个 crop × 基因独立判断收敛。"""
    diagonal = rhs.new_full((rhs.shape[0],), ridge)
    diagonal.index_add_(0, matrix.indices()[1], matrix.values().square())
    diagonal.index_add_(0, edges[:, 0], smoothing * conductance)
    diagonal.index_add_(0, edges[:, 1], smoothing * conductance)
    solution = torch.zeros_like(rhs)
    residual = rhs.clone()
    preconditioned = residual / diagonal[:, None]
    direction = preconditioned.clone()
    group_count = int(groups.max()) + 1

    def group_sum(values):
        return grouped_sum(values, groups, group_count)

    rz = group_sum(residual * preconditioned)
    threshold = tolerance**2 * group_sum(rhs.square()).clamp_min(1e-30)
    active = group_sum(residual.square()) > threshold
    for _ in range(steps):
        if not bool(active.any()):
            break
        product = measurement_operator(direction, conductance, edges, matrix, smoothing, ridge)
        denominator = group_sum(direction * product)
        alpha = torch.where(active, rz / denominator.clamp_min(1e-30), 0)
        solution = solution + direction * alpha[groups]
        residual = residual - product * alpha[groups]
        active = group_sum(residual.square()) > threshold
        preconditioned = residual / diagonal[:, None]
        rz_new = group_sum(residual * preconditioned)
        beta = torch.where(active, rz_new / rz.clamp_min(1e-30), 0)
        direction = torch.where(active[groups], preconditioned + direction * beta[groups], 0)
        rz = rz_new
    return solution


class _MeasurementSolve(torch.autograd.Function):
    @staticmethod
    def forward(ctx, conductance, edges, matrix, rhs, groups, smoothing, ridge, steps, tolerance):
        solution = measurement_pcg(
            conductance, edges, matrix, rhs, groups, smoothing, ridge, steps, tolerance
        )
        ctx.save_for_backward(conductance, edges, matrix, solution, groups)
        ctx.settings = smoothing, ridge, steps, tolerance
        return solution

    @staticmethod
    def backward(ctx, gradient):
        conductance, edges, matrix, solution, groups = ctx.saved_tensors
        smoothing, ridge, steps, tolerance = ctx.settings
        # 同一个对称正定系统的伴随解; 不保存 85 步 × 512 基因的迭代图。
        adjoint = measurement_pcg(
            conductance, edges, matrix, gradient, groups, smoothing, ridge, steps, tolerance
        )
        first, second = edges.T
        edge_gradient = -smoothing * (
            (solution[first] - solution[second]) * (adjoint[first] - adjoint[second])
        ).sum(1)
        return edge_gradient, None, None, adjoint, None, None, None, None, None


def reconstruct_expression(conductance, edges, matrix, expression, prior, *,
                           node_groups=None, smoothing=1.0, ridge=1e-3, steps=85, tolerance=1e-6):
    """用种子 spot 的区域观测重建节点表达; expression/prior 均不含 query 真值。"""
    dtype = conductance.dtype
    matrix = matrix.to(dtype=dtype)
    expression, prior = expression.detach().to(dtype), prior.detach().to(dtype)
    rhs = torch.sparse.mm(matrix.T, expression) + ridge * prior
    if node_groups is None:
        node_groups = torch.zeros(rhs.shape[0], dtype=torch.long, device=rhs.device)
    solution = _MeasurementSolve.apply(
        conductance, edges, matrix, rhs, node_groups, smoothing, ridge, steps, tolerance
    )
    with torch.no_grad():
        residual = measurement_operator(solution, conductance, edges, matrix, smoothing, ridge) - rhs
        relative = (
            grouped_sum(residual.square(), node_groups).sqrt()
            / grouped_sum(rhs.square(), node_groups).sqrt().clamp_min(1e-30)
        ).max()
    # 二次重建并不等于非负最优化; 输出投影到非负空间后再作区域观测。
    return solution.clamp_min(0), relative
