"""可微标签传播的数值对照: 与 scipy 精确解一致 + gate 梯度与有限差分一致。"""

import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import spsolve

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from segmodel.modules.propagation import (
    DifferentiablePropagation,
    build_edges,
    component_labels,
    node_means,
    pcg_solve,
    rgb_to_lab,
)


def synthetic_crop(size=48, max_id=40, seed=3):
    """用平滑随机场量化出超像素分区, 保证块连通且有空间结构。"""
    from scipy.ndimage import zoom

    rng = np.random.default_rng(seed)
    coarse = rng.normal(size=(size // 4 + 1, size // 4 + 1))
    field = zoom(coarse, 4, order=1)[:size, :size]
    quantile = np.quantile(field, np.linspace(0, 1, max_id + 1)[1:-1])
    superpixels = np.digitize(field, quantile).astype(np.int64) + 1
    return torch.from_numpy(superpixels)


def graph_arrays(superpixels, combined, beta):
    """单样本图: 局部节点索引, 边特征, base conductance。"""
    unique = torch.unique(superpixels)
    nodes = unique[unique > 0]
    size = int(nodes.max()) + 1
    lookup = torch.full((size,), -1, dtype=torch.long)
    lookup[nodes] = torch.arange(nodes.numel())
    edges = build_edges(superpixels)
    local = lookup[edges]
    difference = combined[local[:, 0]] - combined[local[:, 1]]
    delta_squared = difference.square().sum(dim=-1)
    delta_l1 = difference.abs().sum(dim=-1)
    base = torch.exp(-beta * delta_squared).clamp_min(1e-6)
    return nodes, local, delta_squared, delta_l1, base


def reference_solution(superpixels, combined, seeds, beta):
    """scipy 参考: 与 propagate_features 同一套公式。

    combined 是已经标准化好的节点特征 (N, C), 即 node_means 的输出拼接。
    """
    superpixels_np = superpixels.numpy().reshape(-1)
    unique = np.unique(superpixels_np)
    nodes = unique[unique > 0]
    size = int(nodes.max()) + 1

    lookup = -np.ones(size, dtype=np.int64)
    lookup[nodes] = np.arange(nodes.size)
    edges = build_edges(superpixels).numpy()
    local = lookup[edges]
    delta = combined[local[:, 0]] - combined[local[:, 1]]
    conductance = np.maximum(np.exp(-beta * (delta**2).sum(axis=1)), 1e-6)

    first, second = local[:, 0], local[:, 1]
    adjacency = coo_matrix(
        (
            np.concatenate((conductance, conductance)),
            (np.concatenate((first, second)), np.concatenate((second, first))),
        ),
        shape=(nodes.size, nodes.size),
    ).tocsr()
    laplacian = diags(np.asarray(adjacency.sum(axis=1)).ravel()) - adjacency

    seed_values = seeds.numpy()
    seeded = seed_values != 255
    component = component_labels(nodes.size, torch.from_numpy(first), torch.from_numpy(second))
    seeded_components = np.unique(component[seeded])
    solvable = ~seeded & np.isin(component, seeded_components)

    probability = np.full(nodes.size, 0.5)
    probability[seed_values == 1] = 1.0
    probability[seed_values == 0] = 0.0
    interior = np.flatnonzero(solvable)
    if interior.size:
        boundary = np.flatnonzero(seeded)
        rhs = -laplacian[interior][:, boundary] @ probability[boundary]
        probability[interior] = spsolve(laplacian[interior][:, interior], rhs)
    return np.clip(probability, 0.0, 1.0), solvable


class PropagationNumericsTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.size, self.max_id, self.beta = 48, 40, 0.5
        self.superpixels = synthetic_crop(self.size, self.max_id)
        generator = torch.Generator().manual_seed(1)
        self.rgb = torch.rand(3, self.size, self.size, generator=generator)
        self.features = torch.rand(8, self.size, self.size, generator=generator)

        # 数值对照要求双精度: loss≈10 的量级下 float32 分辨不出 1e-4 的参数扰动
        unique = torch.unique(self.superpixels)
        nodes = unique[unique > 0]
        color = node_means(self.rgb.double(), self.superpixels, nodes)
        learned = node_means(self.features.double(), self.superpixels, nodes)
        learned = learned * np.sqrt(3.0 / learned.shape[1])
        self.combined = torch.cat((color, learned), dim=1)
        (
            self.nodes,
            self.local_edges,
            self.delta_squared,
            self.delta_l1,
            self.base,
        ) = graph_arrays(self.superpixels, self.combined, self.beta)

        draw_generator = torch.Generator().manual_seed(7)
        draw = torch.rand(nodes.numel(), generator=draw_generator)
        seeds = torch.full((nodes.numel(),), 255, dtype=torch.long)
        seeds[draw < 0.2] = 0
        seeds[(draw >= 0.2) & (draw < 0.4)] = 1
        self.seeds = seeds

    def test_pcg_matches_scipy(self):
        """无 gate 调制时, PCG 解与 scipy 精确解一致。"""
        seeded = self.seeds != 255
        seed_values = (self.seeds == 1).double()
        labels = component_labels(
            self.nodes.numel(), self.local_edges[:, 0], self.local_edges[:, 1]
        )
        seeded_components = np.unique(labels[seeded.numpy()])
        solvable = torch.from_numpy(np.isin(labels, seeded_components)) & ~seeded
        interior = solvable & ~seeded

        values, used = pcg_solve(
            self.base,
            self.local_edges,
            seed_values,
            seeded,
            interior,
            steps=300,
            tolerance=1e-12,
        )
        reference, reference_solvable = reference_solution(
            self.superpixels, self.combined, self.seeds, self.beta
        )
        self.assertTrue(np.array_equal(solvable.numpy(), reference_solvable))
        error = np.abs(values[interior].numpy() - reference[solvable])
        self.assertLess(
            error.max(), 1e-6, f"max error {error.max():.3e}, pcg steps={used}"
        )

    def _dense_loss(self):
        """用稠密精确解做前向: 直接验证 gate 到 loss 的梯度路径。"""
        gate = self.gate(self.delta_squared, self.delta_l1, self.base)
        conductance = self.base * gate
        node_count = self.nodes.numel()

        adjacent = torch.zeros(node_count, node_count, dtype=conductance.dtype)
        adjacent.index_put_(
            (self.local_edges[:, 0], self.local_edges[:, 1]),
            conductance,
            accumulate=True,
        )
        adjacent = adjacent + adjacent.t()
        laplacian = torch.diag(adjacent.sum(dim=1)) - adjacent

        seeded = self.seeds != 255
        seed_values = (self.seeds == 1).double()
        identity = torch.eye(node_count, dtype=conductance.dtype)
        system = torch.where(seeded[:, None], identity, laplacian)
        right_hand_side = torch.where(seeded, seed_values, torch.zeros_like(seed_values))
        values = torch.linalg.solve(system, right_hand_side)

        target = torch.linspace(0.2, 0.8, node_count, dtype=conductance.dtype)
        return (values * target).sum()

    def test_gate_gradient_matches_finite_difference(self):
        """loss 对 gate 参数的解析梯度与全参数有限差分一致 (稠密精确解)。"""
        self.gate = DifferentiablePropagation(label_scale=1, beta=self.beta).gate.double()
        loss = self._dense_loss()
        parameters = list(self.gate.parameters())
        gradients = torch.autograd.grad(loss, parameters)

        epsilon = 1e-4
        analytic_values = []
        numeric_values = []
        for parameter, gradient in zip(parameters, gradients):
            flat = parameter.detach().flatten()
            flat_gradient = gradient.flatten()
            for index in range(flat.numel()):
                original = flat[index].item()
                step = epsilon * max(abs(original), 1e-3)
                flat[index] = original + step
                with torch.no_grad():
                    forward = self._dense_loss().item()
                flat[index] = original - step
                with torch.no_grad():
                    backward = self._dense_loss().item()
                flat[index] = original
                analytic_values.append(flat_gradient[index].item())
                numeric_values.append((forward - backward) / (2 * step))

        analytic_values = torch.tensor(analytic_values)
        numeric_values = torch.tensor(numeric_values)
        relative = (
            (analytic_values - numeric_values).norm()
            / max(analytic_values.norm().item(), numeric_values.norm().item(), 1e-30)
        )
        absolute = (analytic_values - numeric_values).abs().max()
        print(
            f"checked {analytic_values.numel()} parameters: "
            f"relative={relative:.3e} absolute={absolute:.3e}"
        )
        # float64 中心差分的舍入底: loss≈10 时绝对误差 ~1e-7, 相对误差 ~1e-5
        self.assertLess(relative, 1e-4, f"relative gradient error {relative:.3e}")
        self.assertLess(absolute, 1e-6, f"absolute gradient error {absolute:.3e}")


class InputAlignmentTest(unittest.TestCase):
    """输入网格对齐的回归: 形状错位必须报错, 高分辨率 rgb 必须等价于预下采样。"""

    def test_node_means_rejects_misaligned_grid(self):
        values = torch.rand(3, 64, 64)
        superpixels = torch.randint(0, 5, (8, 8))
        with self.assertRaises(ValueError):
            node_means(values, superpixels, torch.arange(1, 5))

    def test_rgb_to_lab_matches_snic(self):
        """自实现 Lab 与 SNIC 预处理的 rgbtolab 一致。"""
        from tools.snic_features import prepare_snic_lab

        rng = np.random.default_rng(7)
        image_rgb = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)
        image_bgr = image_rgb[..., ::-1].copy()
        reference = prepare_snic_lab(image_bgr)  # (H, W, 3) float64
        ours = rgb_to_lab(
            torch.from_numpy(image_rgb.transpose(2, 0, 1)).float()
        ).permute(1, 2, 0)
        difference = np.abs(ours.numpy() - reference).max()
        self.assertLess(difference, 1e-4, f"max Lab difference {difference:.3e}")

    def test_high_resolution_rgb_equals_pre_downsampled(self):
        """rgb 空间分辨率高于超像素网格时内部面积下采样, 与手工下采样等价。"""
        superpixels = (
            torch.arange(16).reshape(4, 4).repeat_interleave(4, 0).repeat_interleave(4, 1)
            + 1
        )
        features = torch.randn(4, 16, 16, generator=torch.Generator().manual_seed(0))
        rgb_high = torch.rand(3, 64, 64, generator=torch.Generator().manual_seed(1)) * 255
        rgb_low = torch.nn.functional.interpolate(
            rgb_high[None], size=(16, 16), mode="area"
        )[0]

        module = DifferentiablePropagation(label_scale=4, beta=0.5, steps=60)
        spots = torch.tensor([[32.0, 32.0], [56.0, 56.0]])
        labels = torch.tensor([0, 1])
        grid = superpixels[None, None].long()
        high = module(grid, features[None], rgb_high[None], [labels], [spots])
        low = module(grid, features[None], rgb_low[None], [labels], [spots])
        self.assertIsNotNone(high)
        self.assertTrue(torch.equal(high[1], low[1]))
        self.assertEqual(len(high[0]), 1)


class FeatureGradientTest(unittest.TestCase):
    """档 2 (固定核 + 开特征): 默认梯度到不了特征, 打开后与有限差分一致。"""

    def setUp(self):
        torch.manual_seed(0)
        self.size, self.max_id = 48, 40
        self.superpixels = synthetic_crop(self.size, self.max_id)
        generator = torch.Generator().manual_seed(1)
        self.rgb = torch.rand(3, self.size, self.size, generator=generator) * 255
        self.features = torch.rand(8, self.size, self.size, generator=generator)

        # 种子取两个前景超像素的任一点: 一癌一非癌
        flat = self.superpixels.reshape(-1)
        nodes = torch.unique(flat)
        nodes = nodes[nodes > 0]
        labels, spots = [], []
        for label, node in ((0, nodes[0]), (1, nodes[-1])):
            index = int((flat == node).nonzero()[0])
            y, x = divmod(index, self.size)
            labels.append(label)
            spots.append([float(x), float(y)])
        self.labels = torch.tensor(labels)
        self.spots = torch.tensor(spots)

    def _module(self, feature_gradients):
        # 双精度: gate 的 Linear 要与双精度输入同 dtype, 数值梯度才可分辨
        return DifferentiablePropagation(
            label_scale=1,
            beta=0.5,
            steps=120,
            tolerance=0.0,
            gate_range=0.0,
            feature_gradients=feature_gradients,
        ).double()

    def _loss(self, module, features):
        result = module(
            self.superpixels[None, None],
            features[None],
            self.rgb[None],
            [self.labels],
            [self.spots],
        )
        self.assertIsNotNone(result)
        soft = result[1][0, 0]
        target = torch.linspace(0.2, 0.8, soft.numel(), dtype=soft.dtype).reshape(
            soft.shape
        )
        return (soft * target).sum()

    def test_features_detached_by_default(self):
        module = self._module(feature_gradients=False)
        features = self.features.clone().double().requires_grad_(True)
        loss = self._loss(module, features)
        gradient = torch.autograd.grad(loss, features, allow_unused=True)[0]
        self.assertIsNone(gradient, "默认档不应有梯度回到模型特征")

    def test_feature_gradients_match_finite_difference(self):
        module = self._module(feature_gradients=True)
        features = self.features.clone().double().requires_grad_(True)
        loss = self._loss(module, features)
        gradient = torch.autograd.grad(loss, features)[0]
        self.assertGreater(float(gradient.abs().max()), 0.0)

        flat = features.detach().reshape(-1)
        flat_gradient = gradient.reshape(-1)
        generator = torch.Generator().manual_seed(3)
        indices = torch.randint(0, flat.numel(), (6,), generator=generator)
        epsilon = 1e-6
        analytic_values, numeric_values = [], []
        for index in indices:
            original = flat[index].item()
            flat[index] = original + epsilon
            with torch.no_grad():
                forward = self._loss(module, features).item()
            flat[index] = original - epsilon
            with torch.no_grad():
                backward = self._loss(module, features).item()
            flat[index] = original
            analytic_values.append(flat_gradient[index].item())
            numeric_values.append((forward - backward) / (2 * epsilon))

        analytic_values = torch.tensor(analytic_values)
        numeric_values = torch.tensor(numeric_values)
        relative = (
            (analytic_values - numeric_values).norm()
            / max(analytic_values.norm().item(), numeric_values.norm().item(), 1e-30)
        )
        print(f"feature gradient: relative={relative:.3e}")
        # float64 中心差分在 loss~O(100) 上的舍入底约 1e-5 相对量级
        self.assertLess(relative, 1e-3, f"relative gradient error {relative:.3e}")


if __name__ == "__main__":
    unittest.main()
