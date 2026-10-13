"""RW 软目标监督的前反向、停止梯度和输入配置检查。"""

import sys
import unittest
from pathlib import Path

import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from segmodel.models.segmentation import SegmentationModel
from segmodel.modules.propagation import DifferentiablePropagation
from segmodel.training.segmentation_module import SegmentationModule


def build_batch(batch=2, crop=112, label_scale=7, pfm_count=2, spot_count=6):
    size = crop // label_scale
    generator = torch.Generator().manual_seed(0)
    image = torch.randint(0, 255, (batch, 3, crop, crop), generator=generator).to(torch.uint8)
    patch = tuple(
        torch.randn(batch, 6, crop // 14, crop // 14, generator=generator)
        for _ in range(pfm_count)
    )
    cls = tuple(
        torch.randn(batch, 6, crop // 28 + 1, crop // 28 + 1, generator=generator)
        for _ in range(pfm_count)
    )
    superpixels = torch.zeros(batch, 1, size, size, dtype=torch.long)
    for index in range(batch):
        grid = torch.arange(6, dtype=torch.long)
        rows = grid[:, None].expand(6, 6).reshape(-1)
        columns = grid[None, :].expand(6, 6).reshape(-1)
        block = (rows * 6 + columns + 1).reshape(6, 6)
        superpixels[index, 0] = block.repeat_interleave(
            size // 6 + 1, dim=0
        )[:size, :].repeat_interleave(size // 6 + 1, dim=1)[:, :size]
    spots = [
        torch.stack(
            (
                torch.rand(spot_count, generator=generator) * (crop - 8) + 4,
                torch.rand(spot_count, generator=generator) * (crop - 8) + 4,
            ),
            dim=1,
        )
        for _ in range(batch)
    ]
    spot_labels = [
        (torch.rand(spot_count, generator=generator) > 0.5).long() for _ in range(batch)
    ]
    return {
        "image": image,
        "patch": patch,
        "cls": cls,
        "pfm_mask": torch.ones(batch, pfm_count, dtype=torch.bool),
        "superpixels": superpixels,
        "spots": spots,
        "spot_labels": spot_labels,
        "cancer_probability": (image[:, :1, ::label_scale, ::label_scale] > 128).float(),
        "valid": torch.ones(batch, 1, size, size, dtype=torch.bool),
        "gene_expression": [torch.rand(spot_count, 8, generator=generator) for _ in range(batch)],
    }


def build_module(propagation=None, gene_count=0, **model_options):
    torch.manual_seed(0)
    model = SegmentationModel(
        pfm_patch_dims=(6, 6),
        pfm_cls_dims=(6, 6),
        encoder_chans=(8, 12, 16),
        decoder_chans=(12, 8),
        global_dim=8,
        spatial_dim=4,
        propagation_channels=4,
        backbone_init="pytorch",
        gene_count=gene_count,
        **model_options,
    )
    image_cfg = OmegaConf.create(
        {
            "lr": 0.001,
            "weight_decay": 0.0,
            "warmup_steps": 0.5,
            "epochs": 2,
            "ema_decay": 0.9,
            "expression_transform": "log1p",
            "gene_correlation_min_spots": 8,
            "loss_weights": {
                "segmentation": 1.0,
                "affinity": 0.0,
                "superpixel": 0.0,
                "propagation": 1.0,
                "pfm_cls_global": 0.0,
                "pfm_orth_global": 0.0,
                "pfm_cls_spatial": 0.0,
                "pfm_orth_spatial": 0.0,
            },
        }
    )
    augment_cfg = OmegaConf.create(
        {
            "enabled": False,
            "mixup": {"enabled": False, "weight": 0.0, "start_epoch": 0, "probability": 0.0},
            "cutmix": {"enabled": False, "weight": 0.0, "start_epoch": 0, "probability": 0.0},
        }
    )
    return SegmentationModule(
        model,
        image_cfg,
        augment_cfg,
        label_scale=7,
        spot_radius_px=5,
        propagation=propagation,
    )


def _to_cuda(value):
    if isinstance(value, torch.Tensor):
        return value.cuda()
    if isinstance(value, dict):
        return {key: _to_cuda(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_to_cuda(item) for item in value)
    if isinstance(value, list):
        return [_to_cuda(item) for item in value]
    return value


class PropagationTrainingStepTest(unittest.TestCase):
    def test_fully_seeded_graph_still_supervises_its_seed_nodes(self):
        propagation = DifferentiablePropagation(label_scale=1, gate_range=0)
        grid = torch.tensor([[[[1, 2], [1, 2]]]])
        result = propagation(
            grid, torch.rand(1, 4, 2, 2), torch.rand(1, 3, 2, 2) * 255,
            [torch.tensor([0, 1])], [torch.tensor([[0., 0.], [1., 0.]])],
        )
        self.assertIsNotNone(result)
        self.assertEqual(result[0], [0])
        torch.testing.assert_close(result[1], torch.tensor([[[[0., 1.], [0., 1.]]]]))
        self.assertTrue(bool(result[2].all()))

    def test_training_step_with_propagation(self):
        propagation = DifferentiablePropagation(
            label_scale=7, beta=0.5, steps=60, tolerance=1e-8
        )
        module = build_module(propagation)
        batch = build_batch()
        loss = module.training_step(batch)
        self.assertTrue(torch.isfinite(loss), loss)

        loss.backward()
        gate_gradients = [
            parameter.grad
            for parameter in propagation.gate.parameters()
            if parameter.grad is not None
        ]
        self.assertEqual(gate_gradients, [])
        total = sum(float(p.grad.abs().sum()) for p in module.model.parameters() if p.grad is not None)
        self.assertGreater(total, 0.0)

    def test_background_pixels_excluded_from_supervision(self):
        """组织外 (超像素 id 0) 的像素不能被当成确定非癌监督。"""
        propagation = DifferentiablePropagation(
            label_scale=7, beta=0.5, steps=40, tolerance=1e-8
        )
        module = build_module(propagation)
        batch = build_batch()
        batch["superpixels"][:, :, :6, :6] = 0  # 左上角设为组织外 (网格共 16x16)
        output = module(batch["image"], batch["patch"], batch["cls"], batch["pfm_mask"])
        result = propagation(
            batch["superpixels"].to(torch.long),
            output["features"],
            batch["image"],
            batch["spot_labels"],
            batch["spots"],
        )
        self.assertIsNotNone(result)
        _, soft, valid, diagnostics = result
        self.assertFalse(bool(valid[:, :, :6, :6].any()), "组织外像素不应参与监督")
        self.assertTrue(bool(valid.any()), "组织内应有可监督像素")
        # 背景的 soft 仍是 0 (查表默认值), 由 valid 挡掉, 不是靠改值
        self.assertTrue(bool((soft[:, :, :6, :6] == 0).all()))
        self.assertGreater(float(diagnostics["valid_fraction"]), 0.0)
        self.assertLessEqual(float(diagnostics["valid_fraction"]), 1.0)

    def test_rw_target_gradient_does_not_change_model_gradients(self):
        """训练目标停止梯度；求解器内部开关不改变模型梯度。"""
        batch = build_batch()
        collected = {}
        for flag in (False, True):
            propagation = DifferentiablePropagation(
                label_scale=7,
                beta=0.5,
                steps=40,
                tolerance=1e-8,
                gate_range=0.0,
                feature_gradients=flag,
            )
            # build_module 内部固定 seed, 两次迭代的模型初始化完全相同
            module = build_module(propagation)
            loss = module.training_step(batch)
            loss.backward()
            collected[flag] = torch.cat(
                [
                    parameter.grad.flatten()
                    for parameter in module.model.parameters()
                    if parameter.grad is not None
                ]
            ).clone()

        torch.testing.assert_close(collected[True], collected[False], rtol=0, atol=0)

    def test_propagation_disabled_falls_back_to_baseline(self):
        module = build_module(None)
        batch = build_batch()
        loss = module.training_step(batch)
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNone(module.propagation)

    def test_no_seeds_falls_back_gracefully(self):
        """所有 spot 都在 crop 内但全部未知: 传播返回 None, 训练不崩。"""
        propagation = DifferentiablePropagation(label_scale=7, beta=0.5, steps=30)
        module = build_module(propagation)
        batch = build_batch()
        size = batch["spots"][0].shape[0]
        batch["spot_labels"] = [torch.full((size,), 255, dtype=torch.long) for _ in batch["spot_labels"]]
        loss = module.training_step(batch)
        self.assertTrue(torch.isfinite(loss))

    def test_config_keys_match_constructor(self):
        """防止 propagation 配置与构造函数签名漂移。"""
        import inspect

        from hydra import compose, initialize_config_dir

        config_dir = str((Path(__file__).resolve().parents[1] / "segmodel/configs").resolve())
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(config_name="config", overrides=["experiment=her2st"])
        keys = set(cfg.experiment.propagation.keys()) - {"enabled", "dynamic_snic"}
        parameters = set(
            inspect.signature(DifferentiablePropagation.__init__).parameters
        )
        self.assertTrue(
            keys <= parameters,
            f"配置键不被构造函数接受: {sorted(keys - parameters)}",
        )
        DifferentiablePropagation(
            label_scale=7,
            **{
                key: value
                for key, value in cfg.experiment.propagation.items()
                if key not in {"enabled", "dynamic_snic"}
            },
        )

    @unittest.skipUnless(torch.cuda.is_available(), "需要 CUDA")
    def test_training_step_on_cuda(self):
        """GPU 回归: 连通分量与求解器不能在 CUDA 张量上直接调 numpy。"""
        propagation = DifferentiablePropagation(
            label_scale=7, beta=0.5, steps=60, tolerance=1e-8
        )
        module = build_module(propagation).cuda()
        batch = _to_cuda(build_batch())
        loss = module.training_step(batch)
        self.assertTrue(torch.isfinite(loss), loss)
        loss.backward()
        gate_gradients = [
            parameter.grad
            for parameter in propagation.gate.parameters()
            if parameter.grad is not None
        ]
        self.assertEqual(gate_gradients, [])


if __name__ == "__main__":
    unittest.main()
