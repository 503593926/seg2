"""端到端冒烟: 传播接进训练步骤后能前向反向, 梯度流到 gate 参数。"""

import sys
import unittest
from pathlib import Path

import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from segmodel.models.segmentation import SegmentationModel
from segmodel.modules.propagation import DifferentiablePropagation
from segmodel.training.distillation import Distiller
from segmodel.training.segmentation_module import SegmentationModule


class _StubGeneVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.latent_dim = 4

    def embedding(self, expression):
        return expression


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
        "spot_ids": [torch.arange(spot_count) for _ in range(batch)],
        "slide_id": torch.zeros(batch, dtype=torch.long),
    }


def build_module(propagation=None, gene_count=0):
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
    )
    distiller = Distiller(
        model,
        _StubGeneVAE(),
        spot_radius_px=50.0,
        label_scale=7,
    )
    image_cfg = OmegaConf.create(
        {
            "lr": 0.001,
            "weight_decay": 0.0,
            "warmup_epochs": 1,
            "epochs": 2,
            "ema_decay": 0.9,
            "loss_weights": {
                "segmentation": 1.0,
                "affinity": 0.0,
                "superpixel": 0.0,
                "molecular": 0.0,
                "relation": 0.0,
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
        distiller,
        image_cfg,
        augment_cfg,
        label_scale=7,
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
        self.assertEqual(len(gate_gradients), len(list(propagation.gate.parameters())))
        total = sum(float(gradient.abs().sum()) for gradient in gate_gradients)
        print(f"loss={loss.item():.4f}, gate gradient L1={total:.3e}")
        self.assertGreater(total, 0.0, "gate 没有收到梯度")

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

    def test_feature_gradients_reach_model_parameters(self):
        """档 2: 打开 feature_gradients 后, 传播损失的回传改变模型参数梯度。"""
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

        difference = (collected[True] - collected[False]).abs()
        print(
            f"feature gradient changes model grads: "
            f"max={float(difference.max()):.3e} mean={float(difference.mean()):.3e}"
        )
        self.assertGreater(
            float(difference.max()), 1e-8, "开特征后模型梯度没有任何变化"
        )

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
        keys = set(cfg.experiment.propagation.keys()) - {"enabled"}
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
                if key != "enabled"
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
        self.assertEqual(len(gate_gradients), len(list(propagation.gate.parameters())))
        total = sum(float(gradient.abs().sum()) for gradient in gate_gradients)
        self.assertGreater(total, 0.0, "gate 在 CUDA 上没有收到梯度")


if __name__ == "__main__":
    unittest.main()
