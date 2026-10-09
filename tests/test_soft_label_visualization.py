"""软标签可视化回调: 配置组合与端到端冒烟 (合成小切片)。"""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from segmodel.dynamic.callback import SoftLabelVisualizationCallback
from segmodel.modules.propagation import DifferentiablePropagation


class StubSlide:
    """满足 predict_slide 与回调所需的最小切片接口。"""

    def __init__(self, height, width, label_scale, superpixels):
        self.height = height
        self.width = width
        self.label_scale = label_scale
        generator = np.random.default_rng(0)
        self.image = generator.integers(
            0, 255, (height, width, 3), dtype=np.uint8
        )
        self.tissue_mask_bl = np.ones(
            (height // label_scale, width // label_scale), dtype=bool
        )
        self.superpixels = superpixels

    def inputs(self, top, left, size):
        crop = self.image[top : top + size, left : left + size]
        return {
            "image": torch.from_numpy(crop.transpose(2, 0, 1).astype(np.float32)),
            "patch": [torch.zeros(2, 2)],
            "cls": [torch.zeros(2, 2)],
        }


class StubModel(torch.nn.Module):
    """logits 恒为 0, 特征随机; 网格尺寸与缓存整数一致。"""

    def forward(self, image, patch, cls):
        height = image.shape[-1]
        tile = height // 4
        return {
            "logits": torch.zeros(image.shape[0], 1, tile, tile),
            "propagation_features": torch.randn(image.shape[0], 6, tile, tile),
        }


def build_case():
    label_scale = 4
    superpixels = np.ones((8, 8), dtype=np.int32)
    superpixels[:4, :4] = 1
    superpixels[:4, 4:] = 2
    superpixels[4:, :4] = 3
    superpixels[4:, 4:] = 4
    slide = StubSlide(32, 32, label_scale, superpixels)
    dataset = SimpleNamespace(
        slides={"B1": slide},
        slide_names=["B1"],
        spots_by_slide={
            "B1": {
                "coords": np.array([[6.0, 6.0], [28.0, 28.0]], dtype=np.float32),
                "indices": np.array([0, 1], dtype=np.int64),
            }
        },
        spot_labels=torch.tensor([1, 0]),
        alignment=16,
        label_scale=label_scale,
    )
    resolution = SimpleNamespace(target_mpp=1.0, label_mpp=4.0)
    datamodule = SimpleNamespace(
        train_dataset=dataset,
        cfg=SimpleNamespace(resolution=resolution),
    )
    trainer = SimpleNamespace(current_epoch=0, world_size=1, datamodule=datamodule)
    propagation = DifferentiablePropagation(
        label_scale=label_scale, beta=0.5, steps=40, tolerance=1e-8
    )
    module = SimpleNamespace(
        model=StubModel(),
        propagation=propagation,
        device=torch.device("cpu"),
    )
    tiling = {"crop_size": 32, "stride": 16, "batch_size": 2}
    return trainer, module, tiling


class SoftLabelVisualizationTest(unittest.TestCase):
    def test_should_run_respects_warmup_and_interval(self):
        callback = SoftLabelVisualizationCallback(
            SimpleNamespace(enabled=True, warmup_epochs=2, interval_epochs=3),
            {"crop_size": 32, "stride": 16, "batch_size": 2},
            "unused",
        )
        self.assertFalse(callback._should_run(1))
        self.assertTrue(callback._should_run(2))
        self.assertFalse(callback._should_run(3))
        self.assertFalse(callback._should_run(4))
        self.assertTrue(callback._should_run(5))
        callback.config.enabled = False
        self.assertFalse(callback._should_run(5))
        callback.config.enabled = True
        callback.output_dir = None
        self.assertFalse(callback._should_run(5))

    def test_disabled_writes_nothing(self):
        trainer, module, tiling = build_case()
        with tempfile.TemporaryDirectory() as directory:
            callback = SoftLabelVisualizationCallback(
                SimpleNamespace(enabled=False, warmup_epochs=0, interval_epochs=1),
                tiling,
                Path(directory),
            )
            callback.on_train_epoch_end(trainer, module)
            self.assertEqual(list(Path(directory).rglob("*")), [])

    def test_visualization_writes_soft_labels(self):
        trainer, module, tiling = build_case()
        with tempfile.TemporaryDirectory() as directory:
            callback = SoftLabelVisualizationCallback(
                SimpleNamespace(enabled=True, warmup_epochs=0, interval_epochs=1),
                tiling,
                Path(directory),
            )
            callback.on_train_epoch_end(trainer, module)
            result_dir = Path(directory) / "epoch_0001" / "B1"
            for name in (
                "overview.jpg",
                "soft_probability.png",
                "soft_probability_heatmap.png",
                "soft_pseudo_mask.png",
                "soft_pseudo_mask_overlay.jpg",
                "spot_seeds_overlay.jpg",
                "metadata.json",
            ):
                self.assertTrue((result_dir / name).exists(), name)
            record = json.loads((result_dir / "metadata.json").read_text())
            self.assertEqual(
                record["seed_counts"],
                {"non_cancer": 1, "cancer": 1, "unknown": 0},
            )
            self.assertGreater(record["soft_class_counts"]["0"], 0)
            self.assertGreater(record["soft_class_counts"]["1"], 0)
            self.assertEqual(record["slide"], "B1")
            self.assertEqual(record["epoch"], 1)

    def test_missing_propagation_is_skipped(self):
        trainer, module, tiling = build_case()
        module.propagation = None
        with tempfile.TemporaryDirectory() as directory:
            callback = SoftLabelVisualizationCallback(
                SimpleNamespace(enabled=True, warmup_epochs=0, interval_epochs=1),
                tiling,
                Path(directory),
            )
            callback.on_train_epoch_end(trainer, module)
            self.assertEqual(list(Path(directory).rglob("*")), [])


if __name__ == "__main__":
    unittest.main()
