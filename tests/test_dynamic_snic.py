"""动态 SNIC 的更新时序、患者隔离、实际训练接线和 RW 图像输出。"""

from contextlib import nullcontext
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from segmodel.dataset.datamodule import FoldDataModule
from segmodel.inference.predictor import predict_slide
from segmodel.inference.visualization import save_soft_label_result
from segmodel.pipeline import _train_segmentation
from segmodel.training.callbacks import DatasetEpochCallback
from segmodel.training.dynamic_snic import DynamicSNICCallback, weighted_snic_features
from test_patient_supervision import write_fixture
from test_tissue_states import state_module


def callback_config(**options):
    cfg = OmegaConf.create(dict(
        enabled=True, warmup_epochs=40, interval_epochs=20,
        superpixel_size_um=14., compactness=10., learned_weight=1.,
    ))
    for name, value in options.items():
        cfg[name] = value
    return cfg


def fixture(directory, workers=0):
    cfg = write_fixture(directory)
    cfg.experiment.dataset.num_workers = workers
    cfg.experiment.training.image.batch_size = 1
    data = FoldDataModule(['A1', 'A2'], ['C1'], 0, cfg)
    data.setup('fit')
    return cfg, data


def callback_trainer(data, epoch):
    return SimpleNamespace(
        current_epoch=epoch - 1, world_size=1, datamodule=data,
        precision_plugin=SimpleNamespace(forward_context=nullcontext),
    )


class DynamicSNICTest(unittest.TestCase):
    def test_schedule_updates_after_epochs_40_60_80_100(self):
        dynamic = callback_config(warmup_epochs=40, interval_epochs=20)
        callback = DynamicSNICCallback(dynamic, {}, '/tmp/unused_snic_test', 1.)
        self.assertEqual([epoch for epoch in range(1, 101) if callback._should_run(epoch)], [40, 60, 80, 100])
        dynamic.enabled = False
        self.assertFalse(callback._should_run(40))
        for invalid in ({'warmup_epochs': 0}, {'interval_epochs': 0}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                DynamicSNICCallback(callback_config(**invalid), {}, '/tmp/unused_snic_test', 1.)

    def test_weighted_features_preserve_lab_and_balance_channel_energy(self):
        rng = np.random.default_rng(1)
        lab = rng.normal(size=(10, 12, 3)) * np.array([10, 20, 30])
        learned = rng.normal(size=(10, 12, 96))
        tissue = np.ones((10, 12), dtype=bool)
        tissue[0] = False
        result = weighted_snic_features(lab, learned, tissue, .5)
        self.assertEqual(result.shape, (10, 12, 99))
        np.testing.assert_allclose(result[tissue, :3], lab[tissue], rtol=1e-6)
        self.assertTrue(np.all(result[~tissue] == 0))
        self.assertAlmostEqual(
            float(np.var(result[tissue, 3:], axis=0).sum()),
            float(np.var(lab[tissue], axis=0).sum() * .5), places=3,
        )
        with self.assertRaises(ValueError):
            weighted_snic_features(lab, learned, np.zeros_like(tissue), 1.)

    def test_native_update_uses_f_and_only_changes_labeled_training_topology(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg, data = fixture(directory)
            module = state_module()
            dataset = data.train_dataset
            original_superpixels = dataset.slides['A1'].superpixels.copy()
            original_mask = dataset.slides['A1'].pseudo_mask_uint8.copy()
            original_expression = dataset.gene_expression.clone()
            original_labels = dataset.spot_labels.clone()
            callback = DynamicSNICCallback(callback_config(), dict(crop_size=112, stride=56, alignment=28, batch_size=2), Path(directory) / 'soft_labels', 1.)
            rng_before = torch.random.get_rng_state()
            with patch('segmodel.training.dynamic_snic.predict_slide', wraps=predict_slide) as inference:
                callback.on_train_epoch_end(callback_trainer(data, 39), module)
                inference.assert_not_called()
                callback.on_train_epoch_end(callback_trainer(data, 40), module)
                self.assertEqual(inference.call_count, 1)
                self.assertEqual(inference.call_args.kwargs['feature_key'], 'features')
                self.assertEqual(inference.call_args.args[1].name, 'A1')
            self.assertTrue(module.model.training)
            self.assertTrue(torch.equal(rng_before, torch.random.get_rng_state()))
            output = Path(directory) / 'soft_labels/epoch_0040/A1'
            metadata = json.loads((output / 'metadata.json').read_text())
            self.assertEqual(metadata['feature_channels'], 8)  # F 是 8 维，A 是 4 维。
            self.assertTrue(metadata['rw_solved'])
            self.assertEqual(metadata['rw_scope'], 'full_slide')
            self.assertEqual(metadata['rw_device'], 'cpu')
            self.assertEqual(metadata['seed_counts']['cancer'], 3)
            self.assertEqual(metadata['previous_superpixel_count'], 256)
            self.assertLess(metadata['superpixel_count'], 256)
            updated = np.load(output / 'superpixels.npy')
            np.testing.assert_array_equal(dataset.slides['A1'].superpixels, updated)
            np.testing.assert_array_equal(dataset._supervision('A1', 0, 0)['superpixels'][0].numpy(), updated)
            self.assertFalse(np.array_equal(original_superpixels, updated))
            np.testing.assert_array_equal(dataset.slides['A1'].pseudo_mask_uint8, original_mask)
            torch.testing.assert_close(dataset.gene_expression, original_expression, rtol=0, atol=0)
            torch.testing.assert_close(dataset.spot_labels, original_labels, rtol=0, atol=0)
            self.assertTrue(np.all(dataset.slides['A2'].superpixels == 0))
            self.assertFalse((output.parent / 'A2').exists())
            self.assertFalse((output.parent / 'C1').exists())
            self.assertFalse((output.parent / 'C2').exists())
            # 不写回预处理数据目录。
            source = Path(directory) / cfg.experiment.dataset.data_root / 'label_mpp/A1/snic/superpixels.npy'
            np.testing.assert_array_equal(np.load(source), original_superpixels)
            probability = np.load(output / 'soft_probability.npy')
            supervised = cv2.imread(str(output / 'supervised_mask.png'), cv2.IMREAD_GRAYSCALE) > 0
            low, high = metadata['thresholds']['non_cancer'], metadata['thresholds']['cancer']
            self.assertTrue(np.all((probability[supervised] <= low) | (probability[supervised] >= high)))

    def test_unsolvable_rw_is_saved_as_ignored_with_explicit_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            _, data = fixture(directory)
            module = state_module()
            callback = DynamicSNICCallback(callback_config(), dict(crop_size=112, stride=56, alignment=28, batch_size=1), Path(directory) / 'soft_labels', 1.)
            with patch.object(type(module.propagation), 'forward', return_value=None):
                callback.on_train_epoch_end(callback_trainer(data, 40), module)
            output = Path(directory) / 'soft_labels/epoch_0040/A1'
            metadata = json.loads((output / 'metadata.json').read_text())
            self.assertFalse(metadata['rw_solved'])
            self.assertEqual(metadata['supervised_fraction'], 0.)
            self.assertTrue(np.all(cv2.imread(str(output / 'soft_pseudo_mask.png'), cv2.IMREAD_GRAYSCALE) == 255))

    def test_pipeline_attaches_callback_and_skips_it_for_expression_only_training(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg, data = fixture(directory)
            for name, value in dict(
                encoder_chans=[8, 12, 16], decoder_chans=[12, 8], global_dim=8, spatial_dim=4,
                propagation_channels=4, spatial_pool=4, top_k=1, backbone_init='pytorch',
                tissue_state_count=4, tissue_state_dim=6,
            ).items():
                cfg.experiment.model.image[name] = value
            for enabled in (True, False):
                cfg.experiment.training.image.loss_weights.propagation = 1.5 if enabled else 0.
                cfg.experiment.training.image.loss_weights.propagation_initial = 0.
                with patch('segmodel.pipeline.pl.Trainer') as trainer:
                    _train_segmentation(cfg, data, {}, Path(directory))
                callbacks = [item for item in trainer.call_args.kwargs['callbacks'] if isinstance(item, DynamicSNICCallback)]
                self.assertEqual(len(callbacks), int(enabled))
                if enabled:
                    self.assertEqual(callbacks[0].output_dir, Path(directory) / 'soft_labels')

    def test_two_epoch_lightning_training_reads_updated_topology(self):
        with tempfile.TemporaryDirectory() as directory:
            _, data = fixture(directory, workers=2)
            data.train_dataset.slide_names = ['A1']
            # 父进程先用过 OpenCV 并行池时，worker 初始化也必须能正常返回。
            cv2.setNumThreads(2)
            cv2.resize(np.zeros((512, 512, 3), dtype=np.uint8), (256, 256))
            loader = data.train_dataloader()
            self.assertTrue(loader.persistent_workers)
            self.assertEqual(loader.prefetch_factor, 1)
            self.assertFalse(loader.pin_memory)
            loader.timeout = 15
            module = state_module()
            callback = DynamicSNICCallback(callback_config(warmup_epochs=1, interval_epochs=1), dict(crop_size=112, stride=56, alignment=28, batch_size=1), Path(directory) / 'soft_labels', 1.)
            seen_topologies = []
            original_forward = module.propagation.forward
            def record_forward(superpixels, *args):
                if not torch.is_inference_mode_enabled():
                    seen_topologies.append(int(superpixels.max()))
                return original_forward(superpixels, *args)
            with patch.object(module.propagation, 'forward', side_effect=record_forward):
                # 回调在 inference_mode 下执行，排除其整片诊断，只记录训练调用。
                trainer = pl.Trainer(
                    accelerator='cpu', devices=1, precision='32-true',
                    max_epochs=2, limit_train_batches=1, check_val_every_n_epoch=10,
                    num_sanity_val_steps=0, logger=False, enable_checkpointing=False,
                    enable_progress_bar=False, enable_model_summary=False,
                    callbacks=[DatasetEpochCallback(), callback],
                )
                trainer.fit(module, datamodule=data)
            self.assertEqual(seen_topologies[0], 256)
            self.assertEqual(len(seen_topologies), 2)
            first_update = np.load(Path(directory) / 'soft_labels/epoch_0001/A1/superpixels.npy')
            self.assertEqual(seen_topologies[1], int(first_update.max()))
            for epoch in (1, 2):
                self.assertTrue((Path(directory) / f'soft_labels/epoch_{epoch:04d}/A1/overview.jpg').exists())


class SoftLabelVisualizationTest(unittest.TestCase):
    def test_images_preserve_thresholds_background_and_file_layout(self):
        tissue = np.ones((8, 9), dtype=bool)
        tissue[:, 0] = False
        probability = np.tile(np.linspace(0, 1, 9, dtype=np.float32), (8, 1))
        supervised = tissue & ((probability <= .2) | (probability >= .8))
        supervised[0, -1] = False  # 求解器排除的位置，即使概率高也不能参与监督。
        labels = np.where(tissue, np.arange(72).reshape(8, 9) + 1, 0).astype(np.int32)
        image = np.full((8, 9, 3), 150, dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            record = save_soft_label_result(
                directory, image, tissue, probability, supervised, labels,
                np.array([[2, 2], [7, 5]], dtype=np.float32), np.array([0, 1]),
                model_probability=probability, non_cancer_threshold=.2, cancer_threshold=.8,
                metadata={'epoch': 40, 'slide': 'A1', 'label_mpp': 7.0},
            )
            root = Path(directory)
            for filename in (
                'soft_probability.png', 'soft_probability_heatmap.png', 'soft_pseudo_mask.png',
                'soft_pseudo_mask_overlay.jpg', 'spot_seeds_overlay.jpg', 'overview.jpg', 'metadata.json',
                'superpixels.npy', 'superpixel_labels.png', 'superpixel_boundary_overlay.png',
            ):
                self.assertTrue((root / filename).exists(), filename)
            mask = cv2.imread(str(root / 'soft_pseudo_mask.png'), cv2.IMREAD_GRAYSCALE)
            self.assertTrue(np.all(mask[~supervised] == 255))
            self.assertTrue(np.all(mask[supervised & (probability <= .2)] == 0))
            self.assertTrue(np.all(mask[supervised & (probability >= .8)] == 1))
            heatmap = cv2.imread(str(root / 'soft_probability_heatmap.png'))
            self.assertTrue(np.all(heatmap[~tissue] == 255))
            saved = cv2.imread(str(root / 'soft_probability.png'), cv2.IMREAD_UNCHANGED)
            self.assertEqual(saved.dtype, np.uint16)
            np.testing.assert_allclose(saved.astype(np.float32) / 65535, probability, atol=1 / 65535)
            self.assertEqual(cv2.imread(str(root / 'overview.jpg')).shape, (1024, 1536, 3))
            self.assertEqual(record['seed_counts'], {'non_cancer': 1, 'cancer': 1, 'unknown': 0})


if __name__ == '__main__':
    unittest.main()
