import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch
from omegaconf import OmegaConf
from torch import nn

from segmodel.dataset.segmentation import SegmentationDataset, collate_samples
from segmodel.models import Distiller, GeneVAE
from segmodel.pipeline import _pretrain_gene_vae
from segmodel.training.losses import relation_distillation_loss
from segmodel.training.segmentation_module import SegmentationModule


class RelationLossTests(unittest.TestCase):
    def test_matches_teacher_geometry_under_rotation(self) -> None:
        gene = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
        image = torch.stack((-gene[:, 1], gene[:, 0]), dim=1).requires_grad_()
        loss, pairs, slides = relation_distillation_loss(
            image, gene, torch.zeros(3, dtype=torch.int64), torch.arange(3),
        )
        self.assertAlmostEqual(loss.item(), 0., places=6)
        self.assertEqual((pairs, slides), (3, 1))

    def test_averages_all_batch_pairs_including_cross_slice_pairs(self) -> None:
        image = torch.tensor([[1., 0.]] * 5, requires_grad=True)
        gene = torch.tensor([[1., 0.], [0., 1.], [1., 0.], [1., 0.], [1., 0.]])
        loss, pairs, slides = relation_distillation_loss(
            image, gene, torch.tensor([0, 0, 1, 1, 1]), torch.arange(5),
        )
        # 10 对中有 4 对关系误差为 1, 包括 3 对跨切片关系.
        self.assertAlmostEqual(loss.item(), 0.4, places=6)
        self.assertEqual((pairs, slides), (10, 2))

    def test_single_spot_per_slice_can_form_cross_slice_pair(self) -> None:
        image = torch.tensor([[1., 1.], [0., 1.]], requires_grad=True)
        gene = torch.tensor([[1., 0.], [0., 1.]])
        loss, pairs, slides = relation_distillation_loss(
            image, gene, torch.tensor([0, 1]), torch.tensor([7, 7]),
        )
        self.assertAlmostEqual(loss.item(), 0.5, places=6)
        self.assertEqual((pairs, slides), (1, 2))
        loss.backward()
        self.assertTrue((image.grad.abs().sum(dim=1) > 0).all())

    def test_merges_duplicate_spot_views_before_comparison(self) -> None:
        image = torch.tensor([[2., 0.], [0., 2.], [0., 1.]], requires_grad=True)
        gene = torch.tensor([[1., 0.], [1., 0.], [0., 1.]])
        loss, pairs, slides = relation_distillation_loss(
            image, gene, torch.zeros(3, dtype=torch.int64), torch.tensor([7, 7, 8]),
        )
        self.assertAlmostEqual(loss.item(), 0.5, places=6)
        self.assertEqual((pairs, slides), (1, 1))
        loss.backward()
        self.assertTrue((image.grad[:2].abs().sum(dim=1) > 0).all())

    def test_duplicate_sampling_does_not_increase_pair_count_or_weight(self) -> None:
        image = torch.tensor([[1., 0.], [1., 1.], [0., 1.]])
        gene = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
        original = relation_distillation_loss(
            image, gene, torch.zeros(3, dtype=torch.int64), torch.arange(3),
        )
        rows = torch.tensor([0, 0, 0, 1, 2])
        repeated = relation_distillation_loss(
            image[rows], gene[rows], torch.zeros(5, dtype=torch.int64), rows,
        )
        torch.testing.assert_close(original[0], repeated[0])
        self.assertEqual(original[1:], repeated[1:])

    def test_empty_singleton_and_only_duplicate_spots_have_zero_loss(self) -> None:
        for slide_ids, spot_ids in (
            ([], []), ([0], [0]), ([0, 0], [3, 3]),
        ):
            with self.subTest(slide_ids=slide_ids, spot_ids=spot_ids):
                image = torch.ones(len(spot_ids), 2, requires_grad=True)
                loss, pairs, slides = relation_distillation_loss(
                    image, image.detach(),
                    torch.tensor(slide_ids, dtype=torch.int64),
                    torch.tensor(spot_ids, dtype=torch.int64),
                )
                self.assertEqual((loss.item(), pairs, slides), (0., 0, 0))
                loss.backward()
                self.assertTrue(torch.isfinite(image.grad).all())

    def test_teacher_is_detached_and_autocast_computes_in_float32(self) -> None:
        image = torch.tensor([[1., 1.], [0., 1.]], requires_grad=True)
        gene = torch.tensor([[1., 0.], [0., 1.]], requires_grad=True)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            loss, _, _ = relation_distillation_loss(
                image, gene, torch.tensor([0, 0]), torch.arange(2),
            )
        self.assertEqual(loss.dtype, torch.float32)
        self.assertAlmostEqual(loss.item(), 0.5, places=6)
        loss.backward()
        self.assertGreater(image.grad.abs().sum().item(), 0.)
        self.assertIsNone(gene.grad)


class RelationIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(3)
        self.teacher = GeneVAE(4, latent_dim=3, hidden_dims=(5,), dropout=0.)
        self.distiller = Distiller(SimpleNamespace(feature_channels=3), self.teacher, 0.6)
        expression = torch.randn(5, 4)
        samples = []
        for slide_id, ids, coordinates in (
            (0, [0, 1], [[1., 1.], [2., 2.]]),
            (0, [0, 2], [[2., 1.], [3., 3.]]),
            (1, [3, 4], [[1., 2.], [3., 2.]]),
        ):
            samples.append({
                'spots': torch.tensor(coordinates),
                'gene_expression': expression[ids],
                'spot_labels': torch.zeros(len(ids), dtype=torch.int64),
                'spot_ids': torch.tensor(ids),
                'slide_id': slide_id,
            })
        self.batch = collate_samples(samples)
        self.features = torch.randn(3, 3, 6, 6, requires_grad=True)

    def test_identity_survives_crop_selection(self) -> None:
        dataset = SegmentationDataset.__new__(SegmentationDataset)
        dataset.spot_radius_px = 0.6
        dataset.crop_size = 6
        dataset.slide_ids = {'slice_a': 2}
        dataset.spots_by_slide = {
            'slice_a': {
                'coords': np.array([[2., 2.], [4., 4.], [8., 8.]], dtype=np.float32),
                'indices': np.array([1, 3, 5], dtype=np.int64),
            },
        }
        dataset.gene_expression = torch.randn(6, 4)
        dataset.spot_labels = torch.arange(6)
        first = dataset._spot_targets('slice_a', 0, 0)
        second = dataset._spot_targets('slice_a', 1, 1)
        batch = collate_samples([first, second])
        self.assertEqual(batch['slide_id'].tolist(), [2, 2])
        self.assertEqual([ids.tolist() for ids in batch['spot_ids']], [[1, 3], [1, 3]])
        torch.testing.assert_close(first['gene_expression'], second['gene_expression'])

    def test_cross_crop_and_cross_slice_pairs_with_frozen_teacher(self) -> None:
        self.distiller.train()
        losses, diagnostics = self.distiller.losses(
            {'features': self.features}, self.batch, relation_enabled=True,
        )
        self.assertEqual(diagnostics['distillation/relation_pairs'], 10.)
        self.assertEqual(diagnostics['distillation/relation_slides'], 2.)
        losses['relation'].backward()
        self.assertGreater(self.features.grad.abs().sum().item(), 0.)
        self.assertTrue(any(p.grad is not None for p in self.distiller.image_parameters))
        self.assertFalse(self.teacher.training)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in self.teacher.parameters()))

    def test_disabled_relation_supports_old_batches(self) -> None:
        old_batch = {key: value for key, value in self.batch.items() if key not in ('slide_id', 'spot_ids')}
        losses, diagnostics = self.distiller.losses({'features': self.features}, old_batch)
        self.assertEqual(losses['relation'].item(), 0.)
        self.assertEqual(diagnostics['distillation/relation_pairs'], 0.)
        self.assertTrue(torch.isfinite(losses['molecular']))

    def test_empty_batch_skips_teacher(self) -> None:
        batch = {'spots': [torch.empty(0, 2)]}
        with patch.object(self.teacher, 'embedding') as forward:
            losses, diagnostics = self.distiller.losses(
                {'features': self.features[:1]}, batch, relation_enabled=True,
            )
        forward.assert_not_called()
        self.assertEqual(diagnostics['distillation/relation_pairs'], 0.)
        (losses['molecular'] + losses['relation']).backward()

    def test_training_gates_and_weighted_loss(self) -> None:
        for molecular, relation in ((0.5, 0.1), (0., 0.1), (0.5, 0.), (0., 0.), (0.5, None)):
            with self.subTest(molecular=molecular, relation=relation):
                features = self.features.detach().clone().requires_grad_()
                batch = dict(self.batch)
                batch.update({
                    'image': torch.zeros_like(features),
                    'patch': (), 'cls': (), 'pfm_mask': torch.ones(3, 1, dtype=torch.bool),
                    'cancer_probability': torch.zeros(3, 1, 6, 6),
                    'valid': torch.ones(3, 1, 6, 6, dtype=torch.bool),
                    'superpixels': torch.zeros(3, 1, 6, 6, dtype=torch.int64),
                })
                output = {'features': features, 'logits': features[:, :1], 'affinity_logits': features[:, :2]}
                for branch in ('global', 'spatial'):
                    output[f'pfm_logits_{branch}'] = torch.zeros(3, 1)
                    output[f'pfm_f1_{branch}'] = features.mean(dim=(-2, -1))
                    output[f'pfm_f2_{branch}'] = features.mean(dim=(-2, -1))
                weights = {'segmentation': 1., 'molecular': molecular}
                if relation is not None:
                    weights['relation'] = relation
                harness = nn.Module()
                harness.forward = lambda *args: output
                harness.model = SimpleNamespace(pfm_count=1)
                harness.distiller = self.distiller
                harness.image_cfg = OmegaConf.create({'loss_weights': weights})
                harness.log = Mock()
                harness.log_dict = Mock()
                harness._mixup_segmentation_loss = lambda batch: None
                harness._cutmix_segmentation_loss = lambda batch: None
                harness._update_segmentation_metrics = Mock()
                with patch.object(self.teacher, 'embedding', wraps=self.teacher.embedding) as forward:
                    loss = SegmentationModule.training_step(harness, batch, 0)
                self.assertEqual(forward.call_count, int(molecular != 0 or bool(relation)))
                logged = harness.log_dict.call_args.args[0]
                expected = sum(weight * logged[f'train/{name}'] for name, weight in weights.items())
                torch.testing.assert_close(loss, expected)
                if relation:
                    self.assertEqual(logged['train/distillation/relation_pairs'], 10.)
                loss.backward()
                self.assertTrue(torch.isfinite(features.grad).all())

    def test_relation_only_requires_teacher_pretraining(self) -> None:
        cfg = OmegaConf.create({
            'experiment': {
                'training': {
                    'runtime': {}, 'vae': {'pretrain_epochs': 0},
                    'image': {'loss_weights': {'molecular': 0., 'relation': 0.1}},
                },
                'model': {'gene': {}},
            },
        })
        with self.assertRaisesRegex(ValueError, 'requires gene VAE pretraining'):
            _pretrain_gene_vae(cfg, None, Path('/tmp'))


if __name__ == '__main__':
    unittest.main()
