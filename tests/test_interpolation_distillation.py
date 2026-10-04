import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch
from omegaconf import OmegaConf
from torch import nn

from segmodel.dataset.segmentation import SegmentationDataset, _augment_geometry, collate_samples
from segmodel.models import Distiller
from segmodel.models.distiller import interpolate_posterior
from segmodel.pipeline import _pretrain_gene_vae
from segmodel.training.segmentation_module import SegmentationModule
from tools.spot_interpolation import get_interpolation_plan


class InterpolationPlanTests(unittest.TestCase):
    def test_complete_square_has_four_edges_and_one_center(self) -> None:
        lattice = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float32)
        pixels = np.array([[2, 2], [6, 2], [2, 6], [6, 6]], dtype=np.float32)
        coords, neighbors, weights = get_interpolation_plan(lattice, pixels)
        self.assertEqual(coords.shape, (5, 2))
        self.assertEqual(neighbors.dtype, np.int64)
        self.assertEqual(weights.dtype, np.float32)
        self.assertEqual(set(map(tuple, coords)), {(4, 2), (2, 4), (4, 4), (4, 6), (6, 4)})
        center = np.flatnonzero((coords == [4, 4]).all(axis=1))[0]
        np.testing.assert_array_equal(neighbors[center], [0, 1, 2, 3])
        np.testing.assert_allclose(weights[center], [0.25] * 4)
        np.testing.assert_allclose(weights.sum(axis=1), 1.)

    def test_missing_corner_does_not_create_center_or_bridge_gaps(self) -> None:
        lattice = np.array([[0, 0], [1, 0], [0, 1], [3, 0]], dtype=np.float32)
        pixels = lattice * 4 + 2
        coords, _, weights = get_interpolation_plan(lattice, pixels)
        self.assertEqual(set(map(tuple, coords)), {(4, 2), (2, 4)})
        np.testing.assert_array_equal(weights, [[0.5, 0.5, 0, 0]] * 2)

    def test_empty_singleton_and_nonadjacent_spots(self) -> None:
        for lattice in ([], [[0, 0]], [[0, 0], [2, 0]]):
            with self.subTest(lattice=lattice):
                lattice = np.array(lattice, dtype=np.float32).reshape(-1, 2)
                coords, neighbors, weights = get_interpolation_plan(lattice, lattice * 4)
                self.assertEqual(coords.shape, (0, 2))
                self.assertEqual(neighbors.shape, (0, 4))
                self.assertEqual(weights.shape, (0, 4))

    def test_rejects_duplicate_and_noninteger_grid_coordinates(self) -> None:
        for lattice in ([[0, 0], [0, 0]], [[0, 0], [0.5, 0]]):
            with self.subTest(lattice=lattice):
                lattice = np.array(lattice, dtype=np.float32)
                with self.assertRaisesRegex(AssertionError, 'unique integer lattice'):
                    get_interpolation_plan(lattice, lattice * 4)


def _dataset() -> SegmentationDataset:
    dataset = SegmentationDataset.__new__(SegmentationDataset)
    dataset.interpolation_enabled = True
    dataset.spot_radius_px = 0.6
    dataset.crop_size = 5
    dataset.label_scale = 1
    coords = np.array([[2, 2], [10, 2], [2, 10], [10, 10]], dtype=np.float32)
    dataset.spots_by_slide = {'a': {'coords': coords, 'indices': np.arange(4, dtype=np.int64)}}
    lattice = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float32)
    queries, neighbors, weights = get_interpolation_plan(lattice, coords)
    dataset.interpolation_plans = {'a': {'coords': queries, 'indices': neighbors, 'weights': weights}}
    dataset.gene_expression = torch.tensor([[0., 1.], [2., 1.], [0., 3.], [2., 3.]])
    dataset.spot_labels = torch.zeros(4, dtype=torch.int64)
    dataset.slide_data = {'a': {'tissue_mask_bl': np.ones((16, 16), dtype=bool)}}
    return dataset


class InterpolationDatasetTests(unittest.TestCase):
    def test_crop_without_measured_spots_keeps_outside_parent_expression(self) -> None:
        dataset = _dataset()
        sample = dataset._spot_targets('a', 4, 4)
        self.assertEqual(sample['spots'].shape, (0, 2))
        torch.testing.assert_close(sample['interpolation_spots'], torch.tensor([[2., 2.]]))
        torch.testing.assert_close(sample['interpolation_expression'], dataset.gene_expression)
        torch.testing.assert_close(sample['interpolation_neighbors'], torch.tensor([[0, 1, 2, 3]]))
        torch.testing.assert_close(sample['interpolation_weights'], torch.full((1, 4), 0.25))

    def test_background_and_boundary_queries_are_excluded_and_collate_handles_empty(self) -> None:
        dataset = _dataset()
        sample = dataset._spot_targets('a', 4, 4)
        boundary = dataset._spot_targets('a', 6, 6)
        self.assertEqual(boundary['interpolation_spots'].shape, (0, 2))
        dataset.slide_data['a']['tissue_mask_bl'][6, 6] = False
        background = dataset._spot_targets('a', 4, 4)
        self.assertEqual(background['interpolation_expression'].shape, (0, 2))
        self.assertEqual(background['interpolation_neighbors'].shape, (0, 4))
        batch = collate_samples([sample, background])
        self.assertEqual([len(v) for v in batch['interpolation_spots']], [1, 0])

    def test_disabled_interpolation_returns_original_fields(self) -> None:
        dataset = _dataset()
        dataset.interpolation_enabled = False
        sample = dataset._spot_targets('a', 0, 0)
        self.assertEqual(set(sample), {'spots', 'gene_expression', 'spot_labels'})

    def test_geometry_moves_queries_with_image_and_keeps_parent_mapping(self) -> None:
        image = torch.zeros(3, 8, 8)
        image[:, 3, 2] = 1
        sample = {
            'spots': torch.tensor([[1., 1.]]),
            'interpolation_spots': torch.tensor([[2., 3.]]),
            'interpolation_neighbors': torch.tensor([[0, 1, 0, 0]]),
            'interpolation_weights': torch.tensor([[0.5, 0.5, 0., 0.]]),
            'image': image,
            'cancer_probability': torch.zeros(1, 8, 8),
            'valid': torch.ones(1, 8, 8, dtype=torch.bool),
            'superpixels': torch.zeros(1, 8, 8, dtype=torch.int64),
            'patch': (image.clone(),), 'cls': (image.clone(),),
        }
        cfg = OmegaConf.create({
            'rotation_quarter_turns': [1],
            'horizontal_flip_probability': 1., 'vertical_flip_probability': 0.,
        })
        result = _augment_geometry(sample, np.random.default_rng(0), 8, cfg)
        torch.testing.assert_close(result['interpolation_spots'], torch.tensor([[4., 5.]]))
        self.assertEqual(result['image'][:, 5, 4].sum().item(), 3.)
        torch.testing.assert_close(result['interpolation_neighbors'], torch.tensor([[0, 1, 0, 0]]))


class _NonlinearTeacher(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.latent_dim = 2
        self.scale = nn.Parameter(torch.ones(2))

    def embedding(self, expression: torch.Tensor) -> torch.Tensor:
        return self.encode(expression)[0]

    def encode(self, expression: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = expression.square() * self.scale
        return mean, torch.zeros_like(mean)


class PosteriorInterpolationTests(unittest.TestCase):
    def test_preserves_posterior_variance_and_parent_disagreement(self) -> None:
        mean = torch.tensor([[0., 2.], [2., 2.]])
        logvar = torch.tensor([[1., 4.], [3., 4.]]).log()
        neighbors = torch.tensor([[0, 1]])
        weights = torch.tensor([[0.5, 0.5]])
        query_mean, query_variance = interpolate_posterior(mean, logvar, neighbors, weights)
        torch.testing.assert_close(query_mean, torch.tensor([[1., 2.]]))
        torch.testing.assert_close(query_variance, torch.tensor([[3., 4.]]))

    def test_equal_parents_do_not_artificially_reduce_posterior_variance(self) -> None:
        mean = torch.tensor([[1., 2.], [1., 2.]])
        logvar = torch.tensor([[4., 9.], [4., 9.]]).log()
        query_mean, query_variance = interpolate_posterior(
            mean, logvar, torch.tensor([[0, 1]]), torch.tensor([[0.5, 0.5]]),
        )
        torch.testing.assert_close(query_mean, mean[:1])
        torch.testing.assert_close(query_variance, torch.tensor([[4., 9.]]))

    def test_zero_weight_padding_is_ignored_and_bf16_inputs_use_float32(self) -> None:
        mean = torch.tensor([[1., 2.], [100., 100.]], dtype=torch.bfloat16)
        logvar = torch.zeros_like(mean)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            query_mean, query_variance = interpolate_posterior(
                mean, logvar, torch.tensor([[0, 1]]), torch.tensor([[1., 0.]]),
            )
        self.assertEqual(query_variance.dtype, torch.float32)
        torch.testing.assert_close(query_mean, torch.tensor([[1., 2.]]))
        torch.testing.assert_close(query_variance, torch.ones(1, 2))


class InterpolationDistillerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.teacher = _NonlinearTeacher()
        self.distiller = Distiller(SimpleNamespace(feature_channels=2), self.teacher, 0.6)
        self.distiller.image_projector = nn.Identity()
        self.batch = collate_samples([{
            'spots': torch.empty(0, 2), 'gene_expression': torch.empty(0, 2),
            'spot_labels': torch.empty(0, dtype=torch.int64),
            'interpolation_spots': torch.tensor([[2., 2.]]),
            'interpolation_expression': torch.tensor([[0., 1.], [2., 1.]]),
            'interpolation_neighbors': torch.tensor([[0, 1, 0, 0]]),
            'interpolation_weights': torch.tensor([[0.5, 0.5, 0., 0.]]),
        }])

    def test_interpolates_latents_instead_of_encoding_mean_expression(self) -> None:
        features = torch.zeros(1, 2, 5, 5)
        features[0, :, 2, 2] = torch.tensor([2., 1.])
        losses, diagnostics = self.distiller.losses(
            {'features': features.requires_grad_()}, self.batch, interpolation_enabled=True,
        )
        self.assertAlmostEqual(losses['interpolation'].item(), 0., places=6)
        self.assertEqual(losses['molecular'].item(), 0.)
        self.assertEqual(diagnostics['distillation/interpolation_spots'], 1.)
        # 均值为 [2, 1], 方差为 [5, 1], 权重为 1 / (1 + 3).
        self.assertAlmostEqual(diagnostics['distillation/interpolation_weight'].item(), 0.25, places=6)

    def test_parent_disagreement_reduces_interpolation_supervision(self) -> None:
        features = torch.ones(1, 2, 5, 5)
        losses, diagnostics = self.distiller.losses(
            {'features': features}, self.batch, interpolation_enabled=True,
        )
        target = torch.tensor([2., 1.])
        expected = (1 - torch.nn.functional.cosine_similarity(torch.ones(2), target, dim=0)) * 0.25
        torch.testing.assert_close(losses['interpolation'], expected)
        self.batch['interpolation_expression'][0] = torch.tensor([[0., 1.], [10., 1.]])
        _, high_disagreement = self.distiller.losses(
            {'features': features}, self.batch, interpolation_enabled=True,
        )
        self.assertGreater(
            high_disagreement['distillation/interpolation_std'].item(),
            diagnostics['distillation/interpolation_std'].item(),
        )
        self.assertLess(
            high_disagreement['distillation/interpolation_weight'].item(),
            diagnostics['distillation/interpolation_weight'].item(),
        )

    def test_scaled_coordinates_bf16_and_gradient_with_frozen_teacher(self) -> None:
        self.distiller.label_scale = 2
        self.distiller.spot_radius_px = 1.2
        self.batch['interpolation_spots'][0] = torch.tensor([[4., 4.]])
        self.distiller.train()
        features = torch.ones(1, 2, 5, 5, requires_grad=True)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            losses, _ = self.distiller.losses(
                {'features': features}, self.batch, interpolation_enabled=True,
            )
        self.assertEqual(losses['interpolation'].dtype, torch.float32)
        losses['interpolation'].backward()
        self.assertGreater(features.grad.abs().sum().item(), 0.)
        self.assertTrue(torch.isfinite(features.grad).all())
        self.assertIsNone(self.teacher.scale.grad)
        self.assertFalse(self.teacher.training)
        self.assertFalse(self.teacher.scale.requires_grad)

    def test_empty_queries_and_disabled_interpolation_skip_teacher(self) -> None:
        features = torch.randn(1, 2, 5, 5, requires_grad=True)
        self.batch['interpolation_spots'][0] = torch.empty(0, 2)
        for enabled in (True, False):
            with self.subTest(enabled=enabled):
                with patch.object(self.teacher, 'encode') as forward:
                    losses, _ = self.distiller.losses(
                        {'features': features}, self.batch, interpolation_enabled=enabled,
                    )
                forward.assert_not_called()
                self.assertEqual(losses['interpolation'].item(), 0.)
                (losses['molecular'] + losses['interpolation']).backward()

    def test_training_weights_and_teacher_gates(self) -> None:
        for molecular, interpolation in ((0.5, 0.1), (0., 0.1), (0.5, 0.), (0., 0.), (0.5, None)):
            with self.subTest(molecular=molecular, interpolation=interpolation):
                features = torch.ones(1, 2, 5, 5, requires_grad=True)
                batch = dict(self.batch)
                batch.update({
                    'image': torch.zeros_like(features), 'patch': (), 'cls': (),
                    'pfm_mask': torch.ones(1, 1, dtype=torch.bool),
                    'cancer_probability': torch.zeros(1, 1, 5, 5),
                    'valid': torch.ones(1, 1, 5, 5, dtype=torch.bool),
                    'superpixels': torch.zeros(1, 1, 5, 5, dtype=torch.int64),
                    'spots': [torch.tensor([[1., 1.]])],
                    'gene_expression': [torch.tensor([[0., 1.]])],
                })
                output = {'features': features, 'logits': features[:, :1], 'affinity_logits': features}
                for branch in ('global', 'spatial'):
                    output[f'pfm_logits_{branch}'] = torch.zeros(1, 1)
                    output[f'pfm_f1_{branch}'] = features.mean(dim=(-2, -1))
                    output[f'pfm_f2_{branch}'] = features.mean(dim=(-2, -1))
                weights = {'segmentation': 1., 'molecular': molecular}
                if interpolation is not None:
                    weights['interpolation'] = interpolation
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
                with patch.object(self.teacher, 'encode', wraps=self.teacher.encode) as forward:
                    loss = SegmentationModule.training_step(harness, batch, 0)
                self.assertEqual(forward.call_count, 2 if interpolation else int(molecular != 0))
                logged = harness.log_dict.call_args.args[0]
                expected = sum(weight * logged[f'train/{name}'] for name, weight in weights.items())
                torch.testing.assert_close(loss, expected)
                loss.backward()
                self.assertTrue(torch.isfinite(features.grad).all())

    def test_interpolation_only_requires_pretraining(self) -> None:
        cfg = OmegaConf.create({
            'experiment': {
                'training': {
                    'runtime': {}, 'vae': {'pretrain_epochs': 0},
                    'image': {'loss_weights': {'molecular': 0., 'interpolation': 0.1}},
                },
                'model': {'gene': {}},
            },
        })
        with self.assertRaisesRegex(ValueError, 'requires gene VAE pretraining'):
            _pretrain_gene_vae(cfg, None, Path('/tmp'))


if __name__ == '__main__':
    unittest.main()
