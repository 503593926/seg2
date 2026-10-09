import unittest
from types import SimpleNamespace

import torch
from torch.nn import functional as F

from segmodel.training.distillation import Distiller
from segmodel.models.gene_vae import GeneVAE
from segmodel.training.losses import relation_distillation_loss


class CropRelationTest(unittest.TestCase):
    def test_singleton_crops_do_not_form_cross_crop_pairs(self) -> None:
        image = torch.tensor([[1., 0.], [1., 0.]], requires_grad=True)
        gene = torch.eye(2)
        loss, pairs, crops = relation_distillation_loss(
            image, gene, torch.tensor([0, 1]), torch.tensor([0, 1]),
        )
        assert (loss.item(), pairs, crops) == (0, 0, 0)
        loss.backward()
        torch.testing.assert_close(image.grad, torch.zeros_like(image))


    def test_valid_crops_have_equal_weight_despite_different_pair_counts(self) -> None:
        image = torch.tensor([
            [1., 0., 0.], [1., 0., 0.],
            [1., 0., 0.], [0., 1., 0.], [0., 0., 1.],
            [1., 0., 0.],
        ])
        gene = torch.tensor([
            [1., 0., 0.], [0., 1., 0.],
            [1., 0., 0.], [0., 1., 0.], [0., 0., 1.],
            [0., 1., 0.],
        ])
        loss, pairs, crops = relation_distillation_loss(
            image, gene, torch.tensor([0, 0, 2, 2, 2, 3]), torch.arange(6),
        )
        # 两个有效 crop 的 MSE 分别为 1 和 0, 单 spot crop 不参与平均.
        torch.testing.assert_close(loss, torch.tensor(0.5))
        assert (pairs, crops) == (4, 2)


    def test_repeated_spots_across_crops_keep_independent_views(self) -> None:
        image = torch.tensor([[1., 0.], [1., 0.], [1., 0.], [0., 1.]])
        gene = torch.eye(2).repeat(2, 1)
        loss, pairs, crops = relation_distillation_loss(
            image, gene, torch.tensor([0, 0, 1, 1]), torch.tensor([7, 8, 7, 8]),
        )
        torch.testing.assert_close(loss, torch.tensor(0.5))
        assert (pairs, crops) == (2, 2)


    def test_duplicate_views_are_merged_only_within_their_crop(self) -> None:
        image = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
        gene = torch.tensor([[1., 0.], [1., 0.], [0., 1.]])
        loss, pairs, crops = relation_distillation_loss(
            image, gene, torch.zeros(3, dtype=torch.long), torch.tensor([0, 0, 1]),
        )
        torch.testing.assert_close(loss, torch.tensor(1.))
        assert (pairs, crops) == (1, 1)


    def test_other_crop_features_do_not_change_first_crop_gradients(self) -> None:
        image = torch.tensor([[1., 0.], [1., 1.], [0., 1.], [1., 2.]], requires_grad=True)
        gene = torch.eye(2).repeat(2, 1).requires_grad_()
        crop_ids = torch.tensor([0, 0, 1, 1])
        spot_ids = torch.tensor([0, 1, 0, 1])
        loss, _, _ = relation_distillation_loss(image, gene, crop_ids, spot_ids)
        loss.backward()
        assert image.grad[:2].abs().sum() > 0
        assert gene.grad is None
        changed_image = image.detach().clone()
        changed_image[2:] = torch.tensor([[3., -1.], [-2., 1.]])
        changed_image.requires_grad_()
        changed_loss, _, _ = relation_distillation_loss(changed_image, gene, crop_ids, spot_ids)
        changed_loss.backward()
        torch.testing.assert_close(changed_image.grad[:2], image.grad[:2])


    def test_empty_input_returns_differentiable_zero(self) -> None:
        image = torch.empty(0, 2, requires_grad=True)
        ids = torch.empty(0, dtype=torch.long)
        loss, pairs, crops = relation_distillation_loss(image, torch.empty(0, 2), ids, ids)
        assert (loss.item(), pairs, crops) == (0, 0, 0)
        loss.backward()
        assert image.grad.shape == image.shape


    def test_bfloat16_autocast_uses_float32_relation_loss(self) -> None:
        image = torch.tensor([[1., 0.], [1., 1.]], dtype=torch.bfloat16, requires_grad=True)
        gene = torch.eye(2, dtype=torch.bfloat16)
        with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
            loss, pairs, crops = relation_distillation_loss(
                image, gene, torch.zeros(2, dtype=torch.long), torch.arange(2),
            )
        assert loss.dtype == torch.float32
        torch.testing.assert_close(loss, torch.tensor(0.5))
        assert (pairs, crops) == (1, 1)
        loss.backward()
        assert torch.isfinite(image.grad).all()
        assert image.grad.abs().sum() > 0


    def test_distiller_groups_by_crop_even_with_shared_slide_and_spot_ids(self) -> None:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(7)
            teacher = GeneVAE(input_dim=3, latent_dim=2, hidden_dims=(8,))
            distiller = Distiller(SimpleNamespace(feature_channels=2), teacher, spot_radius_px=1.)
            features = torch.randn(4, 2, 7, 7, requires_grad=True)
            batch = {
                'spots': [
                    torch.tensor([[2., 2.], [4., 4.]]), torch.empty(0, 2),
                    torch.tensor([[2., 2.], [4., 4.]]), torch.tensor([[2., 2.]]),
                ],
                'gene_expression': [
                    torch.randn(2, 3), torch.empty(0, 3), torch.randn(2, 3), torch.randn(1, 3),
                ],
                'spot_ids': [
                    torch.tensor([0, 1]), torch.empty(0, dtype=torch.long),
                    torch.tensor([0, 1]), torch.tensor([2]),
                ],
                'slide_id': torch.tensor([3, 4, 3, 5]),
            }
            batch['gene_expression'][2] = batch['gene_expression'][0].clone()
            expected = []
            for index in (0, 2):
                image = distiller.image_projector(
                    distiller._pool_spots(features[index], batch['spots'][index])
                )
                with torch.no_grad():
                    gene = teacher.embedding(batch['gene_expression'][index])
                expected.append((
                    F.cosine_similarity(image[:1], image[1:])
                    - F.cosine_similarity(gene[:1], gene[1:])
                ).square().mean())
            losses, diagnostics = distiller.losses({'features': features}, batch, relation_enabled=True)
            torch.testing.assert_close(losses['relation'], torch.stack(expected).mean())
            assert diagnostics['distillation/relation_pairs'] == 2
            assert diagnostics['distillation/relation_crops'] == 2
            assert diagnostics['distillation/relation_slides'] == 1
            batch['slide_id'][2] = 9
            other_losses, other_diagnostics = distiller.losses(
                {'features': features}, batch, relation_enabled=True
            )
            torch.testing.assert_close(other_losses['relation'], losses['relation'])
            assert other_diagnostics['distillation/relation_slides'] == 2
            losses['relation'].backward()
            assert features.grad[[0, 2]].abs().sum() > 0
            torch.testing.assert_close(features.grad[[1, 3]], torch.zeros_like(features.grad[[1, 3]]))
            assert any(
                value.grad is not None and value.grad.abs().sum() > 0
                for value in distiller.image_projector.parameters()
            )
            assert all(value.grad is None for value in teacher.parameters())
