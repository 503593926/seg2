"""spot 均值池化、余弦蒸馏、冻结 VAE 与梯度传播的回归检查。"""

import unittest
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from segmodel.training.distillation import Distiller
from segmodel.models.gene_vae import GeneVAE
from segmodel.training.losses import relation_distillation_loss


class Square(nn.Module):
    def forward(self, value):
        return value.square()


class DistillerTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.teacher = GeneVAE(input_dim=3, latent_dim=2, hidden_dims=(8,))
        self.distiller = Distiller(
            SimpleNamespace(feature_channels=2), self.teacher,
            spot_radius_px=2.0, label_scale=2,
        )

    def batch(self, spots, expression):
        return {
            "spots": spots,
            "gene_expression": expression,
            "spot_ids": [torch.arange(len(value)) for value in spots],
            "slide_id": torch.arange(len(spots)),
        }

    def test_molecular_uses_pool_then_project_and_cosine_distance(self):
        self.teacher = GeneVAE(input_dim=2, latent_dim=2, hidden_dims=(8,))
        self.distiller = Distiller(
            SimpleNamespace(feature_channels=2), self.teacher,
            spot_radius_px=2.0, label_scale=2,
        )
        self.distiller.image_projector = Square()
        features = torch.randn(3, 2, 6, 6)
        spots = [
            torch.tensor([[4.0, 4.0], [6.0, 6.0]]),
            torch.empty(0, 2),
            torch.tensor([[4.5, 5.0]]),
        ]
        expression = [torch.randn(2, 2), torch.empty(0, 2), torch.randn(1, 2)]
        expected, wrong_order = [], []
        rows, columns = torch.meshgrid(torch.arange(6), torch.arange(6), indexing="ij")
        for sample, centers in zip(features, spots):
            for x, y in centers / 2:
                disk = (columns - x).square() + (rows - y).square() <= 1
                values = sample[:, disk].T
                expected.append(values.mean(dim=0).square())
                wrong_order.append(values.square().mean(dim=0))
        with torch.no_grad():
            gene = self.teacher.embedding(torch.cat(expression))
        expected_loss = (1 - F.cosine_similarity(torch.stack(expected), gene, dim=1)).mean()
        wrong_loss = (1 - F.cosine_similarity(torch.stack(wrong_order), gene, dim=1)).mean()
        losses, diagnostics = self.distiller.losses(
            {"features": features}, self.batch(spots, expression),
        )
        torch.testing.assert_close(losses["molecular"], expected_loss)
        self.assertFalse(torch.isclose(expected_loss, wrong_loss).item())
        self.assertEqual(diagnostics["spots"], 3)
        self.assertEqual(losses["relation"].item(), 0)

    def test_gradient_reaches_image_and_projector_but_vae_stays_frozen(self):
        self.distiller.train()
        image_model = nn.Conv2d(1, 2, 1)
        optimizer = torch.optim.AdamW(
            list(image_model.parameters()) + list(self.distiller.image_projector.parameters()), lr=0.01,
        )
        image_before = image_model.weight.detach().clone()
        projector_before = self.distiller.image_projector[0].weight.detach().clone()
        teacher_before = {key: value.clone() for key, value in self.teacher.state_dict().items()}
        features = image_model(torch.randn(1, 1, 6, 6))
        features.retain_grad()
        batch = self.batch([torch.tensor([[4.0, 4.0]])], [torch.randn(1, 3)])
        losses, _ = self.distiller.losses({"features": features}, batch)
        losses["molecular"].backward()
        self.assertGreater(features.grad.abs().sum().item(), 0)
        # 只有 spot 覆盖位置收到 molecular 梯度。
        self.assertEqual(features.grad[:, :, 0, 0].abs().sum().item(), 0)
        self.assertGreater(image_model.weight.grad.abs().sum().item(), 0)
        self.assertGreater(self.distiller.image_projector[0].weight.grad.abs().sum().item(), 0)
        self.assertFalse(self.teacher.training)
        self.assertTrue(all(not value.requires_grad and value.grad is None for value in self.teacher.parameters()))
        optimizer.step()
        self.assertFalse(torch.equal(image_before, image_model.weight))
        self.assertFalse(torch.equal(projector_before, self.distiller.image_projector[0].weight))
        for key, value in self.teacher.state_dict().items():
            torch.testing.assert_close(value, teacher_before[key], rtol=0, atol=0)

    def test_relation_preserves_pool_then_project(self):
        features = torch.randn(1, 2, 6, 6, requires_grad=True)
        spots = [torch.tensor([[4.0, 4.0], [6.0, 6.0], [4.0, 4.0]])]
        expression = [torch.randn(3, 3)]
        expression[0][2] = expression[0][0]
        batch = self.batch(spots, expression)
        batch["spot_ids"][0][2] = 0
        rows, columns = torch.meshgrid(torch.arange(6), torch.arange(6), indexing="ij")
        pooled = []
        for x, y in spots[0] / 2:
            disk = (columns - x).square() + (rows - y).square() <= 1
            pooled.append(features[0, :, disk].mean(dim=1))
        with torch.no_grad():
            gene_embedding = self.teacher.embedding(expression[0])
        expected, pairs, _ = relation_distillation_loss(
            self.distiller.image_projector(torch.stack(pooled)), gene_embedding,
            torch.zeros(3, dtype=torch.long), batch["spot_ids"][0],
        )
        losses, diagnostics = self.distiller.losses({"features": features}, batch, relation_enabled=True)
        torch.testing.assert_close(losses["relation"], expected)
        self.assertEqual(pairs, 1)
        self.assertEqual(diagnostics["distillation/relation_pairs"], pairs)

    def test_empty_batch_returns_differentiable_zero(self):
        features = torch.randn(2, 2, 6, 6, requires_grad=True)
        batch = self.batch([torch.empty(0, 2)] * 2, [torch.empty(0, 3)] * 2)
        losses, diagnostics = self.distiller.losses({"features": features}, batch, relation_enabled=True)
        (losses["molecular"] + losses["relation"]).backward()
        torch.testing.assert_close(features.grad, torch.zeros_like(features))
        self.assertTrue(all(torch.isfinite(value).item() for value in losses.values()))
        self.assertEqual(diagnostics["spots"], 0)

    def test_bfloat16_autocast_keeps_loss_and_gradients_finite(self):
        features = torch.randn(1, 2, 6, 6, requires_grad=True)
        batch = self.batch([torch.tensor([[4.0, 4.0]])], [torch.randn(1, 3)])
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            losses, _ = self.distiller.losses({"features": features}, batch)
        self.assertEqual(losses["molecular"].dtype, torch.float32)
        self.assertTrue(torch.isfinite(losses["molecular"]).item())
        losses["molecular"].backward()
        self.assertTrue(torch.isfinite(features.grad).all().item())
        self.assertGreater(features.grad.abs().sum().item(), 0)

    def test_spot_pooling_scales_coordinates_and_radius(self):
        features = torch.arange(2 * 6 * 6, dtype=torch.float32).reshape(2, 6, 6)
        coordinates = torch.tensor([[2.0, 2.0], [4.5, 5.0], [8.0, 8.0]])
        rows, columns = torch.meshgrid(torch.arange(6), torch.arange(6), indexing="ij")
        expected = []
        for x, y in coordinates / 2:
            disk = (columns - x).square() + (rows - y).square() <= 1
            expected.append(features[:, disk].mean(dim=1))
        torch.testing.assert_close(
            self.distiller._pool_spots(features, coordinates), torch.stack(expected),
        )


if __name__ == "__main__":
    unittest.main()
