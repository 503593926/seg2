"""局部初始化的范围、随机状态、训练和 checkpoint 回归检查。"""
import io
import unittest

import torch
from torch import nn

from segmodel.models.segmentation import SegmentationModel


class ModelInitializationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def build(self, mode):
        torch.manual_seed(37)
        model = SegmentationModel(
            pfm_patch_dims=(6, 8), pfm_cls_dims=(6, 8),
            encoder_chans=(8, 12, 16), decoder_chans=(12, 8),
            global_dim=8, spatial_dim=4, propagation_channels=4,
            backbone_init=mode,
        )
        return model, nn.Linear(model.feature_channels, 4)

    def test_only_image_backbone_changes_and_downstream_rng_is_preserved(self):
        baseline, baseline_projector = self.build('pytorch')
        baseline_rng = torch.get_rng_state().clone()
        model, projector = self.build('convnext')
        self.assertTrue(torch.equal(baseline_rng, torch.get_rng_state()))
        self.assertEqual(list(baseline.state_dict()), list(model.state_dict()))

        # 通过参数名字独立检查范围, 防止误改路由器、调制头和输出分支。
        changed_weights = []
        for name, value in model.state_dict().items():
            ordinary_backbone = name.startswith(('stem.', 'unet.')) and not any(
                part in name for part in ('.cnorm.', '.global_router.', '.spatial_router.')
            )
            if ordinary_backbone and value.ndim >= 2 and name.endswith('.weight'):
                self.assertFalse(torch.equal(value, baseline.state_dict()[name]), name)
                changed_weights.append(value.flatten())
            elif ordinary_backbone and name.endswith('.bias'):
                self.assertEqual(torch.count_nonzero(value).item(), 0, name)
            else:
                torch.testing.assert_close(value, baseline.state_dict()[name], rtol=0, atol=0, msg=name)
            if '.cnorm.' in name:
                self.assertEqual(torch.count_nonzero(value).item(), 0, name)
            if name.endswith('.gamma'):
                torch.testing.assert_close(value, torch.full_like(value, 1e-6), rtol=0, atol=0)
        self.assertAlmostEqual(torch.cat(changed_weights).std().item(), 0.02, delta=0.001)
        for name, value in projector.state_dict().items():
            torch.testing.assert_close(value, baseline_projector.state_dict()[name], rtol=0, atol=0)

    def test_same_seed_repeats_initial_weights(self):
        first, _ = self.build('convnext')
        second, _ = self.build('convnext')
        for name, value in first.state_dict().items():
            torch.testing.assert_close(value, second.state_dict()[name], rtol=0, atol=0)

    def test_training_step_and_checkpoint_round_trip(self):
        model, _ = self.build('convnext')
        image = torch.randint(0, 256, (2, 3, 56, 56), dtype=torch.uint8)
        patch = (torch.randn(2, 6, 4, 4), torch.randn(2, 8, 4, 4))
        cls = (torch.randn(2, 6, 1, 1), torch.randn(2, 8, 1, 1))
        before = model.stem.block1[0].weight.detach().clone()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        output = model(image, patch, cls)
        loss = nn.functional.binary_cross_entropy_with_logits(
            output['logits'], torch.rand_like(output['logits']),
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(model.stem.block1[0].weight.grad.abs().sum().item(), 0)
        for parameter in model.parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all())
        optimizer.step()
        self.assertFalse(torch.equal(before, model.stem.block1[0].weight))

        checkpoint = io.BytesIO()
        torch.save(model.state_dict(), checkpoint)
        checkpoint.seek(0)
        restored, _ = self.build('pytorch')
        restored.load_state_dict(torch.load(checkpoint, weights_only=True), strict=True)
        model.eval()
        restored.eval()
        with torch.no_grad():
            torch.testing.assert_close(
                model(image, patch, cls)['logits'], restored(image, patch, cls)['logits'],
                rtol=0, atol=0,
            )


if __name__ == '__main__':
    unittest.main()
