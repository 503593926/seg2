"""关闭的损失不读取辅助张量；启用的损失保持原来的数值与梯度。"""

import unittest

import torch

from segmodel.training.losses import image_losses, pfm_losses


class ActiveLossesTest(unittest.TestCase):
    def test_disabled_image_and_pfm_losses_need_no_inputs(self):
        self.assertEqual(image_losses({}, {}, weights={}), {})
        self.assertEqual(pfm_losses({}, {}, pfm_count=2, weights={}), {})

    def test_selected_image_losses_preserve_values_and_gradients(self):
        logits = torch.randn(2, 1, 4, 4, requires_grad=True)
        affinity = torch.randn(2, 2, 4, 4, requires_grad=True)
        output = dict(logits=logits, affinity_logits=affinity)
        batch = dict(
            cancer_probability=torch.rand_like(logits),
            superpixels=torch.arange(16).reshape(1, 1, 4, 4).repeat(2, 1, 1, 1) // 2,
            valid=torch.ones_like(logits, dtype=torch.bool),
        )
        expected = image_losses(output, batch)
        for name, output_keys, batch_keys, parameter in (
            ('segmentation', ('logits',), ('cancer_probability', 'valid'), logits),
            ('affinity', ('affinity_logits',), ('superpixels', 'valid'), affinity),
            ('superpixel', ('logits',), ('superpixels', 'valid'), logits),
        ):
            with self.subTest(loss=name):
                actual = image_losses(
                    {key: output[key] for key in output_keys},
                    {key: batch[key] for key in batch_keys}, weights={name: 1},
                )
                self.assertEqual(set(actual), {name})
                torch.testing.assert_close(actual[name], expected[name])
                torch.testing.assert_close(
                    torch.autograd.grad(actual[name], parameter, retain_graph=True)[0],
                    torch.autograd.grad(expected[name], parameter, retain_graph=True)[0],
                )

    def test_selected_pfm_losses_preserve_values_and_gradients(self):
        output = {}
        for branch, rows in (('global', 4), ('spatial', 8)):
            output[f'pfm_logits_{branch}'] = torch.randn(rows, 2, requires_grad=True)
            output[f'pfm_f1_{branch}'] = torch.randn(rows, 3, requires_grad=True)
            output[f'pfm_f2_{branch}'] = torch.randn(rows, 3, requires_grad=True)
        batch = {'pfm_mask': torch.tensor([[True, False], [True, True]])}
        expected = pfm_losses(output, batch, pfm_count=2)
        for name in expected:
            with self.subTest(loss=name):
                branch = name.rsplit('_', 1)[-1]
                keys = (f'pfm_logits_{branch}',) if '_cls_' in name else (
                    f'pfm_f1_{branch}', f'pfm_f2_{branch}',
                )
                actual = pfm_losses(
                    {key: output[key] for key in keys}, batch, pfm_count=2, weights={name: 1},
                )
                self.assertEqual(set(actual), {name})
                torch.testing.assert_close(actual[name], expected[name])
                parameter = output[keys[0]]
                torch.testing.assert_close(
                    torch.autograd.grad(actual[name], parameter, retain_graph=True)[0],
                    torch.autograd.grad(expected[name], parameter, retain_graph=True)[0],
                )


if __name__ == '__main__':
    unittest.main()
