"""共享状态的联合约束、混合监督、数值稳定性及患者推理检查。"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from segmodel.dataset.datamodule import FoldDataModule
from segmodel.models.segmentation import SegmentationModel
from segmodel.modules.propagation import DifferentiablePropagation
from segmodel.modules.tissue_states import TissueStateDecoder
from segmodel.pipeline import _save_final_model, _test_final_model, _train_segmentation
from segmodel.training.losses import gene_prediction_losses, segmentation_loss
from test_patient_supervision import patient_config, write_fixture
from test_propagation_training import build_batch, build_module


def state_module():
    module = build_module(
        DifferentiablePropagation(label_scale=7, gate_range=0),
        gene_count=8, tissue_state_count=4, tissue_state_dim=6,
    )
    module.image_cfg.loss_weights.segmentation = 0
    module.image_cfg.loss_weights.propagation = 1.5
    module.image_cfg.loss_weights.propagation_initial = 0
    module.image_cfg.loss_weights.gene_prediction = 1
    module.image_cfg.loss_weights.gene_correlation = .5
    module.image_cfg.gene_correlation_min_spots = 3
    module.log_dict = lambda *args, **kwargs: None
    module.log = lambda *args, **kwargs: None
    return module


def forward(module, batch):
    return module(batch['image'], batch['patch'], batch['cls'], batch['pfm_mask'])


class TissueStateTest(unittest.TestCase):
    def test_shared_weights_constrain_both_outputs_and_preserve_momu_rng(self):
        baseline = build_module().eval()
        rng = torch.random.get_rng_state()
        module = state_module().eval()
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        for name, value in baseline.model.state_dict().items():
            if not name.startswith(('propagation.', 'segmentation_head.')):
                torch.testing.assert_close(value, module.model.state_dict()[name], rtol=0, atol=0)
        batch = build_batch(batch=1)
        output, original = forward(module, batch), forward(baseline, batch)
        torch.testing.assert_close(output['features'], original['features'], rtol=0, atol=0)
        self.assertFalse(hasattr(module.model, 'segmentation_head'))
        self.assertIsNone(module.model.gene_head)
        self.assertNotIn('initial_logits', output)
        self.assertNotIn('refinement_gate', output)
        decoder = module.model.tissue_state_decoder
        states = output['tissue_states']
        self.assertEqual(tuple(states.shape), (1, 4, 16, 16))
        self.assertTrue(bool((states >= 0).all()))
        torch.testing.assert_close(states.sum(dim=1), torch.ones_like(states[:, 0]))
        templates = decoder.template_logits.softmax(dim=1)
        torch.testing.assert_close(templates.sum(dim=1), torch.ones(4))
        expected = torch.einsum('bkhw,kg->bghw', states, templates) * output['expression_amplitude']
        torch.testing.assert_close(output['gene_expression'], expected)
        probability = torch.einsum('bkhw,k->bhw', states, decoder.class_logits.sigmoid())
        torch.testing.assert_close(output['logits'].sigmoid()[:, 0], probability)
        torch.testing.assert_close(output['gene_expression'].sum(dim=1, keepdim=True), output['expression_amplitude'])

    def test_expression_only_updates_templates_states_amplitude_and_backbone(self):
        module, batch = state_module(), build_batch(batch=1)
        batch['superpixels'].zero_()
        batch['valid'].zero_()
        batch['tissue'] = torch.ones_like(batch['valid'])
        batch['spot_labels'][0].fill_(255)
        output = forward(module, batch)
        losses = {}
        module._update_propagated_loss(losses, output, batch)
        self.assertEqual(set(losses), {'gene_prediction', 'gene_correlation'})
        (losses['gene_prediction'] + .5 * losses['gene_correlation']).backward()
        decoder = module.model.tissue_state_decoder
        for parameter in (decoder.template_logits, decoder.state_head.weight, decoder.amplitude_head.weight,
                          decoder.encoder[0].weight, module.model.stem.block1[0].weight):
            self.assertGreater(float(parameter.grad.abs().sum()), 0)
        self.assertIsNone(decoder.class_logits.grad)

    def test_segmentation_updates_shared_states_without_expression_supervision(self):
        module, batch = state_module(), build_batch(batch=1)
        output = forward(module, batch)
        segmentation_loss(output['logits'], batch['cancer_probability'], batch['valid']).backward()
        decoder = module.model.tissue_state_decoder
        for parameter in (decoder.class_logits, decoder.state_head.weight, decoder.encoder[0].weight):
            self.assertGreater(float(parameter.grad.abs().sum()), 0)
        self.assertIsNone(decoder.template_logits.grad)
        self.assertIsNone(decoder.amplitude_head.weight.grad)

    def test_mixed_supervision_trains_all_new_parameters_once_in_optimizer(self):
        module, batch = state_module(), build_batch(batch=2)
        batch['tissue'] = torch.ones_like(batch['valid'])
        batch['valid'][1].zero_()
        batch['superpixels'][1].zero_()
        batch['spot_labels'][1].fill_(255)
        optimizer = module.configure_optimizers()['optimizer']
        parameter_ids = [id(p) for group in optimizer.param_groups for p in group['params']]
        for parameter in module.model.tissue_state_decoder.parameters():
            self.assertEqual(parameter_ids.count(id(parameter)), 1)
        templates_before = module.model.tissue_state_decoder.template_logits.detach().clone()
        for _ in range(2):
            optimizer.zero_grad()
            loss = module.training_step(batch)
            self.assertTrue(bool(torch.isfinite(loss)))
            loss.backward()
            self.assertTrue(all(bool(torch.isfinite(p.grad).all()) for p in module.parameters() if p.grad is not None))
            for parameter in module.model.tissue_state_decoder.parameters():
                self.assertGreater(float(parameter.grad.abs().sum()), 0)
            optimizer.step()
        self.assertFalse(torch.equal(templates_before, module.model.tissue_state_decoder.template_logits))

    def test_real_expression_changes_loss_without_changing_image_predictions(self):
        module, batch = state_module().eval(), build_batch(batch=1)
        first = forward(module, batch)
        loss, _ = gene_prediction_losses(first, batch, module.spot_radius_px, module.label_scale)
        batch['gene_expression'] = [target + 2 for target in batch['gene_expression']]
        second = forward(module, batch)
        changed_loss, _ = gene_prediction_losses(second, batch, module.spot_radius_px, module.label_scale)
        for key in ('logits', 'gene_expression', 'tissue_states'):
            torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
        self.assertNotEqual(float(loss['gene_prediction'].detach()), float(changed_loss['gene_prediction'].detach()))

    def test_amp_and_extreme_class_logits_have_finite_losses_and_gradients(self):
        for class_value in (-100., 100.):
            decoder = TissueStateDecoder(5, 8, state_count=4, hidden_dim=6)
            with torch.no_grad():
                decoder.class_logits.fill_(class_value)
            features = torch.randn(2, 5, 4, 4, requires_grad=True)
            with torch.autocast('cpu', dtype=torch.bfloat16):
                output = decoder(features)
                loss = F.binary_cross_entropy_with_logits(output['logits'], torch.rand_like(output['logits']))
                loss = loss + output['gene_expression'].log1p().square().mean()
            self.assertEqual(output['logits'].dtype, torch.float32)
            self.assertEqual(output['gene_expression'].dtype, torch.float32)
            self.assertTrue(bool(torch.isfinite(loss)))
            loss.backward()
            for parameter in decoder.parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(bool(torch.isfinite(parameter.grad).all()))

    def test_invalid_or_conflicting_architecture_is_rejected(self):
        for options in (
            {'tissue_state_count': 1}, {'tissue_state_count': -1},
            {'tissue_state_count': 4, 'joint_prediction': True},
            {'tissue_state_count': 4, 'gene_count': 0},
            {'tissue_state_count': 4, 'gene_count': 8, 'tissue_state_dim': 0},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                build_module(**options)

    def test_default_config_and_pipeline_preserve_states_when_expression_loss_is_zero(self):
        cfg = patient_config()
        self.assertEqual(cfg.experiment.model.image.tissue_state_count, 32)
        self.assertEqual(cfg.experiment.model.image.tissue_state_dim, 96)
        self.assertFalse(cfg.experiment.model.image.joint_prediction)
        with tempfile.TemporaryDirectory() as directory:
            cfg = write_fixture(directory)
            for name, value in dict(
                encoder_chans=[8, 12, 16], decoder_chans=[12, 8], global_dim=8, spatial_dim=4,
                propagation_channels=4, spatial_pool=4, top_k=1, backbone_init='pytorch',
                tissue_state_count=4, tissue_state_dim=6,
            ).items():
                cfg.experiment.model.image[name] = value
            cfg.experiment.training.image.loss_weights.gene_prediction = 0
            cfg.experiment.training.image.loss_weights.gene_correlation = 0
            data = FoldDataModule(['A1', 'A2'], ['C1'], 0, cfg)
            data.setup('fit')
            with patch('segmodel.pipeline.pl.Trainer') as trainer:
                module = _train_segmentation(cfg, data, {}, Path(directory))
            trainer.return_value.fit.assert_called_once_with(module, datamodule=data)
            self.assertEqual(module.model.gene_count, 8)
            self.assertEqual(module.model.gene_names, [f'g{i}' for i in range(8)])
            self.assertEqual(tuple(module.model.tissue_state_decoder.template_logits.shape), (4, 8))

    def test_checkpoint_and_patient_evaluation_use_image_only_shared_states(self):
        module = state_module().eval()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = write_fixture(root)
            for name, value in dict(
                encoder_chans=[8, 12, 16], decoder_chans=[12, 8], global_dim=8, spatial_dim=4,
                propagation_channels=4, spatial_pool=4, top_k=1, backbone_init='pytorch',
                tissue_state_count=4, tissue_state_dim=6,
            ).items():
                cfg.experiment.model.image[name] = value
            module.model.gene_names = [f'g{i}' for i in range(8)]
            tiling = dict(crop_size=112, stride=56, alignment=28, batch_size=2)
            _save_final_model(cfg, module.model, tiling, root)
            artifact = torch.load(root / 'final_model.pt', weights_only=True)
            restored = SegmentationModel(**artifact['model_config']).eval()
            restored.load_state_dict(artifact['state_dict'], strict=True)
            restored.gene_names = artifact['gene_names']
            batch = build_batch(batch=1)
            expected, actual = forward(module, batch), restored(batch['image'], batch['patch'], batch['cls'], batch['pfm_mask'])
            for key in ('logits', 'gene_expression', 'tissue_states', 'expression_amplitude'):
                torch.testing.assert_close(expected[key], actual[key], rtol=0, atol=0)
            data = FoldDataModule(['A1', 'A2'], ['C1'], 0, cfg)
            metrics = _test_final_model(cfg, data, restored, 'cpu', tiling, root)
            self.assertIn('dice', metrics['slides']['C1'])
            self.assertNotIn('initial_dice', metrics['slides']['C1'])
            self.assertNotIn('refinement_dice_gain', metrics['slides']['C1'])
            self.assertNotIn('dice', metrics['slides']['C2'])
            for slide in ('C1', 'C2'):
                self.assertTrue((root / 'test' / slide / 'gene_predictions.npz').exists())
            self.assertTrue((root / 'test/C1/mask.png').exists())
            self.assertFalse((root / 'test/C1/initial_mask.png').exists())


if __name__ == '__main__':
    unittest.main()
