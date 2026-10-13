"""条件化表达、单次分子反馈、混合监督及图像推理导出的机制检查。"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from segmodel.dataset.datamodule import FoldDataModule
from segmodel.models.segmentation import SegmentationModel
from segmodel.modules.propagation import DifferentiablePropagation
from segmodel.pipeline import _save_final_model, _test_final_model, _train_segmentation
from segmodel.training.losses import gene_prediction_losses
from test_patient_supervision import patient_config, write_fixture
from test_propagation_training import build_batch, build_module, _to_cuda


def joint_module(detach_phenotype=False, **model_options):
    propagation = DifferentiablePropagation(label_scale=7, gate_range=0, feature_gradients=True)
    module = build_module(
        propagation, gene_count=8, joint_prediction=True,
        phenotype_dim=3, molecular_dim=6, detach_phenotype=detach_phenotype,
        **model_options,
    )
    module.image_cfg.loss_weights.segmentation = 0
    module.image_cfg.loss_weights.propagation_initial = 0.5
    module.image_cfg.loss_weights.gene_prediction = 1
    module.log_dict = lambda *args, **kwargs: None
    module.log = lambda *args, **kwargs: None
    return module


def forward(module, batch):
    return module(batch['image'], batch['patch'], batch['cls'], batch['pfm_mask'])


class JointPredictionTest(unittest.TestCase):
    def test_default_config_selects_shared_states_and_both_tasks(self):
        cfg = patient_config()
        self.assertFalse(cfg.experiment.model.image.joint_prediction)
        self.assertEqual(cfg.experiment.model.image.tissue_state_count, 32)
        weights = cfg.experiment.training.image.loss_weights
        self.assertEqual(weights.propagation, 1.0)
        self.assertEqual(weights.propagation_initial, 0)
        self.assertEqual(weights.gene_prediction, 1)
        self.assertEqual(weights.gene_correlation, .5)
        self.assertTrue(cfg.experiment.propagation.enabled)

    def test_zero_residual_preserves_original_momu_and_initial_segmentation(self):
        baseline = build_module(gene_count=8).eval()
        rng = torch.random.get_rng_state()
        joint = joint_module().eval()
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        for name, value in baseline.model.state_dict().items():
            if not name.startswith('gene_head.'):
                torch.testing.assert_close(value, joint.model.state_dict()[name], rtol=0, atol=0)
        batch = build_batch(batch=1)
        original, output = forward(baseline, batch), forward(joint, batch)
        torch.testing.assert_close(output['features'], original['features'], rtol=0, atol=0)
        torch.testing.assert_close(output['initial_logits'], original['logits'], rtol=0, atol=0)
        torch.testing.assert_close(output['logits'], output['initial_logits'], rtol=0, atol=0)
        self.assertEqual(tuple(output['molecular_latent'].shape), (1, 6, 16, 16))
        self.assertEqual(tuple(output['gene_expression'].shape), (1, 8, 16, 16))
        self.assertTrue(bool((output['gene_expression'] >= 0).all()))
        self.assertTrue(bool((output['refinement_gate'] == 0.5).all()))

    def test_expression_depends_on_soft_probability_without_hard_class_change(self):
        module, batch = joint_module().eval(), build_batch(batch=1)
        handle = module.model.segmentation_head.register_forward_hook(lambda *args: torch.ones_like(args[-1]) * .2)
        first = forward(module, batch)
        handle.remove()
        handle = module.model.segmentation_head.register_forward_hook(lambda *args: torch.ones_like(args[-1]) * .4)
        second = forward(module, batch)
        handle.remove()
        torch.testing.assert_close(first['features'], second['features'], rtol=0, atol=0)
        self.assertTrue(bool((first['initial_probability'] > .5).all()))
        self.assertTrue(bool((second['initial_probability'] > .5).all()))
        self.assertGreater(float((first['gene_expression'] - second['gene_expression']).detach().abs().sum()), 0)

    def test_expression_only_gradient_reaches_p0_unless_detached(self):
        batch = build_batch(batch=1)
        batch['superpixels'].zero_()
        batch['tissue'] = torch.ones_like(batch['valid'])
        batch['valid'].zero_()
        batch['spot_labels'][0].fill_(255)
        for detach in (False, True):
            with self.subTest(detach_phenotype=detach):
                module = joint_module(detach)
                output = forward(module, batch)
                losses, count = gene_prediction_losses(output, batch, module.spot_radius_px, module.label_scale)
                self.assertEqual(count, 6)
                losses['gene_prediction'].backward()
                self.assertGreater(float(module.model.gene_head.weight.grad.abs().sum()), 0)
                self.assertGreater(float(module.model.molecular_encoder[0].weight.grad.abs().sum()), 0)
                gradient = module.model.segmentation_head.weight.grad
                if detach:
                    self.assertTrue(gradient is None or float(gradient.abs().sum()) == 0)
                else:
                    self.assertGreater(float(gradient.abs().sum()), 0)

    def test_disabled_condition_removes_p0_dependence_and_gradient_with_same_capacity(self):
        enabled = joint_module().eval()
        module, batch = joint_module(use_seg_condition=False).eval(), build_batch(batch=1)
        self.assertEqual(
            {name: tuple(value.shape) for name, value in enabled.model.state_dict().items()},
            {name: tuple(value.shape) for name, value in module.model.state_dict().items()},
        )
        outputs = []
        for value in (-1., 1.):
            handle = module.model.segmentation_head.register_forward_hook(
                lambda *args, value=value: args[-1] * 0 + value
            )
            try:
                outputs.append(forward(module, batch))
            finally:
                handle.remove()
        self.assertFalse(torch.equal(outputs[0]['initial_probability'], outputs[1]['initial_probability']))
        for name in ('features', 'molecular_latent', 'gene_expression'):
            torch.testing.assert_close(outputs[0][name], outputs[1][name], rtol=0, atol=0)
        output = forward(module, batch)
        losses, count = gene_prediction_losses(output, batch, module.spot_radius_px, module.label_scale)
        self.assertEqual(count, 6)
        losses['gene_prediction'].backward()
        self.assertGreater(float(module.model.molecular_encoder[0].weight.grad.abs().sum()), 0)
        gradient = module.model.segmentation_head.weight.grad
        self.assertTrue(gradient is None or float(gradient.abs().sum()) == 0)

    def test_disabled_feedback_ignores_trained_residual_and_molecular_perturbation(self):
        module, batch = joint_module(use_gene_feedback=False).eval(), build_batch(batch=1)
        with torch.no_grad():
            module.model.refinement_head[-1].weight.fill_(.1)
            module.model.refinement_head[-1].bias.fill_(1)
        with patch.object(module.model.refinement_head, 'forward', side_effect=AssertionError('disabled')):
            first = forward(module, batch)
        handle = module.model.molecular_encoder.register_forward_hook(lambda *args: args[-1] + 1)
        try:
            second = forward(module, batch)
        finally:
            handle.remove()
        for output in (first, second):
            torch.testing.assert_close(output['logits'], output['initial_logits'], rtol=0, atol=0)
            self.assertEqual(float(output['refinement_gate'].detach().abs().sum()), 0)
            self.assertEqual(float(output['refinement_residual'].detach().abs().sum()), 0)
        torch.testing.assert_close(first['logits'], second['logits'], rtol=0, atol=0)
        self.assertGreater(float((first['gene_expression'] - second['gene_expression']).detach().abs().sum()), 0)
        losses = {}
        module._update_propagated_loss(losses, first, batch)
        losses['propagation'].backward()
        self.assertGreater(float(module.model.segmentation_head.weight.grad.abs().sum()), 0)
        self.assertIsNone(module.model.molecular_encoder[0].weight.grad)
        self.assertIsNone(module.model.refinement_head[-1].weight.grad)

    def test_single_residual_feedback_uses_z_and_trains_it_from_segmentation(self):
        module, batch = joint_module().eval(), build_batch(batch=1)
        with torch.no_grad():
            module.model.refinement_head[-1].weight.fill_(.1)
        first = forward(module, batch)
        torch.testing.assert_close(
            first['logits'], first['initial_logits'] + first['refinement_gate'] * first['refinement_residual'],
        )
        handle = module.model.molecular_encoder.register_forward_hook(lambda *args: args[-1] + 1)
        second = forward(module, batch)
        handle.remove()
        torch.testing.assert_close(first['initial_logits'], second['initial_logits'], rtol=0, atol=0)
        self.assertGreater(float((first['logits'] - second['logits']).detach().abs().sum()), 0)
        losses = {}
        module._update_propagated_loss(losses, first, batch)
        self.assertIn('propagation_initial', losses)
        losses['propagation'].backward()
        self.assertGreater(float(module.model.molecular_encoder[0].weight.grad.abs().sum()), 0)
        self.assertGreater(float(module.model.refinement_gate.weight.grad.abs().sum()), 0)

    def test_mixed_batch_optimizes_mse_correlation_and_segmentation_without_queries(self):
        module, batch = joint_module(), build_batch(batch=2)
        module.image_cfg.loss_weights.gene_correlation = .1
        module.image_cfg.gene_correlation_min_spots = 3
        batch['tissue'] = torch.ones_like(batch['valid'])
        batch['superpixels'][1].zero_()
        batch['valid'][1].zero_()
        batch['spot_labels'][1].fill_(255)
        optimizer = module.configure_optimizers()['optimizer']
        for step in range(2):
            optimizer.zero_grad()
            output = forward(module, batch)
            losses = {}
            module._update_propagated_loss(losses, output, batch)
            self.assertEqual(set(losses), {'propagation', 'propagation_initial', 'gene_prediction', 'gene_correlation'})
            loss = losses['propagation'] + .5 * losses['propagation_initial'] + losses['gene_prediction'] + .1 * losses['gene_correlation']
            self.assertTrue(bool(torch.isfinite(loss)))
            loss.backward()
            self.assertGreater(float(module.model.refinement_head[-1].weight.grad.abs().sum()), 0)
            self.assertGreater(float(module.model.gene_head.weight.grad.abs().sum()), 0)
            self.assertTrue(all(bool(torch.isfinite(p.grad).all()) for p in module.parameters() if p.grad is not None))
            if step:
                self.assertGreater(float(module.model.refinement_gate.weight.grad.abs().sum()), 0)
            optimizer.step()

    def test_training_step_skips_disabled_losses_in_mixed_batch(self):
        module, batch = joint_module(), build_batch(batch=2)
        batch['tissue'] = torch.ones_like(batch['valid'])
        batch['valid'][1].zero_()
        batch['superpixels'][1].zero_()
        batch['spot_labels'][1].fill_(255)
        with (
            patch('segmodel.training.losses.segmentation_loss', side_effect=AssertionError('disabled')),
            patch('segmodel.training.losses.affinity_loss', side_effect=AssertionError('disabled')),
            patch('segmodel.training.losses.superpixel_loss', side_effect=AssertionError('disabled')),
            patch('segmodel.training.losses.cross_correlation_loss', side_effect=AssertionError('disabled')),
        ):
            loss = module.training_step(batch)
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()
        self.assertGreater(float(module.model.gene_head.weight.grad.abs().sum()), 0)
        self.assertGreater(float(module.model.refinement_head[-1].weight.grad.abs().sum()), 0)

    def test_initial_segmentation_can_be_supervised_without_final_loss(self):
        module, batch = joint_module(), build_batch(batch=1)
        for name in module.image_cfg.loss_weights:
            module.image_cfg.loss_weights[name] = float(name == 'propagation_initial')
        losses = {}
        module._update_propagated_loss(losses, forward(module, batch), batch)
        self.assertIn('propagation_initial', losses)
        self.assertNotIn('propagation', losses)
        losses['propagation_initial'].backward()
        self.assertGreater(float(module.model.segmentation_head.weight.grad.abs().sum()), 0)

    def test_pipeline_keeps_joint_architecture_when_expression_losses_are_disabled(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = write_fixture(directory)
            values = dict(
                encoder_chans=[8, 12, 16], decoder_chans=[12, 8], global_dim=8, spatial_dim=4,
                propagation_channels=4, spatial_pool=4, top_k=1, backbone_init='pytorch',
                phenotype_dim=3, molecular_dim=6, use_seg_condition=False, use_gene_feedback=False,
                joint_prediction=True, tissue_state_count=0,
            )
            for name, value in values.items():
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
            self.assertFalse(module.model.use_seg_condition)
            self.assertFalse(module.model.use_gene_feedback)
            self.assertIsNotNone(module.model.gene_head)

    def test_checkpoint_restores_all_interaction_switch_combinations(self):
        cfg = patient_config()
        cfg.experiment.dataset.image_feature_maps = ['feature_0', 'feature_1']
        values = dict(
            encoder_chans=[8, 12, 16], decoder_chans=[12, 8], global_dim=8, spatial_dim=4,
            propagation_channels=4, spatial_pool=4, backbone_init='pytorch',
            phenotype_dim=3, molecular_dim=6,
            joint_prediction=True, tissue_state_count=0,
        )
        for name, value in values.items():
            cfg.experiment.model.image[name] = value
        batch = build_batch(batch=1)
        with tempfile.TemporaryDirectory() as directory:
            for condition in (False, True):
                for feedback in (False, True):
                    with self.subTest(condition=condition, feedback=feedback):
                        module = joint_module(use_seg_condition=condition, use_gene_feedback=feedback).eval()
                        module.model.gene_names = [f'g{i}' for i in range(8)]
                        with torch.no_grad():
                            module.model.refinement_head[-1].weight.fill_(.1)
                        cfg.experiment.model.image.use_seg_condition = condition
                        cfg.experiment.model.image.use_gene_feedback = feedback
                        _save_final_model(cfg, module.model, {}, Path(directory))
                        artifact = torch.load(Path(directory) / 'final_model.pt', weights_only=True)
                        restored = SegmentationModel(**artifact['model_config']).eval()
                        restored.load_state_dict(artifact['state_dict'], strict=True)
                        self.assertEqual(restored.use_seg_condition, condition)
                        self.assertEqual(restored.use_gene_feedback, feedback)
                        expected = forward(module, batch)
                        actual = restored(batch['image'], batch['patch'], batch['cls'], batch['pfm_mask'])
                        for name in ('logits', 'gene_expression', 'molecular_latent', 'refinement_gate'):
                            torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), '需要 CUDA')
    def test_joint_training_step_with_bf16_on_cuda(self):
        module, batch = joint_module().cuda(), _to_cuda(build_batch(batch=2))
        batch['tissue'] = torch.ones_like(batch['valid'])
        batch['valid'][1].zero_()
        batch['superpixels'][1].zero_()
        batch['spot_labels'][1].fill_(255)
        optimizer = module.configure_optimizers()['optimizer']
        for _ in range(3):
            optimizer.zero_grad()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = module.training_step(batch)
            self.assertTrue(bool(torch.isfinite(loss)))
            loss.backward()
            self.assertTrue(all(bool(torch.isfinite(p.grad).all()) for p in module.parameters() if p.grad is not None))
            optimizer.step()
        torch.cuda.synchronize()

    def test_patient_evaluation_and_checkpoint_round_trip_use_image_only_joint_model(self):
        module = joint_module().eval()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = write_fixture(root)
            values = dict(
                encoder_chans=[8, 12, 16], decoder_chans=[12, 8], global_dim=8, spatial_dim=4,
                propagation_channels=4, spatial_pool=4, top_k=1, backbone_init='pytorch',
                joint_prediction=True, phenotype_dim=3, molecular_dim=6, detach_phenotype=False,
                tissue_state_count=0, use_seg_condition=True, use_gene_feedback=True,
            )
            for name, value in values.items():
                cfg.experiment.model.image[name] = value
            module.model.gene_names = [f'g{i}' for i in range(8)]
            tiling = dict(crop_size=112, stride=112, alignment=28, batch_size=1)
            _save_final_model(cfg, module.model, tiling, root)
            artifact = torch.load(root / 'final_model.pt', weights_only=True)
            restored = SegmentationModel(**artifact['model_config']).eval()
            restored.load_state_dict(artifact['state_dict'], strict=True)
            restored.gene_names = artifact['gene_names']
            batch = build_batch(batch=1)
            expected = forward(module, batch)
            actual = restored(batch['image'], batch['patch'], batch['cls'], batch['pfm_mask'])
            for name in ('logits', 'initial_logits', 'gene_expression', 'molecular_latent', 'refinement_gate'):
                torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
            data = FoldDataModule(['A1', 'A2'], ['C1'], 0, cfg)
            metrics = _test_final_model(cfg, data, restored, 'cpu', tiling, root)
            self.assertIn('initial_dice', metrics['slides']['C1'])
            self.assertEqual(metrics['slides']['C1']['refinement_dice_gain'], 0)
            self.assertNotIn('dice', metrics['slides']['C2'])
            self.assertTrue((root / 'test/C1/initial_mask.png').exists())
            self.assertEqual(np.load(root / 'test/C1/refinement_gate.npy').shape, (16, 16))
            self.assertTrue((root / 'test/C2/gene_predictions.npz').exists())
            self.assertFalse((root / 'test/C2/initial_mask.png').exists())


if __name__ == '__main__':
    unittest.main()
