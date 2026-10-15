"""模板拟合的表达尺度、患者隔离、缓存、可训练性和 EMA/保存接线。"""

import json
from itertools import product
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from segmodel.dataset.datamodule import FoldDataModule
from segmodel.models.segmentation import SegmentationModel
from segmodel.modules.tissue_states import TissueStateDecoder
from segmodel.pipeline import _save_final_model, _train_segmentation
from segmodel.training.template_initialization import initialize_expression_templates
from test_patient_supervision import write_fixture
from test_propagation_training import build_batch


def expression_case(mixed=False):
    patterns = np.array([[8., 2., 0., 0.], [0., 0., 3., 7.]], np.float32)
    values = np.repeat(patterns, 8, axis=0)
    names = ['A1'] * 8 + ['A2'] * 8
    if mixed:
        values = np.vstack((values, .25 * patterns[0] + .75 * patterns[1],
                            .75 * patterns[0] + .25 * patterns[1]))
        names += ['A1', 'A2']
    values = np.vstack((values, [1000., 0., 0., 0.], [0., 0., 0., 1000.]))
    names += ['C1', 'C1']
    adata = ad.AnnData(
        np.log1p(values).astype(np.float32),
        obs=pd.DataFrame({'name': names}, index=[f's{i}' for i in range(len(names))]),
        var=pd.DataFrame(index=['g0', 'g1', 'g2', 'g3']),
    )
    return adata, patterns / patterns.sum(axis=1, keepdims=True)


def model_for(adata, count=2):
    return SimpleNamespace(
        tissue_state_count=count,
        tissue_state_decoder=TissueStateDecoder(3, adata.n_vars, count, hidden_dim=3),
    )


def initialize(model, adata, root, method, **options):
    config = dict(method=method, seed=0, max_iter=1000, kmeans_n_init=5,
                  smoothing=.01, min_counts=0, cache_dir='cache')
    config.update(options)
    return initialize_expression_templates(
        model, adata, ['A1', 'A2'], config, expression_transform='log1p',
        project_root=root, output_dir=Path(root) / 'fold',
    )


def quality_expression_case():
    adata, expected = expression_case()
    adata.obs['normalization_total_counts'] = 200.
    adata.obs.loc['s0', 'normalization_total_counts'] = 100.
    extra = ad.AnnData(
        np.log1p(np.array([[10000., 0., 0., 0.], [0., 0., 0., 0.]], np.float32)),
        obs=pd.DataFrame({'name': ['A1', 'A2'], 'normalization_total_counts': [1., 1000.]},
                         index=['low_count', 'zero_expression']),
        var=adata.var.copy(),
    )
    return ad.concat([adata, extra]), expected


class TemplateInitializationTest(unittest.TestCase):
    def test_count_filter_excludes_outliers_for_all_methods_without_changing_training_data(self):
        adata, expected = quality_expression_case()
        before = adata.copy()
        with tempfile.TemporaryDirectory() as root:
            for method in ('nmf', 'kmeans', 'mix'):
                with self.subTest(method=method):
                    model = model_for(adata, count=4 if method == 'mix' else 2)
                    record = initialize(model, adata, root, method, min_counts=100)
                    self.assertEqual(record['input_spots'], 18)
                    self.assertEqual(record['excluded_low_count_spots'], 1)
                    self.assertEqual(record['excluded_zero_expression_spots'], 1)
                    self.assertEqual(record['fit_spots'], 16)  # counts 恰为 100 的 s0 被保留。
                    templates = model.tissue_state_decoder.template_logits.softmax(dim=1).detach().numpy()
                    for block in (templates[:2], templates[2:]) if method == 'mix' else (templates,):
                        order = np.argsort(block[:, 0])[::-1]
                        np.testing.assert_allclose(block[order], .99 * expected + .01 / 4, atol=2e-3)
                    saved = json.loads((Path(root) / 'fold/template_initialization.json').read_text())
                    self.assertEqual(saved['min_counts'], 100)
                    self.assertEqual(saved['excluded_low_count_spots'], 1)
                    np.testing.assert_array_equal(adata.X, before.X)
                    pd.testing.assert_frame_equal(adata.obs, before.obs)

    def test_count_filter_cache_tracks_threshold_and_eligibility_but_ignores_excluded_data(self):
        adata, _ = quality_expression_case()
        with tempfile.TemporaryDirectory() as root:
            model = model_for(adata)
            first = initialize(model, adata, root, 'kmeans', min_counts=100)
            threshold = initialize(model, adata, root, 'kmeans', min_counts=50)
            self.assertNotEqual(first['cache_key'], threshold['cache_key'])
            self.assertEqual(first['expression_sha256'], threshold['expression_sha256'])
            adata.X[adata.obs.index.get_loc('low_count'), 0] += .1
            adata.obs.loc[adata.obs['name'] == 'C1', 'normalization_total_counts'] = np.nan
            with patch('segmodel.training.template_initialization._fit_patterns', side_effect=AssertionError('cache miss')):
                cached = initialize(model, adata, root, 'kmeans', min_counts=100)
            self.assertTrue(cached['cache_hit'])
            self.assertEqual(first['cache_key'], cached['cache_key'])
            adata.obs.loc['low_count', 'normalization_total_counts'] = 100
            changed = initialize(model, adata, root, 'kmeans', min_counts=100)
            self.assertNotEqual(first['cache_key'], changed['cache_key'])
            self.assertEqual(changed['fit_spots'], 17)

    def test_count_filter_reports_missing_invalid_or_fully_filtered_data(self):
        adata, _ = expression_case()
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError, 'normalization_total_counts'):
                initialize(model_for(adata), adata, root, 'nmf', min_counts=100)
            for threshold in (-1, float('nan'), float('inf')):
                with self.subTest(min_counts=threshold), self.assertRaisesRegex(ValueError, 'min_counts'):
                    initialize(model_for(adata), adata, root, 'nmf', min_counts=threshold)
            adata.obs['normalization_total_counts'] = 1.
            with self.assertRaisesRegex(ValueError, 'No nonzero.*min_counts=100'):
                initialize(model_for(adata), adata, root, 'mix', min_counts=100)
            for value in (-1, float('nan'), float('inf')):
                adata.obs.loc['s0', 'normalization_total_counts'] = value
                with self.subTest(counts=value), self.assertRaisesRegex(ValueError, 'finite and nonnegative'):
                    initialize(model_for(adata), adata, root, 'kmeans', min_counts=100)

    def test_kmeans_templates_are_linear_group_means_and_remain_trainable(self):
        adata, expected = expression_case()
        with tempfile.TemporaryDirectory() as root:
            model = model_for(adata)
            decoder = model.tissue_state_decoder
            classes_before = decoder.class_logits.detach().clone()
            record = initialize(model, adata, root, 'kmeans')
            templates = decoder.template_logits.softmax(dim=1)
            order = np.argsort(templates.detach().numpy()[:, 0])[::-1].copy()
            np.testing.assert_allclose(templates.detach().numpy()[order], .99 * expected + .01 / 4, atol=1e-6)
            torch.testing.assert_close(decoder.class_logits, classes_before, rtol=0, atol=0)
            self.assertEqual(record['fit_spots'], 16)
            self.assertEqual(record['train_names'], ['A1', 'A2'])
            self.assertFalse(record['cache_hit'])
            loss = decoder(torch.randn(2, 3, 4, 5))['gene_expression'].log1p().square().mean()
            loss.backward()
            self.assertGreater(float(decoder.template_logits.grad.abs().sum()), 0)
            self.assertTrue(torch.isfinite(decoder.template_logits.grad).all())

    def test_nmf_recovers_composable_expression_programs_and_handles_sparse_data(self):
        adata, expected = expression_case(mixed=True)
        adata.X = sparse.csr_matrix(adata.X)
        with tempfile.TemporaryDirectory() as root:
            model = model_for(adata)
            record = initialize(model, adata, root, 'nmf')
            templates = model.tissue_state_decoder.template_logits.softmax(dim=1).detach().numpy()
            order = np.argsort(templates[:, 0])[::-1]
            np.testing.assert_allclose(templates[order], .99 * expected + .01 / 4, atol=2e-3)
            self.assertLess(record['reconstruction_error'], .05)
            self.assertEqual(record['fit_spots'], 18)

    def test_mix_combines_nmf_programs_and_kmeans_means_for_even_and_odd_counts(self):
        adata, expected = expression_case()
        with tempfile.TemporaryDirectory() as root:
            for count in (4, 3):
                with self.subTest(state_count=count):
                    model = model_for(adata, count=count)
                    record = initialize(model, adata, root, 'mix')
                    templates = model.tissue_state_decoder.template_logits.softmax(dim=1).detach().numpy()
                    self.assertEqual(templates.shape, (count, adata.n_vars))
                    self.assertEqual(record['nmf_state_count'], 2)
                    self.assertEqual(record['kmeans_state_count'], count - 2)
                    self.assertLess(record['nmf']['reconstruction_error'], .05)
                    self.assertGreater(record['kmeans']['iterations'], 0)
                    nmf_order = np.argsort(templates[:2, 0])[::-1]
                    np.testing.assert_allclose(templates[:2][nmf_order], .99 * expected + .01 / 4, atol=2e-3)
                    if count == 4:
                        kmeans_order = np.argsort(templates[2:, 0])[::-1]
                        np.testing.assert_allclose(templates[2:][kmeans_order], .99 * expected + .01 / 4, atol=1e-6)
                    else:
                        np.testing.assert_allclose(templates[2], .99 * expected.mean(axis=0) + .01 / 4, atol=1e-6)
                    np.testing.assert_allclose(templates.sum(axis=1), 1, atol=1e-6)
                    self.assertTrue(np.all(templates > 0))

    def test_mix_cache_tracks_kmeans_restarts_and_ignores_heldout_expression(self):
        adata, _ = expression_case()
        with tempfile.TemporaryDirectory() as root:
            model = model_for(adata, count=4)
            first = initialize(model, adata, root, 'mix')
            initial = model.tissue_state_decoder.template_logits.detach().clone()
            adata.X[-2:] = 100
            with patch('segmodel.training.template_initialization._fit_patterns', side_effect=AssertionError('cache miss')):
                cached = initialize(model, adata, root, 'mix')
            self.assertTrue(cached['cache_hit'])
            self.assertEqual(first['cache_key'], cached['cache_key'])
            torch.testing.assert_close(model.tissue_state_decoder.template_logits, initial, rtol=0, atol=0)
            changed = initialize(model, adata, root, 'mix', kmeans_n_init=2)
            self.assertNotEqual(first['cache_key'], changed['cache_key'])
            self.assertFalse(changed['cache_hit'])

    def test_cache_ignores_heldout_expression_and_preserves_global_random_states(self):
        adata, _ = expression_case()
        with tempfile.TemporaryDirectory() as root:
            model = model_for(adata)
            torch_before = torch.random.get_rng_state().clone()
            numpy_before = np.random.get_state()
            first = initialize(model, adata, root, 'kmeans')
            templates = model.tissue_state_decoder.template_logits.detach().clone()
            self.assertTrue(torch.equal(torch_before, torch.random.get_rng_state()))
            numpy_after = np.random.get_state()
            np.testing.assert_array_equal(numpy_before[1], numpy_after[1])
            self.assertEqual(numpy_before[2:], numpy_after[2:])
            adata.X[-2:] = 100  # 改动留出患者不应改变缓存或训练模板。
            with patch('segmodel.training.template_initialization._fit_patterns', side_effect=AssertionError('cache miss')):
                second = initialize(model, adata, root, 'kmeans')
            self.assertTrue(second['cache_hit'])
            self.assertEqual(first['cache_key'], second['cache_key'])
            torch.testing.assert_close(model.tissue_state_decoder.template_logits, templates, rtol=0, atol=0)
            with np.load(Path(root) / 'fold/template_initialization.npz', allow_pickle=False) as saved:
                self.assertEqual(saved['gene_names'].tolist(), list(adata.var_names))
                self.assertEqual(saved['train_names'].tolist(), ['A1', 'A2'])

    def test_cache_changes_with_training_values_gene_order_and_fit_parameters(self):
        adata, _ = expression_case()
        with tempfile.TemporaryDirectory() as root:
            model = model_for(adata)
            keys = [initialize(model, adata, root, 'kmeans')['cache_key']]
            adata.X[0, 0] += .1
            keys.append(initialize(model, adata, root, 'kmeans')['cache_key'])
            reordered = adata[:, ['g1', 'g0', 'g2', 'g3']].copy()
            keys.append(initialize(model, reordered, root, 'kmeans')['cache_key'])
            keys.append(initialize(model, reordered, root, 'kmeans', seed=1)['cache_key'])
            self.assertEqual(len(set(keys)), 4)

    def test_random_initialization_does_not_read_data_or_write_files(self):
        with tempfile.TemporaryDirectory() as root:
            record = initialize_expression_templates(
                SimpleNamespace(tissue_state_count=0), None, [], {'method': 'random'},
                expression_transform='log1p', project_root=root, output_dir=Path(root) / 'fold',
            )
            self.assertIsNone(record)
            self.assertEqual(list(Path(root).iterdir()), [])

    def test_invalid_settings_data_and_architectures_are_rejected(self):
        adata, _ = expression_case()
        with tempfile.TemporaryDirectory() as root:
            for method, options in [('unknown', {}), ('nmf', {'smoothing': 1}), ('kmeans', {'max_iter': 0})]:
                with self.subTest(method=method, options=options), self.assertRaises(ValueError):
                    initialize(model_for(adata), adata, root, method, **options)
            with self.assertRaisesRegex(ValueError, 'requires tissue_state_count'):
                initialize(SimpleNamespace(tissue_state_count=0), adata, root, 'nmf')
            with self.assertRaisesRegex(ValueError, 'must not exceed'):
                initialize(model_for(adata, count=5), adata, root, 'nmf')
            invalid = adata.copy()
            invalid.X[0, 0] = -1
            with self.assertRaisesRegex(ValueError, 'nonnegative'):
                initialize(model_for(invalid), invalid, root, 'kmeans')
            invalid.X[:] = 0
            with self.assertRaisesRegex(ValueError, 'No nonzero'):
                initialize(model_for(invalid), invalid, root, 'kmeans')

    def test_pipeline_initialization_trainability_ema_and_saved_model(self):
        with tempfile.TemporaryDirectory() as root:
            cfg = write_fixture(root)
            for name, value in dict(
                encoder_chans=[8, 12, 16], decoder_chans=[12, 8], global_dim=8, spatial_dim=4,
                propagation_channels=4, spatial_pool=4, top_k=1, backbone_init='pytorch',
                tissue_state_count=2, tissue_state_dim=6,
            ).items():
                cfg.experiment.model.image[name] = value
            data = FoldDataModule(['A1', 'A2'], ['C1'], 0, cfg)
            data.setup('fit')
            patterns = np.array([[8, 2, 0, 0, 1, 0, 0, 0], [0, 0, 3, 7, 0, 1, 0, 0]], np.float32)
            data.train_gene_adata.X = np.log1p(np.repeat(patterns, 6, axis=0))
            data.train_gene_adata.obs['normalization_total_counts'] = np.tile([99, 100, 200, 200, 200, 200], 2)
            cfg.experiment.training.image.template_init.min_counts = 100
            training_expression = data.train_gene_adata.X.copy()
            cache_keys = {}
            for method, trainable in product(('random', 'nmf', 'kmeans', 'mix'), (True, False)):
                with self.subTest(method=method, trainable=trainable):
                    cfg.experiment.training.image.template_init.method = method
                    cfg.experiment.training.image.template_init.trainable = trainable
                    output_dir = Path(root) / f'{method}_{trainable}'
                    output_dir.mkdir()
                    with patch('segmodel.pipeline.pl.Trainer') as trainer:
                        module = _train_segmentation(cfg, data, {}, output_dir)
                    trainer.return_value.fit.assert_called_once_with(module, datamodule=data)
                    decoder = module.model.tissue_state_decoder
                    initial_logits = decoder.template_logits.detach().clone()
                    initial = initial_logits.softmax(dim=1)
                    if method != 'random':
                        with np.load(output_dir / 'template_initialization.npz', allow_pickle=False) as saved:
                            torch.testing.assert_close(initial, torch.from_numpy(saved['templates']))
                        record = json.loads((output_dir / 'template_initialization.json').read_text())
                        self.assertEqual(record['method'], method)
                        self.assertEqual(record['fit_spots'], 10)
                        self.assertEqual(record['excluded_low_count_spots'], 2)
                        if trainable:
                            cache_keys[method] = record['cache_key']
                        else:
                            self.assertEqual(record['cache_key'], cache_keys[method])
                            self.assertTrue(record['cache_hit'])
                    else:
                        self.assertFalse((output_dir / 'template_initialization.npz').exists())
                    self.assertEqual(decoder.template_logits.requires_grad, trainable)
                    self.assertEqual(data.train_gene_adata.n_obs, 12)
                    np.testing.assert_array_equal(data.train_gene_adata.X, training_expression)
                    torch.testing.assert_close(module.ema_model.tissue_state_decoder.template_logits,
                                               initial_logits, rtol=0, atol=0)
                    _save_final_model(cfg, module.model, {}, output_dir)
                    artifact = torch.load(output_dir / 'final_model.pt', weights_only=True)
                    restored = SegmentationModel(**artifact['model_config'])
                    restored.load_state_dict(artifact['state_dict'], strict=True)
                    torch.testing.assert_close(restored.tissue_state_decoder.template_logits.softmax(dim=1), initial)

                    module.propagation = None
                    module.image_cfg.loss_weights.segmentation = 1
                    module.image_cfg.gene_correlation_min_spots = 3
                    module.log = module.log_dict = lambda *args, **kwargs: None
                    optimizer = module.configure_optimizers()['optimizer']
                    parameter_ids = [id(p) for group in optimizer.param_groups for p in group['params']]
                    self.assertEqual(parameter_ids.count(id(decoder.template_logits)), int(trainable))
                    trained_parameters = (decoder.state_head.weight, decoder.amplitude_head.weight,
                                          decoder.encoder[0].weight, decoder.class_logits,
                                          module.model.stem.block1[0].weight)
                    before = [parameter.detach().clone() for parameter in trained_parameters]
                    batch = build_batch(batch=1)
                    batch['tissue'] = torch.ones_like(batch['valid'])
                    for step in (1, 2):
                        optimizer.zero_grad(set_to_none=True)
                        loss = module.training_step(batch)
                        self.assertTrue(bool(torch.isfinite(loss)))
                        loss.backward()
                        for parameter in trained_parameters:
                            self.assertGreater(float(parameter.grad.abs().sum()), 0)
                            self.assertTrue(bool(torch.isfinite(parameter.grad).all()))
                        if trainable:
                            self.assertGreater(float(decoder.template_logits.grad.abs().sum()), 0)
                        else:
                            self.assertIsNone(decoder.template_logits.grad)
                        optimizer.step()
                        module.trainer = SimpleNamespace(global_step=step)
                        module._update_ema_model()
                        if not trainable:
                            torch.testing.assert_close(decoder.template_logits, initial_logits, rtol=0, atol=0)
                            torch.testing.assert_close(module.ema_model.tissue_state_decoder.template_logits,
                                                       initial_logits, rtol=0, atol=0)
                    if trainable:
                        self.assertFalse(torch.equal(decoder.template_logits, initial_logits))
                    for parameter, previous in zip(trained_parameters, before):
                        self.assertFalse(torch.equal(parameter, previous))


if __name__ == '__main__':
    unittest.main()
