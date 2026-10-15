"""最高 PCC 基因的掩码、排名、共享/归一化色标及每轮验证保存接线。"""

import csv
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from segmodel.inference.metrics import expression_metrics, gene_pearson_correlations
from segmodel.inference.predictor import predict_slide
from segmodel.inference.validation import validate_slide
from segmodel.inference.visualization import _expression_spot_panel, save_top_gene_predictions
from segmodel.modules.tissue_states import TissueStateDecoder
from test_propagation_training import build_module


def expression_records():
    target = torch.tensor([[1., 1., 2., 9., 1., 1.], [2., 3., 2., 6., 2., 2.],
                           [4., 5., 2., 4., 4., 4.], [7., 8., 2., 0., 7., 7.]])
    prediction = target.clone()
    prediction[:, 0] += .5
    prediction[:, 1] = torch.tensor([1000., 1000., 6., 10.])
    prediction[:, 2] = torch.arange(4.)
    prediction[:, 4] = 8 - target[:, 4]
    prediction[:, 5] = 1
    observed = torch.ones_like(target, dtype=torch.bool)
    observed[:2, 1] = False
    observed[:, 3] = False
    coords = np.array([[30., 30.], [65., 50.], [35., 35.], [95., 55.]], np.float32)
    records = {name: {'prediction': prediction[indices], 'target': target[indices],
                      'observed': observed[indices], 'spots': coords[indices]}
               for name, indices in [('A', slice(0, 2)), ('B', slice(2, 4))]}
    image = np.full((80, 140, 3), [210, 150, 190], dtype=np.uint8)
    slides = {name: SimpleNamespace(image=image.copy()) for name in records}
    return records, ['G0', 'Gene/1', 'constant_target', 'missing', 'negative', 'constant_prediction'], slides


def dense_expression_records():
    records, genes, slides = expression_records()
    for offset, (name, record) in enumerate(records.items()):
        slides[name].image = np.full((84, 140, 3), [210, 150, 190], dtype=np.uint8)
        field = np.ones((12, 20, len(genes)), np.float32)
        field[..., 0] = np.expm1(np.linspace(1 + offset * 2, 8 + offset * 2, 240).reshape(12, 20))
        tissue = np.ones((12, 20), dtype=bool)
        tissue[0, 0] = False
        field[~tissue] = 1e8  # 背景极值不能进入色标范围。
        record.update(dense_expression={'prediction': field}, dense_tissue=tissue, dense_mpp=7.)
    return records, genes, slides


class GeneVisualizationTest(unittest.TestCase):
    def test_dense_native_export_uses_unpooled_fields_and_tissue_ranges(self):
        records, genes, slides = dense_expression_records()
        before = deepcopy(records)
        with tempfile.TemporaryDirectory() as root:
            result = save_top_gene_predictions(Path(root) / 'direct', records, genes, slides, top_n=2)
            gene = result['selected_genes'][0]
            self.assertAlmostEqual(gene['dense_vmax'], 10., places=5)
            self.assertAlmostEqual(gene['dense_normalization']['max'], 10., places=5)
            self.assertAlmostEqual(gene['dense_normalization']['min'], 1 + 7 / 239, places=5)
            self.assertEqual(gene['normalization']['predicted'], {'min': 1.5, 'max': 7.5})
            for name in slides:
                info = gene['slides'][name]
                native = cv2.imread(str(Path(root) / 'direct' / info['dense_file']))
                normalized = cv2.imread(str(Path(root) / 'direct' / info['dense_normalized_file']))
                self.assertEqual(native.shape, (12, 20, 3))
                self.assertEqual(normalized.shape, native.shape)
                self.assertEqual(info['dense_mpp'], 7.)
                self.assertEqual(info['dense_shape'], [12, 20])
                np.testing.assert_array_equal(native[0, 0], [255, 255, 255])
                np.testing.assert_array_equal(normalized[0, 0], [255, 255, 255])
                comparison = cv2.imread(str(Path(root) / 'direct' / info['dense_comparison_normalized_file']))
                self.assertEqual(comparison.shape, (576, 1536, 3))
                overview = cv2.imread(str(Path(root) / 'direct' / name / 'overview_dense_normalized.jpg'))
                self.assertEqual(overview.shape, (1152, 1536, 3))
                expected = np.log1p(before[name]['dense_expression']['prediction'][..., 0])
                encoded = np.rint(np.clip(expected / gene['dense_vmax'], 0, 1) * 255).astype(np.uint8)
                expected_colors = cv2.applyColorMap(encoded, cv2.COLORMAP_VIRIDIS)
                np.testing.assert_array_equal(native[record_mask := records[name]['dense_tissue']], expected_colors[record_mask])
            first = cv2.imread(str(Path(root) / 'direct/A/01_G0_dense_normalized.png'))
            last = cv2.imread(str(Path(root) / 'direct/B/01_G0_dense_normalized.png'))
            palette = cv2.applyColorMap(np.array([[0, 255]], np.uint8), cv2.COLORMAP_VIRIDIS)[0]
            np.testing.assert_array_equal(first[0, 1], palette[0])
            np.testing.assert_array_equal(last[-1, -1], palette[1])
            for name, record in records.items():
                np.testing.assert_array_equal(record['dense_expression']['prediction'], before[name]['dense_expression']['prediction'])
                torch.testing.assert_close(record['prediction'], before[name]['prediction'], rtol=0, atol=0)
                # 相同场改用系数/模板表示，输出热力图应完全相同。
                record['dense_expression'] = {'coefficients': record['dense_expression']['prediction'],
                                              'templates': np.eye(len(genes), dtype=np.float32)}
            save_top_gene_predictions(Path(root) / 'factorized', records, genes, slides, top_n=2)
            for name in slides:
                for suffix in ('_dense', '_dense_normalized'):
                    direct = cv2.imread(str(Path(root) / f'direct/{name}/01_G0{suffix}.png'))
                    factorized = cv2.imread(str(Path(root) / f'factorized/{name}/01_G0{suffix}.png'))
                    np.testing.assert_array_equal(direct, factorized)

    def test_predictor_caches_factors_on_native_grid_without_changing_predictions(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.tissue_state_decoder = TissueStateDecoder(3, 6, state_count=4, hidden_dim=3)
                self.calls = 0

            def forward(self, image, patch, cls):
                self.calls += 1
                return self.tissue_state_decoder(torch.nn.functional.avg_pool2d(image.float() / 255, 7))

        image = torch.arange(56 * 84 * 3).reshape(56, 84, 3).remainder(256).to(torch.uint8)
        slide = SimpleNamespace(height=56, width=84)
        slide.inputs = lambda top, left, size: {
            'image': image[top:top + size, left:left + size].permute(2, 0, 1),
            'patch': (torch.zeros(1, 1, 1),), 'cls': (torch.zeros(1, 1, 1),),
        }
        model = Model()
        options = dict(crop_size=28, stride=14, alignment=7, device='cpu', batch_size=3,
                       return_features=True, feature_key='gene_expression')
        expected_probability, expected_field = predict_slide(model, slide, **options)
        model.calls = 0
        factors = {}
        probability, field = predict_slide(model, slide, **options, expression_factors=factors)
        self.assertEqual(model.calls, 5)  # 15 个窗口，batch=3；缓存不增加模型推理。
        self.assertTrue(model.training)
        self.assertEqual(factors['coefficients'].shape, (8, 12, 4))
        self.assertEqual(factors['templates'].shape, (4, 6))
        np.testing.assert_array_equal(probability, expected_probability)
        np.testing.assert_array_equal(field, expected_field)
        reconstructed = np.einsum('hwk,kg->hwg', factors['coefficients'], factors['templates'])
        np.testing.assert_allclose(reconstructed, field, rtol=2e-6, atol=1e-6)

    def test_validation_collects_dense_only_when_requested_and_keeps_spot_metrics(self):
        field = (np.arange(12 * 20 * 6).reshape(12, 20, 6).astype(np.float32) + 1) / 100
        slide = SimpleNamespace(name='A', height=84, width=140, label_scale=7)
        adata = SimpleNamespace(
            obs=pd.DataFrame({'name': ['A'] * 3, 'pixel_x': [14., 42., 84.], 'pixel_y': [14., 28., 56.]}),
            X=np.log1p(np.arange(18).reshape(3, 6).astype(np.float32)),
        )
        for factorized in (False, True):
            def predict(*args, **kwargs):
                if factorized and kwargs.get('expression_factors') is not None:
                    kwargs['expression_factors'].update(coefficients=field, templates=np.eye(6, dtype=np.float32))
                return np.zeros((84, 140), np.float32), field

            with self.subTest(factorized=factorized), patch('segmodel.inference.validation.predict_slide', side_effect=predict) as inference, \
                 patch('segmodel.inference.validation.read_image', return_value=np.ones((12, 20), np.uint8)):
                results = []
                for enabled in (False, True):
                    records = {}
                    _, metrics = validate_slide(SimpleNamespace(joint_prediction=False), slide, '/unused', 'cpu',
                                                28, 14, 7, gene_adata=adata, spot_radius_px=.1,
                                                evaluate_segmentation=False, expression_results=records,
                                                save_dense_expression=enabled)
                    self.assertEqual('dense_expression' in records['A'], enabled)
                    results.append((metrics, records['A']))
                self.assertEqual(inference.call_count, 2)
                self.assertEqual(results[0][0]['gene_prediction'], results[1][0]['gene_prediction'])
                torch.testing.assert_close(results[0][1]['prediction'], results[1][1]['prediction'], rtol=0, atol=0)
                self.assertEqual(results[1][1]['dense_mpp'], 7.)
                self.assertEqual('coefficients' in results[1][1]['dense_expression'], factorized)

    def test_gene_pcc_excludes_missing_and_constants_and_matches_validation(self):
        records, _, _ = expression_records()
        prediction, target, observed = [torch.cat([record[key] for record in records.values()])
                                         for key in ('prediction', 'target', 'observed')]
        pcc = gene_pearson_correlations(prediction, target, observed)
        torch.testing.assert_close(pcc[[0, 1, 4]], torch.tensor([1., 1., -1.]))
        self.assertTrue(bool(torch.isnan(pcc[[2, 3, 5]]).all()))
        metrics = expression_metrics(prediction, target, observed)
        self.assertEqual(metrics['gene_pcc_genes'], 3)
        self.assertAlmostEqual(metrics['gene_prediction_pcc'], 1 / 3, places=6)

    def test_top_genes_export_shared_scale_and_missing_spots_are_excluded(self):
        records, genes, slides = expression_records()
        rng_before = torch.random.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as root:
            result = save_top_gene_predictions(root, records, genes, slides, top_n=2, spot_radius_px=4)
            self.assertEqual(result['eligible_genes'], 3)
            self.assertEqual([gene['gene'] for gene in result['selected_genes']], ['G0', 'Gene/1'])
            second = result['selected_genes'][1]
            self.assertEqual(second['observed_spots'], 2)
            self.assertEqual(second['vmax'], 10)  # 未观测位置的 1000 不进入色标。
            self.assertEqual(second['normalization'], {'observed': {'min': 5., 'max': 8.},
                                                       'predicted': {'min': 6., 'max': 10.}})
            self.assertEqual(second['slides']['A']['observed_spots'], 0)
            self.assertIsNone(second['slides']['A']['pcc'])
            with (Path(root) / 'top_genes.csv').open() as handle:
                self.assertEqual([row['gene'] for row in csv.DictReader(handle)], ['G0', 'Gene/1'])
            saved = json.loads((Path(root) / 'metadata.json').read_text())
            self.assertEqual(saved['ranking_scope'], 'pooled_validation_spots')
            for name in slides:
                image = cv2.imread(str(Path(root) / name / '02_Gene_1.png'))
                self.assertEqual(image.shape, (576, 1024, 3))
                np.testing.assert_array_equal(image[534:546, 20:492], image[534:546, 532:1004])
                overview = cv2.imread(str(Path(root) / name / 'overview.jpg'))
                self.assertEqual(overview.shape, (1152, 1024, 3))
                normalized = cv2.imread(str(Path(root) / second['slides'][name]['normalized_file']))
                self.assertEqual(normalized.shape, (576, 1024, 3))
                normalized_overview = cv2.imread(str(Path(root) / name / 'overview_normalized.jpg'))
                self.assertEqual(normalized_overview.shape, overview.shape)
            self.assertTrue(torch.equal(rng_before, torch.random.get_rng_state()))

    def test_independent_normalization_preserves_cross_slide_ranges_and_original_values(self):
        records, genes, slides = expression_records()
        before = {name: {key: value.clone() if isinstance(value, torch.Tensor) else value.copy()
                         for key, value in record.items()} for name, record in records.items()}
        with tempfile.TemporaryDirectory() as root:
            with patch('segmodel.inference.visualization._expression_spot_panel', wraps=_expression_spot_panel) as panel:
                result = save_top_gene_predictions(root, records, genes, slides, top_n=1, spot_radius_px=4)
            gene = result['selected_genes'][0]
            self.assertEqual(gene['gene'], 'G0')
            self.assertAlmostEqual(gene['pcc'], 1., places=6)
            self.assertEqual(gene['normalization'], {'observed': {'min': 1., 'max': 7.},
                                                      'predicted': {'min': 1.5, 'max': 7.5}})
            for title in ('Observed normalized', 'Predicted normalized'):
                calls = [call for call in panel.call_args_list if call.args[-1] == title]
                self.assertEqual(len(calls), 2)
                np.testing.assert_allclose(calls[0].args[3], [0., 1 / 6])
                np.testing.assert_allclose(calls[1].args[3], [.5, 1.])
                self.assertTrue(all(call.args[4] == 1. for call in calls))
            for name in slides:
                raw = cv2.imread(str(Path(root) / name / '01_G0.png'))
                normalized = cv2.imread(str(Path(root) / name / '01_G0_normalized.png'))
                # 仿射偏差在共享色标图中可见，在独立归一化后两边空间颜色相同。
                self.assertFalse(np.array_equal(raw[108:530, :512], raw[108:530, 512:]))
                np.testing.assert_array_equal(normalized[108:530, :512], normalized[108:530, 512:])
        for name, record in records.items():
            for key, value in record.items():
                if isinstance(value, torch.Tensor):
                    torch.testing.assert_close(value, before[name][key], rtol=0, atol=0)
                else:
                    np.testing.assert_array_equal(value, before[name][key])

    def test_available_genes_limit_top_n_and_no_defined_pcc_is_safe(self):
        records, genes, slides = expression_records()
        with tempfile.TemporaryDirectory() as root:
            result = save_top_gene_predictions(Path(root) / 'all', records, genes, slides, top_n=10)
            self.assertEqual(len(result['selected_genes']), 3)
            for record in records.values():
                record['prediction'] = torch.zeros_like(record['target'])
            empty = save_top_gene_predictions(Path(root) / 'undefined', records, genes, slides)
            self.assertEqual(empty['selected_genes'], [])
            self.assertTrue((Path(root) / 'undefined/metadata.json').exists())
            self.assertFalse((Path(root) / 'undefined/A').exists())
            self.assertEqual(save_top_gene_predictions(Path(root) / 'disabled', records, genes, slides, top_n=0), {})
            self.assertFalse((Path(root) / 'disabled').exists())

    def test_epoch_end_saves_each_epoch_supports_single_slide_and_can_disable(self):
        records, genes, slides = dense_expression_records()
        module = build_module(gene_count=6)
        module.image_cfg.gene_visualization_top_n = 2
        module.log = module.log_dict = lambda *args, **kwargs: None
        data = SimpleNamespace(expression_test_names=list(slides), val_gene_adata=SimpleNamespace(var_names=genes),
                               val_slides=slides)
        module.trainer = SimpleNamespace(datamodule=data, current_epoch=0, is_global_zero=True)
        with tempfile.TemporaryDirectory() as root:
            module.visualization_dir = Path(root)
            for epoch in (0, 1):
                module.trainer.current_epoch = epoch
                module.on_validation_epoch_start()
                module._validation_expression = deepcopy(records)
                module._validation_metrics = {name: expression_metrics(record['prediction'], record['target'], record['observed'])
                                               for name, record in records.items()}
                module.on_validation_epoch_end()
                self.assertTrue((Path(root) / f'epoch_{epoch + 1:04d}/gene_expression/B/01_G0.png').exists())
                self.assertTrue((Path(root) / f'epoch_{epoch + 1:04d}/gene_expression/B/01_G0_normalized.png').exists())
                self.assertTrue((Path(root) / f'epoch_{epoch + 1:04d}/gene_expression/B/01_G0_dense.png').exists())
                self.assertTrue((Path(root) / f'epoch_{epoch + 1:04d}/gene_expression/B/01_G0_dense_normalized.png').exists())
                self.assertNotIn('dense_expression', module._validation_expression['B'])
                self.assertNotIn('dense_tissue', module._validation_expression['B'])
            module.image_cfg.gene_visualization_top_n = 0
            module.trainer.current_epoch = 2
            module.on_validation_epoch_end()
            self.assertFalse((Path(root) / 'epoch_0003').exists())
            module.image_cfg.gene_visualization_top_n = 1
            module.trainer.current_epoch = 3
            data.expression_test_names = ['A']
            module._validation_expression = {'A': records['A']}
            module._validation_metrics = {'A': module._validation_metrics['A']}
            module.on_validation_epoch_end()
            self.assertTrue((Path(root) / 'epoch_0004/gene_expression/A/01_G0.png').exists())


if __name__ == '__main__':
    unittest.main()
