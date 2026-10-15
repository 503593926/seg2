"""最高 PCC 基因的掩码、排名、共享/归一化色标及每轮验证保存接线。"""

import csv
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from segmodel.inference.metrics import expression_metrics, gene_pearson_correlations
from segmodel.inference.visualization import _expression_spot_panel, save_top_gene_predictions
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


class GeneVisualizationTest(unittest.TestCase):
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
        records, genes, slides = expression_records()
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
                module._validation_expression = records
                module._validation_metrics = {name: expression_metrics(record['prediction'], record['target'], record['observed'])
                                               for name, record in records.items()}
                module.on_validation_epoch_end()
                self.assertTrue((Path(root) / f'epoch_{epoch + 1:04d}/gene_expression/B/01_G0.png').exists())
                self.assertTrue((Path(root) / f'epoch_{epoch + 1:04d}/gene_expression/B/01_G0_normalized.png').exists())
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
