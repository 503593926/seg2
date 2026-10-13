"""患者隔离、缺失分割监督和独立表达评估的集成测试。"""

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import anndata as ad
import cv2
import numpy as np
import pandas as pd
import torch
from hydra import compose, initialize_config_dir

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from segmodel.dataset.datamodule import FoldDataModule
from segmodel.dataset.splits import patient_folds, patient_id
from segmodel.inference.metrics import expression_metrics, summarize_metrics
from segmodel.main import get_folds
from segmodel.pipeline import _test_final_model
from test_expression_prediction import case


def patient_config():
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1] / 'segmodel/configs'), version_base=None):
        return compose(config_name='config')


def write_fixture(directory):
    cfg = patient_config()
    cfg.project_root = str(directory)
    dataset = cfg.experiment.dataset
    dataset.all_names = ['A1', 'C1']
    dataset.expression_names = ['A1', 'A2', 'C1', 'C2']
    dataset.image_feature_maps = ['feature_0', 'feature_1']
    dataset.crop_size = 112
    dataset.samples_per_epoch = 32
    dataset.spot_diameter_um = 10.0
    cfg.experiment.augment.enabled = False
    root = Path(directory) / dataset.data_root
    coords = np.array([[14, 14], [42, 14], [70, 42], [42, 70], [70, 70], [98, 98]], np.float32)
    obs = []
    for name in dataset.expression_names:
        sample = root / 'samples' / name
        spatial = sample / 'spatial'
        spatial.mkdir(parents=True)
        cv2.imwrite(str(spatial / 'he_1mpp.png'), np.full((112, 112, 3), 150, np.uint8))
        cv2.imwrite(str(spatial / 'canvas_mask.png'), np.full((112, 112), 255, np.uint8))
        label = root / 'label_mpp' / name
        (label / 'spatial').mkdir(parents=True)
        cv2.imwrite(str(label / 'spatial/tissue_mask.png'), np.ones((16, 16), np.uint8))
        if name in dataset.all_names:
            (sample / 'masks').mkdir()
            cv2.imwrite(str(sample / 'masks/training_mask.png'), np.ones((112, 112), np.uint8))
            cv2.imwrite(str(sample / 'masks/valid_mask.png'), np.full((112, 112), 255, np.uint8))
            pseudo = label / dataset.pseudo_label_dir
            pseudo.mkdir()
            cv2.imwrite(str(pseudo / 'pseudo_mask.png'), np.ones((16, 16), np.uint8))
            cv2.imwrite(str(pseudo / 'cancer_probability.png'), np.full((16, 16), 65535, np.uint16))
            (label / 'snic').mkdir()
            np.save(label / 'snic/superpixels.npy', np.arange(1, 257).reshape(16, 16))
        for feature in dataset.image_feature_maps:
            folder = root / feature / name
            folder.mkdir(parents=True)
            (root / feature / 'metadata.json').write_text('{"patch_mpp": 14, "cls_mpp": 28}')
            np.save(folder / 'patch_map.npy', np.zeros((8, 8, 6), np.float16))
            np.save(folder / 'cls_map.npy', np.zeros((4, 4, 6), np.float16))
        obs.extend({'name': name, 'pixel_x': x, 'pixel_y': y, 'spot_label': index % 2}
                   for index, (x, y) in enumerate(coords))
    expression = np.log1p(np.arange(24 * 8, dtype=np.float32).reshape(24, 8) / 10)
    adata = ad.AnnData(expression, obs=pd.DataFrame(obs, index=[str(i) for i in range(24)]),
                       var=pd.DataFrame(index=[f'g{i}' for i in range(8)]))
    adata.uns.update(train_names=['A1', 'A2'], log1p={'base': None})
    gene_dir = root / dataset.gene_expression_dir
    gene_dir.mkdir(parents=True)
    adata.write_h5ad(gene_dir / 'test_C1.h5ad')
    return cfg


class PatientSplitTest(unittest.TestCase):
    def test_c_fold_excludes_all_six_c_sections(self):
        cfg = patient_config()
        folds = get_folds(cfg)
        self.assertEqual(len(folds), 8)
        fold = next(fold for fold in folds if fold['test'] == ['C1'])
        self.assertEqual(fold['test_expression'], [f'C{i}' for i in range(1, 7)])
        self.assertEqual(len(fold['train']), 30)
        self.assertTrue(all(patient_id(name) != 'C' for name in fold['train']))
        module = FoldDataModule(fold['train'], fold['test'], 0, cfg)
        self.assertEqual(len(module.train_labeled_names), 7)
        self.assertEqual(module.expression_test_names, fold['test_expression'])

    def test_rejects_inconsistent_or_leaking_fold(self):
        cfg = patient_config()
        with self.assertRaises(ValueError):
            patient_folds(['C1'], ['A1'])
        fold = next(fold for fold in get_folds(cfg) if fold['test'] == ['C1'])
        with self.assertRaisesRegex(ValueError, 'held-out'):
            FoldDataModule([*fold['train'], 'C2'], ['C1'], 0, cfg)

    def test_gene_selection_never_receives_heldout_patient(self):
        from derive.get_gene_expression import prepare_tasks
        samples = {}
        for name in ('A1', 'A2', 'C1', 'C2'):
            samples[name] = ad.AnnData(
                np.array([[10, 2, 1], [4, 8, 3]], dtype=np.float32),
                obs=pd.DataFrame({'spot_id': ['s0', 's1'], 'pixel_x': [5., 10.], 'pixel_y': [5., 10.]}, index=['s0', 's1']),
                var=pd.DataFrame(index=['g0', 'g1', 'g2']),
            )
        table = pd.DataFrame({'gene': ['g0', 'g1']})
        with tempfile.TemporaryDirectory() as directory:
            with patch('derive.get_gene_expression.select_genes', return_value=(['g0', 'g1', 'g2'], table)) as select:
                prepare_tasks(samples, list(samples), Path(directory), 2, 10000., 'hegs', 8, 1.,
                              patient_loocv=True, evaluation_names=['C1'])
            self.assertEqual(set(select.call_args.args[0]), {'A1', 'A2'})
            result = ad.read_h5ad(Path(directory) / 'task1_patient_loocv/test_C1.h5ad')
            self.assertEqual(set(result.uns['train_names']), {'A1', 'A2'})
            self.assertEqual(set(result.uns['test_expression_names']), {'C1', 'C2'})
            np.testing.assert_allclose(result.X.toarray()[0], np.log1p(np.array([10., 2.]) / 13 * 10000), rtol=1e-6)


class MixedSupervisionTest(unittest.TestCase):
    def test_loads_expression_only_slide_without_pseudo_labels_and_masks_poisoned_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = write_fixture(directory)
            module = FoldDataModule(['A1', 'A2'], ['C1'], 0, cfg)
            module.setup('fit')
            dataset = module.train_dataset
            slide = module.train_slides['A2']
            self.assertFalse(slide.has_segmentation)
            self.assertTrue((slide.pseudo_mask_uint8 == 255).all())
            self.assertTrue(slide.tissue_mask_bl.all())
            self.assertFalse(slide.superpixels.any())
            rows = dataset.spots_by_slide['A2']['indices']
            self.assertTrue(bool((dataset.spot_labels[rows] == 255).all()))
            sample = dataset._supervision('A2', 0, 0)
            self.assertFalse(bool(sample['valid'].any()))
            self.assertTrue(bool(sample['tissue'].all()))
            self.assertEqual(set(module.val_slides), {'C1', 'C2'})
            self.assertEqual(set(module.val_gene_adata.obs['name']), {'C1', 'C2'})
            rng = np.random.default_rng(0)
            labeled = sum(dataset._sample_origin(rng)[0] == 'A1' for _ in range(200))
            self.assertGreater(labeled, 70)
            self.assertLess(labeled, 130)

    def test_expression_only_batch_trains_gene_head_without_segmentation_gradient(self):
        module, batch = case(batch_size=2)
        batch['tissue'] = torch.ones_like(batch['valid'])
        batch['valid'].zero_()
        batch['superpixels'].zero_()
        batch['spot_labels'] = [torch.full_like(labels, 255) for labels in batch['spot_labels']]
        for key in module.image_cfg.loss_weights:
            module.image_cfg.loss_weights[key] = float(key == 'gene_prediction')
        loss = module.training_step(batch)
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertGreater(float(loss.detach()), 0)
        loss.backward()
        self.assertGreater(float(module.model.gene_head.weight.grad.abs().sum()), 0)
        gradient = module.model.segmentation_head.weight.grad
        self.assertTrue(gradient is None or float(gradient.abs().sum()) == 0)

    def test_mixed_batch_keeps_expression_targets_for_unlabeled_sample(self):
        module, batch = case(batch_size=2)
        batch['tissue'] = torch.ones_like(batch['valid'])
        batch['superpixels'][1].zero_()
        batch['valid'][1].zero_()
        batch['spot_labels'][1].fill_(255)
        output = module(batch['image'], batch['patch'], batch['cls'], batch['pfm_mask'])
        propagated = module.propagation(batch['superpixels'], output['features'], batch['image'],
                                        batch['spot_labels'], batch['spots'])
        from segmodel.training.losses import gene_prediction_losses
        losses, count = gene_prediction_losses(output, batch, module.spot_radius_px, module.label_scale)
        self.assertEqual(count, 12)
        self.assertEqual(propagated[0], [0])
        self.assertGreater(float(losses['gene_prediction'].detach()), 0)


class HeldoutExpressionEvaluationTest(unittest.TestCase):
    def test_final_test_evaluates_masks_only_on_c1_and_expression_on_c1_c2(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = write_fixture(directory)
            data = FoldDataModule(['A1', 'A2'], ['C1'], 0, cfg)
            module, _ = case(batch_size=1)
            module.model.gene_names = [f'g{i}' for i in range(8)]
            output = Path(directory) / 'output'
            output.mkdir()
            summary = _test_final_model(cfg, data, module.model, torch.device('cpu'),
                                        {'crop_size': 112, 'stride': 56, 'alignment': 14}, output)
            self.assertEqual(set(summary['slides']), {'C1', 'C2'})
            self.assertIn('dice', summary['slides']['C1'])
            self.assertNotIn('dice', summary['slides']['C2'])
            self.assertEqual(summary['expression']['micro']['gene_spots'], 12)
            self.assertEqual(summary['micro']['valid_pixels'], 112 * 112)
            self.assertTrue((output / 'test/C1/mask.png').exists())
            self.assertFalse((output / 'test/C2/mask.png').exists())
            with np.load(output / 'test/C2/gene_predictions.npz') as predictions:
                self.assertEqual(predictions['prediction'].shape, (6, 8))
                self.assertEqual(predictions['gene_names'].tolist(), module.model.gene_names)

    def test_validation_aggregates_all_heldout_spots_at_epoch_end(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = write_fixture(directory)
            data = FoldDataModule(['A1', 'A2'], ['C1'], 0, cfg)
            data.setup('fit')
            module, _ = case(batch_size=1)
            module.validation_roots = data.data_roots
            module.tiling = {'crop_size': 112, 'stride': 56, 'alignment': 14}
            module._trainer = SimpleNamespace(datamodule=data)
            logged = {}
            module.log_dict = lambda values, **kwargs: logged.update(values)
            module.log = lambda name, value, **kwargs: logged.update({name: value})
            module.on_validation_epoch_start()
            for index, slide in enumerate(data.val_slides.values()):
                module.validation_step(slide, index)
            module.on_validation_epoch_end()
            self.assertEqual(logged['val/gene_spots'], 12)
            self.assertIn('val/dice', logged)
            self.assertIn('val/C2/gene_prediction', logged)
            self.assertIn('val/gene_prediction_pcc_macro', logged)

    def test_pooled_mse_is_spot_weighted_and_segmentation_ignores_gene_only_slides(self):
        records = {
            'C1': {'prediction': torch.zeros(1, 2), 'target': torch.ones(1, 2)},
            'C2': {'prediction': torch.zeros(3, 2), 'target': torch.full((3, 2), 3.)},
        }
        slides = {name: expression_metrics(record['prediction'], record['target']) for name, record in records.items()}
        slides['C1'].update(tp=1, fp=0, fn=0, tn=1, loss=0., dice=1., iou=1., hd95=0.)
        summary = summarize_metrics(slides, records)
        self.assertAlmostEqual(summary['expression']['micro']['gene_prediction'], 7.)
        self.assertAlmostEqual(summary['expression']['macro']['gene_prediction'], 5.)
        self.assertEqual(summary['micro']['valid_pixels'], 2)

    def test_missing_gene_values_do_not_count_as_zero_expression(self):
        prediction = torch.tensor([[1., 1000.], [2., 1000.], [3., 5.]])
        target = torch.tensor([[1., 0.], [2., 0.], [3., 5.]])
        observed = torch.tensor([[True, False], [True, False], [True, True]])
        metrics = expression_metrics(prediction, target, observed)
        self.assertEqual(metrics['gene_values'], 4)
        self.assertEqual(metrics['gene_prediction'], 0.)
        self.assertEqual(metrics['gene_pcc_genes'], 1)
        self.assertAlmostEqual(metrics['gene_prediction_pcc'], 1., places=6)


if __name__ == '__main__':
    unittest.main()
