"""实际 Lightning 训练中的当前 batch 指标、显示间隔及梯度累积。"""

import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from segmodel.training.callbacks import MetricsBelowRichProgressBar
from test_propagation_training import build_batch, build_module


class RecordProgress(MetricsBelowRichProgressBar):
    def __init__(self, interval):
        self.stream = io.StringIO()
        super().__init__(
            metrics_every_n_steps=interval,
            console_kwargs={'file': self.stream, 'force_terminal': True, 'width': 400},
        )
        self.seen = []

    def get_metrics(self, trainer, pl_module):
        metrics = super().get_metrics(trainer, pl_module)
        if trainer.training:
            self.seen.append(dict(metrics))
        return metrics


def fit(module, batches, progress, **options):
    trainer = pl.Trainer(
        accelerator='cpu', devices=1, precision='32-true', max_epochs=1,
        logger=False, enable_checkpointing=False, enable_model_summary=False,
        callbacks=[progress], **options,
    )
    trainer.fit(module, train_dataloaders=DataLoader(batches, batch_size=None))
    return trainer


class TrainingProgressTest(unittest.TestCase):
    def test_step_dice_iou_and_empty_mask_preserve_global_epoch_metrics(self):
        module = build_module()
        batches = [build_batch(batch=1) for _ in range(3)]
        for index, batch in enumerate(batches):
            batch['image'].fill_(255 if index == 0 else 0)
            batch['cancer_probability'].fill_(1)
            batch['valid'].fill_(index == 0)
        batches[1]['valid'][:, :, :8] = True

        def known_prediction(image, patch, cls, pfm_mask, *, dense_expression=True):
            logits = torch.where(image[:, :1, ::7, ::7] > 128, 4., -4.)
            return {'logits': logits + module.model.segmentation_head.bias.sum() * 0}

        progress = RecordProgress(interval=1)
        with patch.object(module.model, 'forward', side_effect=known_prediction):
            trainer = fit(module, batches, progress)
        self.assertEqual([metrics['step'] for metrics in progress.seen], [1, 2, 3])
        for metrics, expected in zip(progress.seen, (1., 0., 0.)):
            self.assertAlmostEqual(metrics['train/dice'], expected)
            self.assertAlmostEqual(metrics['train/iou'], expected)
            self.assertIn('train/loss', metrics)
            self.assertFalse(any(name.endswith('_epoch') for name in metrics))
        self.assertAlmostEqual(float(trainer.callback_metrics['train/dice_epoch']), .8)
        self.assertAlmostEqual(float(trainer.callback_metrics['train/iou_epoch']), 2 / 3)

    def test_interval_counts_optimizer_steps_and_displays_expression_and_lr(self):
        module = build_module(gene_count=8, tissue_state_count=4, tissue_state_dim=6)
        module.image_cfg.loss_weights.segmentation = 0
        module.image_cfg.loss_weights.propagation = 0
        module.image_cfg.loss_weights.gene_prediction = 1
        module.image_cfg.loss_weights.gene_correlation = .5
        module.image_cfg.gene_correlation_min_spots = 3
        progress = RecordProgress(interval=2)
        trainer = fit(
            module, [build_batch(batch=1) for _ in range(12)], progress,
            max_steps=5, accumulate_grad_batches=2,
        )
        self.assertEqual(trainer.global_step, 5)
        self.assertEqual([metrics['step'] for metrics in progress.seen], [1, 2, 4, 5])
        for metrics in progress.seen:
            for name in ('train/gene_prediction', 'train/gene_correlation', 'train/gene_spots'):
                self.assertIn(name, metrics)
                self.assertTrue(torch.isfinite(torch.tensor(metrics[name])))
            self.assertGreater(metrics['train/gene_spots'], 0)
            self.assertGreater(float(metrics['lr']), 0)
        self.assertAlmostEqual(float(progress.seen[-1]['lr']), 1e-6)
        self.assertIn('train/gene_prediction_epoch', trainer.callback_metrics)
        self.assertIn('train/gene_correlation_epoch', trainer.callback_metrics)
        self.assertIn('lr:', progress.stream.getvalue())
        self.assertIn('train/gene_prediction:', progress.stream.getvalue())


if __name__ == '__main__':
    unittest.main()
