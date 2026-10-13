"""按总优化器 step 比例预热的边界、取整，以及梯度累积和训练步数截断。"""

import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_propagation_training import build_batch, build_module


def scheduled_module(total_steps, batches, accumulation=1):
    module = build_module()
    module.trainer = SimpleNamespace(
        estimated_stepping_batches=total_steps,
        num_training_batches=batches,
        accumulate_grad_batches=accumulation,
    )
    return module


def lr_curve(configuration, steps):
    optimizer = configuration['optimizer']
    scheduler = configuration['lr_scheduler']['scheduler']
    rates = [optimizer.param_groups[0]['lr']]
    for _ in range(steps):
        optimizer.step()
        scheduler.step()
        rates.append(optimizer.param_groups[0]['lr'])
    return rates


class StepScheduleTest(unittest.TestCase):
    def test_ratio_warmup_and_cosine_change_within_epoch_and_reach_boundaries(self):
        module = scheduled_module(1600, 16)
        module.image_cfg.lr = .004
        module.image_cfg.warmup_steps = .1
        configuration = module.configure_optimizers()
        self.assertEqual(configuration['lr_scheduler']['interval'], 'step')
        rates = lr_curve(configuration, 1600)
        self.assertAlmostEqual(rates[0], .0004)
        self.assertAlmostEqual(rates[1], .0004 + (.004 - .0004) / 160)
        self.assertAlmostEqual(rates[160], .004)
        self.assertAlmostEqual(rates[880], (.004 + 1e-6) / 2)
        self.assertAlmostEqual(rates[1600], 1e-6)
        self.assertTrue(all(first < second for first, second in zip(rates[:160], rates[1:161])))
        self.assertTrue(all(first > second for first, second in zip(rates[160:1600], rates[161:])))

    def test_ratio_rounds_up_using_total_optimizer_steps(self):
        module = scheduled_module(8, 7, accumulation=2)
        module.image_cfg.warmup_steps = .3
        rates = lr_curve(module.configure_optimizers(), 8)
        for step, rate in enumerate(rates):
            expected = (.0001 + step * .0003 if step <= 3 else
                        1e-6 + (.001 - 1e-6) * (1 + math.cos(math.pi * (step - 3) / 5)) / 2)
            self.assertAlmostEqual(rate, expected)

    def test_zero_and_full_warmup_have_valid_schedules(self):
        module = scheduled_module(5, 5)
        module.image_cfg.warmup_steps = 0
        rates = lr_curve(module.configure_optimizers(), 5)
        self.assertAlmostEqual(rates[0], .001)
        self.assertAlmostEqual(rates[-1], 1e-6)
        module = scheduled_module(3, 4)
        module.image_cfg.warmup_steps = 1.
        rates = lr_curve(module.configure_optimizers(), 3)
        for actual, expected in zip(rates, [.0001, .0004, .0007, .001]):
            self.assertAlmostEqual(actual, expected)

    def test_small_positive_ratio_still_warms_up_for_one_step(self):
        for total_steps in (1, 3):
            with self.subTest(total_steps=total_steps):
                module = scheduled_module(total_steps, total_steps)
                module.image_cfg.warmup_steps = .1
                rates = lr_curve(module.configure_optimizers(), total_steps)
                self.assertAlmostEqual(rates[0], .0001)
                self.assertAlmostEqual(rates[1], .001)
                if total_steps > 1:
                    self.assertAlmostEqual(rates[-1], 1e-6)

    def test_out_of_range_ratios_and_absolute_step_counts_are_rejected(self):
        module = scheduled_module(8, 7)
        for value in (-.1, 1.1, 3200, float('nan'), float('inf')):
            with self.subTest(warmup_steps=value):
                module.image_cfg.warmup_steps = value
                with self.assertRaisesRegex(ValueError, 'warmup_steps must be a ratio'):
                    module.configure_optimizers()

    def test_lightning_uses_optimizer_steps_with_accumulation_and_max_steps(self):
        module = build_module()
        seen = []

        class RecordLR(pl.Callback):
            def on_before_optimizer_step(self, trainer, pl_module, optimizer):
                seen.append(optimizer.param_groups[0]['lr'])

        trainer = pl.Trainer(
            accelerator='cpu', devices=1, precision='32-true', max_epochs=2,
            max_steps=3, limit_train_batches=4, accumulate_grad_batches=2,
            logger=False, enable_checkpointing=False, enable_progress_bar=False,
            enable_model_summary=False, callbacks=[RecordLR()],
        )
        loader = DataLoader([build_batch(batch=1) for _ in range(8)], batch_size=None)
        trainer.fit(module, train_dataloaders=loader)
        self.assertEqual(trainer.global_step, 3)
        self.assertEqual(len(seen), 3)
        for actual, expected in zip(seen, [.0001, .00055, .001]):
            self.assertAlmostEqual(actual, expected)
        self.assertAlmostEqual(trainer.optimizers[0].param_groups[0]['lr'], 1e-6)


if __name__ == '__main__':
    unittest.main()
