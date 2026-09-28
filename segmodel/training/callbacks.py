"""训练期回调：让随机 crop 随 epoch 变化、把指标放到进度条下一行。"""

import pytorch_lightning as pl
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.callbacks.progress.rich_progress import CustomProgress
from rich.table import Column, Table


class DatasetEpochCallback(pl.Callback):
    """让随机 crop 随训练 epoch 改变，同时保持同一 epoch 可复现。"""

    def on_train_epoch_start(self, trainer, pl_module):
        trainer.datamodule.train_dataset.epoch = trainer.current_epoch


class _MetricsBelowProgress(CustomProgress):
    def get_renderables(self):
        progress_columns = self.columns[:-1]
        metrics_column = self.columns[-1]
        for task in self.tasks:
            if not task.visible:
                continue
            table_columns = [
                Column(no_wrap=True)
                if isinstance(column, str)
                else column.get_table_column().copy()
                for column in progress_columns
            ]
            table = Table.grid(
                *table_columns,
                padding=(0, 1),
                expand=self.expand,
            )
            table.add_row(
                *(
                    column.format(task=task)
                    if isinstance(column, str)
                    else column(task)
                    for column in progress_columns
                )
            )
            yield table

            metrics = (
                metrics_column.format(task=task)
                if isinstance(metrics_column, str)
                else metrics_column(task)
            )
            if str(metrics):
                yield metrics


class MetricsBelowRichProgressBar(RichProgressBar):
    """将 Rich 进度条指标放到进度信息的下一行。"""

    def _init_progress(self, trainer):
        super()._init_progress(trainer)
        if self.progress is not None:
            self.progress.__class__ = _MetricsBelowProgress
