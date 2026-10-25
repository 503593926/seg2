import pytorch_lightning as pl
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.callbacks.progress.rich_progress import CustomProgress
from rich.table import Column, Table


class DatasetEpochCallback(pl.Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        data_module = trainer.datamodule
        data_module.train_dataset.epoch = trainer.current_epoch


class _MetricsBelowProgress(CustomProgress):
    def get_renderables(self):
        progress_columns = self.columns[:-1]
        metrics_column = self.columns[-1]
        for task in self.tasks:
            if not task.visible:
                continue
            table_columns = [
                Column(no_wrap=True) if isinstance(column, str) else column.get_table_column().copy()
                for column in progress_columns
            ]
            table = Table.grid(*table_columns, padding=(0, 1), expand=self.expand)
            table.add_row(
                *(
                    column.format(task=task) if isinstance(column, str) else column(task)
                    for column in progress_columns
                )
            )
            yield table
            metrics = (
                metrics_column.format(task=task) if isinstance(metrics_column, str) else metrics_column(task)
            )
            if str(metrics):
                yield metrics


class MetricsBelowRichProgressBar(RichProgressBar):
    def __init__(self, metrics_every_n_steps=100, **kwargs):
        super().__init__(**kwargs)
        self.metrics_every_n_steps = int(metrics_every_n_steps)
        self._last_metric_step = None

    def _init_progress(self, trainer):
        super()._init_progress(trainer)
        if self.progress is not None:
            self.progress.__class__ = _MetricsBelowProgress

    def get_metrics(self, trainer, pl_module):
        metrics = {
            name.removesuffix("_step"): value
            for name, value in super().get_metrics(trainer, pl_module).items()
            if not (name.startswith("train/") and name.endswith("_epoch"))
        }
        metrics["step"] = int(trainer.global_step)
        if trainer.optimizers:
            metrics["lr"] = f"{trainer.optimizers[0].param_groups[0]['lr']:.3e}"
        return metrics

    def _update_metrics(self, trainer, pl_module):
        if not self.is_enabled or self._metric_component is None:
            return
        if trainer.training:
            step = int(trainer.global_step)
            if step == 0 or step == self._last_metric_step:
                return
            if self._last_metric_step is not None and step % self.metrics_every_n_steps:
                return
            self._last_metric_step = step
        super()._update_metrics(trainer, pl_module)

    def on_train_epoch_end(self, trainer, pl_module):
        if int(trainer.global_step) != self._last_metric_step:
            self._last_metric_step = None
        super().on_train_epoch_end(trainer, pl_module)
