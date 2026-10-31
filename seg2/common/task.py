"""Locate expression inputs independently of prepared image/count directories."""

from pathlib import Path

from omegaconf import DictConfig


def gene_expression_path(
    cfg: DictConfig, fold_name: str, *, target: bool = False
) -> Path:
    directory = Path(str(cfg.task.gene_expression_dir)).expanduser().resolve()
    if str(cfg.task.split_mode) == "transfer":
        dataset = cfg.target_dataset if target else cfg.source_dataset
        return directory / f"{dataset.dataset_name}.h5ad"
    return directory / f"{fold_name}.h5ad"
