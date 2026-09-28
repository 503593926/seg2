from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from segmodel.pipeline import run_fold


def get_folds(cfg: DictConfig) -> list[dict]:
    dataset_cfg = cfg.experiment.dataset
    split_mode = dataset_cfg.split_mode
    folds = []
    if split_mode == "loocv":
        all_names = dataset_cfg.all_names
        for test_name in all_names:
            train_names = [name for name in all_names if name != test_name]
            folds.append({"train": train_names, "test": [test_name]})

    elif split_mode == "transfer":
        folds.append(
            {"train": list(dataset_cfg.ref_names), "test": list(dataset_cfg.tgt_names)}
        )
    else:
        raise ValueError(f"Unsupported split mode: {split_mode}")
    return folds


def fold_output_dir(run_output_dir, fold, fold_index, seed):
    test_id = "-".join(fold["test"])
    return (
        Path(run_output_dir)
        / f"seed_{seed}"
        / f"fold_{fold_index:02d}_test_{test_id}"
    )


def experiment(
    fold: dict,
    fold_index: int,
    seed: int,
    cfg: DictConfig,
    run_output_dir: Path,
) -> None:
    output_dir = fold_output_dir(run_output_dir, fold, fold_index, seed)
    run_fold(fold["train"], fold["test"], seed, cfg, output_dir)


def run(cfg: DictConfig, run_output_dir: Path) -> None:
    folds = get_folds(cfg)
    for seed in cfg.seeds:
        for fold_index, fold in enumerate(folds):
            experiment(fold, fold_index, seed, cfg, run_output_dir)


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig):
    run(cfg, Path(HydraConfig.get().runtime.output_dir))


if __name__ == "__main__":
    main()
