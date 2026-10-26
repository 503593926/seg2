from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from seg2.validation.callback import ValidationCallback
from seg2.train.pipeline import run_fold


def get_folds(cfg: DictConfig) -> list[dict]:
    folds = []
    if cfg.dataset.split_mode == "patient_loocv":
        data_dir = Path(str(cfg.dataset.data_dir)).expanduser().resolve()
        samples_root = data_dir / "samples"
        sample_names = sorted(
            (
                path.parent.name
                for path in samples_root.glob("*/gene_counts.tsv.gz")
                if path.parent.name[0] in cfg.dataset.patient_names
            ),
            key=lambda name: (name[0], int(name[1:])),
        )
        for patient in cfg.dataset.patient_names:
            fold_name = f"test_{patient}"
            train_names = [name for name in sample_names if name[0] != patient]
            folds.append(
                {
                    "name": fold_name,
                    "train_names": train_names,
                    "data_dir": data_dir
                }
            )

    return folds


# def run(cfg: DictConfig, run_output_dir: Path) -> None:
#     folds = get_folds(cfg)
#     for seed in cfg.seeds:
#         for fold in folds:
#             log_dir = run_output_dir / fold["name"] / f"seed_{seed}"
#             log_dir.mkdir(parents=True, exist_ok=True)
#             run_fold(fold, seed, cfg, log_dir)


def run(cfg: DictConfig, run_output_dir: Path) -> None:
    folds = get_folds(cfg)
    for seed in cfg.seeds:
        for fold in folds:
            log_dir = run_output_dir / fold["name"] / f"seed_{seed}"
            log_dir.mkdir(parents=True, exist_ok=True)
            data_dir = fold["data_dir"]
            heldout = sorted(
                path.parent.name
                for path in data_dir.glob("samples/*/gene_counts.tsv.gz")
                if path.parent.name[0] in cfg.dataset.patient_names
                and path.parent.name not in fold["train_names"]
            )
            gene_path = (
                    data_dir / str(cfg.dataset.gene_expression_dir) / f"{fold['name']}.h5ad"
            )
            callback = ValidationCallback(
                cfg, heldout, data_dir, gene_path, log_dir / "validation"
            )
            run_fold(fold, seed, cfg, log_dir, extra_callbacks=[callback])

@hydra.main(version_base="1.3", config_path="configs", config_name="validation")
def main(cfg: DictConfig):
    run(cfg, Path(HydraConfig.get().runtime.output_dir))


if __name__ == "__main__":
    main()
