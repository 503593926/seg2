from pathlib import Path

import anndata as ad
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from seg2.common.task import gene_expression_path
from seg2.train.pipeline import run_fold


def get_folds(cfg: DictConfig) -> list[dict]:
    folds = []
    if cfg.task.split_mode == "transfer":
        data_dir = Path(str(cfg.source_dataset.data_dir)).expanduser().resolve()
        path = gene_expression_path(cfg, str(cfg.experiment_name))
        data = ad.read_h5ad(path, backed="r")
        try:
            if data.uns.get("data_role") != "train" or not data.uns.get(
                "external_task"
            ):
                raise ValueError(
                    "Transfer mode requires the external task's training H5AD"
                )
            names = list(map(str, data.uns["train_names"]))
            if set(data.obs["name"].astype(str)) != set(names):
                raise ValueError(
                    "Target test samples must not enter source training data"
                )
        finally:
            data.file.close()
        return [
            {
                "name": str(cfg.experiment_name),
                "train_names": names,
                "data_dir": data_dir,
            }
        ]
    if cfg.task.split_mode == "patient_loocv":
        data_dir = Path(str(cfg.source_dataset.data_dir)).expanduser().resolve()
        samples_root = data_dir / "samples"
        sample_names = sorted(
            (
                path.parent.name
                for path in samples_root.glob("*/gene_counts.tsv.gz")
                if path.parent.name[0] in cfg.source_dataset.patient_names
            ),
            key=lambda name: (name[0], int(name[1:])),
        )
        for patient in cfg.source_dataset.patient_names:
            fold_name = f"test_{patient}"
            train_names = [name for name in sample_names if name[0] != patient]
            folds.append(
                {"name": fold_name, "train_names": train_names, "data_dir": data_dir}
            )

        return folds
    raise ValueError(f"Unknown split mode: {cfg.task.split_mode}")



# def run(cfg: DictConfig, run_output_dir: Path) -> None:
#     folds = get_folds(cfg)
#     for seed in cfg.seeds:
#         for fold in folds:
#             log_dir = run_output_dir / fold["name"] / f"seed_{seed}"
#             log_dir.mkdir(parents=True, exist_ok=True)
#
#             run_fold(fold, seed, cfg, log_dir)
#
#             dataset, gene_path, names = evaluation_inputs(cfg, fold["name"])
#             test_dir = (
#                 "external_test"
#                 if cfg.task.split_mode == "transfer"
#                 else "test"
#             )
#             run_test(
#                 **evaluation_parameters(
#                     cfg,
#                     dataset=dataset,
#                     artifact_path=model_artifact(
#                         log_dir, cfg.evaluation.test.weights
#                     ),
#                     gene_path=gene_path,
#                     output_dir=log_dir / test_dir,
#                     names=names,
#                 )
#             )


def run(cfg: DictConfig, run_output_dir: Path) -> None:
    validate = bool(cfg.evaluation.validation.enabled)
    folds = get_folds(cfg)
    for seed in cfg.seeds:
        for fold in folds:
            log_dir = run_output_dir / fold["name"] / f"seed_{seed}"
            log_dir.mkdir(parents=True, exist_ok=True)
            callbacks = []
            if validate:
                from seg2.test.configuration import evaluation_inputs
                from seg2.validation.callback import ValidationCallback

                dataset, gene_path, names = evaluation_inputs(cfg, fold["name"])
                callbacks.append(
                    ValidationCallback(
                        cfg,
                        names,
                        dataset.data_dir,
                        gene_path,
                        log_dir / "validation",
                        dataset=dataset,
                    )
                )
            run_fold(fold, seed, cfg, log_dir, extra_callbacks=callbacks)
            if cfg.evaluation.test.after_train:
                from seg2.test import run_test
                from seg2.test.configuration import (
                    evaluation_inputs,
                    evaluation_parameters,
                    model_artifact,
                )

                dataset, gene_path, names = evaluation_inputs(cfg, fold["name"])
                test_dir = (
                    "external_test" if cfg.task.split_mode == "transfer" else "test"
                )
                run_test(
                    **evaluation_parameters(
                        cfg,
                        dataset=dataset,
                        artifact_path=model_artifact(
                            log_dir, cfg.evaluation.test.weights
                        ),
                        gene_path=gene_path,
                        output_dir=log_dir / test_dir,
                        names=names,
                    )
                )

def main():
    config_name = "config"

    @hydra.main(version_base="1.3", config_path="configs", config_name=config_name)
    def launch(cfg: DictConfig):
        run(cfg, Path(HydraConfig.get().runtime.output_dir))

    launch()


if __name__ == "__main__":
    main()
