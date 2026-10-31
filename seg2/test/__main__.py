from pathlib import Path

from hydra import compose, initialize_config_dir

from seg2.test import run_test
from seg2.test.configuration import (
    evaluation_inputs,
    evaluation_parameters,
    model_artifact,
)


def main():
    project_root = Path(__file__).resolve().parents[2]
    config_name = "config"  # 实验在 configs/config.yaml 中选择
    model_dir = project_root / "outputs/her2st/2026-10-30/10-17-31/test_A/seed_0"

    config_dir = Path(__file__).resolve().parents[1] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        cfg = compose(config_name=config_name)
    dataset, gene_path, names = evaluation_inputs(cfg, model_dir.parent.name)
    test_dir = "external_test" if cfg.task.split_mode == "transfer" else "test"
    run_test(
        **evaluation_parameters(
            cfg,
            dataset=dataset,
            artifact_path=model_artifact(model_dir, cfg.evaluation.test.weights),
            gene_path=gene_path,
            output_dir=model_dir / test_dir,
            names=names,
        )
    )


if __name__ == "__main__":
    main()
