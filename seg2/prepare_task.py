"""Prepare task expression targets from dataset inputs; choose a task in main()."""

from pathlib import Path

from hydra import compose, initialize_config_dir

from seg2.derive.gene_expression import run as prepare_expression


def run(cfg):
    prepare_expression(cfg=cfg)


def main() -> None:
    task_name = "her2st_to_visium_wu_2021"

    config_dir = Path(__file__).resolve().parent / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        cfg = compose(
            config_name="prepare_task",
            overrides=[f"task@_global_={task_name}"],
        )
    run(cfg=cfg)


if __name__ == "__main__":
    main()
