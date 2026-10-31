from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from seg2.main import run


def main():
    config_name = "config"  # 验证开关在 configs/evaluation/base.yaml 中设置

    @hydra.main(version_base="1.3", config_path="../configs", config_name=config_name)
    def launch(cfg: DictConfig):
        run(cfg, Path(HydraConfig.get().runtime.output_dir))

    launch()


if __name__ == "__main__":
    main()
