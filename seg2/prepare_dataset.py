from importlib import import_module
from pathlib import Path

from hydra import compose, initialize_config_dir


def run(cfg, skip_features=False):
    name = str(cfg.dataset.dataset_name)
    cfg.prep.resolution.target_mpp = float(cfg.dataset.target_mpp)
    ingest_stages = cfg.prep.ingest_stages
    derive_stages = cfg.prep.derive_stages

    for stage in ingest_stages:
        print(f"== ingest.{name}.{stage} ==", flush=True)
        import_module(f"seg2.ingest.{name}.{stage}").run(cfg=cfg)

    for stage in derive_stages:
        if skip_features and stage == "pfm_features":
            continue
        print(f"== derive.{stage} ==", flush=True)
        import_module(f"seg2.derive.{stage}").run(cfg=cfg)


def main() -> None:
    dataset_name = "visium_wu_2021"
    skip_features = False

    config_dir = Path(__file__).resolve().parent / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        cfg = compose(
            config_name="prepare_dataset",
            overrides=[f"dataset={dataset_name}", f"prepare@prep={dataset_name}"],
        )
    run(cfg=cfg, skip_features=skip_features)


if __name__ == "__main__":
    main()
