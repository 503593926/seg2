from importlib import import_module
from pathlib import Path

from hydra import compose, initialize_config_dir


def main() -> None:
    dataset = "her2st"
    ingest_stages = ("mpp", "masks", "canvas")
    derive_stages = (
        "seed_selection",
        "label_grid",
        "superpixels",
        "pseudo_label",
        "pfm_features",
        "gene_expression",
    )
    force = False

    config_dir = Path(__file__).resolve().parent / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        cfg = compose(config_name="prepare", overrides=[f"dataset={dataset}"])

    for stage in ingest_stages:
        print(f"== ingest.{dataset}.{stage} ==", flush=True)
        import_module(f"seg2.ingest.{dataset}.{stage}").run(cfg, force)

    for stage in derive_stages:
        print(f"== derive.{stage} ==", flush=True)
        import_module(f"seg2.derive.{stage}").run(cfg, force)


if __name__ == "__main__":
    main()
