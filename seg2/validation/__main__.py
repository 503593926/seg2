from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig

from seg2.main import get_folds
from seg2.train.pipeline import run_fold
from seg2.validation.callback import ValidationCallback


@hydra.main(version_base="1.3", config_path="../configs", config_name="validation")
def main(cfg):
    root = Path(HydraConfig.get().runtime.output_dir)
    dataset = cfg.dataset
    data_root = Path(str(dataset.data_dir)).expanduser().resolve()
    names = sorted(
        path.parent.name
        for path in data_root.glob("samples/*/gene_counts.tsv.gz")
        if path.parent.name[0] in dataset.patient_names
    )
    for seed in cfg.seeds:
        for fold in get_folds(cfg):
            output = root / f"seed_{seed}" / fold["name"]
            output.mkdir(parents=True, exist_ok=True)
            heldout = [name for name in names if name not in fold["train_names"]]
            gene_path = (
                data_root / str(dataset.gene_expression_dir) / f"{fold['name']}.h5ad"
            )
            callback = ValidationCallback(
                cfg, heldout, data_root, gene_path, output / "validation"
            )
            run_fold(fold, seed, cfg, output, extra_callbacks=[callback])


if __name__ == "__main__":
    main()
