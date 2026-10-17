from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from segmodel.pipeline import run_fold


def get_folds(cfg: DictConfig) -> list[dict]:
    dataset_cfg = cfg.experiment.dataset
    project_root = Path(str(cfg.project_root))
    if dataset_cfg.split_mode == "patient_loocv":
        # 按患者确定全部训练和测试切片。
        data_root = project_root / str(dataset_cfg.data_root)
        samples_root = data_root / "samples"
        sample_names = sorted(
            (path.parent.name for path in samples_root.glob("*/gene_counts.tsv.gz")
             if path.parent.name[0] in dataset_cfg.patient_names),
            key=lambda name: (name[0], int(name[1:])),
        )
        folds = []
        for patient in dataset_cfg.patient_names:
            fold_name = f"test_{patient}"
            folds.append({
                "name": fold_name,
                "train_names": [name for name in sample_names if name[0] != patient],
                "test_names": [name for name in sample_names if name[0] == patient],
                "test_patient": patient,
                "data_roots": {name: data_root for name in sample_names},
                "gene_path": data_root / str(dataset_cfg.gene_expression_dir) / f"{fold_name}.h5ad",
            })
    elif dataset_cfg.split_mode == "transfer":
        # 源集全部训练，目标集统一测试；每个 seed 只生成一个实验。
        data_base = project_root / "data"
        mpp_dir = f"{float(cfg.resolution.target_mpp):g}mpp"
        ref_root = data_base / str(dataset_cfg.ref_dataset) / mpp_dir
        tgt_root = data_base / str(dataset_cfg.tgt_dataset) / mpp_dir
        train_names = sorted(path.parent.name for path in ref_root.glob("samples/*/gene_counts.tsv.gz"))
        test_names = sorted(path.parent.name for path in tgt_root.glob("samples/*/gene_counts.tsv.gz"))
        if set(train_names) & set(test_names):
            raise ValueError("Transfer source and target slide names must be distinct in the combined expression file")
        transfer_name = f"{dataset_cfg.ref_dataset}_to_{dataset_cfg.tgt_dataset}"
        folds = [{
            "name": transfer_name,
            "train_names": train_names,
            "test_names": test_names,
            "data_roots": {**dict.fromkeys(train_names, ref_root), **dict.fromkeys(test_names, tgt_root)},
            "gene_path": data_base / transfer_name / str(dataset_cfg.gene_expression_file),
        }]
    else:
        raise ValueError(f"Unsupported split mode: {dataset_cfg.split_mode}")
    return folds


def run(cfg: DictConfig, run_output_dir: Path) -> None:
    folds = get_folds(cfg)
    for seed in cfg.seeds:
        for fold in folds:
            output_dir = Path(run_output_dir) / f"seed_{seed}" / fold["name"]
            output_dir.mkdir(parents=True, exist_ok=True)
            run_fold(fold, seed, cfg, output_dir)


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig):
    run(cfg, Path(HydraConfig.get().runtime.output_dir))


if __name__ == "__main__":
    main()
