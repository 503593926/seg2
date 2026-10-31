from pathlib import Path

from omegaconf import DictConfig

from seg2.ingest.utils.gene_mapping import prepare_counts


def run(cfg: DictConfig) -> None:
    dataset = cfg.dataset
    raw = Path(str(dataset.raw_root)).expanduser().resolve() / "ST-cnts"
    sources = {
        path.name.removesuffix(".tsv.gz"): path
        for patient in sorted(dataset.patient_names)
        for path in sorted(raw.glob(f"{patient}*.tsv.gz"))
    }
    prepare_counts(
        sources=sources,
        data_dir=Path(str(dataset.data_dir)),
        reference=Path(str(cfg.prep.gene.mapping_reference)),
        task_root=Path(str(cfg.paths.task_root)),
        dataset_name=str(dataset.dataset_name),
    )
