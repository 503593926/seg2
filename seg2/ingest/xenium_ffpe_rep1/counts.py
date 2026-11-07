from pathlib import Path

from seg2.ingest.utils.gene_mapping import prepare_counts


def run(cfg):
    dataset = cfg.dataset
    bundle = Path(str(dataset.raw_root)) / "raw" / str(dataset.bundle_name)
    prepare_counts(
        sources={str(dataset.dataset_name): bundle / "outs/cell_feature_matrix.h5"},
        data_dir=Path(str(dataset.data_dir)),
        reference=Path(str(cfg.prep.gene.mapping_reference)),
        task_root=Path(str(cfg.paths.task_root)),
        dataset_name=str(dataset.dataset_name),
    )
