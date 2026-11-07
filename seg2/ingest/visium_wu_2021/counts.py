from pathlib import Path

from seg2.ingest.utils.gene_mapping import prepare_counts
from seg2.ingest.visium_wu_2021.index import build_index


def run(cfg):
    dataset = cfg.dataset
    index = build_index(raw_root=dataset.raw_root)
    prepare_counts(
        sources={
            str(row["name"]): Path(row["count_dir"])
            for row in index.to_dict(orient="records")
        },
        data_dir=Path(str(dataset.data_dir)),
        reference=Path(str(cfg.prep.gene.mapping_reference)),
        task_root=Path(str(cfg.paths.task_root)),
        dataset_name=str(dataset.dataset_name),
    )
