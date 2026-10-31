from pathlib import Path

from omegaconf import DictConfig

from seg2.ingest.stnet.index import build_index
from seg2.ingest.utils.gene_mapping import prepare_counts


def run(cfg: DictConfig) -> None:
    dataset = cfg.dataset
    index = build_index(raw_root=dataset.raw_root)
    sources = {
        str(record["section_name"]): Path(str(record["count_path"]))
        for record in index.to_dict("records")
    }
    prepare_counts(
        sources=sources,
        data_dir=Path(str(dataset.data_dir)),
        reference=Path(str(cfg.prep.gene.mapping_reference)),
        task_root=Path(str(cfg.paths.task_root)),
        dataset_name=str(dataset.dataset_name),
    )
