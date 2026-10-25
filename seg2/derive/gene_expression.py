from collections.abc import Mapping
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import csr_matrix

from seg2.derive.gene_selection import (
    rank_genes,
    select_genes,
    select_hvgs,
    select_stable_svgs,
    select_union,
)

__all__ = [
    "load_raw_gene_data",
    "prepare_tasks",
    "process_gene_expression",
    "rank_genes",
    "run",
    "select_genes",
    "select_hvgs",
    "select_stable_svgs",
    "select_union",
]


def load_raw_gene_data(
    samples_root: Path, names: list[str], target_mpp: float
) -> dict[str, ad.AnnData]:
    sample_adatas: dict[str, ad.AnnData] = {}
    for name in names:
        print(f"[{name}] loading raw gene counts", flush=True)
        sample_root = samples_root / name
        count_path = sample_root / "gene_counts.tsv.gz"
        spatial_path = sample_root / "spatial" / f"spatial_{target_mpp:g}mpp.tsv"
        counts = pd.read_csv(count_path, sep="\t", index_col=0)
        spatial = pd.read_csv(spatial_path, sep="\t")
        counts.index = counts.index.astype(str)
        counts.columns = counts.columns.astype(str)
        spatial["spot_id"] = spatial["spot_id"].astype(str)
        usable = (
            spatial["selected"].eq(1)
            & spatial["in_content"]
            & spatial["has_expression"]
        )
        spatial = spatial.loc[usable].copy()
        spot_ids: list[str] = spatial["spot_id"].tolist()
        obs = spatial.set_index("spot_id", drop=False)
        sample_adata = ad.AnnData(
            X=csr_matrix(counts.loc[spot_ids].to_numpy(dtype=np.int32)),
            obs=obs,
            var=pd.DataFrame(index=counts.columns),
        )
        sample_adatas[name] = sample_adata
    return sample_adatas


def process_gene_expression(
    sample_adatas: Mapping[str, ad.AnnData],
    names: list[str],
    common_genes: list[str],
    selected_gene_table: pd.DataFrame,
    target_sum: float,
    target_mpp: float,
) -> ad.AnnData:
    combined = ad.concat(
        {name: sample_adatas[name] for name in names},
        join="outer",
        label="name",
        index_unique="-",
        fill_value=0,
    )
    combined = combined[:, common_genes].copy()
    counts = csr_matrix(combined.X, dtype=np.float32)
    combined.X = counts
    obs = combined.obs
    obs["normalization_total_counts"] = np.asarray(counts.sum(axis=1)).reshape(-1)
    combined.obs_names = (
        obs["name"].astype(str) + ":" + obs["spot_id"].astype(str)
    ).tolist()
    combined.obs_names.name = "observation_id"
    sc.pp.normalize_total(combined, target_sum=target_sum)
    sc.pp.log1p(combined)
    selected_genes: list[str] = selected_gene_table["gene"].tolist()
    processed = combined[:, selected_genes].copy()
    processed.X = csr_matrix(processed.X, dtype=np.float32)
    processed.var = selected_gene_table.set_index("gene").loc[selected_genes].copy()
    processed.obsm[f"spatial_{target_mpp:g}mpp"] = processed.obs[
        ["pixel_x", "pixel_y"]
    ].to_numpy(dtype=np.float32)
    return processed


def prepare_tasks(
    sample_adatas: dict[str, ad.AnnData],
    sample_names: list[str],
    output_root: Path,
    n_top_genes: int,
    target_sum: float,
    selection_mode: str,
    svg_n_neighbors: int,
    target_mpp: float,
    union_rank_and_truncate: bool = False,
    *,
    patient_names: list[str],
) -> None:
    fold_root = output_root
    fold_root.mkdir(parents=True, exist_ok=True)
    for test_patient in patient_names:
        train_names = [name for name in sample_names if name[0] != test_patient]
        test_names = [name for name in sample_names if name[0] == test_patient]
        train_adatas = {name: sample_adatas[name] for name in train_names}
        common_genes, selected_gene_table = select_genes(
            train_adatas,
            n_top_genes,
            selection_mode,
            target_sum,
            svg_n_neighbors,
            union_rank_and_truncate,
        )
        selected_genes: list[str] = selected_gene_table["gene"].tolist()
        missing_by_slide = {
            name: [
                gene
                for gene in selected_genes
                if gene not in sample_adatas[name].var_names
            ]
            for name in test_names
        }
        processed = process_gene_expression(
            sample_adatas,
            sample_names,
            common_genes,
            selected_gene_table,
            target_sum,
            target_mpp,
        )
        valid_counts = processed.obs["normalization_total_counts"].to_numpy() > 0
        excluded_count = int((~valid_counts).sum())
        processed = processed[valid_counts].copy()
        processed.uns.update(
            test_patient=test_patient,
            train_names=train_names,
            test_names=test_names,
            gene_selection_mode=selection_mode,
            n_top_genes=n_top_genes,
            target_sum=target_sum,
            union_rank_and_truncate=union_rank_and_truncate,
            missing_selected_genes_by_slide={
                name: np.asarray(genes, dtype=str)
                for name, genes in missing_by_slide.items()
            },
        )
        processed.write_h5ad(
            fold_root / f"test_{test_patient}.h5ad", compression="gzip"
        )
        missing_count = len(
            {gene for genes in missing_by_slide.values() for gene in genes}
        )
        print(
            f"[patient={test_patient}] common={len(common_genes)}, selected={len(selected_genes)}, missing_selected={missing_count}, excluded_zero_count_spots={excluded_count}, shape={processed.shape}",
            flush=True,
        )


def run(cfg, force=False):
    gene = cfg.prep.gene
    root = Path(str(cfg.dataset.data_dir)).expanduser().resolve()
    patients = sorted(cfg.dataset.patient_names)
    names = [
        path.parent.name
        for patient in patients
        for path in sorted((root / "samples").glob(f"{patient}*/gene_counts.tsv.gz"))
    ]
    output_root = root / str(cfg.dataset.gene_expression_dir)
    outputs = [output_root / f"test_{patient}.h5ad" for patient in patients]
    if outputs and (not force) and all(path.is_file() for path in outputs):
        return
    target_mpp = float(cfg.prep.resolution.target_mpp)
    samples = load_raw_gene_data(root / "samples", names, target_mpp)
    prepare_tasks(
        samples,
        names,
        output_root,
        int(gene.n_top_genes),
        float(gene.target_sum),
        str(gene.selection_mode),
        int(gene.svg_n_neighbors),
        target_mpp,
        bool(gene.union_rank_and_truncate),
        patient_names=patients,
    )
