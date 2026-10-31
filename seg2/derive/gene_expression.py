import hashlib
import json
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
from seg2.ingest.utils.gene_mapping import counts_gene_info

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
    samples_root: Path,
    names: list[str],
    target_mpp: float,
    *,
    gene_info: pd.DataFrame | None = None,
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
        var = pd.DataFrame(index=counts.columns)
        if gene_info is not None:
            unknown = counts.columns.difference(gene_info.index)
            if len(unknown):
                raise ValueError(
                    f"[{name}] Counts contain unresolved gene columns: {unknown.tolist()}"
                )
            var = gene_info.loc[counts.columns].copy()
        sample_adata = ad.AnnData(
            X=csr_matrix(counts.loc[spot_ids].to_numpy(dtype=np.int32)),
            obs=obs,
            var=var,
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
    if not isinstance(obs, pd.DataFrame):
        raise TypeError("Gene preprocessing requires in-memory observation metadata")
    # Annotation IDs can be strings in HER2ST and integers in STNet. HDF5
    # cannot store a mixed object column; keep identifiers as strings and
    # preserve missing values without changing numeric expression fields.
    # Categorical encoding also handles a dataset-specific column becoming
    # entirely missing after observations are split into training/test files.
    for column in obs.select_dtypes(include="object"):
        obs[column] = pd.Categorical(
            obs[column].map(lambda value: str(value) if pd.notna(value) else None)
        )
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
    source_var = sample_adatas[names[0]].var
    if not isinstance(source_var, pd.DataFrame):
        raise TypeError("Gene preprocessing requires in-memory gene metadata")
    if "hgnc_id" in source_var:
        processed.var["hgnc_id"] = source_var.loc[selected_genes, "hgnc_id"].to_numpy(
            dtype=str
        )
    processed.obsm[f"spatial_{target_mpp:g}mpp"] = obs[["pixel_x", "pixel_y"]].to_numpy(
        dtype=np.float32
    )
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
    if not sample_names:
        raise ValueError("No samples available for gene preprocessing")
    for name in sample_names:
        if not sample_adatas[name].var_names.is_unique:
            raise ValueError(f"[{name}] Gene names must be unique")
    common_genes = sorted(
        set.intersection(*(set(sample_adatas[name].var_names) for name in sample_names))
    )
    if not common_genes:
        raise ValueError("Training and test samples have no common genes")

    # Fix the normalization universe using all participating train/test samples.
    # Gene ranking below still receives training samples only.
    aligned_adatas: dict[str, ad.AnnData] = {}
    excluded_by_slide: dict[str, int] = {}
    for name in sample_names:
        sample = sample_adatas[name][:, common_genes].copy()
        if sample.X is None:
            raise ValueError(f"[{name}] Raw counts are missing")
        totals = np.asarray(csr_matrix(sample.X).sum(axis=1)).reshape(-1)
        usable = totals > 0
        excluded_by_slide[name] = int((~usable).sum())
        aligned_adatas[name] = sample[usable].copy()
    excluded_count = sum(excluded_by_slide.values())

    fold_root = output_root
    fold_root.mkdir(parents=True, exist_ok=True)
    for test_patient in patient_names:
        train_names = [name for name in sample_names if name[0] != test_patient]
        test_names = [name for name in sample_names if name[0] == test_patient]
        if not train_names or not test_names:
            raise ValueError(
                f"[patient={test_patient}] Training and test samples are required"
            )
        if selection_mode in ("svgs", "union"):
            for name in train_names:
                if aligned_adatas[name].n_obs < 2:
                    raise ValueError(
                        f"[{name}] SVG selection requires at least two spots with "
                        "positive counts over the common genes"
                    )
        train_adatas = {name: aligned_adatas[name] for name in train_names}
        _, selected_gene_table = select_genes(
            train_adatas,
            n_top_genes,
            selection_mode,
            target_sum,
            svg_n_neighbors,
            union_rank_and_truncate,
        )
        selected_genes: list[str] = selected_gene_table["gene"].tolist()
        if not selected_genes:
            raise ValueError(
                f"[patient={test_patient}] No common genes passed gene selection"
            )
        missing_by_slide = {
            name: [
                gene
                for gene in selected_genes
                if gene not in sample_adatas[name].var_names
            ]
            for name in test_names
        }
        processed = process_gene_expression(
            aligned_adatas,
            sample_names,
            common_genes,
            selected_gene_table,
            target_sum,
            target_mpp,
        )
        processed.uns.update(
            test_patient=test_patient,
            train_names=train_names,
            test_names=test_names,
            gene_selection_mode=selection_mode,
            gene_selection_scope="train_only",
            n_top_genes=n_top_genes,
            n_selected_genes=len(selected_genes),
            target_sum=target_sum,
            union_rank_and_truncate=union_rank_and_truncate,
            common_gene_source="train_test_columns",
            common_gene_scope="train_test",
            common_gene_sample_names=np.asarray(sample_names, dtype=str),
            common_genes=np.asarray(common_genes, dtype=str),
            normalization_genes=np.asarray(common_genes, dtype=str),
            excluded_zero_count_spots=excluded_count,
            excluded_zero_count_spots_by_slide=excluded_by_slide,
            missing_selected_genes_by_slide={
                name: np.asarray(genes, dtype=str)
                for name, genes in missing_by_slide.items()
            },
        )
        if "hgnc_id" in processed.var:
            processed.uns["gene_identifier_type"] = "hgnc_approved_symbol"
            first_var = aligned_adatas[sample_names[0]].var
            if not isinstance(first_var, pd.DataFrame):
                raise TypeError("HGNC metadata requires in-memory gene annotations")
            processed.uns["normalization_hgnc_ids"] = first_var.loc[
                common_genes, "hgnc_id"
            ].to_numpy(dtype=str)
            processed.uns["preprocessing_id"] = hashlib.sha256(
                json.dumps(
                    {
                        "task": f"patient_loocv/test_{test_patient}",
                        "normalization_genes": common_genes,
                        "gene_names": selected_genes,
                        "target_sum": float(target_sum),
                        "expression_transform": "log1p",
                        "gene_hgnc_ids": processed.var["hgnc_id"].tolist(),
                        "normalization_hgnc_ids": processed.uns[
                            "normalization_hgnc_ids"
                        ].tolist(),
                    },
                    sort_keys=True,
                ).encode()
            ).hexdigest()
        output = fold_root / f"test_{test_patient}.h5ad"
        temporary = output.with_suffix(".h5ad.tmp")
        processed.write_h5ad(temporary, compression="gzip")
        temporary.replace(output)
        missing_count = len(
            {gene for genes in missing_by_slide.values() for gene in genes}
        )
        print(
            f"[patient={test_patient}] common={len(common_genes)}, selected={len(selected_genes)}, missing_selected={missing_count}, excluded_zero_count_spots={excluded_count}, shape={processed.shape}",
            flush=True,
        )


def run(cfg):
    if str(cfg.task.split_mode) == "transfer":
        run_external(cfg=cfg)
        return
    if str(cfg.task.split_mode) != "patient_loocv":
        raise ValueError(f"Unknown split mode: {cfg.task.split_mode}")
    dataset = cfg.source_dataset
    gene = cfg.task.gene
    root = Path(str(dataset.data_dir)).expanduser().resolve()
    patients = sorted(dataset.patient_names)
    names = [
        path.parent.name
        for patient in patients
        for path in sorted((root / "samples").glob(f"{patient}*/gene_counts.tsv.gz"))
    ]
    output_root = Path(str(cfg.task.gene_expression_dir)).expanduser().resolve()
    target_mpp = float(dataset.target_mpp)
    gene_info = counts_gene_info(data_dir=root)
    samples = load_raw_gene_data(
        samples_root=root / "samples",
        names=names,
        target_mpp=target_mpp,
        gene_info=gene_info,
    )
    prepare_tasks(
        sample_adatas=samples,
        sample_names=names,
        output_root=output_root,
        n_top_genes=int(gene.n_top_genes),
        target_sum=float(gene.target_sum),
        selection_mode=str(gene.selection_mode),
        svg_n_neighbors=int(gene.svg_n_neighbors),
        target_mpp=target_mpp,
        union_rank_and_truncate=bool(gene.union_rank_and_truncate),
        patient_names=patients,
    )


def prepare_external_task(
    train_samples,
    test_samples,
    train_output,
    test_output,
    *,
    n_top_genes=512,
    target_sum=10000.0,
    selection_mode="svgs",
    svg_n_neighbors=8,
    target_mpp=1.0,
    union_rank_and_truncate=False,
    task_name="her2st_to_stnet",
):
    """Fix the joint gene universe, then rank genes using training slices only."""
    train_names, test_names = list(train_samples), list(test_samples)
    if not train_names or not test_names or set(train_names) & set(test_names):
        raise ValueError("External tasks require nonempty, disjoint sample names")
    samples = {**train_samples, **test_samples}
    if any(not sample.var_names.is_unique for sample in samples.values()):
        raise ValueError("External task gene names must be unique after ID mapping")
    common = sorted(
        set.intersection(*(set(sample.var_names) for sample in samples.values()))
    )
    if not common:
        raise ValueError("Training and external test data have no common genes")
    aligned, excluded = {}, {}
    for name, sample in samples.items():
        sample = sample[:, common].copy()
        usable = np.asarray(sample.X.sum(axis=1)).reshape(-1) > 0
        excluded[name] = int((~usable).sum())
        aligned[name] = sample[usable].copy()
    if any(aligned[name].n_obs < 2 for name in train_names):
        raise ValueError("Training slices need at least two positive-count spots")
    _, table = select_genes(
        {name: aligned[name] for name in train_names},
        n_top_genes,
        selection_mode,
        target_sum,
        svg_n_neighbors,
        union_rank_and_truncate,
    )
    if table.empty:
        raise ValueError("No genes passed training-only SVG selection")
    spec = {
        "task": str(task_name),
        "common_gene_source": "train_test_columns",
        "normalization_genes": common,
        "gene_names": table.gene.tolist(),
        "target_sum": float(target_sum),
        "expression_transform": "log1p",
        "gene_selection_mode": selection_mode,
        "gene_selection_scope": "train_only",
        "n_top_genes": int(n_top_genes),
        "svg_n_neighbors": int(svg_n_neighbors),
        "union_rank_and_truncate": bool(union_rank_and_truncate),
        "train_names": train_names,
        "test_names": test_names,
    }
    first_var = aligned[train_names[0]].var
    if "hgnc_id" in first_var:
        spec["gene_hgnc_ids"] = first_var.loc[table.gene.tolist(), "hgnc_id"].tolist()
        spec["normalization_hgnc_ids"] = first_var.loc[common, "hgnc_id"].tolist()
    spec["preprocessing_id"] = hashlib.sha256(
        json.dumps(spec, sort_keys=True).encode()
    ).hexdigest()
    processed = process_gene_expression(
        aligned, list(aligned), common, table, target_sum, target_mpp
    )
    processed.uns.update(
        external_task=str(task_name),
        preprocessing_id=spec["preprocessing_id"],
        train_names=train_names,
        test_names=test_names,
        normalization_genes=np.asarray(common, dtype=str),
        common_genes=np.asarray(common, dtype=str),
        common_gene_source="train_test_columns",
        common_gene_scope="train_test",
        gene_selection_scope="train_only",
        gene_selection_mode=selection_mode,
        target_sum=float(target_sum),
        n_top_genes=n_top_genes,
        n_selected_genes=len(table),
        union_rank_and_truncate=bool(union_rank_and_truncate),
        common_gene_sample_names=np.asarray(list(aligned), dtype=str),
        excluded_zero_count_spots=sum(excluded.values()),
        excluded_zero_count_spots_by_slide=excluded,
        missing_selected_genes_by_slide={
            name: np.asarray([], dtype=str) for name in test_names
        },
    )
    if "gene_hgnc_ids" in spec:
        processed.uns["gene_identifier_type"] = "hgnc_approved_symbol"
        processed.uns["normalization_hgnc_ids"] = np.asarray(
            spec["normalization_hgnc_ids"], dtype=str
        )
    for role, names, path in (
        ("train", train_names, Path(train_output)),
        ("test", test_names, Path(test_output)),
    ):
        data = processed[processed.obs["name"].isin(names).to_numpy()].copy()
        data.uns["data_role"] = role
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".h5ad.tmp")
        data.write_h5ad(temporary, compression="gzip")
        temporary.replace(path)
        print(
            f"[external {role}] common={len(common)}, shape={data.shape}, output={path}",
            flush=True,
        )
    spec_path = Path(train_output).parent / "gene_spec.json"
    spec_path.write_text(json.dumps(spec, indent=2) + "\n")
    return spec


def run_external(cfg):
    from seg2.common.task import gene_expression_path
    from seg2.derive import sample_names

    root = Path(str(cfg.target_dataset.data_dir)).expanduser().resolve()
    reference = Path(str(cfg.source_dataset.data_dir)).expanduser().resolve()
    name = str(cfg.task.name)
    train_output = gene_expression_path(cfg=cfg, fold_name=name)
    test_output = gene_expression_path(cfg=cfg, fold_name=name, target=True)
    train_names = sample_names(dataset=cfg.source_dataset)
    test_names = sample_names(dataset=cfg.target_dataset)
    mpp = float(cfg.source_dataset.target_mpp)
    if mpp != float(cfg.target_dataset.target_mpp):
        raise ValueError("External expression tasks require matching prepared MPP")
    gene = cfg.task.gene
    train_info = counts_gene_info(data_dir=reference)
    test_info = counts_gene_info(data_dir=root)
    shared = train_info.index.intersection(test_info.index)
    if not train_info.loc[shared, "hgnc_id"].equals(test_info.loc[shared, "hgnc_id"]):
        raise ValueError("Common gene symbols have inconsistent HGNC identities")
    prepare_external_task(
        train_samples=load_raw_gene_data(
            samples_root=reference / "samples",
            names=train_names,
            target_mpp=mpp,
            gene_info=train_info,
        ),
        test_samples=load_raw_gene_data(
            samples_root=root / "samples",
            names=test_names,
            target_mpp=mpp,
            gene_info=test_info,
        ),
        train_output=train_output,
        test_output=test_output,
        n_top_genes=int(gene.n_top_genes),
        target_sum=float(gene.target_sum),
        selection_mode=str(gene.selection_mode),
        svg_n_neighbors=int(gene.svg_n_neighbors),
        target_mpp=mpp,
        union_rank_and_truncate=bool(gene.union_rank_and_truncate),
        task_name=name,
    )
