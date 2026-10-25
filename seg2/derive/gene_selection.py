import math
from collections.abc import Mapping

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq
from numpy.typing import NDArray
from scipy.sparse import csr_matrix


def rank_genes(
    common_genes: list[str], scores: NDArray[np.floating], n_top_genes: int, score_column: str
) -> pd.DataFrame:
    ranking_scores = np.nan_to_num(scores, nan=-np.inf)
    ranked_indices = np.argsort(-ranking_scores, kind="stable")[:n_top_genes]
    return pd.DataFrame(
        {
            score_column: scores[ranked_indices],
            "selection_rank": np.arange(len(ranked_indices), dtype=np.int64),
        },
        index=pd.Index(np.asarray(common_genes, dtype=object)[ranked_indices]),
    )


def select_stable_svgs(
    sample_adatas: Mapping[str, ad.AnnData],
    common_genes: list[str],
    n_top_genes: int,
    target_sum: float,
    svg_n_neighbors: int,
) -> pd.DataFrame:
    morans_by_sample: list[NDArray[np.float64]] = []
    for name, adata in sample_adatas.items():
        sample = adata[:, common_genes].copy()
        sample.X = csr_matrix(sample.X, dtype=np.float32)
        sc.pp.normalize_total(sample, target_sum=target_sum)
        sc.pp.log1p(sample)
        sample.obsm["spatial"] = sample.obs[["pixel_x", "pixel_y"]].to_numpy(dtype=np.float64)
        sq.gr.spatial_neighbors_knn(
            sample, spatial_key="spatial", n_neighs=min(svg_n_neighbors, sample.n_obs - 1)
        )
        moran_table = sq.gr.spatial_autocorr(
            sample,
            genes=common_genes,
            mode="moran",
            corr_method=None,
            copy=True,
            n_jobs=1,
            show_progress_bar=False,
        )
        morans_by_sample.append(moran_table.reindex(common_genes)["I"].to_numpy())
    morans = np.stack(morans_by_sample)
    finite = np.isfinite(morans)
    finite_counts = finite.sum(axis=0)
    median_morans = np.full(len(common_genes), np.nan, dtype=np.float64)
    usable = finite_counts > 0
    median_morans[usable] = np.nanmedian(morans[:, usable], axis=0)
    positive_counts = ((morans > 0) & finite).sum(axis=0)
    positive_fractions = positive_counts / np.maximum(finite_counts, 1)
    required_positive_samples = max(1, math.ceil(0.7 * len(sample_adatas)))
    consistent = positive_counts >= required_positive_samples
    ranking_scores = np.nan_to_num(median_morans, nan=-np.inf)
    ranking = pd.DataFrame(
        {
            "gene": common_genes,
            "morans_i": median_morans,
            "morans_i_positive_samples": positive_counts,
            "morans_i_positive_fraction": positive_fractions,
            "morans_i_consistent": consistent,
            "morans_i_required_positive_samples": required_positive_samples,
            "selection_score": ranking_scores,
        }
    )
    ranking = ranking.loc[ranking["morans_i_consistent"]]
    ranking = ranking.sort_values(["selection_score", "gene"], ascending=[False, True], kind="stable").head(
        n_top_genes
    )
    ranking["selection_rank"] = np.arange(len(ranking), dtype=np.int64)
    return ranking.set_index("gene").rename_axis(None)


def select_hvgs(
    sample_adatas: Mapping[str, ad.AnnData], common_genes: list[str], n_top_genes: int
) -> pd.DataFrame:
    combined = ad.concat(
        {name: adata[:, common_genes] for name, adata in sample_adatas.items()},
        join="inner",
        label="name",
        index_unique="-",
    )
    sc.pp.highly_variable_genes(combined, flavor="seurat_v3", n_top_genes=n_top_genes, batch_key="name")
    gene_stats = combined.var
    selected = gene_stats.loc[gene_stats["highly_variable"]].copy()
    selected["selection_score"] = selected["variances_norm"]
    selected["selection_rank"] = selected["highly_variable_rank"]
    return selected


def select_union(
    sample_adatas: Mapping[str, ad.AnnData],
    common_genes: list[str],
    n_top_genes: int,
    target_sum: float,
    svg_n_neighbors: int,
    union_rank_and_truncate: bool = False,
) -> pd.DataFrame:
    svg = select_stable_svgs(sample_adatas, common_genes, n_top_genes, target_sum, svg_n_neighbors)
    hvg = select_hvgs(sample_adatas, common_genes, n_top_genes)
    union = sorted(set(svg.index) | set(hvg.index))
    table = pd.DataFrame(index=pd.Index(union))
    table["svg_rank"] = svg["selection_rank"].reindex(union)
    table["hvg_rank"] = hvg["selection_rank"].reindex(union)
    if union_rank_and_truncate:
        in_both = table[["svg_rank", "hvg_rank"]].notna().all(axis=1)
        order = (
            pd.DataFrame({"in_both": in_both, "best_rank": table[["svg_rank", "hvg_rank"]].min(axis=1)})
            .sort_values(["in_both", "best_rank"], ascending=[False, True], kind="stable")
            .head(n_top_genes)
        )
        table = table.loc[order.index]
    table["selection_score"] = -table[["svg_rank", "hvg_rank"]].min(axis=1)
    table["selection_rank"] = np.arange(len(table), dtype=np.int64)
    return table


def select_genes(
    sample_adatas: Mapping[str, ad.AnnData],
    n_top_genes: int,
    mode: str,
    target_sum: float,
    svg_n_neighbors: int,
    union_rank_and_truncate: bool = False,
) -> tuple[list[str], pd.DataFrame]:
    adatas = list(sample_adatas.values())
    common_gene_set = set(adatas[0].var_names).intersection(*(set(adata.var_names) for adata in adatas[1:]))
    common_genes = sorted(common_gene_set)
    n_selected = min(n_top_genes, len(common_genes))
    if mode == "hvgs":
        selected = select_hvgs(sample_adatas, common_genes, n_selected)
    elif mode == "union":
        selected = select_union(
            sample_adatas, common_genes, n_selected, target_sum, svg_n_neighbors, union_rank_and_truncate
        )
    elif mode == "svgs":
        selected = select_stable_svgs(sample_adatas, common_genes, n_selected, target_sum, svg_n_neighbors)
    else:
        combined = ad.concat(
            {name: adata[:, common_genes] for name, adata in sample_adatas.items()},
            join="inner",
            label="name",
            index_unique="-",
        )
        sc.pp.normalize_total(combined, target_sum=target_sum)
        mean_expression: NDArray[np.floating] = (
            np.asarray(combined.X.mean(axis=0)).reshape(-1).astype(np.float32)
        )
        selected = rank_genes(common_genes, mean_expression, n_selected, "mean_expression")
        selected["selection_score"] = selected["mean_expression"]
    selected.insert(0, "gene", selected.index)
    selected = selected.sort_values("gene", kind="stable").reset_index(drop=True)
    selected.insert(1, "input_channel", np.arange(len(selected)))
    selected.insert(2, "selection_mode", mode)
    return (common_genes, selected)
