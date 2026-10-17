import math
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq
import yaml
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

from derive.dataset_config import load_dataset_config


def rank_genes(
    common_genes: list[str],
    scores: NDArray[np.floating],
    n_top_genes: int,
    score_column: str,
) -> pd.DataFrame:
    ranking_scores = np.nan_to_num(scores, nan=-np.inf)
    ranked_indices = np.argsort(-ranking_scores, kind='stable')[:n_top_genes]
    return pd.DataFrame(
        {
            score_column: scores[ranked_indices],
            'selection_rank': np.arange(len(ranked_indices), dtype=np.int64),
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
        assert adata.n_obs >= 2, f'{name}: SVG selection needs at least two spots'

        sample = adata[:, common_genes].copy()
        sample.X = csr_matrix(sample.X, dtype=np.float32)
        sc.pp.normalize_total(sample, target_sum=target_sum)
        sc.pp.log1p(sample)
        sample.obsm['spatial'] = cast(pd.DataFrame, sample.obs)[['pixel_x', 'pixel_y']].to_numpy(
            dtype=np.float64
        )
        sq.gr.spatial_neighbors_knn(
            sample,
            spatial_key='spatial',
            n_neighs=min(svg_n_neighbors, sample.n_obs - 1),
        )
        moran_table = sq.gr.spatial_autocorr(
            sample,
            genes=common_genes,
            mode='moran',
            corr_method=None,
            copy=True,
            n_jobs=1,
            show_progress_bar=False,
        )
        assert moran_table is not None
        morans_by_sample.append(moran_table.reindex(common_genes)['I'].to_numpy())

    morans = np.stack(morans_by_sample)
    finite = np.isfinite(morans)
    finite_counts = finite.sum(axis=0)
    median_morans = np.full(len(common_genes), np.nan, dtype=np.float64)
    usable = finite_counts > 0
    median_morans[usable] = np.nanmedian(morans[:, usable], axis=0)
    positive_counts = ((morans > 0) & finite).sum(axis=0)
    positive_fractions = positive_counts / np.maximum(finite_counts, 1)
    # 跨切片一致: 至少 ceil(0.7 * 切片数) 张切片 Moran's I > 0 才算稳定
    required_positive_samples = max(1, math.ceil(0.7 * len(sample_adatas)))
    consistent = positive_counts >= required_positive_samples

    ranking_scores = np.nan_to_num(median_morans, nan=-np.inf)
    ranking = pd.DataFrame(
        {
            'gene': common_genes,
            'morans_i': median_morans,
            'morans_i_positive_samples': positive_counts,
            'morans_i_positive_fraction': positive_fractions,
            'morans_i_consistent': consistent,
            'morans_i_required_positive_samples': required_positive_samples,
            'selection_score': ranking_scores,
        }
    )
    ranking = ranking.loc[ranking['morans_i_consistent']]
    assert len(ranking) >= n_top_genes, (
        f'only {len(ranking)} genes passed the cross-slice Moran I consistency filter, '
        f'fewer than the requested {n_top_genes}'
    )
    ranking = ranking.sort_values(
        ['selection_score', 'gene'],
        ascending=[False, True],
        kind='stable',
    ).head(n_top_genes)
    ranking['selection_rank'] = np.arange(len(ranking), dtype=np.int64)
    return ranking.set_index('gene').rename_axis(None)


def select_hvgs(
    sample_adatas: Mapping[str, ad.AnnData],
    common_genes: list[str],
    n_top_genes: int,
) -> pd.DataFrame:
    combined = ad.concat(
        {name: adata[:, common_genes] for name, adata in sample_adatas.items()},
        join='inner',
        label='name',
        index_unique='-',
    )
    sc.pp.highly_variable_genes(
        combined,
        flavor='seurat_v3',
        n_top_genes=n_top_genes,
        batch_key='name',
    )
    gene_stats = cast(pd.DataFrame, combined.var)
    selected = gene_stats.loc[gene_stats['highly_variable']].copy()
    selected['selection_score'] = selected['variances_norm']
    selected['selection_rank'] = selected['highly_variable_rank']
    return selected


def select_union(
    sample_adatas: Mapping[str, ad.AnnData],
    common_genes: list[str],
    n_top_genes: int,
    target_sum: float,
    svg_n_neighbors: int,
    union_rank_and_truncate: bool = False,
) -> pd.DataFrame:
    # SVG 与 HVG 各取 n_top_genes 个; 可选交集优先排序并截取
    svg = select_stable_svgs(
        sample_adatas,
        common_genes,
        n_top_genes,
        target_sum,
        svg_n_neighbors,
    )
    hvg = select_hvgs(sample_adatas, common_genes, n_top_genes)
    union = sorted(set(svg.index) | set(hvg.index))
    table = pd.DataFrame(index=pd.Index(union))
    table['svg_rank'] = svg['selection_rank'].reindex(union)
    table['hvg_rank'] = hvg['selection_rank'].reindex(union)
    if union_rank_and_truncate:
        in_both = table[['svg_rank', 'hvg_rank']].notna().all(axis=1)
        order = pd.DataFrame(
            {'in_both': in_both, 'best_rank': table[['svg_rank', 'hvg_rank']].min(axis=1)}
        ).sort_values(
            ['in_both', 'best_rank'],
            ascending=[False, True],
            kind='stable',
        ).head(n_top_genes)
        table = table.loc[order.index]
    table['selection_score'] = -table[['svg_rank', 'hvg_rank']].min(axis=1)
    table['selection_rank'] = np.arange(len(table), dtype=np.int64)
    return table


def select_genes(
    sample_adatas: Mapping[str, ad.AnnData],
    n_top_genes: int,
    mode: str,
    target_sum: float,
    svg_n_neighbors: int,
    union_rank_and_truncate: bool = False,
) -> tuple[list[str], pd.DataFrame]:
    assert mode in {'hvgs', 'svgs', 'hegs', 'union'}, (
        f'unsupported gene selection mode: {mode}'
    )

    adatas = list(sample_adatas.values())
    common_gene_set = set(adatas[0].var_names).intersection(
        *(set(adata.var_names) for adata in adatas[1:])
    )
    common_genes = sorted(common_gene_set)
    assert common_genes, 'samples do not share any genes'

    n_selected = min(n_top_genes, len(common_genes))
    if mode == 'hvgs':
        selected = select_hvgs(sample_adatas, common_genes, n_selected)
    elif mode == 'union':
        selected = select_union(
            sample_adatas,
            common_genes,
            n_selected,
            target_sum,
            svg_n_neighbors,
            union_rank_and_truncate,
        )
    elif mode == 'svgs':
        selected = select_stable_svgs(
            sample_adatas,
            common_genes,
            n_selected,
            target_sum,
            svg_n_neighbors,
        )
    else:
        # hegs: normalize 后的平均表达降序; combined 只用于这个均值, 就地归一化
        combined = ad.concat(
            {name: adata[:, common_genes] for name, adata in sample_adatas.items()},
            join='inner',
            label='name',
            index_unique='-',
        )
        sc.pp.normalize_total(combined, target_sum=target_sum)
        mean_expression: NDArray[np.floating] = np.asarray(
            cast(csr_matrix | NDArray[np.floating], combined.X).mean(axis=0),
        ).reshape(-1).astype(np.float32)
        selected = rank_genes(
            common_genes,
            mean_expression,
            n_selected,
            'mean_expression',
        )
        selected['selection_score'] = selected['mean_expression']

    selected.insert(0, 'gene', selected.index)
    selected = selected.sort_values('gene', kind='stable').reset_index(drop=True)
    selected.insert(1, 'input_channel', np.arange(len(selected)))
    selected.insert(2, 'selection_mode', mode)
    return common_genes, selected


def load_raw_gene_data(
    samples_root: Path,
    names: list[str],
    target_mpp: float,
) -> dict[str, ad.AnnData]:
    sample_adatas: dict[str, ad.AnnData] = {}
    for name in names:
        print(f'[{name}] loading raw gene counts', flush=True)
        sample_root = samples_root / name
        count_path = sample_root / 'gene_counts.tsv.gz'
        spatial_path = sample_root / 'spatial' / f'spatial_{target_mpp:g}mpp.tsv'
        counts = pd.read_csv(count_path, sep='\t', index_col=0)
        spatial = pd.read_csv(spatial_path, sep='\t')

        required_columns = {
            'spot_id',
            'selected',
            'has_expression',
            'in_content',
            'pixel_x',
            'pixel_y',
        }
        missing = required_columns.difference(spatial.columns)
        assert not missing, f'{spatial_path}: missing columns {sorted(missing)}'

        counts.index = counts.index.astype(str)
        counts.columns = counts.columns.astype(str)
        spatial['spot_id'] = spatial['spot_id'].astype(str)

        usable = (
            spatial['selected'].eq(1)
            & spatial['in_content']
            & spatial['has_expression']
        )

        spatial = spatial.loc[usable].copy()

        spot_ids: list[str] = spatial['spot_id'].tolist()
        obs = spatial.set_index('spot_id', drop=False)
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
        join='outer',
        label='name',
        index_unique='-',
        fill_value=0,
    )
    combined = combined[:, common_genes].copy()
    counts = csr_matrix(combined.X, dtype=np.float32)
    combined.X = counts
    obs = cast(pd.DataFrame, combined.obs)
    obs['normalization_total_counts'] = np.asarray(counts.sum(axis=1)).reshape(-1)
    combined.obs_names = (
        obs['name'].astype(str)
        + ':'
        + obs['spot_id'].astype(str)
    ).tolist()
    combined.obs_names.name = 'observation_id'

    sc.pp.normalize_total(combined, target_sum=target_sum)
    sc.pp.log1p(combined)

    selected_genes: list[str] = selected_gene_table['gene'].tolist()
    processed = combined[:, selected_genes].copy()
    processed.X = csr_matrix(processed.X, dtype=np.float32)
    processed.var = selected_gene_table.set_index('gene').loc[selected_genes].copy()
    processed.obsm[f'spatial_{target_mpp:g}mpp'] = cast(pd.DataFrame, processed.obs)[
        ['pixel_x', 'pixel_y']
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
    """按患者留出选基因；所有切片使用训练患者确定的基因集合。"""
    fold_root = output_root / 'patient_loocv'
    fold_root.mkdir(parents=True, exist_ok=True)

    for test_patient in patient_names:
        train_names = [name for name in sample_names if name[0] != test_patient]
        test_names = [name for name in sample_names if name[0] == test_patient]
        # 测试患者的所有切片均不参与基因筛选。
        train_adatas = {name: sample_adatas[name] for name in train_names}
        common_genes, selected_gene_table = select_genes(
            train_adatas,
            n_top_genes,
            selection_mode,
            target_sum,
            svg_n_neighbors,
            union_rank_and_truncate,
        )
        selected_genes: list[str] = selected_gene_table['gene'].tolist()
        missing_by_slide = {
            name: [gene for gene in selected_genes if gene not in sample_adatas[name].var_names]
            for name in test_names
        }
        # 使用训练患者确定的基因集合处理全部切片。
        processed = process_gene_expression(
            sample_adatas,
            sample_names,
            common_genes,
            selected_gene_table,
            target_sum,
            target_mpp,
        )

        valid_counts = cast(pd.DataFrame, processed.obs)['normalization_total_counts'].to_numpy() > 0
        excluded_count = int((~valid_counts).sum())
        processed = processed[valid_counts].copy()

        # 保存划分、处理参数和表达评估需要的缺失基因信息。
        processed.uns.update(
            test_patient=test_patient,
            train_names=train_names,
            test_names=test_names,
            gene_selection_mode=selection_mode,
            n_top_genes=n_top_genes,
            target_sum=target_sum,
            union_rank_and_truncate=union_rank_and_truncate,
            missing_selected_genes_by_slide={name: np.asarray(genes, dtype=str) for name, genes in missing_by_slide.items()},
        )
        processed.write_h5ad(
            fold_root / f'test_{test_patient}.h5ad',
            compression='gzip',
        )
        missing_count = len({gene for genes in missing_by_slide.values() for gene in genes})
        print(
            f'[patient={test_patient}] common={len(common_genes)}, '
            f'selected={len(selected_genes)}, missing_selected={missing_count}, '
            f'excluded_zero_count_spots={excluded_count}, '
            f'shape={processed.shape}',
            flush=True,
        )


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    dataset_config = load_dataset_config()
    gene_config = dataset_config['gene']

    with open(str(project_root / "config/resolution.yaml"), "r", encoding="utf-8") as f:
        resolution = cast(dict[str, float], yaml.safe_load(f))

    target_mpp = float(resolution['target_mpp'])
    sample_names = cast(list[str], dataset_config['expression_names'])
    patient_names = cast(list[str], dataset_config['patient_names'])
    selection_mode = cast(str, gene_config['selection_mode'])
    n_top_genes = cast(int, gene_config['n_top_genes'])
    target_sum = cast(float, gene_config['target_sum'])
    svg_n_neighbors = cast(int, gene_config.get('svg_n_neighbors', 8))
    union_rank_and_truncate = cast(
        bool, gene_config.get('union_rank_and_truncate', False)
    )

    mpp_root = (
        project_root
        / 'data'
        / str(dataset_config['dataset'])
        / f'{target_mpp:g}mpp'
    )
    sample_adatas = load_raw_gene_data(
        mpp_root / 'samples',
        sample_names,
        target_mpp,
    )
    prepare_tasks(
        sample_adatas,
        sample_names,
        mpp_root / 'gene_expression',
        n_top_genes,
        target_sum,
        selection_mode,
        svg_n_neighbors,
        target_mpp,
        union_rank_and_truncate,
        patient_names=patient_names,
    )
    print(f'Wrote {mpp_root / "gene_expression"}', flush=True)


if __name__ == '__main__':
    main()
