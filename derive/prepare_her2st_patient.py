"""准备36张HER2ST切片及患者留出表达目标; 参数见同目录her2st_patient.yaml。"""

import gc
import json
import sys
from pathlib import Path

# 支持IDE直接运行本文件。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import yaml

from derive.dataset_config import load_dataset_config
from derive.get_gene_expression import load_raw_gene_data, prepare_tasks
from derive.get_label_mpp import process_sample as prepare_label_grid
from ingest.her2st.prepare_target_mpp import process_sample as prepare_spatial
from ingest.utils.estimate_mpp_from_spatial import estimate_mpp_from_spatial


def prepare_data(config):
    dataset = load_dataset_config()
    if dataset['dataset'] != 'her2st':
        raise ValueError('This preparation script requires dataset: her2st')
    raw_root = Path(config['raw_data_root'])
    names = sorted(path.name.removesuffix('.tsv.gz') for path in (raw_root / 'ST-cnts').glob('*.tsv.gz'))
    if not names:
        raise FileNotFoundError(f'No HER2ST count matrices in {raw_root / "ST-cnts"}')
    resolution = yaml.safe_load((PROJECT_ROOT / 'config/resolution.yaml').read_text())
    target_mpp = float(resolution['target_mpp'])
    label_mpp = float(resolution['label_mpp'])
    data_root = PROJECT_ROOT / 'data/her2st'
    root = data_root / f'{target_mpp:g}mpp'
    mpp_path = data_root / 'mpp.csv'
    mpp_table = pd.read_csv(mpp_path, index_col=0) if mpp_path.exists() else pd.DataFrame()
    records = []
    for name in names:
        if name in mpp_table.index:
            continue
        spatial = pd.read_csv(raw_root / 'ST-spotfiles' / f'{name}_selection.tsv', sep='\t')
        selected = spatial.loc[spatial['selected'] == 1].rename(columns={'new_x': 'array_x', 'new_y': 'array_y'})
        axis, affine = estimate_mpp_from_spatial(selected, 200.0)
        records.append({'name': name, **axis.add_prefix('axis_').to_dict(), **affine.add_prefix('affine_').to_dict()})
    if records:
        mpp_table = pd.concat([mpp_table, pd.DataFrame(records).set_index('name')]).sort_index()
        mpp_table.to_csv(mpp_path)

    classes = [
        {'id': 1, 'label': 'invasive cancer', 'rgb': (236, 28, 36)},
        {'id': 2, 'label': 'cancer in situ', 'rgb': (255, 127, 39)},
        {'id': 3, 'label': 'immune infiltrate', 'rgb': (255, 242, 0)},
        {'id': 4, 'label': 'breast glands', 'rgb': (14, 209, 69)},
        {'id': 5, 'label': 'connective tissue', 'rgb': (63, 72, 204)},
        {'id': 6, 'label': 'adipose tissue', 'rgb': (112, 227, 223)},
    ]
    spot_labels = {item['label']: int(item['id'] <= 2) for item in classes}
    spot_labels['undetermined'] = 255
    for name in names:
        sample = root / 'samples' / name
        ready = all(path.exists() for path in (
            sample / 'spatial' / f'he_{target_mpp:g}mpp.png',
            sample / 'spatial' / f'spatial_{target_mpp:g}mpp.tsv',
            sample / 'gene_counts.tsv.gz',
        ))
        if not config['skip_existing'] or not ready:
            print(f'[{name}] preparing H&E and spatial coordinates', flush=True)
            prepare_spatial(raw_root, root / 'samples', data_root / 'original_mask', mpp_table.loc[name], target_mpp,
                            classes, spot_labels, int(resolution['tile_size']))
        label_spatial = root / 'label_mpp' / name / 'spatial'
        if not config['skip_existing'] or not all(path.exists() for path in (
            label_spatial / 'tissue_mask.png', label_spatial / f'spatial_{label_mpp:g}mpp.tsv',
        )):
            prepare_label_grid(root / 'samples', root / 'label_mpp', name, target_mpp, label_mpp)

    if config['prepare_genes']:
        gene = dataset['gene']
        sample_adatas = load_raw_gene_data(root / 'samples', names, target_mpp)
        prepare_tasks(
            sample_adatas, names, root / 'gene_expression', int(gene['n_top_genes']),
            float(gene['target_sum']), str(gene['selection_mode']), int(gene.get('svg_n_neighbors', 8)),
            target_mpp, bool(gene.get('union_rank_and_truncate', False)), patient_loocv=True,
            evaluation_names=list(config['evaluation_names']) or list(dataset['names']),
        )
        del sample_adatas
        gc.collect()
    if config['prepare_features']:
        prepare_features(root, names, target_mpp, int(resolution['tile_size']), config)
    print(f'Patient-level data preparation complete: {root}', flush=True)


def prepare_features(root, names, target_mpp, tile_size, config):
    import torch
    from derive.get_he_feature_map import process_sample as extract_sample
    from pretrained_model.h_optimus_0 import HO0Extractor
    from pretrained_model.h_optimus_1 import HO1Extractor
    from pretrained_model.uni2 import UNI2Extractor
    from pretrained_model.virchow import VirchowExtractor
    from pretrained_model.virchow2 import Virchow2Extractor

    models = {'ho0': HO0Extractor, 'ho1': HO1Extractor, 'uni2': UNI2Extractor,
              'virchow': VirchowExtractor, 'virchow2': Virchow2Extractor}
    # models = {'ho1': HO1Extractor, 'uni2': UNI2Extractor,
    #           'virchow2': Virchow2Extractor}
    tag = f'tile_{tile_size}_mpp_{target_mpp:g}_resize_1_overlap_1_cls_overlap_1'
    for name in config['feature_models']:
        output = root / f'feature_{name}' / tag
        pending = [slide for slide in names if not config['skip_existing'] or not all(
            (output / slide / file).exists() for file in ('patch_map.npy', 'cls_map.npy')
        )]
        if not pending:
            continue
        print(f'[{name}] extracting {len(pending)} missing slides', flush=True)
        extractor = models[name](batch_size=int(config['feature_batch_size']), device=config['feature_device'],
                                 tile_size=tile_size, resize=True, overlap_average=True, overlap_cls=True)
        output.mkdir(parents=True, exist_ok=True)
        metadata = {'patch_mpp': tile_size * target_mpp / extractor.patch_grid[0],
                    'cls_mpp': extractor.cls_stride * target_mpp}
        metadata_path = output / 'metadata.json'
        if metadata_path.exists() and json.loads(metadata_path.read_text()) != metadata:
            raise ValueError(f'{metadata_path}: existing feature geometry differs')
        metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')
        for slide in pending:
            extract_sample(extractor, slide, root / 'samples' / slide / 'spatial', output / slide, target_mpp)
        del extractor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    config = yaml.safe_load(Path(__file__).with_name('her2st_patient.yaml').read_text())
    prepare_data(config)


if __name__ == '__main__':
    main()
