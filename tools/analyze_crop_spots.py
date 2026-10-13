"""仅用坐标和掩码重放训练采样，统计每 crop 的有效表达 spot 数。"""

import argparse
import csv
import json
from math import lcm
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import anndata as ad
import cv2
import numpy as np
from omegaconf import OmegaConf
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from segmodel.dataset.augment import augment_geometry, augment_image
from segmodel.dataset.loading import fold_paths, load_feature_scales, read_image
from segmodel.dataset.segmentation import SegmentationDataset
from segmodel.modules.spot_pooling import prepare_spot_pool


def advance_image_rng(rng, cfg):
    # get_augmented_inputs 的 stain seed；HEStain 使用独立的随机状态。
    rng.integers(0, np.iinfo(np.uint32).max)
    if cfg.enabled:
        for name, bounds in (('brightness', 'multiplier_range'), ('contrast', 'factor_range'),
                             ('low_resolution', 'scale_range')):
            options = cfg[name]
            probability = float(options.probability)
            if probability > 0 and rng.random() < probability:
                rng.uniform(*options[bounds])


def verify_image_rng(cfg):
    for seed in range(100):
        actual, replay = np.random.default_rng(seed), np.random.default_rng(seed)
        actual.integers(0, np.iinfo(np.uint32).max)
        if cfg.enabled:
            augment_image(torch.zeros(3, 4, 4, dtype=torch.uint8), np.ones((4, 4), bool), actual, cfg)
        advance_image_rng(replay, cfg)
        assert actual.bit_generator.state == replay.bit_generator.state, 'Image augmentation RNG differs'


def metadata_dataset(cfg, split, seed):
    roots, gene_path = fold_paths(Path(str(cfg.project_root)), cfg.experiment.dataset, split['test'])
    adata = ad.read_h5ad(gene_path, backed='r')
    try:
        obs = adata.obs.copy()
    finally:
        adata.file.close()
    dataset = SegmentationDataset.__new__(SegmentationDataset)
    dataset.seed, dataset.epoch = seed, 0
    dataset.slide_names = sorted(split['train'])
    dataset.labeled_names = sorted(split['train_labeled'])
    dataset.expression_only_names = sorted(set(split['train']) - set(split['train_labeled']))
    dataset.dataset_cfg = cfg.experiment.dataset
    dataset.augment_cfg = cfg.experiment.augment
    dataset.crop_size = int(dataset.dataset_cfg.crop_size)
    dataset.label_scale = round(float(cfg.resolution.label_mpp) / float(cfg.resolution.target_mpp))
    dataset.spot_radius_px = float(dataset.dataset_cfg.spot_diameter_um) / (2 * float(cfg.resolution.target_mpp))
    patch_scale, _ = load_feature_scales(roots[dataset.slide_names[0]], dataset.dataset_cfg.image_feature_maps, cfg.resolution)
    dataset.alignment = lcm(dataset.label_scale, patch_scale)
    dataset.gene_expression = torch.empty(len(obs), 0)
    dataset.spot_labels = torch.full((len(obs),), 255, dtype=torch.long)
    dataset.slides, dataset.spots_by_slide, dataset.crop_origins = {}, {}, {}
    names = obs['name'].astype(str).to_numpy()
    for name in dataset.slide_names:
        label_dir = Path(roots[name]) / 'label_mpp' / name
        tissue = read_image(label_dir / 'spatial/tissue_mask.png', cv2.IMREAD_GRAYSCALE) > 0
        labeled = name in dataset.labeled_names
        pseudo = (read_image(label_dir / str(dataset.dataset_cfg.pseudo_label_dir) / 'pseudo_mask.png', cv2.IMREAD_GRAYSCALE)
                  if labeled else np.full(tissue.shape, 255, np.uint8))
        dataset.slides[name] = SimpleNamespace(has_segmentation=labeled, tissue_mask_bl=tissue, pseudo_mask_uint8=pseudo)
        indices = np.flatnonzero(names == name)
        dataset.spots_by_slide[name] = {
            'coords': obs.iloc[indices][['pixel_x', 'pixel_y']].to_numpy(dtype=np.float32),
            'indices': indices.astype(np.int64),
        }
        dataset._refresh_slide(name)
    return dataset


def describe(values):
    return {
        'crops': int(len(values)), 'mean': float(values.mean()), 'std': float(values.std()),
        'min': int(values.min()), 'max': int(values.max()),
        **{f'p{q:02d}': float(np.percentile(values, q)) for q in (5, 10, 25, 50, 75, 90, 95)},
        'zero_crops': int((values == 0).sum()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--split', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=0)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    threadpool_limits(limits=1, user_api='blas')
    cfg = OmegaConf.load(args.config)
    split = json.loads(args.split.read_text())
    verify_image_rng(cfg.experiment.augment)
    dataset = metadata_dataset(cfg, split, args.seed)
    samples = args.samples or int(dataset.dataset_cfg.samples_per_epoch)
    print('metadata_ready', 'slides', len(dataset.slides), 'samples', samples,
          'alignment', dataset.alignment, 'candidate_crops', sum(map(len, dataset.crop_origins.values())), flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    fields = ['index', 'slide', 'group', 'top', 'left', 'centers_in_crop', 'full_spots', 'valid_spots']
    rows = []
    start = time.monotonic()
    size = dataset.crop_size // dataset.label_scale
    with (args.output / 'crops.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index in range(samples):
            rng = np.random.default_rng(np.random.SeedSequence([args.seed, args.epoch, index]))
            name, top, left = dataset._sample_origin(rng)
            local = dataset.spots_by_slide[name]['coords'] - np.array([left, top], np.float32)
            centers = int(((local >= 0) & (local <= dataset.crop_size - 1)).all(axis=1).sum())
            sample = dataset._spot_targets(name, top, left)
            full = len(sample['spots'])
            y, x = top // dataset.label_scale, left // dataset.label_scale
            sample['tissue'] = torch.from_numpy(dataset.slides[name].tissue_mask_bl[y:y + size, x:x + size][None].copy())
            sample['patch'], sample['cls'] = (), ()
            advance_image_rng(rng, dataset.augment_cfg)
            if dataset.augment_cfg.enabled and dataset.augment_cfg.geometry.enabled:
                sample = augment_geometry(sample, rng, dataset.crop_size, dataset.augment_cfg.geometry)
            valid = prepare_spot_pool(sample['tissue'][0], sample['spots'], dataset.spot_radius_px, dataset.label_scale)['count']
            assert 0 <= valid <= full <= centers
            row = dict(zip(fields, [index, name, 'labeled' if name in dataset.labeled_names else 'expression_only',
                                    top, left, centers, full, valid]))
            writer.writerow(row)
            rows.append(row)
            if (index + 1) % 2000 == 0:
                counts = np.array([row['valid_spots'] for row in rows])
                print('progress', index + 1, 'mean', round(float(counts.mean()), 4),
                      'below_8_pct', round(float((counts < 8).mean() * 100), 3),
                      'seconds', round(time.monotonic() - start, 1), flush=True)

    counts = np.array([row['valid_spots'] for row in rows])
    thresholds = []
    for threshold in (3, 4, 6, 8, 10, 12, 16):
        eligible = counts >= threshold
        thresholds.append({'min_spots': threshold, 'eligible_crops': int(eligible.sum()),
                           'eligible_crop_pct': float(eligible.mean() * 100),
                           'spots_in_eligible_crops_pct': float(counts[eligible].sum() / max(1, counts.sum()) * 100)})
    with (args.output / 'thresholds.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(thresholds[0]))
        writer.writeheader()
        writer.writerows(thresholds)
    by_slide = []
    for name in dataset.slide_names:
        selected = np.array([row['valid_spots'] for row in rows if row['slide'] == name])
        if len(selected):
            by_slide.append({'slide': name, 'group': 'labeled' if name in dataset.labeled_names else 'expression_only',
                             **describe(selected), 'below_8_pct': float((selected < 8).mean() * 100)})
    with (args.output / 'slides.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(by_slide[0]))
        writer.writeheader()
        writer.writerows(by_slide)
    checks = []
    metrics_file = args.split.parent / 'segmentation/metrics.csv'
    batch_size = int(cfg.experiment.training.image.batch_size)
    if metrics_file.exists():
        with metrics_file.open(newline='') as stream:
            for row in csv.DictReader(stream):
                if row.get('epoch') != str(args.epoch) or not row.get('train/gene_spots_step'):
                    continue
                step = int(row['step'])
                first = step * batch_size
                if first + batch_size <= samples:
                    actual = int(float(row['train/gene_spots_step']))
                    replayed = int(counts[first:first + batch_size].sum())
                    checks.append({'step': step, 'logged': actual, 'replayed': replayed, 'matches': actual == replayed})
    summary = {
        'source_config': str(args.config), 'source_split': str(args.split), 'seed': args.seed, 'epoch': args.epoch,
        'crop_size': dataset.crop_size, 'label_scale': dataset.label_scale, 'alignment': dataset.alignment,
        'spot_radius_px': dataset.spot_radius_px, 'train_slides': dataset.slide_names,
        'overall': describe(counts),
        'centers_in_crop_mean': float(np.mean([row['centers_in_crop'] for row in rows])),
        'full_spots_mean': float(np.mean([row['full_spots'] for row in rows])),
        'tissue_filter_removed_spots': int(sum(row['full_spots'] - row['valid_spots'] for row in rows)),
        'groups': {group: describe(np.array([row['valid_spots'] for row in rows if row['group'] == group]))
                   for group in ('labeled', 'expression_only')},
        'thresholds': thresholds, 'csv_cross_checks': checks,
    }
    (args.output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if checks:
        assert all(check['matches'] for check in checks), 'Spot counts differ from live training CSV'


if __name__ == '__main__':
    main()
