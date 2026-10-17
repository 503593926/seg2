import gc
import json
from pathlib import Path
from typing import cast

import numpy as np
import yaml
from numpy.typing import NDArray
from PIL import Image

from derive.dataset_config import load_dataset_config
from derive.qc_visualize import qc_visualize_feature_map
from pretrained_model.common import FeatureMapExtractor
from pretrained_model.h_optimus_1 import HO1Extractor
from pretrained_model.uni2 import UNI2Extractor
from pretrained_model.virchow2 import Virchow2Extractor


def process_sample(
    extractor: FeatureMapExtractor,
    name: str,
    spatial_dir: Path,
    output_dir: Path,
    target_mpp: float,
) -> None:
    he_path = spatial_dir / f'he_{target_mpp:g}mpp.png'
    with Image.open(he_path) as source:
        image: NDArray[np.uint8] = np.asarray(source.convert('RGB'))
    height, width = image.shape[:2]

    # 不足一整块 tile 的右侧、底部不参与提取; patch 网格由模型实际输入尺寸决定
    tile_size = extractor.tile_size
    patch_rows, patch_columns = extractor.patch_grid
    tile_rows, tile_columns = height // tile_size, width // tile_size
    usable_height, usable_width = tile_rows * tile_size, tile_columns * tile_size

    patch_map, cls_map = extractor.extract(image)

    expected_patch_shape = (
        tile_rows * patch_rows,
        tile_columns * patch_columns,
        extractor.patch_dimension,
    )
    expected_cls_shape = (
        (usable_height - tile_size) // extractor.cls_stride + 1,
        (usable_width - tile_size) // extractor.cls_stride + 1,
        extractor.cls_dimension,
    )
    assert patch_map.shape == expected_patch_shape and cls_map.shape == expected_cls_shape, (
        f'{name}: patch/CLS shapes {patch_map.shape}, {cls_map.shape}; '
        f'expected {expected_patch_shape}, {expected_cls_shape}'
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / 'patch_map.npy', patch_map)
    np.save(output_dir / 'cls_map.npy', cls_map)

    print(f'[{name}] patch_map={patch_map.shape} cls_map={cls_map.shape}', flush=True)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    dataset_config = load_dataset_config()
    # 与 segmodel 的 image_feature_maps 对应，逐个加载和释放基础模型。
    extractor_types = [HO1Extractor, UNI2Extractor, Virchow2Extractor]
    batch_size = 512
    device = 'cuda'

    with open(project_root / 'config/resolution.yaml', 'r', encoding='utf-8') as f:
        resolution = cast(dict[str, float], yaml.safe_load(f))
    target_mpp = float(resolution['target_mpp'])
    tile_size = int(resolution['tile_size'])
    expression_names = cast(list[str], dataset_config['expression_names'])

    mpp_root = project_root / 'data' / dataset_config['dataset'] / f'{target_mpp:g}mpp'
    samples_root = mpp_root / 'samples'
    for extractor_type in extractor_types:
        extractor = extractor_type(
            batch_size=batch_size,
            device=device,
            tile_size=tile_size,
            resize=True,
            overlap_average=True,
            overlap_cls=False,
        )
        patch_rows, _ = extractor.patch_grid
        patch_mpp = extractor.tile_size * target_mpp / patch_rows
        cls_mpp = extractor.cls_stride * target_mpp

        # 提取配置进目录名，保留模型真实的 patch / CLS 网格。
        model_tag = extractor.__class__.__name__.removesuffix('Extractor').lower()
        config_tag = f'tile_{extractor.tile_size}_mpp_{target_mpp:g}_resize_{int(extractor.resize)}'
        if extractor.overlap_average:
            config_tag += '_overlap_1'
        if extractor.overlap_cls:
            config_tag += '_cls_overlap_1'
        output_root = mpp_root / f'feature_{model_tag}' / config_tag
        output_root.mkdir(parents=True, exist_ok=True)

        metadata = {'patch_mpp': patch_mpp, 'cls_mpp': cls_mpp}
        (output_root / 'metadata.json').write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + '\n',
            encoding='utf-8',
        )
        print(
            f'Model: {extractor.__class__.__name__} | '
            f'target {target_mpp:g} MPP | patch {patch_mpp:g} MPP | CLS {cls_mpp:g} MPP',
            flush=True,
        )
        for name in expression_names:
            process_sample(
                extractor,
                name,
                spatial_dir=samples_root / name / 'spatial',
                output_dir=output_root / name,
                target_mpp=target_mpp,
            )
        del extractor
        gc.collect()
        qc_visualize_feature_map(
            output_root,
            expression_names,
            patch_mpp=patch_mpp,
            cls_mpp=cls_mpp,
        )


if __name__ == '__main__':
    main()
