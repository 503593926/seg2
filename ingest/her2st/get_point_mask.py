from pathlib import Path
from typing import cast

import cv2
import numpy as np
import pandas as pd
import yaml
from numpy.typing import NDArray

from ingest.her2st.qc_visualize import qc_visualize_point_mask


def _write_image(path: Path, image: NDArray[np.uint8], parameters: list[int]) -> None:
    if not cv2.imwrite(str(path), image, parameters):
        raise RuntimeError(f'Cannot write image: {path}')


def get_point_mask(
    spot_x: NDArray[np.int64],
    spot_y: NDArray[np.int64],
    spot_labels: NDArray[np.uint8],
    shape: tuple[int, int],
) -> NDArray[np.uint8]:
    # 0: non-tumor spot center, 1: tumor spot center, 255: unlabeled.
    point_mask: NDArray[np.uint8] = np.full(shape, 255, dtype=np.uint8)
    point_mask[spot_y, spot_x] = spot_labels
    return point_mask


def get_supervised_flags(
    n_spots: int,
    ratio: float,
    rng: np.random.Generator,
) -> NDArray[np.bool_]:
    supervised: NDArray[np.bool_] = np.zeros(n_spots, dtype=np.bool_)
    supervised[rng.choice(n_spots, size=round(ratio * n_spots), replace=False)] = True
    return supervised


def process_sample(
    samples_root: Path,
    name: str,
    name_index: int,
    target_mpp: float,
    ratios: list[float],
    seed: int,
) -> None:
    sample_dir = samples_root / name
    spatial_dir = sample_dir / 'spatial'
    masks_dir = sample_dir / 'masks'

    image_mat = cv2.imread(str(spatial_dir / f'he_{target_mpp:g}mpp.png'), cv2.IMREAD_COLOR)
    assert image_mat is not None, f'{name}: he_{target_mpp:g}mpp.png not found'
    image_bgr = np.asarray(image_mat, dtype=np.uint8)
    canvas_shape = (image_bgr.shape[0], image_bgr.shape[1])

    spatial = pd.read_csv(spatial_dir / f'spatial_{target_mpp:g}mpp.tsv', sep='\t')
    required = {'spot_id', 'selected', 'in_content', 'spot_label', 'pixel_x', 'pixel_y'}
    missing = required.difference(spatial.columns)
    assert not missing, f'{name}: missing columns {sorted(missing)}'

    # 候选集与 derive 的标签传播一致: selected & in_content; 抽样只在有标签的 spot 上做
    spots = spatial.loc[
        spatial['selected'].eq(1) & spatial['in_content'] & spatial['spot_label'].ne(255)
    ]
    assert len(spots) > 0, f'{name}: no labeled spot'

    spot_x = np.rint(spots['pixel_x'].to_numpy(dtype=np.float64)).astype(np.int64)
    spot_y = np.rint(spots['pixel_y'].to_numpy(dtype=np.float64)).astype(np.int64)
    assert spot_x.min() >= 0 and spot_x.max() < canvas_shape[1], f'{name}: spot x outside canvas'
    assert spot_y.min() >= 0 and spot_y.max() < canvas_shape[0], f'{name}: spot y outside canvas'
    spot_labels: NDArray[np.uint8] = spots['spot_label'].to_numpy(dtype=np.uint8)

    selection = pd.DataFrame({'spot_id': spots['spot_id'].to_numpy()})
    for ratio in ratios:
        percent = round(ratio * 100)
        # 每个比例都用同一个 seed 新建 rng, 抽样结果不受比例顺序影响
        flags = get_supervised_flags(len(spots), ratio, np.random.default_rng([seed, name_index]))
        _write_image(
            masks_dir / f'point_mask_{percent}.png',
            get_point_mask(spot_x[flags], spot_y[flags], spot_labels[flags], canvas_shape),
            [cv2.IMWRITE_PNG_COMPRESSION, 9],
        )
        selection[f'ratio@{ratio:g}'] = flags
        qc_visualize_point_mask(sample_dir, percent, image_bgr, spots, flags)
        print(f'  ratio {ratio:g}: {int(flags.sum())}/{len(spots)} spots supervised', flush=True)

    selection.to_csv(spatial_dir / 'point_selection.tsv', sep='\t', index=False)


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    names = ['A1', 'B1', 'C1', 'D1', 'E1', 'F1', 'G2', 'H1']

    with open(str(project_root / 'config/resolution.yaml'), 'r', encoding='utf-8') as f:
        resolution = cast(dict[str, float], yaml.safe_load(f))
    target_mpp: float = resolution['target_mpp']

    with open(str(project_root / 'ingest/point_mask.yaml'), 'r', encoding='utf-8') as f:
        point_config = cast(dict[str, object], yaml.safe_load(f))
    ratios = [float(ratio) for ratio in cast(list[float], point_config['ratios'])]
    seed = int(cast(int, point_config['seed']))

    samples_root = project_root / 'data/her2st' / f'{target_mpp:g}mpp/samples'
    for name_index, name in enumerate(names):
        print(f'[{name}] generating point masks for {ratios}', flush=True)
        process_sample(samples_root, name, name_index, target_mpp, ratios, seed)


if __name__ == '__main__':
    main()
