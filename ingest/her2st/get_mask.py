from pathlib import Path
import cv2
import numpy as np
from numpy.typing import NDArray
from ingest.her2st.qc_visualize import qc_visualize_mask
from ingest.utils.annotation2mask import ClassInfo, get_sem_mask

def _write_image(path: Path, image: NDArray[np.uint8], parameters: list[int]) -> None:
    if not cv2.imwrite(str(path), image, parameters):
        raise RuntimeError(f'Cannot write image: {path}')

def _binary_mask(mask: NDArray[np.uint8], ids: list[int]) -> NDArray[np.uint8]:
    return np.where(np.isin(mask, ids), 255, 0).astype(np.uint8)

def save_results(
    semantic_mask: NDArray[np.uint8],
    name: str,
    save_root: Path,
) -> Path:
    output_dir = save_root / name
    output_dir.mkdir(parents=True, exist_ok=True)
    png_parameters = [cv2.IMWRITE_PNG_COMPRESSION, 9]

    _write_image(output_dir / 'semantic_mask.png', semantic_mask, png_parameters)
    _write_image(
        output_dir / 'invasive_cancer_mask.png',
        _binary_mask(semantic_mask, [1]),
        png_parameters,
    )
    _write_image(
        output_dir / 'cancer_in_situ_mask.png',
        _binary_mask(semantic_mask, [2]),
        png_parameters,
    )
    _write_image(output_dir / 'tumor_mask.png', _binary_mask(semantic_mask, [1, 2]), png_parameters)
    _write_image(
        output_dir / 'valid_mask.png',
        np.where(semantic_mask > 0, 255, 0).astype(np.uint8),
        png_parameters,
    )

    # 0: known non-tumor, 1: tumor, 255: unlabeled/ignore.
    training = np.full(semantic_mask.shape, 255, dtype=np.uint8)
    training[np.isin(semantic_mask, [3, 4, 5, 6])] = 0
    training[np.isin(semantic_mask, [1, 2])] = 1
    _write_image(output_dir / 'training_mask.png', training, png_parameters)

    return output_dir


def process(
    raw_data_root: Path,
    name: str,
    classes: list[ClassInfo],
    save_root: Path,
) -> None:

    img_dir = raw_data_root / 'ST-imgs' / name[0] / name
    img_path = next(iter(img_dir.glob('*.jpg')))

    annotation_name = 'G1_annotated.png' if name == 'G2' else f'{name}_annotated.png'
    annotation_path = raw_data_root / 'ST-pat' / 'img' / annotation_name

    ann_mat = cv2.imread(annotation_path, cv2.IMREAD_COLOR)
    assert ann_mat is not None

    semantic_mask = get_sem_mask(np.asarray(ann_mat, dtype=np.uint8), classes)

    save_results(semantic_mask, name, save_root)
    qc_visualize_mask(semantic_mask, name, classes, img_path, raw_data_root, save_root)


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    raw_data_root = Path('/home/yfm/data/bioinfo/her2st-master/data')
    save_root = project_root / 'data/her2st/original_mask'
    slice_names = ['A1', 'B1', 'C1', 'D1', 'E1', 'F1', 'G2', 'H1']

    classes: list[ClassInfo] = [
        {"id": 1, "label": "invasive cancer", "rgb": (236, 28, 36)},
        {"id": 2, "label": "cancer in situ", "rgb": (255, 127, 39)},
        {"id": 3, "label": "immune infiltrate", "rgb": (255, 242, 0)},
        {"id": 4, "label": "breast glands", "rgb": (14, 209, 69)},
        {"id": 5, "label": "connective tissue", "rgb": (63, 72, 204)},
        {"id": 6, "label": "adipose tissue", "rgb": (112, 227, 223)},
    ]

    for name in slice_names:
        process(raw_data_root, name, classes, save_root)
        print('processed:', name)


if __name__ == '__main__':
    main()
