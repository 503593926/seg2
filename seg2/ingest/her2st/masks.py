from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from seg2.ingest.her2st.qc import qc_visualize_mask
from seg2.ingest.utils.annotation2mask import ClassInfo, get_sem_mask


def _write_image(path: Path, image: NDArray[np.uint8], parameters: list[int]) -> None:
    cv2.imwrite(str(path), image, parameters)


def _binary_mask(mask: NDArray[np.uint8], ids: list[int]) -> NDArray[np.uint8]:
    return np.where(np.isin(mask, ids), 255, 0).astype(np.uint8)


def save_results(semantic_mask: NDArray[np.uint8], name: str, save_root: Path) -> Path:
    output_dir = save_root / name
    output_dir.mkdir(parents=True, exist_ok=True)
    png_parameters = [cv2.IMWRITE_PNG_COMPRESSION, 9]
    _write_image(output_dir / "semantic_mask.png", semantic_mask, png_parameters)
    _write_image(
        output_dir / "invasive_cancer_mask.png",
        _binary_mask(semantic_mask, [1]),
        png_parameters,
    )
    _write_image(
        output_dir / "cancer_in_situ_mask.png",
        _binary_mask(semantic_mask, [2]),
        png_parameters,
    )
    _write_image(
        output_dir / "tumor_mask.png",
        _binary_mask(semantic_mask, [1, 2]),
        png_parameters,
    )
    _write_image(
        output_dir / "valid_mask.png",
        np.where(semantic_mask > 0, 255, 0).astype(np.uint8),
        png_parameters,
    )
    training = np.full(semantic_mask.shape, 255, dtype=np.uint8)
    training[np.isin(semantic_mask, [3, 4, 5, 6])] = 0
    training[np.isin(semantic_mask, [1, 2])] = 1
    _write_image(output_dir / "training_mask.png", training, png_parameters)
    return output_dir


def process(
    raw_data_root: Path, name: str, classes: list[ClassInfo], save_root: Path
) -> None:
    img_dir = raw_data_root / "ST-imgs" / name[0] / name
    img_path = next(iter(img_dir.glob("*.jpg")))
    annotation_name = "G1_annotated.png" if name == "G2" else f"{name}_annotated.png"
    annotation_path = raw_data_root / "ST-pat" / "img" / annotation_name
    ann_mat = cv2.imread(annotation_path, cv2.IMREAD_COLOR)
    semantic_mask = get_sem_mask(np.asarray(ann_mat, dtype=np.uint8), classes)
    save_results(semantic_mask, name, save_root)
    qc_visualize_mask(semantic_mask, name, classes, img_path, raw_data_root, save_root)


def run(cfg, force=False):
    from seg2.ingest.her2st.annotation import CLASSES

    dataset = cfg.dataset
    save_root = (
        Path(str(cfg.dataset.data_dir)).expanduser().resolve().parent / "original_mask"
    )
    for name in dataset.labeled_names:
        if force or not all(
            (save_root / name / f).is_file()
            for f in ("training_mask.png", "valid_mask.png")
        ):
            process(Path(str(dataset.raw_root)), name, CLASSES, save_root)
