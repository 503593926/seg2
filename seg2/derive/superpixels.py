from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from seg2.common.pseudo_label import requested_superpixel_count
from seg2.common.snic import prepare_snic_lab, run_snic
from seg2.derive.qc.visualization import qc_visualize_superpixels


def process_sample(
    input_root: Path,
    output_root: Path,
    name: str,
    label_mpp: float,
    superpixel_size_um: float,
    compactness: float,
) -> None:
    sample_dir = input_root / name / "spatial"
    output_dir = output_root / name / "snic"
    output_dir.mkdir(parents=True, exist_ok=True)
    image_bgr = cv2.imread(
        str(sample_dir / f"he_{label_mpp:g}mpp.png"), cv2.IMREAD_COLOR
    )
    tissue_mask = cv2.imread(str(sample_dir / "tissue_mask.png"), cv2.IMREAD_GRAYSCALE)
    tissue: NDArray[np.bool_] = tissue_mask > 0
    requested_superpixels = requested_superpixel_count(
        int(tissue.sum()), superpixel_size_um, label_mpp
    )
    superpixels, _ = run_snic(
        prepare_snic_lab(image_bgr),
        tissue,
        requested_superpixels,
        compactness=compactness,
    )
    np.save(output_dir / "superpixels.npy", superpixels)
    qc_visualize_superpixels(image_bgr, superpixels, label_mpp, output_dir)


def run(cfg, force=False):
    target_mpp, label_mpp = (
        float(cfg.prep.resolution.target_mpp),
        float(cfg.prep.resolution.label_mpp),
    )
    output = Path(str(cfg.dataset.data_dir)).expanduser().resolve() / "label_mpp"
    if round(label_mpp / target_mpp) != 1:
        output /= f"{label_mpp:g}mpp"
    for name in cfg.dataset.labeled_names:
        if force or not (output / name / "snic/superpixels.npy").is_file():
            process_sample(
                output,
                output,
                name,
                label_mpp,
                float(cfg.prep.snic.superpixel_size_um),
                float(cfg.prep.snic.compactness),
            )
