from pathlib import Path
import cv2
import numpy as np
import yaml
from numpy.typing import NDArray
from typing import cast

from derive.dataset_config import load_dataset_config
from derive.qc_visualize import qc_visualize_superpixels
from tools.snic_features import prepare_snic_lab, run_snic
from tools.pseudo_label import requested_superpixel_count
from tools.label_grid import label_grid_root


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
        str(sample_dir / f"he_{label_mpp:g}mpp.png"),
        cv2.IMREAD_COLOR,
    )
    assert image_bgr is not None
    image_bgr = cast(NDArray[np.uint8], image_bgr)

    tissue_mask = cv2.imread(
        str(sample_dir / "tissue_mask.png"),
        cv2.IMREAD_GRAYSCALE,
    )
    assert tissue_mask is not None
    if image_bgr.shape[:2] != tissue_mask.shape:
        raise ValueError(f"{sample_dir}: H&E image and tissue mask do not align")

    tissue: NDArray[np.bool_] = tissue_mask > 0
    requested_superpixels = requested_superpixel_count(
        int(tissue.sum()),
        superpixel_size_um,
        label_mpp,
    )

    superpixels, _ = run_snic(
        prepare_snic_lab(image_bgr),
        tissue,
        requested_superpixels,
        compactness=compactness,
    )

    np.save(output_dir / "superpixels.npy", superpixels)
    qc_visualize_superpixels(image_bgr, superpixels, label_mpp, output_dir)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    dataset_config = load_dataset_config()
    data_root = project_root / "data" / dataset_config["dataset"]

    with open(str(project_root / "config/pseudo_mask.yaml"), "r", encoding="utf-8") as f:
        pseudo_mask_config = cast(dict[str, float], yaml.safe_load(f))
    superpixel_size_um = float(pseudo_mask_config["superpixel_size_um"])
    compactness = float(pseudo_mask_config["compactness"])

    with open(str(project_root / "config/resolution.yaml"), "r", encoding="utf-8") as f:
        resolution = cast(dict[str, float], yaml.safe_load(f))
    target_mpp: float = resolution["target_mpp"]
    label_mpp = float(resolution["label_mpp"])
    output_root = label_grid_root(data_root / f"{target_mpp:g}mpp", target_mpp, label_mpp)

    for name in dataset_config["labeled_names"]:
        print(f"[{name}] generating SNIC superpixels", flush=True)
        process_sample(
            input_root=output_root,
            output_root=output_root,
            name=name,
            label_mpp=label_mpp,
            superpixel_size_um=superpixel_size_um,
            compactness=compactness,
        )


if __name__ == "__main__":
    main()
