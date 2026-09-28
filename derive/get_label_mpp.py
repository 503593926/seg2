from pathlib import Path
import cv2
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from typing import cast
import yaml
from derive.dataset_config import load_dataset_config


def _write_image(path: Path, image: NDArray[np.uint8], parameters: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image, parameters):
        raise RuntimeError(f"Cannot write image: {path}")


def process_sample(
    target_root: Path,
    output_root: Path,
    name: str,
    target_mpp: float,
    label_mpp: float,
) -> None:
    target_spatial = target_root / name / "spatial"
    output_spatial = output_root / name / "spatial"
    output_spatial.mkdir(parents=True, exist_ok=True)

    image_path = target_spatial / f"he_{target_mpp:g}mpp.png"
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    assert image_bgr is not None

    tissue_path = target_spatial / "tissue_mask.png"
    tissue_mask = cv2.imread(str(tissue_path), cv2.IMREAD_GRAYSCALE)
    assert tissue_mask is not None

    target_shape = tissue_mask.shape
    scale = target_mpp / label_mpp
    processing_shape = (
        max(1, round(target_shape[0] * scale)),
        max(1, round(target_shape[1] * scale)),
    )
    output_size = (processing_shape[1], processing_shape[0])
    processing_image: NDArray[np.uint8] = cv2.resize(
        image_bgr,
        output_size,
        interpolation=cv2.INTER_AREA,
    ).astype(np.uint8, copy=False)
    processing_tissue: NDArray[np.uint8] = cv2.resize(
        tissue_mask,
        output_size,
        interpolation=cv2.INTER_NEAREST,
    ).astype(np.uint8, copy=False)

    spatial_path = target_spatial / f"spatial_{target_mpp:g}mpp.tsv"
    spatial = pd.read_csv(spatial_path, sep="\t")

    scale_x = processing_shape[1] / target_shape[1]
    scale_y = processing_shape[0] / target_shape[0]

    spatial["pixel_x"] *= scale_x
    spatial["pixel_y"] *= scale_y
    rounded_x = np.clip(
        np.rint(spatial["pixel_x"]).astype(np.int64), 0, processing_shape[1] - 1
    )
    rounded_y = np.clip(
        np.rint(spatial["pixel_y"]).astype(np.int64), 0, processing_shape[0] - 1
    )
    spatial["in_tissue_mask"] = processing_tissue[rounded_y, rounded_x] > 0

    _write_image(
        output_spatial / f"he_{label_mpp:g}mpp.png",
        processing_image,
        [cv2.IMWRITE_PNG_COMPRESSION, 3],
    )
    _write_image(
        output_spatial / "tissue_mask.png",
        processing_tissue,
        [cv2.IMWRITE_PNG_COMPRESSION, 3],
    )
    spatial.to_csv(
        output_spatial / f"spatial_{label_mpp:g}mpp.tsv",
        sep="\t",
        index=False,
    )


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    dataset_config = load_dataset_config()
    data_root = project_root / "data" / dataset_config["dataset"]

    with open(str(project_root / "config/resolution.yaml"), "r", encoding="utf-8") as f:
        resolution = cast(dict[str, float], yaml.safe_load(f))
    target_mpp: float = resolution["target_mpp"]
    label_mpp: float = float(resolution["label_mpp"])
    
    output_root = data_root / f"{target_mpp:g}mpp/label_mpp"
    samples_root = data_root / f"{target_mpp:g}mpp" / "samples"

    for name in dataset_config["names"]:
        print(f"[{name}] preparing aligned {label_mpp:g}-MPP label grid")
        process_sample(
            samples_root,
            output_root,
            name,
            target_mpp=target_mpp,
            label_mpp=label_mpp,
        )


if __name__ == "__main__":
    main()
