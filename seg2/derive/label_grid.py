import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def _read_image(path, mode):
    image = cv2.imread(str(path), mode)
    return image


def _write_image(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image, [cv2.IMWRITE_PNG_COMPRESSION, 3])


def process_sample(target_root, output_root, name, target_mpp, label_mpp):
    source = Path(target_root) / name
    destination = Path(output_root) / name
    image = _read_image(source / f"spatial/he_{target_mpp:g}mpp.png", cv2.IMREAD_COLOR)
    scale = round(label_mpp / target_mpp)
    height, width = image.shape[:2]
    shape = (height // scale, width // scale)

    def resize(array, interpolation):
        array = array[: shape[0] * scale, : shape[1] * scale]
        return cv2.resize(array, (shape[1], shape[0]), interpolation=interpolation)

    _write_image(
        destination / f"spatial/he_{label_mpp:g}mpp.png", resize(image, cv2.INTER_AREA)
    )
    tissue = resize(
        _read_image(source / "spatial/tissue_mask.png", cv2.IMREAD_GRAYSCALE),
        cv2.INTER_NEAREST,
    )
    _write_image(destination / "spatial/tissue_mask.png", tissue)
    canvas = resize(
        _read_image(source / "spatial/canvas_mask.png", cv2.IMREAD_GRAYSCALE),
        cv2.INTER_NEAREST,
    )
    _write_image(destination / "spatial/canvas_mask.png", canvas)
    if (source / "masks/training_mask.png").is_file():
        for filename in ("training_mask.png", "valid_mask.png"):
            values = _read_image(source / "masks" / filename, cv2.IMREAD_GRAYSCALE)
            _write_image(
                destination / "masks" / filename, resize(values, cv2.INTER_NEAREST)
            )
    spatial = pd.read_csv(source / f"spatial/spatial_{target_mpp:g}mpp.tsv", sep="\t")
    spatial["pixel_x"] /= scale
    spatial["pixel_y"] /= scale
    inside = spatial.pixel_x.between(0, shape[1] - 1) & spatial.pixel_y.between(
        0, shape[0] - 1
    )
    spatial["in_content"] = spatial["in_content"] & inside
    spatial["in_tissue_mask"] = False
    selected = spatial.loc[spatial.in_content]
    x = np.rint(selected.pixel_x).astype(np.int64).clip(0, shape[1] - 1)
    y = np.rint(selected.pixel_y).astype(np.int64).clip(0, shape[0] - 1)
    spatial.loc[selected.index, "in_tissue_mask"] = tissue[y, x] > 0
    spatial.to_csv(
        destination / f"spatial/spatial_{label_mpp:g}mpp.tsv", sep="\t", index=False
    )
    (destination / "grid.json").write_text(
        json.dumps(
            {
                "target_mpp": float(target_mpp),
                "label_mpp": float(label_mpp),
                "label_scale": scale,
                "input_shape": [height, width],
                "prediction_shape": list(shape),
            },
            indent=2,
        )
        + "\n"
    )


def run(cfg, force=False):
    target_mpp, label_mpp = (
        float(cfg.prep.resolution[key]) for key in ("target_mpp", "label_mpp")
    )
    root = Path(str(cfg.dataset.data_dir)).expanduser().resolve()
    output = root / "label_mpp"
    if round(label_mpp / target_mpp) != 1:
        output /= f"{label_mpp:g}mpp"
    names = {
        path.parent.name
        for patient in cfg.dataset.patient_names
        for path in (root / "samples").glob(f"{patient}*/gene_counts.tsv.gz")
    } | set(cfg.dataset.labeled_names)
    for name in sorted(names):
        if force or not (output / name / "grid.json").is_file():
            process_sample(root / "samples", output, name, target_mpp, label_mpp)
