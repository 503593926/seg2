import json
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import tifffile

from seg2.common.imaging.detect_tissue import detect_tissue
from seg2.common.imaging.pad_to_tile import pad_to_tile
from seg2.common.imaging.tissue_bbox_in_image import tissue_bbox_in_image


def transform_points(points, matrix):
    points = np.asarray(points, dtype=np.float64)
    homogeneous = np.column_stack((points, np.ones(len(points)))) @ np.asarray(matrix).T
    return homogeneous[:, :2] / homogeneous[:, 2:3]


def input_transform(source_shape, resized_shape, bbox):
    sy, sx = np.asarray(resized_shape, dtype=float) / np.asarray(
        source_shape, dtype=float
    )
    x0, y0 = bbox[:2]
    return np.array(
        [[sx, 0, (sx - 1) / 2 - x0], [0, sy, (sy - 1) / 2 - y0], [0, 0, 1]],
        dtype=np.float64,
    )


def prepare_image(
    source_root, data_root, target_mpp=1.0, tile_size=224, sample="XeHBC_Rep1"
):
    raw_name = "Xenium_FFPE_Human_Breast_Cancer_Rep1"
    raw = Path(source_root) / "raw" / raw_name
    image_path = raw / "inputs" / f"{raw_name}_he_image.ome.tif"
    alignment_path = raw / "inputs" / f"{raw_name}_he_imagealignment.csv"
    with tifffile.TiffFile(image_path) as image_file:
        pixels = next(
            element
            for element in ET.fromstring(image_file.ome_metadata).iter()
            if element.tag.rsplit("}", 1)[-1] == "Pixels"
        )
        source_mpp = np.array(
            [float(pixels.attrib[f"PhysicalSize{axis}"]) for axis in "XY"]
        )
        source_shape = image_file.series[0].shape[:2]
        resized_shape = tuple(
            np.rint(np.asarray(source_shape) * source_mpp[::-1] / target_mpp).astype(
                int
            )
        )
        level_index = 0
        rgb = image_file.series[0].levels[level_index].asarray()
    image = np.asarray(
        cv2.resize(
            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            resized_shape[::-1],
            interpolation=cv2.INTER_AREA,
        ),
        dtype=np.uint8,
    )
    del rgb
    detection = detect_tissue(image, 4.0, 112.0, source_mpp=target_mpp)
    bbox = tissue_bbox_in_image(
        detection, image.shape[1], image.shape[0], round(112 / target_mpp)
    )
    x0, y0, x1, y1 = bbox
    tissue = np.asarray(
        cv2.resize(detection, image.shape[1::-1], interpolation=cv2.INTER_NEAREST),
        dtype=np.uint8,
    )
    image = pad_to_tile(image[y0:y1, x0:x1], tile_size)
    tissue = pad_to_tile(tissue[y0:y1, x0:x1], tile_size, (0,))
    canvas = np.zeros(image.shape[:2], dtype=np.uint8)
    canvas[: y1 - y0, : x1 - x0] = 255
    he_to_input = input_transform(source_shape, resized_shape, bbox)
    he_to_xenium = np.loadtxt(alignment_path, delimiter=",")
    experiment = json.loads((raw / "outs/experiment.xenium").read_text())
    xenium_mpp = float(experiment["pixel_size"])
    um_to_input = (
        he_to_input
        @ np.linalg.inv(he_to_xenium)
        @ np.diag([1 / xenium_mpp, 1 / xenium_mpp, 1])
    )
    geometry = {
        "dataset": "hbc_10x_xenium_ffpe_rep1",
        "sample": sample,
        "source_raw": str(raw.resolve()),
        "source_image": str(image_path.resolve()),
        "alignment_file": str(alignment_path.resolve()),
        "source_mpp_xy": source_mpp.tolist(),
        "xenium_mpp": xenium_mpp,
        "target_mpp": float(target_mpp),
        "source_shape": list(source_shape),
        "read_level": level_index,
        "resized_shape": list(map(int, resized_shape)),
        "crop_bbox": list(map(int, bbox)),
        "content_shape": [y1 - y0, x1 - x0],
        "input_shape": list(image.shape[:2]),
        "tile_size": int(tile_size),
        "coordinate_convention": "zero-based pixel centers; x=column, y=row",
        "source_he_to_input": he_to_input.tolist(),
        "he_to_xenium_pixels": he_to_xenium.tolist(),
        "xenium_um_to_input": um_to_input.tolist(),
        "input_to_xenium_um": np.linalg.inv(um_to_input).tolist(),
    }
    spatial = Path(data_root) / "samples" / sample / "spatial"
    spatial.mkdir(parents=True, exist_ok=True)
    for name, array in (
        (f"he_{target_mpp:g}mpp.png", image),
        ("tissue_mask.png", tissue),
        ("canvas_mask.png", canvas),
    ):
        cv2.imwrite(str(spatial / name), array, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    (spatial / "geometry.json").write_text(json.dumps(geometry, indent=2) + "\n")
    print(
        f"[{sample}] read level {level_index}; {target_mpp:g} mpp input {image.shape[:2]}; {spatial}",
        flush=True,
    )
    return geometry


def run(cfg, force=False):
    dataset = cfg.dataset
    root = Path(str(dataset.data_dir)).expanduser().resolve()
    if (
        force
        or not (root / "samples" / dataset.sample / "spatial/geometry.json").is_file()
    ):
        return prepare_image(
            Path(str(dataset.raw_root)),
            root,
            float(cfg.prep.resolution.target_mpp),
            int(cfg.prep.resolution.tile_size),
            str(dataset.sample),
        )
