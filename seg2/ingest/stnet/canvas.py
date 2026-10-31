import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from seg2.common.imaging.detect_tissue import detect_tissue
from seg2.common.imaging.pad_to_tile import pad_to_tile
from seg2.common.imaging.resize_image_to_mpp import resize_image_to_mpp
from seg2.common.imaging.tissue_bbox_in_image import tissue_bbox_in_image
from seg2.ingest.her2st.qc import qc_visualize_target_mpp
from seg2.ingest.stnet.index import build_index
from seg2.ingest.stnet.spots import align_spatial_columns, read_spots
from seg2.ingest.utils.spatial_qc import write_spot_overlay


def write_image(path, values):
    if not cv2.imwrite(str(path), values, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
        raise OSError(f"Could not write {path}")


def process_sample(record, spatial, mpp, destination, target_mpp, tile_size):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    image = cv2.imread(record["image_path"], cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read {record['image_path']}")
    image = np.asarray(image, dtype=np.uint8)
    source_height, source_width = image.shape[:2]
    mpp_x, mpp_y = float(mpp["affine_mpp_x"]), float(mpp["affine_mpp_y"])
    image = resize_image_to_mpp(image, mpp_x, mpp_y, target_mpp)
    height, width = image.shape[:2]
    scale_x, scale_y = width / source_width, height / source_height
    spatial = spatial.copy()
    spatial["pixel_x"] = spatial.source_pixel_x * scale_x
    spatial["pixel_y"] = spatial.source_pixel_y * scale_y
    if not (
        spatial.pixel_x.between(0, width - 1) & spatial.pixel_y.between(0, height - 1)
    ).all():
        raise ValueError(f"[{record['section_name']}] Spots fall outside the source image")
    detection = detect_tissue(image, 4.0, 112.0, source_mpp=target_mpp)
    if not detection.any():
        raise ValueError(f"[{record['section_name']}] No tissue detected")
    x0, y0, x1, y1 = tissue_bbox_in_image(
        detection, width, height, round(112 / target_mpp)
    )
    radius = float(spatial.spot_diameter_um.max()) / (2 * target_mpp)
    x0 = max(0, min(x0, math.floor(spatial.pixel_x.min() - radius - 1)))
    y0 = max(0, min(y0, math.floor(spatial.pixel_y.min() - radius - 1)))
    x1 = min(width, max(x1, math.ceil(spatial.pixel_x.max() + radius + 2)))
    y1 = min(height, max(y1, math.ceil(spatial.pixel_y.max() + radius + 2)))
    full_tissue = np.asarray(
        cv2.resize(detection, (width, height), interpolation=cv2.INTER_NEAREST),
        dtype=np.uint8,
    )
    padded = pad_to_tile(image[y0:y1, x0:x1], tile_size)
    tissue = pad_to_tile(full_tissue[y0:y1, x0:x1], tile_size, (0,))
    canvas = np.zeros(tissue.shape, dtype=np.uint8)
    canvas[: y1 - y0, : x1 - x0] = 255
    spatial["pixel_x"] -= x0
    spatial["pixel_y"] -= y0
    spatial["spot_radius_px"] = spatial.spot_diameter_um / (2 * target_mpp)
    spatial["in_content"] = spatial.pixel_x.between(
        0, x1 - x0 - 1
    ) & spatial.pixel_y.between(0, y1 - y0 - 1)
    x = np.rint(spatial.pixel_x).astype(int).clip(0, tissue.shape[1] - 1)
    y = np.rint(spatial.pixel_y).astype(int).clip(0, tissue.shape[0] - 1)
    spatial["in_tissue_mask"] = tissue[y, x] > 0
    spatial = align_spatial_columns(spatial)
    write_image(destination / f"he_{target_mpp:g}mpp.png", padded)
    write_image(destination / "tissue_mask.png", tissue)
    write_image(destination / "canvas_mask.png", canvas)
    spatial.to_csv(
        destination / f"spatial_{target_mpp:g}mpp.tsv", sep="\t", index=False
    )
    qc_visualize_target_mpp(
        destination.parent,
        image,
        full_tissue,
        padded,
        tissue,
        None,
        spatial,
        (x0, y0, x1, y1),
        [],
    )
    write_spot_overlay(destination, padded, spatial, radius)


def run(cfg):
    dataset = cfg.dataset
    root = Path(str(dataset.data_dir)).expanduser().resolve().parent
    scales = pd.read_csv(root / "mpp.csv", index_col=0)
    for record in build_index(raw_root=dataset.raw_root).to_dict("records"):
        destination = (
            Path(str(dataset.data_dir)) / "samples" / record["section_name"] / "spatial"
        )
        spatial = read_spots(
            record=record, spot_diameter_um=float(dataset.spot_diameter_um)
        )
        process_sample(
            record=record,
            spatial=spatial,
            mpp=scales.loc[record["section_name"]],
            destination=destination,
            target_mpp=float(cfg.prep.resolution.target_mpp),
            tile_size=int(cfg.prep.resolution.tile_size),
        )
        print(f"[{record['section_name']}] canvas complete", flush=True)
