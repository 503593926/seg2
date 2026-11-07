"""Shared physical scale fitting and canvas generation for standard Visium."""

import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tifffile

from seg2.common.imaging.detect_tissue import detect_tissue
from seg2.common.imaging.estimate_mpp_from_spatial import estimate_mpp_from_spatial
from seg2.common.imaging.pad_to_tile import pad_to_tile
from seg2.common.imaging.resize_image_to_mpp import resize_image_to_mpp
from seg2.common.imaging.tissue_bbox_in_image import tissue_bbox_in_image
from seg2.ingest.utils.spatial_qc import write_canvas_qc, write_spot_overlay


def estimate_visium_mpp(spatial, array_pitch_um):
    # Visium columns stagger: adjacent spots are 2 columns apart within a row.
    # These coordinates express both axes in units of the neighbour spacing.
    lattice = pd.DataFrame(
        data={
            "array_x": spatial.array_col / 2,
            "array_y": spatial.array_row * np.sqrt(3) / 2,
            "pixel_x": spatial.source_pixel_x,
            "pixel_y": spatial.source_pixel_y,
        }
    )
    axis, affine = estimate_mpp_from_spatial(
        spatial=lattice, spot_pitch_um=array_pitch_um
    )
    scales = affine[["mpp_x", "mpp_y"]].to_numpy(dtype=float)
    if not np.isfinite(scales).all() or (scales <= 0).any() or affine["rank"] != 3:
        raise ValueError("Visium positions do not define a valid spatial scale")
    return axis, affine


def process_sample(
    image_path,
    spatial,
    mpp_x,
    mpp_y,
    destination,
    target_mpp,
    tile_size,
    *,
    coordinate_scale=1.0,
):
    if Path(image_path).suffix.lower() == ".png":
        source = cv2.imread(filename=str(image_path), flags=cv2.IMREAD_COLOR)
        if source is None:
            raise FileNotFoundError(image_path)
        source = np.asarray(cv2.cvtColor(src=source, code=cv2.COLOR_BGR2RGB), dtype=np.uint8)
    else:
        # Use the original file directly when possible; other TIFFs need decoding.
        with tifffile.TiffFile(image_path) as handle:
            source = (
                tifffile.memmap(image_path, mode="r")
                if handle.pages[0].is_memmappable
                else handle.series[0].asarray()
            )
    if source.ndim != 3 or source.shape[2] != 3 or source.dtype != np.uint8:
        raise ValueError("Expected an RGB uint8 Visium H&E image")
    source_height, source_width = source.shape[:2]
    image = resize_image_to_mpp(
        img_bgr=source, mpp_x=mpp_x, mpp_y=mpp_y, target_mpp=target_mpp
    )
    del source
    image = np.asarray(cv2.cvtColor(src=image, code=cv2.COLOR_RGB2BGR), dtype=np.uint8)
    height, width = image.shape[:2]
    spatial = spatial.copy()
    spatial["pixel_x"] = (
        spatial.source_pixel_x * coordinate_scale * width / source_width
    )
    spatial["pixel_y"] = (
        spatial.source_pixel_y * coordinate_scale * height / source_height
    )
    if not (
        spatial.pixel_x.between(0, width - 1) & spatial.pixel_y.between(0, height - 1)
    ).all():
        raise ValueError("Visium spot coordinates fall outside the original H&E frame")
    detection = detect_tissue(
        image_bgr=image,
        detection_mpp=4.0,
        closing_diameter_um=112.0,
        source_mpp=target_mpp,
    )
    if not detection.any():
        raise ValueError("No tissue detected in Visium H&E image")
    x0, y0, x1, y1 = tissue_bbox_in_image(
        tissue_detection_mask=detection,
        image_width=width,
        image_height=height,
        margin_px=round(112 / target_mpp),
    )
    selected = spatial.loc[spatial.selected.eq(1)]
    radius = float(selected.spot_diameter_um.max()) / (2 * target_mpp)
    # Retain full selected disks even where image-based tissue detection is pale.
    x0 = max(0, min(x0, math.floor(selected.pixel_x.min() - radius - 1)))
    y0 = max(0, min(y0, math.floor(selected.pixel_y.min() - radius - 1)))
    x1 = min(width, max(x1, math.ceil(selected.pixel_x.max() + radius + 2)))
    y1 = min(height, max(y1, math.ceil(selected.pixel_y.max() + radius + 2)))
    tissue = np.asarray(
        cv2.resize(
            src=detection, dsize=(width, height), interpolation=cv2.INTER_NEAREST
        ),
        dtype=np.uint8,
    )
    image = pad_to_tile(image=image[y0:y1, x0:x1], tile_size=tile_size)
    tissue = pad_to_tile(
        image=tissue[y0:y1, x0:x1], tile_size=tile_size, background=(0,)
    )
    canvas = np.zeros(shape=tissue.shape, dtype=np.uint8)
    canvas[: y1 - y0, : x1 - x0] = 255
    spatial["pixel_x"] -= x0
    spatial["pixel_y"] -= y0
    spatial["spot_radius_px"] = spatial.spot_diameter_um / (2 * target_mpp)
    spatial["in_content"] = spatial.pixel_x.between(
        0, x1 - x0 - 1
    ) & spatial.pixel_y.between(0, y1 - y0 - 1)
    x = np.rint(spatial.pixel_x).astype(int).clip(0, tissue.shape[1] - 1)
    y = np.rint(spatial.pixel_y).astype(int).clip(0, tissue.shape[0] - 1)
    spatial["in_tissue_mask"] = spatial.in_content & (tissue[y, x] > 0)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for filename, values in (
        (f"he_{target_mpp:g}mpp.png", image),
        ("tissue_mask.png", tissue),
        ("canvas_mask.png", canvas),
    ):
        if not cv2.imwrite(
            filename=str(destination / filename),
            img=values,
            params=[cv2.IMWRITE_PNG_COMPRESSION, 3],
        ):
            raise OSError(f"Could not write {filename}")
    spatial.to_csv(
        path_or_buf=destination / f"spatial_{target_mpp:g}mpp.tsv",
        sep="\t",
        index=False,
    )
    write_canvas_qc(spatial_dir=destination, image=image, tissue_mask=tissue, canvas_mask=canvas)
    write_spot_overlay(
        spatial_dir=destination, image=image, spots=spatial, radius_px=radius
    )
