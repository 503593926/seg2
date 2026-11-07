"""QC overlays for prepared spatial transcriptomics samples."""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from numpy.typing import NDArray


def write_canvas_qc(spatial_dir, image, tissue_mask, canvas_mask):
    """Preview registered H&E with tissue (green) and image support (yellow)."""
    if tissue_mask.shape != image.shape[:2] or canvas_mask.shape != image.shape[:2]:
        raise ValueError("QC image and mask shapes differ")
    height, width = image.shape[:2]
    scale = min(1.0, 1600 / max(height, width))
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    thumbnail = cv2.resize(src=image, dsize=size, interpolation=cv2.INTER_AREA)
    tissue = cv2.resize(src=tissue_mask, dsize=size, interpolation=cv2.INTER_NEAREST) > 0
    canvas = cv2.resize(src=canvas_mask, dsize=size, interpolation=cv2.INTER_NEAREST) > 0
    spatial_dir = Path(spatial_dir)
    spatial_dir.mkdir(parents=True, exist_ok=True)
    for filename, filled in (("qc_thumbnail.jpg", False), ("qc_crop_overview.jpg", True)):
        overlay = thumbnail.copy()
        if filled:
            overlay[tissue] = (0.75 * overlay[tissue] + 0.25 * np.array([0, 180, 0])).astype(np.uint8)
            overlay[~canvas] = (0.3 * overlay[~canvas] + 0.7 * 255).astype(np.uint8)
        for mask, color in ((tissue, (0, 180, 0)), (canvas, (0, 200, 255))):
            contours, _ = cv2.findContours(
                image=mask.astype(np.uint8), mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(image=overlay, contours=contours, contourIdx=-1, color=color, thickness=2)
        text = "Tissue: green | H&E support: yellow"
        for color, thickness in (((0, 0, 0), 4), ((255, 255, 255), 1)):
            cv2.putText(
                img=overlay, text=text, org=(12, 28), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.65, color=color, thickness=thickness, lineType=cv2.LINE_AA,
            )
        output = spatial_dir / filename
        if not cv2.imwrite(filename=str(output), img=overlay, params=[cv2.IMWRITE_JPEG_QUALITY, 92]):
            raise OSError(f"Could not write {output}")


def write_region_overlay(spatial_dir, image, region_map, region_ids, labels):
    """Show native polygon labels on the registered H&E output grid (BGR)."""
    height, width = image.shape[:2]
    scale = min(1.0, 1600 / max(height, width))
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    overlay = cv2.resize(src=image, dsize=size, interpolation=cv2.INTER_AREA)
    regions = cv2.resize(src=region_map, dsize=size, interpolation=cv2.INTER_NEAREST)
    colors = np.full((max(int(region_map.max(initial=0)), int(np.max(region_ids, initial=0))) + 1, 3), 120, dtype=np.uint8)
    for value, color in ((1, (0, 0, 255)), (0, (255, 120, 0))):
        colors[np.asarray(region_ids)[np.asarray(labels) == value]] = color
    selected = regions > 0
    overlay[selected] = (0.55 * overlay[selected] + 0.45 * colors[regions[selected]]).astype(np.uint8)
    output = Path(spatial_dir) / "region_qc_overlay.jpg"
    if not cv2.imwrite(filename=str(output), img=overlay, params=[cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise OSError(f"Could not write {output}")
    return output


def write_spot_overlay(
    spatial_dir: Path,
    image: NDArray[np.uint8],
    spots: pd.DataFrame,
    radius_px: float,
) -> Path:
    height, width = image.shape[:2]
    scale = min(1.0, 1600 / max(height, width))
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    overlay = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    sx, sy = size[0] / width, size[1] / height
    selected = spots.loc[spots["selected"].eq(1) & spots["in_content"]]
    points = selected[["pixel_x", "pixel_y"]].to_numpy(dtype=np.float64)
    labels = selected["spot_label"].to_numpy(dtype=np.int64)
    radii = (
        selected["spot_radius_px"].to_numpy(dtype=np.float64)
        if "spot_radius_px" in selected
        else np.full(len(selected), radius_px, dtype=np.float64)
    )
    colors = {1: (0, 0, 255), 0: (255, 120, 0), 255: (120, 120, 120)}
    for (x, y), label, radius in zip(points, labels, radii):
        cv2.ellipse(
            overlay,
            (round(float(x) * sx), round(float(y) * sy)),
            (max(1, round(float(radius) * sx)), max(1, round(float(radius) * sy))),
            0,
            0,
            360,
            colors[int(label)],
            1,
            cv2.LINE_AA,
        )
    spatial_dir.mkdir(parents=True, exist_ok=True)
    output = spatial_dir / "spot_qc_overlay.jpg"
    if not cv2.imwrite(str(output), overlay, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise OSError(f"Could not write {output}")
    return output
