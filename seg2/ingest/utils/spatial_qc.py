"""Shared spot QC overlays for prepared spatial transcriptomics samples."""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from numpy.typing import NDArray


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
