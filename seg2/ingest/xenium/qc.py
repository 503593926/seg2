import json
from pathlib import Path

import anndata as ad
import cv2
import numpy as np
import pandas as pd

from seg2.ingest.xenium.image import transform_points


def write_png(path, image):
    cv2.imwrite(str(path), image, [cv2.IMWRITE_PNG_COMPRESSION, 3])


def visualize(data_root, sample="XeHBC_Rep1"):
    directory = Path(data_root) / "samples" / sample
    spatial = directory / "spatial"
    output = directory / "qc"
    output.mkdir(exist_ok=True)
    geometry = json.loads((spatial / "geometry.json").read_text())
    mpp = float(geometry["target_mpp"])
    image = cv2.imread(str(spatial / f"he_{mpp:g}mpp.png"))
    backed = ad.read_h5ad(directory / "cells.h5ad", backed="r")
    try:
        cells = backed.obs.copy()
    finally:
        backed.file.close()
    valid = cells.loc[cells.in_content & cells.in_tissue_mask & cells.boundary_complete]
    overview = image
    boundary_counts = {}
    for kind, color in (("cell", (40, 210, 40)), ("nucleus", (240, 200, 0))):
        table = pd.read_parquet(spatial / f"{kind}_boundaries.parquet")
        table = table.loc[table.cell_id.isin(valid.index)].sort_values(
            ["cell_id", "vertex_index"]
        )
        points = np.rint(table[["pixel_x", "pixel_y"]].to_numpy() * 16).astype(np.int32)
        counts = table.groupby("cell_id", sort=False).size().to_numpy(dtype=np.int64)
        outlines = np.split(points, counts.cumsum()[:-1])
        cv2.polylines(overview, outlines, True, color, 1, cv2.LINE_AA, shift=4)
        boundary_counts[kind] = len(counts)
        del table, points, counts, outlines
    cv2.putText(
        overview,
        "cell: green; nucleus: cyan",
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 0),
        2,
    )
    write_png(output / "overview.png", overview)
    (output / "overview.jpg").unlink(missing_ok=True)
    for index in range(1, 5):
        (output / f"cells_{index:02d}.jpg").unlink(missing_ok=True)
    coordinates = cells[["x_centroid", "y_centroid"]].to_numpy()
    restored = transform_points(
        cells[["pixel_x", "pixel_y"]].to_numpy(), geometry["input_to_xenium_um"]
    )
    report = {
        "cells": len(cells),
        "overlay_cells": len(valid),
        "max_coordinate_roundtrip_error_um": float(
            np.max(np.abs(restored - coordinates))
        ),
        "boundary_counts": boundary_counts,
        "input_mpp": mpp,
        "overview_mpp": mpp,
        "overview_shape": list(overview.shape[:2]),
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"QC saved to {output}: {report}", flush=True)
    return report


def run(cfg, force=False):
    return visualize(
        Path(str(cfg.dataset.data_dir)).expanduser().resolve(), str(cfg.dataset.sample)
    )
