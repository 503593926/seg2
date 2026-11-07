from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from seg2.derive import sample_names
from seg2.ingest.utils.spatial_qc import write_region_overlay
from seg2.test.data import label_sample_dir


def rasterize_regions(boundaries, shape, label_scale):
    """Rasterize original polygons directly on the model output grid (0 = background)."""
    regions = np.zeros(shape=shape, dtype=np.int32)
    ordered = boundaries.sort_values(by="region_id", kind="stable")
    ids, starts = np.unique(ordered.region_id.to_numpy(), return_index=True)
    vertices = (
        ordered[["vertex_x", "vertex_y"]].to_numpy(dtype=np.float64) / label_scale
    )
    fixed_vertices = np.rint(vertices * 256).astype(np.int32)
    for region_id, start, stop in zip(ids, starts, np.r_[starts[1:], len(vertices)]):
        if stop - start < 3 or int(region_id) <= 0:
            raise ValueError(f"Invalid polygon for cell {region_id}")
        cv2.fillPoly(
            img=regions, pts=[fixed_vertices[start:stop]], color=int(region_id), shift=8
        )
    return regions


def run(cfg):
    dataset = cfg.dataset
    if str(dataset.region_type) != "polygon":
        raise ValueError("The regions stage requires polygon geometry")
    root = Path(str(dataset.data_dir))
    target_mpp, label_mpp = (
        float(cfg.prep.resolution.target_mpp),
        float(cfg.prep.resolution.label_mpp),
    )
    for name in sample_names(dataset=dataset):
        source = root / "samples" / name / "spatial"
        destination = (
            label_sample_dir(
                root=root, name=name, label_mpp=label_mpp, target_mpp=target_mpp
            )
            / "spatial"
        )
        image = cv2.imread(filename=str(destination / f"he_{label_mpp:g}mpp.png"))
        if image is None:
            raise FileNotFoundError(destination / f"he_{label_mpp:g}mpp.png")
        boundaries = pd.read_parquet(path=source / "region_boundaries.parquet")
        regions = rasterize_regions(
            boundaries=boundaries,
            shape=image.shape[:2],
            label_scale=label_mpp / target_mpp,
        )
        np.save(file=destination / "region_map.npy", arr=regions)
        spatial = pd.read_csv(
            filepath_or_buffer=source / f"spatial_{target_mpp:g}mpp.tsv", sep="\t"
        )
        write_region_overlay(
            spatial_dir=destination,
            image=image,
            region_map=regions,
            region_ids=spatial.spot_id.to_numpy(dtype=np.int64),
            labels=spatial.spot_label.to_numpy(),
        )
        print(
            f"[{name}] regions complete: {np.count_nonzero(np.bincount(regions.ravel())[1:])} cells on {label_mpp:g}mpp grid",
            flush=True,
        )
