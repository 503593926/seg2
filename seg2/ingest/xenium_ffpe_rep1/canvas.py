from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tifffile

from seg2.common.imaging.detect_tissue import detect_tissue
from seg2.common.imaging.pad_to_tile import pad_to_tile
from seg2.ingest.utils.spatial_qc import write_canvas_qc
from seg2.ingest.xenium_ffpe_rep1.cells import read_boundaries, read_cells


def run(cfg):
    dataset = cfg.dataset
    name, target_mpp = str(dataset.dataset_name), float(cfg.prep.resolution.target_mpp)
    tile_size = int(cfg.prep.resolution.tile_size)
    bundle = Path(str(dataset.raw_root)) / "raw" / str(dataset.bundle_name)
    mpp = (
        pd.read_csv(
            filepath_or_buffer=Path(str(dataset.data_dir)).parent / "mpp.csv",
            index_col="name",
        )
        .loc[name]
        .to_dict()
    )
    alignment = np.loadtxt(
        fname=bundle / "inputs" / f"{dataset.bundle_name}_he_imagealignment.csv",
        delimiter=",",
    )
    with tifffile.TiffFile(
        bundle / "inputs" / f"{dataset.bundle_name}_he_image.ome.tif"
    ) as handle:
        series = handle.series[0]
        height, width = series.shape[:2]
        level = series.levels[0]
        for candidate in series.levels:
            if (
                max(
                    float(mpp["he_mpp_x"]) * width / candidate.shape[1],
                    float(mpp["he_mpp_y"]) * height / candidate.shape[0],
                )
                <= target_mpp
            ):
                level = candidate
        level_shape = level.shape[:2]
        image = level.asarray()
    transform = alignment @ np.diag(
        [width / image.shape[1], height / image.shape[0], 1.0]
    )
    transform[:2] *= float(mpp["xenium_mpp"]) / target_mpp
    size = (
        int(
            np.ceil(
                float(mpp["reference_width"]) * float(mpp["xenium_mpp"]) / target_mpp
            )
        ),
        int(
            np.ceil(
                float(mpp["reference_height"]) * float(mpp["xenium_mpp"]) / target_mpp
            )
        ),
    )
    image = cv2.warpAffine(
        src=image,
        M=transform[:2],
        dsize=size,
        flags=cv2.INTER_LINEAR,
        borderValue=(255, 255, 255),
    )
    # Source support excludes unregistered H&E margins and padding from evaluation.
    canvas = cv2.warpAffine(
        src=np.full(level_shape, 255, dtype=np.uint8),
        M=transform[:2],
        dsize=size,
        flags=cv2.INTER_NEAREST,
    )
    image = np.asarray(cv2.cvtColor(src=image, code=cv2.COLOR_RGB2BGR), dtype=np.uint8)
    detection = detect_tissue(
        image_bgr=image,
        detection_mpp=4.0,
        closing_diameter_um=224.0,
        source_mpp=target_mpp,
    )
    tissue = cv2.resize(src=detection, dsize=size, interpolation=cv2.INTER_NEAREST)
    tissue[canvas == 0] = 0
    spatial = read_cells(
        raw_root=dataset.raw_root, bundle_name=str(dataset.bundle_name)
    )
    spatial[["pixel_x", "pixel_y"]] /= target_mpp
    x, y = (
        np.rint(spatial[key]).astype(int).clip(0, size[axis] - 1)
        for axis, key in enumerate(("pixel_x", "pixel_y"))
    )
    spatial["in_content"] = (
        spatial.pixel_x.between(0, size[0] - 1)
        & spatial.pixel_y.between(0, size[1] - 1)
        & (canvas[y, x] > 0)
    )
    spatial["in_tissue_mask"] = tissue[y, x] > 0
    boundaries = read_boundaries(
        raw_root=dataset.raw_root, bundle_name=str(dataset.bundle_name)
    )
    if set(boundaries.region_id.astype(str)) != set(spatial.spot_id):
        raise ValueError("Cell boundary and centroid IDs differ")
    boundaries[["vertex_x", "vertex_y"]] /= target_mpp
    destination = Path(str(dataset.data_dir)) / "samples" / name / "spatial"
    destination.mkdir(parents=True, exist_ok=True)
    image = pad_to_tile(image=image, tile_size=tile_size, background=(255, 255, 255))
    tissue = pad_to_tile(image=np.asarray(tissue, dtype=np.uint8), tile_size=tile_size, background=(0,))
    canvas = pad_to_tile(image=np.asarray(canvas, dtype=np.uint8), tile_size=tile_size, background=(0,))
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
    boundaries.to_parquet(path=destination / "region_boundaries.parquet", index=False)
    write_canvas_qc(spatial_dir=destination, image=image, tissue_mask=tissue, canvas_mask=canvas)
    print(f"[{name}] canvas complete: {len(spatial)} cells", flush=True)
