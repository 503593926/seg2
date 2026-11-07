from pathlib import Path

import pandas as pd

from seg2.ingest.utils.visium import process_sample
from seg2.ingest.visium_block_a_section_1.spots import read_spots


def run(cfg):
    dataset = cfg.dataset
    name = str(dataset.dataset_name)
    root = Path(str(dataset.data_dir)).expanduser().resolve()
    source = Path(str(dataset.raw_root)) / "raw" / str(dataset.bundle_name)
    mpp = (
        pd.read_csv(filepath_or_buffer=root.parent / "mpp.csv", index_col="name")
        .loc[name]
        .to_dict()
    )
    spatial = read_spots(
        raw_root=dataset.raw_root,
        bundle_name=str(dataset.bundle_name),
        spot_diameter_um=float(dataset.spot_diameter_um),
    )
    process_sample(
        image_path=source / "image.tif",
        spatial=spatial,
        mpp_x=float(mpp["affine_mpp_x"]),
        mpp_y=float(mpp["affine_mpp_y"]),
        destination=root / "samples" / name / "spatial",
        target_mpp=float(cfg.prep.resolution.target_mpp),
        tile_size=int(cfg.prep.resolution.tile_size),
    )
    print(
        f"[{name}] canvas complete: {int(spatial.selected.sum())} tissue spots",
        flush=True,
    )
