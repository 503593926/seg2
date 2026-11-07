from pathlib import Path

import pandas as pd

from seg2.ingest.utils.visium import process_sample
from seg2.ingest.visium_wu_2021.index import build_index
from seg2.ingest.visium_wu_2021.spots import read_spots


def run(cfg):
    dataset = cfg.dataset
    root = Path(str(dataset.data_dir)).expanduser().resolve()
    scales = pd.read_csv(filepath_or_buffer=root.parent / "mpp.csv", index_col="name")
    for row in build_index(raw_root=dataset.raw_root).to_dict(orient="records"):
        name = str(row["name"])
        mpp = scales.loc[name].to_dict()
        spatial = read_spots(
            record=row, spot_diameter_um=float(dataset.spot_diameter_um)
        )
        process_sample(
            image_path=Path(row["image_path"]),
            spatial=spatial,
            mpp_x=float(mpp["mpp_x"]),
            mpp_y=float(mpp["mpp_y"]),
            coordinate_scale=float(mpp["coordinate_scale"]),
            destination=root / "samples" / name / "spatial",
            target_mpp=float(cfg.prep.resolution.target_mpp),
            tile_size=int(cfg.prep.resolution.tile_size),
        )
        print(
            f"[{name}] hires canvas complete; native MPP={mpp['mpp_x']:.3f}/{mpp['mpp_y']:.3f}",
            flush=True,
        )
