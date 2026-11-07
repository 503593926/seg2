from pathlib import Path

import pandas as pd

from seg2.ingest.utils.visium import estimate_visium_mpp
from seg2.ingest.visium_block_a_section_1.spots import read_spots


def run(cfg):
    dataset = cfg.dataset
    spatial = read_spots(
        raw_root=dataset.raw_root,
        bundle_name=str(dataset.bundle_name),
        spot_diameter_um=float(dataset.spot_diameter_um),
    )
    axis, affine = estimate_visium_mpp(
        spatial=spatial, array_pitch_um=float(dataset.array_pitch_um)
    )
    root = Path(str(dataset.data_dir)).expanduser().resolve().parent
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        data=[
            {
                "name": str(dataset.dataset_name),
                **{f"axis_{key}": value for key, value in axis.items()},
                **{f"affine_{key}": value for key, value in affine.items()},
            }
        ]
    ).set_index(keys="name").to_csv(path_or_buf=root / "mpp.csv")
