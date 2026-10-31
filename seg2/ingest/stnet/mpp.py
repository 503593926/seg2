from pathlib import Path

import numpy as np
import pandas as pd

from seg2.common.imaging.estimate_mpp_from_spatial import estimate_mpp_from_spatial
from seg2.ingest.stnet.index import build_index
from seg2.ingest.stnet.spots import read_spots


def run(cfg):
    dataset = cfg.dataset
    root = Path(str(dataset.data_dir)).expanduser().resolve().parent
    output = root / "mpp.csv"
    records = []
    for record in build_index(raw_root=dataset.raw_root).to_dict("records"):
        name = record["section_name"]
        spatial = read_spots(
            record=record, spot_diameter_um=float(dataset.spot_diameter_um)
        ).rename(
            columns={"source_pixel_x": "pixel_x", "source_pixel_y": "pixel_y"}
        )
        axis, affine = estimate_mpp_from_spatial(
            spatial=spatial, spot_pitch_um=float(dataset.array_pitch_um)
        )
        scales = affine[["mpp_x", "mpp_y"]].to_numpy(dtype=float)
        if not np.isfinite(scales).all() or (scales <= 0).any() or affine["rank"] != 3:
            raise ValueError(f"[{name}] Invalid spatial scale fit")
        records.append(
            {
                "name": name,
                **{f"axis_{key}": value for key, value in axis.items()},
                **{f"affine_{key}": value for key, value in affine.items()},
            }
        )
    pd.DataFrame(records).set_index("name").to_csv(output)
