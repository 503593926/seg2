import json
from pathlib import Path

import numpy as np
import pandas as pd

from seg2.ingest.utils.visium import estimate_visium_mpp
from seg2.ingest.visium_wu_2021.index import build_index
from seg2.ingest.visium_wu_2021.spots import read_spots


def run(cfg):
    dataset = cfg.dataset
    records = []
    for row in build_index(raw_root=dataset.raw_root).to_dict(orient="records"):
        spatial = read_spots(
            record=row, spot_diameter_um=float(dataset.spot_diameter_um)
        )
        axis, affine = estimate_visium_mpp(
            spatial=spatial, array_pitch_um=float(dataset.array_pitch_um)
        )
        factors = json.loads(Path(row["scale_path"]).read_text())
        scale = float(factors["tissue_hires_scalef"])
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"[{row['name']}] Invalid fullres-to-hires scale")
        records.append(
            {
                "name": row["name"],
                "image_source": "hires",
                "coordinate_scale": scale,
                "fullres_mpp_x": float(affine["mpp_x"]),
                "fullres_mpp_y": float(affine["mpp_y"]),
                "mpp_x": float(affine["mpp_x"]) / scale,
                "mpp_y": float(affine["mpp_y"]) / scale,
                **{f"axis_{key}": value for key, value in axis.items()},
                **{f"affine_{key}": value for key, value in affine.items()},
            }
        )
    root = Path(str(dataset.data_dir)).expanduser().resolve().parent
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data=records).set_index(keys="name").to_csv(
        path_or_buf=root / "mpp.csv"
    )
