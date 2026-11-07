import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from seg2.ingest.utils.visium import estimate_visium_mpp
from seg2.ingest.visium_cytassist_ffpe.spots import read_spots


def read_ome_mpp(image_path):
    with tifffile.TiffFile(image_path) as handle:
        if handle.ome_metadata is None:
            raise ValueError("CytAssist H&E image has no OME metadata")
        pixels = ET.fromstring(text=handle.ome_metadata).find(path="{*}Image/{*}Pixels")
        if pixels is None:
            raise ValueError("OME metadata has no pixel calibration")
        scales = []
        for axis in ("X", "Y"):
            if pixels.attrib.get(f"PhysicalSize{axis}Unit", "µm") not in {"µm", "um"}:
                raise ValueError("OME physical pixel sizes must be in micrometres")
            scales.append(float(pixels.attrib[f"PhysicalSize{axis}"]))
    if not np.isfinite(scales).all() or min(scales) <= 0:
        raise ValueError("OME pixel sizes must be finite and positive")
    return scales[0], scales[1]


def run(cfg):
    dataset = cfg.dataset
    source = Path(str(dataset.raw_root)) / "raw" / str(dataset.bundle_name)
    mpp_x, mpp_y = read_ome_mpp(image_path=source / "image.tif")
    spatial = read_spots(
        raw_root=dataset.raw_root,
        bundle_name=str(dataset.bundle_name),
        spot_diameter_um=float(dataset.spot_diameter_um),
    )
    axis, affine = estimate_visium_mpp(
        spatial=spatial, array_pitch_um=float(dataset.array_pitch_um)
    )
    # Use the microscope calibration; the known lattice spacing supplies a QC check.
    if not np.allclose(
        [mpp_x, mpp_y], affine[["mpp_x", "mpp_y"]].to_numpy(dtype=float), rtol=0.02
    ):
        raise ValueError("CytAssist OME calibration and Visium lattice scales disagree")
    root = Path(str(dataset.data_dir)).expanduser().resolve().parent
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        data=[
            {
                "name": str(dataset.dataset_name),
                "mpp_x": mpp_x,
                "mpp_y": mpp_y,
                **{f"axis_{key}": value for key, value in axis.items()},
                **{f"affine_{key}": value for key, value in affine.items()},
            }
        ]
    ).set_index(keys="name").to_csv(path_or_buf=root / "mpp.csv")
