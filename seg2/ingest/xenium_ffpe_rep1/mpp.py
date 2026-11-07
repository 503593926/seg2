import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile


def run(cfg):
    dataset = cfg.dataset
    bundle = Path(str(dataset.raw_root)) / "raw" / str(dataset.bundle_name)
    experiment = json.loads((bundle / "outs/experiment.xenium").read_text())
    xenium_mpp = float(experiment["pixel_size"])
    with tifffile.TiffFile(
        bundle / "inputs" / f"{dataset.bundle_name}_he_image.ome.tif"
    ) as handle:
        if handle.ome_metadata is None:
            raise ValueError("H&E image has no OME metadata")
        pixels = ET.fromstring(text=handle.ome_metadata).find(path="{*}Image/{*}Pixels")
        if pixels is None:
            raise ValueError("OME metadata has no pixel calibration")
        he_mpp_x, he_mpp_y = (
            float(pixels.attrib[f"PhysicalSize{axis}"]) for axis in ("X", "Y")
        )
        if any(
            pixels.attrib.get(f"PhysicalSize{axis}Unit", "µm") not in {"µm", "um"}
            for axis in ("X", "Y")
        ):
            raise ValueError("H&E OME pixel sizes must be in micrometres")
    with tifffile.TiffFile(bundle / "outs/morphology_mip.ome.tif") as handle:
        height, width = handle.series[0].shape[-2:]
    alignment = np.loadtxt(
        fname=bundle / "inputs" / f"{dataset.bundle_name}_he_imagealignment.csv",
        delimiter=",",
    )
    if alignment.shape != (3, 3) or not np.allclose(alignment[2], [0, 0, 1]):
        raise ValueError("Expected an affine H&E-to-Xenium alignment matrix")
    effective = np.linalg.norm(alignment[:2, :2], axis=0) * xenium_mpp
    if xenium_mpp <= 0 or not np.allclose(effective, [he_mpp_x, he_mpp_y], rtol=0.02):
        raise ValueError("Alignment and OME physical pixel sizes disagree")
    root = Path(str(dataset.data_dir)).parent
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        data=[
            {
                "name": str(dataset.dataset_name),
                "he_mpp_x": he_mpp_x,
                "he_mpp_y": he_mpp_y,
                "xenium_mpp": xenium_mpp,
                "reference_width": width,
                "reference_height": height,
            }
        ]
    ).set_index(keys="name").to_csv(path_or_buf=root / "mpp.csv")
