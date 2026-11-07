from pathlib import Path
from typing import cast

import h5py
import numpy as np
import pandas as pd


def read_spots(raw_root, bundle_name, spot_diameter_um):
    root = Path(raw_root) / "raw" / bundle_name
    spatial = pd.read_csv(
        filepath_or_buffer=root / "spatial/tissue_positions.csv",
        dtype={"barcode": str},
    ).rename(
        columns={
            "barcode": "spot_id",
            "pxl_col_in_fullres": "source_pixel_x",
            "pxl_row_in_fullres": "source_pixel_y",
        }
    )
    annotation = pd.read_excel(
        io=root / "labels.xlsx", sheet_name="Visium", dtype={"Barcode": str}
    ).rename(
        columns={
            "Barcode": "spot_id",
            "Cluster": "cluster_id",
            "Annotation": "annotation",
        }
    )
    with h5py.File(name=root / "filtered_feature_bc_matrix.h5", mode="r") as handle:
        barcodes = np.asarray(cast(h5py.Dataset, handle["matrix/barcodes"]).asstr()[()])
    if (
        not spatial.spot_id.is_unique
        or not annotation.spot_id.is_unique
        or len(np.unique(barcodes)) != len(barcodes)
    ):
        raise ValueError(
            "CytAssist position, count and annotation barcodes must be unique"
        )
    tissue_ids = set(spatial.loc[spatial.in_tissue.eq(1), "spot_id"])
    if set(barcodes) != tissue_ids or set(annotation.spot_id) != tissue_ids:
        raise ValueError(
            "Filtered counts and annotations must cover the CytAssist tissue barcodes exactly"
        )
    if (
        not spatial.in_tissue.isin([0, 1]).all()
        or not np.isfinite(
            spatial[
                ["array_row", "array_col", "source_pixel_x", "source_pixel_y"]
            ].to_numpy(dtype=float)
        ).all()
    ):
        raise ValueError(
            "CytAssist positions contain invalid tissue flags or coordinates"
        )
    spatial = spatial.merge(
        right=annotation, on="spot_id", how="left", validate="one_to_one"
    )
    positive = {"invasive", "DCIS #1", "DCIS #2"}
    negative = {
        "stromal/endothelial",
        "stromal/endothelial/immune",
        "stromal",
        "adipocytes",
        "immune",
        "myoepithelial/stromal/immune",
    }
    spatial["spot_label"] = np.select(
        condlist=[spatial.annotation.isin(positive), spatial.annotation.isin(negative)],
        choicelist=[1, 0],
        default=255,
    ).astype(np.uint8)
    spatial["has_annotation"] = spatial.annotation.notna()
    spatial["has_expression"] = spatial.spot_id.isin(barcodes)
    spatial["selected"] = spatial.in_tissue.astype(np.uint8)
    spatial["annotation"] = spatial.annotation.fillna("undetermined")
    spatial["spot_diameter_um"] = float(spot_diameter_um)
    return spatial
