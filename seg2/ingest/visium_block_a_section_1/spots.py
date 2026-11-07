from pathlib import Path
from typing import cast

import h5py
import numpy as np
import pandas as pd


def read_spots(raw_root, bundle_name, spot_diameter_um):
    root = Path(raw_root) / "raw" / bundle_name
    spatial = pd.read_csv(
        filepath_or_buffer=root / "spatial/tissue_positions_list.csv",
        header=None,
        names=[
            "spot_id",
            "in_tissue",
            "array_row",
            "array_col",
            "source_pixel_y",
            "source_pixel_x",
        ],
        dtype={"spot_id": str},
    )
    annotation = pd.read_csv(
        filepath_or_buffer=root / "metadata.tsv", sep="\t", dtype=str
    ).rename(
        columns={
            "ID": "spot_id",
            "annot_type": "annotation",
            "fine_annot_type": "fine_annotation",
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
            "Visium position, count and annotation barcodes must be unique"
        )
    tissue_ids = set(spatial.loc[spatial.in_tissue.eq(1), "spot_id"])
    if set(barcodes) != tissue_ids or set(annotation.spot_id) != tissue_ids:
        raise ValueError(
            "Filtered counts and pathology annotations must cover the tissue spot barcodes exactly"
        )
    if (
        not spatial.in_tissue.isin([0, 1]).all()
        or not np.isfinite(
            spatial[
                ["array_row", "array_col", "source_pixel_x", "source_pixel_y"]
            ].to_numpy(dtype=float)
        ).all()
    ):
        raise ValueError("Visium positions contain invalid tissue flags or coordinates")
    spatial = spatial.merge(
        right=annotation, on="spot_id", how="left", validate="one_to_one"
    )
    spatial["spot_label"] = (
        spatial.annotation.map({"Invasive": 1, "Tumor": 1, "Healthy": 0})
        .fillna(255)
        .astype(np.uint8)
    )
    spatial["has_annotation"] = spatial.annotation.notna()
    spatial["has_expression"] = spatial.spot_id.isin(barcodes)
    spatial["selected"] = spatial.in_tissue.astype(np.uint8)
    spatial["annotation"] = spatial.annotation.fillna("undetermined")
    spatial["fine_annotation"] = spatial.fine_annotation.fillna("undetermined")
    spatial["spot_diameter_um"] = float(spot_diameter_um)
    return spatial
