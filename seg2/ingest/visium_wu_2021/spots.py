from pathlib import Path

import numpy as np
import pandas as pd

from seg2.ingest.utils.gene_mapping import read_mtx_table


def read_spots(record, spot_diameter_um):
    spatial = pd.read_csv(
        filepath_or_buffer=record["position_path"],
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
    metadata = pd.read_csv(
        filepath_or_buffer=record["annotation_path"],
        index_col=0,
        dtype={"patientid": str},
    )
    metadata.index = metadata.index.astype(str)
    barcodes = read_mtx_table(path=Path(record["count_dir"]) / "barcodes.tsv.gz")[0]
    if (
        not spatial.spot_id.is_unique
        or not metadata.index.is_unique
        or not barcodes.is_unique
    ):
        raise ValueError(
            f"[{record['name']}] Position, count and metadata barcodes must be unique"
        )
    if set(metadata.index) != set(barcodes) or not set(barcodes) <= set(
        spatial.spot_id
    ):
        raise ValueError(
            f"[{record['name']}] Counts and metadata must match and have spatial positions"
        )
    if not metadata.patientid.eq(str(record["name"])).all():
        raise ValueError(
            f"[{record['name']}] Metadata patient IDs do not match the section"
        )
    if (
        not spatial.in_tissue.isin([0, 1]).all()
        or not np.isfinite(
            spatial[
                ["array_row", "array_col", "source_pixel_x", "source_pixel_y"]
            ].to_numpy(dtype=float)
        ).all()
    ):
        raise ValueError(f"[{record['name']}] Invalid tissue flags or coordinates")
    metadata = metadata.rename(
        columns={"patientid": "patient_id", "Classification": "annotation"}
    )
    metadata = metadata.rename_axis(index="spot_id").reset_index()
    spatial = spatial.merge(
        right=metadata, on="spot_id", how="left", validate="one_to_one"
    )
    positive = {
        "DCIS",
        "Invasive cancer",
        "Invasive cancer + lymphocytes",
        "Invasive cancer + stroma",
        "Invasive cancer + stroma + lymphocytes",
        "Invasive cancer + adipose tissue + lymphocytes",
        "Cancer trapped in lymphocyte aggregation",
    }
    negative = {
        "Adipose tissue",
        "Lymphocytes",
        "Stroma",
        "Stroma + adipose tissue",
        "Normal duct",
        "Normal glands + lymphocytes",
        "Normal + stroma + lymphocytes",
        "TLS",
    }
    spatial["spot_label"] = np.select(
        condlist=[spatial.annotation.isin(positive), spatial.annotation.isin(negative)],
        choicelist=[1, 0],
        default=255,
    ).astype(np.uint8)
    spatial["has_annotation"] = spatial.annotation.notna()
    spatial["has_expression"] = spatial.spot_id.isin(barcodes)
    # The filtered matrices are a subset of positions, including a few outside auto-tissue.
    spatial["selected"] = (
        spatial.in_tissue.eq(1) & spatial.annotation.ne("Artefact")
    ).astype(np.uint8)
    spatial["patient_id"] = spatial.patient_id.fillna(str(record["name"]))
    spatial["annotation"] = spatial.annotation.fillna("undetermined")
    spatial["spot_diameter_um"] = float(spot_diameter_um)
    return spatial
