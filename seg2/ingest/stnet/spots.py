import numpy as np
import pandas as pd


def align_spatial_columns(spatial):
    """Expose common HER2ST fields while retaining STNet source metadata."""
    spatial = spatial.copy()
    spatial["x"], spatial["y"] = spatial["array_x"], spatial["array_y"]
    spatial["new_x"] = spatial["annotation_array_x"].fillna(spatial["array_x"])
    spatial["new_y"] = spatial["annotation_array_y"].fillna(spatial["array_y"])
    spatial["annotation"] = spatial["annotation"].fillna("undetermined")
    common = [
        "spot_id",
        "x",
        "y",
        "new_x",
        "new_y",
        "source_pixel_x",
        "source_pixel_y",
        "selected",
        "annotation",
        "spot_label",
        "has_expression",
        "pixel_x",
        "pixel_y",
        "in_content",
        "in_tissue_mask",
    ]
    return spatial[
        [key for key in common if key in spatial]
        + [key for key in spatial if key not in common]
    ]


def read_spots(record, spot_diameter_um=100.0):
    spatial = pd.read_csv(record["spot_path"], index_col=0)
    spatial.index = spatial.index.astype(str)
    if not spatial.index.is_unique:
        raise ValueError(f"[{record['section_name']}] Duplicate spot IDs")
    spatial = spatial.rename(columns={"X": "source_pixel_x", "Y": "source_pixel_y"})
    spatial.index.name = "spot_id"
    spatial = spatial.reset_index()
    coordinates = spatial.spot_id.str.split("x", expand=True).astype(float)
    spatial["array_x"], spatial["array_y"] = coordinates[0], coordinates[1]
    # The published header omits the annotation ID column and has a trailing tab.
    annotation = pd.read_csv(
        record["annotation_path"],
        sep="\t",
        skiprows=1,
        header=None,
        names=[
            "annotation_id",
            "annotation_array_x",
            "annotation_array_y",
            "label_id",
            "annotation",
        ],
    )
    annotation["spot_id"] = [
        f"{int(np.rint(x))}x{int(np.rint(y))}"
        for x, y in zip(annotation.annotation_array_x, annotation.annotation_array_y)
    ]
    if annotation.spot_id.duplicated().any():
        raise ValueError(
            f"[{record['section_name']}] Rounded annotation coordinates are not unique"
        )
    unmatched = set(annotation.spot_id) - set(spatial.spot_id)
    if unmatched:
        raise ValueError(
            f"[{record['section_name']}] Annotations without positions: {sorted(unmatched)}"
        )
    spatial = spatial.merge(annotation, on="spot_id", how="left", validate="one_to_one")
    labels = spatial.annotation.map({"tumor": 1, "non": 0})
    spatial["spot_label"] = labels.fillna(255).astype(np.uint8)
    spatial["has_annotation"] = labels.notna()
    expression_ids = (
        pd.read_csv(record["count_path"], sep="\t", usecols=[0]).iloc[:, 0].astype(str)
    )
    if expression_ids.duplicated().any():
        raise ValueError(f"[{record['section_name']}] Duplicate count row IDs")
    spatial["has_expression"] = spatial.spot_id.isin(expression_ids)
    spatial["selected"] = 1
    spatial["spot_diameter_um"] = float(spot_diameter_um)
    spatial["name"] = record["section_name"]
    if (
        not np.isfinite(
            spatial[["source_pixel_x", "source_pixel_y", "array_x", "array_y"]]
        )
        .all()
        .all()
    ):
        raise ValueError(f"[{record['section_name']}] Nonfinite spot coordinates")
    return align_spatial_columns(spatial)
