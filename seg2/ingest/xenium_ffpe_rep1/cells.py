from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def read_cells(raw_root, bundle_name):
    root = Path(raw_root) / "raw"
    cells = pd.read_parquet(path=root / bundle_name / "outs/cells.parquet")
    annotations = pd.read_excel(
        io=root / "Cell_Barcode_Type_Matrices.xlsx",
        sheet_name="Xenium R1 Fig1-5 (supervised)",
        usecols=["Barcode", "Cluster"],
    )
    cells["cell_id"] = cells.cell_id.astype(np.int64)
    annotations["Barcode"] = annotations.Barcode.astype(np.int64)
    cells = cells.merge(
        right=annotations,
        left_on="cell_id",
        right_on="Barcode",
        how="left",
        validate="one_to_one",
    )
    if cells.Cluster.isna().any():
        raise ValueError("Supervised R1 annotations do not cover all Xenium cells")
    positive = {"Invasive_Tumor", "Prolif_Invasive_Tumor", "DCIS_1", "DCIS_2"}
    negative = {
        "Stromal",
        "Macrophages_1",
        "Endothelial",
        "CD4+_T_Cells",
        "Myoepi_ACTA2+",
        "CD8+_T_Cells",
        "B_Cells",
        "Myoepi_KRT15+",
        "Macrophages_2",
        "Perivascular-Like",
        "Stromal_&_T_Cell_Hybrid",
        "IRF7+_DCs",
        "LAMP3+_DCs",
        "Mast_Cells",
    }
    return pd.DataFrame(
        data={
            "spot_id": cells.cell_id.astype(str),
            "cell_type": cells.Cluster,
            "spot_label": np.select(
                [cells.Cluster.isin(positive), cells.Cluster.isin(negative)],
                [1, 0],
                default=255,
            ),
            "pixel_x": cells.x_centroid,
            "pixel_y": cells.y_centroid,
            "selected": 1,
            "has_expression": cells.transcript_counts.gt(0),
        }
    )


def read_boundaries(raw_root, bundle_name):
    path = Path(raw_root) / "raw" / bundle_name / "outs/cell_boundaries.parquet"
    # Read every row group: this release's aggregate footer row count is truncated.
    batches = pq.ParquetFile(source=path).iter_batches(
        columns=["cell_id", "vertex_x", "vertex_y"]
    )
    boundaries = pd.concat(
        objs=[batch.to_pandas() for batch in batches], ignore_index=True
    )
    boundaries = boundaries.rename(columns={"cell_id": "region_id"})
    boundaries["region_id"] = boundaries.region_id.astype(np.int32)
    if not np.isfinite(boundaries[["vertex_x", "vertex_y"]].to_numpy()).all():
        raise ValueError("Cell boundaries contain nonfinite coordinates")
    return boundaries
