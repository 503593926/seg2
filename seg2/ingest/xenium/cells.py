import json
from pathlib import Path

import anndata as ad
import cv2
import h5py
import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix

from seg2.ingest.xenium.image import transform_points


def read_cell_counts(path):
    with h5py.File(path, "r") as file:
        matrix = file["matrix"]

        def read(key: str) -> np.ndarray:
            return np.asarray(matrix[key][:])

        counts = csc_matrix(
            (read("data"), read("indices"), read("indptr")), shape=tuple(read("shape"))
        )
        keep = read("features/feature_type").astype(str) == "Gene Expression"
        genes = read("features/name").astype(str)[keep]
        var = pd.DataFrame(
            {"gene_id": read("features/id").astype(str)[keep]}, index=genes
        )
        ids = read("barcodes").astype(str)
    return (counts[keep].T.tocsr(), ids, var)


def prepare_cells(data_root, sample="XeHBC_Rep1"):
    directory = Path(data_root) / "samples" / sample
    spatial = directory / "spatial"
    geometry = json.loads((spatial / "geometry.json").read_text())
    raw = Path(geometry["source_raw"]) / "outs"
    matrix = np.asarray(geometry["xenium_um_to_input"])
    counts, ids, var = read_cell_counts(raw / "cell_feature_matrix.h5")
    cells = pd.read_parquet(raw / "cells.parquet")
    cells["cell_id"] = cells.cell_id.astype(str)
    cells = cells.set_index("cell_id", drop=False).loc[ids].copy()
    cells.index.name = "observation_id"
    coordinates = transform_points(
        cells[["x_centroid", "y_centroid"]].to_numpy(), matrix
    )
    cells["pixel_x"], cells["pixel_y"] = coordinates.T
    cells["name"] = sample
    height, width = geometry["content_shape"]
    cells["in_content"] = cells.pixel_x.between(
        -0.5, width - 0.5
    ) & cells.pixel_y.between(-0.5, height - 0.5)
    tissue = cv2.imread(str(spatial / "tissue_mask.png"), cv2.IMREAD_GRAYSCALE)
    x = np.rint(coordinates[:, 0]).astype(int).clip(0, tissue.shape[1] - 1)
    y = np.rint(coordinates[:, 1]).astype(int).clip(0, tissue.shape[0] - 1)
    cells["in_tissue_mask"] = cells.in_content & (tissue[y, x] > 0)
    cells["gene_counts"] = np.asarray(counts.sum(axis=1)).ravel()
    for kind in ("cell", "nucleus"):
        boundaries = pd.read_parquet(raw / f"{kind}_boundaries.parquet")
        boundaries["cell_id"] = boundaries.cell_id.astype(str)
        vertices = transform_points(
            boundaries[["vertex_x", "vertex_y"]].to_numpy(), matrix
        )
        boundaries["pixel_x"], boundaries["pixel_y"] = vertices.T
        boundaries["vertex_index"] = boundaries.groupby(
            "cell_id", sort=False
        ).cumcount()
        boundaries.to_parquet(spatial / f"{kind}_boundaries.parquet", index=False)
        if kind == "cell":
            bounds = (
                boundaries.groupby("cell_id")
                .agg(
                    min_x=("pixel_x", "min"),
                    max_x=("pixel_x", "max"),
                    min_y=("pixel_y", "min"),
                    max_y=("pixel_y", "max"),
                )
                .reindex(cells.index)
            )
            cells["boundary_complete"] = (
                bounds.min_x.ge(-0.5)
                & bounds.max_x.le(width - 0.5)
                & bounds.min_y.ge(-0.5)
                & bounds.max_y.le(height - 0.5)
            )
    data = ad.AnnData(X=counts, obs=cells, var=var)
    data.obsm["spatial"] = coordinates
    data.uns.update(
        dataset=geometry["dataset"],
        count_space="raw",
        target_mpp=geometry["target_mpp"],
        coordinate_convention=geometry["coordinate_convention"],
    )
    data.write_h5ad(directory / "cells.h5ad", compression="gzip")
    summary = {
        "cells": data.n_obs,
        "genes": data.n_vars,
        "cells_with_gene_counts": int((cells.gene_counts > 0).sum()),
        "centers_in_content": int(cells.in_content.sum()),
        "centers_in_tissue": int(cells.in_tissue_mask.sum()),
        "complete_cell_boundaries": int(cells.boundary_complete.sum()),
    }
    (directory / "cells_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    return summary


def run(cfg, force=False):
    dataset = cfg.dataset
    root = Path(str(dataset.data_dir)).expanduser().resolve()
    if (
        force
        or not (root / "samples" / dataset.sample / "cells_summary.json").is_file()
    ):
        return prepare_cells(root, str(dataset.sample))
