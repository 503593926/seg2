from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from seg2.common.pseudo_label import build_pseudo_mask, snap_spots_to_tissue
from seg2.common.rw import (
    build_random_walker_topology,
    propagate_random_walker,
    save_random_walker_topology,
)
from seg2.common.snic import prepare_snic_lab
from seg2.derive.qc.visualization import qc_visualize_pseudo_label


def _write_image(
    path: Path, image: NDArray[np.uint8] | NDArray[np.uint16], parameters: list[int]
) -> None:
    cv2.imwrite(str(path), image, parameters)


def process_sample(
    processing_root: Path,
    samples_root: Path,
    name: str,
    label_mpp: float,
    ratio: float,
    max_snap_distance_um: float,
    beta: float,
    non_cancer_threshold: float,
    cancer_threshold: float,
    qc_spot_diameter_um: float,
) -> None:
    sample_dir = processing_root / name / "spatial"
    snic_dir = processing_root / name / "snic"
    output_dir = processing_root / name / f"pseudo_label_ratio@{ratio * 100:g}"
    output_dir.mkdir(parents=True, exist_ok=True)
    image_bgr = cv2.imread(
        str(sample_dir / f"he_{label_mpp:g}mpp.png"), cv2.IMREAD_COLOR
    )
    tissue_mask_uint8 = cv2.imread(
        str(sample_dir / "tissue_mask.png"), cv2.IMREAD_GRAYSCALE
    )
    tissue_mask_bl: NDArray[np.bool_] = tissue_mask_uint8 > 0
    superpixels: NDArray[np.int32] = np.load(snic_dir / "superpixels.npy")
    spatial = pd.read_csv(sample_dir / f"spatial_{label_mpp:g}mpp.tsv", sep="\t")
    spatial = spatial.loc[spatial["selected"].eq(1) & spatial["in_content"]]
    spot_x: NDArray[np.int64] = np.rint(
        spatial["pixel_x"].to_numpy(dtype=np.float64)
    ).astype(np.int64)
    spot_y: NDArray[np.int64] = np.rint(
        spatial["pixel_y"].to_numpy(dtype=np.float64)
    ).astype(np.int64)
    spot_labels = spatial["spot_label"].to_numpy(dtype=np.uint8).copy()
    point_selection = pd.read_csv(
        samples_root / name / "spatial/point_selection.tsv", sep="\t"
    )
    ratio_column = f"ratio@{ratio:g}"
    supervised_ids = set(point_selection.loc[point_selection[ratio_column], "spot_id"])
    supervised = spatial["spot_id"].isin(supervised_ids).to_numpy()
    spot_labels[~supervised] = 255
    max_snap_distance_px = round(max_snap_distance_um / label_mpp)
    propagation_spot_x, propagation_spot_y = snap_spots_to_tissue(
        spot_x, spot_y, spot_labels, superpixels, max_snap_distance_px
    )
    lab = prepare_snic_lab(image_bgr)
    topology = build_random_walker_topology(superpixels)
    probability_map, _ = propagate_random_walker(
        superpixels,
        lab,
        propagation_spot_x,
        propagation_spot_y,
        spot_labels,
        beta=beta,
        topology=topology,
    )
    pseudo_mask = build_pseudo_mask(
        probability_map, tissue_mask_bl, non_cancer_threshold, cancer_threshold
    )
    del lab
    save_random_walker_topology(output_dir / "rw_topology.npz", topology)
    _write_image(
        output_dir / "pseudo_mask.png", pseudo_mask, [cv2.IMWRITE_PNG_COMPRESSION, 3]
    )
    probability_image: NDArray[np.uint16] = np.rint(
        probability_map * np.iinfo(np.uint16).max
    ).astype(np.uint16)
    _write_image(
        output_dir / "cancer_probability.png",
        probability_image,
        [cv2.IMWRITE_PNG_COMPRESSION, 3],
    )
    qc_visualize_pseudo_label(
        output_dir,
        image_bgr,
        superpixels,
        pseudo_mask,
        spot_x,
        spot_y,
        spot_labels,
        label_mpp,
        qc_spot_diameter_um,
    )


def run(cfg, force=False):
    root = Path(str(cfg.dataset.data_dir)).expanduser().resolve()
    target_mpp, label_mpp = (
        float(cfg.prep.resolution.target_mpp),
        float(cfg.prep.resolution.label_mpp),
    )
    processing = root / "label_mpp"
    if round(label_mpp / target_mpp) != 1:
        processing /= f"{label_mpp:g}mpp"
    processing.mkdir(parents=True, exist_ok=True)
    ratio = float(cfg.prep.rw.pseudo_label_ratio)
    options = cfg.prep.rw
    for name in cfg.dataset.labeled_names:
        destination = processing / name / f"pseudo_label_ratio@{ratio * 100:g}"
        if force or not all(
            (destination / f).is_file()
            for f in ("pseudo_mask.png", "cancer_probability.png")
        ):
            process_sample(
                processing,
                root / "samples",
                name,
                label_mpp,
                ratio,
                float(options.max_snap_distance_um),
                float(options.beta),
                float(options.non_cancer_threshold),
                float(options.cancer_threshold),
                float(cfg.dataset.spot_diameter_um),
            )
