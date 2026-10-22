from pathlib import Path
from typing import cast

import cv2
import numpy as np
import pandas as pd
import yaml
from numpy.typing import NDArray

from derive.dataset_config import load_dataset_config
from derive.qc_visualize import qc_visualize_pseudo_label
from tools.snic_features import prepare_snic_lab
from tools.propagate_random_walker import (
    build_random_walker_topology,
    propagate_random_walker,
    save_random_walker_topology,
)
from tools.pseudo_label import build_pseudo_mask, snap_spots_to_tissue
from tools.label_grid import label_grid_root

def _write_image(
    path: Path,
    image: NDArray[np.uint8] | NDArray[np.uint16],
    parameters: list[int],
) -> None:
    if not cv2.imwrite(str(path), image, parameters):
        raise RuntimeError(f"Cannot write image: {path}")

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
    output_dir = processing_root / name / f"pseudo_label_ratio@{ratio*100:g}"
    output_dir.mkdir(parents=True, exist_ok=True)

    image_bgr = cv2.imread(
        str(sample_dir / f"he_{label_mpp:g}mpp.png"),
        cv2.IMREAD_COLOR,
    )
    assert image_bgr is not None, f"{sample_dir}: H&E image is not readable"
    image_bgr = cast(NDArray[np.uint8], image_bgr)

    tissue_mask_uint8 = cv2.imread(
        str(sample_dir / "tissue_mask.png"),
        cv2.IMREAD_GRAYSCALE,
    )
    assert tissue_mask_uint8 is not None, f"{sample_dir}: tissue mask is not readable"

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

    spot_labels = cast(
        NDArray[np.uint8],
        spatial["spot_label"].to_numpy(dtype=np.uint8).copy(),
    )

    point_selection = pd.read_csv(
        samples_root / name / "spatial/point_selection.tsv",
        sep="\t",
    )
    ratio_column = f"ratio@{ratio:g}"
    assert ratio_column in point_selection.columns, (
        f"{samples_root / name}: point_selection.tsv has no column {ratio_column}"
    )

    supervised_ids = set(point_selection.loc[point_selection[ratio_column], "spot_id"])
    supervised = spatial["spot_id"].isin(supervised_ids).to_numpy()
    spot_labels[~supervised] = 255

    max_snap_distance_px = round(max_snap_distance_um / label_mpp)
    propagation_spot_x, propagation_spot_y = snap_spots_to_tissue(
        spot_x,
        spot_y,
        spot_labels,
        superpixels,
        max_snap_distance_px,
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
        probability_map,
        tissue_mask_bl,
        non_cancer_threshold,
        cancer_threshold,
    )
    del lab

    save_random_walker_topology(output_dir / "rw_topology.npz", topology)

    _write_image(
        output_dir / "pseudo_mask.png",
        pseudo_mask,
        [cv2.IMWRITE_PNG_COMPRESSION, 3],
    )

    probability_image: NDArray[np.uint16] = np.rint(probability_map * np.iinfo(np.uint16).max).astype(np.uint16)
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

def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    dataset_config = load_dataset_config()
    data_root = project_root / "data" / dataset_config["dataset"]

    with open(project_root / "config/pseudo_mask.yaml", "r", encoding="utf-8") as f:
        pseudo_mask_config = cast(dict[str, float], yaml.safe_load(f))
    with open(project_root / "config/resolution.yaml", "r", encoding="utf-8") as f:
        resolution = cast(dict[str, float], yaml.safe_load(f))
    with open(str(Path(__file__).resolve().parent / 'dataset.yaml'), 'r', encoding="utf-8") as f:
        dataset_file = yaml.safe_load(f)

    target_mpp = float(resolution["target_mpp"])
    label_mpp = float(resolution["label_mpp"])
    samples_root = data_root / f"{target_mpp:g}mpp" / "samples"
    processing_root = label_grid_root(data_root / f"{target_mpp:g}mpp", target_mpp, label_mpp)
    processing_root.mkdir(parents=True, exist_ok=True)
    ratio = float(dataset_file["radio"])

    for name in dataset_config["labeled_names"]:
        print(f"[{name}] generating pseudo label", flush=True)
        process_sample(
            processing_root,
            samples_root,
            name,
            label_mpp=label_mpp,
            ratio=ratio,
            max_snap_distance_um=float(pseudo_mask_config["max_snap_distance_um"]),
            beta=float(pseudo_mask_config["rw_beta"]),
            non_cancer_threshold=float(pseudo_mask_config["non_cancer_threshold"]),
            cancer_threshold=float(pseudo_mask_config["cancer_threshold"]),
            qc_spot_diameter_um=float(dataset_config["spot_diameter_um"]),
        )

if __name__ == "__main__":
    main()
