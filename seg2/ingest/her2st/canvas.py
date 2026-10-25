import shutil
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from seg2.common.imaging.detect_tissue import detect_tissue
from seg2.common.imaging.pad_to_tile import pad_to_tile
from seg2.common.imaging.resize_image_to_mpp import resize_image_to_mpp
from seg2.common.imaging.tissue_bbox_in_image import tissue_bbox_in_image
from seg2.ingest.her2st.qc import qc_visualize_target_mpp
from seg2.ingest.utils.annotation2mask import ClassInfo


def _write_image(path: Path, image: NDArray[np.uint8], parameters: list[int]) -> None:
    cv2.imwrite(str(path), image, parameters)


def resize_discrete_mask(
    path: Path,
    full_width: int,
    full_height: int,
    bbox: tuple[int, int, int, int],
    tile_size: int,
) -> NDArray[np.uint8]:
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    background = 255 if path.name == "training_mask.png" else 0
    mask_target_mpp = cv2.resize(
        mask, (full_width, full_height), interpolation=cv2.INTER_NEAREST
    )
    x0, y0, x1, y1 = bbox
    return pad_to_tile(mask_target_mpp[y0:y1, x0:x1], tile_size, (background,))


def transform_spatial(
    selection_path: Path,
    annotation_path: Path,
    count_path: Path,
    bbox: tuple[int, int, int, int],
    scale_x: float,
    scale_y: float,
    tissue_mask: NDArray[np.uint8],
    spot_label_map: dict[str, int],
) -> pd.DataFrame:
    spatial = pd.read_csv(selection_path, sep="\t")
    if annotation_path.exists():
        annotations = pd.read_csv(annotation_path, sep="\t").dropna(subset=["x", "y"])
        annotations.rename(
            columns={"x": "new_x", "y": "new_y", "label": "annotation"}, inplace=True
        )
        spatial = spatial.merge(
            annotations[["new_x", "new_y", "annotation"]],
            how="left",
            on=["new_x", "new_y"],
            validate="one_to_one",
        )
    else:
        spatial["annotation"] = "undetermined"
    spatial["spot_label"] = (
        spatial["annotation"].map(spot_label_map).fillna(255).astype(np.uint8)
    )
    x0, y0, x1, y1 = bbox
    content_width = x1 - x0
    content_height = y1 - y0
    spatial.insert(
        0,
        "spot_id",
        [f"{float(x):g}x{float(y):g}" for x, y in zip(spatial["x"], spatial["y"])],
    )
    count_spot_ids = set(
        pd.read_csv(count_path, sep="\t", usecols=[0]).iloc[:, 0].astype(str)
    )
    spatial["has_expression"] = spatial["spot_id"].isin(count_spot_ids)
    spatial.rename(
        columns={"pixel_x": "source_pixel_x", "pixel_y": "source_pixel_y"}, inplace=True
    )
    spatial["pixel_x"] = spatial["source_pixel_x"] * scale_x - x0
    spatial["pixel_y"] = spatial["source_pixel_y"] * scale_y - y0
    spatial["in_content"] = spatial["pixel_x"].between(0, content_width - 1) & spatial[
        "pixel_y"
    ].between(0, content_height - 1)
    inside = spatial["in_content"]
    content_rows = spatial.loc[inside]
    px = np.rint(content_rows["pixel_x"]).astype(int).clip(0, content_width - 1)
    py = np.rint(content_rows["pixel_y"]).astype(int).clip(0, content_height - 1)
    spatial["in_tissue_mask"] = False
    spatial.loc[inside, "in_tissue_mask"] = tissue_mask[py, px] > 0
    return spatial


def process_sample(
    data_root: Path,
    samples_root: Path,
    mask_root: Path,
    mpp_record: pd.Series,
    target_mpp: float,
    classes: Sequence[ClassInfo],
    spot_label_map: dict[str, int],
    tile_size: int = 112,
) -> None:
    detection_mpp = 4.0
    margin_um = 112
    closing_diameter_um = 112
    weak_threshold_ratio = 0.85
    min_tissue_area_um2 = 100000.0
    name = str(mpp_record.name)
    output_dir = samples_root / name
    spatial_dir = output_dir / "spatial"
    masks_dir = output_dir / "masks"
    spatial_dir.mkdir(parents=True, exist_ok=True)
    image_path = next((data_root / "ST-imgs" / name[0] / name).glob("*.jpg"))
    img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    mpp_x = float(mpp_record["axis_mpp_x"])
    mpp_y = float(mpp_record["axis_mpp_y"])
    scale_x = mpp_x / target_mpp
    scale_y = mpp_y / target_mpp
    image_target_mpp = resize_image_to_mpp(img_bgr, mpp_x, mpp_y, target_mpp)
    full_height, full_width = image_target_mpp.shape[:2]
    del img_bgr
    tissue_detection = detect_tissue(
        image_target_mpp,
        detection_mpp,
        closing_diameter_um,
        source_mpp=target_mpp,
        weak_threshold_ratio=weak_threshold_ratio,
        min_tissue_area_um2=min_tissue_area_um2,
    )
    bbox = tissue_bbox_in_image(
        tissue_detection, full_width, full_height, round(margin_um / target_mpp)
    )
    x0, y0, x1, y1 = bbox
    crop_width = x1 - x0
    crop_height = y1 - y0
    padded_image = pad_to_tile(image_target_mpp[y0:y1, x0:x1], tile_size)
    canvas_height, canvas_width = padded_image.shape[:2]
    tissue_target_mpp = cv2.resize(
        tissue_detection, (full_width, full_height), interpolation=cv2.INTER_NEAREST
    )
    tissue_padded = pad_to_tile(tissue_target_mpp[y0:y1, x0:x1], tile_size, (0,))
    canvas_mask = np.zeros((canvas_height, canvas_width), dtype=np.uint8)
    canvas_mask[:crop_height, :crop_width] = 255
    _write_image(
        spatial_dir / f"he_{target_mpp:g}mpp.png",
        padded_image,
        [cv2.IMWRITE_PNG_COMPRESSION, 3],
    )
    _write_image(
        spatial_dir / "tissue_mask.png", tissue_padded, [cv2.IMWRITE_PNG_COMPRESSION, 9]
    )
    _write_image(
        spatial_dir / "canvas_mask.png", canvas_mask, [cv2.IMWRITE_PNG_COMPRESSION, 9]
    )
    semantic_mask_target_mpp: NDArray[np.uint8] | None = None
    for source_mask_path in (mask_root / name).glob("*_mask.png"):
        masks_dir.mkdir(parents=True, exist_ok=True)
        output_mask = resize_discrete_mask(
            source_mask_path, full_width, full_height, bbox, tile_size
        )
        output_mask_path = masks_dir / source_mask_path.name
        _write_image(output_mask_path, output_mask, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        if source_mask_path.name == "semantic_mask.png":
            semantic_mask_target_mpp = output_mask
    selection_path = data_root / "ST-spotfiles" / f"{name}_selection.tsv"
    annotation_path = data_root / "ST-pat" / "lbl" / f"{name}_labeled_coordinates.tsv"
    count_path = data_root / "ST-cnts" / f"{name}.tsv.gz"
    spatial = transform_spatial(
        selection_path,
        annotation_path,
        count_path,
        bbox,
        scale_x,
        scale_y,
        tissue_padded,
        spot_label_map,
    )
    spatial.to_csv(
        spatial_dir / f"spatial_{target_mpp:g}mpp.tsv", sep="\t", index=False
    )
    shutil.copyfile(count_path, output_dir / "gene_counts.tsv.gz")
    qc_visualize_target_mpp(
        output_dir,
        image_target_mpp,
        tissue_target_mpp,
        padded_image,
        tissue_padded,
        semantic_mask_target_mpp,
        spatial,
        bbox,
        classes,
    )


def run(cfg, force=False):
    from seg2.ingest.her2st.annotation import CLASSES

    dataset = cfg.dataset
    raw_data_root = Path(str(dataset.raw_root))
    data_directory = Path(str(cfg.dataset.data_dir)).expanduser().resolve().parent
    mask_root = data_directory / "original_mask"
    target_mpp, tile_size = (
        float(cfg.prep.resolution.target_mpp),
        int(cfg.prep.resolution.tile_size),
    )
    samples_root = Path(str(cfg.dataset.data_dir)).expanduser().resolve() / "samples"
    samples_root.mkdir(parents=True, exist_ok=True)
    mpp_table = pd.read_csv(data_directory / "mpp.csv", index_col=0)
    spot_label_map = {
        "invasive cancer": 1,
        "cancer in situ": 1,
        "immune infiltrate": 0,
        "breast glands": 0,
        "connective tissue": 0,
        "adipose tissue": 0,
        "undetermined": 255,
    }
    for patient in sorted(dataset.patient_names):
        for selection in sorted(
            (raw_data_root / "ST-spotfiles").glob(f"{patient}*_selection.tsv")
        ):
            name = selection.stem.removesuffix("_selection")
            if force or not all(
                (samples_root / name / f).is_file()
                for f in (
                    "gene_counts.tsv.gz",
                    f"spatial/he_{target_mpp:g}mpp.png",
                    "spatial/canvas_mask.png",
                )
            ):
                process_sample(
                    raw_data_root,
                    samples_root,
                    mask_root,
                    mpp_table.loc[name],
                    target_mpp,
                    classes=CLASSES,
                    spot_label_map=spot_label_map,
                    tile_size=tile_size,
                )
