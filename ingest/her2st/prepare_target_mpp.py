import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import cv2
import numpy as np
import pandas as pd
import yaml
from numpy.typing import NDArray

from ingest.her2st.qc_visualize import qc_visualize_target_mpp
from ingest.utils.annotation2mask import ClassInfo
from ingest.utils.detect_tissue import detect_tissue
from ingest.utils.pad_to_tile import pad_to_tile
from ingest.utils.resize_image_to_mpp import resize_image_to_mpp
from ingest.utils.tissue_bbox_in_image import tissue_bbox_in_image


def _write_image(path: Path, image: NDArray[np.uint8], parameters: list[int]) -> None:
    if not cv2.imwrite(str(path), image, parameters):
        raise RuntimeError(f"Cannot write image: {path}")


def resize_discrete_mask(
    path: Path,
    full_width: int,
    full_height: int,
    bbox: tuple[int, int, int, int],
    tile_size: int,
) -> NDArray[np.uint8]:
    """把离散类别 mask 对齐到目标 MPP 网格,裁剪到 bbox 并按 tile 补齐。"""
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    assert mask is not None

    background = 255 if path.name == "training_mask.png" else 0
    mask_target_mpp = cv2.resize(
        mask,
        (full_width, full_height),
        interpolation=cv2.INTER_NEAREST,
    )

    x0, y0, x1, y1 = bbox
    annotated = mask_target_mpp != background
    if annotated.any():
        y, x = np.nonzero(annotated)
        if x.min() < x0 or x.max() >= x1 or y.min() < y0 or y.max() >= y1:
            raise RuntimeError(
                f"{path}: annotated pixels fall outside the H&E-derived crop. "
                "Tune tissue detection instead of expanding the crop from labels."
            )

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
    """合并 spot 标注，统一标签并变换到裁剪后画布坐标系。"""
    spatial = pd.read_csv(selection_path, sep="\t")
    required = {"x", "y", "pixel_x", "pixel_y", "selected"}
    missing = required.difference(spatial.columns)
    if missing:
        raise ValueError(f"{selection_path}: missing columns {sorted(missing)}")

    annotations = pd.read_csv(annotation_path, sep="\t").dropna(subset=["x", "y"])
    annotations.rename(
        columns={"x": "new_x", "y": "new_y", "label": "annotation"},
        inplace=True,
    )
    spatial = spatial.merge(
        annotations[["new_x", "new_y", "annotation"]],
        how="left",
        on=["new_x", "new_y"],
        validate="one_to_one",
    )
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
        columns={"pixel_x": "source_pixel_x", "pixel_y": "source_pixel_y"},
        inplace=True,
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

    selected = spatial["selected"] == 1
    if not spatial.loc[selected, "in_content"].all():
        outside = spatial.loc[selected & ~spatial["in_content"], "spot_id"].tolist()
        raise RuntimeError(f"Selected spots outside crop: {outside[:10]}")

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
    """为一个样本生成 target-MPP 空间数据。"""
    detection_mpp = 2.0
    margin_um = 112
    closing_diameter_um = 112.0

    name = str(mpp_record.name)
    output_dir = samples_root / name
    spatial_dir = output_dir / "spatial"
    masks_dir = output_dir / "masks"
    spatial_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    # 读取 H&E 并把整张图统一到目标 MPP 网格
    image_path = next((data_root / "ST-imgs" / name[0] / name).glob("*.jpg"))
    img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    assert img_bgr is not None

    mpp_x = float(mpp_record["axis_mpp_x"])
    mpp_y = float(mpp_record["axis_mpp_y"])
    scale_x = mpp_x / target_mpp
    scale_y = mpp_y / target_mpp
    image_target_mpp = resize_image_to_mpp(img_bgr, mpp_x, mpp_y, target_mpp)
    full_height, full_width = image_target_mpp.shape[:2]
    del img_bgr

    # 组织检测,并据此确定裁剪框
    tissue_detection = detect_tissue(
        image_target_mpp,
        detection_mpp,
        closing_diameter_um,
        source_mpp=target_mpp,
    )
    bbox = tissue_bbox_in_image(
        tissue_detection,
        full_width,
        full_height,
        round(margin_um / target_mpp),
    )
    x0, y0, x1, y1 = bbox
    crop_width = x1 - x0
    crop_height = y1 - y0

    # 裁剪并按 tile 补齐,同时生成 tissue / canvas 两个 mask
    padded_image = pad_to_tile(image_target_mpp[y0:y1, x0:x1], tile_size)
    canvas_height, canvas_width = padded_image.shape[:2]

    tissue_target_mpp = cv2.resize(
        tissue_detection,
        (full_width, full_height),
        interpolation=cv2.INTER_NEAREST,
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
        spatial_dir / "tissue_mask.png",
        tissue_padded,
        [cv2.IMWRITE_PNG_COMPRESSION, 9],
    )
    _write_image(
        spatial_dir / "canvas_mask.png",
        canvas_mask,
        [cv2.IMWRITE_PNG_COMPRESSION, 9],
    )

    # 病理标注 mask:逐个对齐到目标 MPP、裁剪、补齐
    semantic_mask_target_mpp: NDArray[np.uint8] | None = None
    for source_mask_path in (mask_root / name).glob("*_mask.png"):
        output_mask = resize_discrete_mask(
            source_mask_path,
            full_width,
            full_height,
            bbox,
            tile_size,
        )
        output_mask_path = masks_dir / source_mask_path.name
        _write_image(output_mask_path, output_mask, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        if source_mask_path.name == "semantic_mask.png":
            semantic_mask_target_mpp = output_mask

    # spot 坐标变换
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
        spatial_dir / f"spatial_{target_mpp:g}mpp.tsv",
        sep="\t",
        index=False,
    )
    # process 阶段只读取 data 目录；counts 保持统一的 spot × gene TSV 格式。
    shutil.copyfile(count_path, output_dir / "gene_counts.tsv.gz")

    # QC 图
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


def main() -> None:
    raw_data_root = Path("/home/yfm/data/bioinfo/her2st-master/data")
    names = ["A1", "B1", "C1", "D1", "E1", "F1", "G2", "H1"]

    classes: list[ClassInfo] = [
        {"id": 1, "label": "invasive cancer", "rgb": (236, 28, 36)},
        {"id": 2, "label": "cancer in situ", "rgb": (255, 127, 39)},
        {"id": 3, "label": "immune infiltrate", "rgb": (255, 242, 0)},
        {"id": 4, "label": "breast glands", "rgb": (14, 209, 69)},
        {"id": 5, "label": "connective tissue", "rgb": (63, 72, 204)},
        {"id": 6, "label": "adipose tissue", "rgb": (112, 227, 223)},
    ]
    spot_label_map = {
        "invasive cancer": 1,
        "cancer in situ": 1,
        "immune infiltrate": 0,
        "breast glands": 0,
        "connective tissue": 0,
        "adipose tissue": 0,
        "undetermined": 255,
    }

    project_root = Path(__file__).resolve().parents[2]
    data_root = project_root / "data/her2st"
    mask_root = data_root / "original_mask/"
    with open(str(project_root / "config/resolution.yaml"), "r", encoding="utf-8") as f:
        resolution = cast(dict[str, float], yaml.safe_load(f))
    target_mpp: float = resolution["target_mpp"]
    tile_size: int = int(resolution["tile_size"])

    samples_root = data_root / f"{target_mpp:g}mpp/samples"
    samples_root.mkdir(parents=True, exist_ok=True)

    mpp_table = pd.read_csv(data_root / "mpp.csv", index_col=0)
    for name in names:
        print(f"[{name}] preparing {target_mpp}-MPP spatial data", flush=True)
        process_sample(
            raw_data_root,
            samples_root,
            mask_root,
            cast(pd.Series, mpp_table.loc[name]),
            target_mpp,
            classes=classes,
            spot_label_map=spot_label_map,
            tile_size=tile_size,
        )


if __name__ == "__main__":
    main()
