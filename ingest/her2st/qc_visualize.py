from collections.abc import Sequence
from pathlib import Path
from typing import cast
import cv2
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from ingest.utils.annotation2mask import ClassInfo


def qc_visualize_mask(
    semantic_mask: NDArray[np.uint8],
    name: str,
    classes: list[ClassInfo],
    img_path: Path,
    raw_data_root: Path,
    save_root: Path,
) -> Path:
    output_dir = save_root / name
    output_dir.mkdir(parents=True, exist_ok=True)

    qc_classes: list[ClassInfo] = [*classes, {'id': 0, 'label': 'undetermined spot', 'rgb': (128, 128, 128)}]

    # overlay: H&E 底图 + 按类别染色
    overlay_mat = cv2.imread(img_path, cv2.IMREAD_COLOR)
    assert overlay_mat is not None, 'Raw image is not exist'
    overlay = np.asarray(overlay_mat, dtype=np.uint8)

    mask_mat = cv2.resize(
        semantic_mask,
        (overlay.shape[1], overlay.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    mask = np.asarray(mask_mat, dtype=np.uint8)

    tint_alpha = 0.34
    for info in classes:
        sel = mask == info['id']
        if not sel.any():
            continue
        color = np.asarray(info['rgb'][::-1], dtype=np.float32)  # BGR
        overlay[sel] = ((1 - tint_alpha) * overlay[sel].astype(np.float32)
                        + tint_alpha * color).astype(np.uint8)

    # legend: 左上角淡入图例,版面尺寸由文字度量自适应推算
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 5
    font_size, thickness = 0.8 * scale, 2 * scale
    pad = 14 * scale  # 外缘留白,兼作行间距与色块-文字间距

    items = [(f'{info["id"]}: {info["label"]}', info['rgb'][::-1]) for info in qc_classes]
    metrics = [cv2.getTextSize(text, font, font_size, thickness) for text, _ in items]
    text_w = max(m[0][0] for m in metrics)
    text_h = max(m[0][1] for m in metrics)
    baseline = max(m[1] for m in metrics)

    swatch = text_h  # 方形色块,与文字等高
    row_h = text_h + pad
    height = 2 * pad + len(items) * row_h
    width = 2 * pad + swatch + pad + text_w

    legend = np.full((height, width, 3), 255, dtype=np.uint8)
    for i, (text, bgr) in enumerate(items):
        top = pad + i * row_h + (row_h - text_h) // 2  # 行内垂直居中
        cv2.rectangle(legend, (pad, top), (pad + swatch, top + swatch), bgr, -1)
        cv2.putText(legend, text, (pad + swatch + pad, top + text_h - baseline),
                    font, font_size, (15, 15, 15), thickness, cv2.LINE_AA)

    legend_alpha = 0.82
    roi = overlay[:height, :width]
    cv2.addWeighted(legend, legend_alpha, roi, 1 - legend_alpha, 0, dst=roi)

    # spots: 合并标注坐标与 spot 选点表,按真实类别画圆点,预测错误再叠白色叉号
    labels = pd.read_csv(
        raw_data_root / 'ST-pat' / 'lbl' / f'{name}_labeled_coordinates.tsv',
        sep='\t',
    ).dropna(subset=['x', 'y'])
    selection = pd.read_csv(
        raw_data_root / 'ST-spotfiles' / f'{name}_selection.tsv',
        sep='\t',
    )
    rows = labels.merge(
        selection,
        how='left',
        left_on=['x', 'y'],
        right_on=['new_x', 'new_y'],
        suffixes=('_label', '_he'),
        validate='one_to_one',
    )
    assert not rows[['pixel_x_he', 'pixel_y_he']].isna().any().any(), \
        f'{name}: some spots cannot be mapped'

    label_to_class = {info['label']: info for info in qc_classes}
    label_to_class['undetermined'] = qc_classes[-1]
    color_of = {info['id']: info['rgb'][::-1] for info in qc_classes}

    radius = 25
    for row in rows.itertuples(index=False):
        x = int(np.clip(round(cast(float, row.pixel_x_he)), 0, mask.shape[1] - 1))
        y = int(np.clip(round(cast(float, row.pixel_y_he)), 0, mask.shape[0] - 1))
        true_id = label_to_class[cast(str, row.label)]['id']
        cv2.circle(overlay, (x, y), radius + 2, (20, 20, 20), -1, cv2.LINE_AA)
        cv2.circle(overlay, (x, y), radius, color_of[true_id], -1, cv2.LINE_AA)
        if true_id != int(mask[y, x]):
            r = radius + 6
            cv2.line(overlay, (x - r, y - r), (x + r, y + r), (255, 255, 255), 4, cv2.LINE_AA)
            cv2.line(overlay, (x - r, y + r), (x + r, y - r), (255, 255, 255), 4, cv2.LINE_AA)

    # thumbnail
    thumb = cv2.resize(overlay, None, fx=0.1, fy=0.1, interpolation=cv2.INTER_AREA)
    qc_path = output_dir / 'spot_qc_overlay.jpg'
    _write_image(qc_path, thumb, [cv2.IMWRITE_JPEG_QUALITY, 95])

    return qc_path


def qc_visualize_target_mpp(
    output_dir: Path,
    image_target_mpp: NDArray[np.uint8],
    tissue_target_mpp: NDArray[np.uint8],
    padded_image: NDArray[np.uint8],
    tissue_padded: NDArray[np.uint8],
    semantic_mask: NDArray[np.uint8] | None,
    spots: pd.DataFrame,
    bbox: tuple[int, int, int, int],
    classes: Sequence[ClassInfo],
) -> None:
    max_side = 1600
    spot_radius = 6
    spatial_dir = output_dir / 'spatial'
    selected = spots.loc[spots['selected'] == 1]
    x0, y0, x1, y1 = bbox

    # qc_crop_overview: 整张 H&E 概览, 绿=组织轮廓, 黄=裁剪框, 红=selected spot
    overview_height, overview_width = image_target_mpp.shape[:2]
    overview_scale = min(1.0, max_side / max(overview_height, overview_width))
    overview_size = (
        max(1, round(overview_width * overview_scale)),
        max(1, round(overview_height * overview_scale)),
    )
    overview = cv2.resize(image_target_mpp, overview_size, interpolation=cv2.INTER_AREA)
    overview_tissue = cv2.resize(
        tissue_target_mpp,
        overview_size,
        interpolation=cv2.INTER_NEAREST,
    )
    overview_contours, _ = cv2.findContours(
        overview_tissue,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(overview, overview_contours, -1, (0, 180, 0), 2)
    cv2.rectangle(
        overview,
        (round(x0 * overview_scale), round(y0 * overview_scale)),
        (round((x1 - 1) * overview_scale), round((y1 - 1) * overview_scale)),
        (0, 255, 255),
        2,
    )
    for pixel_x, pixel_y in selected[['pixel_x', 'pixel_y']].to_numpy(dtype=float):
        spot_x = round((pixel_x + x0) * overview_scale)
        spot_y = round((pixel_y + y0) * overview_scale)
        cv2.circle(overview, (spot_x, spot_y), spot_radius + 2, (20, 20, 20), -1, cv2.LINE_AA)
        cv2.circle(overview, (spot_x, spot_y), spot_radius, (0, 0, 255), -1, cv2.LINE_AA)
    _write_image(
        spatial_dir / 'qc_crop_overview.jpg',
        overview,
        [cv2.IMWRITE_JPEG_QUALITY, 92],
    )

    # qc_thumbnail: 裁剪补齐后的画布, 绿=组织轮廓, 红=selected spot
    canvas_height, canvas_width = padded_image.shape[:2]
    canvas_scale = min(1.0, max_side / max(canvas_height, canvas_width))
    canvas_size = (
        max(1, round(canvas_width * canvas_scale)),
        max(1, round(canvas_height * canvas_scale)),
    )
    thumbnail = cv2.resize(padded_image, canvas_size, interpolation=cv2.INTER_AREA)
    thumbnail_tissue = cv2.resize(
        tissue_padded,
        canvas_size,
        interpolation=cv2.INTER_NEAREST,
    )
    thumbnail_contours, _ = cv2.findContours(
        thumbnail_tissue,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(thumbnail, thumbnail_contours, -1, (0, 180, 0), 2)
    for pixel_x, pixel_y in selected[['pixel_x', 'pixel_y']].to_numpy(dtype=float):
        spot_x = round(pixel_x * canvas_scale)
        spot_y = round(pixel_y * canvas_scale)
        cv2.circle(thumbnail, (spot_x, spot_y), spot_radius + 2, (20, 20, 20), -1, cv2.LINE_AA)
        cv2.circle(thumbnail, (spot_x, spot_y), spot_radius, (0, 0, 255), -1, cv2.LINE_AA)
    _write_image(
        spatial_dir / 'qc_thumbnail.jpg',
        thumbnail,
        [cv2.IMWRITE_JPEG_QUALITY, 92],
    )

    if semantic_mask is None:
        return

    # spot_qc_overlay: 画布 + 类别染色 + 图例 + 按所在类别着色的 spot
    overlay = cv2.resize(padded_image, canvas_size, interpolation=cv2.INTER_AREA)
    semantic_small = cv2.resize(
        semantic_mask,
        canvas_size,
        interpolation=cv2.INTER_NEAREST,
    )
    tint_alpha = 0.34
    for info in classes:
        class_pixels = semantic_small == info['id']
        if not class_pixels.any():
            continue
        color = np.asarray(info['rgb'][::-1], dtype=np.float32)  # BGR
        overlay[class_pixels] = np.rint(
            overlay[class_pixels].astype(np.float32) * (1 - tint_alpha) + color * tint_alpha
        ).astype(np.uint8)

    # 每个 spot 按它所在位置的语义类别着色, 未标注处灰色
    color_of = {info['id']: info['rgb'][::-1] for info in classes}
    color_of[0] = (128, 128, 128)
    mask_height, mask_width = semantic_mask.shape
    content_spots = spots.loc[spots['in_content'], ['pixel_x', 'pixel_y']].to_numpy(dtype=float)
    for pixel_x, pixel_y in content_spots:
        row = min(round(pixel_y), mask_height - 1)
        col = min(round(pixel_x), mask_width - 1)
        spot_color = color_of.get(int(semantic_mask[row, col]), (128, 128, 128))
        spot_x = round(pixel_x * canvas_scale)
        spot_y = round(pixel_y * canvas_scale)
        cv2.circle(overlay, (spot_x, spot_y), spot_radius + 2, (20, 20, 20), -1, cv2.LINE_AA)
        cv2.circle(overlay, (spot_x, spot_y), spot_radius, spot_color, -1, cv2.LINE_AA)

    # 清理旧命名的 QC 图
    (output_dir / 'masks' / 'qc_semantic_overlay.png').unlink(missing_ok=True)
    _write_image(
        output_dir / 'masks' / 'spot_qc_overlay.jpg',
        overlay,
        [cv2.IMWRITE_JPEG_QUALITY, 92],
    )


def qc_visualize_point_mask(
    sample_dir: Path,
    percent: int,
    image_bgr: NDArray[np.uint8],
    spots: pd.DataFrame,
    flags: NDArray[np.bool_],
) -> Path:
    max_side = 1600
    spot_radius = 4
    height, width = image_bgr.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    thumbnail = cv2.resize(image_bgr, size, interpolation=cv2.INTER_AREA)

    # 点只有 1 个像素, QC 放大成圆点: 红=选中的肿瘤 spot, 绿=选中的非肿瘤 spot, 灰=未选中
    color_of = {0: (0, 180, 0), 1: (0, 0, 255)}
    spot_x = np.rint(spots['pixel_x'].to_numpy(dtype=np.float64)).astype(np.int64)
    spot_y = np.rint(spots['pixel_y'].to_numpy(dtype=np.float64)).astype(np.int64)
    spot_labels = spots['spot_label'].to_numpy(dtype=np.uint8)
    for x, y, label, supervised in zip(spot_x, spot_y, spot_labels, flags):
        point = (round(float(x) * scale), round(float(y) * scale))
        if supervised:
            cv2.circle(thumbnail, point, spot_radius, color_of[int(label)], -1, cv2.LINE_AA)
        else:
            cv2.circle(thumbnail, point, spot_radius, (150, 150, 150), 1, cv2.LINE_AA)

    # 左上角标题, 先描深色边再写白字, 免得压在组织上看不清
    font = cv2.FONT_HERSHEY_SIMPLEX
    caption = f'ratio={percent}% supervised={int(flags.sum())}/{len(flags)}'
    cv2.putText(thumbnail, caption, (12, 34), font, 1.0, (20, 20, 20), 3, cv2.LINE_AA)
    cv2.putText(thumbnail, caption, (12, 34), font, 1.0, (255, 255, 255), 1, cv2.LINE_AA)

    qc_path = sample_dir / 'masks' / f'qc_point_mask_{percent}.jpg'
    _write_image(qc_path, thumbnail, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return qc_path


def _write_image(path: Path, image: NDArray[np.uint8], parameters: list[int]) -> None:
    if not cv2.imwrite(str(path), image, parameters):
        raise RuntimeError(f'Cannot write image: {path}')
