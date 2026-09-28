import numpy as np
from numpy.typing import NDArray
from typing import TypedDict
import cv2
import scipy

class ClassInfo(TypedDict):
    id: int
    label: str
    rgb: tuple[int, int, int]

class _RegionInfo(TypedDict):
    class_id: int
    area: int
    bbox: tuple[slice, slice]
    mask: NDArray[np.bool_]

def get_sem_mask(
    ann_bgr: NDArray[np.uint8],
    classes: list[ClassInfo],
) -> NDArray[np.uint8]:

    lines = _get_lines(ann_bgr, classes)
    regions = _get_regions(lines, classes)

    semantic_mask: NDArray[np.uint8] = np.zeros_like(next(iter(lines.values())), dtype=np.uint8)
    for region in sorted(regions, key=lambda item: item['area'], reverse=True):
        roi = semantic_mask[region['bbox']]
        roi[region['mask']] = region['class_id']

    for info in classes:
        semantic_mask[lines[info['id']] > 0] = info['id']

    return semantic_mask


def _get_lines(
    ann_bgr: NDArray[np.uint8],
    classes: list[ClassInfo],
    kernel_size: int = 11,
) -> dict[int, NDArray[np.uint8]]:
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    lines: dict[int, NDArray[np.uint8]] = {}

    for info in classes:
        r, g, b = info['rgb']

        line_mask_ui8 = cv2.inRange(
            ann_bgr,
            np.array((b, g, r), dtype=np.uint8),
            np.array((b, g, r), dtype=np.uint8),
        )

        line_mask_ui8 = cv2.morphologyEx(
            line_mask_ui8,
            cv2.MORPH_CLOSE,
            kernel,
        )

        lines[info['id']] = line_mask_ui8

    return lines


def _get_regions(
    lines: dict[int, NDArray[np.uint8]],
    classes: list[ClassInfo],
    area_threshold: int = 800,
) -> list[_RegionInfo]:
    regions: list[_RegionInfo] = []

    for info in classes:
        class_id = info['id']

        line_bl = lines[class_id] > 0
        interior = scipy.ndimage.binary_fill_holes(line_bl) & ~line_bl

        count, component_map, stats, _ = cv2.connectedComponentsWithStats(
            interior.astype(np.uint8),
            connectivity=8,
        )

        for component_id in range(1, count):
            area = int(stats[component_id, cv2.CC_STAT_AREA])
            if area < area_threshold:
                continue
            x = int(stats[component_id, cv2.CC_STAT_LEFT])
            y = int(stats[component_id, cv2.CC_STAT_TOP])
            w = int(stats[component_id, cv2.CC_STAT_WIDTH])
            h = int(stats[component_id, cv2.CC_STAT_HEIGHT])
            regions.append({
                'class_id': class_id,
                'area': area,
                'bbox': (slice(y, y + h), slice(x, x + w)),
                'mask': component_map[y:y + h, x:x + w] == component_id,
            })

    return regions
