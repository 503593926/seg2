import numpy as np
from numpy.typing import NDArray

def snap_spots_to_tissue(
    spot_x: NDArray[np.int64],
    spot_y: NDArray[np.int64],
    spot_labels: NDArray[np.uint8],
    superpixels: NDArray[np.int32],
    max_distance_px: int,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    # 组织外的已知 spot 移到窗口内最近的前景像素; 距离用整数像素的平方比较, 并列取行主序第一个
    # (动态标签更新里另有 EDT 版本, 两者的边界/并列行为不同, 不要互换)
    height, width = superpixels.shape
    snapped_x = spot_x.copy()
    snapped_y = spot_y.copy()
    outside_known = np.flatnonzero(
        (spot_labels != 255) & ~(superpixels[spot_y, spot_x] > 0)
    )
    for spot_index in outside_known:
        x = int(spot_x[spot_index])
        y = int(spot_y[spot_index])
        x0 = max(0, x - max_distance_px)
        y0 = max(0, y - max_distance_px)
        x1 = min(width, x + max_distance_px + 1)
        y1 = min(height, y + max_distance_px + 1)
        candidate_y, candidate_x = np.where(superpixels[y0:y1, x0:x1] > 0)
        if len(candidate_x) == 0:
            continue

        candidate_x += x0
        candidate_y += y0
        distance_squared = (candidate_x - x) ** 2 + (candidate_y - y) ** 2
        nearest = int(np.argmin(distance_squared))
        if distance_squared[nearest] <= max_distance_px**2:
            snapped_x[spot_index] = candidate_x[nearest]
            snapped_y[spot_index] = candidate_y[nearest]

    return snapped_x, snapped_y

def build_pseudo_mask(
    probability: NDArray[np.floating],
    tissue: NDArray[np.bool_],
    non_cancer_threshold: float,
    cancer_threshold: float,
) -> NDArray[np.uint8]:
    mask = np.full(probability.shape, 255, dtype=np.uint8)
    mask[tissue & (probability < non_cancer_threshold)] = 0
    mask[tissue & (probability > cancer_threshold)] = 1

    return mask

def requested_superpixel_count(
    tissue_pixels: int,
    superpixel_size_um: float,
    mpp: float,
) -> int:
    return max(1, round(tissue_pixels / (superpixel_size_um / mpp) ** 2))
