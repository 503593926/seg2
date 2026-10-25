import cv2
import numpy as np
import scipy
from numpy.typing import NDArray


def _odd_kernel_size(diameter_um: float, mpp: float) -> int:
    size = max(1, round(diameter_um / mpp))
    return size if size % 2 == 1 else size + 1


def detect_tissue(
    image_bgr: NDArray[np.uint8],
    detection_mpp: float,
    closing_diameter_um: float,
    source_mpp: float = 1.0,
    weak_threshold_ratio: float = 0.85,
    min_tissue_area_um2: float = 100000.0,
) -> NDArray[np.uint8]:
    source_height, source_width = image_bgr.shape[:2]
    detection_width = round(source_width * source_mpp / detection_mpp)
    detection_height = round(source_height * source_mpp / detection_mpp)
    thumbnail = cv2.resize(image_bgr, (detection_width, detection_height), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2HSV)
    gray_threshold, dark_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, saturated_mask = cv2.threshold(hsv[:, :, 1], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    strong_mask = (dark_mask > 0) | (saturated_mask > 0)
    weak_mask = (gray <= 255.0 - weak_threshold_ratio * (255.0 - gray_threshold)) | (saturated_mask > 0)
    raw_mask = (
        scipy.ndimage.binary_propagation(strong_mask, structure=np.ones((3, 3), dtype=bool), mask=weak_mask)
        * 255
    ).astype(np.uint8)
    closing_kernel_size = _odd_kernel_size(closing_diameter_um, detection_mpp)
    opening_kernel_size = _odd_kernel_size(max(16.0, closing_diameter_um / 4.0), detection_mpp)
    closing_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (closing_kernel_size, closing_kernel_size))
    opening_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (opening_kernel_size, opening_kernel_size))
    cleaned = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, closing_kernel)
    cleaned = (scipy.ndimage.binary_fill_holes(cleaned > 0) * 255).astype(np.uint8)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, opening_kernel)
    _, labels, stats, _ = cv2.connectedComponentsWithStats((cleaned > 0).astype(np.uint8), connectivity=8)
    labels = labels
    keep = stats[:, cv2.CC_STAT_AREA] * detection_mpp**2 >= min_tissue_area_um2
    keep[0] = False
    kept: NDArray[np.uint8] = (scipy.ndimage.binary_fill_holes(keep[labels]) * 255).astype(np.uint8)
    return kept
