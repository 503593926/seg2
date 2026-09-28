import math

import numpy as np
from numpy.typing import NDArray

def tissue_bbox_in_image(
    tissue_detection_mask: NDArray[np.uint8],
    image_width: int,
    image_height: int,
    margin_px: int,
) -> tuple[int, int, int, int]:
    y_indices, x_indices = np.nonzero(tissue_detection_mask)
    assert x_indices.size > 0, 'Tissue mask is empty'
    detection_height, detection_width = tissue_detection_mask.shape

    x0 = math.floor(x_indices.min() * image_width / detection_width)
    x1 = math.ceil((x_indices.max() + 1) * image_width / detection_width)
    y0 = math.floor(y_indices.min() * image_height / detection_height)
    y1 = math.ceil((y_indices.max() + 1) * image_height / detection_height)

    return (
        max(0, x0 - margin_px),
        max(0, y0 - margin_px),
        min(image_width, x1 + margin_px),
        min(image_height, y1 + margin_px),
    )
