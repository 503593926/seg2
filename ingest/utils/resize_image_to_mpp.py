import cv2
import numpy as np
from numpy.typing import NDArray

def resize_image_to_mpp(
    img_bgr: NDArray[np.uint8],
    mpp_x: float,
    mpp_y: float,
    target_mpp: float,
) -> NDArray[np.uint8]:
    source_height, source_width = img_bgr.shape[:2]

    scale_x = mpp_x / target_mpp
    scale_y = mpp_y / target_mpp

    full_width = round(source_width * scale_x)
    full_height = round(source_height * scale_y)
    interpolation = cv2.INTER_AREA if scale_x <= 1 and scale_y <= 1 else cv2.INTER_CUBIC
    image_target_mpp = cv2.resize(
        img_bgr,
        (full_width, full_height),
        interpolation=interpolation,
    )

    return image_target_mpp
