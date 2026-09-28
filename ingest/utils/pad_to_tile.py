from collections.abc import Sequence
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

def pad_to_tile(
    image: NDArray[np.uint8],
    tile_size: int,
    background: Sequence[float] = (255, 255, 255),
) -> NDArray[np.uint8]:
    height, width = image.shape[:2]

    pad_bottom = (-height) % tile_size
    pad_right = (-width) % tile_size

    return cast(
        NDArray[np.uint8],
        cv2.copyMakeBorder(
            image,
            0,
            pad_bottom,
            0,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=background,
        ),
    )
