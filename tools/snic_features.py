import ctypes
from pathlib import Path
from typing import cast

import cv2
import numpy as np
import skimage
from numpy.typing import NDArray

SNIC_DIR = Path(__file__).resolve().parents[1] / 'SNIC' / 'snic_python_interface'
_SNIC_LIBRARY_PATH = SNIC_DIR / 'libsnic.so'

if not _SNIC_LIBRARY_PATH.is_file():
    raise FileNotFoundError(
        f'{_SNIC_LIBRARY_PATH} is missing; build it with:\n'
        'gcc -O3 -march=native -ffast-math -fPIC -shared '
        'SNIC/snic_python_interface/snic.c -I SNIC/snic_python_interface '
        '-lm -o SNIC/snic_python_interface/libsnic.so'
    )

_snic_library = ctypes.CDLL(str(_SNIC_LIBRARY_PATH))

_snic_main = _snic_library.SNIC_main
_snic_main.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_double,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
]
_snic_main.restype = None

_rgb_to_lab = _snic_library.rgbtolab
_rgb_to_lab.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_double,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
]
_rgb_to_lab.restype = None


def prepare_snic_lab(image_bgr: NDArray[np.uint8]) -> NDArray[np.float64]:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    planar_lab = cast(
        NDArray[np.float64],
        np.ascontiguousarray(image_rgb.transpose(2, 0, 1), dtype=np.float64),
    )
    channel_pointers = [
        planar_lab[channel].ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        for channel in range(3)
    ]
    _rgb_to_lab(
        channel_pointers[0],
        channel_pointers[1],
        channel_pointers[2],
        float(image_bgr.shape[0] * image_bgr.shape[1]),
        channel_pointers[0],
        channel_pointers[1],
        channel_pointers[2],
    )

    return planar_lab.transpose(1, 2, 0)


def run_snic(
    features: NDArray[np.generic],
    mask: NDArray[np.bool_],
    n_segments: int,
    *,
    compactness: float,
) -> tuple[NDArray[np.int32], dict[str, int]]:
    rows = np.flatnonzero(np.any(mask, axis=1))
    columns = np.flatnonzero(np.any(mask, axis=0))
    y0, y1 = int(rows[0]), int(rows[-1]) + 1
    x0, x1 = int(columns[0]), int(columns[-1]) + 1
    mask_crop = mask[y0:y1, x0:x1]
    feature_crop = features[y0:y1, x0:x1]
    height, width, channels = feature_crop.shape

    official_requested = min(
        mask_crop.size,
        max(1, round(max(1, n_segments) / float(mask_crop.mean()))),
    )
    planar_features = np.ascontiguousarray(
        np.moveaxis(feature_crop, -1, 0),
        dtype=np.float64,
    )
    # 官方接口没有 mask; 把背景特征顶到极大值, 阻止背景种子进入组织
    planar_features[:, ~mask_crop] = 1_000_000.0
    raw_labels = np.empty((height, width), dtype=np.int32)
    official_actual = ctypes.c_int(0)

    _snic_main(
        planar_features.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        width,
        height,
        channels,
        official_requested,
        float(compactness),
        0,
        raw_labels.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        ctypes.byref(official_actual),
    )

    # SNIC 标签从 0 起, 统一成 1..K, 背景置 0
    raw_labels += 1
    raw_labels[~mask_crop] = 0

    labels_crop = skimage.measure.label(
        raw_labels,
        background=0,
        connectivity=1,
    ).astype(np.int32, copy=False)

    labels = cast(NDArray[np.int32], np.zeros(mask.shape, dtype=np.int32))
    labels[y0:y1, x0:x1] = labels_crop

    return labels, {
        'official_requested_superpixels': official_requested,
        'official_actual_superpixels': int(official_actual.value),
    }
