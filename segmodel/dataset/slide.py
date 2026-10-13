"""共用整片输入和对齐取窗，训练监督由 TrainingSlide 单独保存。"""

from dataclasses import dataclass
from math import lcm

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset


@dataclass
class Slide:
    name: str
    image: NDArray[np.uint8]
    patch_maps: tuple[NDArray[np.float16], ...]
    cls_maps: tuple[NDArray[np.float16], ...]
    cls_scale: int
    label_scale: int

    @property
    def height(self) -> int:
        return self.image.shape[0]

    @property
    def width(self) -> int:
        return self.image.shape[1]

    @property
    def patch_scale(self) -> tuple[int, int]:
        shapes = {patch_map.shape[:2] for patch_map in self.patch_maps}
        if len(shapes) != 1:
            raise ValueError(f"{self.name}: PFM patch grids differ: {sorted(shapes)}")
        rows, columns = shapes.pop()
        return self.height // rows, self.width // columns

    @property
    def alignment(self) -> int:
        return lcm(self.label_scale, self.patch_scale[0])

    @property
    def patch_dims(self) -> tuple[int, ...]:
        return tuple(patch_map.shape[-1] for patch_map in self.patch_maps)

    @property
    def cls_dims(self) -> tuple[int, ...]:
        return tuple(cls_map.shape[-1] for cls_map in self.cls_maps)

    def inputs(
        self,
        top,
        left,
        size,
    ):
        scale_y, scale_x = self.patch_scale
        image = _crop_array(self.image, top, left, size, size, fill=255)
        patch_maps = tuple(
            _crop_array(
                patch_map,
                top // scale_y,
                left // scale_x,
                size // scale_y,
                size // scale_x,
            )
            for patch_map in self.patch_maps
        )

        cls_maps = tuple(
            _crop_array(
                cls_map,
                (top + self.cls_scale // 2) // self.cls_scale,
                (left + self.cls_scale // 2) // self.cls_scale,
                size // self.cls_scale,
                size // self.cls_scale,
            )
            for cls_map in self.cls_maps
        )
        return {
            "image": torch.from_numpy(image).permute(2, 0, 1),
            "patch": tuple(
                torch.from_numpy(patch_map).permute(2, 0, 1)
                for patch_map in patch_maps
            ),
            "cls": tuple(
                torch.from_numpy(cls_map).permute(2, 0, 1) for cls_map in cls_maps
            ),
        }


@dataclass
class TrainingSlide(Slide):
    spot_radius_px: float
    pseudo_mask_uint8: NDArray[np.uint8]
    cancer_probability: NDArray[np.float32]
    superpixels: NDArray[np.int32]
    tissue_mask_bl: NDArray[np.bool_]
    has_segmentation: bool = True


class SlideDataset(Dataset):
    def __init__(self, slides):
        self.slides = list(slides)

    def __len__(self):
        return len(self.slides)

    def __getitem__(self, index):
        return self.slides[index]



def _crop_array(array, top, left, height, width, fill=0):
    """裁剪 (H,W,C) 数组，并在右侧和下侧不足时补齐。"""
    output = np.full(
        (height, width, *array.shape[2:]),
        fill,
        dtype=array.dtype,
    )
    source = array[top : top + height, left : left + width]
    output[: source.shape[0], : source.shape[1]] = source
    return output
