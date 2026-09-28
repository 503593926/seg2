"""只读整片输入：训练和推理共用同一套对齐取窗。"""

from dataclasses import dataclass

import cv2
import numpy as np
import torch


def read_image(path, flag) -> np.ndarray:
    image = cv2.imread(str(path), flag)
    if image is None:
        raise FileNotFoundError(path)
    return np.asarray(image)


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


def _image_tensor(
    image_rgb,
    stain_transform=None,
    stain_seed=None,
    stain_mask=None,
):
    if stain_transform is not None:
        stain_transform.set_random_seed(stain_seed)
        augmented = stain_transform(image=image_rgb)["image"]

        if stain_mask is not None:
            mask = stain_mask.astype(bool, copy=False)
            np.copyto(
                augmented,
                image_rgb,
                where=~mask[..., None],
            )

        image_rgb = augmented

    return torch.from_numpy(image_rgb).permute(2, 0, 1)


def _pad_cls_map(cls_map, image_shape, cls_scale):
    """cls 网格按边缘补齐到整片尺寸的整数倍，取窗时才能整除。"""
    padding_rows = image_shape[0] // cls_scale - cls_map.shape[0]
    padding_columns = image_shape[1] // cls_scale - cls_map.shape[1]
    return np.pad(
        cls_map,
        ((0, padding_rows), (0, padding_columns), (0, 0)),
        mode="edge",
    )


@dataclass
class Slide:
    """整片输入视图：image 和各 PFM 的特征图均保持 HWC 布局。

    每个 PFM 一份 patch_map（spatial token）与一份 cls_map（tile 级 embedding），
    两者维度可以不同（例如 CONCH 的 patch 与 cls 维度就不一致）。
    """

    name: str
    image: np.ndarray
    patch_maps: tuple
    cls_maps: tuple
    cls_scale: int

    @property
    def height(self):
        return self.image.shape[0]

    @property
    def width(self):
        return self.image.shape[1]

    @property
    def patch_scale(self):
        """特征图一个格子对应多少个目标分辨率像素；各 PFM 必须落在同一张网格。"""
        shapes = {patch_map.shape[:2] for patch_map in self.patch_maps}
        if len(shapes) != 1:
            raise ValueError(f"{self.name}: PFM patch grids differ: {sorted(shapes)}")
        rows, columns = shapes.pop()
        if self.height % rows or self.width % columns:
            raise ValueError(f"{self.name}: image and patch grids are not divisible")
        return self.height // rows, self.width // columns

    @property
    def patch_dims(self):
        return tuple(patch_map.shape[-1] for patch_map in self.patch_maps)

    @property
    def cls_dims(self):
        return tuple(cls_map.shape[-1] for cls_map in self.cls_maps)

    def inputs(
        self,
        top,
        left,
        size,
        alignment,
        stain_transform=None,
        stain_seed=None,
        stain_mask=None,
    ):
        """取一个按 alignment 对齐的窗口：H&E、各 PFM 的 patch 特征与 CLS 上下文。"""
        if (top < 0 or left < 0 or size <= 0
            or any(value % alignment for value in (top, left, size))
        ):
            raise ValueError(
                f"Crop origins must be nonnegative and size must be positive; "
                f"all values must be {alignment}px multiples"
            )

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
        # cls 网格粗(约 112mpp), 窗口起点只按 alignment 对齐, 这里四舍五入到最近的格,
        # 上下文特征最多偏移半格, 对全局上下文分支可接受
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
            "image": _image_tensor(
                image,
                stain_transform,
                stain_seed,
                stain_mask,
            ),
            "patch": tuple(
                torch.from_numpy(patch_map).permute(2, 0, 1)
                for patch_map in patch_maps
            ),
            "cls": tuple(
                torch.from_numpy(cls_map).permute(2, 0, 1) for cls_map in cls_maps
            ),
        }

    @classmethod
    def from_sample(cls, name, slide_data):
        """训练/验证用：复用已经加载在内存里的整片数组。"""
        return cls(
            name,
            slide_data["image"],
            slide_data["patch_map"],
            slide_data["cls_map"],
            slide_data["cls_scale"],
        )

    @classmethod
    def from_disk(cls, name, image_path, feature_dirs, cls_scale):
        image = cv2.cvtColor(
            read_image(image_path, cv2.IMREAD_COLOR),
            cv2.COLOR_BGR2RGB,
        )
        patch_maps = tuple(
            np.load(directory / "patch_map.npy", mmap_mode='r') for directory in feature_dirs
        )
        cls_maps = tuple(
            _pad_cls_map(np.load(directory / "cls_map.npy", mmap_mode='r'), image.shape, cls_scale)
            for directory in feature_dirs
        )
        return cls(name, image, patch_maps, cls_maps, cls_scale)
