import numpy as np
import torch


def window_inputs(sample, top, left, size):
    patch_scale = sample["patch_scale"]
    cls_scale = sample["cls_scale"]
    image = _crop_array(sample["image"], top, left, size, size, fill=255)
    patch_maps = tuple(
        (
            _crop_array(patch_map, top // patch_scale, left // patch_scale, size // patch_scale, size // patch_scale)
            for patch_map in sample["patch_maps"]
        )
    )
    cls_maps = tuple(
        (
            _crop_array(cls_map, top // cls_scale, left // cls_scale, size // cls_scale, size // cls_scale,)
            for cls_map in sample["cls_maps"]
        )
    )
    return {
        "image": torch.from_numpy(image).permute(2, 0, 1),
        "patch": tuple((torch.from_numpy(patch_map).permute(2, 0, 1) for patch_map in patch_maps)),
        "cls": tuple((torch.from_numpy(cls_map).permute(2, 0, 1) for cls_map in cls_maps)),
    }


def _crop_array(array, top, left, height, width, fill=0):
    output = np.full((height, width, *array.shape[2:]), fill, dtype=array.dtype)
    source = array[top : top + height, left : left + width]
    output[: source.shape[0], : source.shape[1]] = source
    return output
