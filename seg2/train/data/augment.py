from functools import partial

import albumentations as A
import cv2
import numpy as np
import torch

from seg2.infer.window import window_inputs


def build_stain_transform(augment_cfg):
    if not bool(augment_cfg.enabled):
        return None
    stain_cfg = augment_cfg.stain
    probability = float(stain_cfg.probability)
    if probability <= 0:
        return None
    return A.HEStain(
        method=stain_cfg.method,
        preset=stain_cfg.preset,
        intensity_scale_range=stain_cfg.intensity_scale_range,
        intensity_shift_range=stain_cfg.intensity_shift_range,
        augment_background=bool(stain_cfg.augment_background),
        p=probability,
    )


def get_augmented_inputs(slide, top, left, rng, cfg, crop_size, stain_transform):
    stain_seed = int(rng.integers(0, np.iinfo(np.uint32).max))
    mask_size = crop_size // slide["label_scale"]
    mask_top, mask_left = (top // slide["label_scale"], left // slide["label_scale"])
    tissue_mask = slide["tissue_mask_bl"][
        mask_top : mask_top + mask_size, mask_left : mask_left + mask_size
    ]
    tissue_mask = cv2.resize(
        tissue_mask.astype(np.uint8),
        (crop_size, crop_size),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    sample = window_inputs(slide, top, left, crop_size)
    if cfg.enabled and stain_transform is not None:
        image_rgb = sample["image"].permute(1, 2, 0).numpy()
        stain_transform.set_random_seed(stain_seed)
        augmented = stain_transform(image=image_rgb)["image"]
        stain_mask = tissue_mask if bool(cfg.stain.tissue_only) else None
        if stain_mask is not None:
            mask = stain_mask.astype(bool, copy=False)
            np.copyto(augmented, image_rgb, where=~mask[..., None])
        sample["image"] = torch.from_numpy(augmented).permute(2, 0, 1)
    if cfg.enabled:
        sample["image"] = augment_image(sample["image"], tissue_mask, rng, cfg)
    return sample


def augment_image(image, tissue_mask, rng, cfg):
    transforms = (
        ("brightness", _brightness),
        ("contrast", partial(_contrast, tissue_mask=tissue_mask)),
        ("low_resolution", _low_resolution),
    )
    augmented = None
    for name, transform in transforms:
        transform_cfg = cfg[name]
        probability = float(transform_cfg.probability)
        if probability <= 0 or rng.random() >= probability:
            continue
        if augmented is None:
            augmented = image.permute(1, 2, 0).numpy().astype(np.float32)
        transformed = transform(augmented, rng, transform_cfg)
        tissue_only = bool(transform_cfg.tissue_only)
        if tissue_only:
            np.copyto(transformed, augmented, where=~tissue_mask[..., None])
        augmented = np.clip(transformed, 0, 255)
    if augmented is None:
        return image
    return torch.from_numpy(np.rint(augmented).astype(np.uint8)).permute(2, 0, 1)


def augment_geometry(sample, rng, size, cfg):
    turns = int(rng.choice(cfg.rotation_quarter_turns))
    flip_x = rng.random() < float(cfg.horizontal_flip_probability)
    flip_y = rng.random() < float(cfg.vertical_flip_probability)
    coordinates = sample["spots"]
    for _ in range(turns):
        coordinates = torch.stack(
            (coordinates[:, 1], size - 1 - coordinates[:, 0]), dim=1
        )
    if flip_x:
        coordinates[:, 0] = size - 1 - coordinates[:, 0]
    if flip_y:
        coordinates[:, 1] = size - 1 - coordinates[:, 1]
    sample["spots"] = coordinates
    for key in ("image", "cancer_probability", "valid", "tissue"):
        if key in sample:
            sample[key] = _transform_geometry(sample[key], turns, flip_x, flip_y)
    for key in ("patch", "cls"):
        sample[key] = tuple(
            _transform_geometry(value, turns, flip_x, flip_y) for value in sample[key]
        )
    return sample


def _brightness(image, rng, cfg):
    multiplier = rng.uniform(*cfg.multiplier_range)
    return image * multiplier


def _contrast(image, rng, cfg, *, tissue_mask):
    factor = rng.uniform(*cfg.factor_range)
    values = image[tissue_mask] if bool(cfg.tissue_only) else image.reshape(-1, 3)
    mean = values.mean(axis=0)
    return (image - mean) * factor + mean


def _low_resolution(image, rng, cfg):
    height, width = image.shape[:2]
    scale = rng.uniform(*cfg.scale_range)
    downsampled = cv2.resize(
        image,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return cv2.resize(downsampled, (width, height), interpolation=cv2.INTER_LINEAR)


def _transform_geometry(value, turns, flip_x, flip_y):
    array = np.rot90(value.numpy(), turns, axes=(-2, -1))
    if flip_x:
        array = np.flip(array, axis=-1)
    if flip_y:
        array = np.flip(array, axis=-2)
    array = (
        array.copy(order="C")
        if any(stride < 0 for stride in array.strides)
        else np.ascontiguousarray(array)
    )
    return torch.from_numpy(array)
