"""将预加载的整张切片数据转换为训练 crop。"""

import os

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, default_collate

from .slide import Slide


def _build_stain_transform(augment_cfg):
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


def _augment_image(image, tissue_mask, rng, cfg):
    """只改变 H&E 明暗、对比度和清晰度，保持图像尺寸与其他监督不变。"""
    augmented = None
    for name in ("brightness", "contrast", "low_resolution"):
        transform_cfg = cfg[name]
        probability = float(transform_cfg.probability)
        if probability <= 0 or rng.random() >= probability:
            continue
        tissue_only = bool(transform_cfg.tissue_only)
        if tissue_only and not tissue_mask.any():
            continue
        if augmented is None:
            augmented = image.permute(1, 2, 0).numpy().astype(np.float32)

        if name == "brightness":
            multiplier = rng.uniform(*transform_cfg.multiplier_range)
            transformed = augmented * multiplier
        elif name == "contrast":
            factor = rng.uniform(*transform_cfg.factor_range)
            values = augmented[tissue_mask] if tissue_only else augmented.reshape(-1, 3)
            mean = values.mean(axis=0)
            transformed = (augmented - mean) * factor + mean
        else:
            height, width = augmented.shape[:2]
            scale = rng.uniform(*transform_cfg.scale_range)
            downsampled = cv2.resize(
                augmented,
                (max(1, round(width * scale)), max(1, round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
            transformed = cv2.resize(
                downsampled, (width, height), interpolation=cv2.INTER_LINEAR
            )

        if tissue_only:
            np.copyto(transformed, augmented, where=~tissue_mask[..., None])
        augmented = np.clip(transformed, 0, 255)

    if augmented is None:
        return image
    return torch.from_numpy(np.rint(augmented).astype(np.uint8)).permute(2, 0, 1)


def _transform_geometry(value, turns, flip_x, flip_y):
    value = torch.rot90(value, turns, (-2, -1))
    if flip_x:
        value = value.flip(-1)
    if flip_y:
        value = value.flip(-2)
    return value.contiguous()


def _augment_geometry(sample, rng, size, cfg):
    """对图像、各 PFM 特征图、监督图和 spot 坐标执行同一组几何增强。"""
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

    for key in ("image", "cancer_probability", "valid", "superpixels"):
        sample[key] = _transform_geometry(sample[key], turns, flip_x, flip_y)
    for key in ("patch", "cls"):
        sample[key] = tuple(
            _transform_geometry(value, turns, flip_x, flip_y)
            for value in sample[key]
        )
    return sample


class SegmentationDataset(Dataset):
    def __init__(
        self,
        slide_data,
        gene_adata,
        slide_names,
        cfg,
        label_scale,
        alignment,
        spot_radius_px,
        seed=0,
    ):
        self.slide_data = slide_data
        self.gene_expression = torch.from_numpy(gene_adata.X)
        self.spot_labels = torch.from_numpy(gene_adata.obs["spot_label"].to_numpy(dtype=np.int64))
        self.slide_names = list(slide_names)
        self.seed = int(seed)
        self.epoch = 0

        obs_names = gene_adata.obs["name"].astype(str).to_numpy()
        self.spots_by_slide = {}
        for name in self.slide_names:
            indices = np.flatnonzero(obs_names == name)
            coordinates = gene_adata.obs.iloc[indices][["pixel_x", "pixel_y"]].to_numpy(
                dtype=np.float32
            )
            self.spots_by_slide[name] = {
                "coords": coordinates,
                "indices": indices.astype(np.int64),
            }

        self.dataset_cfg = cfg.experiment.dataset
        self.augment_cfg = cfg.experiment.augment
        self.stain_transform = _build_stain_transform(self.augment_cfg)
        self.crop_size = int(self.dataset_cfg.crop_size)

        first_slide = next(iter(self.slide_data.values()))
        self.pfm_count = len(first_slide["patch_map"])

        self.label_scale = int(label_scale)
        self.alignment = int(alignment)
        self.spot_radius_px = float(spot_radius_px)

        self.slides = {
            name: Slide.from_sample(name, self.slide_data[name])
            for name in self.slide_names
        }
        if self.crop_size % self.alignment:
            raise ValueError(f"crop_size must be divisible by {self.alignment}")
        self.label_crop_size = self.crop_size // self.label_scale
        self.label_step = self.alignment // self.label_scale
        self.crop_origins = {}

        self.refresh()

    def _refresh_slide(self, name):
        """由当前伪标签重算一张切片的派生监督和候选训练窗口。"""
        slide_data = self.slide_data[name]
        valid = slide_data["pseudo_mask"] != 255
        slide_data["valid"] = valid
        self.crop_origins[name] = self.candidate_origins(valid)

    def candidate_origins(self, valid):
        """返回满足有效像素比例且对齐图像、标签和特征图的窗口起点。"""
        # 标签网格积分图上按候选步长抽样：四角相减即窗口内有效像素数。
        size, step = self.label_crop_size, self.label_step
        integral = cv2.integral(valid.view(np.uint8), sdepth=cv2.CV_64F)
        area = (
            integral[size::step, size::step]
            - integral[:-size:step, size::step]
            - integral[size::step, :-size:step]
            + integral[:-size:step, :-size:step]
        )
        keep = area >= self.dataset_cfg.min_valid_fraction * size * size
        y, x = np.nonzero(keep)
        return np.stack((y * step, x * step), axis=1) * self.label_scale

    def refresh(self):
        """重算所有训练切片的派生监督和候选训练窗口。"""
        for name in self.slide_names:
            self._refresh_slide(name)

    def update_supervision(
        self,
        name,
        pseudo_mask,
        cancer_probability,
        superpixels,
    ):
        """替换一张训练切片的动态监督，并只刷新该切片的训练窗口。"""
        slide_data = self.slide_data[name]
        slide_data["pseudo_mask"] = pseudo_mask
        slide_data["cancer_probability"] = cancer_probability
        slide_data["superpixels"] = superpixels
        self._refresh_slide(name)

    def __len__(self):
        return int(self.dataset_cfg.samples_per_epoch)

    def _get_inputs(self, name, top, left, rng):
        tissue_mask = None
        stain_seed = int(rng.integers(0, np.iinfo(np.uint32).max))
        needs_tissue_mask = (
            self.stain_transform is not None and bool(self.augment_cfg.stain.tissue_only)
        ) or (
            bool(self.augment_cfg.enabled)
            and any(
                float(self.augment_cfg[name].probability) > 0
                and bool(self.augment_cfg[name].tissue_only)
                for name in ("brightness", "contrast", "low_resolution")
            )
        )
        if needs_tissue_mask:
            mask_size = self.crop_size // self.label_scale
            mask_top, mask_left = top // self.label_scale, left // self.label_scale
            tissue_mask = self.slide_data[name]["tissue_mask_bl"][
                mask_top : mask_top + mask_size,
                mask_left : mask_left + mask_size,
            ]
            tissue_mask = cv2.resize(
                tissue_mask.astype(np.uint8),
                (self.crop_size, self.crop_size),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        sample = self.slides[name].inputs(
            top,
            left,
            self.crop_size,
            self.alignment,
            stain_transform=self.stain_transform,
            stain_seed=stain_seed,
            stain_mask=tissue_mask if bool(self.augment_cfg.stain.tissue_only) else None,
        )
        if self.augment_cfg.enabled:
            sample["image"] = _augment_image(
                sample["image"], tissue_mask, rng, self.augment_cfg
            )
        return sample

    def __getitem__(self, index):
        # sample = {                              # C=crop_size, S=patch_scale, L=label_scale
        #     "image": (3, C, C),                 # uint8
        #     "patch": ((D_i, C // S, C // S),),  # float16, 每个 PFM 一份 spatial token
        #     "cls": ((D_i, C // 112, C // 112),),  # float16, 每个 PFM 一份 tile/2 网格 CLS
        #     "pfm_mask": (K,),                   # bool, True 表示该 PFM 这步可用
        #     "cancer_probability": (1, C // L, C // L),  # float32, 监督在 7mpp 网格
        #     "valid": (1, C // L, C // L),       # bool
        #     "superpixels": (1, C // L, C // L), # int64
        #     "spots": (k, 2),                    # float32，以下三项仅用于蒸馏
        #     "gene_expression": (k, G),          # float32
        #     "spot_labels": (k,),                # int64
        # }
        rng = np.random.default_rng(
            np.random.SeedSequence([self.seed, self.epoch, index])
        )
        name, top, left = self._sample_origin(rng)
        sample = self._get_inputs(name, top, left, rng)
        sample.update(self._supervision(name, top, left))
        sample.update(self._spot_targets(name, top, left))
        if self.augment_cfg.enabled:
            sample = _augment_geometry(
                sample, rng, self.crop_size, self.augment_cfg.geometry
            )
        # 放在几何增强之后取随机数：p=0 时与加这个机制之前的增强序列完全一致
        sample["pfm_mask"] = self._pfm_mask(rng)
        return sample

    def _pfm_mask(self, rng):
        """按配置概率与数量范围屏蔽 PFM；max_count 为 None 时最多屏蔽 K-1 个。

        概率为 0 时也返回全 True 的掩码，下游就不必判断该字段是否存在。
        被屏蔽 PFM 的输入特征不在这里清零——那属于模型的语义，dataset 只负责
        声明"这一步哪些 expert 可用"。
        """
        cfg = self.augment_cfg.pfm_mask
        available = torch.ones(self.pfm_count, dtype=torch.bool)
        if float(cfg.probability) and rng.random() < float(cfg.probability):
            max_count = self.pfm_count - 1 if cfg.max_count is None else int(cfg.max_count)
            count = int(rng.integers(int(cfg.min_count), max_count + 1))
            masked = rng.choice(self.pfm_count, size=count, replace=False)
            available[masked] = False
        return available

    def _sample_origin(self, rng):
        """选一张切片和一个候选窗口，一部分靠近真实 spot 以保证分子监督有样本。"""
        name = self.slide_names[int(rng.integers(len(self.slide_names)))]
        origins = self.crop_origins[name]
        index = int(rng.integers(len(origins)))
        spot_data = self.spots_by_slide[name]

        fraction = float(self.augment_cfg.sampling.spot_sampling_fraction)
        if len(spot_data["coords"]) and rng.random() < fraction:
            center = spot_data["coords"][int(rng.integers(len(spot_data["coords"])))]
            desired = center[::-1] - self.crop_size / 2
            index = int(np.argmin(((origins - desired) ** 2).sum(axis=1)))
        top, left = origins[index]
        return name, int(top), int(left)

    def _supervision(self, name, top, left):
        """把标签网格裁到窗口；监督与 logits 同在 7mpp 网格，不上采样。"""
        slide_data = self.slide_data[name]
        label_top, label_left = top // self.label_scale, left // self.label_scale
        size = self.crop_size // self.label_scale
        arrays = {
            "cancer_probability": slide_data["cancer_probability"],
            "valid": slide_data["valid"],
            "superpixels": slide_data["superpixels"],
        }
        supervision = {}
        for key, array in arrays.items():
            crop = array[label_top : label_top + size, label_left : label_left + size]
            supervision[key] = torch.from_numpy(crop[None].copy())
        supervision["valid"] = supervision["valid"].bool()
        supervision["superpixels"] = supervision["superpixels"].int()
        return supervision

    def _spot_targets(self, name, top, left):
        """返回窗口内完整 spot 的局部坐标、表达和标签。"""
        spot_data = self.spots_by_slide[name]
        radius = self.spot_radius_px
        local = spot_data["coords"] - np.array([left, top], dtype=np.float32)
        keep = ((local >= radius) & (local <= self.crop_size - 1 - radius)).all(axis=1)
        indices = torch.from_numpy(spot_data["indices"][keep])
        return {
            "spots": torch.from_numpy(local[keep].astype(np.float32)),
            "gene_expression": self.gene_expression[indices],
            "spot_labels": self.spot_labels[indices],
        }


def collate_samples(samples):
    """spot 字段的长度随窗口内完整 spot 数变化，堆不了，保留为逐样本列表；其余按默认规则堆叠。"""
    variable_keys = {"spots", "gene_expression", "spot_labels"}
    batch = {}
    for key in samples[0]:
        values = [sample[key] for sample in samples]
        batch[key] = values if key in variable_keys else default_collate(values)
    return batch
