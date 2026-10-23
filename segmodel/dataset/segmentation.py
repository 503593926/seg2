import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, default_collate

from segmodel.dataset.augment import (
    augment_geometry,
    build_stain_transform,
    get_augmented_inputs,
    pfm_mask,
)
# 保留原导入路径，采样实现集中在 sampling.py。
from segmodel.dataset.sampling import EpochSampler as EpochSampler
from segmodel.modules.spot_pooling import collate_spot_pools, prepare_spot_pool


class SegmentationDataset(Dataset):
    def __init__(
        self,
        slides,
        gene_adata,
        slide_names,
        cfg,
        seed=0,
    ):
        self.slides = slides
        self.slide_names = sorted(slide_names)
        self.labeled_names = [name for name in self.slide_names if slides[name].has_segmentation]
        self.expression_only_names = [name for name in self.slide_names if not slides[name].has_segmentation]
        self.gene_expression = torch.from_numpy(gene_adata.X)
        if "spot_label" in gene_adata.obs:
            labels = gene_adata.obs["spot_label"].fillna(255).to_numpy(dtype=np.int64)
        else:
            labels = np.full(gene_adata.n_obs, 255, dtype=np.int64)
        labels = labels.copy()
        labels[~gene_adata.obs["name"].astype(str).isin(self.labeled_names).to_numpy()] = 255
        self.spot_labels = torch.from_numpy(labels)
        self.seed = int(seed)
        self.epoch = 0
        self.shared_supervision = {}
        self.shared_crop_origins = {}
        if int(cfg.experiment.dataset.num_workers) > 0:
            # 动态监督共享更新；固定超像素也共享，避免 spawn 复制整片数组。
            for name in self.labeled_names:
                slide = self.slides[name]
                self.shared_supervision[name] = {}
                for key, attribute in (("valid", "pseudo_mask_uint8"),
                                       ("cancer_probability", "cancer_probability"),
                                       ("superpixels", "superpixels")):
                    shared = torch.from_numpy(getattr(slide, attribute)).share_memory_()
                    self.shared_supervision[name][key] = shared
                    setattr(slide, attribute, shared.numpy())
        weights = cfg.experiment.training.image.loss_weights
        self.prepare_expression = any(
            float(weights.get(name, 0)) for name in ("gene_prediction", "gene_correlation")
        )

        self.spots_by_slide = {}
        obs_names = gene_adata.obs["name"].astype(str).to_numpy()
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
        self.stain_transform = build_stain_transform(self.augment_cfg)
        self.crop_size = int(self.dataset_cfg.crop_size)

        first_slide = next(iter(self.slides.values()))
        self.pfm_count = len(first_slide.patch_maps)
        self.label_scale = first_slide.label_scale
        self.alignment = first_slide.alignment
        self.spot_radius_px = first_slide.spot_radius_px

        if self.crop_size % self.alignment:
            raise ValueError(f"crop_size must be divisible by {self.alignment}")

        self.crop_origins = {}
        for name in self.slide_names:
            if name in self.shared_supervision:
                size = self.crop_size // self.label_scale
                step = self.alignment // self.label_scale
                height, width = self.slides[name].tissue_mask_bl.shape
                capacity = max(0, (height - size) // step + 1) * max(0, (width - size) // step + 1)
                self.shared_crop_origins[name] = (
                    torch.empty((capacity, 2), dtype=torch.int64).share_memory_(),
                    torch.zeros((), dtype=torch.int64).share_memory_(),
                )
            self._refresh_slide(name)

    def _refresh_slide(self, name):
        # pseudo_mask_uint8: 0, 1, 255
        slide = self.slides[name]
        usable = slide.pseudo_mask_uint8 != 255 if slide.has_segmentation else slide.tissue_mask_bl
        self.crop_origins[name] = self.candidate_origins(usable)
        if not len(self.crop_origins[name]):
            raise ValueError(f"{name}: no usable training crops")
        if name in self.shared_crop_origins:
            buffer, count = self.shared_crop_origins[name]
            origins = self.crop_origins[name]
            buffer[:len(origins)].copy_(torch.from_numpy(origins))
            count.fill_(len(origins))

    def update_supervision(self, name, mask, probability):
        """原地提交整片监督，保持常驻 worker 使用的共享存储。"""
        slide = self.slides[name]
        slide.pseudo_mask_uint8[...] = mask
        slide.cancer_probability[...] = probability
        self._refresh_slide(name)

    def candidate_origins(self, valid):
        size, step = self.crop_size // self.label_scale, self.alignment // self.label_scale
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

    def __len__(self):
        return int(self.dataset_cfg.samples_per_epoch)

    def __getitem__(self, index):
        epoch, index, *origin = index if isinstance(index, tuple) else (self.epoch, index)
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, epoch, index]))
        name, top, left = origin if origin else self._sample_origin(rng)
        sample = get_augmented_inputs(
            self.slides[name],
            top,
            left,
            rng,
            self.augment_cfg,
            self.crop_size,
            self.stain_transform,
        )
        sample.update(self._supervision(name, top, left))
        sample.update(self._spot_targets(name, top, left))

        if self.augment_cfg.enabled:
            if self.augment_cfg.geometry.enabled:
                sample = augment_geometry(sample, rng, self.crop_size, self.augment_cfg.geometry)
            sample["pfm_mask"] = pfm_mask(self.pfm_count, rng, self.augment_cfg.pfm_mask)
        else:
            sample["pfm_mask"] = torch.ones(self.pfm_count, dtype=torch.bool)
        if self.prepare_expression:
            sample["spot_pool"] = prepare_spot_pool(
                sample["tissue"][0], sample["spots"], self.spot_radius_px, self.label_scale,
            )
        return sample

    def _sample_origin(self, rng):
        names = self.slide_names
        fraction = self.dataset_cfg.get("labeled_sampling_fraction")
        if fraction is not None and self.labeled_names and self.expression_only_names:
            if not 0 < float(fraction) < 1:
                raise ValueError("labeled_sampling_fraction must be between 0 and 1")
            names = self.labeled_names if rng.random() < float(fraction) else self.expression_only_names
        name = names[int(rng.integers(len(names)))]
        top, left = self._sample_slide_origins(name, 1, rng)[0]
        return name, top, left

    def _sample_slide_origins(self, name, count, rng):
        origins = self.crop_origins[name]
        if name in self.shared_crop_origins:
            buffer, valid_count = self.shared_crop_origins[name]
            origins = buffer[:int(valid_count)].numpy()
        spot_data = self.spots_by_slide[name]
        fraction = float(self.augment_cfg.sampling.spot_sampling_fraction)
        expression_only = not self.slides[name].has_segmentation
        available = np.arange(len(origins))
        selected = []
        for _ in range(count):
            if not len(available):
                available = np.arange(len(origins))
            index = int(rng.integers(len(available)))
            if len(spot_data["coords"]) and (expression_only or (self.augment_cfg.enabled and rng.random() < fraction)):
                center = spot_data["coords"][int(rng.integers(len(spot_data["coords"])))]
                desired = center[::-1] - self.crop_size / 2
                index = int(np.argmin(((origins[available] - desired) ** 2).sum(axis=1)))
            top, left = origins[available[index]]
            selected.append((int(top), int(left)))
            available = np.delete(available, index)
        return selected

    def _supervision(self, name, top, left):
        slide = self.slides[name]
        label_top, label_left = top // self.label_scale, left // self.label_scale
        size = self.crop_size // self.label_scale
        arrays = {
            "cancer_probability": slide.cancer_probability,
            "valid": slide.pseudo_mask_uint8,
            "tissue": slide.tissue_mask_bl,
        }
        supervision = {}
        for key, array in arrays.items():
            if key in self.shared_supervision.get(name, {}):
                array = self.shared_supervision[name][key].numpy()
            crop = array[label_top : label_top + size, label_left : label_left + size]
            supervision[key] = torch.from_numpy(crop[None].copy())

        supervision["valid"] = supervision["valid"] != 255
        return supervision

    def _spot_targets(self, name, top, left):
        spot_data = self.spots_by_slide[name]
        radius = self.spot_radius_px
        local = spot_data["coords"] - np.array([left, top], dtype=np.float32)
        keep = ((local >= radius) & (local <= self.crop_size - 1 - radius)).all(axis=1)
        indices = torch.from_numpy(spot_data["indices"][keep])
        return {
            "slide": name,
            "spots": torch.from_numpy(local[keep].astype(np.float32)),
            "gene_expression": self.gene_expression[indices],
        }


def collate_samples(samples):
    variable_keys = {"spots", "gene_expression"}
    batch = {}
    for key in samples[0]:
        values = [sample[key] for sample in samples]
        if key == "spot_pool":
            batch[key] = collate_spot_pools(values, samples[0]["tissue"].shape[-2:])
        else:
            batch[key] = values if key in variable_keys else default_collate(values)
    return batch
