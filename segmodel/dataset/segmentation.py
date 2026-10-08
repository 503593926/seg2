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
        self.gene_expression = torch.from_numpy(gene_adata.X)
        self.spot_labels = torch.from_numpy(gene_adata.obs["spot_label"].to_numpy(dtype=np.int64))
        self.slide_names = sorted(list(slide_names))
        self.slide_ids = {name: index for index, name in enumerate(self.slide_names)}
        self.seed = int(seed)
        self.epoch = 0

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
            self._refresh_slide(name)

    def _refresh_slide(self, name):
        # pseudo_mask_uint8: 0, 1, 255
        self.crop_origins[name] = self.candidate_origins(self.slides[name].pseudo_mask_uint8 != 255)

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

    def update_supervision(
        self,
        name,
        pseudo_mask_uint8,
        cancer_probability,
        superpixels,
    ):
        """替换一张训练切片的动态监督，并只刷新该切片的训练窗口。"""
        self.slides[name].update_supervision(pseudo_mask_uint8, cancer_probability, superpixels)
        self._refresh_slide(name)

    def __len__(self):
        return int(self.dataset_cfg.samples_per_epoch)

    def __getitem__(self, index):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index]))
        name, top, left = self._sample_origin(rng)
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
        return sample

    def _sample_origin(self, rng):
        name = self.slide_names[int(rng.integers(len(self.slide_names)))]
        origins = self.crop_origins[name]
        index = int(rng.integers(len(origins)))
        spot_data = self.spots_by_slide[name]

        fraction = float(self.augment_cfg.sampling.spot_sampling_fraction)
        if self.augment_cfg.enabled and len(spot_data["coords"]) and rng.random() < fraction:
            center = spot_data["coords"][int(rng.integers(len(spot_data["coords"])))]
            desired = center[::-1] - self.crop_size / 2
            index = int(np.argmin(((origins - desired) ** 2).sum(axis=1)))
        top, left = origins[index]
        return name, int(top), int(left)

    def _supervision(self, name, top, left):
        slide = self.slides[name]
        label_top, label_left = top // self.label_scale, left // self.label_scale
        size = self.crop_size // self.label_scale
        arrays = {
            "cancer_probability": slide.cancer_probability,
            "valid": slide.pseudo_mask_uint8,
            "superpixels": slide.superpixels,
        }
        supervision = {}
        for key, array in arrays.items():
            crop = array[label_top : label_top + size, label_left : label_left + size]
            supervision[key] = torch.from_numpy(crop[None].copy())

        supervision["valid"] = supervision["valid"] != 255
        supervision["superpixels"] = supervision["superpixels"].int()
        return supervision

    def _spot_targets(self, name, top, left):
        spot_data = self.spots_by_slide[name]
        radius = self.spot_radius_px
        local = spot_data["coords"] - np.array([left, top], dtype=np.float32)
        keep = ((local >= radius) & (local <= self.crop_size - 1 - radius)).all(axis=1)
        indices = torch.from_numpy(spot_data["indices"][keep])
        return {
            "spots": torch.from_numpy(local[keep].astype(np.float32)),
            "gene_expression": self.gene_expression[indices],
            "spot_labels": self.spot_labels[indices],
            'spot_ids': indices,
            'slide_id': self.slide_ids[name],
        }


def collate_samples(samples):
    variable_keys = {"spots", "gene_expression", "spot_labels", 'spot_ids'}
    batch = {}
    for key in samples[0]:
        values = [sample[key] for sample in samples]
        batch[key] = values if key in variable_keys else default_collate(values)
    return batch
