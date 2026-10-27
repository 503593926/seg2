import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, default_collate

from seg2.models.spot_pooling import collate_spot_pools, prepare_spot_pool
from seg2.train.data.augment import (
    augment_geometry,
    build_stain_transform,
    get_augmented_inputs,
)


class TrainingDataset(Dataset):
    def __init__(self, slides, gene_adata, slide_names, cfg, seed=0):
        self.cfg = cfg
        self.seed = seed
        self.epoch = 0

        self.slides = slides
        self.slide_names = sorted(slide_names)

        self.labeled_names = [
            name for name in self.slide_names if slides[name]["has_segmentation"]
        ]
        self.expression_only_names = [
            name for name in self.slide_names if not slides[name]["has_segmentation"]
        ]

        self.gene_expression = torch.from_numpy(gene_adata.X)

        labels = gene_adata.obs["spot_label"].to_numpy(dtype=np.int64, copy=True)
        self.spot_labels = torch.from_numpy(labels)

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

        self.shared_supervision = {}
        if int(cfg.dataset.num_workers) > 0:
            for name in self.labeled_names:
                slide = self.slides[name]
                self.shared_supervision[name] = {}
                for key in ("pseudo_mask", "probability", "superpixels"):
                    shared = torch.from_numpy(slide[key]).share_memory_()
                    self.shared_supervision[name][key] = shared
                    slide[key] = shared.numpy()

        self.stain_transform = build_stain_transform(self.cfg.augment)
        self.crop_size = int(self.cfg.dataset.crop_size)

        first_slide = next(iter(self.slides.values()))
        self.label_scale = first_slide["label_scale"]
        self.alignment = first_slide["alignment"]
        self.spot_radius_px = first_slide["spot_radius_px"]
        auxiliary_weights = {
            str(scale): float(weight)
            for scale, weight in cfg.training.image.get("auxiliary_weights", {}).items()
        }
        task_weights = cfg.training.image.loss_weights
        self.auxiliary_pool_scales = (
            tuple(
                int(scale) for scale in cfg.model.image.get("auxiliary_scales", ())
                if float(auxiliary_weights.get(str(scale), 0))
            )
            if any(float(task_weights[name]) for name in ("gene_prediction", "gene_correlation"))
            else ()
        )
        if any(
            scale < self.label_scale or scale % self.label_scale or self.crop_size % scale
            for scale in self.auxiliary_pool_scales
        ):
            raise ValueError("Auxiliary scales must divide crop_size and be multiples of label_scale")

        self.crop_origins = {}
        for name in self.slide_names:
            self._refresh_slide(name)

    def __len__(self):
        return int(self.cfg.dataset.samples_per_epoch)

    def __getitem__(self, index):
        epoch, index, name, top, left = index
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, epoch, index]))

        sample = get_augmented_inputs(
            self.slides[name],
            top,
            left,
            rng,
            self.cfg.augment,
            self.crop_size,
            self.stain_transform,
        )

        sample.update(self._supervision(name, top, left))
        sample.update(self._spot_targets(name, top, left))
        if self.cfg.augment.enabled and self.cfg.augment.geometry.enabled:
            sample = augment_geometry(
                sample, rng, self.crop_size, self.cfg.augment.geometry
            )

        sample["spot_pool"] = prepare_spot_pool(
            sample["tissue"][0],
            sample["spots"],
            self.spot_radius_px,
            self.label_scale,
        )
        if self.auxiliary_pool_scales:
            # Rebuild after geometry augmentation, using each native grid's pixel spacing.
            sample["auxiliary_spot_pools"] = {}
            for scale in self.auxiliary_pool_scales:
                factor = scale // self.label_scale
                tissue = sample["tissue"][0, ::factor, ::factor]
                sample["auxiliary_spot_pools"][str(scale)] = prepare_spot_pool(
                    tissue, sample["spots"], self.spot_radius_px, scale,
                )
        return sample

    def _refresh_slide(self, name):
        slide = self.slides[name]
        if slide["has_segmentation"]:
            usable = slide["pseudo_mask"] != 255
        else:
            usable = slide["tissue_mask_bl"]

        self.crop_origins[name] = self.candidate_origins(usable)

    def update_supervision(self, name, mask, probability):
        slide = self.slides[name]
        slide["pseudo_mask"][...] = mask
        slide["probability"][...] = probability
        self._refresh_slide(name)

    def candidate_origins(self, valid):
        size, step = (
            self.crop_size // self.label_scale,
            self.alignment // self.label_scale,
        )
        integral = cv2.integral(valid.view(np.uint8), sdepth=cv2.CV_64F)
        area = (
            integral[size::step, size::step]
            - integral[:-size:step, size::step]
            - integral[size::step, :-size:step]
            + integral[:-size:step, :-size:step]
        )
        keep = area >= self.cfg.dataset.min_valid_fraction * size * size
        y, x = np.nonzero(keep)
        return np.stack((y * step, x * step), axis=1) * self.label_scale

    def _sample_slide_origins(self, name, count, rng):
        origins = self.crop_origins[name]
        spot_data = self.spots_by_slide[name]
        fraction = float(self.cfg.augment.sampling.spot_sampling_fraction)
        expression_only = not self.slides[name]["has_segmentation"]
        available = np.arange(len(origins))
        selected = []
        for _ in range(count):
            if not len(available):
                available = np.arange(len(origins))
            index = int(rng.integers(len(available)))
            if len(spot_data["coords"]) and (
                expression_only
                or (self.cfg.augment.enabled and rng.random() < fraction)
            ):
                center = spot_data["coords"][
                    int(rng.integers(len(spot_data["coords"])))
                ]
                desired = center[::-1] - self.crop_size / 2
                index = int(
                    np.argmin(((origins[available] - desired) ** 2).sum(axis=1))
                )
            top, left = origins[available[index]]
            selected.append((int(top), int(left)))
            available = np.delete(available, index)
        return selected

    def _supervision(self, name, top, left):
        slide = self.slides[name]
        label_top, label_left = (top // self.label_scale, left // self.label_scale)
        size = self.crop_size // self.label_scale
        keys = ["tissue_mask_bl"]
        if slide["has_segmentation"]:
            keys.extend(("pseudo_mask", "probability"))
        crops = {}
        for key in keys:
            if name in self.shared_supervision and key in self.shared_supervision[name]:
                array = self.shared_supervision[name][key].numpy()
            else:
                array = slide[key]
            crop = array[label_top : label_top + size, label_left : label_left + size]
            crops[key] = torch.from_numpy(crop[None].copy())
        tissue = crops["tissue_mask_bl"]
        if slide["has_segmentation"]:
            valid = crops["pseudo_mask"] != 255
            probability = crops["probability"]
        else:
            valid = torch.zeros_like(tissue)
            probability = torch.zeros_like(tissue, dtype=torch.float32)
        return {"tissue": tissue, "valid": valid, "cancer_probability": probability}

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
        elif key == "auxiliary_spot_pools":
            height, width = samples[0]["image"].shape[-2:]
            batch[key] = {
                scale: collate_spot_pools(
                    [value[scale] for value in values],
                    (height // int(scale), width // int(scale)),
                )
                for scale in values[0]
            }
        else:
            batch[key] = values if key in variable_keys else default_collate(values)
    return batch
