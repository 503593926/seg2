import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler, default_collate

from segmodel.modules.propagation import prepare_graph
from segmodel.modules.spot_pooling import collate_spot_pools, prepare_spot_pool

from segmodel.dataset.augment import (
    augment_geometry,
    build_stain_transform,
    get_augmented_inputs,
    pfm_mask,
)

class EpochSampler(Sampler):
    """把 epoch 放进任务索引，预取和持久化 worker 不依赖本地 epoch 副本。"""

    def __init__(self, dataset):
        self.dataset = dataset

    def set_epoch(self, epoch):
        self.dataset.epoch = int(epoch)

    def __iter__(self):
        epoch = self.dataset.epoch
        return iter((epoch, index) for index in range(len(self.dataset)))

    def __len__(self):
        return len(self.dataset)


class SegmentationDataset(Dataset):
    def __init__(
        self,
        slides,
        gene_adata,
        slide_names,
        cfg,
        seed=0,
        labeled_names=None,
    ):
        self.slides = slides
        self.gene_expression = torch.from_numpy(gene_adata.X)
        if "spot_label" in gene_adata.obs:
            labels = gene_adata.obs["spot_label"].fillna(255).to_numpy(dtype=np.int64)
        else:
            labels = np.full(gene_adata.n_obs, 255, dtype=np.int64)
        labeled_names = set(slide_names if labeled_names is None else labeled_names)
        labels = labels.copy()
        labels[~gene_adata.obs["name"].astype(str).isin(labeled_names).to_numpy()] = 255
        self.spot_labels = torch.from_numpy(labels)
        self.slide_names = sorted(list(slide_names))
        self.labeled_names = [name for name in self.slide_names if name in labeled_names]
        self.expression_only_names = [name for name in self.slide_names if name not in labeled_names]
        self.seed = int(seed)
        self.epoch = 0
        self.shared_superpixels = {}
        if int(cfg.experiment.dataset.num_workers) > 0:
            for name, slide in self.slides.items():
                if slide.has_segmentation:
                    shared = torch.from_numpy(slide.superpixels).share_memory_()
                    self.shared_superpixels[name] = shared
                    slide.superpixels = shared.numpy()
        weights = cfg.experiment.training.image.loss_weights
        self.prepare_expression = any(
            float(weights.get(name, 0)) for name in ("gene_prediction", "gene_correlation")
        )
        self.prepare_propagation = bool(cfg.experiment.propagation.enabled) and any(
            float(weights.get(name, 0)) for name in ("propagation", "propagation_initial")
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
            self._refresh_slide(name)

    def _refresh_slide(self, name):
        # pseudo_mask_uint8: 0, 1, 255
        slide = self.slides[name]
        usable = slide.pseudo_mask_uint8 != 255 if slide.has_segmentation else slide.tissue_mask_bl
        self.crop_origins[name] = self.candidate_origins(usable)
        if not len(self.crop_origins[name]):
            raise ValueError(f"{name}: no usable training crops")

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
        epoch, index = index if isinstance(index, tuple) else (self.epoch, index)
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, epoch, index]))
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
        if self.prepare_expression:
            sample["spot_pool"] = prepare_spot_pool(
                sample["tissue"][0], sample["spots"], self.spot_radius_px, self.label_scale,
            )
        if self.prepare_propagation:
            sample["propagation_graph"] = prepare_graph(
                sample["superpixels"][0], sample["spot_labels"], sample["spots"], self.label_scale,
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
        origins = self.crop_origins[name]
        index = int(rng.integers(len(origins)))
        spot_data = self.spots_by_slide[name]

        fraction = float(self.augment_cfg.sampling.spot_sampling_fraction)
        expression_only = not self.slides[name].has_segmentation
        if len(spot_data["coords"]) and (expression_only or (self.augment_cfg.enabled and rng.random() < fraction)):
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
            "tissue": slide.tissue_mask_bl,
        }
        supervision = {}
        for key, array in arrays.items():
            if key == "superpixels" and name in self.shared_superpixels:
                array = self.shared_superpixels[name].numpy()
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
        }


def collate_samples(samples):
    variable_keys = {"spots", "gene_expression", "spot_labels"}
    batch = {}
    for key in samples[0]:
        values = [sample[key] for sample in samples]
        if key == "spot_pool":
            batch[key] = collate_spot_pools(values, samples[0]["tissue"].shape[-2:])
        elif key == "propagation_graph":
            batch["propagation_graphs"] = values
        else:
            batch[key] = values if key in variable_keys else default_collate(values)
    return batch
