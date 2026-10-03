from math import lcm
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset, TensorDataset

from .loading import (
    fold_paths,
    load_gene_adata,
    load_inference_slides,
    load_training_slides,
)
from .segmentation import SegmentationDataset, collate_samples


class SlideDataset(Dataset):
    def __init__(self, slides):
        self.slides = list(slides)

    def __len__(self):
        return len(self.slides)

    def __getitem__(self, index):
        return self.slides[index]


class FoldDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_names: list[str],
        test_names: list[str],
        seed: int,
        cfg: DictConfig,
    ):
        super().__init__()
        self.train_names = train_names
        self.test_names = test_names
        self.seed = seed
        self.cfg = cfg
        self.project_root = Path(str(cfg.project_root))
        self.data_roots, self.gene_path = fold_paths(
            self.project_root,
            cfg.experiment.dataset,
            test_names,
        )
        self.train_slide_data = None
        self.train_gene_adata = None
        self.train_dataset = None

        self.val_gene_adata = None
        self.val_slides = None

        self.test_slides = None
        self.predict_slides = None

    def _prepare_geometry(self):
        """派生网格几何量，整个训练过程只在这里算一次。

        训练切片同源，取第一张的 patch 步长即可。label_scale 是分辨率换算，
        alignment 取监督网格与特征网格的最小公倍数——取窗起点必须同时落在两张网格上，
        数据集（候选窗口步长）与推理滑窗都取这一份。
        """
        self.patch_scale = next(iter(self.train_slide_data.values()))["patch_scale"]
        target_mpp = float(self.cfg.resolution.target_mpp)
        self.spot_radius_px = (
            float(self.cfg.experiment.dataset.spot_diameter_um) / (2.0 * target_mpp)
        )
        self.label_scale = round(
            float(self.cfg.resolution.label_mpp) / target_mpp
        )
        self.alignment = lcm(self.label_scale, self.patch_scale)

    def setup(self, stage=None):
        dataset_cfg = self.cfg.experiment.dataset
        if stage in (None, "fit") and self.train_dataset is None:
            self.train_slide_data = load_training_slides(
                self.data_roots,
                self.train_names,
                dataset_cfg,
                self.cfg.resolution,
            )
            self.train_gene_adata = load_gene_adata(self.gene_path, self.train_names)
            self.val_gene_adata = load_gene_adata(self.gene_path, self.test_names)
            self.val_slides = load_inference_slides(
                self.data_roots,
                self.test_names,
                dataset_cfg,
                self.cfg.resolution,
            )
            self._prepare_geometry()
            self.train_dataset = SegmentationDataset(
                self.train_slide_data,
                self.train_gene_adata,
                self.train_names,
                self.cfg,
                label_scale=self.label_scale,
                alignment=self.alignment,
                spot_radius_px=self.spot_radius_px,
                seed=self.seed,
            )

        if stage in (None, "test") and self.test_slides is None:
            self.test_slides = load_inference_slides(
                self.data_roots,
                self.test_names,
                dataset_cfg,
                self.cfg.resolution,
            )

        if stage in (None, "predict") and self.predict_slides is None:
            self.predict_slides = load_inference_slides(
                self.data_roots,
                self.test_names,
                dataset_cfg,
                self.cfg.resolution,
            )

    def teardown(self, stage=None):
        """释放当前 stage 的整片数组，避免切换阶段后同时占用内存。"""
        if stage in (None, "fit"):
            self.train_slide_data = None
            self.train_gene_adata = None
            self.val_gene_adata = None
            self.train_dataset = None
            self.val_slides = None
        if stage in (None, "test"):
            self.test_slides = None
        if stage in (None, "predict"):
            self.predict_slides = None

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=int(self.cfg.experiment.training.image.batch_size),
            num_workers=int(self.cfg.experiment.dataset.num_workers),
            collate_fn=collate_samples,
            persistent_workers=False,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            SlideDataset(self.val_slides.values()),
            batch_size=None,
            num_workers=0,
        )

    def _vae_dataloader(self, gene_adata, shuffle):
        expression = torch.from_numpy(gene_adata.X)
        spot_labels = torch.from_numpy(
            gene_adata.obs["spot_label"].to_numpy(dtype=np.int64)
        )
        return DataLoader(
            TensorDataset(expression, spot_labels),
            batch_size=int(self.cfg.experiment.training.vae.pretrain_batch_size),
            shuffle=shuffle,
            num_workers=0,
        )

    def vae_dataloader(self):
        return self._vae_dataloader(self.train_gene_adata, shuffle=True)

    def vae_val_dataloader(self):
        return self._vae_dataloader(self.val_gene_adata, shuffle=False)

    def test_dataloader(self):
        return DataLoader(
            SlideDataset(self.test_slides.values()),
            batch_size=None,
            num_workers=0,
        )

    def predict_dataloader(self):
        return DataLoader(
            SlideDataset(self.predict_slides.values()),
            batch_size=None,
            num_workers=0,
        )
