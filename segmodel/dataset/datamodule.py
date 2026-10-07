from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, TensorDataset

from segmodel.dataset.loading import (
    fold_paths,
    load_gene_adata,
    load_inference_slides,
    load_training_slides,
)
from segmodel.dataset.segmentation import SegmentationDataset, collate_samples
from segmodel.dataset.slide import SlideDataset


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
        self.train_slides = None
        self.train_gene_adata = None
        self.train_dataset = None

        self.val_generator = torch.Generator().manual_seed(seed + 10000)
        self.vae_val_generator = torch.Generator().manual_seed(seed + 20000)
        self.val_gene_adata = None
        self.val_slides = None

        self.test_slides = None
        self.predict_slides = None

    def setup(self, stage=None):
        dataset_cfg = self.cfg.experiment.dataset
        # fit 同时准备训练 crop、VAE 表达矩阵和留出切片的整片验证输入。
        if stage in (None, "fit") and self.train_dataset is None:
            self.train_slides = load_training_slides(
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

            self.train_dataset = SegmentationDataset(
                self.train_slides,
                self.train_gene_adata,
                self.train_names,
                self.cfg,
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
            self.train_slides = None
            self.train_gene_adata = None
            self.val_gene_adata = None
            self.train_dataset = None
            self.val_slides = None
        if stage in (None, "test"):
            self.test_slides = None
        if stage in (None, "predict"):
            self.predict_slides = None

    def train_dataloader(self):
        assert self.train_dataset is not None, "Call setup('fit') before train_dataloader()"
        return DataLoader(
            self.train_dataset,
            batch_size=int(self.cfg.experiment.training.image.batch_size),
            num_workers=int(self.cfg.experiment.dataset.num_workers),
            collate_fn=collate_samples,
            persistent_workers=False,
            pin_memory=True,
        )

    def val_dataloader(self):
        assert self.val_slides is not None, "Call setup('fit') before val_dataloader()"
        return DataLoader(
            SlideDataset(self.val_slides.values()),
            batch_size=None,
            num_workers=0,
            generator=self.val_generator,
        )

    def _vae_dataloader(
        self, gene_adata, shuffle, generator: torch.Generator | None = None
    ):
        expression = torch.from_numpy(gene_adata.X)
        spot_labels = torch.from_numpy(
            gene_adata.obs["spot_label"].to_numpy(dtype=np.int64)
        )
        return DataLoader(
            TensorDataset(expression, spot_labels),
            batch_size=int(self.cfg.experiment.training.vae.pretrain_batch_size),
            shuffle=shuffle,
            num_workers=0,
            generator=generator,
        )

    def vae_dataloader(self):
        return self._vae_dataloader(self.train_gene_adata, shuffle=True)

    def vae_val_dataloader(self):
        return self._vae_dataloader(
            self.val_gene_adata, shuffle=False, generator=self.vae_val_generator
        )

    def test_dataloader(self):
        assert self.test_slides is not None, "Call setup('test') before test_dataloader()"
        return DataLoader(
            SlideDataset(self.test_slides.values()),
            batch_size=None,
            num_workers=0,
        )

    def predict_dataloader(self):
        assert self.predict_slides is not None, "Call setup('predict') before predict_dataloader()"
        return DataLoader(
            SlideDataset(self.predict_slides.values()),
            batch_size=None,
            num_workers=0,
        )
