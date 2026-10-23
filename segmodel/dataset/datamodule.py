import os
from typing import Any, cast

import cv2
import pytorch_lightning as pl
import torch
from lightning_fabric.utilities.seed import pl_worker_init_function
from omegaconf import DictConfig
from threadpoolctl import threadpool_limits
from torch.utils.data import DataLoader

from segmodel.dataset.loading import (
    load_gene_adata,
    load_inference_slides,
    load_training_slides,
)
from segmodel.dataset.sampling import EpochSampler
from segmodel.dataset.segmentation import (
    SegmentationDataset,
    collate_samples,
)
from segmodel.dataset.slide import SlideDataset


def initialize_training_worker(worker_id):
    """保留 Lightning worker 种子，并限制每个数据加载进程的原生线程池。"""
    if int(os.environ.get("PL_SEED_WORKERS", "0")):
        pl_worker_init_function(worker_id)
    cv2.setNumThreads(1)
    # 不使用 with，让限制持续覆盖该 worker 的整个生命周期。
    threadpool_limits(limits=1, user_api="blas")


class FoldDataModule(pl.LightningDataModule):
    def __init__(
        self,
        fold: dict,
        seed: int,
        cfg: DictConfig,
    ):
        super().__init__()
        self.train_names = fold["train_names"]
        self.test_names = fold["test_names"]
        self.data_roots = fold["data_roots"]
        self.gene_path = fold["gene_path"]
        self.seed = seed
        self.cfg = cfg
        self.train_slides = None
        self.train_gene_adata = None
        self.train_dataset = None
        self._train_loader = None
        self.train_generator = torch.Generator().manual_seed(seed + 1000)

        self.val_generator = torch.Generator().manual_seed(seed + 10000)
        self.val_gene_adata = None
        self.val_slides = {}

        self.test_slides = {}

    def setup(self, stage=None):
        dataset_cfg = self.cfg.experiment.dataset
        if stage in (None, "fit") and self.train_dataset is None:
            self.train_slides = load_training_slides(
                self.data_roots,
                self.train_names,
                dataset_cfg,
                self.cfg.resolution,
            )

            self.train_gene_adata = load_gene_adata(self.gene_path, self.train_names)
            if set(self.train_gene_adata.uns["train_names"]) != set(self.train_names):
                raise ValueError("Gene selection metadata does not match the training slides")

            self.val_slides = {}
            if dataset_cfg.split_mode == "patient_loocv":
                self.val_gene_adata = load_gene_adata(self.gene_path, self.test_names)
                self.val_slides = load_inference_slides(
                    self.data_roots, self.test_names, dataset_cfg, self.cfg.resolution,
                )

            self.train_dataset = SegmentationDataset(
                self.train_slides,
                self.train_gene_adata,
                self.train_names,
                self.cfg,
                seed=self.seed,
            )

        if stage in (None, "test") and not self.test_slides:
            self.test_slides = load_inference_slides(
                self.data_roots,
                self.test_names,
                dataset_cfg,
                self.cfg.resolution,
            )

    def teardown(self, stage=None):
        """释放当前 stage 的整片数组，避免切换阶段后同时占用内存。"""
        if stage in (None, "fit"):
            if self._train_loader is not None:
                iterator = self._train_loader._iterator
                if iterator is not None:
                    cast(Any, iterator)._shutdown_workers()
                self._train_loader = None
            self.train_slides = None
            self.train_gene_adata = None
            self.val_gene_adata = None
            self.train_dataset = None
            self.val_slides = {}
        if stage in (None, "test"):
            self.test_slides = {}

    def train_dataloader(self):
        if self._train_loader is None:
            dataset = cast(SegmentationDataset, self.train_dataset)
            dataset_cfg = self.cfg.experiment.dataset
            workers = int(dataset_cfg.num_workers)
            if workers:
                # fork 前关闭父进程的 OpenCV 并行池，避免子进程调整继承的池时死锁。
                cv2.setNumThreads(1)
            options: dict[str, Any] = {"prefetch_factor": int(dataset_cfg.prefetch_factor)} if workers else {}
            batch_size = int(self.cfg.experiment.training.image.batch_size)
            self._train_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=EpochSampler(dataset, batch_size, dataset_cfg.get("slides_per_batch")),
                num_workers=workers,
                collate_fn=collate_samples,
                worker_init_fn=initialize_training_worker,
                persistent_workers=workers > 0 and bool(dataset_cfg.persistent_workers),
                pin_memory=bool(dataset_cfg.pin_memory),
                generator=self.train_generator,
                **options,
            )
        return self._train_loader

    def val_dataloader(self):
        return DataLoader(
            SlideDataset(self.val_slides.values()),
            batch_size=None,
            num_workers=0,
            generator=self.val_generator,
        )

    def test_dataloader(self):
        return DataLoader(
            SlideDataset(self.test_slides.values()),
            batch_size=None,
            num_workers=0,
        )
