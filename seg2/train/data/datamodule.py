import os
from pathlib import Path
from typing import Any

import cv2
import pytorch_lightning as pl
import torch
from lightning_fabric.utilities.seed import pl_worker_init_function
from omegaconf import DictConfig
from threadpoolctl import threadpool_limits
from torch.utils.data import DataLoader
from typing import Any, cast
from torch.utils.data.dataloader import _MultiProcessingDataLoaderIter
from seg2.train.data.dataset import TrainingDataset, collate_samples
from seg2.train.data.loading import load_gene_adata, load_training_slides
from seg2.train.data.sampling import EpochSampler


def initialize_training_worker(worker_id):
    if int(os.environ.get("PL_SEED_WORKERS", "0")):
        pl_worker_init_function(worker_id)
    cv2.setNumThreads(1)
    threadpool_limits(limits=1, user_api="blas")


class FoldDataModule(pl.LightningDataModule):
    def __init__(self, fold: dict, seed: int, cfg: DictConfig):
        super().__init__()
        self.fold = fold
        self.seed = seed
        self.cfg = cfg
        self.train_slides = None
        self.train_gene_adata = None
        self.train_dataset = None
        self._train_loader = None
        self.train_generator = torch.Generator().manual_seed(seed + 1000)

    def setup(self, stage=None):
        if stage in (None, "fit") and self.train_dataset is None:
            self.train_slides = load_training_slides(self.fold, self.cfg)
            self.train_gene_adata = load_gene_adata(self.fold, self.cfg)
            self.train_dataset = TrainingDataset(
                self.train_slides,
                self.train_gene_adata,
                self.fold["train_names"],
                self.cfg,
                seed=self.seed,
            )

    def teardown(self, stage=None):
        if stage in (None, "fit"):
            if self._train_loader is not None:
                iterator = self._train_loader._iterator
                if iterator is not None:
                    cast(_MultiProcessingDataLoaderIter, iterator)._shutdown_workers()
                self._train_loader = None
            self.train_slides = None
            self.train_gene_adata = None
            self.train_dataset = None

    def train_dataloader(self):
        if self._train_loader is None:
            dataset = self.train_dataset
            assert dataset is not None

            workers = int(self.cfg.dataset.num_workers)
            if workers:
                cv2.setNumThreads(1)
            batch_size = int(self.cfg.training.image.batch_size)
            self._train_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=EpochSampler(dataset, batch_size, self.cfg.dataset.slides_per_batch),
                num_workers=workers,
                collate_fn=collate_samples,
                worker_init_fn=initialize_training_worker,
                persistent_workers=workers > 0 and bool(self.cfg.dataset.persistent_workers),
                pin_memory=bool(self.cfg.dataset.pin_memory),
                generator=self.train_generator,
                prefetch_factor=int(self.cfg.dataset.prefetch_factor) if workers > 0 else None,
            )
        return self._train_loader
