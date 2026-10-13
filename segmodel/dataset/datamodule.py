import os
from pathlib import Path

import cv2
import pytorch_lightning as pl
import torch
from lightning_fabric.utilities.seed import pl_worker_init_function
from omegaconf import DictConfig
from threadpoolctl import threadpool_limits
from torch.utils.data import DataLoader

from segmodel.dataset.loading import (
    fold_paths,
    load_gene_adata,
    load_inference_slides,
    load_training_slides,
)
from segmodel.dataset.segmentation import EpochSampler, SegmentationDataset, collate_samples
from segmodel.dataset.slide import SlideDataset
from segmodel.dataset.splits import patient_folds


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
        dataset_cfg = cfg.experiment.dataset
        self.train_labeled_names = list(train_names)
        self.expression_test_names = list(test_names)
        if dataset_cfg.split_mode == "patient_loocv":
            fold = next(
                fold for fold in patient_folds(dataset_cfg.all_names, dataset_cfg.expression_names)
                if fold["test"] == list(test_names)
            )
            if set(train_names) != set(fold["train"]):
                raise ValueError("Patient split contains held-out slides or omits training slides")
            self.train_labeled_names = [name for name in train_names if name in dataset_cfg.all_names]
            self.expression_test_names = fold["test_expression"]
        self.evaluation_names = list(dict.fromkeys([*test_names, *self.expression_test_names]))
        self.project_root = Path(str(cfg.project_root))
        self.data_roots, self.gene_path = fold_paths(
            self.project_root,
            cfg.experiment.dataset,
            test_names,
        )
        self.train_slides = None
        self.train_gene_adata = None
        self.train_dataset = None
        self._train_loader = None
        self.train_generator = torch.Generator().manual_seed(seed + 1000)

        self.val_generator = torch.Generator().manual_seed(seed + 10000)
        self.val_gene_adata = None
        self.val_slides = None

        self.test_slides = None
        self.predict_slides = None

    def setup(self, stage=None):
        dataset_cfg = self.cfg.experiment.dataset
        # 准备训练 crop、表达目标和留出患者的整片验证输入。
        if stage in (None, "fit") and self.train_dataset is None:
            if dataset_cfg.split_mode == "patient_loocv":
                self._check_patient_files()
            self.train_slides = load_training_slides(
                self.data_roots,
                self.train_names,
                dataset_cfg,
                self.cfg.resolution,
                labeled_names=self.train_labeled_names,
            )

            self.train_gene_adata = load_gene_adata(self.gene_path, self.train_names)
            if dataset_cfg.split_mode == "patient_loocv":
                if set(self.train_gene_adata.uns.get("train_names", [])) != set(self.train_names):
                    raise ValueError("Patient-fold gene selection metadata does not match the training patients; rerun derive/prepare_her2st_patient.py")

            self.val_gene_adata = load_gene_adata(self.gene_path, self.expression_test_names)
            self.val_slides = load_inference_slides(
                self.data_roots,
                self.evaluation_names,
                dataset_cfg,
                self.cfg.resolution,
            )

            self.train_dataset = SegmentationDataset(
                self.train_slides,
                self.train_gene_adata,
                self.train_names,
                self.cfg,
                seed=self.seed,
                labeled_names=self.train_labeled_names,
            )

        if stage in (None, "test") and self.test_slides is None:
            self.test_slides = load_inference_slides(
                self.data_roots,
                self.evaluation_names,
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

    def _check_patient_files(self):
        target_mpp = float(self.cfg.resolution.target_mpp)
        missing = []
        for name in self.evaluation_names + self.train_names:
            root = self.data_roots[name]
            paths = [
                root / 'samples' / name / f'spatial/he_{target_mpp:g}mpp.png',
                root / 'label_mpp' / name / 'spatial/tissue_mask.png',
            ]
            paths.extend(root / feature / name / filename
                         for feature in self.cfg.experiment.dataset.image_feature_maps
                         for filename in ('patch_map.npy', 'cls_map.npy'))
            missing.extend(str(path) for path in paths if not path.exists())
        if not self.gene_path.exists():
            missing.append(str(self.gene_path))
        if missing:
            raise FileNotFoundError(
                f'Patient training is missing {len(missing)} input files. '
                'Run derive/prepare_her2st_patient.py in your IDE first. Examples:\n'
                + '\n'.join(missing[:5])
            )

    def teardown(self, stage=None):
        """释放当前 stage 的整片数组，避免切换阶段后同时占用内存。"""
        if stage in (None, "fit"):
            if self._train_loader is not None:
                iterator = self._train_loader._iterator
                if iterator is not None:
                    iterator._shutdown_workers()
                self._train_loader = None
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
        if self._train_loader is None:
            dataset_cfg = self.cfg.experiment.dataset
            workers = int(dataset_cfg.num_workers)
            if workers:
                # fork 前关闭父进程的 OpenCV 并行池，避免子进程调整继承的池时死锁。
                cv2.setNumThreads(1)
            options = {"prefetch_factor": int(dataset_cfg.get("prefetch_factor", 2))} if workers else {}
            self._train_loader = DataLoader(
                self.train_dataset,
                batch_size=int(self.cfg.experiment.training.image.batch_size),
                sampler=EpochSampler(self.train_dataset),
                num_workers=workers,
                collate_fn=collate_samples,
                worker_init_fn=initialize_training_worker,
                persistent_workers=workers > 0 and bool(dataset_cfg.get("persistent_workers", True)),
                pin_memory=bool(dataset_cfg.get("pin_memory", True)),
                generator=self.train_generator,
                **options,
            )
        return self._train_loader

    def val_dataloader(self):
        assert self.val_slides is not None, "Call setup('fit') before val_dataloader()"
        return DataLoader(
            SlideDataset(self.val_slides.values()),
            batch_size=None,
            num_workers=0,
            generator=self.val_generator,
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
