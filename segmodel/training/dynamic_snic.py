"""按轮次重划分训练切片的 SNIC，并保存同一特征上的整片 RW 诊断。"""

from copy import deepcopy
from pathlib import Path
from typing import cast

import cv2
import numpy as np
import pytorch_lightning as pl
import torch

from segmodel.inference.predictor import predict_slide
from segmodel.inference.visualization import save_soft_label_result
from segmodel.modules.propagation import DifferentiablePropagation


def weighted_snic_features(lab, learned, tissue, learned_weight):
    """保留 Lab 尺度，按组织内通道标准差归一化 F，并平衡通道数量。"""
    reference = learned[tissue]
    if not len(reference):
        raise ValueError("Dynamic SNIC requires a nonempty tissue mask")
    reference_scale = np.sqrt(np.mean(lab[tissue].std(axis=0) ** 2))
    features = np.zeros((*tissue.shape, lab.shape[-1] + learned.shape[-1]), dtype=np.float32)
    features[tissue, :lab.shape[-1]] = lab[tissue]
    features[tissue, lab.shape[-1]:] = (
        (reference - reference.mean(axis=0)) / np.maximum(reference.std(axis=0), 1e-6)
        * np.sqrt(lab.shape[-1] * learned_weight / learned.shape[-1]) * reference_scale
    )
    return features


class DynamicSNICCallback(pl.Callback):
    def __init__(self, config, tiling, output_dir, target_mpp):
        self.config = config
        self.tiling = tiling
        self.output_dir = Path(output_dir)
        self.target_mpp = float(target_mpp)
        if int(config.warmup_epochs) < 1 or int(config.interval_epochs) < 1:
            raise ValueError("Dynamic SNIC warmup_epochs and interval_epochs must be positive")
        if float(config.superpixel_size_um) <= 0 or float(config.compactness) < 0 or float(config.learned_weight) < 0:
            raise ValueError("Invalid dynamic SNIC size, compactness or learned_weight")

    def _should_run(self, epoch):
        warmup = int(self.config.warmup_epochs)
        return bool(self.config.enabled) and epoch >= warmup and (epoch - warmup) % int(self.config.interval_epochs) == 0

    @torch.inference_mode()
    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        if not self._should_run(epoch):
            return
        if trainer.world_size != 1:
            raise RuntimeError("Dynamic SNIC currently requires a single training process")
        dataset = getattr(trainer, "datamodule").train_dataset
        # 整片诊断在 CPU 求解，避免整片 F 和图额外占用训练显存；参数和训练 RW 一致。
        propagation = deepcopy(cast(DifferentiablePropagation, pl_module.propagation)).cpu().eval()
        for name in dataset.labeled_names:
            print(f"dynamic SNIC epoch {epoch}: {name}, extracting F", flush=True)
            with trainer.precision_plugin.forward_context():
                record = self._update_slide(dataset, name, pl_module.model, pl_module.device, propagation, epoch)
            print(
                f"dynamic SNIC epoch {epoch}: {name}, nodes={record['superpixel_count']}, "
                f"RW coverage={record['supervised_fraction']:.3f}", flush=True,
            )

    def _update_slide(self, dataset, name, model, device, propagation, epoch):
        from tools.snic_features import prepare_snic_lab, run_snic

        slide = dataset.slides[name]
        model_probability, learned = predict_slide(
            model, slide, device=device,
            crop_size=int(self.tiling["crop_size"]), stride=int(self.tiling["stride"]),
            alignment=int(self.tiling["alignment"]), batch_size=int(self.tiling["batch_size"]),
            return_features=True, feature_key="features",
        )
        tissue = slide.tissue_mask_bl
        if learned is None or learned.shape[:2] != tissue.shape:
            raise ValueError(f"{name}: MoMu F and SNIC tissue grid must match")
        if not np.isfinite(learned[tissue]).all():
            raise FloatingPointError(f"{name}: nonfinite MoMu features during dynamic SNIC")
        height, width = tissue.shape
        image = cv2.resize(slide.image, (width, height), interpolation=cv2.INTER_AREA)
        lab = prepare_snic_lab(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        label_mpp = self.target_mpp * dataset.label_scale
        segments = max(1, round(tissue.sum() / (float(self.config.superpixel_size_um) / label_mpp) ** 2))
        superpixels, snic_stats = run_snic(
            weighted_snic_features(lab, learned, tissue, float(self.config.learned_weight)),
            tissue, segments, compactness=float(self.config.compactness),
        )
        spot_data = dataset.spots_by_slide[name]
        labels = dataset.spot_labels[torch.from_numpy(spot_data["indices"])]
        # 原始 spot 坐标交给现有求解器换算，保持与训练时相同的种子投票协议。
        with torch.autocast(device_type="cpu", enabled=False):
            result = propagation(
                torch.from_numpy(superpixels)[None, None].long(),
                torch.from_numpy(learned).permute(2, 0, 1)[None],
                torch.from_numpy(image).permute(2, 0, 1)[None],
                [labels], [torch.from_numpy(spot_data["coords"])],
            )
        probability = np.where(tissue, 0.5, 0).astype(np.float32)
        supervised = np.zeros(tissue.shape, dtype=bool)
        diagnostics = {}
        if result is not None:
            _, soft, valid, diagnostics = result
            probability = soft[0, 0].cpu().numpy()
            supervised = valid[0, 0].cpu().numpy()
        if not np.isfinite(probability[tissue]).all():
            raise FloatingPointError(f"{name}: nonfinite whole-slide RW probability")
        record = save_soft_label_result(
            self.output_dir / f"epoch_{epoch:04d}" / name,
            image, tissue, probability, supervised, superpixels,
            spot_data["coords"] / dataset.label_scale, labels.numpy(),
            model_probability=cv2.resize(model_probability, (width, height), interpolation=cv2.INTER_AREA),
            non_cancer_threshold=propagation.non_cancer_threshold,
            cancer_threshold=propagation.cancer_threshold,
            metadata={
                "epoch": int(epoch), "slide": name,
                "round": (epoch - int(self.config.warmup_epochs)) // int(self.config.interval_epochs) + 1,
                "source_model": "current", "feature_key": "features",
                "feature_channels": int(learned.shape[-1]), "rw_scope": "full_slide",
                "rw_device": "cpu", "rw_solved": result is not None,
                "target_mpp": self.target_mpp, "label_mpp": label_mpp,
                "snic_recomputed": True, "superpixel_size_um": float(self.config.superpixel_size_um),
                "compactness": float(self.config.compactness), "snic_learned_weight": float(self.config.learned_weight),
                "previous_superpixel_count": int(np.unique(slide.superpixels[slide.superpixels > 0]).size),
                "superpixel_count": int(superpixels.max()),
                "rw_beta": propagation.beta, "rw_learned_weight": propagation.learned_weight,
                "rw_steps_limit": propagation.steps, "rw_tolerance": propagation.tolerance,
                **snic_stats, **diagnostics,
            },
        )
        # 只更新训练拓扑。RW 仍逐 crop 生成目标，表达、类别种子和采样掩码保持原数据。
        # 保留共享存储，让跨 epoch 复用的 worker 立即读取新拓扑。
        slide.superpixels[...] = superpixels
        return record
