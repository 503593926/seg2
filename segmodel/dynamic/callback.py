from pathlib import Path

import cv2
import numpy as np
import pytorch_lightning as pl
import torch

from tools.propagate_random_walker import build_random_walker_topology

from segmodel.dynamic.propagation import (
    build_pseudo_mask,
    label_change_stats,
    propagate_features,
    snap_spots,
    weighted_features,
)
from segmodel.dynamic.visualization import save_dynamic_result, save_soft_label_result
from segmodel.inference.predictor import predict_slide


class DynamicPropagationCallback(pl.Callback):
    """按训练轮次使用模型特征重新生成训练切片的伪标签。"""

    def __init__(self, config, tiling, output_dir=None):
        self.config = config
        self.tiling = tiling
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.rounds = 0
        self.topologies = {}

    def state_dict(self):
        return {"rounds": self.rounds}

    def load_state_dict(self, state_dict):
        self.rounds = state_dict["rounds"]

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        if not self._should_run(epoch):
            return
        if trainer.world_size != 1:
            raise RuntimeError(
                "dynamic propagation currently requires one training process"
            )

        self.rounds += 1
        dataset = trainer.datamodule.train_dataset
        for name in dataset.slide_names:
            report = self._update_slide(
                dataset,
                name,
                pl_module.model,
                pl_module.device,
                trainer.datamodule.cfg.resolution,
                epoch,
            )
            print(
                f"dynamic round {self.rounds}: {name}, "
                f"crops={report['candidate_crop_count']}, "
                f"coverage={report['coverage_ratio']:.3f}, "
                f"flip={report['flip_fraction']:.3f}",
                flush=True,
            )

    def _should_run(self, epoch):
        if not self.config.enabled or self.rounds >= self.config.max_rounds:
            return False
        if epoch < self.config.warmup_epochs:
            return False
        return (epoch - self.config.warmup_epochs) % self.config.interval_epochs == 0

    def _should_save_intermediate(self):
        return self.output_dir is not None and bool(
            self.config.get("save_intermediate", True)
        )

    def _update_slide(self, dataset, name, model, device, resolution, epoch):
        # 当前普通模型整片推理, 获取传播特征。
        slide = dataset.slides[name]
        model_probability, learned = predict_slide(
            model,
            slide,
            crop_size=self.tiling["crop_size"],
            stride=self.tiling["stride"],
            alignment=dataset.alignment,
            device=device,
            batch_size=self.tiling["batch_size"],
        )

        tissue = slide.tissue_mask_bl
        if learned.shape[:2] != tissue.shape:
            raise ValueError(
                f"{name}: propagation features {learned.shape[:2]} "
                f"do not match label grid {tissue.shape}"
            )
        image = cv2.resize(
            slide.image,
            (tissue.shape[1], tissue.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
        from tools.snic_features import prepare_snic_lab, run_snic

        # 按轮次重算超像素, 其余轮次复用拓扑。
        lab = prepare_snic_lab(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        recompute_snic = self.rounds % int(self.config.snic_every_rounds) == 0
        superpixels = slide.superpixels
        if recompute_snic:
            features = weighted_features(
                lab,
                learned,
                tissue,
                float(self.config.learned_weight),
            )
            label_mpp = float(resolution.label_mpp)
            segments = max(
                1,
                round(
                    tissue.sum()
                    / (float(self.config.superpixel_size_um) / label_mpp) ** 2
                ),
            )
            superpixels, _ = run_snic(
                features,
                tissue,
                segments,
                compactness=float(self.config.compactness),
            )
            topology = build_random_walker_topology(superpixels)
        elif name in self.topologies:
            topology = self.topologies[name]
        else:
            topology = build_random_walker_topology(superpixels)

        # spot 吸附后作为随机游走种子, 传播结果阈值化为新伪标签。
        spot_data = dataset.spots_by_slide[name]
        labels = dataset.spot_labels.numpy()[spot_data["indices"]]
        label_mpp = float(resolution.label_mpp)
        x, y, labels = snap_spots(
            spot_data["coords"],
            labels,
            superpixels,
            dataset.label_scale,
            float(self.config.max_snap_distance_um) / label_mpp,
        )
        probability = propagate_features(
            superpixels,
            lab,
            learned,
            x,
            y,
            labels,
            float(self.config.beta),
            float(self.config.learned_weight),
            topology,
        )
        mask = build_pseudo_mask(
            probability,
            tissue,
            float(self.config.non_cancer_threshold),
            float(self.config.cancer_threshold),
        )
        coverage_ratio, flip_fraction = label_change_stats(
            slide.pseudo_mask_uint8,
            mask,
            tissue,
        )
        candidate_crop_count = len(dataset.candidate_origins(mask != 255))
        report = {
            "coverage_ratio": coverage_ratio,
            "flip_fraction": flip_fraction,
            "candidate_crop_count": int(candidate_crop_count),
            "snic_recomputed": recompute_snic,
        }
        # 下面两条不是质量判断，而是"这次更新不可用"：概率含 NaN、或没有任何可用训练窗口
        if not candidate_crop_count or not np.isfinite(probability).all():
            report["skipped"] = True
            return report
        if self._should_save_intermediate():
            model_probability = cv2.resize(
                model_probability,
                (tissue.shape[1], tissue.shape[0]),
                interpolation=cv2.INTER_AREA,
            )
            known = (labels == 0) | (labels == 1)
            seeded_superpixels = np.unique(superpixels[y[known], x[known]])
            seeded_superpixels = seeded_superpixels[seeded_superpixels > 0]
            result_dir = self.output_dir / f"epoch_{epoch:04d}" / name
            save_dynamic_result(
                result_dir,
                image,
                tissue,
                model_probability,
                probability,
                mask,
                slide.pseudo_mask_uint8,
                superpixels,
                learned,
                {
                    "epoch": int(epoch),
                    "round": int(self.rounds),
                    "slide": name,
                    "coverage_ratio": float(report["coverage_ratio"]),
                    "flip_fraction": float(report["flip_fraction"]),
                    "candidate_crop_count": int(candidate_crop_count),
                    "snic_recomputed": bool(recompute_snic),
                    "superpixel_count": int(superpixels.max()),
                    "known_spots": int(known.sum()),
                    "seeded_superpixels": len(seeded_superpixels),
                    "target_mpp": float(resolution.target_mpp),
                    "label_mpp": float(resolution.label_mpp),
                    "non_cancer_threshold": float(self.config.non_cancer_threshold),
                    "cancer_threshold": float(self.config.cancer_threshold),
                },
            )
            report["output_dir"] = str(result_dir)

        # 提交有效监督并刷新该切片下一轮可采样窗口。
        dataset.update_supervision(
            name,
            mask,
            np.asarray(probability, dtype=np.float32),
            np.asarray(superpixels, dtype=np.int32),
        )
        self.topologies[name] = topology
        return report


class SoftLabelVisualizationCallback(pl.Callback):
    """按轮次对训练切片整片推理, 保存可微传播产出的软标签可视化。

    与 DynamicPropagationCallback 的区别: 只读地用 spot 真值 + 当前模型特征
    生成软标签图, 不写回数据集、不影响训练。用于人工检查软标签质量随训练
    的演化 (阈值边界、模糊区间、与模型概率的差异)。
    """

    def __init__(self, config, tiling, output_dir=None):
        self.config = config
        self.tiling = tiling
        self.output_dir = Path(output_dir) if output_dir is not None else None

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        if not self._should_run(epoch):
            return
        if trainer.world_size != 1:
            return
        propagation = pl_module.propagation
        if propagation is None:
            return
        dataset = trainer.datamodule.train_dataset
        resolution = trainer.datamodule.cfg.resolution
        for name in dataset.slide_names:
            report = self._visualize_slide(
                dataset,
                name,
                pl_module,
                propagation,
                resolution,
                epoch,
            )
            print(
                f"soft label epoch {epoch}: {name}, "
                f"seeds={report['seed_counts']['cancer'] + report['seed_counts']['non_cancer']}, "
                f"supervised={report['supervised_fraction']:.3f}, "
                f"pcg={report['pcg_steps']:.0f}",
                flush=True,
            )

    def _should_run(self, epoch):
        if not bool(self.config.enabled) or self.output_dir is None:
            return False
        warmup = int(self.config.warmup_epochs)
        if epoch < warmup:
            return False
        return (epoch - warmup) % int(self.config.interval_epochs) == 0

    def _visualize_slide(self, dataset, name, pl_module, propagation, resolution, epoch):
        slide = dataset.slides[name]
        model_probability, learned = predict_slide(
            pl_module.model,
            slide,
            crop_size=self.tiling["crop_size"],
            stride=self.tiling["stride"],
            alignment=dataset.alignment,
            device=pl_module.device,
            batch_size=self.tiling["batch_size"],
        )
        tissue = slide.tissue_mask_bl
        if learned.shape[:2] != tissue.shape:
            raise ValueError(
                f"{name}: propagation features {learned.shape[:2]} "
                f"do not match label grid {tissue.shape}"
            )
        height, width = tissue.shape
        image = cv2.resize(
            slide.image,
            (width, height),
            interpolation=cv2.INTER_AREA,
        )
        model_probability = cv2.resize(
            model_probability,
            (width, height),
            interpolation=cv2.INTER_AREA,
        )
        superpixels = np.asarray(slide.superpixels, dtype=np.int64)

        # spot 坐标保持目标 mpp 像素单位, 由传播模块自己除以 label_scale;
        # 叠加图上的种子点画在标签网格上, 因此单独换算一份显示坐标。
        spot_data = dataset.spots_by_slide[name]
        labels = dataset.spot_labels.numpy()[spot_data["indices"]]
        coordinates = np.asarray(spot_data["coords"], dtype=np.float32)
        display = np.rint(coordinates / dataset.label_scale).astype(np.int64)
        display[:, 0] = display[:, 0].clip(0, width - 1)
        display[:, 1] = display[:, 1].clip(0, height - 1)

        device = pl_module.device
        with torch.no_grad():
            result = propagation(
                torch.from_numpy(superpixels)[None, None].to(device),
                torch.from_numpy(np.ascontiguousarray(learned.transpose(2, 0, 1)))
                .float()[None]
                .to(device),
                torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))
                .float()[None]
                .to(device),
                [torch.from_numpy(np.ascontiguousarray(labels)).to(device)],
                [torch.from_numpy(np.ascontiguousarray(coordinates)).to(device)],
            )
        if result is None:
            return {
                "seed_counts": {"cancer": 0, "non_cancer": 0},
                "supervised_fraction": 0.0,
                "pcg_steps": 0.0,
            }
        kept, soft, _, diagnostics = result
        if kept != [0]:
            raise ValueError(f"{name}: expected the single sample to be solvable")

        result_dir = self.output_dir / f"epoch_{epoch:04d}" / name
        record = save_soft_label_result(
            result_dir,
            image,
            tissue,
            soft[0, 0].cpu().numpy(),
            superpixels,
            display[:, 0],
            display[:, 1],
            labels,
            non_cancer_threshold=propagation.non_cancer_threshold,
            cancer_threshold=propagation.cancer_threshold,
            metadata={
                "epoch": int(epoch),
                "slide": name,
                "pcg_steps": float(diagnostics["pcg_steps"]),
                "gate_mean": float(diagnostics["gate_mean"]),
                "gate_std": float(diagnostics["gate_std"]),
                "superpixel_count": int(superpixels.max()),
                "target_mpp": float(resolution.target_mpp),
                "label_mpp": float(resolution.label_mpp),
            },
            model_probability=model_probability,
        )
        return {
            "seed_counts": record["seed_counts"],
            "supervised_fraction": float(
                (record["soft_class_counts"]["0"] + record["soft_class_counts"]["1"])
                / max(int(tissue.sum()), 1)
            ),
            "pcg_steps": float(diagnostics["pcg_steps"]),
        }
