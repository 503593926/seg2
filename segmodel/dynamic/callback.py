from pathlib import Path

import cv2
import numpy as np
import pytorch_lightning as pl

from tools.propagate_random_walker import build_random_walker_topology

from ..inference.predictor import predict_slide
from .propagation import (
    build_pseudo_mask,
    label_change_stats,
    propagate_features,
    snap_spots,
    weighted_features,
)
from .visualization import save_dynamic_result


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
        slide = dataset.slides[name]
        slide_data = dataset.slide_data[name]
        model_probability, learned = predict_slide(
            model,
            slide,
            crop_size=self.tiling["crop_size"],
            stride=self.tiling["stride"],
            alignment=dataset.alignment,
            device=device,
            batch_size=self.tiling["batch_size"],
        )

        tissue = slide_data["tissue_mask_bl"]
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

        lab = prepare_snic_lab(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        recompute_snic = self.rounds % int(self.config.snic_every_rounds) == 0
        superpixels = slide_data["superpixels"]
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
            slide_data["pseudo_mask"],
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
                slide_data["pseudo_mask"],
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

        dataset.update_supervision(
            name,
            mask,
            np.asarray(probability, dtype=np.float32),
            np.asarray(superpixels, dtype=np.int32),
        )
        self.topologies[name] = topology
        return report
