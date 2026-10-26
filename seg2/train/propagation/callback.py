from pathlib import Path

import cv2
import pytorch_lightning as pl
import torch

from seg2.infer.memory import release_inference_cache
from seg2.infer.predictor import predict_slide
from seg2.train.propagation.propagation import (
    build_pseudo_mask,
    label_change_stats,
    prepare_propagation,
    propagate_features,
)
from seg2.train.propagation.visualization import save_soft_label_result


class DynamicPropagationCallback(pl.Callback):
    def __init__(self, config, rw_config, output_dir=None):
        self.config = config
        self.rw = rw_config
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.rounds = 0
        self.slide_cache = {}

    def state_dict(self):
        return {"rounds": self.rounds}

    def load_state_dict(self, state_dict):
        self.rounds = int(state_dict["rounds"])
        self.slide_cache.clear()

    def on_fit_end(self, trainer, pl_module):
        self.slide_cache.clear()

    def _should_run(self, epoch):
        warmup = int(self.config.warmup_epochs)
        return (
            bool(self.config.enabled)
            and epoch >= warmup
            and ((epoch - warmup) % int(self.config.interval_epochs) == 0)
        )

    @torch.inference_mode()
    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        if not self._should_run(epoch):
            return
        self.rounds += 1
        data_module = trainer.datamodule
        dataset = data_module.train_dataset
        release_inference_cache(pl_module.device, f"dynamic epoch {epoch} begin")
        try:
            for name in dataset.labeled_names:
                print(
                    f"dynamic propagation epoch {epoch}: {name}, extracting F",
                    flush=True,
                )
                with trainer.precision_plugin.forward_context():
                    report = self._update_slide(
                        dataset,
                        name,
                        pl_module.model,
                        pl_module.device,
                        data_module.cfg.prep.resolution,
                        epoch,
                    )
                print(
                    f"dynamic round {self.rounds}: {name}, crops={report['candidate_crop_count']}, coverage={report['coverage_ratio']:.3f}, flip={report['flip_fraction']:.3f}",
                    flush=True,
                )
        finally:
            release_inference_cache(pl_module.device, f"dynamic epoch {epoch} end")

    def _update_slide(self, dataset, name, model, device, resolution, epoch):
        from seg2.common.snic import prepare_snic_lab

        slide = dataset.slides[name]
        superpixels = slide["superpixels"]
        tissue = slide["tissue_mask_bl"]
        height, width = tissue.shape
        image = cv2.resize(
            slide["image"], (width, height), interpolation=cv2.INTER_AREA
        )
        label_mpp = float(resolution.label_mpp)
        if name not in self.slide_cache:
            lab = prepare_snic_lab(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            spot_data = dataset.spots_by_slide[name]
            labels = dataset.spot_labels.numpy()[spot_data["indices"]]
            self.slide_cache[name] = prepare_propagation(
                superpixels,
                lab,
                spot_data["coords"],
                labels,
                dataset.label_scale,
                float(self.rw.max_snap_distance_um) / label_mpp,
            )
            del lab
        cache = self.slide_cache[name]
        topology = cache["topology"]
        _, learned = predict_slide(
            model,
            slide,
            device=device,
            crop_size=dataset.crop_size,
            stride=int(self.config.stride),
            batch_size=int(self.config.batch_size),
            return_features=True,
            superpixel_pool=cache["feature_pool"],
        )
        probability = propagate_features(
            superpixels,
            learned,
            float(self.rw.beta),
            float(self.config.learned_weight),
            cache,
        )
        mask = build_pseudo_mask(
            probability,
            tissue,
            float(self.rw.non_cancer_threshold),
            float(self.rw.cancer_threshold),
        )
        coverage, flip = label_change_stats(slide["pseudo_mask"], mask, tissue)
        crop_count = len(dataset.candidate_origins(mask != 255))
        report = {
            "coverage_ratio": coverage,
            "flip_fraction": flip,
            "candidate_crop_count": int(crop_count),
        }
        if self.output_dir is not None and bool(self.config.save_intermediate):
            result_dir = self.output_dir / f"epoch_{epoch:04d}" / name
            save_soft_label_result(
                result_dir,
                image,
                tissue,
                probability,
                mask,
                cache["spot_labels"],
                non_cancer_threshold=float(self.rw.non_cancer_threshold),
                cancer_threshold=float(self.rw.cancer_threshold),
                metadata={
                    **report,
                    "epoch": int(epoch),
                    "round": self.rounds,
                    "slide": name,
                    "source_model": "current",
                    "feature_key": "features",
                    "feature_channels": int(learned.shape[-1]),
                    "rw_scope": "full_slide",
                    "rw_device": "cpu",
                    "target_mpp": float(resolution.target_mpp),
                    "label_mpp": label_mpp,
                    "superpixel_count": int(topology.max_label),
                    "rw_beta": float(self.rw.beta),
                    "learned_weight": float(self.config.learned_weight),
                },
            )
            report["output_dir"] = str(result_dir)
        dataset.update_supervision(name, mask, probability)
        return report
