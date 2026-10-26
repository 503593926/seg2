import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytorch_lightning as pl

from seg2.infer.memory import release_inference_cache
from seg2.train.pipeline import save_model


class ValidationCallback(pl.Callback):
    def __init__(self, cfg, names, data_root, gene_path, output_dir):
        self.cfg = cfg
        self.names = list(names)
        self.data_root = Path(data_root)
        self.gene_path = Path(gene_path)
        self.output_dir = Path(output_dir)

    def on_train_epoch_end(self, trainer, module):
        epoch = trainer.current_epoch + 1
        options = self.cfg.validation
        if epoch % int(options.every_n_epochs):
            return
        output = self.output_dir / f"epoch_{epoch:04d}"
        dataset = self.cfg.dataset
        model = module.ema_model if str(options.weights) == "ema" else module.model
        with tempfile.TemporaryDirectory(prefix="seg2_validation_model_") as directory:
            artifact = Path(directory) / "model.pt"
            save_model(model, artifact)
            command = [
                sys.executable,
                "-m",
                "seg2.test",
                "--artifact",
                str(artifact),
                "--data-root",
                str(self.data_root),
                "--gene-path",
                str(self.gene_path),
                "--output",
                str(output),
                "--slides",
                *self.names,
                "--feature-maps",
                *map(str, dataset.image_feature_maps),
                "--target-mpp",
                str(float(self.cfg.prep.resolution.target_mpp)),
                "--spot-diameter-um",
                str(float(dataset.spot_diameter_um)),
                "--crop-size",
                str(int(options.crop_size)),
                "--inference-batch-size",
                str(int(options.batch_size)),
                "--device",
                str(module.device),
                "--precision",
                str(options.precision),
                "--expression-transform",
                str(options.expression_transform),
                "--visualization-top-n",
                str(int(options.top_genes)),
            ]
            if options.stride is not None:
                command.extend(["--stride", str(int(options.stride))])
            release_inference_cache(
                module.device, f"development validation epoch {epoch} begin"
            )
            try:
                subprocess.run(
                    command, check=True, cwd=Path(__file__).resolve().parents[2]
                )
            finally:
                release_inference_cache(
                    module.device, f"development validation epoch {epoch} end"
                )
        summary = json.loads((output / "metrics.json").read_text())
        metrics = {f"val/{key}": value for key, value in summary["macro"].items()}
        expression = summary["expression"]
        if expression:
            metrics.update(
                {f"val/{key}": value for key, value in expression["micro"].items()}
            )
            metrics["val/gene_prediction_pcc_macro"] = expression["macro"][
                "gene_prediction_pcc"
            ]
        module.log_dict(
            metrics, on_step=False, on_epoch=True, prog_bar=True, batch_size=1
        )
