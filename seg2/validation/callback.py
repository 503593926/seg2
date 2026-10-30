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

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        options = self.cfg.validation
        if epoch % int(options.every_n_epochs):
            return
        output = self.output_dir / f"epoch_{epoch:04d}"
        dataset = self.cfg.dataset
        model = pl_module.ema_model if str(options.weights) == "ema" else pl_module.model
        with tempfile.TemporaryDirectory(prefix="seg2_validation_model_") as directory:
            artifact = Path(directory) / "model.pt"
            save_model(model, artifact)
            command = [
                sys.executable,
                "-c",
                "import json, sys\n"
                "from seg2.test import run_test\n"
                "run_test(**json.load(sys.stdin))\n",
            ]
            parameters = {
                "artifact_path": str(artifact),
                "data_root": str(self.data_root),
                "gene_path": str(self.gene_path),
                "output_dir": str(output),
                "names": self.names,
                "feature_maps": list(map(str, dataset.image_feature_maps)),
                "target_mpp": float(self.cfg.prep.resolution.target_mpp),
                "spot_diameter_um": float(dataset.spot_diameter_um),
                "crop_size": int(options.crop_size),
                "stride": None if options.stride is None else int(options.stride),
                "inference_batch_size": int(options.batch_size),
                "device": str(pl_module.device),
                "precision": str(options.precision),
                "expression_transform": str(options.expression_transform),
                "visualization_top_n": int(options.top_genes),
            }
            release_inference_cache(
                pl_module.device, f"development validation epoch {epoch} begin"
            )
            try:
                subprocess.run(
                    command,
                    input=json.dumps(parameters),
                    text=True,
                    check=True,
                    cwd=Path(__file__).resolve().parents[2],
                )
            finally:
                release_inference_cache(
                    pl_module.device, f"development validation epoch {epoch} end"
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
        pl_module.log_dict(
            metrics, on_step=False, on_epoch=True, prog_bar=True, batch_size=1
        )
