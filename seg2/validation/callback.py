import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytorch_lightning as pl

from seg2.infer.memory import release_inference_cache
from seg2.test.configuration import evaluation_parameters
from seg2.train.pipeline import save_model


class ValidationCallback(pl.Callback):
    def __init__(self, cfg, names, data_root, gene_path, output_dir, *, dataset):
        self.cfg = cfg
        self.dataset = dataset
        self.names = list(names)
        self.data_root = Path(data_root)
        self.gene_path = Path(gene_path)
        self.output_dir = Path(output_dir)

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        options = self.cfg.evaluation.validation
        if epoch % int(options.every_n_epochs):
            return
        output = self.output_dir / f"epoch_{epoch:04d}"
        if str(options.weights) not in {"ema", "model"}:
            raise ValueError("Validation weights must be 'ema' or 'model'")
        model = (
            pl_module.ema_model if str(options.weights) == "ema" else pl_module.model
        )
        with tempfile.TemporaryDirectory(prefix="seg2_validation_model_") as directory:
            artifact = Path(directory) / "model.pt"
            save_model(model, artifact)
            command = [
                sys.executable,
                "-c",
                (
                    "import json, sys\n"
                    "from seg2.test import run_test\n"
                    "run_test(**json.load(sys.stdin))\n"
                ),
            ]
            parameters = evaluation_parameters(
                self.cfg,
                dataset=self.dataset,
                artifact_path=artifact,
                data_root=self.data_root,
                gene_path=self.gene_path,
                output_dir=output,
                names=self.names,
                device=pl_module.device,
                validation=True,
            )
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
                {f"val/{key}": value for key, value in expression["macro"].items()}
            )
            metrics["val/gene_prediction_pcc_macro"] = expression["macro"][
                "gene_prediction_pcc"
            ]
        region_metrics = summary.get("region_classification", {}).get("macro", {})
        metrics.update(
            {
                f"val/region_{key}": region_metrics[key]
                for key in ("auc", "ap", "f1", "acc")
                if region_metrics.get(key) is not None
            }
        )
        pl_module.log_dict(
            metrics, on_step=False, on_epoch=True, prog_bar=True, batch_size=1
        )
