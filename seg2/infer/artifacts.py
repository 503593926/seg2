from pathlib import Path

import torch

from seg2.models.model import JointModel


def load_model(path, device="cpu"):
    artifact = torch.load(Path(path), map_location="cpu", weights_only=True)
    model = JointModel(**artifact["model_config"])
    model.load_state_dict(artifact["state_dict"])
    model.model_config = artifact["model_config"]
    model.gene_names = list(map(str, artifact["gene_names"]))
    return (model.to(device).eval(), artifact)
