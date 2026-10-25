import json
from pathlib import Path

import timm
import torch

from .common import FeatureMapExtractor, create_preprocess, get_patch_grid


class UNI2Extractor(FeatureMapExtractor):
    def __init__(
        self,
        weights_root: str | Path,
        batch_size: int = 8,
        device: str | torch.device | None = None,
        tile_size: int = 112,
        resize: bool = True,
        overlap_average: bool = False,
        overlap_cls: bool = False,
    ) -> None:
        patch_grid = get_patch_grid(tile_size, resize, 14)
        local_dir = Path(weights_root) / "UNI2"
        timm_kwargs = {
            "model_name": "vit_giant_patch14_224",
            "img_size": 224,
            "patch_size": 14,
            "depth": 24,
            "num_heads": 24,
            "init_values": 1e-05,
            "embed_dim": 1536,
            "mlp_ratio": 2.66667 * 2,
            "num_classes": 0,
            "no_embed_class": True,
            "mlp_layer": timm.layers.SwiGLUPacked,
            "act_layer": torch.nn.SiLU,
            "reg_tokens": 8,
            "dynamic_img_size": True,
            "global_pool": "token",
        }
        model = timm.create_model(pretrained=False, **timm_kwargs)
        model.load_state_dict(
            torch.load(
                local_dir / "pytorch_model.bin", map_location="cpu", weights_only=True
            ),
            strict=True,
        )
        with (local_dir / "config.json").open("r", encoding="utf-8") as file:
            config = json.load(file)
        model.pretrained_cfg = config["pretrained_cfg"]
        transform = create_preprocess(
            resize,
            mean=tuple(config["pretrained_cfg"]["mean"]),
            std=tuple(config["pretrained_cfg"]["std"]),
        )
        super().__init__(
            model,
            transform,
            patch_grid=patch_grid,
            patch_dimension=model.num_features,
            cls_dimension=model.num_features,
            batch_size=batch_size,
            device=device,
            tile_size=tile_size,
            resize=resize,
            overlap_average=overlap_average,
            overlap_cls=overlap_cls,
        )
