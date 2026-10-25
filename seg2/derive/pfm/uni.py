import json
from pathlib import Path

import timm
import torch

from .common import FeatureMapExtractor, create_preprocess, get_patch_grid


class UNIExtractor(FeatureMapExtractor):
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
        patch_grid = get_patch_grid(tile_size, resize, 16)
        local_dir = Path(weights_root) / "UNI"
        model = timm.create_model(
            "vit_large_patch16_224",
            pretrained=False,
            img_size=224,
            init_values=1.0,
            num_classes=0,
            global_pool="token",
            dynamic_img_size=True,
        )
        model.load_state_dict(
            torch.load(
                local_dir / "pytorch_model.bin", map_location="cpu", weights_only=True
            ),
            strict=True,
        )
        config = json.loads((local_dir / "config.json").read_text(encoding="utf-8"))
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
