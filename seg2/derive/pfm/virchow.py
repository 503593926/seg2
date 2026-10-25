import json
from pathlib import Path

import timm
import torch
from timm.layers import SwiGLUPacked
from torchvision.transforms import InterpolationMode

from .common import FeatureMapExtractor, create_preprocess, get_patch_grid


class VirchowExtractor(FeatureMapExtractor):
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
        local_dir = Path(weights_root) / "Virchow"
        model = timm.create_model(
            "vit_huge_patch14_224",
            pretrained=False,
            img_size=224,
            init_values=1e-05,
            num_classes=0,
            mlp_ratio=5.3375,
            global_pool="",
            dynamic_img_size=True,
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU,
        )
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
            interpolation=InterpolationMode.BICUBIC,
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

    def _forward_tiles(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        model = self.model
        tokens = model.forward_features(images)
        return (tokens[:, model.num_prefix_tokens :], tokens[:, 0])
