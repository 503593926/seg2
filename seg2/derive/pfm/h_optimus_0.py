from pathlib import Path

import timm
import torch

from .common import FeatureMapExtractor, create_preprocess, get_patch_grid


class HO0Extractor(FeatureMapExtractor):
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
        local_dir = Path(weights_root) / "HOptimus0"
        model = timm.create_model(
            "vit_giant_patch14_reg4_dinov2",
            pretrained=False,
            num_classes=0,
            global_pool="token",
            img_size=224,
            init_values=1e-05,
            dynamic_img_size=True,
        )
        model.load_state_dict(
            torch.load(
                local_dir / "pytorch_model.bin", map_location="cpu", weights_only=True
            ),
            strict=True,
        )
        transform = create_preprocess(
            resize,
            mean=(0.707223, 0.578729, 0.703617),
            std=(0.211883, 0.230117, 0.177517),
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
