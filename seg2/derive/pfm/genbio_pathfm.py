from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from safetensors.torch import load_file

from .common import FeatureMapExtractor, create_preprocess, get_patch_grid

if TYPE_CHECKING:
    from .GenBio_PathFM.modeling_genbio_pathfm import GenBioPathFMModel


class GenBioPathFMExtractor(FeatureMapExtractor):
    model: GenBioPathFMModel

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
        from .GenBio_PathFM.configuration_genbio_pathfm import GenBioPathFMConfig
        from .GenBio_PathFM.modeling_genbio_pathfm import GenBioPathFMModel

        local_dir = Path(weights_root) / "GenBio_PathFM"
        config = GenBioPathFMConfig(
            **json.loads((local_dir / "config.json").read_text(encoding="utf-8"))
        )
        model = GenBioPathFMModel(config)
        model.load_state_dict(load_file(local_dir / "model.safetensors"), strict=True)
        transform = create_preprocess(
            resize, mean=tuple(config.image_mean), std=tuple(config.image_std)
        )
        dimension = config.embed_dim * 3
        super().__init__(
            model,
            transform,
            patch_grid=patch_grid,
            patch_dimension=dimension,
            cls_dimension=dimension,
            batch_size=batch_size,
            device=device,
            tile_size=tile_size,
            resize=resize,
            overlap_average=overlap_average,
            overlap_cls=overlap_cls,
        )

    def _forward_tiles(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cls_features, patch_features = self.model.forward_with_patches(images)
        return (patch_features, cls_features)
