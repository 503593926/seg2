from pathlib import Path

import torch
from timm.models.vision_transformer import VisionTransformer
from torchvision.transforms import InterpolationMode

from .common import FeatureMapExtractor, create_preprocess, get_patch_grid


class _AttentionPool(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.query = torch.nn.Parameter(torch.randn(1, 512))
        self.attn = torch.nn.MultiheadAttention(512, 8, kdim=768, vdim=768)
        self.ln_q = torch.nn.LayerNorm(512)
        self.ln_k = torch.nn.LayerNorm(768)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        context = self.ln_k(tokens).to(tokens.dtype).permute(1, 0, 2)
        query = self.ln_q(self.query).to(self.query.dtype)
        query = query.unsqueeze(1).repeat(1, tokens.shape[0], 1)
        pooled, _ = self.attn(query, context, context, need_weights=False)
        return pooled.permute(1, 0, 2)


class _CONCHVisual(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.trunk = VisionTransformer(
            img_size=448,
            patch_size=16,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            num_classes=0,
            dynamic_img_size=True,
        )
        self.attn_pool_contrast = _AttentionPool()
        self.ln_contrast = torch.nn.LayerNorm(512)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.trunk.forward_features(images)
        pooled = self.ln_contrast(self.attn_pool_contrast(tokens)[:, 0])
        return (tokens[:, self.trunk.num_prefix_tokens :], pooled)


class CONCHExtractor(FeatureMapExtractor):
    model: _CONCHVisual

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
        local_dir = Path(weights_root) / "CONCH"
        model = _CONCHVisual()
        checkpoint = torch.load(
            local_dir / "pytorch_model.bin", map_location="cpu", weights_only=True
        )
        prefixes = (
            "visual.trunk.",
            "visual.attn_pool_contrast.",
            "visual.ln_contrast.",
        )
        visual_weights = {
            key.removeprefix("visual."): value
            for key, value in checkpoint.items()
            if key.startswith(prefixes)
        }
        model.load_state_dict(visual_weights, strict=True)
        transform = create_preprocess(
            resize,
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
            interpolation=InterpolationMode.BICUBIC,
        )
        super().__init__(
            model,
            transform,
            patch_grid=patch_grid,
            patch_dimension=768,
            cls_dimension=512,
            batch_size=batch_size,
            device=device,
            tile_size=tile_size,
            resize=resize,
            overlap_average=overlap_average,
            overlap_cls=overlap_cls,
        )

    def _forward_tiles(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model.forward(images)
