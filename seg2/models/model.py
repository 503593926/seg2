import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from seg2.models.backbone import ConvNeXtUnet, Stem, init_convnext_backbone
from seg2.models.conditioning import PFMProjector
from seg2.models.states import TissueStateDecoder


class JointModel(nn.Module):
    gene_names: list[str]
    model_config: dict

    def __init__(
        self,
        pfm_patch_dims,
        pfm_cls_dims,
        gene_count,
        *,
        top_k,
        encoder_chans,
        decoder_chans,
        global_dim,
        spatial_dim,
        skip_drop_path,
        drop_path,
        tissue_state_count,
        tissue_state_dim,
        output_scale,
        gradient_checkpointing,
        downsample_factor,
        encoder_spatial,
        decoder_spatial,
        auxiliary_scales=(),
    ):
        super().__init__()
        self.pfm_patch_dims = tuple(pfm_patch_dims)
        self.pfm_cls_dims = tuple(pfm_cls_dims)
        self.pfm_count = len(self.pfm_patch_dims)
        self.tissue_state_count = int(tissue_state_count)
        self.output_scale = int(output_scale)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.downsample_factor = int(downsample_factor)
        self.auxiliary_scales = tuple(int(scale) for scale in auxiliary_scales)
        decoder_scales = [
            self.output_scale * self.downsample_factor ** (len(encoder_chans) - 2 - i)
            for i in range(len(decoder_chans))
        ]
        if len(set(self.auxiliary_scales)) != len(self.auxiliary_scales) or any(
            scale not in decoder_scales[:-1] for scale in self.auxiliary_scales
        ):
            raise ValueError(
                f"auxiliary_scales must be distinct intermediate decoder scales: {decoder_scales[:-1]}"
            )
        self._auxiliary_stage_indices = {
            str(scale): decoder_scales.index(scale) for scale in self.auxiliary_scales
        }
        self.stem = Stem(
            in_chans=3, out_chans=encoder_chans[0], down_factor=self.output_scale
        )
        self.unet = ConvNeXtUnet(
            encoder_chans=encoder_chans,
            decoder_chans=decoder_chans,
            drop_path=drop_path,
            blocks=2,
            global_dim=global_dim,
            spatial_dim=spatial_dim,
            pfm_count=self.pfm_count,
            g_top_k=top_k,
            s_top_k=top_k,
            skip_drop_path=skip_drop_path,
            gradient_checkpointing=self.gradient_checkpointing,
            downsample_factor=self.downsample_factor,
            encoder_spatial=encoder_spatial,
            decoder_spatial=decoder_spatial,
        )
        self.feature_channels = int(decoder_chans[-1])
        self.global_projector = PFMProjector(pfm_cls_dims, global_dim, spatial=False)
        self.spatial_projector = PFMProjector(pfm_patch_dims, spatial_dim, spatial=True)
        with torch.random.fork_rng(devices=[]):
            init_convnext_backbone(self.stem)
            init_convnext_backbone(self.unet)
        self.gene_count = int(gene_count)
        with torch.random.fork_rng(devices=[]):
            self.tissue_state_decoder = TissueStateDecoder(
                self.feature_channels,
                self.gene_count,
                self.tissue_state_count,
                tissue_state_dim,
            )
        # Construct auxiliary adapters after the main model without changing its RNG state.
        with torch.random.fork_rng(devices=[]):
            self.auxiliary_projections = nn.ModuleDict(
                {
                    scale: nn.Conv2d(decoder_chans[index], self.feature_channels, 1)
                    for scale, index in self._auxiliary_stage_indices.items()
                }
            )
            init_convnext_backbone(self.auxiliary_projections)

    def forward(
        self,
        image,
        patch,
        cls,
        compute_expression=True,
        state_usage_mask=None,
        state_usage_grad=True,
        compute_auxiliary=False,
    ):
        task_global = self.global_projector(
            [value.float().mean(dim=(-2, -1)) for value in cls]
        )
        task_spatial = (
            self.spatial_projector(
                patch
                if torch.is_autocast_enabled(image.device.type)
                else [value.float() for value in patch]
            )
            if self.unet.use_spatial
            else []
        )
        if image.dtype == torch.uint8:
            image = image.float().mul_(2.0 / 255.0).sub_(1.0)
        use_auxiliary = compute_auxiliary and bool(self.auxiliary_scales)
        decoded = self.unet(
            self.stem(image), task_global, task_spatial,
            return_decoder_features=use_auxiliary,
        )
        if use_auxiliary:
            decoded_fm, decoder_features = decoded
        else:
            decoded_fm = decoded
        output = {"features": decoded_fm}
        decoder_args = (
            decoded_fm,
            compute_expression,
            state_usage_mask,
            state_usage_grad,
        )
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            states = checkpoint(
                self.tissue_state_decoder, *decoder_args, use_reentrant=False
            )
        else:
            states = self.tissue_state_decoder(*decoder_args)
        output.update(states)
        if use_auxiliary:
            auxiliary = {}
            for scale, index in self._auxiliary_stage_indices.items():
                features = self.auxiliary_projections[scale](decoder_features[index])
                if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                    auxiliary[scale] = checkpoint(
                        self.tissue_state_decoder, features, compute_expression,
                        use_reentrant=False,
                    )
                else:
                    auxiliary[scale] = self.tissue_state_decoder(features, compute_expression)
            output["auxiliary"] = auxiliary
        return output
