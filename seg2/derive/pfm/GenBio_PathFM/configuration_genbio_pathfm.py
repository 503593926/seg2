from transformers import PretrainedConfig

class GenBioPathFMConfig(PretrainedConfig):
    model_type = 'genbio_pathfm'

    def __init__(self, img_size=224, patch_size=16, in_chans=1, embed_dim=1536, depth=40, num_heads=24, ffn_ratio=4, n_storage_tokens=4, ffn_layer='swiglu64', layerscale_init=1e-05, qkv_bias=False, proj_bias=True, ffn_bias=True, norm_layer='layernorm', drop_path_rate=0.0, pos_embed_rope_base=100.0, pos_embed_rope_min_period=None, pos_embed_rope_max_period=None, pos_embed_rope_normalize_coords='separate', pos_embed_rope_shift_coords=None, pos_embed_rope_jitter_coords=True, pos_embed_rope_rescale_coords=2, pos_embed_rope_dtype='bf16', image_mean=(0.697, 0.575, 0.728), image_std=(0.188, 0.24, 0.187), **kwargs):
        super().__init__(**kwargs)
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.ffn_ratio = ffn_ratio
        self.n_storage_tokens = n_storage_tokens
        self.ffn_layer = ffn_layer
        self.layerscale_init = layerscale_init
        self.qkv_bias = qkv_bias
        self.proj_bias = proj_bias
        self.ffn_bias = ffn_bias
        self.norm_layer = norm_layer
        self.drop_path_rate = drop_path_rate
        self.pos_embed_rope_base = pos_embed_rope_base
        self.pos_embed_rope_min_period = pos_embed_rope_min_period
        self.pos_embed_rope_max_period = pos_embed_rope_max_period
        self.pos_embed_rope_normalize_coords = pos_embed_rope_normalize_coords
        self.pos_embed_rope_shift_coords = pos_embed_rope_shift_coords
        self.pos_embed_rope_jitter_coords = pos_embed_rope_jitter_coords
        self.pos_embed_rope_rescale_coords = pos_embed_rope_rescale_coords
        self.pos_embed_rope_dtype = pos_embed_rope_dtype
        self.image_mean = image_mean
        self.image_std = image_std
