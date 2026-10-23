"""共享软状态：由同一组混合权重解码类别和非负线性表达。"""

import torch
from torch import nn
from torch.nn import functional as F

from segmodel.modules.state_usage import state_log_usage


class TissueStateDecoder(nn.Module):
    def __init__(self, in_channels, gene_count, state_count, hidden_dim):
        super().__init__()
        if gene_count <= 0 or state_count < 2 or hidden_dim <= 0:
            raise ValueError("Tissue states require genes, at least two states and a positive hidden dimension")
        self.gene_count = int(gene_count)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1),
            nn.GELU(),
        )
        self.state_head = nn.Conv2d(hidden_dim, state_count, 1)
        self.amplitude_head = nn.Conv2d(hidden_dim, 1, 1)
        # 全部模板随机初始化，并与状态头一起训练。
        self.template_logits = nn.Parameter(torch.empty(state_count, gene_count).normal_(std=0.5))
        # 避免所有类别倾向相同，使首次分割反向就能更新状态权重。
        self.class_logits = nn.Parameter(torch.empty(state_count).uniform_(-2, 2))

    def forward(self, features, compute_expression=True,
                state_usage_mask=None, state_usage_grad=True):
        latent = self.encoder(features)
        state_logits = self.state_head(latent)
        amplitude_logits = self.amplitude_head(latent) if compute_expression else None
        output = {}
        # 混合、模板归一化和概率转 logits 使用 FP32，兼容 AMP。
        with torch.autocast(device_type=features.device.type, enabled=False):
            log_states = state_logits.float().log_softmax(dim=1)
            if state_usage_mask is not None:
                # 在 checkpoint 内归约，只返回 K 维统计，不额外保留完整 log_states。
                with torch.set_grad_enabled(torch.is_grad_enabled() and state_usage_grad):
                    log_usage, pixels = state_log_usage(log_states, state_usage_mask)
                output.update(tissue_state_log_usage=log_usage, tissue_state_pixels=pixels)
            # RW 只使用 F 和分割概率；训练保留原表达分支的运算顺序。
            if amplitude_logits is not None:
                states = log_states.exp()
                templates = self.template_logits.float().softmax(dim=1)
                # 模板逐行和为 1；乘 G 让初始化的逐基因表达保持 O(1)。
                amplitude = F.softplus(amplitude_logits.float()) * self.gene_count
                output.update(tissue_states=states, expression_amplitude=amplitude, expression_templates=templates)
            classes = self.class_logits.float()[None, :, None, None]
            # log P - log(1-P)，不先 clamp 概率，避免极值处切断梯度。
            positive = torch.logsumexp(log_states + F.logsigmoid(classes), dim=1, keepdim=True)
            negative = torch.logsumexp(log_states + F.logsigmoid(-classes), dim=1, keepdim=True)
        output = {"logits": positive - negative, **output}
        return output
