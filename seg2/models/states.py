import torch
from torch import nn
from torch.nn import functional as F

STATE_USAGE_CHUNK_ELEMENTS = 4194304


class TissueStateDecoder(nn.Module):
    def __init__(self, in_channels, gene_count, state_count, hidden_dim):
        super().__init__()
        self.gene_count = int(gene_count)
        self.encoder = nn.Sequential(nn.Conv2d(in_channels, hidden_dim, 1), nn.GELU())
        self.state_head = nn.Conv2d(hidden_dim, state_count, 1)
        self.amplitude_head = nn.Conv2d(hidden_dim, 1, 1)
        self.template_logits = nn.Parameter(torch.empty(state_count, gene_count).normal_(std=0.5))
        self.class_logits = nn.Parameter(torch.empty(state_count).uniform_(-2, 2))

    def forward(self, features, compute_expression=True, state_usage_mask=None, state_usage_grad=True):
        latent = self.encoder(features)
        state_logits = self.state_head(latent)
        amplitude_logits = self.amplitude_head(latent) if compute_expression else None
        output = {}
        with torch.autocast(device_type=features.device.type, enabled=False):
            log_states = state_logits.float().log_softmax(dim=1)
            if state_usage_mask is not None:
                with torch.set_grad_enabled(torch.is_grad_enabled() and state_usage_grad):
                    log_usage, pixels = state_log_usage(log_states, state_usage_mask)
                output.update(tissue_state_log_usage=log_usage, tissue_state_pixels=pixels)
            if amplitude_logits is not None:
                states = log_states.exp()
                templates = self.template_logits.float().softmax(dim=1)
                amplitude = F.softplus(amplitude_logits.float()) * self.gene_count
                output.update(
                    tissue_states=states, expression_amplitude=amplitude, expression_templates=templates
                )
            classes = self.class_logits.float()[None, :, None, None]
            positive = torch.logsumexp(log_states + F.logsigmoid(classes), dim=1, keepdim=True)
            negative = torch.logsumexp(log_states + F.logsigmoid(-classes), dim=1, keepdim=True)
        output = {"logits": positive - negative, **output}
        return output


def state_log_usage(log_states, mask, *, chunk_rows=None):
    batch, count, _, width = log_states.shape
    log_states = log_states.float()
    mask = mask.to(device=log_states.device)
    pixels = mask.sum()
    mask = mask.to(dtype=log_states.dtype)
    if chunk_rows is None:
        chunk_rows = max(1, STATE_USAGE_CHUNK_ELEMENTS // (batch * count * width))
    chunk_rows = max(1, int(chunk_rows))
    log_mass = log_states.new_zeros(count)
    seen = pixels.new_zeros((), dtype=torch.bool)
    for state_chunk, mask_chunk in zip(log_states.split(chunk_rows, dim=2), mask.split(chunk_rows, dim=2)):
        present = mask_chunk.sum() > 0
        log_mask = torch.where(present, mask_chunk, 1.0).log()
        chunk_mass = torch.logsumexp(state_chunk + log_mask, dim=(0, 2, 3))
        merged = torch.where(seen, torch.logaddexp(log_mass, chunk_mass), chunk_mass)
        log_mass = torch.where(present, merged, log_mass)
        seen = seen | present
    normalizer = torch.where(pixels > 0, pixels, 1).to(dtype=log_mass.dtype)
    return (log_mass - normalizer.log(), pixels)
