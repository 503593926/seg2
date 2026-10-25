import torch

from seg2.models.states import STATE_USAGE_CHUNK_ELEMENTS


@torch.no_grad()
def state_diagnostics(output, batch):
    metrics = {}
    states = output["tissue_states"].detach()
    valid = batch["valid"].to(dtype=states.dtype)
    chunk_rows = max(1, STATE_USAGE_CHUNK_ELEMENTS // (states.shape[0] * states.shape[1] * states.shape[3]))
    entropy = states.new_zeros((), dtype=torch.float32)
    usage = states.new_zeros(states.shape[1], dtype=torch.float32)
    valid_pixels = states.new_zeros((), dtype=torch.float32)
    for chunk, valid_chunk in zip(states.split(chunk_rows, dim=2), valid.split(chunk_rows, dim=2)):
        terms = chunk.clamp_min(1e-08).log_().mul_(chunk)
        entropy.sub_(terms.sum())
        del terms
        usage.add_((chunk * valid_chunk).sum(dim=(0, 2, 3)))
        valid_pixels.add_(valid_chunk.sum())
    usage = usage / valid_pixels.clamp_min(1)
    metrics["tissue_state_entropy"] = entropy / (states.shape[0] * states.shape[2] * states.shape[3])
    metrics["tissue_state_usage_entropy"] = -(usage * usage.clamp_min(1e-08).log()).sum()
    tissue_log_usage = output["tissue_state_log_usage"]
    tissue_pixels = output["tissue_state_pixels"]
    log_usage_tissue = tissue_log_usage.detach()
    usage_tissue = log_usage_tissue.exp()
    tissue_entropy = -(usage_tissue * log_usage_tissue).sum()
    metrics["state_effective_states"] = torch.where(
        tissue_pixels > 0, tissue_entropy.exp(), usage_tissue.new_zeros(())
    )
    metrics["expression_amplitude_mean"] = output["expression_amplitude"].detach().mean()
    return metrics
