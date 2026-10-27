import math

import torch
from torch.nn import functional as F

from seg2.models.spot_pooling import pool_spot_batch


def _gene_correlation_terms(predictions, targets, slide_names, min_spots, std_floor):
    prediction_residuals, target_residuals = ([], [])
    for name in dict.fromkeys(slide_names):
        prediction = torch.cat([value for value, slide in zip(predictions, slide_names) if slide == name])
        if len(prediction) < max(3, min_spots):
            continue
        target = torch.cat([value for value, slide in zip(targets, slide_names) if slide == name])
        prediction_residuals.append(prediction - prediction.mean(dim=0))
        target_residuals.append(target - target.mean(dim=0))
    if not prediction_residuals:
        return predictions[0].sum(dim=0)[:0]
    prediction = torch.cat(prediction_residuals)
    target = torch.cat(target_residuals)
    var_p = prediction.square().mean(dim=0)
    var_t = target.square().mean(dim=0)
    usable = var_t > 1e-12
    scale_p = (var_p + std_floor**2).sqrt()
    scale_t = (var_t + std_floor**2).sqrt()
    correlation = (prediction / scale_p * (target / scale_t)).mean(dim=0).clamp(-1, 1)
    return (1 - correlation)[usable]


def gene_prediction_losses(
    output, batch, *, expression_transform, weights, correlation_min_spots, correlation_std_floor
):
    losses = {}
    field = output["tissue_states"] * output["expression_amplitude"]
    plan = batch["spot_pool"]
    with torch.autocast(device_type=field.device.type, enabled=False):
        pooled = pool_spot_batch(field, plan) @ output["expression_templates"]
    predictions = [value for value, count in zip(pooled.split(plan["counts"]), plan["counts"]) if count]
    targets = [
        expression[indices].detach().float()
        for expression, indices, count in zip(batch["gene_expression"], plan["indices"], plan["counts"])
        if count
    ]
    zero = pooled.sum() * 0
    if not predictions:
        return ({name: zero for name in ("gene_prediction", "gene_correlation") if float(weights[name])}, 0)
    with torch.autocast(device_type=predictions[0].device.type, enabled=False):
        if expression_transform == "log1p":
            predictions = [prediction.log1p() for prediction in predictions]
        if float(weights["gene_prediction"]):
            losses["gene_prediction"] = F.mse_loss(torch.cat(predictions), torch.cat(targets))
        if float(weights["gene_correlation"]):
            slide_names = [name for name, count in zip(batch["slide"], plan["counts"]) if count]
            terms = _gene_correlation_terms(
                predictions, targets, slide_names, correlation_min_spots, correlation_std_floor
            )
            losses["gene_correlation"] = terms.sum() / max(terms.numel(), 1)
    return (losses, sum((len(prediction) for prediction in predictions)))


def _masked_mean(values, weights):
    return (values * weights).sum() / weights.sum().clamp_min(1)


def segmentation_loss(logits, target, valid, *, soft_targets=False):
    logits, target, valid = (logits.float(), target.float(), valid.float())
    if not soft_targets:
        target = (target >= 0.5).float()
    bce = _masked_mean(F.binary_cross_entropy_with_logits(logits, target, reduction="none"), valid)
    probability = logits.sigmoid()
    intersection = (probability * target * valid).sum()
    dice = 1 - (2 * intersection + 1) / ((probability * valid).sum() + (target * valid).sum() + 1)
    return bce + dice


def auxiliary_prediction_losses(
    auxiliary, batch, *, weights, expression_transform,
    correlation_min_spots, correlation_std_floor,
):
    losses, spot_counts = {}, {}
    for scale, output in auxiliary.items():
        branch_losses = {}
        if float(weights.get("segmentation", 0)):
            shape = output["logits"].shape[-2:]
            valid = batch["valid"].float()
            positive = (batch["cancer_probability"] >= 0.5).float() * valid
            area = (valid.shape[-2] / shape[0]) * (valid.shape[-1] / shape[1])
            native_valid = F.adaptive_avg_pool2d(valid, shape) * area
            native_target = (
                F.adaptive_avg_pool2d(positive, shape) * area / native_valid.clamp_min(1e-12)
            )
            # A coarse pixel represents the positive fraction of its valid source pixels.
            branch_losses["segmentation"] = segmentation_loss(
                output["logits"], native_target, native_valid, soft_targets=True,
            )
        if any(float(weights.get(name, 0)) for name in ("gene_prediction", "gene_correlation")):
            native_batch = {**batch, "spot_pool": batch["auxiliary_spot_pools"][scale]}
            gene_losses, count = gene_prediction_losses(
                output, native_batch, weights=weights,
                expression_transform=expression_transform,
                correlation_min_spots=correlation_min_spots,
                correlation_std_floor=correlation_std_floor,
            )
            branch_losses.update(gene_losses)
            spot_counts[scale] = count
        losses[scale] = branch_losses
    return losses, spot_counts


def state_balance_loss(log_usage, pixels, *, alpha=0.1):
    count = log_usage.numel()
    pixels = torch.as_tensor(pixels, device=log_usage.device)
    loss = F.relu(math.log(alpha) - math.log(count) - log_usage).mean()
    return loss * (pixels > 0).to(dtype=loss.dtype)


def template_orthogonality_loss(templates, eps=1e-08):
    log_templates = templates.float().clamp_min(eps).log()
    centered = log_templates - log_templates.mean(dim=1, keepdim=True)
    unit = centered / centered.norm(dim=1, keepdim=True).clamp_min(eps)
    gram = unit @ unit.transpose(0, 1)
    count = gram.shape[0] * (gram.shape[0] - 1)
    off_diagonal = gram - torch.diag_embed(torch.diagonal(gram))
    return off_diagonal.square().sum() / count


def template_entropy_loss(templates, eps=1e-08):
    probabilities = templates.float().clamp_min(eps)
    return -(probabilities * probabilities.log()).sum(dim=1).mean()


def image_losses(output, batch, weights):
    losses = {}
    if weights is None or float(weights["segmentation"]):
        losses["segmentation"] = segmentation_loss(
            output["logits"], batch["cancer_probability"], batch["valid"]
        )
    return losses
