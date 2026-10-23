import math

import torch
from torch.nn import functional as F

from segmodel.modules.spot_pooling import pool_spot_batch


def _gene_correlation_terms(predictions, targets, slide_names, min_spots, std_floor):
    """按 slide 筛选并中心化后，计算带软方差下限的 batch 相关性损失。"""
    prediction_residuals, target_residuals = [], []
    # 同一 slide 的多个 crop 合并后中心化，只使用当前 batch 的有效 spot。
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
    # 两侧标准差加入软下限，保留常数预测的监督并限制近零方差处的梯度。
    scale_p = (var_p + std_floor**2).sqrt()
    scale_t = (var_t + std_floor**2).sqrt()
    correlation = (
        (prediction / scale_p) * (target / scale_t)
    ).mean(dim=0).clamp(-1, 1)
    return (1 - correlation)[usable]


def gene_prediction_losses(
    output, batch, *, expression_transform, weights,
    correlation_min_spots, correlation_std_floor,
):
    """计算 spot 表达 MSE，以及按 slide 去均值后的 batch 正则化相关性损失。"""
    losses = {}
    field = output["tissue_states"] * output["expression_amplitude"]
    plan = batch["spot_pool"]
    with torch.autocast(device_type=field.device.type, enabled=False):
        pooled = pool_spot_batch(field, plan) @ output["expression_templates"]
    predictions = [value for value, count in zip(pooled.split(plan["counts"]), plan["counts"]) if count]
    targets = [expression[indices].detach().float()
               for expression, indices, count in zip(batch["gene_expression"], plan["indices"], plan["counts"])
               if count]
    zero = pooled.sum() * 0
    if not predictions:
        return {
            name: zero for name in ("gene_prediction", "gene_correlation")
            if float(weights.get(name, 0))
        }, 0
    with torch.autocast(device_type=predictions[0].device.type, enabled=False):
        if expression_transform == "log1p":
            predictions = [prediction.log1p() for prediction in predictions]
        if float(weights.get("gene_prediction", 0)):
            losses["gene_prediction"] = F.mse_loss(torch.cat(predictions), torch.cat(targets))
        if float(weights.get("gene_correlation", 0)):
            slide_names = [name for name, count in zip(batch["slide"], plan["counts"]) if count]
            terms = _gene_correlation_terms(
                predictions, targets, slide_names, correlation_min_spots, correlation_std_floor,
            )
            losses["gene_correlation"] = terms.sum() / max(terms.numel(), 1)
    return losses, sum(len(prediction) for prediction in predictions)


def _masked_mean(values, weights):
    return (values * weights).sum() / weights.sum().clamp_min(1)


def segmentation_loss(logits, target, valid):
    logits, target, valid = logits.float(), target.float(), valid.float()
    target = (target >= 0.5).float()
    bce = _masked_mean(
        F.binary_cross_entropy_with_logits(logits, target, reduction="none"), valid
    )
    probability = logits.sigmoid()
    intersection = (probability * target * valid).sum()
    dice = 1 - (2 * intersection + 1) / (
        (probability * valid).sum() + (target * valid).sum() + 1
    )
    return bce + dice


def state_balance_loss(log_usage, pixels, *, alpha=0.1):
    """最低用量损失 mean(relu(log(alpha / K) - log(q)))。"""
    if not 0 < alpha <= 1:
        raise ValueError("state_balance_alpha must be in (0, 1]")
    count = log_usage.numel()
    if count < 2:
        return log_usage.sum() * 0
    pixels = torch.as_tensor(pixels, device=log_usage.device)
    loss = F.relu(math.log(alpha) - math.log(count) - log_usage).mean()
    return loss * (pixels > 0).to(dtype=loss.dtype)


def template_orthogonality_loss(templates, eps=1e-8):
    """模板去冗余：log 模板按基因去均值后两两余弦，取非对角项平方均值。"""
    if templates.ndim != 2:
        raise ValueError("templates must be a K x G matrix")
    log_templates = templates.float().clamp_min(eps).log()
    # 模板行和恒为 1，必须先去掉共同的平均 profile 再比较形状。
    centered = log_templates - log_templates.mean(dim=1, keepdim=True)
    unit = centered / centered.norm(dim=1, keepdim=True).clamp_min(eps)
    gram = unit @ unit.transpose(0, 1)
    count = gram.shape[0] * (gram.shape[0] - 1)
    if count == 0:
        return gram.sum() * 0
    off_diagonal = gram - torch.diag_embed(torch.diagonal(gram))
    return off_diagonal.square().sum() / count


def template_entropy_loss(templates, eps=1e-8):
    """模板熵（nats/state）：正权重压熵使模板更稀疏，负权重抬熵使模板更弥散。"""
    if templates.ndim != 2:
        raise ValueError("templates must be a K x G matrix")
    probabilities = templates.float().clamp_min(eps)
    return -(probabilities * probabilities.log()).sum(dim=1).mean()


def image_losses(output, batch, weights=None):
    """只计算已启用的损失；不传 weights 时保留全部计算的用法。"""
    losses = {}
    if weights is None or float(weights.get("segmentation", 0)):
        losses["segmentation"] = segmentation_loss(
            output["logits"], batch["cancer_probability"], batch["valid"]
        )
    return losses
