import math

import torch
from torch.nn import functional as F

from segmodel.modules.spot_pooling import collate_spot_pools, pool_spot_batch, prepare_spot_pool


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
    output, batch, spot_radius_px, label_scale, expression_transform="log1p",
    *, weights=None, correlation_min_spots=8, correlation_std_floor=0.02,
):
    """计算 spot 表达 MSE，以及按 slide 去均值后的 batch 正则化相关性损失。"""
    if weights is None:
        weights = {"gene_prediction": 1.0}
    losses = {}
    factorized = "expression_templates" in output
    field = (output["tissue_states"] * output["expression_amplitude"]
             if factorized else output["gene_expression"])
    plan = batch.get("spot_pool")
    if plan is None:
        tissue = batch.get("tissue", batch["superpixels"])[:, 0]
        plans = [prepare_spot_pool(grid, spots, spot_radius_px, label_scale)
                 for grid, spots in zip(tissue, batch["spots"])]
        plan = collate_spot_pools(plans, field.shape[-2:])
        plan = {key: [item.to(field.device) for item in value] if key == "indices"
                else value.to(field.device) if isinstance(value, torch.Tensor) else value
                for key, value in plan.items()}
    with torch.autocast(device_type=field.device.type, enabled=False):
        pooled = pool_spot_batch(field, plan)
        if factorized:
            pooled = pooled @ output["expression_templates"]
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


def image_losses(output, batch, weights=None):
    """只计算已启用的损失；不传 weights 时保留全部计算的用法。"""
    losses = {}
    if weights is None or float(weights.get("segmentation", 0)):
        losses["segmentation"] = segmentation_loss(
            output["logits"], batch["cancer_probability"], batch["valid"]
        )
    return losses
