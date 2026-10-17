import torch
from torch.nn import functional as F

from segmodel.modules.spot_pooling import collate_spot_pools, pool_spot_batch, prepare_spot_pool


def _gene_correlation_terms(prediction, target, min_spots):
    """同一 crop 内逐基因计算 1-PCC；常数预测仍参与监督。"""
    if len(prediction) < max(3, min_spots):
        return prediction.sum(dim=0)[:0]
    prediction = prediction - prediction.mean(dim=0)
    target = target - target.mean(dim=0)
    usable = target.square().mean(dim=0) > 1e-12
    # normalize 对零向量的反向仍有限，不能因预测为常数而跳过该基因。
    correlation = (
        F.normalize(prediction, dim=0, eps=1e-6)
        * F.normalize(target, dim=0, eps=1e-6)
    ).sum(dim=0).clamp(-1, 1)
    return (1 - correlation)[usable]


def gene_prediction_losses(
    output, batch, spot_radius_px, label_scale, expression_transform="log1p",
    *, weights=None, correlation_min_spots=8,
):
    """spot 均值变换后计算 MSE 和 crop 内逐基因 PCC，返回未加权损失。"""
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
            terms = torch.cat([
                _gene_correlation_terms(prediction, target, correlation_min_spots)
                for prediction, target in zip(predictions, targets)
            ])
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


def image_losses(output, batch, weights=None):
    """只计算已启用的损失；不传 weights 时保留全部计算的用法。"""
    losses = {}
    if weights is None or float(weights.get("segmentation", 0)):
        losses["segmentation"] = segmentation_loss(
            output["logits"], batch["cancer_probability"], batch["valid"]
        )
    return losses
