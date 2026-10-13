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


def soft_segmentation_loss(logits, target, valid, weights=None):
    """软目标版本: 不做阈值化, 梯度可以穿过 target 回到传播算子。

    target 是 [0,1] 的连续值 (传播输出), weights 可选地给出逐像素置信度。
    与 segmentation_loss 的区别只在 target 不二值化、且 BCE 对软标签求导。
    """
    logits, target, valid = logits.float(), target.float(), valid.float()
    if weights is not None:
        weights = weights.float()
    else:
        weights = torch.ones_like(target)
    weights = weights * valid
    bce = _masked_mean(
        F.binary_cross_entropy_with_logits(logits, target, reduction="none"), weights
    )
    probability = logits.sigmoid()
    intersection = (probability * target * weights).sum()
    dice = 1 - (2 * intersection + 1) / (
        (probability * weights).sum() + (target * weights).sum() + 1
    )
    return bce + dice


def affinity_loss(logits, superpixels, valid):
    ids = superpixels[:, 0]
    usable = valid[:, 0].bool()
    right = ids[:, :, :-1] == ids[:, :, 1:]
    down = ids[:, :-1] == ids[:, 1:]
    horizontal = _masked_mean(
        F.binary_cross_entropy_with_logits(
            logits[:, 0, :, :-1].float(), right.float(), reduction="none"
        ),
        (usable[:, :, :-1] & usable[:, :, 1:]).float(),
    )
    vertical = _masked_mean(
        F.binary_cross_entropy_with_logits(
            logits[:, 1, :-1].float(), down.float(), reduction="none"
        ),
        (usable[:, :-1] & usable[:, 1:]).float(),
    )
    return (horizontal + vertical) / 2


def superpixel_loss(logits, superpixels, valid):
    probability = logits.float().sigmoid()
    loss = probability.sum() * 0
    group_count = 0
    for values, ids, keep in zip(probability, superpixels, valid.bool()):
        if not keep.any():
            continue
        values = values[keep]
        _, inverse = torch.unique(ids[keep], return_inverse=True)
        counts = torch.bincount(inverse).to(values.dtype)
        sums = torch.zeros_like(counts).scatter_add(0, inverse, values)
        means = sums / counts
        variance = torch.zeros_like(counts).scatter_add(
            0, inverse, (values - means[inverse]).square()
        )
        loss += (variance / counts).sum()
        group_count += counts.numel()
    return loss / max(group_count, 1)


def cross_correlation_loss(f1, f2, available=None):
    """f1 与 f2 的互协方差压零：按维标准化后取 C = F1ᵀF2/N，损失为 ‖C‖²/d²。

    整个矩阵压零（不像 Barlow Twins 那样把对角线拉向 1）：这里 f1/f2 是两个不同投影，
    目标是让 f1 不留下可以被 f2 线性预测的成分。注意它只能去掉线性相关的部分。

    被屏蔽 PFM 的行直接剔除而不是留着：它这一步的 f1/f2 是常量，参与统计会让
    L_orth 的量级随屏蔽率漂移，后续消融就分不清是机制还是尺度变了。
    """
    f1, f2 = f1.float(), f2.float()
    if available is not None:
        f1, f2 = f1[available], f2[available]
    if not f1.shape[0]:
        # 整个 batch 的 PFM 都被屏蔽时该项无定义；返回 0 而不是让空张量算出 NaN
        return f1.sum() * 0
    normalized_1 = (f1 - f1.mean(dim=0)) / f1.std(dim=0, unbiased=False).clamp_min(1e-6)
    normalized_2 = (f2 - f2.mean(dim=0)) / f2.std(dim=0, unbiased=False).clamp_min(1e-6)
    dimension = normalized_1.shape[1]
    correlation = (
        normalized_1.transpose(0, 1) @ normalized_2 / normalized_1.shape[0]
    )
    return correlation.square().sum() / (dimension**2)


def pfm_losses(output, batch, pfm_count, weights=None):
    """PFM 解耦损失：f2 的身份分类 + f1/f2 去相关，global 与 spatial 各一组。

    统计量按 PFM 优先展平，第 k*B+b 行对应第 k 个 PFM，因此目标是 repeat_interleave；
    spatial 侧每个 (样本, PFM) 还多了池化位置那一维，用行数比例反推重复倍数。
    """
    if weights is not None and not any(
        float(weights.get(f"pfm_{kind}_{branch}", 0))
        for branch in ("global", "spatial") for kind in ("cls", "orth")
    ):
        return {}
    # (B, K) -> (K, B) -> (K*B,)，与输出里 PFM 优先的行序一致
    available = batch["pfm_mask"].transpose(0, 1).reshape(-1)
    losses = {}
    for branch in ("global", "spatial"):
        if weights is None or float(weights.get(f"pfm_cls_{branch}", 0)):
            logits = output[f"pfm_logits_{branch}"].float()
            targets = torch.arange(pfm_count, device=logits.device).repeat_interleave(
                logits.shape[0] // pfm_count
            )
            losses[f"pfm_cls_{branch}"] = _masked_mean(
                F.cross_entropy(logits, targets, reduction="none"),
                available.repeat_interleave(logits.shape[0] // available.shape[0]).float(),
            )
        if weights is None or float(weights.get(f"pfm_orth_{branch}", 0)):
            f1 = output[f"pfm_f1_{branch}"]
            f2 = output[f"pfm_f2_{branch}"]
            losses[f"pfm_orth_{branch}"] = cross_correlation_loss(
                f1,
                f2,
                available.repeat_interleave(f1.shape[0] // available.shape[0]),
            )
    return losses


def image_losses(output, batch, weights=None):
    """只计算已启用的损失；不传 weights 时保留全部计算的用法。"""
    losses = {}
    if weights is None or float(weights.get("segmentation", 0)):
        losses["segmentation"] = segmentation_loss(
            output["logits"], batch["cancer_probability"], batch["valid"]
        )
    if weights is None or float(weights.get("affinity", 0)):
        losses["affinity"] = affinity_loss(
            output["affinity_logits"], batch["superpixels"], batch["valid"]
        )
    if weights is None or float(weights.get("superpixel", 0)):
        losses["superpixel"] = superpixel_loss(
            output["logits"], batch["superpixels"], batch["valid"]
        )
    return losses
