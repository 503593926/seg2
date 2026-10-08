import torch
from torch.nn import functional as F


def get_gene_losses(reconstruction, expression, mean, logvar):
    """无监督基因 VAE 的重构与 KL 损失。"""
    kl = -0.5 * (1 + logvar - mean.square() - logvar.exp()).mean()
    return {
        "reconstruction": F.mse_loss(reconstruction, expression),
        "kl": kl,
    }


def distillation_loss(embedding, target_embedding, weights):
    """单向蒸馏：按 spot 权重把 embedding 拉向 target_embedding（加权 mean(1 - cos)）。"""
    distance = 1 - F.cosine_similarity(embedding, target_embedding, dim=1)
    return (weights * distance).mean()


def relation_distillation_loss(
    embedding: torch.Tensor,
    target_embedding: torch.Tensor,
    crop_ids: torch.Tensor,
    spot_ids: torch.Tensor,
) -> tuple[torch.Tensor, int, int]:
    with torch.autocast(device_type=embedding.device.type, enabled=False):
        embedding = embedding.float()
        target_embedding = target_embedding.detach().float()
        # 同一 spot 在不同 crop 中保留各自视图, 只合并 crop 内重复项.
        identities = torch.stack((crop_ids, spot_ids), dim=1)
        unique_ids, inverse, counts = torch.unique(
            identities, dim=0, return_inverse=True, return_counts=True,
        )
        size = unique_ids.shape[0]
        if size < 2:
            return embedding.sum() * 0, 0, 0

        image = embedding.new_zeros((size, embedding.shape[1]))
        gene = target_embedding.new_zeros((size, target_embedding.shape[1]))
        image = image.index_add(0, inverse, embedding) / counts[:, None]
        gene = gene.index_add(0, inverse, target_embedding) / counts[:, None]
        image = F.normalize(image, dim=1, eps=1e-8)
        gene = F.normalize(gene, dim=1, eps=1e-8)
        loss = image.sum() * 0
        pair_count, crop_count = 0, 0
        for crop_id in torch.unique(unique_ids[:, 0]):
            keep = unique_ids[:, 0] == crop_id
            crop_image, crop_gene = image[keep], gene[keep]
            size = crop_image.shape[0]
            if size < 2:
                continue
            image_similarity = crop_image @ crop_image.transpose(0, 1)
            gene_similarity = crop_gene @ crop_gene.transpose(0, 1)
            rows, columns = torch.triu_indices(size, size, offset=1, device=image.device)
            loss = loss + (
                image_similarity[rows, columns] - gene_similarity[rows, columns]
            ).square().mean()
            pair_count += rows.numel()
            crop_count += 1
        # 每个有效 crop 等权, 避免 spot 多的 crop 主导监督.
        return loss / max(crop_count, 1), pair_count, crop_count


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


def pfm_losses(output, batch, pfm_count):
    """PFM 解耦损失：f2 的身份分类 + f1/f2 去相关，global 与 spatial 各一组。

    统计量按 PFM 优先展平，第 k*B+b 行对应第 k 个 PFM，因此目标是 repeat_interleave；
    spatial 侧每个 (样本, PFM) 还多了池化位置那一维，用行数比例反推重复倍数。
    """
    # (B, K) -> (K, B) -> (K*B,)，与输出里 PFM 优先的行序一致
    available = batch["pfm_mask"].transpose(0, 1).reshape(-1)
    losses = {}
    for branch in ("global", "spatial"):
        logits = output[f"pfm_logits_{branch}"].float()
        f1 = output[f"pfm_f1_{branch}"]
        f2 = output[f"pfm_f2_{branch}"]
        targets = torch.arange(pfm_count, device=logits.device).repeat_interleave(
            logits.shape[0] // pfm_count
        )
        losses[f"pfm_cls_{branch}"] = _masked_mean(
            F.cross_entropy(logits, targets, reduction="none"),
            available.repeat_interleave(logits.shape[0] // available.shape[0]).float(),
        )
        losses[f"pfm_orth_{branch}"] = cross_correlation_loss(
            f1,
            f2,
            available.repeat_interleave(f1.shape[0] // available.shape[0]),
        )
    return losses


def image_losses(output, batch):
    return {
        "segmentation": segmentation_loss(
            output["logits"], batch["cancer_probability"], batch["valid"]
        ),
        "affinity": affinity_loss(
            output["affinity_logits"], batch["superpixels"], batch["valid"]
        ),
        "superpixel": superpixel_loss(
            output["logits"], batch["superpixels"], batch["valid"]
        ),
    }
