"""将非负线性表达按 spot 圆盘内组织面积聚合。"""

import math

import torch


def _spot_windows(spots, shape, radius_px, label_scale):
    height, width = shape
    radius = radius_px / label_scale
    for center in spots / label_scale:
        x, y = center.tolist()
        left, right = max(0, math.ceil(x - radius)), min(width, math.floor(x + radius) + 1)
        top, bottom = max(0, math.ceil(y - radius)), min(height, math.floor(y + radius) + 1)
        if left >= right or top >= bottom:
            yield top, bottom, left, right, None
            continue
        yy = torch.arange(top, bottom, device=spots.device)[:, None]
        xx = torch.arange(left, right, device=spots.device)[None, :]
        disk = (xx - x).square() + (yy - y).square() <= radius**2
        yield top, bottom, left, right, disk



def prepare_spot_pool(tissue, spots, radius_px, label_scale):
    """在 CPU 上准备稀疏圆盘索引；只保留有组织像素的 spot。"""
    tissue, spots = tissue.cpu(), spots.cpu()
    pixels, rows, weights, valid_indices = [], [], [], []
    width = tissue.shape[1]
    for index, (top, bottom, left, right, disk) in enumerate(
        _spot_windows(spots, tissue.shape, radius_px, label_scale)
    ):
        if disk is not None:
            disk = disk & (tissue[top:bottom, left:right] > 0)
        if disk is None or not bool(disk.any()):
            continue
        y, x = disk.nonzero(as_tuple=True)
        pixels.append((y + top) * width + x + left)
        rows.append(torch.full_like(x, len(valid_indices)))
        weights.append(torch.full((len(x),), 1.0 / len(x)))
        valid_indices.append(index)
    return {
        "pixels": torch.cat(pixels) if pixels else torch.empty(0, dtype=torch.long),
        "rows": torch.cat(rows) if rows else torch.empty(0, dtype=torch.long),
        "weights": torch.cat(weights) if weights else torch.empty(0),
        "indices": torch.tensor(valid_indices, dtype=torch.long),
        "count": len(valid_indices),
        "spot_count": len(spots),
    }


def collate_spot_pools(plans, shape):
    """合并不同 crop 的像素索引，GPU 上一次聚合整个 batch。"""
    area = shape[0] * shape[1]
    offsets, total = [], 0
    for plan in plans:
        offsets.append(total)
        total += plan["count"]
    return {
        "pixels": torch.cat([plan["pixels"] + index * area for index, plan in enumerate(plans)]),
        "rows": torch.cat([plan["rows"] + offset for plan, offset in zip(plans, offsets)]),
        "weights": torch.cat([plan["weights"] for plan in plans]),
        "indices": [plan["indices"] for plan in plans],
        "counts": [plan["count"] for plan in plans],
        "count": total,
    }


def pool_spot_batch(field, plan):
    """(B,C,H,W) → (有效 spot,C)，索引与归一化权重均已准备好。"""
    pixels = field.float().permute(0, 2, 3, 1).reshape(-1, field.shape[1])
    selected = pixels[plan["pixels"]] * plan["weights"][:, None]
    return selected.new_zeros(plan["count"], field.shape[1]).index_add(0, plan["rows"], selected)


def pool_spot_expression(field, superpixels, spots, radius_px, label_scale):
    """整片评估逐圆盘聚合，限制临时内存；训练使用 pool_spot_batch。"""
    predictions, valid = [], []
    for top, bottom, left, right, disk in _spot_windows(spots, superpixels.shape, radius_px, label_scale):
        if disk is not None:
            disk = disk & (superpixels[top:bottom, left:right] > 0)
        usable = disk is not None and bool(disk.any())
        valid.append(usable)
        predictions.append(
            field[:, top:bottom, left:right][:, disk].float().mean(1)
            if usable else field.new_zeros(field.shape[0], dtype=torch.float32)
        )
    values = torch.stack(predictions) if predictions else field.new_zeros(0, field.shape[0], dtype=torch.float32)
    return values, torch.tensor(valid, dtype=torch.bool, device=field.device)

