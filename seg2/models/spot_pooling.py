import math

import torch


def _spot_windows(spots, shape, radius_px, label_scale):
    height, width = shape
    radius = radius_px / label_scale
    for center in spots / label_scale:
        x, y = center.tolist()
        left, right = (max(0, math.ceil(x - radius)), min(width, math.floor(x + radius) + 1))
        top, bottom = (max(0, math.ceil(y - radius)), min(height, math.floor(y + radius) + 1))
        if left >= right or top >= bottom:
            yield (top, bottom, left, right, None)
            continue
        yy = torch.arange(top, bottom, device=spots.device)[:, None]
        xx = torch.arange(left, right, device=spots.device)[None, :]
        disk = (xx - x).square() + (yy - y).square() <= radius**2
        yield (top, bottom, left, right, disk)


def prepare_spot_pool(tissue, spots, radius_px, label_scale):
    tissue, spots = (tissue.cpu(), spots.cpu())
    pixels, rows, weights, valid_indices = ([], [], [], [])
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
    }


def collate_spot_pools(plans, shape):
    area = shape[0] * shape[1]
    offsets, total = ([], 0)
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
    pixels = field.float().flatten(2).transpose(1, 2)
    area = pixels.shape[1]
    selected = pixels[plan["pixels"] // area, plan["pixels"] % area] * plan["weights"][:, None]
    return selected.new_zeros(plan["count"], field.shape[1]).index_add(0, plan["rows"], selected)
