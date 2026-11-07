"""Build geometry-independent pixel plans for circular and polygon regions."""

import math

import numpy as np
import torch


def _circle_windows(coordinates, shape, radius_px, label_scale):
    height, width = shape
    radius = radius_px / label_scale
    for center in coordinates / label_scale:
        x, y = center.tolist()
        left, right = (
            max(0, math.ceil(x - radius)),
            min(width, math.floor(x + radius) + 1),
        )
        top, bottom = (
            max(0, math.ceil(y - radius)),
            min(height, math.floor(y + radius) + 1),
        )
        if left >= right or top >= bottom:
            yield top, bottom, left, right, None
            continue
        yy = torch.arange(top, bottom, device=coordinates.device)[:, None]
        xx = torch.arange(left, right, device=coordinates.device)[None, :]
        yield (
            top,
            bottom,
            left,
            right,
            (xx - x).square() + (yy - y).square() <= radius**2,
        )


def prepare_region_pool(
    valid_mask,
    *,
    coordinates=None,
    radius_px=None,
    label_scale=1,
    region_map=None,
    region_ids=None,
):
    """Return one common plan; indices refer to rows in the requested region order.

    Circles use input-grid coordinates/radius and an output-grid label_scale.
    Polygon maps already lie on the output grid, with positive integer region IDs.
    Background ID 0 and regions without valid pixels are excluded.
    """
    valid_mask = torch.as_tensor(valid_mask).bool().cpu()
    if region_map is not None:
        labels = np.asarray(region_map)
        ids = np.asarray(region_ids, dtype=np.int64)
        if labels.shape != tuple(valid_mask.shape) or ids.ndim != 1:
            raise ValueError("Region map and valid-mask grids must match")
        if len(np.unique(ids)) != len(ids) or (ids <= 0).any() or (labels < 0).any():
            raise ValueError(
                "Region IDs must be unique positive integers; map background is 0"
            )
        lookup = np.full(
            max(int(labels.max(initial=0)), int(ids.max(initial=0))) + 1,
            -1,
            dtype=np.int64,
        )
        lookup[ids] = np.arange(len(ids))
        source_rows = lookup[labels.ravel()]
        expected = np.bincount(source_rows[source_rows >= 0], minlength=len(ids))
        pixels = np.flatnonzero(valid_mask.numpy().ravel() & (source_rows >= 0))
        original_rows = source_rows[pixels]
        counts = np.bincount(original_rows, minlength=len(ids))
        indices = np.flatnonzero(counts > 0)
        compact = np.full(len(ids), -1, dtype=np.int64)
        compact[indices] = np.arange(len(indices))
        rows = compact[original_rows]
        weights = (1.0 / counts[original_rows]).astype(np.float32)
        return {
            "pixels": torch.from_numpy(pixels),
            "rows": torch.from_numpy(rows),
            "weights": torch.from_numpy(weights),
            "indices": torch.from_numpy(indices),
            "count": len(indices),
            "expected_counts": torch.from_numpy(expected),
        }
    if coordinates is None or radius_px is None or float(radius_px) <= 0:
        raise ValueError("Circular regions require coordinates and a positive radius")
    coordinates = torch.as_tensor(coordinates).cpu()
    pixels, rows, weights, indices, expected = [], [], [], [], []
    width = valid_mask.shape[1]
    radius = float(radius_px) / float(label_scale)
    for center in coordinates / label_scale:
        x, y = center.tolist()
        xx = torch.arange(math.ceil(x - radius), math.floor(x + radius) + 1)
        yy = torch.arange(math.ceil(y - radius), math.floor(y + radius) + 1)
        expected.append(
            int(
                (
                    (xx[None, :] - x).square() + (yy[:, None] - y).square() <= radius**2
                ).sum()
            )
        )
    for index, (top, bottom, left, right, disk) in enumerate(
        _circle_windows(
            coordinates=coordinates,
            shape=valid_mask.shape,
            radius_px=radius_px,
            label_scale=label_scale,
        )
    ):
        if disk is not None:
            disk = disk & valid_mask[top:bottom, left:right]
        if disk is None or not bool(disk.any()):
            continue
        y, x = disk.nonzero(as_tuple=True)
        pixels.append((y + top) * width + x + left)
        rows.append(torch.full_like(x, len(indices)))
        weights.append(torch.full((len(x),), 1.0 / len(x)))
        indices.append(index)
    return {
        "pixels": torch.cat(pixels) if pixels else torch.empty(0, dtype=torch.long),
        "rows": torch.cat(rows) if rows else torch.empty(0, dtype=torch.long),
        "weights": torch.cat(weights) if weights else torch.empty(0),
        "indices": torch.tensor(indices, dtype=torch.long),
        "count": len(indices),
        "expected_counts": torch.tensor(expected, dtype=torch.long),
    }


def collate_region_pools(plans, shape):
    area = shape[0] * shape[1]
    offsets, total = [], 0
    for plan in plans:
        offsets.append(total)
        total += plan["count"]
    return {
        "pixels": torch.cat(
            [plan["pixels"] + index * area for index, plan in enumerate(plans)]
        ),
        "rows": torch.cat(
            [plan["rows"] + offset for plan, offset in zip(plans, offsets)]
        ),
        "weights": torch.cat([plan["weights"] for plan in plans]),
        "indices": [plan["indices"] for plan in plans],
        "counts": [plan["count"] for plan in plans],
        "count": total,
    }


def pool_region_batch(field, plan):
    pixels = field.float().flatten(2).transpose(1, 2)
    area = pixels.shape[1]
    selected = (
        pixels[plan["pixels"] // area, plan["pixels"] % area] * plan["weights"][:, None]
    )
    return selected.new_zeros(plan["count"], field.shape[1]).index_add(
        0, plan["rows"], selected
    )
