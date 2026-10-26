from math import ceil
from typing import cast

import cv2
import numpy as np
import torch

from seg2.infer.prefetch import _prefetch_batches


def tile_origins(length, crop_size, stride, alignment):
    if length <= crop_size:
        return [0]
    last = ceil((length - crop_size) / alignment) * alignment
    origins = list(range(0, last + 1, stride))
    if origins[-1] != last:
        origins.append(last)
    return origins


def _pool_pixels(field, pixels, rows, pixel_counts):
    pixels = torch.as_tensor(pixels, device=field.device, dtype=torch.long)
    rows = torch.as_tensor(rows, device=field.device, dtype=torch.long)
    counts = torch.as_tensor(pixel_counts, device=field.device, dtype=torch.float64)
    sums = torch.zeros(
        (len(counts), field.shape[0]), device=field.device, dtype=torch.float64
    )
    values = field.flatten(1).T
    for offset in range(0, len(pixels), 262144):
        part = slice(offset, offset + 262144)
        selected = values.index_select(0, pixels[part]).to(torch.float64)
        sums.index_add_(0, rows[part], selected)
        del selected
    return sums / counts.clamp_min(1)[:, None]


class _PredictionGrid:
    def __init__(
        self, height, width, crop_size, tile_shape, origins, device, cached_weights=None
    ):
        _, tile = tile_shape
        scale = crop_size // tile
        self.shape = (height // scale, width // scale)
        window = np.maximum(
            np.outer(np.hanning(crop_size), np.hanning(crop_size)), 0.01
        ).astype(np.float32)
        window = cv2.resize(window, (tile, tile), interpolation=cv2.INTER_AREA)
        window = torch.as_tensor(window, device=device)
        self.windows = {}
        for top, left in origins:
            y, x = (top // scale, left // scale)
            h, w = (
                min(crop_size, height - top) // scale,
                min(crop_size, width - left) // scale,
            )
            self.windows[top, left] = (slice(y, y + h), slice(x, x + w), window[:h, :w])
        if cached_weights is None:
            self.weights = torch.zeros(self.shape, device=device, dtype=torch.float32)
            for y, x, weight in self.windows.values():
                self.weights[y, x].add_(weight)
        else:
            self.weights = torch.as_tensor(cached_weights, device=device)
        self.denominator = self.weights.clamp_min(1e-6)


@torch.inference_mode()
def predict_slide(
    model,
    slide,
    crop_size,
    device,
    batch_size=1,
    stride=None,
    return_features=True,
    expression_factors=None,
    superpixel_pool=None,
    spot_pool=None,
):
    batch_size = int(batch_size)
    alignment = slide["alignment"]
    stride = (
        max(alignment, crop_size // 2 // alignment * alignment)
        if stride is None
        else int(stride)
    )
    height, width = slide["image"].shape[:2]
    rows = tile_origins(height, crop_size, stride, alignment)
    columns = tile_origins(width, crop_size, stride, alignment)
    origins = [(top, left) for top in rows for left in columns]
    grid = None
    probability_sum = None
    templates = None
    dense_sums = {}
    compute_expression = expression_factors is not None or spot_pool is not None
    batches = _prefetch_batches(slide, origins, crop_size, batch_size, device)
    was_training = model.training
    model.eval()
    try:
        for batch_origins, inputs in batches:
            output = model(
                inputs["image"][0],
                inputs["patch"],
                inputs["cls"],
                compute_expression=compute_expression,
            )
            probability_grid = output["logits"].float().sigmoid()[:, 0]
            dense_fields = {}
            if return_features:
                dense_fields["features"] = output["features"].float()
            if compute_expression:
                dense_fields["coefficients"] = (
                    output["tissue_states"].float()
                    * output["expression_amplitude"].float()
                )
                if templates is None:
                    templates = output["expression_templates"].float()
            if grid is None:
                grid = _PredictionGrid(
                    height,
                    width,
                    crop_size,
                    probability_grid.shape[-2:],
                    origins,
                    probability_grid.device,
                    superpixel_pool["weights"] if superpixel_pool is not None else None,
                )
                probability_sum = torch.zeros(
                    grid.shape, device=probability_grid.device, dtype=torch.float32
                )
                if superpixel_pool is not None:
                    superpixel_pool["weights"] = grid.weights
                dense_sums = {
                    name: torch.zeros(
                        (field.shape[1], *grid.shape),
                        device=field.device,
                        dtype=(
                            torch.float64
                            if (name == "features" and superpixel_pool is not None)
                            or (name == "coefficients" and spot_pool is not None)
                            else torch.float32
                        ),
                    )
                    for name, field in dense_fields.items()
                }
            probability_sum = cast(torch.Tensor, probability_sum)
            for index, origin in enumerate(batch_origins):
                y, x, weight = grid.windows[origin]
                h, w = weight.shape
                probability_sum[y, x].addcmul_(probability_grid[index, :h, :w], weight)
                for name, field in dense_fields.items():
                    dense_sums[name][:, y, x].addcmul_(
                        field[index, :, :h, :w], weight
                    )
            del output, inputs, dense_fields
    finally:
        batches.close()
        model.train(was_training)
    grid = cast(_PredictionGrid, grid)
    probability_sum = cast(torch.Tensor, probability_sum)
    probability_sum.div_(grid.denominator)
    for field in dense_sums.values():
        field.div_(grid.denominator)
    results = {"probability": probability_sum}
    if return_features:
        field = dense_sums["features"]
        if superpixel_pool is not None:
            labels = superpixel_pool["superpixels"].ravel()
            pixels = np.flatnonzero(labels)
            results["features"] = _pool_pixels(
                field, pixels, labels[pixels], superpixel_pool["pixel_counts"]
            )
            results["features"][0] = 0
        else:
            results["features"] = field.permute(1, 2, 0).contiguous()
    if compute_expression:
        results["templates"] = cast(torch.Tensor, templates)
    if spot_pool is not None:
        results["spot_coefficients"] = _pool_pixels(
            dense_sums["coefficients"],
            spot_pool["pixels"],
            spot_pool["rows"],
            spot_pool["pixel_counts"],
        ).float()
    if expression_factors is not None:
        results["dense_coefficients"] = (
            dense_sums["coefficients"].float().permute(1, 2, 0).contiguous()
        )
    results = {name: value.cpu().numpy() for name, value in results.items()}
    if spot_pool is not None:
        spot_pool.update(
            coefficients=results["spot_coefficients"], templates=results["templates"]
        )
    if expression_factors is not None:
        expression_factors.update(
            coefficients=results["dense_coefficients"], templates=results["templates"]
        )
    return (results["probability"], results.get("features"))
