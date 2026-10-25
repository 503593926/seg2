from math import ceil
from tempfile import TemporaryFile

import cv2
import numpy as np
import torch

from seg2.infer.prefetch import _prefetch_batches


def _prediction_buffer(shape):
    if np.prod(shape) * np.dtype(np.float32).itemsize <= 256 * 1024**2:
        return np.zeros(shape, dtype=np.float32)
    with TemporaryFile(prefix="seg2_prediction_") as file:
        return np.memmap(file, dtype=np.float32, mode="w+", shape=shape)


def tile_origins(length, crop_size, stride, alignment):
    if length <= crop_size:
        return [0]
    last = ceil((length - crop_size) / alignment) * alignment
    origins = list(range(0, last + 1, stride))
    if origins[-1] != last:
        origins.append(last)
    return origins


def _accumulate_pixels(field, pixels, rows, weights, sums):
    pixels = torch.as_tensor(pixels, device=sums.device)
    rows = torch.as_tensor(rows, device=sums.device, dtype=torch.long)
    weights = torch.as_tensor(weights, device=sums.device)
    values = field.flatten(1).T
    for offset in range(0, len(pixels), 262144):
        part = slice(offset, offset + 262144)
        selected = values.index_select(0, pixels[part]).to(torch.float64)
        selected.mul_(weights[part, None])
        sums.index_add_(0, rows[part], selected)
        del selected


class _PredictionGrid:
    def __init__(
        self, height, width, crop_size, tile_shape, origins, cached_weights=None
    ):
        _, tile = tile_shape
        scale = crop_size // tile
        self.shape = (height // scale, width // scale)
        window = np.maximum(
            np.outer(np.hanning(crop_size), np.hanning(crop_size)), 0.01
        ).astype(np.float32)
        window = cv2.resize(window, (tile, tile), interpolation=cv2.INTER_AREA)
        self.windows = {}
        for top, left in origins:
            y, x = (top // scale, left // scale)
            h, w = (
                min(crop_size, height - top) // scale,
                min(crop_size, width - left) // scale,
            )
            self.windows[top, left] = (slice(y, y + h), slice(x, x + w), window[:h, :w])
        if cached_weights is None:
            self.weights = np.zeros(self.shape, dtype=np.float32)
            for y, x, weight in self.windows.values():
                self.weights[y, x] += weight
        else:
            self.weights = cached_weights
        self.denominator = np.maximum(self.weights, 1e-06)


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
    height, width = (slide["height"], slide["width"])
    rows = tile_origins(height, crop_size, stride, alignment)
    columns = tile_origins(width, crop_size, stride, alignment)
    origins = [(top, left) for top in rows for left in columns]
    grid = None
    probability_sum = feature_sum = coefficient_sum = None
    templates = node_sum = spot_sum = None
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
            probability_grid = output["logits"].float().sigmoid()[:, 0].cpu().numpy()
            dense_fields = {}
            if return_features and superpixel_pool is None:
                dense_fields["features"] = (
                    output["features"].float().cpu().numpy().transpose(0, 2, 3, 1)
                )
            coefficients = None
            if compute_expression:
                coefficients = (
                    output["tissue_states"].float()
                    * output["expression_amplitude"].float()
                )
                if expression_factors is not None:
                    dense_fields["coefficients"] = (
                        coefficients.cpu().numpy().transpose(0, 2, 3, 1)
                    )
                if templates is None:
                    templates = output["expression_templates"].float().cpu().numpy()
            if grid is None:
                grid = _PredictionGrid(
                    height,
                    width,
                    crop_size,
                    probability_grid.shape[-2:],
                    origins,
                    superpixel_pool["weights"] if superpixel_pool is not None else None,
                )
                probability_sum = np.zeros(grid.shape, dtype=np.float32)
                if superpixel_pool is not None:
                    superpixel_pool["weights"] = grid.weights
                if spot_pool is not None:
                    spot_sum = torch.zeros(
                        (len(spot_pool["pixel_counts"]), coefficients.shape[1]),
                        dtype=torch.float64,
                        device=coefficients.device,
                    )
                if "coefficients" in dense_fields:
                    coefficient_sum = _prediction_buffer(
                        (*grid.shape, dense_fields["coefficients"].shape[-1])
                    )
                if return_features:
                    if superpixel_pool is None:
                        feature_sum = _prediction_buffer(
                            (*grid.shape, dense_fields["features"].shape[-1])
                        )
                    else:
                        node_sum = torch.zeros(
                            (
                                len(superpixel_pool["pixel_counts"]),
                                output["features"].shape[1],
                            ),
                            dtype=torch.float64,
                            device=output["features"].device,
                        )
            dense_sums = {"features": feature_sum, "coefficients": coefficient_sum}
            for index, origin in enumerate(batch_origins):
                y, x, weight = grid.windows[origin]
                h, w = weight.shape
                probability_sum[y, x] += probability_grid[index, :h, :w] * weight
                for name in dense_fields:
                    dense_sums[name][y, x] += (
                        dense_fields[name][index, :h, :w] * weight[..., None]
                    )
                if spot_sum is not None or node_sum is not None:
                    normalized = (
                        weight.astype(np.float64) / grid.denominator[y, x]
                    ).ravel()
                if spot_sum is not None:
                    grid_width = grid.shape[1]
                    first, last = np.searchsorted(
                        spot_pool["pixels"], [y.start * grid_width, y.stop * grid_width]
                    )
                    pixels = spot_pool["pixels"][first:last]
                    columns = pixels % grid_width
                    inside = (columns >= x.start) & (columns < x.stop)
                    local_pixels = (
                        (pixels[inside] // grid_width - y.start) * w
                        + columns[inside]
                        - x.start
                    )
                    _accumulate_pixels(
                        coefficients[index, :, :h, :w],
                        local_pixels,
                        spot_pool["rows"][first:last][inside],
                        normalized[local_pixels],
                        spot_sum,
                    )
                if node_sum is not None:
                    labels = superpixel_pool["superpixels"][y, x].ravel()
                    foreground = np.flatnonzero(labels)
                    _accumulate_pixels(
                        output["features"][index, :, :h, :w],
                        foreground,
                        labels[foreground],
                        normalized[foreground],
                        node_sum,
                    )
            del output, inputs, coefficients, dense_fields, dense_sums
    finally:
        batches.close()
        model.train(was_training)
    for field in (feature_sum, coefficient_sum):
        if field is not None:
            field /= grid.denominator[..., None]
    features = feature_sum
    if node_sum is not None:
        features = node_sum.cpu().numpy()
        features /= np.maximum(superpixel_pool["pixel_counts"][:, None], 1)
        features[0] = 0
    if spot_sum is not None:
        spot_pool["coefficients"] = (
            spot_sum.cpu().numpy() / np.maximum(spot_pool["pixel_counts"][:, None], 1)
        ).astype(np.float32)
        spot_pool["templates"] = templates
    if coefficient_sum is not None:
        expression_factors.update(coefficients=coefficient_sum, templates=templates)
    return (probability_sum / grid.denominator, features)
